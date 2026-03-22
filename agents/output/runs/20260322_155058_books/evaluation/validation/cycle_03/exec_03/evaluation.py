import os
import sys
import json
import ast
import re
from pathlib import Path
from collections import Counter

import pandas as pd

from PyDI.io import load_csv
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEvaluator,
    tokenized_match,
    year_only_match,
    set_equality_match,
    numeric_tolerance_match,
    exact_match,
    boolean_match,
)

try:
    from PyDI.normalization import normalize_country
except Exception:
    normalize_country = None

try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _candidates = [
        Path.cwd(),
        Path.cwd() / "agents",
        Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd(),
        Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd(),
        Path(__file__).resolve().parent.parent.parent if "__file__" in globals() else Path.cwd(),
    ]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            resolved = str(_path.resolve())
            if resolved not in sys.path:
                sys.path.append(resolved)
    from list_normalization import detect_list_like_columns, normalize_list_like_columns


OUTPUT_DIR = "output/runs/20260322_155058_books"
FUSION_PATH = "output/runs/20260322_155058_books/data_fusion/fusion_data.csv"
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_OUTPUT = "output/runs/20260322_155058_books/pipeline_evaluation/pipeline_evaluation.json"
EVAL_STAGE = "validation"
EVAL_SET_PATH = "input/datasets/books/testsets/validation_set.csv"

os.makedirs(EVAL_DIR, exist_ok=True)


def _safe_is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    try:
        result = pd.isna(value)
        if hasattr(result, "__len__") and not isinstance(result, str):
            return False
        return bool(result)
    except Exception:
        return False


def _parse_source_ids(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text or text[0] not in "[({":
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(v) for v in parsed if str(v)]
        except Exception:
            pass
    return []


def _detect_gold_prefix(gold_ids):
    prefixes = [g.split("_", 1)[0] for g in gold_ids if "_" in g]
    if not prefixes:
        return None
    return Counter(prefixes).most_common(1)[0][0] + "_"


def build_eval_view(fused_df, gold_df):
    gold_ids = set(gold_df["id"].dropna().astype(str).tolist()) if "id" in gold_df.columns else set()
    gold_prefix = _detect_gold_prefix(gold_ids)
    expanded_rows = []

    for _, row in fused_df.iterrows():
        row_dict = row.to_dict()
        cluster_id = str(row_dict.get("_id", ""))
        explicit_eval_id = row_dict.get("eval_id")
        source_ids = _parse_source_ids(row_dict.get("_fusion_sources"))

        candidates = []
        if not _safe_is_missing(explicit_eval_id):
            candidates.append(str(explicit_eval_id))
        if gold_prefix:
            candidates.extend([sid for sid in source_ids if sid.startswith(gold_prefix)])
        candidates.extend([sid for sid in source_ids if sid in gold_ids])

        if not candidates and cluster_id:
            candidates = [cluster_id]

        seen = set()
        for eval_id in candidates:
            eval_id = str(eval_id)
            if eval_id in seen:
                continue
            seen.add(eval_id)
            enriched = dict(row_dict)
            enriched["eval_id"] = eval_id
            enriched["_eval_cluster_id"] = cluster_id
            expanded_rows.append(enriched)

    if not expanded_rows:
        fallback = fused_df.copy()
        if "eval_id" not in fallback.columns:
            if "_id" in fallback.columns:
                fallback["eval_id"] = fallback["_id"].astype(str)
            else:
                fallback["eval_id"] = fallback.index.astype(str)
        fallback["_eval_cluster_id"] = fallback["eval_id"].astype(str)
        return fallback

    fused_eval = pd.DataFrame(expanded_rows)
    if "_fusion_confidence" in fused_eval.columns:
        fused_eval["_fusion_confidence"] = pd.to_numeric(
            fused_eval["_fusion_confidence"], errors="coerce"
        ).fillna(0.0)
        fused_eval = fused_eval.sort_values("_fusion_confidence", ascending=False)
    fused_eval = fused_eval.drop_duplicates(subset=["eval_id"], keep="first")
    return fused_eval


def _infer_country_output_format(gold_df, country_columns):
    values = []
    for col in country_columns:
        if col not in gold_df.columns:
            continue
        values.extend(gold_df[col].dropna().astype(str).head(500).tolist())
    if not values:
        return "name"

    total = max(len(values), 1)
    alpha2 = sum(1 for v in values if re.fullmatch(r"[A-Z]{2}", v.strip()))
    alpha3 = sum(1 for v in values if re.fullmatch(r"[A-Z]{3}", v.strip()))
    numeric = sum(1 for v in values if re.fullmatch(r"\d{3}", v.strip()))

    if alpha2 / total >= 0.70:
        return "alpha_2"
    if alpha3 / total >= 0.70:
        return "alpha_3"
    if numeric / total >= 0.70:
        return "numeric"
    return "name"


def _normalize_country_safe(value, output_format):
    if _safe_is_missing(value) or normalize_country is None:
        return value
    text = str(value).strip()
    if not text:
        return value
    try:
        normalized = normalize_country(text, output_format=output_format)
        return normalized if normalized else text
    except Exception:
        return text


def _looks_like_serialized_list(text):
    if not isinstance(text, str):
        return False
    t = text.strip()
    return len(t) >= 2 and t[0] in "[({" and t[-1] in "])}"


def _to_text_list(value):
    if _safe_is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []

    if _looks_like_serialized_list(text):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except Exception:
                pass

    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]

    return [text]


def _normalize_list_text_case(value):
    return [str(v).strip().lower() for v in _to_text_list(value) if str(v).strip()]


def _normalize_text_cell(value):
    if _safe_is_missing(value):
        return value
    if isinstance(value, (list, tuple, set)):
        return value
    return re.sub(r"\s+", " ", str(value).strip())


def _normalize_isbn_like(value):
    if _safe_is_missing(value):
        return value
    text = str(value).strip()
    if not text:
        return text
    if re.fullmatch(r"\d+(\.0+)?", text):
        text = str(int(float(text)))
    text = re.sub(r"[^0-9Xx]", "", text)
    if not text:
        return text
    if text[:-1].isdigit() if text.endswith(("X", "x")) else text.isdigit():
        if text.endswith(("X", "x")):
            body = text[:-1].lstrip("0") or "0"
            text = body + text[-1].upper()
        else:
            text = text.lstrip("0") or "0"
    return text.upper()


def _coerce_numeric(df, cols):
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].dropna()
        if len(non_null) and non_null.apply(float.is_integer).all():
            df[col] = df[col].astype("Int64")
    return df


def _drop_derived_subcolumns(fused_df, gold_df):
    gold_cols = set(gold_df.columns)
    keep_cols = []
    for col in fused_df.columns:
        if col in gold_cols or col.startswith("_") or col == "eval_id":
            keep_cols.append(col)
            continue
        parent = col.split("_", 1)[0]
        if parent in gold_cols and col not in gold_cols:
            continue
        keep_cols.append(col)
    return fused_df[keep_cols]


def _is_meaningful_gold_column(df, col):
    if col == "id" or col.endswith("_provenance"):
        return False
    if col not in df.columns:
        return False
    series = df[col]
    if series.notna().sum() == 0:
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    if non_null.astype(str).str.strip().eq("").all():
        return False
    return True


def _strip_unevaluable_subcols(fused_df, gold_df):
    gold_cols = set(gold_df.columns)
    fused_cols = list(fused_df.columns)
    to_drop = []
    for fc in fused_cols:
        if "_" in fc and fc not in gold_cols:
            parent = fc.split("_")[0]
            if parent in gold_cols:
                to_drop.append(fc)
    if to_drop:
        fused_df = fused_df.drop(columns=to_drop, errors="ignore")
        print(f"Dropped unevaluable sub-columns: {to_drop}")
    return fused_df


def _coerce_numeric_cols(*dfs):
    numeric_hints = {
        "year", "count", "score", "rating", "page", "pages", "sales", "price",
        "rank", "assets", "profits", "revenue", "duration", "founded"
    }
    for df in dfs:
        for col in df.columns:
            col_l = col.lower().replace("_", "").replace("-", "")
            if any(h in col_l for h in numeric_hints):
                try:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    if numeric.dropna().empty:
                        continue
                    if numeric.dropna().apply(float.is_integer).all():
                        df[col] = numeric.astype("Int64")
                    else:
                        df[col] = numeric
                except Exception:
                    pass


fused = pd.read_csv(FUSION_PATH)
gold = load_csv(EVAL_SET_PATH, name=f"{EVAL_STAGE}_set")
fused_eval = build_eval_view(fused, gold)

if "eval_id" not in fused_eval.columns:
    if "_id" in fused_eval.columns:
        fused_eval["eval_id"] = fused_eval["_id"].astype(str)
    else:
        fused_eval["eval_id"] = fused_eval.index.astype(str)

fused_eval = _drop_derived_subcolumns(fused_eval, gold)
fused_eval = _strip_unevaluable_subcols(fused_eval, gold)

gold_meaningful_cols = [
    col for col in gold.columns
    if _is_meaningful_gold_column(gold, col) and col in fused_eval.columns
]

for col in gold_meaningful_cols:
    fused_eval[col] = fused_eval[col].apply(_normalize_text_cell)
    gold[col] = gold[col].apply(_normalize_text_cell)

if "isbn_clean" in gold_meaningful_cols:
    fused_eval["isbn_clean"] = fused_eval["isbn_clean"].apply(_normalize_isbn_like)
    gold["isbn_clean"] = gold["isbn_clean"].apply(_normalize_isbn_like)

list_eval_columns = []
try:
    list_eval_columns = detect_list_like_columns(
        [
            fused_eval[["eval_id"] + [c for c in gold_meaningful_cols if c in fused_eval.columns]].copy(),
            gold[["id"] + gold_meaningful_cols].copy(),
        ],
        exclude_columns={"id", "eval_id", "_id", "_eval_cluster_id", "_fusion_confidence"},
    )
except Exception:
    list_eval_columns = []

list_eval_columns = [c for c in list_eval_columns if c in gold_meaningful_cols and not c.startswith("_")]

for candidate in ["genres", "categories", "tags", "tracks"]:
    if candidate in gold_meaningful_cols and candidate not in list_eval_columns:
        list_eval_columns.append(candidate)

for col in list_eval_columns:
    if col in fused_eval.columns:
        fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
    if col in gold.columns:
        gold[col] = gold[col].apply(_normalize_list_text_case)

country_eval_columns = [c for c in gold_meaningful_cols if "country" in c.lower()]
if country_eval_columns:
    country_format = _infer_country_output_format(gold, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        gold[col] = gold[col].apply(lambda x: _normalize_country_safe(x, country_format))

numeric_cols = [
    c for c in gold_meaningful_cols
    if any(token in c.lower() for token in ["year", "count", "rating", "score", "price", "pages", "page_count"])
]
fused_eval = _coerce_numeric(fused_eval, numeric_cols)
gold = _coerce_numeric(gold, numeric_cols)
_coerce_numeric_cols(fused_eval, gold)

gold_ids = set(gold["id"].dropna().astype(str).tolist()) if "id" in gold.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0

print(json.dumps({
    "stage": EVAL_STAGE,
    "id_alignment": {
        "gold_rows": len(gold_ids),
        "direct__id_coverage": direct_cov,
        "mapped_eval_id_coverage": mapped_cov,
    }
}, indent=2))

strategy = DataFusionStrategy("books_evaluation_strategy")
eval_funcs_summary = {}


def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name


for col in gold_meaningful_cols:
    c = col.lower()

    if c in {"isbn_clean"} or c.endswith("_id") or c in {"id", "code"}:
        register_eval(col, exact_match, "exact_match")
    elif c in {"publish_year", "founded", "year"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.0)
    elif c in {"page_count", "numratings"} or "count" in c:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.0)
    elif c in {"rating", "price", "score"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.01)
    elif c.startswith("is_") or c.startswith("has_") or c in {"available", "explicit"}:
        register_eval(col, boolean_match, "boolean_match")
    elif col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif "date" in c:
        register_eval(col, year_only_match, "year_only_match")
    elif c in {"title", "author", "publisher", "language"}:
        threshold = {
            "title": 0.85,
            "author": 0.85,
            "publisher": 0.80,
            "language": 0.90,
        }.get(c, 0.85)
        register_eval(col, tokenized_match, "tokenized_match", threshold=threshold)
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.80)

compact_summary = " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items()))
print(f"[EVAL_FUNCS] {compact_summary}")

evaluator = DataFusionEvaluator(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=os.path.join(EVAL_DIR, "debug_fusion_eval.jsonl"),
)


# [GUARDRAIL] Normalize list separators to match gold standard format
def _normalize_list_separators(fused_df, gold_df):
    """Align list-like column separators between fused and gold DataFrames."""
    _LIST_HINTS = {"genre", "categor", "tag", "topic", "keyword", "subject", "track"}
    shared = set(fused_df.columns) & set(gold_df.columns)
    for col in shared:
        col_l = col.lower()
        if not any(h in col_l for h in _LIST_HINTS):
            continue
        gold_sample = gold_df[col].dropna().astype(str).head(5).tolist()
        if not gold_sample:
            continue
        # Detect gold separator
        gold_sep = ", "
        for sep in ["; ", " | ", "|", " / ", "/"]:
            if any(sep in s for s in gold_sample):
                gold_sep = sep
                break
        # Normalize fused to match gold separator
        fused_sample = fused_df[col].dropna().astype(str).head(5).tolist()
        for sep in ["; ", " | ", "|", " / ", "/"]:
            if sep != gold_sep and any(sep in s for s in fused_sample):
                fused_df[col] = fused_df[col].astype(str).str.replace(sep, gold_sep)
                print(f"  Normalized {col} separator: '{sep}' -> '{gold_sep}'")
                break
    return fused_df

fused_eval = _normalize_list_separators(fused_eval, gold)
evaluation_results = evaluator.evaluate(
    fused_df=fused_eval,
    fused_id_column="eval_id",
    gold_df=gold,
    gold_id_column="id",
)

print(json.dumps({
    "stage": EVAL_STAGE,
    "evaluated_columns": gold_meaningful_cols,
    "list_like_columns": list_eval_columns,
    "metrics": evaluation_results,
}, indent=2, default=str))

with open(EVAL_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4, default=str)