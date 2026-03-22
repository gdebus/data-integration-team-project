import pandas as pd
import json
import ast
import re
import os
from pathlib import Path
import sys

from PyDI.io import load_csv
from PyDI.normalization import normalize_country
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
    from list_normalization import detect_list_like_columns
except ModuleNotFoundError:
    _candidates = [
        Path.cwd(),
        Path.cwd() / "agents",
        Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd(),
        (Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()),
        (Path(__file__).resolve().parent.parent.parent if "__file__" in globals() else Path.cwd()),
    ]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            if str(_path.resolve()) not in sys.path:
                sys.path.append(str(_path.resolve()))
    from list_normalization import detect_list_like_columns


# ================================
# 0. OUTPUT / INPUT PATHS
# ================================
OUTPUT_DIR = "output/runs/20260322_155058_books"
FUSION_PATH = "output/runs/20260322_155058_books/data_fusion/fusion_data.csv"
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_OUTPUT = "output/runs/20260322_155058_books/pipeline_evaluation/pipeline_evaluation.json"
DEBUG_OUTPUT = os.path.join(EVAL_DIR, "debug_fusion_eval.jsonl")

EVAL_STAGE = "validation"
GOLD_PATH = "input/datasets/books/testsets/validation_set.csv"

os.makedirs(EVAL_DIR, exist_ok=True)


# ================================
# 1. HELPERS
# ================================
def _safe_is_missing(value):
    if value is None:
        return True
    try:
        is_na = pd.isna(value)
        if hasattr(is_na, "__len__") and not isinstance(is_na, str):
            return False
        return bool(is_na)
    except Exception:
        return False


def _parse_source_ids(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text:
        return []
    if text[0] not in "[({":
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(v) for v in parsed if str(v)]
        except Exception:
            pass
    return []


def build_eval_view(fused_df, gold_df):
    gold_ids = set(gold_df["id"].dropna().astype(str).tolist()) if "id" in gold_df.columns else set()

    if "eval_id" in fused_df.columns:
        out = fused_df.copy()
        out["eval_id"] = out["eval_id"].astype(str)
        if "_fusion_confidence" in out.columns:
            out["_fusion_confidence"] = pd.to_numeric(out["_fusion_confidence"], errors="coerce").fillna(0.0)
            out = out.sort_values("_fusion_confidence", ascending=False)
        out = out.drop_duplicates(subset=["eval_id"], keep="first")
        return out

    expanded_rows = []
    for _, row in fused_df.iterrows():
        row_dict = row.to_dict()
        cluster_id = str(row_dict.get("_id", ""))
        source_ids = _parse_source_ids(row_dict.get("_fusion_sources"))
        candidates = [sid for sid in source_ids if sid in gold_ids]
        if not candidates and cluster_id in gold_ids:
            candidates = [cluster_id]

        seen = set()
        for eval_id in candidates:
            if eval_id in seen:
                continue
            seen.add(eval_id)
            enriched = dict(row_dict)
            enriched["eval_id"] = str(eval_id)
            enriched["_eval_cluster_id"] = cluster_id
            expanded_rows.append(enriched)

    if not expanded_rows:
        fallback = fused_df.copy()
        fallback["eval_id"] = fallback["_id"].astype(str) if "_id" in fallback.columns else fallback.index.astype(str)
        fallback["_eval_cluster_id"] = fallback["eval_id"]
        return fallback

    out = pd.DataFrame(expanded_rows)
    if "_fusion_confidence" in out.columns:
        out["_fusion_confidence"] = pd.to_numeric(out["_fusion_confidence"], errors="coerce").fillna(0.0)
        out = out.sort_values("_fusion_confidence", ascending=False)
    out = out.drop_duplicates(subset=["eval_id"], keep="first")
    return out


def _normalize_country_safe(value, output_format):
    if _safe_is_missing(value):
        return value
    text = str(value).strip()
    if not text:
        return value
    try:
        normalized = normalize_country(text, output_format=output_format)
        return normalized if normalized else text
    except Exception:
        return text


def _infer_country_output_format(gold_df, country_columns):
    values = []
    for col in country_columns:
        if col in gold_df.columns:
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


def _canonicalize_text_list(value):
    if _safe_is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            parsed_ok = False
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    if isinstance(parsed, (list, tuple, set)):
                        items = list(parsed)
                        parsed_ok = True
                        break
                except Exception:
                    continue
            if not parsed_ok:
                items = re.split(r"\s*[;,]\s*", text)
        else:
            items = re.split(r"\s*[;,]\s*", text)

    cleaned = []
    for item in items:
        t = str(item).strip().lower()
        if t:
            cleaned.append(t)
    return cleaned


def _coerce_numeric_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].dropna()
        if len(non_null) and non_null.apply(float.is_integer).all():
            df[col] = df[col].astype("Int64")
    return df


def _meaningful_common_columns(fused_df, gold_df):
    shared = []
    gold_cols = set(gold_df.columns)
    fused_cols = set(fused_df.columns)

    for col in sorted(gold_cols & fused_cols):
        if col in {"id", "_id", "eval_id", "_eval_cluster_id"}:
            continue
        if str(col).startswith("_"):
            continue

        gold_non_null = gold_df[col].dropna()
        fused_non_null = fused_df[col].dropna()

        if len(gold_non_null) == 0 or len(fused_non_null) == 0:
            continue

        shared.append(col)

    return shared


def _strip_unevaluable_subcols(fused_df, gold_df):
    gold_cols = set(gold_df.columns)
    fused_cols = list(fused_df.columns)
    to_drop = []
    for fc in fused_cols:
        if fc in {"eval_id", "_id", "_eval_cluster_id", "_fusion_confidence", "_fusion_sources"}:
            continue
        if "_" in fc and fc not in gold_cols:
            parent = fc.split("_")[0]
            if parent in gold_cols:
                to_drop.append(fc)
    if to_drop:
        fused_df = fused_df.drop(columns=to_drop, errors="ignore")
        print(f"[DROP SUBCOLS] dropped={to_drop}")
    return fused_df


def _serialize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_serialize_for_json(v) for v in obj]
    if isinstance(obj, set):
        return [_serialize_for_json(v) for v in sorted(obj, key=lambda x: str(x))]
    if pd.isna(obj) if not isinstance(obj, (list, tuple, set, dict)) else False:
        return None
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return obj


# ================================
# 2. LOAD DATA
# ================================
fused = pd.read_csv(FUSION_PATH)
gold = load_csv(GOLD_PATH, name=f"{EVAL_STAGE}_set")
fused_eval = build_eval_view(fused, gold)

print(f"[INFO] evaluation_stage={EVAL_STAGE}")
print(f"[INFO] fused_path={FUSION_PATH}")
print(f"[INFO] gold_path={GOLD_PATH}")
print(f"[INFO] fused_rows={len(fused)} mapped_eval_rows={len(fused_eval)} gold_rows={len(gold)}")

gold_ids = set(gold["id"].dropna().astype(str).tolist()) if "id" in gold.columns else set()
mapped_cov = len(set(fused_eval["eval_id"].dropna().astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0
print(f"[ID ALIGNMENT] mapped eval_id coverage: {mapped_cov}/{len(gold_ids)}")


# ================================
# 3. DROP PROVENANCE COLUMNS / NON-1:1 DERIVED COLUMNS
# ================================
gold = gold[[c for c in gold.columns if not str(c).endswith("_provenance")]].copy()
fused_eval = _strip_unevaluable_subcols(fused_eval, gold)

shared_columns = _meaningful_common_columns(fused_eval, gold)

keep_fused_cols = [c for c in fused_eval.columns if c in shared_columns or c in {"eval_id", "_id", "_eval_cluster_id", "_fusion_confidence", "_fusion_sources"}]
keep_gold_cols = [c for c in gold.columns if c in shared_columns or c == "id"]

fused_eval = fused_eval[keep_fused_cols].copy()
gold = gold[keep_gold_cols].copy()


# ================================
# 4. NORMALIZE REPRESENTATION TO GOLD STYLE
# ================================
numeric_like_cols = [
    c for c in shared_columns
    if any(k in c.lower() for k in ["year", "count", "rating", "score", "price", "page"])
]

fused_eval = _coerce_numeric_columns(fused_eval, numeric_like_cols)
gold = _coerce_numeric_columns(gold, numeric_like_cols)

list_eval_columns = detect_list_like_columns(
    [fused_eval, gold],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence", "_fusion_sources", "isbn_clean"},
)

if "genres" in shared_columns and "genres" not in list_eval_columns:
    list_eval_columns.append("genres")

list_eval_columns = [c for c in list_eval_columns if c in shared_columns and not str(c).startswith("_")]

for col in list_eval_columns:
    if col in fused_eval.columns:
        fused_eval[col] = fused_eval[col].apply(_canonicalize_text_list)
    if col in gold.columns:
        gold[col] = gold[col].apply(_canonicalize_text_list)

country_eval_columns = [c for c in shared_columns if "country" in c.lower()]
if country_eval_columns:
    country_format = _infer_country_output_format(gold, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        gold[col] = gold[col].apply(lambda x: _normalize_country_safe(x, country_format))
    print(f"[COUNTRY NORMALIZATION] format={country_format} columns={country_eval_columns}")

print(f"[LIST NORMALIZATION] columns={list_eval_columns if list_eval_columns else []}")
print(f"[NUMERIC COERCION] columns={numeric_like_cols if numeric_like_cols else []}")


# ================================
# 5. REGISTER EVALUATION FUNCTIONS
# ================================
strategy = DataFusionStrategy("books_evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, label, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items())
    eval_funcs_summary[column] = f"{label}({args})" if args else label


for col in shared_columns:
    col_l = col.lower()

    if col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif col in {"isbn_clean"} or col_l.endswith("_id") or col_l in {"ean", "upc"}:
        register_eval(col, exact_match, "exact_match")
    elif any(k in col_l for k in ["year", "page_count", "count", "rating", "score", "price"]):
        if "year" in col_l:
            register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.0)
        elif "rating" in col_l or "score" in col_l:
            register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.05)
        elif "price" in col_l:
            register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.01)
        else:
            register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.0)
    elif any(k in col_l for k in ["date", "founded", "released"]):
        register_eval(col, year_only_match, "year_only_match")
    elif col_l.startswith("is_") or col_l.startswith("has_") or col_l in {"active", "available"}:
        register_eval(col, boolean_match, "boolean_match")
    else:
        threshold = 0.85
        if col_l in {"language"}:
            threshold = 0.90
        register_eval(col, tokenized_match, "tokenized_match", threshold=threshold)

compact_summary = " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items()))
print(f"[EVAL FUNCTIONS] {compact_summary}")


# ================================
# 6. RUN EVALUATION
# ================================
evaluator = DataFusionEvaluator(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=DEBUG_OUTPUT,
)


# [GUARDRAIL] Coerce numeric-like columns to consistent types
def _coerce_numeric_cols(*dfs):
    """Cast numeric-like columns to Int64 in all DataFrames to prevent float vs int mismatches."""
    _NUMERIC_HINTS = {"year", "count", "score", "rating", "page", "sales", "price",
                      "rank", "assets", "profits", "revenue", "duration", "founded"}
    for df in dfs:
        for col in df.columns:
            col_l = col.lower().replace("_", "").replace("-", "")
            if any(h in col_l for h in _NUMERIC_HINTS):
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

_coerce_numeric_cols(fused_eval, gold)

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


# ================================
# 7. PRINT STRUCTURED METRICS
# ================================
safe_results = _serialize_for_json(evaluation_results)
print("[EVALUATION RESULTS]")
print(json.dumps(safe_results, indent=2, ensure_ascii=False))


# ================================
# 8. SAVE RESULTS
# ================================
with open(EVAL_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(safe_results, f, indent=2, ensure_ascii=False)

print(f"[DONE] wrote evaluation to {EVAL_OUTPUT}")