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
from PyDI.normalization import normalize_country

try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
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
            resolved = str(_path.resolve())
            if resolved not in sys.path:
                sys.path.append(resolved)
    from list_normalization import detect_list_like_columns, normalize_list_like_columns


# ================================
# 0. PATHS
# ================================
OUTPUT_DIR = "output/runs/20260322_155058_books"
FUSION_PATH = "output/runs/20260322_155058_books/data_fusion/fusion_data.csv"
EVAL_SET_PATH = "input/datasets/books/testsets/validation_set.csv"
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)


# ================================
# 1. HELPERS
# ================================
def _safe_is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    try:
        out = pd.isna(value)
        return bool(out) if not hasattr(out, "__array__") else False
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
        source_ids = _parse_source_ids(row_dict.get("_fusion_sources"))

        candidates = []
        if "eval_id" in row_dict and not _safe_is_missing(row_dict.get("eval_id")):
            candidates.append(str(row_dict["eval_id"]))
        if gold_prefix:
            candidates.extend([sid for sid in source_ids if sid.startswith(gold_prefix)])
        candidates.extend([sid for sid in source_ids if sid in gold_ids])

        if not candidates and cluster_id:
            candidates = [cluster_id]

        seen = set()
        for eval_id in candidates:
            if not eval_id or eval_id in seen:
                continue
            seen.add(eval_id)
            enriched = dict(row_dict)
            enriched["eval_id"] = str(eval_id)
            enriched["_eval_cluster_id"] = cluster_id
            expanded_rows.append(enriched)

    if not expanded_rows:
        fallback = fused_df.copy()
        fallback["eval_id"] = fallback["eval_id"].astype(str) if "eval_id" in fallback.columns else fallback["_id"].astype(str)
        fallback["_eval_cluster_id"] = fallback["_id"].astype(str) if "_id" in fallback.columns else fallback["eval_id"].astype(str)
        return fallback

    fused_eval = pd.DataFrame(expanded_rows)
    if "_fusion_confidence" in fused_eval.columns:
        fused_eval["_fusion_confidence"] = pd.to_numeric(fused_eval["_fusion_confidence"], errors="coerce").fillna(0.0)
        fused_eval = fused_eval.sort_values("_fusion_confidence", ascending=False)
    fused_eval = fused_eval.drop_duplicates(subset=["eval_id"], keep="first")
    return fused_eval


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


def _split_text_list(value):
    if _safe_is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        parts = [str(v).strip() for v in value if str(v).strip()]
        return parts

    text = str(value).strip()
    if not text:
        return []

    if text[0] in "[({":
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except Exception:
                pass

    parts = re.split(r"\s*[,;|]\s*", text)
    return [p.strip() for p in parts if p.strip()]


def _canonicalize_list_text(value):
    items = _split_text_list(value)
    canon = []
    seen = set()
    for item in items:
        x = re.sub(r"\s+", " ", str(item).strip().lower())
        if x and x not in seen:
            seen.add(x)
            canon.append(x)
    return canon


def _coerce_numeric_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].dropna()
        if len(non_null) > 0:
            try:
                if non_null.apply(float.is_integer).all():
                    df[col] = df[col].astype("Int64")
            except Exception:
                pass
    return df


def _has_meaningful_values(df, col):
    if col not in df.columns:
        return False
    series = df[col]
    if series.dropna().empty:
        return False
    if series.astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}).dropna().empty:
        return False
    return True


def _compact_json(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ================================
# 2. LOAD DATA
# ================================
fused = pd.read_csv(FUSION_PATH)
gold = load_csv(EVAL_SET_PATH, name="validation_set")

fused_eval = build_eval_view(fused, gold)

# use eval_id if present, per requirement
if "eval_id" not in fused_eval.columns:
    raise ValueError("Expected an eval_id column in fused evaluation view, but none was found.")

# ================================
# 3. DROP NON-DIRECT / PROVENANCE COLUMNS
# ================================
gold_direct_columns = {
    c for c in gold.columns
    if not str(c).endswith("_provenance")
}

fused_eval = fused_eval[[c for c in fused_eval.columns if not (c.startswith("_") and c not in {"_id"}) or c in {"eval_id"}]]

# Keep only columns with direct gold matches plus technical IDs
keep_cols = {"eval_id", "_id"} | (set(fused_eval.columns) & gold_direct_columns)
fused_eval = fused_eval[[c for c in fused_eval.columns if c in keep_cols]]

# ================================
# 4. SHARED NORMALIZATION TO VALIDATION REPRESENTATION
# ================================
# Numeric normalization
numeric_candidates = [
    c for c in set(fused_eval.columns) & set(gold.columns)
    if any(k in c.lower() for k in ["year", "count", "rating", "score", "price", "isbn", "page"])
]
fused_eval = _coerce_numeric_columns(fused_eval, numeric_candidates)
gold = _coerce_numeric_columns(gold, numeric_candidates)

# List/text-list normalization
common_nontech = [
    c for c in (set(fused_eval.columns) & set(gold.columns))
    if c not in {"id", "eval_id", "_id"}
    and not c.endswith("_provenance")
]

list_like_columns = detect_list_like_columns(
    [fused_eval, gold],
    exclude_columns={"id", "eval_id", "_id"},
)

# Force known textual multi-value column(s) from validation representation
for forced_col in ["genres"]:
    if forced_col in common_nontech and forced_col not in list_like_columns:
        list_like_columns.append(forced_col)

list_like_columns = [c for c in list_like_columns if c in common_nontech]

if list_like_columns:
    try:
        fused_eval, gold = normalize_list_like_columns([fused_eval, gold], list_like_columns)
    except Exception:
        pass
    for col in list_like_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_canonicalize_list_text)
        if col in gold.columns:
            gold[col] = gold[col].apply(_canonicalize_list_text)

# Country normalization if any
country_columns = [c for c in common_nontech if "country" in c.lower()]
if country_columns:
    country_format = _infer_country_output_format(gold, country_columns)
    for col in country_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        gold[col] = gold[col].apply(lambda x: _normalize_country_safe(x, country_format))
else:
    country_format = None

# ================================
# 5. SELECT EVALUABLE COLUMNS ONLY
# ================================
candidate_columns = []
for col in common_nontech:
    if col in {"id", "eval_id", "_id"}:
        continue
    if col.endswith("_provenance"):
        continue
    if not _has_meaningful_values(fused_eval, col):
        continue
    if not _has_meaningful_values(gold, col):
        continue
    candidate_columns.append(col)

# Avoid evaluating derived sub-columns without direct gold counterpart
candidate_columns = [c for c in candidate_columns if c in gold.columns]

# ================================
# 6. REGISTER EVALUATION FUNCTIONS
# ================================
strategy = DataFusionStrategy("books_validation_evaluation")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

for col in candidate_columns:
    cl = col.lower()

    if col in {"id", "eval_id", "_id"}:
        register_eval(col, exact_match, "exact_match")
    elif col in list_like_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif any(k in cl for k in ["isbn", "ean", "upc", "doi", "issn"]):
        register_eval(col, exact_match, "exact_match")
    elif any(k in cl for k in ["publish_year", "page_count", "numratings", "rating", "score", "price", "count"]):
        tol = 0.0 if cl in {"publish_year", "page_count", "numratings"} else 0.01
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=tol)
    elif "date" in cl and "year" not in cl:
        register_eval(col, year_only_match, "year_only_match")
    elif "country" in cl:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.65)
    elif any(k in cl for k in ["title", "author", "publisher", "language", "name", "label", "city", "street"]):
        threshold = 0.85
        if "publisher" in cl:
            threshold = 0.80
        if "language" in cl:
            threshold = 0.90
        register_eval(col, tokenized_match, "tokenized_match", threshold=threshold)
    elif any(k in cl for k in ["flag", "is_", "has_", "enabled", "active", "available"]):
        register_eval(col, boolean_match, "boolean_match")
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)

# compact one-line summary
ordered_summary = {k: eval_funcs_summary[k] for k in sorted(eval_funcs_summary)}
print("[EVAL_FUNCTIONS]", _compact_json(ordered_summary))

# ================================
# 7. ALIGN IDS AND RUN EVALUATION
# ================================
gold_ids = set(gold["id"].dropna().astype(str).tolist()) if "id" in gold.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids)

print("[ID_ALIGNMENT]", _compact_json({
    "direct__id_coverage": {"matched": direct_cov, "total_gold": len(gold_ids)},
    "mapped_eval_id_coverage": {"matched": mapped_cov, "total_gold": len(gold_ids)},
    "country_format": country_format,
    "list_like_columns": sorted(list_like_columns),
    "evaluated_columns": sorted(candidate_columns),
}))

evaluator = DataFusionEvaluator(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=os.path.join(EVAL_DIR, "debug_fusion_eval.jsonl"),
)


# [GUARDRAIL] Strip unevaluable sub-columns from evaluation
def _strip_unevaluable_subcols(fused_df, gold_df):
    """Drop fused sub-columns that have no matching gold column (e.g. tracks_track_name when gold only has tracks)."""
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
        print(f"  Dropped {len(to_drop)} unevaluable sub-column(s): {to_drop}")
    return fused_df

fused_eval = _strip_unevaluable_subcols(fused_eval, gold)

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
    expected_df=gold,
    expected_id_column="id",
)

# structured metrics print
print("[EVALUATION_METRICS]", json.dumps(evaluation_results, indent=2, ensure_ascii=False))

# ================================
# 8. WRITE RESULTS
# ================================
evaluation_output = "output/runs/20260322_155058_books/pipeline_evaluation/pipeline_evaluation.json"
with open(evaluation_output, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)