import pandas as pd
import json
import ast
import os
import re
import sys
from pathlib import Path

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
            if str(_path.resolve()) not in sys.path:
                sys.path.append(str(_path.resolve()))
    from list_normalization import detect_list_like_columns, normalize_list_like_columns


# === 0. OUTPUT DIRECTORY ===
OUTPUT_DIR = "output/runs/20260323_041229_restaurant/"
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)


# ================================
# HELPERS
# ================================

def _safe_is_missing(value):
    if value is None:
        return True
    try:
        result = pd.isna(value)
        if hasattr(result, "__len__") and not isinstance(result, str):
            return False
        return bool(result)
    except Exception:
        return False


def _parse_list_like(value):
    if _safe_is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if not _safe_is_missing(v) and str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple, set)):
                out = []
                for item in parsed:
                    if isinstance(item, (list, tuple, set)):
                        out.extend(str(v).strip() for v in item if not _safe_is_missing(v) and str(v).strip())
                    else:
                        item_text = str(item).strip()
                        if item_text and item_text.lower() != "nan":
                            nested_done = False
                            if item_text[:1] in "[({":
                                for nested_parser in (json.loads, ast.literal_eval):
                                    try:
                                        nested = nested_parser(item_text)
                                        if isinstance(nested, (list, tuple, set)):
                                            out.extend(
                                                str(v).strip()
                                                for v in nested
                                                if not _safe_is_missing(v) and str(v).strip() and str(v).strip().lower() != "nan"
                                            )
                                            nested_done = True
                                            break
                                    except Exception:
                                        pass
                            if not nested_done:
                                out.append(item_text)
                return out
        except Exception:
            pass
    if ";" in text:
        return [t.strip() for t in text.split(";") if t.strip() and t.strip().lower() != "nan"]
    return [text] if text.lower() != "nan" else []


def _normalize_list_text_case(value):
    items = _parse_list_like(value)
    return sorted({str(v).strip().lower() for v in items if str(v).strip()})


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


def _detect_gold_prefixes(gold_ids):
    prefixes = set()
    for g in gold_ids:
        g = str(g)
        if g.startswith("yelp-"):
            prefixes.add("yelp-")
        elif g.startswith("uber_"):
            prefixes.add("uber_")
        elif g.startswith("kaggle380k-"):
            prefixes.add("kaggle380k-")
    return prefixes


def build_eval_view(fused_df, gold_df):
    gold_ids = set(gold_df["id"].dropna().astype(str).tolist()) if "id" in gold_df.columns else set()
    gold_prefixes = _detect_gold_prefixes(gold_ids)
    expanded_rows = []

    for _, row in fused_df.iterrows():
        row_dict = row.to_dict()
        cluster_id = str(row_dict.get("_id", ""))
        source_ids = _parse_source_ids(row_dict.get("_fusion_sources"))
        candidates = [sid for sid in source_ids if sid in gold_ids]

        if not candidates and gold_prefixes:
            candidates = [sid for sid in source_ids if any(sid.startswith(p) for p in gold_prefixes)]

        if not candidates and "eval_id" in row_dict and str(row_dict.get("eval_id", "")).strip():
            candidates = [str(row_dict["eval_id"])]

        if not candidates and cluster_id:
            candidates = [cluster_id]

        seen = set()
        for eval_id in candidates:
            if eval_id in seen:
                continue
            seen.add(eval_id)
            enriched = dict(row_dict)
            enriched["eval_id"] = eval_id
            enriched["_eval_cluster_id"] = cluster_id
            expanded_rows.append(enriched)

    if not expanded_rows:
        fallback = fused_df.copy()
        fallback["eval_id"] = fallback["eval_id"].astype(str) if "eval_id" in fallback.columns else fallback["_id"].astype(str)
        fallback["_eval_cluster_id"] = fallback["_id"].astype(str)
        return fallback

    fused_eval = pd.DataFrame(expanded_rows)
    if "_fusion_confidence" in fused_eval.columns:
        fused_eval["_fusion_confidence"] = pd.to_numeric(fused_eval["_fusion_confidence"], errors="coerce").fillna(0.0)
        fused_eval = fused_eval.sort_values("_fusion_confidence", ascending=False)
    fused_eval = fused_eval.drop_duplicates(subset=["eval_id"], keep="first")
    return fused_eval


def _coerce_numeric_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].dropna()
        if len(non_null) > 0:
            try:
                if non_null.apply(lambda x: float(x).is_integer()).all():
                    df[col] = df[col].astype("Int64")
            except Exception:
                pass
    return df


def _meaningful_in_both(fused_df, gold_df, col):
    if col not in fused_df.columns or col not in gold_df.columns:
        return False
    fused_non_null = fused_df[col].dropna()
    gold_non_null = gold_df[col].dropna()
    return len(fused_non_null) > 0 and len(gold_non_null) > 0


# ================================
# 1. LOAD FUSED OUTPUT AND GOLD SET
# ================================

fused = pd.read_csv(os.path.join(FUSION_DIR, "fusion_data.csv"))
gold = pd.read_csv("input/datasets/restaurant/testsets/Restaurant_Fusion_Validation_Set.csv")

fused_eval = build_eval_view(fused, gold)

# ================================
# 2. NORMALIZE LIST-LIKE COLUMNS TO GOLD REPRESENTATION
# ================================

list_eval_columns = detect_list_like_columns(
    [fused_eval, gold],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if not c.startswith("_")]

if list_eval_columns:
    fused_eval, gold = normalize_list_like_columns([fused_eval, gold], list_eval_columns)
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
        if col in gold.columns:
            gold[col] = gold[col].apply(_normalize_list_text_case)
    print(f"[LIST NORMALIZATION] columns={list_eval_columns}")

# ================================
# 3. COERCE NUMERIC-LIKE COLUMNS CONSISTENTLY
# ================================

numeric_candidates = [
    "latitude", "longitude", "rating", "rating_count",
    "postal_code", "house_number", "phone_raw", "phone_e164"
]
numeric_eval_columns = [c for c in numeric_candidates if c in fused_eval.columns and c in gold.columns]
fused_eval = _coerce_numeric_columns(fused_eval, numeric_eval_columns)
gold = _coerce_numeric_columns(gold, numeric_eval_columns)

# keep phone/postal as string-ish IDs for robust comparison if both are mostly identifier-like
for col in ["postal_code", "phone_raw", "phone_e164", "house_number"]:
    if col in fused_eval.columns and col in gold.columns:
        fused_eval[col] = fused_eval[col].astype("string")
        gold[col] = gold[col].astype("string")

# text cleanup aligned to validation representation
shared_text_cols = set(fused_eval.columns).intersection(gold.columns)
for col in shared_text_cols:
    if col in list_eval_columns:
        continue
    if col.startswith("_"):
        continue
    if col in numeric_candidates:
        continue
    if pd.api.types.is_object_dtype(fused_eval[col]) or pd.api.types.is_string_dtype(fused_eval[col]):
        fused_eval[col] = fused_eval[col].apply(lambda x: str(x).strip() if not _safe_is_missing(x) else x)
    if pd.api.types.is_object_dtype(gold[col]) or pd.api.types.is_string_dtype(gold[col]):
        gold[col] = gold[col].apply(lambda x: str(x).strip() if not _safe_is_missing(x) else x)

# ================================
# 4. ID ALIGNMENT DIAGNOSTICS
# ================================

gold_ids = set(gold["id"].dropna().astype(str).tolist()) if "id" in gold.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
eval_cov = len(set(fused["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0

print(json.dumps({
    "stage": "validation",
    "rows_fused_raw": int(len(fused)),
    "rows_fused_eval": int(len(fused_eval)),
    "rows_gold": int(len(gold)),
    "id_alignment": {
        "direct__id_coverage": f"{direct_cov}/{len(gold_ids)}",
        "raw_eval_id_coverage": f"{eval_cov}/{len(gold_ids)}",
        "mapped_eval_id_coverage": f"{mapped_cov}/{len(gold_ids)}",
    }
}, indent=2))

# ================================
# 5. REGISTER EVALUATION FUNCTIONS
# ================================

strategy = DataFusionStrategy("restaurant_validation_evaluation")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items())
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

# text columns
for col, threshold in [
    ("name", 0.85),
    ("name_norm", 0.85),
    ("address_line1", 0.80),
    ("address_line2", 0.80),
    ("street", 0.75),
    ("city", 0.85),
    ("state", 0.90),
    ("country", 0.80),   # no country normalization per instructions
    ("website", 0.75),
    ("map_url", 0.70),
    ("source", 0.85),
]:
    if _meaningful_in_both(fused_eval, gold, col):
        register_eval(col, tokenized_match, "tokenized_match", threshold=threshold)

# exact/code/id-like columns
for col in ["postal_code", "phone_raw", "phone_e164", "house_number", "kaggle380k_id", "yelp_id", "uber_eats_id"]:
    if _meaningful_in_both(fused_eval, gold, col):
        register_eval(col, exact_match, "exact_match")

# numeric columns
for col, tol in [
    ("latitude", 0.001),
    ("longitude", 0.001),
    ("rating", 0.05),
    ("rating_count", 0.01),
]:
    if _meaningful_in_both(fused_eval, gold, col):
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=tol)

# date-like if present
for col in ["founded", "year", "publish_year"]:
    if _meaningful_in_both(fused_eval, gold, col):
        register_eval(col, year_only_match, "year_only_match")

# boolean-like if present
for col in fused_eval.columns:
    if col in gold.columns and re.search(r"(is_|has_|_flag$|_bool$|^active$)", str(col)):
        if _meaningful_in_both(fused_eval, gold, col):
            register_eval(col, boolean_match, "boolean_match")

# list-like columns
for col in list_eval_columns:
    if _meaningful_in_both(fused_eval, gold, col):
        register_eval(col, set_equality_match, "set_equality_match")

# compact one-line summary
summary_line = " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items()))
print(f"[EVAL_FUNCS] {summary_line}")

# ================================
# 6. RUN EVALUATION
# ================================

evaluator = DataFusionEvaluator(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=os.path.join(EVAL_DIR, "debug_fusion_eval.jsonl"),
    fusion_debug_logs=os.path.join(FUSION_DIR, "debug_fusion_data.jsonl"),
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
                    if numeric.dropna().apply(lambda x: float(x).is_integer()).all():
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

print(json.dumps({
    "stage": "validation",
    "evaluation_output": os.path.join(EVAL_DIR, "pipeline_evaluation.json"),
    "metrics": evaluation_results
}, indent=2, default=str))

# ================================
# 8. WRITE RESULTS
# ================================

evaluation_output = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
with open(evaluation_output, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4, default=str)