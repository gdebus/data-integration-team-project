import pandas as pd
import json
import ast
import os
import re
import sys
from pathlib import Path

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
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _candidates = [
        Path.cwd(),
        Path.cwd() / "agents",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
    ]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            if str(_path.resolve()) not in sys.path:
                sys.path.append(str(_path.resolve()))
    from list_normalization import detect_list_like_columns, normalize_list_like_columns


# ================================
# 0. OUTPUT DIRECTORY
# ================================
OUTPUT_DIR = "output/runs/20260323_041229_restaurant/"
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
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
                        out.extend([str(v).strip() for v in item if not _safe_is_missing(v) and str(v).strip()])
                    else:
                        item_text = str(item).strip()
                        if item_text.startswith("[") and item_text.endswith("]"):
                            try:
                                nested = ast.literal_eval(item_text)
                                if isinstance(nested, (list, tuple, set)):
                                    out.extend(
                                        [str(v).strip() for v in nested if not _safe_is_missing(v) and str(v).strip()]
                                    )
                                    continue
                            except Exception:
                                pass
                        if item_text and item_text.lower() != "nan":
                            out.append(item_text)
                return out
        except Exception:
            pass

    if ";" in text:
        return [p.strip() for p in text.split(";") if p.strip() and p.strip().lower() != "nan"]

    return [text] if text.lower() != "nan" else []


def _normalize_list_text_case(value):
    if _safe_is_missing(value):
        return value
    parsed = _parse_list_like(value)
    normalized = []
    for v in parsed:
        t = str(v).strip().lower()
        if t and t != "nan":
            normalized.append(t)
    return sorted(set(normalized))


def _coerce_numeric_column(df, col):
    if col not in df.columns:
        return
    df[col] = pd.to_numeric(df[col], errors="coerce")
    non_null = df[col].dropna()
    if len(non_null) > 0:
        try:
            if non_null.apply(lambda x: float(x).is_integer()).all():
                df[col] = df[col].astype("Int64")
        except Exception:
            pass


def _meaningful_in_both(fused_df, gold_df, col):
    if col not in fused_df.columns or col not in gold_df.columns:
        return False
    fused_non_null = fused_df[col].dropna()
    gold_non_null = gold_df[col].dropna()
    return len(fused_non_null) > 0 and len(gold_non_null) > 0


def _looks_nested_structural(col):
    parts = str(col).split("_")
    return len(parts) >= 3 and parts[0] not in {"address", "phone", "postal", "name", "rating"}


# ================================
# 2. LOAD FUSED OUTPUT AND GOLD SET
# ================================
fused = pd.read_csv(os.path.join(FUSION_DIR, "fusion_data.csv"))
gold = load_csv(
    "input/datasets/restaurant/testsets/Restaurant_Fusion_Test_Set.csv",
    name="fusion_test_set",
)

# Use precomputed eval_id if present; otherwise create a fallback.
if "eval_id" not in fused.columns:
    fused["eval_id"] = fused["_id"].astype(str) if "_id" in fused.columns else fused["id"].astype(str)

fused["eval_id"] = fused["eval_id"].astype(str)
gold["id"] = gold["id"].astype(str)

# Keep best fused row per eval_id if duplicates exist.
if "_fusion_confidence" in fused.columns:
    fused["_fusion_confidence"] = pd.to_numeric(fused["_fusion_confidence"], errors="coerce").fillna(0.0)
    fused = fused.sort_values("_fusion_confidence", ascending=False)
fused_eval = fused.drop_duplicates(subset=["eval_id"], keep="first").copy()


# ================================
# 3. ALIGN REPRESENTATIONS
# Use validation/test-set representation as target.
# Do NOT normalize countries with normalize_country.
# ================================
# Numeric columns relevant for restaurant fusion
numeric_cols = [
    "latitude",
    "longitude",
    "rating",
    "rating_count",
    "postal_code",
    "house_number",
    "phone_e164",
    "phone_raw",
    "_fusion_confidence",
]

for col in numeric_cols:
    if col in fused_eval.columns:
        _coerce_numeric_column(fused_eval, col)
    if col in gold.columns:
        _coerce_numeric_column(gold, col)

# Detect list-like columns jointly, then canonicalize consistently.
list_eval_columns = detect_list_like_columns(
    [fused_eval, gold],
    exclude_columns={"id", "_id", "eval_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if not str(c).startswith("_")]

if list_eval_columns:
    fused_eval, gold = normalize_list_like_columns([fused_eval, gold], list_eval_columns)
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
        if col in gold.columns:
            gold[col] = gold[col].apply(_normalize_list_text_case)
    print(f"[LIST NORMALIZATION] columns={list_eval_columns}")

# Light text cleanup on shared non-list text columns.
shared_cols = set(fused_eval.columns) & set(gold.columns)
text_like_cols = [
    c for c in shared_cols
    if c not in list_eval_columns
    and c not in numeric_cols
    and c not in {"id", "_id", "eval_id", "_fusion_sources", "_fusion_source_datasets", "_fusion_metadata"}
]

for col in text_like_cols:
    fused_eval[col] = fused_eval[col].apply(lambda x: str(x).strip() if not _safe_is_missing(x) else x)
    gold[col] = gold[col].apply(lambda x: str(x).strip() if not _safe_is_missing(x) else x)


# ================================
# 4. ID ALIGNMENT DIAGNOSTICS
# ================================
gold_ids = set(gold["id"].dropna().astype(str).tolist())
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
eval_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids)

print(json.dumps({
    "stage": "sealed_test",
    "id_alignment": {
        "gold_records": len(gold_ids),
        "direct__id_coverage": direct_cov,
        "eval_id_coverage": eval_cov,
    }
}, indent=2))


# ================================
# 5. REGISTER EVALUATION FUNCTIONS
# Only evaluate columns meaningful in BOTH datasets.
# Avoid strict exact matching except for identifiers/booleans/codes.
# ================================
strategy = DataFusionStrategy("restaurant_evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

candidate_columns = [
    c for c in sorted(shared_cols)
    if c not in {
        "_id", "id", "eval_id", "_fusion_sources", "_fusion_source_datasets",
        "_fusion_metadata", "_fusion_confidence",
        "kaggle380k_id", "yelp_id", "uber_eats_id"
    }
    and not _looks_nested_structural(c)
    and _meaningful_in_both(fused_eval, gold, c)
]

for col in candidate_columns:
    col_l = col.lower()

    if col in list_eval_columns or col_l == "categories":
        register_eval(col, set_equality_match, "set_equality_match")
    elif col_l in {"latitude", "longitude"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.001)
    elif col_l in {"rating"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.15)
    elif col_l in {"rating_count", "postal_code", "house_number", "phone_e164", "phone_raw"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.0)
    elif "date" in col_l or "founded" in col_l or "year" in col_l:
        register_eval(col, year_only_match, "year_only_match")
    elif col_l.startswith("is_") or col_l.startswith("has_") or col_l in {"open", "closed", "active"}:
        register_eval(col, boolean_match, "boolean_match")
    elif col_l in {"country"}:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.65)
    elif col_l in {"state", "city", "street", "address_line1", "address_line2", "website", "map_url", "source", "name", "name_norm"}:
        threshold = {
            "name": 0.85,
            "name_norm": 0.90,
            "city": 0.80,
            "state": 0.80,
            "street": 0.75,
            "address_line1": 0.75,
            "address_line2": 0.75,
            "website": 0.75,
            "map_url": 0.70,
            "source": 0.85,
        }.get(col_l, 0.80)
        register_eval(col, tokenized_match, "tokenized_match", threshold=threshold)
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.80)

print("[EVAL_FUNCS]", " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items())))


# ================================
# 6. RUN EVALUATION
# IMPORTANT: use fused_id_column='eval_id'
# ================================
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
    "stage": "sealed_test",
    "evaluation_functions": eval_funcs_summary,
    "metrics": evaluation_results,
}, indent=2, default=str))


# ================================
# 8. WRITE RESULTS
# ================================
evaluation_output = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
with open(evaluation_output, "w") as f:
    json.dump(evaluation_results, f, indent=4, default=str)