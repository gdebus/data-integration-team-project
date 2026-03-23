import pandas as pd
import json
import ast
import os
import sys
from pathlib import Path

from PyDI.io import load_csv
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEvaluator,
    tokenized_match,
    set_equality_match,
    numeric_tolerance_match,
    exact_match,
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


# ================================
# 0. OUTPUT / INPUT PATHS
# ================================
OUTPUT_DIR = "output/runs/20260323_041229_restaurant"
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

FUSED_PATH = "output/runs/20260323_041229_restaurant/data_fusion/fusion_data.csv"
GOLD_PATH = "input/datasets/restaurant/testsets/Restaurant_Fusion_Validation_Set.csv"
EVAL_OUTPUT_PATH = "output/runs/20260323_041229_restaurant/pipeline_evaluation/pipeline_evaluation.json"


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


def _parse_possible_list(value):
    if _safe_is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if not _safe_is_missing(v) and str(v).strip() and str(v).strip().lower() != "nan"]

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
                        out.extend(
                            str(v).strip() for v in item
                            if not _safe_is_missing(v) and str(v).strip() and str(v).strip().lower() != "nan"
                        )
                    else:
                        s = str(item).strip()
                        if s and s.lower() != "nan":
                            out.append(s)
                return out
        except Exception:
            pass

    if ";" in text:
        parts = [p.strip() for p in text.split(";")]
        return [p for p in parts if p and p.lower() != "nan"]

    return [text] if text.lower() != "nan" else []


def _normalize_text_scalar(value):
    if _safe_is_missing(value):
        return value
    text = str(value).strip()
    if not text:
        return text
    return text.lower()


def _normalize_list_value(value):
    parsed = _parse_possible_list(value)
    cleaned = []
    for x in parsed:
        s = str(x).strip().lower()
        if s and s != "nan":
            cleaned.append(s)
    return sorted(set(cleaned))


def _coerce_numeric_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            non_null = df[col].dropna()
            if len(non_null) > 0:
                try:
                    if non_null.apply(lambda x: float(x).is_integer()).all():
                        df[col] = df[col].astype("Int64")
                except Exception:
                    pass
    return df


def _meaningful_columns(df):
    meaningful = set()
    for col in df.columns:
        series = df[col]
        if series.notna().sum() == 0:
            continue
        non_empty = series.dropna().astype(str).str.strip()
        if (non_empty != "").sum() == 0:
            continue
        meaningful.add(col)
    return meaningful


def _compact_metric_view(results):
    compact = {}
    for k, v in results.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            compact[k] = v
    return compact


# ================================
# 1. LOAD DATA
# ================================
fused = pd.read_csv(FUSED_PATH)
gold = load_csv(GOLD_PATH, name="restaurant_fusion_validation_set")

# Ensure required ID columns exist
if "eval_id" not in fused.columns:
    raise ValueError("Fused output must contain 'eval_id' column for validation evaluation.")
if "id" not in gold.columns:
    if "_id" in gold.columns:
        gold["id"] = gold["_id"].astype(str)
    else:
        raise ValueError("Gold evaluation set must contain 'id' or '_id'.")

fused["eval_id"] = fused["eval_id"].astype(str)
gold["id"] = gold["id"].astype(str)

# Prefer validation-set representation as target
# Normalize duplicate ID rows in fused by confidence if present
if "_fusion_confidence" in fused.columns:
    fused["_fusion_confidence"] = pd.to_numeric(fused["_fusion_confidence"], errors="coerce").fillna(0.0)
    fused = fused.sort_values("_fusion_confidence", ascending=False)
fused = fused.drop_duplicates(subset=["eval_id"], keep="first")

# ================================
# 2. ALIGN COLUMN SET
# ================================
shared_columns = set(fused.columns) & set(gold.columns)
shared_columns.discard("_fusion_metadata")
shared_columns.discard("_fusion_source_datasets")

gold_meaningful = _meaningful_columns(gold)
fused_meaningful = _meaningful_columns(fused)

candidate_columns = sorted(
    c for c in shared_columns
    if c in gold_meaningful and c in fused_meaningful
)

# Keep IDs for alignment but not for attribute evaluation
id_like_columns = {"id", "_id", "eval_id"}
eval_columns = [c for c in candidate_columns if c not in id_like_columns]

# ================================
# 3. NORMALIZE LIST-LIKE COLUMNS
# ================================
list_eval_columns = detect_list_like_columns(
    [fused[["eval_id"] + eval_columns].copy(), gold[["id"] + eval_columns].copy()],
    exclude_columns={"id", "_id", "eval_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if c in eval_columns and not c.startswith("_")]

if list_eval_columns:
    fused, gold = normalize_list_like_columns([fused, gold], list_eval_columns)
    for col in list_eval_columns:
        if col in fused.columns:
            fused[col] = fused[col].apply(_normalize_list_value)
        if col in gold.columns:
            gold[col] = gold[col].apply(_normalize_list_value)

# ================================
# 4. NORMALIZE TEXT + NUMERIC REPRESENTATIONS
#    IMPORTANT: do NOT normalize country with normalize_country.
# ================================
numeric_eval_columns = [
    c for c in eval_columns
    if c in {
        "latitude", "longitude", "rating", "rating_count",
        "phone_raw", "phone_e164", "house_number", "postal_code",
    }
]

fused = _coerce_numeric_columns(fused, numeric_eval_columns)
gold = _coerce_numeric_columns(gold, numeric_eval_columns)

text_like_columns = [
    c for c in eval_columns
    if c not in set(list_eval_columns) and c not in set(numeric_eval_columns)
]

for col in text_like_columns:
    fused[col] = fused[col].apply(_normalize_text_scalar)
    gold[col] = gold[col].apply(_normalize_text_scalar)

# Recompute eval columns after normalization, dropping all-null gold columns
eval_columns = [
    c for c in eval_columns
    if c in fused.columns and c in gold.columns
    and gold[c].notna().sum() > 0
]

# ================================
# 5. REGISTER EVALUATION FUNCTIONS
# ================================
strategy = DataFusionStrategy("restaurant_validation_evaluation")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

for col in eval_columns:
    if col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif col in {"latitude", "longitude"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.001)
    elif col in {"rating"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.05)
    elif col in {"rating_count", "phone_raw", "phone_e164", "house_number", "postal_code"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.0)
    elif col in {"source"}:
        register_eval(col, exact_match, "exact_match")
    elif col in {"country"}:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.8)
    elif col in {"name", "name_norm"}:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)
    elif col in {"address_line1", "address_line2", "street", "city", "state", "website", "map_url"}:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.75)
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.8)

# Compact one-line summary
print(
    "[EVAL FUNCTIONS] " +
    " | ".join(f"{col}:{eval_funcs_summary[col]}" for col in sorted(eval_funcs_summary))
)

# ================================
# 6. DIAGNOSTICS
# ================================
gold_ids = set(gold["id"].dropna().astype(str))
fused_ids = set(fused["eval_id"].dropna().astype(str))
mapped_cov = len(fused_ids & gold_ids)

print(json.dumps({
    "stage": "validation",
    "fused_rows": int(len(fused)),
    "gold_rows": int(len(gold)),
    "shared_eval_columns": eval_columns,
    "list_eval_columns": list_eval_columns,
    "numeric_eval_columns": numeric_eval_columns,
    "id_alignment": {
        "mapped_eval_id_coverage": mapped_cov,
        "gold_id_count": len(gold_ids),
    },
}, indent=2))

# ================================
# 7. RUN EVALUATION
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

fused = _strip_unevaluable_subcols(fused, gold)

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

_coerce_numeric_cols(fused, gold)

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

fused = _normalize_list_separators(fused, gold)
evaluation_results = evaluator.evaluate(
    fused_df=fused,
    fused_id_column="eval_id",
    gold_df=gold,
    gold_id_column="id",
)

# ================================
# 8. PRINT STRUCTURED METRICS
# ================================
print(json.dumps({
    "stage": "validation",
    "metrics": _compact_metric_view(evaluation_results),
}, indent=2, default=str))

# ================================
# 9. WRITE RESULTS
# ================================
with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4, default=str)