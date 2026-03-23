import pandas as pd
import json
import ast
import os
import sys
from pathlib import Path
from collections import OrderedDict

from PyDI.io import load_xml
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


# ================================
# 0. PATHS
# ================================
OUTPUT_DIR = "output/runs/20260323_005631_music"
FUSION_PATH = "output/runs/20260323_005631_music/data_fusion/fusion_data.csv"
EVAL_SET_PATH = "input/datasets/music/testsets/test_set.xml"
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)


# ================================
# 1. HELPERS
# ================================
def _safe_is_missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _parse_list_like(value):
    if _safe_is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value]
    text = str(value).strip()
    if not text:
        return []
    if text[0] in "[({":
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [v for v in parsed]
            except Exception:
                pass
    return [value]


def _canon_text(value):
    if _safe_is_missing(value):
        return None
    text = str(value).strip()
    return text.lower() if text else None


def _canon_list(value):
    items = _parse_list_like(value)
    cleaned = []
    for x in items:
        cx = _canon_text(x)
        if cx:
            cleaned.append(cx)
    return sorted(set(cleaned))


def _coerce_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            non_null = df[col].dropna()
            if len(non_null) and non_null.apply(lambda x: float(x).is_integer()).all():
                df[col] = df[col].astype("Int64")
    return df


def _meaningful_columns(df):
    cols = []
    for c in df.columns:
        series = df[c]
        non_null = series.dropna()
        if len(non_null) == 0:
            continue
        if non_null.astype(str).str.strip().eq("").all():
            continue
        cols.append(c)
    return set(cols)


def _compact_metrics(obj):
    if isinstance(obj, dict):
        wanted = OrderedDict()
        for k in ["overall_accuracy", "accuracy", "micro_accuracy", "macro_accuracy", "coverage", "matched_records"]:
            if k in obj:
                wanted[k] = obj[k]
        if wanted:
            return wanted
    return obj


# ================================
# 2. LOAD DATA
# ================================
fused = pd.read_csv(FUSION_PATH)
gold = load_xml(EVAL_SET_PATH, name="fusion_validation_set", nested_handling="aggregate")

# Use eval_id if present, per requirement
fused_id_column = "eval_id" if "eval_id" in fused.columns else "_id"

# ================================
# 3. NORMALIZE REPRESENTATION FOR EVALUATION
#    Important: do NOT apply country normalization here.
# ================================
list_eval_columns = detect_list_like_columns(
    [fused, gold],
    exclude_columns={"id", "_id", "eval_id", "_fusion_confidence"},
)

# Drop structurally difficult nested track columns from evaluation
nested_track_cols = {
    c for c in set(fused.columns).intersection(set(gold.columns))
    if c.startswith("tracks_")
}
drop_from_eval = set(nested_track_cols)

if list_eval_columns:
    fused, gold = normalize_list_like_columns([fused, gold], list_eval_columns)

# Canonicalize list/text consistently for list comparisons
for col in list_eval_columns:
    if col in fused.columns:
        fused[col] = fused[col].apply(_canon_list)
    if col in gold.columns:
        gold[col] = gold[col].apply(_canon_list)

# Numeric coercion
numeric_cols = [c for c in ["duration"] if c in fused.columns and c in gold.columns]
fused = _coerce_numeric(fused, numeric_cols)
gold = _coerce_numeric(gold, numeric_cols)

# Boolean-like coercion if ever present
for col in set(fused.columns).intersection(set(gold.columns)):
    if col.lower().startswith("is_") or col.lower().startswith("has_"):
        def _to_bool(x):
            if _safe_is_missing(x):
                return pd.NA
            s = str(x).strip().lower()
            if s in {"true", "1", "yes"}:
                return True
            if s in {"false", "0", "no"}:
                return False
            return pd.NA
        fused[col] = fused[col].apply(_to_bool)
        gold[col] = gold[col].apply(_to_bool)


# ================================
# 4. COLUMN SELECTION
# ================================
shared = set(fused.columns).intersection(set(gold.columns))
meaningful_shared = (
    shared
    & _meaningful_columns(fused)
    & _meaningful_columns(gold)
)

excluded_system_cols = {
    "id", "_id", "eval_id", "_fusion_sources", "_fusion_source_datasets",
    "_fusion_confidence", "_fusion_metadata"
}
candidate_eval_columns = sorted(
    c for c in meaningful_shared
    if c not in excluded_system_cols and c not in drop_from_eval
)

# ================================
# 5. REGISTER EVALUATION FUNCTIONS
# ================================
strategy = DataFusionStrategy("music_evaluation_strategy")
eval_funcs_summary = OrderedDict()

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items())
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

for col in candidate_eval_columns:
    col_l = col.lower()

    if col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif col in {"duration"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.01)
    elif "date" in col_l or col_l in {"founded", "birth_year", "year"}:
        register_eval(col, year_only_match, "year_only_match")
    elif col_l.endswith("_id") or col_l in {"code"}:
        register_eval(col, exact_match, "exact_match")
    elif col_l.startswith("is_") or col_l.startswith("has_") or col_l in {"active", "explicit"}:
        register_eval(col, boolean_match, "boolean_match")
    else:
        threshold = 0.85
        if col_l in {"artist", "name", "label"}:
            threshold = 0.85
        elif "country" in col_l:
            threshold = 0.70
        elif "genre" in col_l:
            threshold = 0.75
        register_eval(col, tokenized_match, "tokenized_match", threshold=threshold)

print("[EVAL_FUNCTIONS] " + " | ".join(f"{k}:{v}" for k, v in eval_funcs_summary.items()))

# ================================
# 6. RUN EVALUATION
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
    fused_id_column=fused_id_column,
    gold_df=gold,
    gold_id_column="id",
)

# ================================
# 7. PRINT STRUCTURED METRICS
# ================================
structured_print = {
    "evaluation_stage": "validation",
    "fusion_path": FUSION_PATH,
    "gold_path": EVAL_SET_PATH,
    "fused_id_column": fused_id_column,
    "evaluated_columns": candidate_eval_columns,
    "dropped_nested_columns": sorted(drop_from_eval),
    "metrics": _compact_metrics(evaluation_results),
}

print(json.dumps(structured_print, indent=2, default=str))
print(json.dumps(evaluation_results, indent=2, default=str))

# ================================
# 8. WRITE OUTPUT
# ================================
evaluation_output = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
with open(evaluation_output, "w") as f:
    json.dump(evaluation_results, f, indent=4, default=str)