import pandas as pd
import json
import ast
import re
import os
from pathlib import Path
import sys

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
# 0. PATHS
# ================================
OUTPUT_DIR = "output/runs/20260322_175903_games"
FUSION_PATH = "output/runs/20260322_175903_games/data_fusion/fusion_data.csv"
EVAL_SET_PATH = "input/datasets/games/testsets/test_set_fusion.xml"
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_OUTPUT_PATH = "output/runs/20260322_175903_games/pipeline_evaluation/pipeline_evaluation.json"
DEBUG_OUTPUT_PATH = os.path.join(EVAL_DIR, "debug_fusion_eval.jsonl")
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
        is_na = pd.isna(value)
        return bool(is_na) if not hasattr(is_na, "__array__") else False
    except Exception:
        return False


def _normalize_list_text_case(value):
    if _safe_is_missing(value):
        return value
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    text = str(value).strip()
    return text.lower() if text else text


def _is_meaningful_column(df, col):
    return col in df.columns and df[col].notna().any()


def _parse_possible_list(value):
    if value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return list(value)
    text = str(value).strip()
    if not text:
        return value
    if text[0] in "[({":
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, (list, tuple, set)):
                    return list(parsed)
            except Exception:
                pass
    return value


def _coerce_numeric_columns(df, cols):
    for col in cols:
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


def _drop_derived_subcolumns_without_gold_match(fused_df, gold_df):
    gold_cols = set(gold_df.columns)
    keep_cols = []
    for col in fused_df.columns:
        if col in gold_cols:
            keep_cols.append(col)
            continue
        if col.startswith("_") or col in {"eval_id", "id", "_id"}:
            keep_cols.append(col)
            continue
        parent = col.split("_", 1)[0]
        if parent in gold_cols and col not in gold_cols:
            continue
        keep_cols.append(col)
    return fused_df[keep_cols].copy()


# ================================
# 2. LOAD DATA
# ================================
fused = pd.read_csv(FUSION_PATH)

fusion_test_set = load_xml(
    EVAL_SET_PATH,
    name="fusion_test_set",
    nested_handling="aggregate",
)

# Respect precomputed eval_id if present
if "eval_id" not in fused.columns:
    raise ValueError("Expected fused output to contain 'eval_id' column for evaluation alignment.")

fused_eval = fused.copy()
fused_eval = _drop_derived_subcolumns_without_gold_match(fused_eval, fusion_test_set)

# Prefer highest-confidence row if duplicate eval_id exists
if "_fusion_confidence" in fused_eval.columns:
    fused_eval["_fusion_confidence"] = pd.to_numeric(fused_eval["_fusion_confidence"], errors="coerce").fillna(0.0)
    fused_eval = fused_eval.sort_values("_fusion_confidence", ascending=False)
fused_eval = fused_eval.drop_duplicates(subset=["eval_id"], keep="first")


# ================================
# 3. SHARED NORMALIZATION TO VALIDATION REPRESENTATION
# ================================
# Parse potential serialized lists in both sides first
for df in (fused_eval, fusion_test_set):
    for col in df.columns:
        if col.startswith("_"):
            continue
        df[col] = df[col].apply(_parse_possible_list)

# List-like normalization based on both dataframes, targeting validation representation
list_eval_columns = detect_list_like_columns(
    [fused_eval, fusion_test_set],
    exclude_columns={"id", "_id", "eval_id", "_fusion_confidence", "_fusion_sources", "_fusion_source_datasets"},
)
list_eval_columns = [c for c in list_eval_columns if not c.startswith("_")]

if list_eval_columns:
    fused_eval, fusion_test_set = normalize_list_like_columns(
        [fused_eval, fusion_test_set],
        list_eval_columns,
    )
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
        if col in fusion_test_set.columns:
            fusion_test_set[col] = fusion_test_set[col].apply(_normalize_list_text_case)

# Explicit numeric coercion on shared numeric-like columns
numeric_candidates = [
    "criticScore",
    "userScore",
    "globalSales",
    "releaseYear",
]
shared_numeric_cols = [c for c in numeric_candidates if c in fused_eval.columns and c in fusion_test_set.columns]
fused_eval = _coerce_numeric_columns(fused_eval, shared_numeric_cols)
fusion_test_set = _coerce_numeric_columns(fusion_test_set, shared_numeric_cols)


# ================================
# 4. COLUMN SELECTION
# Only evaluate direct 1:1 columns present and meaningful in both.
# ================================
shared_cols = set(fused_eval.columns) & set(fusion_test_set.columns)
shared_cols -= {"id", "_id", "eval_id", "_fusion_confidence", "_fusion_sources", "_fusion_source_datasets", "_fusion_metadata"}

eval_columns = []
for col in sorted(shared_cols):
    if not _is_meaningful_column(fused_eval, col):
        continue
    if not _is_meaningful_column(fusion_test_set, col):
        continue
    eval_columns.append(col)


# ================================
# 5. REGISTER EVALUATION FUNCTIONS
# ================================
strategy = DataFusionStrategy("evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

# Games-specific direct-match columns from validation sample and fused schema
for col in eval_columns:
    if col == "name":
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)
    elif col == "developer":
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.80)
    elif col == "publisher":
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.80)
    elif col == "platform":
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.75)
    elif col == "series":
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.80)
    elif col == "releaseYear":
        register_eval(col, year_only_match, "year_only_match")
    elif col in {"criticScore"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=2.0)
    elif col in {"userScore"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.25)
    elif col in {"globalSales"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=1.0)
    elif col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif re.search(r"(flag|is_|has_|boolean|active|enabled)$", col, flags=re.IGNORECASE):
        register_eval(col, boolean_match, "boolean_match")
    elif re.search(r"(code|isbn|issn|ean|upc|gtin)$", col, flags=re.IGNORECASE):
        register_eval(col, exact_match, "exact_match")
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.80)

print("[EVALUATION_FUNCTIONS] " + " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items())))


# ================================
# 6. ID ALIGNMENT DIAGNOSTICS
# ================================
gold_ids = set(fusion_test_set["id"].dropna().astype(str).tolist()) if "id" in fusion_test_set.columns else set()
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0
print(json.dumps({
    "stage": "validation",
    "fused_rows": int(len(fused)),
    "fused_eval_rows": int(len(fused_eval)),
    "gold_rows": int(len(fusion_test_set)),
    "gold_id_count": int(len(gold_ids)),
    "mapped_eval_id_coverage": int(mapped_cov),
    "mapped_eval_id_coverage_rate": round(mapped_cov / len(gold_ids), 6) if gold_ids else None,
    "evaluated_columns": eval_columns,
    "list_eval_columns": list_eval_columns,
}, indent=2))


# ================================
# 7. RUN EVALUATION
# ================================
evaluator = DataFusionEvaluator(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=DEBUG_OUTPUT_PATH,
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

fused_eval = _strip_unevaluable_subcols(fused_eval, fusion_test_set)

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

_coerce_numeric_cols(fused_eval, fusion_test_set)

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

fused_eval = _normalize_list_separators(fused_eval, fusion_test_set)
evaluation_results = evaluator.evaluate(
    fused_df=fused_eval,
    fused_id_column="eval_id",
    gold_df=fusion_test_set,
    gold_id_column="id",
)


# ================================
# 8. PRINT STRUCTURED METRICS
# ================================
print(json.dumps(evaluation_results, indent=2, default=str))


# ================================
# 9. WRITE RESULTS
# ================================
with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=2, default=str)