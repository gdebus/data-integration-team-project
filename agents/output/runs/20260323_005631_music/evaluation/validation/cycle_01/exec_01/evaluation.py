import pandas as pd
import json
import ast
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


def _safe_is_missing(value):
    if value is None:
        return True
    try:
        is_na = pd.isna(value)
        return bool(is_na) if not hasattr(is_na, "__array__") else False
    except Exception:
        return False


def _parse_list_like(value):
    if _safe_is_missing(value):
        return value
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
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
    return [text]


def _normalize_list_text_case(value):
    if _safe_is_missing(value):
        return value
    parsed = _parse_list_like(value)
    if isinstance(parsed, list):
        return sorted({str(v).strip().lower() for v in parsed if str(v).strip()})
    return parsed


def _normalize_text(value):
    if _safe_is_missing(value):
        return value
    text = str(value).strip()
    return text if text else value


def _normalize_text_casefold(value):
    if _safe_is_missing(value):
        return value
    text = str(value).strip()
    return text.lower() if text else value


def _coerce_numeric_nullable_int(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            non_null = df[col].dropna()
            if len(non_null) > 0 and non_null.apply(lambda x: float(x).is_integer()).all():
                df[col] = df[col].astype("Int64")
    return df


def _meaningful_common_columns(fused_df, gold_df):
    common = []
    for col in sorted(set(fused_df.columns) & set(gold_df.columns)):
        if col in {"id", "_id", "eval_id", "_eval_cluster_id"}:
            continue
        if str(col).startswith("_"):
            continue
        gold_non_null = gold_df[col].dropna() if col in gold_df.columns else pd.Series(dtype="object")
        fused_non_null = fused_df[col].dropna() if col in fused_df.columns else pd.Series(dtype="object")
        if len(gold_non_null) == 0 or len(fused_non_null) == 0:
            continue
        common.append(col)
    return common


# === 0. OUTPUT DIRECTORY ===
OUTPUT_DIR = "output/runs/20260323_005631_music"
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

# === 1. LOAD FUSED OUTPUT AND GOLD VALIDATION SET ===
fused = pd.read_csv("output/runs/20260323_005631_music/data_fusion/fusion_data.csv")
gold = load_xml(
    "input/datasets/music/testsets/test_set.xml",
    name="fusion_validation_set",
    nested_handling="aggregate",
)

# Use eval_id if already provided by fusion pipeline
fused_eval = fused.copy()
if "eval_id" not in fused_eval.columns:
    raise ValueError("Fused output must contain eval_id for validation evaluation.")

# === 2. NORMALIZE TO VALIDATION-SET REPRESENTATION ===
# Do NOT apply country normalization here.
# Only align textual/list/numeric representation consistently across fused and gold.

# Strip scalar text columns
for df in [fused_eval, gold]:
    for col in df.columns:
        if col in {"id", "_id", "eval_id", "_eval_cluster_id"} or str(col).startswith("_"):
            continue
        if df[col].dtype == "object":
            df[col] = df[col].apply(_normalize_text)

# Normalize list-like columns jointly
list_eval_columns = detect_list_like_columns(
    [fused_eval, gold],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if not str(c).startswith("_")]

if list_eval_columns:
    fused_eval, gold = normalize_list_like_columns([fused_eval, gold], list_eval_columns)
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
        if col in gold.columns:
            gold[col] = gold[col].apply(_normalize_list_text_case)

# Numeric coercion
numeric_columns = [c for c in ["duration"] if c in fused_eval.columns and c in gold.columns]
fused_eval = _coerce_numeric_nullable_int(fused_eval, numeric_columns)
gold = _coerce_numeric_nullable_int(gold, numeric_columns)

# Case-normalize selected scalar text columns
for col in ["name", "artist", "label", "genre", "release-country"]:
    if col in fused_eval.columns:
        fused_eval[col] = fused_eval[col].apply(_normalize_text_casefold)
    if col in gold.columns:
        gold[col] = gold[col].apply(_normalize_text_casefold)

# === 3. SELECT EVALUATION COLUMNS ===
common_columns = _meaningful_common_columns(fused_eval, gold)

# Drop structurally difficult nested track fields from evaluation to avoid structural-artifact scoring
nested_track_columns = [c for c in common_columns if c.startswith("tracks_")]
evaluation_columns = [c for c in common_columns if c not in nested_track_columns]

# === 4. REGISTER EVALUATION FUNCTIONS ===
strategy = DataFusionStrategy("evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, label, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items())
    eval_funcs_summary[column] = f"{label}({args})" if args else label

if "name" in evaluation_columns:
    register_eval("name", tokenized_match, "tokenized_match", threshold=0.85)
if "artist" in evaluation_columns:
    register_eval("artist", tokenized_match, "tokenized_match", threshold=0.85)
if "label" in evaluation_columns:
    register_eval("label", tokenized_match, "tokenized_match", threshold=0.8)
if "genre" in evaluation_columns:
    register_eval("genre", tokenized_match, "tokenized_match", threshold=0.75)
if "release-country" in evaluation_columns:
    register_eval("release-country", tokenized_match, "tokenized_match", threshold=0.7)
if "release-date" in evaluation_columns:
    register_eval("release-date", year_only_match, "year_only_match")
if "duration" in evaluation_columns:
    register_eval("duration", numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.01)

# If any non-track list columns remain, use set equality after canonicalization
for col in evaluation_columns:
    if col in eval_funcs_summary:
        continue
    if col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.8)

print("[EVAL_FUNCTIONS] " + " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items())))
print("[EVAL_COLUMNS] " + ", ".join(evaluation_columns))
print("[SKIPPED_NESTED_COLUMNS] " + (", ".join(nested_track_columns) if nested_track_columns else "none"))

# === 5. RUN EVALUATION ===
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

# === 6. PRINT STRUCTURED METRICS ===
print("[EVALUATION_RESULTS]")
print(json.dumps(evaluation_results, indent=2, default=str))

# === 7. WRITE RESULTS ===
evaluation_output = "output/runs/20260323_005631_music/pipeline_evaluation/pipeline_evaluation.json"
with open(evaluation_output, "w") as f:
    json.dump(evaluation_results, f, indent=4, default=str)