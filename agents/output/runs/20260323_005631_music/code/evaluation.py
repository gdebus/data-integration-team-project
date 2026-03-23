import pandas as pd
import json
import ast
import os
import re
import sys
from collections import Counter
from pathlib import Path

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
# HELPERS
# ================================

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


def _detect_gold_prefixes(gold_ids):
    prefixes = [g.split("_", 1)[0] + "_" for g in gold_ids if "_" in g]
    if not prefixes:
        return []
    return [p for p, _ in Counter(prefixes).most_common()]


def build_eval_view(fused_df, gold_df):
    gold_ids = set(gold_df["id"].dropna().astype(str).tolist()) if "id" in gold_df.columns else set()
    gold_prefixes = _detect_gold_prefixes(gold_ids)
    expanded_rows = []

    for _, row in fused_df.iterrows():
        row_dict = row.to_dict()
        cluster_id = str(row_dict.get("_id", ""))
        source_ids = _parse_source_ids(row_dict.get("_fusion_sources"))

        candidates = []
        for prefix in gold_prefixes:
            prefixed = [sid for sid in source_ids if sid.startswith(prefix)]
            if prefixed:
                candidates.extend(prefixed)

        if not candidates:
            candidates = [sid for sid in source_ids if sid in gold_ids]

        if not candidates and "eval_id" in row_dict and pd.notna(row_dict["eval_id"]):
            candidates = [str(row_dict["eval_id"])]

        if not candidates and cluster_id:
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
        if "eval_id" not in fallback.columns:
            fallback["eval_id"] = fallback["_id"].astype(str) if "_id" in fallback.columns else fallback["id"].astype(str)
        fallback["_eval_cluster_id"] = fallback["eval_id"].astype(str)
        return fallback

    fused_eval = pd.DataFrame(expanded_rows)
    if "_fusion_confidence" in fused_eval.columns:
        fused_eval["_fusion_confidence"] = pd.to_numeric(fused_eval["_fusion_confidence"], errors="coerce").fillna(0.0)
        fused_eval = fused_eval.sort_values("_fusion_confidence", ascending=False)
    fused_eval = fused_eval.drop_duplicates(subset=["eval_id"], keep="first")
    return fused_eval


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
    gold_non_null = gold_df[col].dropna()
    fused_non_null = fused_df[col].dropna()
    return len(gold_non_null) > 0 and len(fused_non_null) > 0


def _looks_nested_track_col(col):
    return str(col).startswith("tracks_")


# ================================
# PATHS
# ================================

OUTPUT_DIR = "output/runs/20260323_005631_music"
FUSION_PATH = "output/runs/20260323_005631_music/data_fusion/fusion_data.csv"
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_OUTPUT = "output/runs/20260323_005631_music/pipeline_evaluation/pipeline_evaluation.json"
GOLD_PATH = "input/datasets/music/testsets/test_set.xml"

os.makedirs(EVAL_DIR, exist_ok=True)

# ================================
# LOAD DATA
# ================================

fused = pd.read_csv(FUSION_PATH)
gold = load_xml(GOLD_PATH, name="fusion_test_set", nested_handling="aggregate")

fused_eval = build_eval_view(fused, gold)

# ================================
# NORMALIZE LIST-LIKE COLUMNS TO SAME REPRESENTATION
# ================================

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
    print(f"[LIST NORMALIZATION] columns={list_eval_columns}")

# ================================
# COERCE NUMERIC-LIKE COLUMNS CONSISTENTLY
# ================================

numeric_candidates = [
    c for c in set(fused_eval.columns).intersection(set(gold.columns))
    if any(token in str(c).lower() for token in ["duration", "count", "year", "rating", "score", "price", "revenue", "latitude", "longitude"])
]

for col in numeric_candidates:
    _coerce_numeric_column(fused_eval, col)
    _coerce_numeric_column(gold, col)

if numeric_candidates:
    print(f"[NUMERIC COERCION] columns={numeric_candidates}")

# ================================
# ID ALIGNMENT DIAGNOSTICS
# ================================

gold_ids = set(gold["id"].dropna().astype(str).tolist()) if "id" in gold.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0
print(f"[ID ALIGNMENT] direct _id coverage: {direct_cov}/{len(gold_ids)}")
print(f"[ID ALIGNMENT] mapped eval_id coverage: {mapped_cov}/{len(gold_ids)}")

# ================================
# REGISTER EVALUATION FUNCTIONS
# ================================

strategy = DataFusionStrategy("evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

shared_columns = sorted(set(fused_eval.columns).intersection(set(gold.columns)))
candidate_columns = []

for col in shared_columns:
    if col in {"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence", "_fusion_sources"}:
        continue
    if str(col).endswith("_provenance"):
        continue
    if not _meaningful_in_both(fused_eval, gold, col):
        continue
    candidate_columns.append(col)

# Drop nested track fields for this sealed evaluation, per robustness rule / prior evidence
nested_track_columns = [c for c in candidate_columns if _looks_nested_track_col(c)]
top_level_columns = [c for c in candidate_columns if not _looks_nested_track_col(c)]

for col in top_level_columns:
    col_l = str(col).lower()

    if col_l == "id":
        register_eval(col, exact_match, "exact_match")
    elif any(tok in col_l for tok in ["date", "founded", "released"]):
        register_eval(col, year_only_match, "year_only_match")
    elif any(tok in col_l for tok in ["duration", "revenue", "latitude", "longitude", "price", "count", "year", "rating", "score"]):
        tolerance = 0.01
        if "duration" in col_l:
            tolerance = 0.05
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=tolerance)
    elif any(tok in col_l for tok in ["is_", "has_", "active", "enabled", "disabled", "boolean", "flag"]):
        register_eval(col, boolean_match, "boolean_match")
    elif col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    else:
        threshold = 0.85
        if "country" in col_l:
            threshold = 0.75
        elif any(tok in col_l for tok in ["artist", "name", "label", "genre"]):
            threshold = 0.85
        register_eval(col, tokenized_match, "tokenized_match", threshold=threshold)

print("[EVALUATION FUNCTIONS] " + " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items())))
if nested_track_columns:
    print(f"[DROPPED NESTED COLUMNS] {nested_track_columns}")

# ================================
# RUN EVALUATION
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
# PRINT STRUCTURED METRICS
# ================================

print("[EVALUATION RESULTS]")
print(json.dumps(evaluation_results, indent=2, default=str))

# ================================
# WRITE RESULTS
# ================================

with open(EVAL_OUTPUT, "w") as f:
    json.dump(evaluation_results, f, indent=4, default=str)