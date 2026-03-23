import pandas as pd
import json
import ast
import os
import sys
from pathlib import Path
from collections import Counter

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
# 0. OUTPUT DIRECTORY / PATHS
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
    if not text:
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
    """
    Map fused clusters back to gold IDs through _fusion_sources.
    If eval_id already exists, keep it and deduplicate on eval_id.
    """
    if "eval_id" in fused_df.columns:
        out = fused_df.copy()
        out["eval_id"] = out["eval_id"].astype(str)
        if "_fusion_confidence" in out.columns:
            out["_fusion_confidence"] = pd.to_numeric(out["_fusion_confidence"], errors="coerce").fillna(0.0)
            out = out.sort_values("_fusion_confidence", ascending=False)
        return out.drop_duplicates(subset=["eval_id"], keep="first")

    gold_ids = set(gold_df["id"].dropna().astype(str).tolist()) if "id" in gold_df.columns else set()
    gold_prefix = _detect_gold_prefix(gold_ids)
    expanded_rows = []

    for _, row in fused_df.iterrows():
        row_dict = row.to_dict()
        cluster_id = str(row_dict.get("_id", ""))
        source_ids = _parse_source_ids(row_dict.get("_fusion_sources"))

        candidates = []
        if gold_prefix:
            candidates = [sid for sid in source_ids if sid.startswith(gold_prefix)]
        if not candidates:
            candidates = [sid for sid in source_ids if sid in gold_ids]
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
        fallback["eval_id"] = fallback["_id"].astype(str) if "_id" in fallback.columns else fallback.index.astype(str)
        fallback["_eval_cluster_id"] = fallback["eval_id"]
        return fallback

    fused_eval = pd.DataFrame(expanded_rows)
    if "_fusion_confidence" in fused_eval.columns:
        fused_eval["_fusion_confidence"] = pd.to_numeric(
            fused_eval["_fusion_confidence"], errors="coerce"
        ).fillna(0.0)
        fused_eval = fused_eval.sort_values("_fusion_confidence", ascending=False)
    fused_eval = fused_eval.drop_duplicates(subset=["eval_id"], keep="first")
    return fused_eval


def _normalize_list_text_case(value):
    if _safe_is_missing(value):
        return value
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    text = str(value).strip()
    return text.lower() if text else text


def _canonicalize_list(value):
    if _safe_is_missing(value):
        return value
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    return value


def _maybe_numeric_series(series):
    converted = pd.to_numeric(series, errors="coerce")
    if converted.notna().sum() == 0:
        return series
    non_null = converted.dropna()
    if len(non_null) > 0 and non_null.apply(lambda x: float(x).is_integer()).all():
        return converted.astype("Int64")
    return converted


def _is_meaningful_column(df, col):
    return col in df.columns and df[col].notna().sum() > 0


def _looks_nested_track_column(col):
    return str(col).startswith("tracks_")


# ================================
# 2. LOAD FUSED OUTPUT AND GOLD SET
# ================================
fused = pd.read_csv(FUSION_PATH)
fusion_gold = load_xml(EVAL_SET_PATH, name="fusion_validation_set", nested_handling="aggregate")
fused_eval = build_eval_view(fused, fusion_gold)

print(f"[LOAD] fused rows={len(fused)}")
print(f"[LOAD] fused eval rows={len(fused_eval)}")
print(f"[LOAD] gold rows={len(fusion_gold)}")


# ================================
# 3. ALIGN REPRESENTATION
# Use validation-set representation as target.
# Do NOT apply country normalization here.
# ================================
list_eval_columns = detect_list_like_columns(
    [fused_eval, fusion_gold],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if not c.startswith("_")]

if list_eval_columns:
    fused_eval, fusion_gold = normalize_list_like_columns([fused_eval, fusion_gold], list_eval_columns)
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_canonicalize_list)
        if col in fusion_gold.columns:
            fusion_gold[col] = fusion_gold[col].apply(_canonicalize_list)
    print(f"[LIST NORMALIZATION] columns={list_eval_columns}")

# Numeric coercion to avoid float/int false mismatches
numeric_columns = []
for candidate in ["duration"]:
    if candidate in fused_eval.columns and candidate in fusion_gold.columns:
        numeric_columns.append(candidate)

for col in numeric_columns:
    fused_eval[col] = _maybe_numeric_series(fused_eval[col])
    fusion_gold[col] = _maybe_numeric_series(fusion_gold[col])

# Normalize scalar text lightly and consistently, excluding IDs and list columns
shared_columns = sorted(set(fused_eval.columns) & set(fusion_gold.columns))
protected_non_text = {"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"}
for col in shared_columns:
    if col in protected_non_text or col in list_eval_columns:
        continue
    if col in numeric_columns:
        continue
    if "date" in col.lower():
        continue
    if fused_eval[col].dtype == "object":
        fused_eval[col] = fused_eval[col].apply(
            lambda x: str(x).strip() if not _safe_is_missing(x) else x
        )
    if fusion_gold[col].dtype == "object":
        fusion_gold[col] = fusion_gold[col].apply(
            lambda x: str(x).strip() if not _safe_is_missing(x) else x
        )


# ================================
# 4. ID ALIGNMENT DIAGNOSTICS
# ================================
gold_ids = set(fusion_gold["id"].dropna().astype(str).tolist()) if "id" in fusion_gold.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0
print(f"[ID ALIGNMENT] direct _id coverage: {direct_cov}/{len(gold_ids)}")
print(f"[ID ALIGNMENT] mapped eval_id coverage: {mapped_cov}/{len(gold_ids)}")


# ================================
# 5. REGISTER EVALUATION FUNCTIONS
# Only evaluate columns meaningful in both fused and gold.
# Drop nested track subfields for robustness in this pass.
# ================================
strategy = DataFusionStrategy("evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name


candidate_columns = []
for col in shared_columns:
    if col in {"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"}:
        continue
    if not _is_meaningful_column(fused_eval, col):
        continue
    if not _is_meaningful_column(fusion_gold, col):
        continue
    if _looks_nested_track_column(col):
        continue
    candidate_columns.append(col)

for col in candidate_columns:
    col_lower = col.lower()

    if col in {"id", "eval_id"}:
        register_eval(col, exact_match, "exact_match")
    elif col_lower.endswith("date") or "date" in col_lower:
        register_eval(col, year_only_match, "year_only_match")
    elif col_lower in {"duration", "revenue", "price", "rating", "score", "latitude", "longitude"}:
        tolerance = 0.05 if col_lower == "duration" else 0.01
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=tolerance)
    elif col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif col_lower.startswith("is_") or col_lower.startswith("has_") or col_lower in {"active", "deleted", "enabled"}:
        register_eval(col, boolean_match, "boolean_match")
    else:
        threshold = 0.85
        if "country" in col_lower:
            threshold = 0.75
        elif col_lower in {"artist", "label"}:
            threshold = 0.85
        elif col_lower == "name":
            threshold = 0.85
        register_eval(col, tokenized_match, "tokenized_match", threshold=threshold)

summary_one_line = " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items()))
print(f"[EVAL FUNCS] {summary_one_line}")


# ================================
# 6. RUN EVALUATION
# IMPORTANT: use eval_id if present
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

fused_eval = _strip_unevaluable_subcols(fused_eval, fusion_gold)

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

_coerce_numeric_cols(fused_eval, fusion_gold)

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

fused_eval = _normalize_list_separators(fused_eval, fusion_gold)
evaluation_results = evaluator.evaluate(
    fused_df=fused_eval,
    fused_id_column="eval_id",
    gold_df=fusion_gold,
    gold_id_column="id",
)


# ================================
# 7. PRINT STRUCTURED METRICS
# ================================
print("[EVALUATION RESULTS JSON]")
print(json.dumps(evaluation_results, indent=2))


# ================================
# 8. WRITE RESULTS
# ================================
evaluation_output = "output/runs/20260323_005631_music/pipeline_evaluation/pipeline_evaluation.json"
with open(evaluation_output, "w") as f:
    json.dump(evaluation_results, f, indent=4)