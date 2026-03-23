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
OUTPUT_DIR = "output/runs/20260323_005631_music"
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

FUSED_PATH = "output/runs/20260323_005631_music/data_fusion/fusion_data.csv"
GOLD_PATH = "input/datasets/music/testsets/test_set.xml"
EVAL_OUTPUT_PATH = "output/runs/20260323_005631_music/pipeline_evaluation/pipeline_evaluation.json"


# ================================
# 1. HELPERS
# ================================
def _safe_is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
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
        explicit_eval_id = row_dict.get("eval_id", None)

        candidates = []
        if explicit_eval_id is not None and str(explicit_eval_id).strip():
            candidates = [str(explicit_eval_id).strip()]

        if not candidates:
            source_ids = _parse_source_ids(row_dict.get("_fusion_sources"))
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
            enriched["eval_id"] = eval_id
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


def _normalize_list_text_case(value):
    if _safe_is_missing(value):
        return value
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    text = str(value).strip()
    return text.lower() if text else text


def _normalize_scalar_text(value):
    if _safe_is_missing(value):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_normalize_scalar_text(v) for v in value]
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def _coerce_datetime_columns(df, columns):
    for col in columns:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            df[col] = parsed
    return df


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


def _is_meaningful_series(series):
    if series is None:
        return False
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    if non_null.astype(str).str.strip().eq("").all():
        return False
    return True


# ================================
# 2. LOAD DATA
# ================================
fused = pd.read_csv(FUSED_PATH)
fusion_test_set = load_xml(GOLD_PATH, name="fusion_test_set", nested_handling="aggregate")
fused_eval = build_eval_view(fused, fusion_test_set)

print(f"[LOAD] fused rows={len(fused)}")
print(f"[LOAD] gold rows={len(fusion_test_set)}")
print(f"[LOAD] eval-view rows={len(fused_eval)}")


# ================================
# 3. ALIGN REPRESENTATIONS
# Use validation/test representation as target.
# Do NOT run country normalization here.
# ================================
shared_columns = sorted((set(fused_eval.columns) & set(fusion_test_set.columns)) - {"id", "_id", "eval_id", "_eval_cluster_id"})
list_eval_columns = detect_list_like_columns(
    [fused_eval, fusion_test_set],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if c in shared_columns and not c.startswith("_")]

if list_eval_columns:
    fused_eval, fusion_test_set = normalize_list_like_columns([fused_eval, fusion_test_set], list_eval_columns)
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
        if col in fusion_test_set.columns:
            fusion_test_set[col] = fusion_test_set[col].apply(_normalize_list_text_case)
    print(f"[LIST NORMALIZATION] columns={list_eval_columns}")

date_columns = [c for c in shared_columns if "date" in c.lower() or c.lower() in {"founded", "created", "published"}]
numeric_columns = [
    c for c in shared_columns
    if c.lower() in {"duration", "revenue", "latitude", "longitude", "price", "rating", "score", "count", "year"}
    or any(tok in c.lower() for tok in ["duration", "revenue", "price", "rating", "score", "count"])
]

fused_eval = _coerce_datetime_columns(fused_eval, date_columns)
fusion_test_set = _coerce_datetime_columns(fusion_test_set, date_columns)

fused_eval = _coerce_numeric_columns(fused_eval, numeric_columns)
fusion_test_set = _coerce_numeric_columns(fusion_test_set, numeric_columns)

for col in shared_columns:
    if col in list_eval_columns or col in date_columns or col in numeric_columns:
        continue
    fused_eval[col] = fused_eval[col].apply(_normalize_scalar_text)
    fusion_test_set[col] = fusion_test_set[col].apply(_normalize_scalar_text)


# ================================
# 4. ID ALIGNMENT DIAGNOSTICS
# ================================
gold_ids = set(fusion_test_set["id"].dropna().astype(str).tolist()) if "id" in fusion_test_set.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0
print(f"[ID ALIGNMENT] direct _id coverage: {direct_cov}/{len(gold_ids)}")
print(f"[ID ALIGNMENT] mapped eval_id coverage: {mapped_cov}/{len(gold_ids)}")


# ================================
# 5. SELECT EVALUATION COLUMNS
# Drop absent/all-null columns.
# Drop structurally difficult nested track fields for this run.
# ================================
nested_track_prefix = "tracks_"
candidate_columns = []
for col in shared_columns:
    if col.startswith("_"):
        continue
    if col == "id":
        continue
    if col.startswith(nested_track_prefix):
        continue
    if not _is_meaningful_series(fused_eval[col]) or not _is_meaningful_series(fusion_test_set[col]):
        continue
    candidate_columns.append(col)

print(f"[EVALUATION COLUMNS] {candidate_columns}")


# ================================
# 6. REGISTER EVALUATION FUNCTIONS
# Use non-exact defaults for textual fields.
# ================================
strategy = DataFusionStrategy("evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items())
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

for col in candidate_columns:
    col_lower = col.lower()

    if col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif "date" in col_lower or col_lower in {"founded", "created", "published"}:
        register_eval(col, year_only_match, "year_only_match")
    elif col_lower in {"duration", "revenue", "latitude", "longitude", "price", "rating", "score", "count", "year"} or any(
        tok in col_lower for tok in ["duration", "revenue", "price", "rating", "score", "count"]
    ):
        tolerance = 0.01 if col_lower != "duration" else 0.05
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=tolerance)
    elif "country" in col_lower:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.80)
    elif col_lower in {"id", "isbn", "issn", "doi", "code"}:
        register_eval(col, exact_match, "exact_match")
    elif col_lower.startswith("is_") or col_lower.startswith("has_") or col_lower in {"active", "deleted", "explicit"}:
        register_eval(col, boolean_match, "boolean_match")
    else:
        threshold = 0.85
        if col_lower in {"name", "artist", "label"}:
            threshold = 0.85
        register_eval(col, tokenized_match, "tokenized_match", threshold=threshold)

compact_summary = " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items()))
print(f"[EVAL FUNCS] {compact_summary}")


# ================================
# 7. RUN EVALUATION
# IMPORTANT: use eval_id when present
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
print("[EVALUATION RESULTS]")
print(json.dumps(evaluation_results, indent=2, default=str))


# ================================
# 9. WRITE RESULTS
# ================================
with open(EVAL_OUTPUT_PATH, "w") as f:
    json.dump(evaluation_results, f, indent=4, default=str)

print(f"[SAVED] {EVAL_OUTPUT_PATH}")