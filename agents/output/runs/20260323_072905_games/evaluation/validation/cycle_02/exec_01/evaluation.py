import pandas as pd
import json
import ast
import os
import re
import sys
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
            if str(_path.resolve()) not in sys.path:
                sys.path.append(str(_path.resolve()))
    from list_normalization import detect_list_like_columns, normalize_list_like_columns


# === 0. OUTPUT DIRECTORY ===
OUTPUT_DIR = "output/runs/20260323_072905_games"
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)


# ================================
# HELPERS
# ================================

def _is_missing(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, tuple, set)) and len(value) == 0:
        return True
    return False


def _parse_source_ids(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(v) for v in parsed if str(v).strip()]
        except Exception:
            pass
    return []


def _meaningful_series(series):
    if series is None:
        return pd.Series(dtype="object")
    s = series.dropna()
    if s.empty:
        return s
    s = s.astype(str).str.strip()
    s = s.replace("", pd.NA).dropna()
    return s


def _has_meaningful_values(df, col):
    return col in df.columns and not _meaningful_series(df[col]).empty


def _canonicalize_list_text(value):
    if _is_missing(value):
        return value
    if isinstance(value, (list, tuple, set)):
        cleaned = []
        for v in value:
            if _is_missing(v):
                continue
            t = str(v).strip().lower()
            if t:
                cleaned.append(t)
        return sorted(set(cleaned))
    text = str(value).strip()
    return text.lower() if text else text


def _coerce_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            non_null = df[col].dropna()
            if not non_null.empty:
                try:
                    if non_null.apply(lambda x: float(x).is_integer()).all():
                        df[col] = df[col].astype("Int64")
                except Exception:
                    pass


def _coerce_boolean_text(df, cols):
    bool_map = {
        "true": True, "false": False,
        "yes": True, "no": False,
        "1": True, "0": False,
        "y": True, "n": False,
        "t": True, "f": False,
    }
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: bool_map.get(str(x).strip().lower(), x) if not _is_missing(x) else x
            )


def build_eval_view(fused_df, gold_df):
    """
    Build an evaluation-aligned fused view.
    Priority:
    1) If eval_id exists and covers many gold IDs, use it.
    2) Otherwise expand rows by _fusion_sources so every gold ID found in a cluster gets a row.
    3) Fallback to _id.
    """
    gold_ids = set(gold_df["id"].dropna().astype(str).tolist()) if "id" in gold_df.columns else set()

    if "eval_id" in fused_df.columns:
        eval_cov = len(set(fused_df["eval_id"].dropna().astype(str)) & gold_ids)
        if eval_cov >= max(1, int(0.5 * len(gold_ids))) or eval_cov >= len(gold_ids) * 0.3:
            view = fused_df.copy()
            view["eval_id"] = view["eval_id"].astype(str)
            if "_id" in view.columns and "_eval_cluster_id" not in view.columns:
                view["_eval_cluster_id"] = view["_id"].astype(str)
            return view.drop_duplicates(subset=["eval_id"], keep="first")

    expanded_rows = []
    for _, row in fused_df.iterrows():
        row_dict = row.to_dict()
        cluster_id = str(row_dict.get("_id", ""))
        source_ids = _parse_source_ids(row_dict.get("_fusion_sources"))
        matched_gold_ids = [sid for sid in source_ids if sid in gold_ids]

        if matched_gold_ids:
            for gid in matched_gold_ids:
                enriched = dict(row_dict)
                enriched["eval_id"] = gid
                enriched["_eval_cluster_id"] = cluster_id
                expanded_rows.append(enriched)

    if expanded_rows:
        fused_eval = pd.DataFrame(expanded_rows)
        if "_fusion_confidence" in fused_eval.columns:
            fused_eval["_fusion_confidence"] = pd.to_numeric(
                fused_eval["_fusion_confidence"], errors="coerce"
            ).fillna(0.0)
            fused_eval = fused_eval.sort_values("_fusion_confidence", ascending=False)
        fused_eval = fused_eval.drop_duplicates(subset=["eval_id"], keep="first")
        return fused_eval

    fallback = fused_df.copy()
    fallback["eval_id"] = fallback["_id"].astype(str) if "_id" in fallback.columns else fallback.index.astype(str)
    fallback["_eval_cluster_id"] = fallback["eval_id"]
    return fallback


def _choose_eval_function(col, fused_df, gold_df, list_cols):
    col_l = col.lower()

    if col in list_cols:
        return (set_equality_match, "set_equality_match", {})

    if any(k in col_l for k in ["releaseyear", "date", "founded", "birth", "death"]):
        return (year_only_match, "year_only_match", {})

    if any(k in col_l for k in ["criticscore", "critic_score"]):
        return (numeric_tolerance_match, "numeric_tolerance_match", {"tolerance": 5.0})

    if any(k in col_l for k in ["userscore", "user_score", "rating", "score"]):
        return (numeric_tolerance_match, "numeric_tolerance_match", {"tolerance": 1.0})

    if any(k in col_l for k in ["sales", "revenue", "price", "duration", "latitude", "longitude", "count", "year"]):
        return (numeric_tolerance_match, "numeric_tolerance_match", {"tolerance": 0.01})

    if any(k in col_l for k in ["id", "code"]) and col != "id":
        return (exact_match, "exact_match", {})

    fused_non_null = fused_df[col].dropna() if col in fused_df.columns else pd.Series(dtype="object")
    gold_non_null = gold_df[col].dropna() if col in gold_df.columns else pd.Series(dtype="object")

    sample = pd.concat([fused_non_null.head(50), gold_non_null.head(50)], ignore_index=True)
    sample_text = sample.astype(str).str.strip().str.lower() if not sample.empty else pd.Series(dtype="object")
    bool_vocab = {"true", "false", "yes", "no", "1", "0", "y", "n", "t", "f"}
    if not sample_text.empty and sample_text.isin(bool_vocab).all():
        return (boolean_match, "boolean_match", {})

    if any(k in col_l for k in ["name", "title"]):
        return (tokenized_match, "tokenized_match", {"threshold": 0.85})

    if any(k in col_l for k in ["developer", "publisher", "studio", "artist", "author", "platform", "series", "genre", "country", "city", "street"]):
        return (tokenized_match, "tokenized_match", {"threshold": 0.75})

    return (tokenized_match, "tokenized_match", {"threshold": 0.8})


# ================================
# 1. LOAD FUSED OUTPUT AND GOLD SET
# ================================
fused = pd.read_csv("output/runs/20260323_072905_games/data_fusion/fusion_data.csv")

evaluation_set = load_xml(
    "input/datasets/games/testsets/validation_set_fusion.xml",
    name="validation_fusion_set",
    nested_handling="aggregate",
)

fused_eval = build_eval_view(fused, evaluation_set)

# ================================
# 2. NORMALIZE LIST-LIKE COLUMNS TO VALIDATION REPRESENTATION
# ================================
list_eval_columns = detect_list_like_columns(
    [fused_eval, evaluation_set],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if not c.startswith("_")]

if list_eval_columns:
    fused_eval, evaluation_set = normalize_list_like_columns(
        [fused_eval, evaluation_set], list_eval_columns
    )
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_canonicalize_list_text)
        if col in evaluation_set.columns:
            evaluation_set[col] = evaluation_set[col].apply(_canonicalize_list_text)
    print(json.dumps({
        "stage": "list_normalization",
        "columns": list_eval_columns
    }, indent=2))

# ================================
# 3. TYPE COERCION FOR ROBUST EVALUATION
# ================================
numeric_candidates = []
for col in set(fused_eval.columns) & set(evaluation_set.columns):
    cl = col.lower()
    if any(k in cl for k in ["year", "score", "rating", "sales", "revenue", "count", "price", "duration", "latitude", "longitude"]):
        numeric_candidates.append(col)

_coerce_numeric(fused_eval, numeric_candidates)
_coerce_numeric(evaluation_set, numeric_candidates)

boolean_candidates = [
    c for c in (set(fused_eval.columns) & set(evaluation_set.columns))
    if any(k in c.lower() for k in ["is_", "has_", "active", "enabled", "available", "bool", "flag"])
]
_coerce_boolean_text(fused_eval, boolean_candidates)
_coerce_boolean_text(evaluation_set, boolean_candidates)

# ================================
# 4. ID ALIGNMENT DIAGNOSTICS
# ================================
gold_ids = set(evaluation_set["id"].dropna().astype(str).tolist()) if "id" in evaluation_set.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
eval_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0

print(json.dumps({
    "stage": "id_alignment",
    "gold_ids": len(gold_ids),
    "direct_id_coverage": direct_cov,
    "mapped_eval_id_coverage": eval_cov
}, indent=2))

# ================================
# 5. REGISTER EVALUATION FUNCTIONS
#    Only for columns meaningful in BOTH fused and gold.
# ================================
strategy = DataFusionStrategy("games_validation_evaluation")
eval_funcs_summary = {}

shared_columns = sorted(
    c for c in (set(fused_eval.columns) & set(evaluation_set.columns))
    if c not in {"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence", "_fusion_sources"}
    and not c.startswith("_")
)

# Optionally skip structurally nested fields if needed later; do not drop genres_genre.
nested_like_columns = [c for c in shared_columns if len(re.findall(r"_", c)) >= 2]
skip_columns = set()

for col in shared_columns:
    if col in skip_columns:
        continue
    if not _has_meaningful_values(fused_eval, col):
        continue
    if not _has_meaningful_values(evaluation_set, col):
        continue

    fn, fn_name, kwargs = _choose_eval_function(col, fused_eval, evaluation_set, list_eval_columns)
    strategy.add_evaluation_function(col, fn, **kwargs)
    args = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[col] = f"{fn_name}({args})" if args else fn_name

print(
    "[EVAL_FUNCS] " +
    " | ".join(f"{col}:{desc}" for col, desc in eval_funcs_summary.items())
)

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

fused_eval = _strip_unevaluable_subcols(fused_eval, evaluation_set)

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

_coerce_numeric_cols(fused_eval, evaluation_set)

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

fused_eval = _normalize_list_separators(fused_eval, evaluation_set)
evaluation_results = evaluator.evaluate(
    fused_df=fused_eval,
    fused_id_column="eval_id",
    gold_df=evaluation_set,
    gold_id_column="id",
)

# ================================
# 7. PRINT STRUCTURED METRICS
# ================================
print(json.dumps({
    "stage": "evaluation_results",
    "metrics": evaluation_results
}, indent=2, default=str))

# ================================
# 8. WRITE RESULTS
# ================================
evaluation_output = "output/runs/20260323_072905_games/pipeline_evaluation/pipeline_evaluation.json"
with open(evaluation_output, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4, ensure_ascii=False)