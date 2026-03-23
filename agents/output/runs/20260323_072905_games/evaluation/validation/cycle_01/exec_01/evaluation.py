import pandas as pd
import json
import ast
import os
import re
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
OUTPUT_DIR = "output/runs/20260323_072905_games"
FUSION_PATH = os.path.join(OUTPUT_DIR, "data_fusion", "fusion_data.csv")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_OUTPUT_PATH = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
EVAL_SET_PATH = "input/datasets/games/testsets/validation_set_fusion.xml"

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


def _is_meaningful_series(series):
    if series is None:
        return False
    cleaned = series.dropna()
    if cleaned.empty:
        return False

    def _meaningful(v):
        if _safe_is_missing(v):
            return False
        if isinstance(v, (list, tuple, set)):
            return len([x for x in v if str(x).strip()]) > 0
        return str(v).strip() != ""

    return cleaned.apply(_meaningful).any()


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


def build_eval_view_from_sources(fused_df, gold_df):
    gold_ids = set(gold_df["id"].dropna().astype(str).tolist()) if "id" in gold_df.columns else set()
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
        else:
            enriched = dict(row_dict)
            enriched["eval_id"] = cluster_id if cluster_id else str(row_dict.get("id", ""))
            enriched["_eval_cluster_id"] = cluster_id if cluster_id else str(row_dict.get("id", ""))
            expanded_rows.append(enriched)

    fused_eval = pd.DataFrame(expanded_rows)

    if fused_eval.empty:
        fused_eval = fused_df.copy()
        fused_eval["eval_id"] = fused_eval["_id"].astype(str) if "_id" in fused_eval.columns else fused_eval["id"].astype(str)
        fused_eval["_eval_cluster_id"] = fused_eval["eval_id"]

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


def _coerce_numeric_like(df, cols):
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
    return df


def _coerce_datetime_like(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _looks_like_nested_subfield(col):
    parts = str(col).split("_")
    return len(parts) >= 3


# ================================
# 2. LOAD DATA
# ================================
fused = pd.read_csv(FUSION_PATH)
gold = load_xml(EVAL_SET_PATH, name="fusion_validation_set", nested_handling="aggregate")

fused_eval = build_eval_view_from_sources(fused, gold)

gold_ids = set(gold["id"].dropna().astype(str).tolist()) if "id" in gold.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
eval_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0

print(json.dumps({
    "stage": "validation",
    "id_alignment": {
        "gold_records": len(gold_ids),
        "direct_id_coverage": direct_cov,
        "mapped_eval_id_coverage": eval_cov
    }
}, indent=2))


# ================================
# 3. NORMALIZE TO VALIDATION-SET REPRESENTATION
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

date_cols = [c for c in ["releaseYear"] if c in fused_eval.columns and c in gold.columns]
numeric_cols = [c for c in ["criticScore", "userScore", "globalSales"] if c in fused_eval.columns and c in gold.columns]
boolean_cols = [c for c in (set(fused_eval.columns) & set(gold.columns)) if re.fullmatch(r"is[A-Z].*|has[A-Z].*|.*_flag", str(c) or "")]

fused_eval = _coerce_datetime_like(fused_eval, date_cols)
gold = _coerce_datetime_like(gold, date_cols)

fused_eval = _coerce_numeric_like(fused_eval, numeric_cols)
gold = _coerce_numeric_like(gold, numeric_cols)


# ================================
# 4. CHOOSE EVALUATION COLUMNS
# ================================
shared_columns = sorted(set(fused_eval.columns) & set(gold.columns))
skip_columns = {"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence", "_fusion_sources"}

candidate_columns = []
for col in shared_columns:
    if col in skip_columns:
        continue
    if str(col).startswith("_"):
        continue
    if _looks_like_nested_subfield(col):
        continue
    if not _is_meaningful_series(gold[col]):
        continue
    if not _is_meaningful_series(fused_eval[col]):
        continue
    candidate_columns.append(col)


# ================================
# 5. REGISTER EVALUATION FUNCTIONS
# ================================
strategy = DataFusionStrategy("games_validation_evaluation")
eval_funcs_summary = {}

def register_eval(column, fn, label, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items())
    eval_funcs_summary[column] = f"{label}({args})" if args else label

for col in candidate_columns:
    if col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif col in boolean_cols:
        register_eval(col, boolean_match, "boolean_match")
    elif col == "releaseYear":
        register_eval(col, year_only_match, "year_only_match")
    elif col == "criticScore":
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=5.0)
    elif col == "userScore":
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=1.0)
    elif col == "globalSales":
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.01)
    elif col.lower() in {"id", "code"} or col.endswith("Id") or col.endswith("_id"):
        register_eval(col, exact_match, "exact_match")
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)

print("EVAL_FUNCTIONS|" + " ; ".join(f"{k}={v}" for k, v in sorted(eval_funcs_summary.items())))


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
    fused_id_column="eval_id" if "eval_id" in fused_eval.columns else "_id",
    gold_df=gold,
    gold_id_column="id",
)

print(json.dumps({
    "stage": "validation",
    "evaluation_columns": candidate_columns,
    "list_columns": list_eval_columns,
    "metrics": evaluation_results
}, indent=2, default=str))


# ================================
# 7. WRITE RESULTS
# ================================
with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4, default=str)