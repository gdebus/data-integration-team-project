import pandas as pd
import json
import ast
import os
import re
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
# 0. PATHS
# ================================
OUTPUT_DIR = "output/runs/20260323_072905_games"
FUSION_PATH = "output/runs/20260323_072905_games/data_fusion/fusion_data.csv"
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_OUTPUT = "output/runs/20260323_072905_games/pipeline_evaluation/pipeline_evaluation.json"
DEBUG_OUTPUT = os.path.join(EVAL_DIR, "debug_fusion_eval.jsonl")
EVAL_SET_PATH = "input/datasets/games/testsets/test_set_fusion.xml"
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
        pass
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    if isinstance(value, str):
        return value.strip() == ""
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


def _detect_gold_prefix(gold_ids):
    prefixes = [g.split("_", 1)[0] for g in gold_ids if "_" in g]
    if not prefixes:
        return None
    return Counter(prefixes).most_common(1)[0][0] + "_"


def _normalize_text_scalar(value):
    if _safe_is_missing(value):
        return value
    return re.sub(r"\s+", " ", str(value).strip())


def _normalize_list_text_case(value):
    if _safe_is_missing(value):
        return value
    if isinstance(value, (list, tuple, set)):
        out = []
        for v in value:
            if _safe_is_missing(v):
                continue
            s = re.sub(r"\s+", " ", str(v).strip()).lower()
            if s:
                out.append(s)
        return out
    text = re.sub(r"\s+", " ", str(value).strip()).lower()
    return text if text else value


def _coerce_numeric_like(df, columns):
    for col in columns:
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


def _meaningful_nonempty_series(series):
    if series is None:
        return pd.Series(dtype="object")
    if isinstance(series, pd.Series):
        s = series.copy()
    else:
        s = pd.Series(series)
    s = s.dropna()

    def _has_content(x):
        if isinstance(x, (list, tuple, set)):
            return len([v for v in x if not _safe_is_missing(v) and str(v).strip() != ""]) > 0
        return str(x).strip() != ""

    return s[s.apply(_has_content)]


def _is_boolean_like(series):
    vals = set()
    for x in _meaningful_nonempty_series(series).head(200):
        sx = str(x).strip().lower()
        vals.add(sx)
    return len(vals) > 0 and vals.issubset({"true", "false", "0", "1", "yes", "no"})


def _is_nested_structural_column(col):
    parts = str(col).split("_")
    return len(parts) >= 3


def _compact_eval_summary(summary_dict):
    return " | ".join(f"{k}:{v}" for k, v in sorted(summary_dict.items()))


def build_eval_view(fused_df, gold_df):
    """
    Robust ID alignment:
    1) map fused rows to all gold IDs found in _fusion_sources
    2) if no source match, fall back to prefix-based matches
    3) if still no match, use cluster _id
    """
    gold_ids = set(gold_df["id"].dropna().astype(str).tolist()) if "id" in gold_df.columns else set()
    gold_prefix = _detect_gold_prefix(gold_ids)
    expanded_rows = []

    for _, row in fused_df.iterrows():
        row_dict = row.to_dict()
        cluster_id = str(row_dict.get("_id", ""))
        source_ids = _parse_source_ids(row_dict.get("_fusion_sources"))

        candidates = [sid for sid in source_ids if sid in gold_ids]

        if not candidates and gold_prefix:
            candidates = [sid for sid in source_ids if sid.startswith(gold_prefix) and sid in gold_ids]

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
        fused_eval["_fusion_confidence"] = pd.to_numeric(fused_eval["_fusion_confidence"], errors="coerce").fillna(0.0)
        fused_eval = fused_eval.sort_values("_fusion_confidence", ascending=False)
    fused_eval = fused_eval.drop_duplicates(subset=["eval_id"], keep="first")
    return fused_eval


# ================================
# 2. LOAD DATA
# ================================
fused = pd.read_csv(FUSION_PATH)
fusion_gold = load_xml(EVAL_SET_PATH, name="fusion_test_set", nested_handling="aggregate")

# Prefer robust remapping via _fusion_sources regardless of existing eval_id.
fused_eval = build_eval_view(fused, fusion_gold)


# ================================
# 3. NORMALIZE TO SAME REPRESENTATION
# Use validation/test representation directly; do not country-normalize.
# ================================
for df in [fused_eval, fusion_gold]:
    for col in df.columns:
        if col.startswith("_") or col in {"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_sources"}:
            continue
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: _normalize_text_scalar(x) if not isinstance(x, (list, tuple, set)) else x
            )

list_eval_columns = detect_list_like_columns(
    [fused_eval, fusion_gold],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if not c.startswith("_")]

if list_eval_columns:
    fused_eval, fusion_gold = normalize_list_like_columns([fused_eval, fusion_gold], list_eval_columns)
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
        if col in fusion_gold.columns:
            fusion_gold[col] = fusion_gold[col].apply(_normalize_list_text_case)
    print(json.dumps({"list_normalization": {"columns": list_eval_columns}}, indent=2))

numeric_cols = []
for col in set(fused_eval.columns).intersection(fusion_gold.columns):
    cl = str(col).lower()
    if any(tok in cl for tok in ["score", "sales", "rating", "count", "year", "price", "duration"]):
        if col not in list_eval_columns and col not in {"id", "_id", "eval_id", "_eval_cluster_id"}:
            numeric_cols.append(col)

fused_eval = _coerce_numeric_like(fused_eval, numeric_cols)
fusion_gold = _coerce_numeric_like(fusion_gold, numeric_cols)

if "releaseYear" in fused_eval.columns:
    fused_eval["releaseYear"] = pd.to_datetime(fused_eval["releaseYear"], errors="coerce")
if "releaseYear" in fusion_gold.columns:
    fusion_gold["releaseYear"] = pd.to_datetime(fusion_gold["releaseYear"], errors="coerce")


# ================================
# 4. ID ALIGNMENT DIAGNOSTICS
# ================================
gold_ids = set(fusion_gold["id"].dropna().astype(str).tolist()) if "id" in fusion_gold.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
existing_eval_cov = len(set(fused["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0

print(json.dumps({
    "id_alignment": {
        "gold_records": len(gold_ids),
        "direct__id_coverage": direct_cov,
        "existing_eval_id_coverage": existing_eval_cov,
        "rebuilt_eval_id_coverage": mapped_cov
    }
}, indent=2))


# ================================
# 5. REGISTER EVALUATION FUNCTIONS
# Use attribute-aware functions; do not default to exact everywhere.
# Skip empty/non-meaningful columns in either dataset.
# Skip structurally nested columns that are likely XML artifacts.
# ================================
strategy = DataFusionStrategy("evaluation_strategy")
eval_funcs_summary = {}
skipped_columns = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

shared_columns = sorted(set(fused_eval.columns).intersection(fusion_gold.columns))
shared_columns = [c for c in shared_columns if c not in {"id", "_id", "eval_id", "_eval_cluster_id"} and not c.startswith("_")]

for col in shared_columns:
    fused_nonempty = _meaningful_nonempty_series(fused_eval[col])
    gold_nonempty = _meaningful_nonempty_series(fusion_gold[col])

    if fused_nonempty.empty or gold_nonempty.empty:
        skipped_columns[col] = "empty_or_non_meaningful_in_fused_or_gold"
        continue

    if _is_nested_structural_column(col):
        skipped_columns[col] = "nested_structural_xml_artifact_skipped"
        continue

    col_l = str(col).lower()

    if col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif col in {"id"} or col_l.endswith("id") and col != "id":
        register_eval(col, exact_match, "exact_match")
    elif _is_boolean_like(pd.concat([fused_nonempty, gold_nonempty], ignore_index=True)):
        register_eval(col, boolean_match, "boolean_match")
    elif "release" in col_l or col_l.endswith("date"):
        register_eval(col, year_only_match, "year_only_match")
    elif "criticscore" in col_l or ("critic" in col_l and "score" in col_l):
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=5.0)
    elif "userscore" in col_l or ("user" in col_l and "score" in col_l) or "rating" in col_l:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=1.0)
    elif "sales" in col_l:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.5)
    elif "year" in col_l and pd.api.types.is_numeric_dtype(fused_eval[col]) and pd.api.types.is_numeric_dtype(fusion_gold[col]):
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.0)
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)

print(json.dumps({
    "evaluation_functions_one_line": _compact_eval_summary(eval_funcs_summary),
    "registered_columns": sorted(eval_funcs_summary.keys()),
    "skipped_columns": skipped_columns
}, indent=2))


# ================================
# 6. RUN EVALUATION
# ================================
evaluator = DataFusionEvaluator(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=DEBUG_OUTPUT,
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
print(json.dumps({
    "evaluation_stage": "sealed_test",
    "fusion_output_path": FUSION_PATH,
    "evaluation_set_path": EVAL_SET_PATH,
    "metrics": evaluation_results
}, indent=2, default=str))


# ================================
# 8. WRITE RESULTS
# ================================
with open(EVAL_OUTPUT, "w") as f:
    json.dump(evaluation_results, f, indent=4, default=str)