import pandas as pd
import json
import ast
import re
import os
import sys
from collections import Counter
from pathlib import Path

from PyDI.io import load_xml
from PyDI.normalization import normalize_country
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
# 0. OUTPUT / INPUT PATHS
# ================================
OUTPUT_DIR = "output/runs/20260322_175903_games"
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

FUSED_PATH = "output/runs/20260322_175903_games/data_fusion/fusion_data.csv"
EVAL_SET_PATH = "input/datasets/games/testsets/test_set_fusion.xml"
EVAL_OUTPUT_PATH = "output/runs/20260322_175903_games/pipeline_evaluation/pipeline_evaluation.json"


# ================================
# HELPERS
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
    gold_ids = set(gold_df["id"].dropna().astype(str).tolist()) if "id" in gold_df.columns else set()
    gold_prefix = _detect_gold_prefix(gold_ids)
    expanded_rows = []

    for _, row in fused_df.iterrows():
        row_dict = row.to_dict()
        cluster_id = str(row_dict.get("_id", ""))
        source_ids = _parse_source_ids(row_dict.get("_fusion_sources"))

        candidates = []
        if "eval_id" in row_dict and str(row_dict.get("eval_id", "")).strip():
            candidates = [str(row_dict["eval_id"]).strip()]

        if not candidates and gold_prefix:
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
        fallback["eval_id"] = fallback["_id"].astype(str)
        fallback["_eval_cluster_id"] = fallback["_id"].astype(str)
        return fallback

    fused_eval = pd.DataFrame(expanded_rows)
    if "_fusion_confidence" in fused_eval.columns:
        fused_eval["_fusion_confidence"] = pd.to_numeric(
            fused_eval["_fusion_confidence"], errors="coerce"
        ).fillna(0.0)
        fused_eval = fused_eval.sort_values("_fusion_confidence", ascending=False)
    fused_eval = fused_eval.drop_duplicates(subset=["eval_id"], keep="first")
    return fused_eval


def _infer_country_output_format(gold_df, country_columns):
    values = []
    for col in country_columns:
        if col not in gold_df.columns:
            continue
        values.extend(gold_df[col].dropna().astype(str).head(500).tolist())
    if not values:
        return "name"
    total = max(len(values), 1)
    alpha2 = sum(1 for v in values if re.fullmatch(r"[A-Z]{2}", v.strip()))
    alpha3 = sum(1 for v in values if re.fullmatch(r"[A-Z]{3}", v.strip()))
    numeric = sum(1 for v in values if re.fullmatch(r"\d{3}", v.strip()))
    if alpha2 / total >= 0.70:
        return "alpha_2"
    if alpha3 / total >= 0.70:
        return "alpha_3"
    if numeric / total >= 0.70:
        return "numeric"
    return "name"


def _normalize_country_safe(value, output_format):
    if _safe_is_missing(value):
        return value
    text = str(value).strip()
    if not text:
        return value
    try:
        normalized = normalize_country(text, output_format=output_format)
        return normalized if normalized else text
    except Exception:
        return text


def _normalize_list_text_case(value):
    if _safe_is_missing(value):
        return value
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    text = str(value).strip()
    return text.lower() if text else text


def _drop_derived_subcolumns_without_gold_pair(fused_df, gold_df):
    gold_cols = set(gold_df.columns)
    to_drop = []
    for col in fused_df.columns:
        if col in gold_cols:
            continue
        if "_" in col and col.split("_", 1)[0] in gold_cols:
            to_drop.append(col)
    if to_drop:
        fused_df = fused_df.drop(columns=to_drop, errors="ignore")
        print(f"[DROP DERIVED SUBCOLUMNS] dropped={to_drop}")
    return fused_df


def _meaningful_common_columns(fused_df, gold_df, id_cols):
    common = []
    for col in sorted(set(fused_df.columns) & set(gold_df.columns)):
        if col in id_cols or col.startswith("_"):
            continue
        gold_non_null = gold_df[col].notna().sum() if col in gold_df.columns else 0
        fused_non_null = fused_df[col].notna().sum() if col in fused_df.columns else 0
        if gold_non_null == 0 or fused_non_null == 0:
            continue
        common.append(col)
    return common


def _coerce_numeric_consistently(df_left, df_right, columns):
    for col in columns:
        if col in df_left.columns:
            df_left[col] = pd.to_numeric(df_left[col], errors="coerce")
            non_null = df_left[col].dropna()
            if len(non_null) > 0 and non_null.apply(lambda x: float(x).is_integer()).all():
                df_left[col] = df_left[col].astype("Int64")
        if col in df_right.columns:
            df_right[col] = pd.to_numeric(df_right[col], errors="coerce")
            non_null = df_right[col].dropna()
            if len(non_null) > 0 and non_null.apply(lambda x: float(x).is_integer()).all():
                df_right[col] = df_right[col].astype("Int64")


# ================================
# 1. LOAD FUSED OUTPUT AND GOLD SET
# ================================
fused = pd.read_csv(FUSED_PATH)
fusion_eval_set = load_xml(EVAL_SET_PATH, name="fusion_validation_set", nested_handling="aggregate")

fused_eval = build_eval_view(fused, fusion_eval_set)
fused_eval = _drop_derived_subcolumns_without_gold_pair(fused_eval, fusion_eval_set)

print(f"[LOAD] fused_rows={len(fused)} fused_eval_rows={len(fused_eval)} gold_rows={len(fusion_eval_set)}")
print(f"[LOAD] fused_cols={len(fused.columns)} gold_cols={len(fusion_eval_set.columns)}")


# ================================
# 2. NORMALIZE LIST-LIKE COLUMNS
# ================================
list_eval_columns = detect_list_like_columns(
    [fused_eval, fusion_eval_set],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if not c.startswith("_")]
if list_eval_columns:
    fused_eval, fusion_eval_set = normalize_list_like_columns([fused_eval, fusion_eval_set], list_eval_columns)
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
        if col in fusion_eval_set.columns:
            fusion_eval_set[col] = fusion_eval_set[col].apply(_normalize_list_text_case)
    print(f"[LIST NORMALIZATION] columns={list_eval_columns}")


# ================================
# 3. NORMALIZE COUNTRY COLUMNS
# ================================
country_eval_columns = [
    c for c in (set(fused_eval.columns) & set(fusion_eval_set.columns))
    if "country" in str(c).lower()
]
if country_eval_columns:
    country_format = _infer_country_output_format(fusion_eval_set, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        fusion_eval_set[col] = fusion_eval_set[col].apply(lambda x: _normalize_country_safe(x, country_format))
    print(f"[COUNTRY NORMALIZATION] format={country_format} columns={country_eval_columns}")


# ================================
# 4. COERCE NUMERIC-LIKE COLUMNS CONSISTENTLY
# ================================
numeric_hint_cols = []
for col in set(fused_eval.columns) & set(fusion_eval_set.columns):
    lc = col.lower()
    if any(token in lc for token in ["year", "score", "sales", "count", "rating", "price", "duration", "latitude", "longitude"]):
        numeric_hint_cols.append(col)

_coerce_numeric_consistently(fused_eval, fusion_eval_set, numeric_hint_cols)
print(f"[NUMERIC COERCION] columns={sorted(numeric_hint_cols)}")


# ================================
# 5. ID ALIGNMENT DIAGNOSTICS
# ================================
gold_ids = set(fusion_eval_set["id"].dropna().astype(str).tolist()) if "id" in fusion_eval_set.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
eval_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0
print(f"[ID ALIGNMENT] direct_cluster_id_coverage={direct_cov}/{len(gold_ids)}")
print(f"[ID ALIGNMENT] eval_id_coverage={eval_cov}/{len(gold_ids)}")


# ================================
# 6. CHOOSE EVALUABLE COLUMNS
# ================================
evaluable_columns = _meaningful_common_columns(
    fused_eval,
    fusion_eval_set,
    id_cols={"id", "_id", "eval_id", "_eval_cluster_id"},
)
print(f"[EVALUABLE COLUMNS] {evaluable_columns}")


# ================================
# 7. REGISTER EVALUATION FUNCTIONS
# ================================
strategy = DataFusionStrategy("games_fusion_evaluation_strategy")
eval_funcs_summary = {}


def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name


for col in evaluable_columns:
    lc = col.lower()

    if col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif lc in {"id", "eval_id"} or lc.endswith("id"):
        register_eval(col, exact_match, "exact_match")
    elif any(token in lc for token in ["releaseyear", "founded", "date", "year"]):
        register_eval(col, year_only_match, "year_only_match")
    elif any(token in lc for token in ["score", "sales", "rating", "price", "duration", "latitude", "longitude", "count"]):
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.02)
    elif any(token in lc for token in ["is_", "has_", "flag", "active", "enabled", "available"]):
        register_eval(col, boolean_match, "boolean_match")
    elif "country" in lc:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.65)
    elif any(token in lc for token in ["name", "title", "developer", "publisher", "platform", "series", "genre", "street", "city"]):
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.80)

compact_summary = " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items()))
print(f"[EVAL FUNCTIONS] {compact_summary}")


# ================================
# 8. RUN EVALUATION
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

fused_eval = _strip_unevaluable_subcols(fused_eval, fusion_eval_set)

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

_coerce_numeric_cols(fused_eval, fusion_eval_set)

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

fused_eval = _normalize_list_separators(fused_eval, fusion_eval_set)
evaluation_results = evaluator.evaluate(
    fused_df=fused_eval,
    fused_id_column="eval_id" if "eval_id" in fused_eval.columns else "_id",
    gold_df=fusion_eval_set,
    gold_id_column="id",
)


# ================================
# 9. PRINT STRUCTURED METRICS
# ================================
print("[EVALUATION RESULTS]")
print(json.dumps(evaluation_results, indent=2, default=str))


# ================================
# 10. WRITE RESULTS
# ================================
with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4, default=str)

print(f"[WRITE] saved={EVAL_OUTPUT_PATH}")