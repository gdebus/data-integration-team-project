import pandas as pd
import json
import ast
import re
import os
from collections import Counter
from pathlib import Path
import sys

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
# HELPERS
# ================================

def _parse_source_ids(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text or text[0] not in "[({":
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
    """Expand fused rows via _fusion_sources to align fused clusters with gold source IDs."""
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
    official_like = sum(1 for v in values if any(tok in v.lower() for tok in ["republic", "kingdom", "states of", "federation"]))

    if alpha2 / total >= 0.70:
        return "alpha_2"
    if alpha3 / total >= 0.70:
        return "alpha_3"
    if numeric / total >= 0.70:
        return "numeric"
    if official_like / total >= 0.70:
        return "official_name"
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


def _meaningful_columns(df, exclude=None):
    exclude = set(exclude or [])
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if str(c).startswith("_"):
            continue
        series = df[c]
        if series.dropna().empty:
            continue
        cols.append(c)
    return cols


def _drop_derived_subcolumns(fused_cols, gold_cols):
    gold_set = set(gold_cols)
    keep = []
    for col in fused_cols:
        parent_candidates = [g for g in gold_set if col.startswith(f"{g}_")]
        if parent_candidates and col not in gold_set:
            continue
        keep.append(col)
    return keep


def _choose_eval_rule(column, fused_series, gold_series, detected_list_cols):
    col_lower = str(column).lower()

    if column in detected_list_cols:
        return tokenized_match, "tokenized_match", {"threshold": 0.75}

    if any(k in col_lower for k in ["founded", "date", "year", "birth", "released"]):
        return year_only_match, "year_only_match", {}

    if any(k in col_lower for k in ["country"]):
        return tokenized_match, "tokenized_match", {"threshold": 0.65}

    if any(k in col_lower for k in ["name", "city", "street", "industry", "sector", "continent", "keypeople"]):
        threshold = 0.85 if "name" in col_lower else 0.75
        return tokenized_match, "tokenized_match", {"threshold": threshold}

    if any(k in col_lower for k in ["revenue", "sales", "profit", "profits", "asset", "assets", "market value", "rank", "latitude", "longitude"]):
        return numeric_tolerance_match, "numeric_tolerance_match", {"tolerance": 0.01}

    fused_non_null = fused_series.dropna()
    gold_non_null = gold_series.dropna()

    if not fused_non_null.empty and not gold_non_null.empty:
        try:
            pd.to_numeric(fused_non_null.head(50), errors="raise")
            pd.to_numeric(gold_non_null.head(50), errors="raise")
            return numeric_tolerance_match, "numeric_tolerance_match", {"tolerance": 0.01}
        except Exception:
            pass

    bool_vals = {"true", "false", "0", "1", "yes", "no"}
    sample_vals = set(str(v).strip().lower() for v in pd.concat([fused_non_null.head(20), gold_non_null.head(20)]).tolist())
    if sample_vals and sample_vals.issubset(bool_vals):
        return boolean_match, "boolean_match", {}

    return tokenized_match, "tokenized_match", {"threshold": 0.85}


# ================================
# 0. OUTPUT / PATHS
# ================================

OUTPUT_DIR = "output/runs/20260322_102925_companies"
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

FUSED_PATH = "output/runs/20260322_102925_companies/data_fusion/fusion_data.csv"
EVAL_SET_PATH = "input/datasets/companies/testsets/test_set.xml"

# sealed_test => use test set
EVALUATION_STAGE = "sealed_test"

# ================================
# 1. LOAD DATA
# ================================

fused = pd.read_csv(FUSED_PATH)
fusion_eval_set = load_xml(EVAL_SET_PATH, name="fusion_eval_set", nested_handling="aggregate")
fused_eval = build_eval_view(fused, fusion_eval_set)

# ================================
# 2. COLUMN FILTERING / DIRECT MATCH ONLY
# ================================

fused_meaningful = _meaningful_columns(
    fused_eval,
    exclude={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence", "_fusion_sources"},
)
gold_meaningful = _meaningful_columns(
    fusion_eval_set,
    exclude={"id"},
)

common_columns = sorted(set(fused_meaningful) & set(gold_meaningful))
common_columns = _drop_derived_subcolumns(common_columns, gold_meaningful)

# Only evaluate columns that are meaningful in both datasets
final_eval_columns = []
for col in common_columns:
    if fusion_eval_set[col].dropna().empty or fused_eval[col].dropna().empty:
        continue
    final_eval_columns.append(col)

# ================================
# 3. NORMALIZE LIST-LIKE COLUMNS
# ================================

list_eval_columns = detect_list_like_columns(
    [fused_eval[["eval_id"] + [c for c in final_eval_columns if c in fused_eval.columns]],
     fusion_eval_set[["id"] + [c for c in final_eval_columns if c in fusion_eval_set.columns]]],
    exclude_columns={"id", "eval_id"},
)
list_eval_columns = [c for c in list_eval_columns if c in final_eval_columns and not c.startswith("_")]

if list_eval_columns:
    fused_eval, fusion_eval_set = normalize_list_like_columns([fused_eval, fusion_eval_set], list_eval_columns)
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
        if col in fusion_eval_set.columns:
            fusion_eval_set[col] = fusion_eval_set[col].apply(_normalize_list_text_case)
    print(json.dumps({"list_normalization": {"columns": list_eval_columns}}, indent=2))

# ================================
# 4. NORMALIZE COUNTRY COLUMNS
# ================================

country_eval_columns = [c for c in final_eval_columns if "country" in str(c).lower()]
if country_eval_columns:
    country_format = _infer_country_output_format(fusion_eval_set, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        fusion_eval_set[col] = fusion_eval_set[col].apply(lambda x: _normalize_country_safe(x, country_format))
    print(json.dumps({"country_normalization": {"format": country_format, "columns": country_eval_columns}}, indent=2))

# ================================
# 5. ID ALIGNMENT DIAGNOSTICS
# ================================

gold_ids = set(fusion_eval_set["id"].dropna().astype(str).tolist()) if "id" in fusion_eval_set.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0
print(json.dumps({
    "id_alignment": {
        "direct_id_coverage": {"matched": direct_cov, "total_gold": len(gold_ids)},
        "mapped_eval_id_coverage": {"matched": mapped_cov, "total_gold": len(gold_ids)},
    }
}, indent=2))

# ================================
# 6. REGISTER EVALUATION FUNCTIONS
# ================================

strategy = DataFusionStrategy("evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

for col in final_eval_columns:
    fn, fn_name, kwargs = _choose_eval_rule(col, fused_eval[col], fusion_eval_set[col], set(list_eval_columns))
    register_eval(col, fn, fn_name, **kwargs)

compact_summary = " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items()))
print(f"[EVAL_FUNCS] {compact_summary}")

# ================================
# 7. RUN EVALUATION
# ================================

evaluator = DataFusionEvaluator(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=os.path.join(EVAL_DIR, "debug_fusion_eval.jsonl"),
)

evaluation_results = evaluator.evaluate(
    fused_df=fused_eval,
    fused_id_column="eval_id" if "eval_id" in fused_eval.columns else "_id",
    gold_df=fusion_eval_set,
    gold_id_column="id",
)

# ================================
# 8. STRUCTURED PRINTING
# ================================

structured_output = {
    "evaluation_stage": EVALUATION_STAGE,
    "evaluation_set_path": EVAL_SET_PATH,
    "fused_path": FUSED_PATH,
    "evaluated_columns": final_eval_columns,
    "evaluation_functions": eval_funcs_summary,
    "metrics": evaluation_results,
}
print(json.dumps(structured_output, indent=2, default=str))

# ================================
# 9. WRITE RESULTS
# ================================

evaluation_output = "output/runs/20260322_102925_companies/pipeline_evaluation/pipeline_evaluation.json"
with open(evaluation_output, "w") as f:
    json.dump(evaluation_results, f, indent=4, default=str)