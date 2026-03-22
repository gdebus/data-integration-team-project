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
        try:
            if (_path / "list_normalization.py").is_file():
                if str(_path.resolve()) not in sys.path:
                    sys.path.append(str(_path.resolve()))
        except Exception:
            pass
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


def _detect_gold_prefix(gold_ids):
    prefixes = [g.split("_", 1)[0] for g in gold_ids if "_" in g]
    if not prefixes:
        return None
    return Counter(prefixes).most_common(1)[0][0] + "_"


def build_eval_view(fused_df, gold_df):
    """
    Map fused cluster rows back to gold IDs using _fusion_sources.
    If a fused row corresponds to multiple source IDs, duplicate it per gold ID.
    """
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
        if not candidates and cluster_id in gold_ids:
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


def _drop_unmatched_derived_columns(fused_df, gold_df):
    gold_cols = set(gold_df.columns)
    drop_cols = []
    for col in fused_df.columns:
        if col.startswith("_"):
            continue
        if col in gold_cols:
            continue
        for gold_col in gold_cols:
            if col.startswith(f"{gold_col}_"):
                drop_cols.append(col)
                break
    if drop_cols:
        fused_df = fused_df.drop(columns=sorted(set(drop_cols)), errors="ignore")
    return fused_df, sorted(set(drop_cols))


def _meaningful_in_gold(df, col):
    if col not in df.columns:
        return False
    series = df[col]
    if len(series) == 0:
        return False
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    if non_null.astype(str).str.strip().eq("").all():
        return False
    return True


def _choose_eval_rule(col, fused_df, gold_df, list_like_cols):
    cl = col.lower()

    if col in list_like_cols:
        return tokenized_match, "tokenized_match", {"threshold": 0.75}

    if cl == "id":
        return exact_match, "exact_match", {}

    if cl.endswith("_id") or cl in {"code", "identifier", "isbn", "ean", "upc"}:
        return exact_match, "exact_match", {}

    if cl.startswith("is_") or cl.startswith("has_") or cl in {"active", "public", "listed"}:
        return boolean_match, "boolean_match", {}

    if any(k in cl for k in ["date", "founded", "released", "born", "died", "year"]):
        return year_only_match, "year_only_match", {}

    if "country" in cl:
        return tokenized_match, "tokenized_match", {"threshold": 0.65}

    if any(k in cl for k in ["name", "title", "label", "industry", "city", "street", "keypeople"]):
        threshold = 0.85 if "name" in cl else 0.75
        return tokenized_match, "tokenized_match", {"threshold": threshold}

    fused_num = pd.to_numeric(fused_df[col], errors="coerce") if col in fused_df.columns else pd.Series(dtype=float)
    gold_num = pd.to_numeric(gold_df[col], errors="coerce") if col in gold_df.columns else pd.Series(dtype=float)
    if fused_num.notna().mean() >= 0.6 and gold_num.notna().mean() >= 0.6:
        tolerance = 0.01
        if cl in {"latitude", "longitude"}:
            tolerance = 0.001
        return numeric_tolerance_match, "numeric_tolerance_match", {"tolerance": tolerance}

    return tokenized_match, "tokenized_match", {"threshold": 0.80}


# ================================
# 0. OUTPUT / INPUT PATHS
# ================================

OUTPUT_DIR = "output/runs/20260322_102925_companies"
FUSION_PATH = "output/runs/20260322_102925_companies/data_fusion/fusion_data.csv"
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_SET_PATH = "input/datasets/companies/testsets/validation_set.xml"

os.makedirs(EVAL_DIR, exist_ok=True)


# ================================
# 1. LOAD FUSED OUTPUT AND GOLD SET
# ================================

fused = pd.read_csv(FUSION_PATH)
fusion_eval_set = load_xml(
    EVAL_SET_PATH,
    name="validation_set",
    nested_handling="aggregate",
)

fused_eval = build_eval_view(fused, fusion_eval_set)
fused_eval, dropped_derived = _drop_unmatched_derived_columns(fused_eval, fusion_eval_set)


# ================================
# 2. NORMALIZE LIST-LIKE COLUMNS TO SHARED REPRESENTATION
# ================================

list_eval_columns = detect_list_like_columns(
    [fused_eval, fusion_eval_set],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if not c.startswith("_")]

if list_eval_columns:
    fused_eval, fusion_eval_set = normalize_list_like_columns(
        [fused_eval, fusion_eval_set],
        list_eval_columns,
    )
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
        if col in fusion_eval_set.columns:
            fusion_eval_set[col] = fusion_eval_set[col].apply(_normalize_list_text_case)


# ================================
# 3. NORMALIZE COUNTRY COLUMNS TO VALIDATION REPRESENTATION
# ================================

country_eval_columns = [
    c for c in (set(fused_eval.columns) & set(fusion_eval_set.columns))
    if "country" in str(c).lower()
]

country_format = None
if country_eval_columns:
    country_format = _infer_country_output_format(fusion_eval_set, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        fusion_eval_set[col] = fusion_eval_set[col].apply(lambda x: _normalize_country_safe(x, country_format))


# ================================
# 4. SELECT DIRECTLY EVALUABLE COLUMNS ONLY
# ================================

shared_columns = sorted(set(fused_eval.columns) & set(fusion_eval_set.columns))
excluded_columns = {
    "id",
    "_id",
    "eval_id",
    "_eval_cluster_id",
    "_fusion_confidence",
    "_fusion_sources",
}
provenance_like = {c for c in shared_columns if c.endswith("_provenance") or c.startswith("_")}

candidate_columns = [
    c for c in shared_columns
    if c not in excluded_columns
    and c not in provenance_like
    and _meaningful_in_gold(fusion_eval_set, c)
]

# Keep only direct 1:1 columns that exist in both tables and are meaningful in gold
candidate_columns = [c for c in candidate_columns if c in fused_eval.columns and c in fusion_eval_set.columns]


# ================================
# 5. REGISTER EVALUATION FUNCTIONS
# ================================

strategy = DataFusionStrategy("validation_evaluation_strategy")
eval_funcs_summary = {}

for col in candidate_columns:
    fn, fn_name, kwargs = _choose_eval_rule(col, fused_eval, fusion_eval_set, list_eval_columns)
    strategy.add_evaluation_function(col, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items())
    eval_funcs_summary[col] = f"{fn_name}({args})" if args else fn_name

print("[EVALUATION_FUNCTIONS] " + " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items())))


# ================================
# 6. DIAGNOSTICS
# ================================

gold_ids = set(fusion_eval_set["id"].dropna().astype(str).tolist()) if "id" in fusion_eval_set.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0

diagnostics = {
    "stage": "validation",
    "fusion_path": FUSION_PATH,
    "evaluation_set_path": EVAL_SET_PATH,
    "fused_rows": int(len(fused)),
    "fused_eval_rows": int(len(fused_eval)),
    "gold_rows": int(len(fusion_eval_set)),
    "direct_id_coverage": {"matched": int(direct_cov), "total_gold": int(len(gold_ids))},
    "mapped_eval_id_coverage": {"matched": int(mapped_cov), "total_gold": int(len(gold_ids))},
    "country_normalization_format": country_format,
    "list_like_columns": list_eval_columns,
    "dropped_derived_columns": dropped_derived,
    "evaluated_columns": candidate_columns,
}

print(json.dumps({"diagnostics": diagnostics}, indent=2))


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
    fused_id_column="eval_id",
    expected_df=fusion_eval_set,
    expected_id_column="id",
)

structured_output = {
    "stage": "validation",
    "evaluation_functions": eval_funcs_summary,
    "diagnostics": diagnostics,
    "metrics": evaluation_results,
}

print(json.dumps({"metrics": evaluation_results}, indent=2))


# ================================
# 8. WRITE RESULTS
# ================================

evaluation_output = "output/runs/20260322_102925_companies/pipeline_evaluation/pipeline_evaluation.json"
with open(evaluation_output, "w", encoding="utf-8") as f:
    json.dump(structured_output, f, indent=4)