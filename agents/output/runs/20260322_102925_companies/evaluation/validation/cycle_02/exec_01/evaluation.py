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
# HELPERS: Build evaluation view by mapping fused cluster IDs to gold test set IDs
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


# ================================
# HELPERS: Normalization / filtering
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


def _normalize_text_cell(value):
    if _safe_is_missing(value):
        return value
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def _normalize_numeric_string(value):
    if _safe_is_missing(value):
        return value
    text = str(value).strip().replace(",", "")
    return text


def _gold_column_meaningful(df, col):
    return col in df.columns and df[col].notna().any()


def _choose_eval_function(column, fused_df, gold_df, list_eval_columns):
    c = str(column).lower()

    if c == "id" or c.endswith("_id"):
        return exact_match, "exact_match", {}

    if c in list_eval_columns:
        return tokenized_match, "tokenized_match", {"threshold": 0.75}

    if "founded" in c or "date" in c or "year" in c:
        return year_only_match, "year_only_match", {}

    if c.startswith("is_") or c.startswith("has_") or c.endswith("_flag") or c.endswith("_bool"):
        return boolean_match, "boolean_match", {}

    numeric_like_names = {"revenue", "assets", "sales", "profits", "market value", "market_value", "latitude", "longitude", "rank"}
    if c in numeric_like_names:
        if c in {"latitude", "longitude"}:
            return numeric_tolerance_match, "numeric_tolerance_match", {"tolerance": 0.001}
        if c == "rank":
            return numeric_tolerance_match, "numeric_tolerance_match", {"tolerance": 0.0}
        return numeric_tolerance_match, "numeric_tolerance_match", {"tolerance": 0.01}

    if "country" in c:
        return tokenized_match, "tokenized_match", {"threshold": 0.65}
    if "city" in c or "street" in c or "address" in c:
        return tokenized_match, "tokenized_match", {"threshold": 0.75}
    if "name" in c or "industry" in c or "sector" in c or "continent" in c:
        return tokenized_match, "tokenized_match", {"threshold": 0.85}

    return tokenized_match, "tokenized_match", {"threshold": 0.85}


# ================================
# 0. OUTPUT DIRECTORY
# ================================
OUTPUT_DIR = "output/runs/20260322_102925_companies"

FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

# ================================
# 1. LOAD FUSED OUTPUT AND GOLD VALIDATION SET
# ================================
fused = pd.read_csv(os.path.join(FUSION_DIR, "fusion_data.csv"))
fusion_eval_set = load_xml(
    "input/datasets/companies/testsets/validation_set.xml",
    name="fusion_validation_set",
    nested_handling="aggregate",
)

fused_eval = build_eval_view(fused, fusion_eval_set)

# ================================
# 2. DROP NON-DIRECT / PROVENANCE COLUMNS FROM EVALUATION SPACE
# ================================
gold_columns = set(fusion_eval_set.columns)
fused_columns = set(fused_eval.columns)

eval_exclude = {
    "_id", "_eval_cluster_id", "_fusion_confidence", "_fusion_sources",
    "eval_id",
}
eval_exclude.update({c for c in fused_columns if c.startswith("_")})
eval_exclude.update({c for c in gold_columns if c.endswith("_provenance")})

# Drop fused derived sub-columns when no direct gold counterpart exists
derived_drop = []
for col in fused_eval.columns:
    if col in eval_exclude or col in gold_columns:
        continue
    prefixes = [g for g in gold_columns if col.startswith(f"{g}_")]
    if prefixes:
        derived_drop.append(col)

if derived_drop:
    fused_eval = fused_eval.drop(columns=[c for c in derived_drop if c in fused_eval.columns])

# Keep only direct 1:1 shared meaningful columns
shared_columns = [
    c for c in fused_eval.columns
    if c in fusion_eval_set.columns
    and c not in eval_exclude
    and _gold_column_meaningful(fusion_eval_set, c)
]

# ================================
# 3. NORMALIZE LIST-LIKE COLUMNS FOR EVALUATION
# ================================
list_eval_columns = detect_list_like_columns(
    [fused_eval[shared_columns + ["eval_id"]] if "eval_id" in fused_eval.columns else fused_eval[shared_columns],
     fusion_eval_set[shared_columns + ["id"]] if "id" in fusion_eval_set.columns else fusion_eval_set[shared_columns]],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if c in shared_columns and not c.startswith("_")]

if list_eval_columns:
    fused_eval, fusion_eval_set = normalize_list_like_columns([fused_eval, fusion_eval_set], list_eval_columns)
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
        if col in fusion_eval_set.columns:
            fusion_eval_set[col] = fusion_eval_set[col].apply(_normalize_list_text_case)

# ================================
# 4. NORMALIZE COUNTRY COLUMNS TO VALIDATION REPRESENTATION
# ================================
country_eval_columns = [c for c in shared_columns if "country" in str(c).lower()]
if country_eval_columns:
    country_format = _infer_country_output_format(fusion_eval_set, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        fusion_eval_set[col] = fusion_eval_set[col].apply(lambda x: _normalize_country_safe(x, country_format))
else:
    country_format = None

# ================================
# 5. LIGHT TEXT / NUMERIC CANONICALIZATION
# ================================
for col in shared_columns:
    c = str(col).lower()
    if col in list_eval_columns:
        continue
    if "country" in c:
        continue
    if c in {"revenue", "assets", "sales", "profits", "market value", "market_value", "rank", "latitude", "longitude"}:
        fused_eval[col] = fused_eval[col].apply(_normalize_numeric_string)
        fusion_eval_set[col] = fusion_eval_set[col].apply(_normalize_numeric_string)
    elif "founded" in c or "date" in c or "year" in c:
        continue
    else:
        fused_eval[col] = fused_eval[col].apply(_normalize_text_cell)
        fusion_eval_set[col] = fusion_eval_set[col].apply(_normalize_text_cell)

# ================================
# 6. ID ALIGNMENT DIAGNOSTICS
# ================================
gold_ids = set(fusion_eval_set["id"].dropna().astype(str).tolist()) if "id" in fusion_eval_set.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0

print(json.dumps({
    "stage": "validation",
    "id_alignment": {
        "gold_ids": len(gold_ids),
        "direct_id_coverage": direct_cov,
        "mapped_eval_id_coverage": mapped_cov,
    }
}, indent=2))

# ================================
# 7. REGISTER EVALUATION FUNCTIONS
# ================================
strategy = DataFusionStrategy("evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

for col in shared_columns:
    if col == "id":
        continue
    if not _gold_column_meaningful(fusion_eval_set, col):
        continue
    fn, fn_name, kwargs = _choose_eval_function(col, fused_eval, fusion_eval_set, list_eval_columns)
    register_eval(col, fn, fn_name, **kwargs)

print("EVAL_FUNCTIONS|" + " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items())))

# ================================
# 8. RUN EVALUATION
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
# 9. PRINT STRUCTURED METRICS
# ================================
print(json.dumps({
    "stage": "validation",
    "evaluation_target": "input/datasets/companies/testsets/validation_set.xml",
    "fused_output": "output/runs/20260322_102925_companies/data_fusion/fusion_data.csv",
    "country_normalization_format": country_format,
    "evaluated_columns": sorted(eval_funcs_summary.keys()),
    "list_like_columns": sorted(list_eval_columns),
    "metrics": evaluation_results,
}, indent=2, default=str))

# ================================
# 10. WRITE RESULTS
# ================================
evaluation_output = "output/runs/20260322_102925_companies/pipeline_evaluation/pipeline_evaluation.json"
with open(evaluation_output, "w") as f:
    json.dump(evaluation_results, f, indent=4, default=str)