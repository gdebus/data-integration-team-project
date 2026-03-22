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
    official_name_hits = sum(
        1 for v in values
        if any(tok in v.lower() for tok in ["republic", "kingdom", "states of", "federation", "union"])
    )
    if official_name_hits / total >= 0.50:
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


def _meaningful_in_gold(df, col):
    if col not in df.columns:
        return False
    series = df[col]
    if series.dropna().empty:
        return False
    non_empty = series.dropna().astype(str).str.strip()
    return (non_empty != "").any()


def _drop_derived_subcolumns(fused_df, gold_df):
    keep_cols = []
    gold_cols = set(gold_df.columns)
    for col in fused_df.columns:
        if col.startswith("_") or col in {"eval_id", "id"}:
            keep_cols.append(col)
            continue
        parent = col.split("_", 1)[0] if "_" in col else None
        if parent and parent in gold_cols and col not in gold_cols:
            continue
        keep_cols.append(col)
    return fused_df[keep_cols].copy()


OUTPUT_DIR = "output/runs/20260322_102925_companies"
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

fused = pd.read_csv(os.path.join(FUSION_DIR, "fusion_data.csv"))
fusion_eval_set = load_xml(
    "input/datasets/companies/testsets/validation_set.xml",
    name="fusion_validation_set",
    nested_handling="aggregate",
)

fused_eval = build_eval_view(fused, fusion_eval_set)
fused_eval = _drop_derived_subcolumns(fused_eval, fusion_eval_set)

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

country_eval_columns = [
    c for c in (set(fused_eval.columns) & set(fusion_eval_set.columns))
    if "country" in str(c).lower() and _meaningful_in_gold(fusion_eval_set, c)
]
if country_eval_columns:
    country_format = _infer_country_output_format(fusion_eval_set, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        fusion_eval_set[col] = fusion_eval_set[col].apply(lambda x: _normalize_country_safe(x, country_format))
else:
    country_format = None

gold_ids = set(fusion_eval_set["id"].dropna().astype(str).tolist()) if "id" in fusion_eval_set.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0

print(json.dumps({
    "stage": "validation",
    "id_alignment": {
        "direct_id_coverage": {"matched": direct_cov, "total_gold": len(gold_ids)},
        "mapped_eval_id_coverage": {"matched": mapped_cov, "total_gold": len(gold_ids)},
    },
    "country_normalization": {
        "applied": bool(country_eval_columns),
        "format": country_format,
        "columns": sorted(country_eval_columns),
    },
    "list_normalization": {
        "applied": bool(list_eval_columns),
        "columns": sorted(list_eval_columns),
    },
}, indent=2))

strategy = DataFusionStrategy("evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name


shared_columns = set(fused_eval.columns) & set(fusion_eval_set.columns)
candidate_columns = []
for col in sorted(shared_columns):
    if col in {"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"}:
        continue
    if col.startswith("_"):
        continue
    if not _meaningful_in_gold(fusion_eval_set, col):
        continue
    candidate_columns.append(col)

for col in candidate_columns:
    col_l = col.lower()
    if col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif any(tok in col_l for tok in ["name", "city", "industry", "sector", "continent", "street", "keypeople"]):
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.75 if col_l in {"city"} else 0.85)
    elif "country" in col_l:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.65)
    elif any(tok in col_l for tok in ["founded", "date", "year"]):
        register_eval(col, year_only_match, "year_only_match")
    elif any(tok in col_l for tok in ["revenue", "sales", "profits", "assets", "market value", "market_value", "rank"]):
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.01)
    elif any(tok in col_l for tok in ["latitude", "longitude", "lat", "lon"]):
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.001)
    elif any(tok in col_l for tok in ["is_", "has_", "active", "enabled", "flag", "bool"]):
        register_eval(col, boolean_match, "boolean_match")
    elif col_l == "id" or col_l.endswith("_id") or col_l.endswith("_code") or col_l.endswith("code"):
        register_eval(col, exact_match, "exact_match")
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)

compact_summary = " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items()))
print(f"[EVAL_FUNCS] {compact_summary}")

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

print(json.dumps({
    "stage": "validation",
    "metrics": evaluation_results,
}, indent=2, default=str))

evaluation_output = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
with open(evaluation_output, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4, default=str)