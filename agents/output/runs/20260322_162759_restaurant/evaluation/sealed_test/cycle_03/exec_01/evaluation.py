import pandas as pd
import json
import ast
import re
import os
from collections import Counter
from pathlib import Path
import sys

from PyDI.io import load_csv
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


OUTPUT_DIR = "output/runs/20260322_162759_restaurant"
FUSION_PATH = os.path.join(OUTPUT_DIR, "data_fusion", "fusion_data.csv")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_OUTPUT_PATH = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
EVAL_DEBUG_PATH = os.path.join(EVAL_DIR, "debug_fusion_eval.jsonl")
GOLD_PATH = "input/datasets/restaurant/testsets/Restaurant_Fusion_Test_Set.csv"

os.makedirs(EVAL_DIR, exist_ok=True)


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


def _detect_gold_prefixes(gold_ids):
    prefixes = [g.split("-", 1)[0] + "-" for g in gold_ids if "-" in g]
    prefixes += [g.split("_", 1)[0] + "_" for g in gold_ids if "_" in g]
    return [p for p, _ in Counter(prefixes).most_common()]


def build_eval_view(fused_df, gold_df):
    gold_ids = set(gold_df["id"].dropna().astype(str).tolist()) if "id" in gold_df.columns else set()
    gold_prefixes = _detect_gold_prefixes(gold_ids)
    expanded_rows = []

    for _, row in fused_df.iterrows():
        row_dict = row.to_dict()
        cluster_id = str(row_dict.get("_id", ""))
        eval_id_existing = row_dict.get("eval_id", None)

        candidates = []
        if not _safe_is_missing(eval_id_existing):
            candidates.append(str(eval_id_existing))

        source_ids = _parse_source_ids(row_dict.get("_fusion_sources"))
        if source_ids:
            for prefix in gold_prefixes:
                candidates.extend([sid for sid in source_ids if sid.startswith(prefix)])
            candidates.extend([sid for sid in source_ids if sid in gold_ids])

        for id_col in ["id", "kaggle380k_id", "uber_eats_id", "yelp_id"]:
            if id_col in row_dict and not _safe_is_missing(row_dict[id_col]):
                candidates.append(str(row_dict[id_col]))

        if cluster_id:
            candidates.append(cluster_id)

        seen = set()
        ordered_candidates = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                ordered_candidates.append(c)

        matched_candidates = [c for c in ordered_candidates if c in gold_ids]
        if not matched_candidates and ordered_candidates:
            matched_candidates = [ordered_candidates[0]]

        for eval_id in matched_candidates:
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


def _infer_country_output_format(gold_df, country_columns):
    values = []
    for col in country_columns:
        if col in gold_df.columns:
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


def _normalize_text_scalar(value):
    if _safe_is_missing(value):
        return value
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def _canonicalize_list_value(value):
    if _safe_is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        text = str(value).strip()
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                break
            except Exception:
                continue
        if isinstance(parsed, (list, tuple, set)):
            raw_items = list(parsed)
        elif text:
            raw_items = [text]
        else:
            raw_items = []

    cleaned = []
    for item in raw_items:
        if _safe_is_missing(item):
            continue
        item_text = str(item).strip().strip('"').strip("'")
        if not item_text or item_text.lower() in {"nan", "none", "null"}:
            continue

        nested = None
        for parser in (json.loads, ast.literal_eval):
            try:
                nested = parser(item_text)
                break
            except Exception:
                continue
        if isinstance(nested, (list, tuple, set)):
            for sub in nested:
                if _safe_is_missing(sub):
                    continue
                sub_text = re.sub(r"\s+", " ", str(sub).strip().lower())
                if sub_text and sub_text not in {"nan", "none", "null"}:
                    cleaned.append(sub_text)
        else:
            item_text = re.sub(r"\s+", " ", item_text.lower())
            if item_text and item_text not in {"nan", "none", "null"}:
                cleaned.append(item_text)

    return sorted(set(cleaned))


def _coerce_numeric_columns(df, numeric_cols):
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            non_null = df[col].dropna()
            if len(non_null) > 0:
                try:
                    if non_null.apply(float.is_integer).all():
                        df[col] = df[col].astype("Int64")
                except Exception:
                    pass
    return df


def _meaningful_gold_columns(df):
    cols = []
    for col in df.columns:
        series = df[col]
        if series.dropna().empty:
            continue
        cols.append(col)
    return set(cols)


fused = pd.read_csv(FUSION_PATH)
gold = load_csv(GOLD_PATH, name="restaurant_fusion_test_set")

if "_id" in gold.columns and "id" not in gold.columns:
    gold["id"] = gold["_id"].astype(str)
elif "id" in gold.columns:
    gold["id"] = gold["id"].astype(str)

fused_eval = build_eval_view(fused, gold)

shared_columns = set(fused_eval.columns) & set(gold.columns)
shared_columns = {c for c in shared_columns if not c.startswith("_")}
gold_meaningful = _meaningful_gold_columns(gold)
shared_columns = {c for c in shared_columns if c in gold_meaningful}

parent_columns = set(shared_columns)
drop_derived = set()
for col in list(fused_eval.columns):
    if col.startswith("_"):
        continue
    for parent in parent_columns:
        if col != parent and col.startswith(parent + "_") and parent in gold.columns:
            drop_derived.add(col)
shared_columns -= drop_derived

list_eval_columns = detect_list_like_columns(
    [fused_eval[list(shared_columns)].copy(), gold[list(shared_columns)].copy()],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if c in shared_columns and not c.startswith("_")]

if list_eval_columns:
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_canonicalize_list_value)
        if col in gold.columns:
            gold[col] = gold[col].apply(_canonicalize_list_value)

country_eval_columns = [c for c in shared_columns if "country" in c.lower()]
if country_eval_columns:
    country_format = _infer_country_output_format(gold, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        gold[col] = gold[col].apply(lambda x: _normalize_country_safe(x, country_format))
else:
    country_format = None

text_like_columns = [
    c for c in shared_columns
    if c not in set(list_eval_columns)
    and c not in {"id", "eval_id", "_id"}
]
for col in text_like_columns:
    if col in fused_eval.columns and fused_eval[col].dtype == object:
        fused_eval[col] = fused_eval[col].apply(_normalize_text_scalar)
    if col in gold.columns and gold[col].dtype == object:
        gold[col] = gold[col].apply(_normalize_text_scalar)

numeric_cols = []
for col in shared_columns:
    if col in {"latitude", "longitude", "rating", "rating_count", "house_number", "postal_code"}:
        numeric_cols.append(col)
    elif any(tok in col.lower() for tok in ["year", "count", "rating", "score", "price", "lat", "lon", "number"]):
        numeric_cols.append(col)

fused_eval = _coerce_numeric_columns(fused_eval, numeric_cols)
gold = _coerce_numeric_columns(gold, numeric_cols)

strategy = DataFusionStrategy("restaurant_evaluation_strategy")
eval_funcs_summary = {}


def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name


if "name" in shared_columns:
    register_eval("name", tokenized_match, "tokenized_match", threshold=0.85)
if "name_norm" in shared_columns:
    register_eval("name_norm", tokenized_match, "tokenized_match", threshold=0.90)
if "address_line1" in shared_columns:
    register_eval("address_line1", tokenized_match, "tokenized_match", threshold=0.80)
if "address_line2" in shared_columns:
    register_eval("address_line2", tokenized_match, "tokenized_match", threshold=0.80)
if "street" in shared_columns:
    register_eval("street", tokenized_match, "tokenized_match", threshold=0.80)
if "city" in shared_columns:
    register_eval("city", tokenized_match, "tokenized_match", threshold=0.85)
if "state" in shared_columns:
    register_eval("state", exact_match, "exact_match")
if "country" in shared_columns:
    register_eval("country", tokenized_match, "tokenized_match", threshold=0.90)
if "website" in shared_columns:
    register_eval("website", tokenized_match, "tokenized_match", threshold=0.75)
if "map_url" in shared_columns:
    register_eval("map_url", tokenized_match, "tokenized_match", threshold=0.70)
if "phone_raw" in shared_columns:
    register_eval("phone_raw", exact_match, "exact_match")
if "phone_e164" in shared_columns:
    register_eval("phone_e164", exact_match, "exact_match")
if "postal_code" in shared_columns:
    register_eval("postal_code", exact_match, "exact_match")
if "house_number" in shared_columns:
    register_eval("house_number", exact_match, "exact_match")
if "latitude" in shared_columns:
    register_eval("latitude", numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.001)
if "longitude" in shared_columns:
    register_eval("longitude", numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.001)
if "rating" in shared_columns:
    register_eval("rating", numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.05)
if "rating_count" in shared_columns:
    register_eval("rating_count", numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.01)
if "founded" in shared_columns:
    register_eval("founded", year_only_match, "year_only_match")

for col in list_eval_columns:
    if col not in eval_funcs_summary:
        register_eval(col, set_equality_match, "set_equality_match")

for col in sorted(shared_columns):
    if col in eval_funcs_summary or col in {"id", "eval_id"}:
        continue
    gold_non_null = gold[col].dropna() if col in gold.columns else pd.Series(dtype=object)
    if gold_non_null.empty:
        continue
    sample = gold_non_null.iloc[0]
    if isinstance(sample, bool):
        register_eval(col, boolean_match, "boolean_match")
    elif pd.api.types.is_numeric_dtype(gold[col]) and col not in eval_funcs_summary:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.01)
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)

gold_ids = set(gold["id"].dropna().astype(str).tolist()) if "id" in gold.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0

print(json.dumps({
    "evaluation_stage": "sealed_test",
    "fusion_path": FUSION_PATH,
    "gold_path": GOLD_PATH,
    "output_path": EVAL_OUTPUT_PATH,
    "id_alignment": {
        "gold_rows": len(gold_ids),
        "direct__id_coverage": direct_cov,
        "mapped_eval_id_coverage": mapped_cov,
    },
    "country_normalization": {
        "columns": country_eval_columns,
        "format": country_format,
    },
    "list_columns": list_eval_columns,
    "dropped_derived_columns": sorted(drop_derived),
    "evaluated_columns": sorted(eval_funcs_summary.keys()),
}, indent=2))

print("EVAL_FUNCTIONS:", "; ".join(f"{k}={v}" for k, v in sorted(eval_funcs_summary.items())))

evaluator = DataFusionEvaluator(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=EVAL_DEBUG_PATH,
    fusion_debug_logs=os.path.join(OUTPUT_DIR, "data_fusion", "debug_fusion_data.jsonl"),
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
                    if numeric.dropna().apply(float.is_integer).all():
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

print(json.dumps(evaluation_results, indent=2, default=str))

with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4, default=str)