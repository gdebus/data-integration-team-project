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


# ================================
# 0. OUTPUT / INPUT PATHS
# ================================
OUTPUT_DIR = "output/runs/20260322_162759_restaurant"
FUSION_PATH = os.path.join(OUTPUT_DIR, "data_fusion", "fusion_data.csv")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_OUTPUT_PATH = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
EVAL_DEBUG_PATH = os.path.join(EVAL_DIR, "debug_fusion_eval.jsonl")
GOLD_PATH = "input/datasets/restaurant/testsets/Restaurant_Fusion_Validation_Set.csv"

os.makedirs(EVAL_DIR, exist_ok=True)


# ================================
# HELPERS
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


def _parse_source_ids(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
    return []


def _detect_gold_prefixes(gold_ids):
    prefixes = [g.split("-", 1)[0] + "-" for g in gold_ids if "-" in g]
    prefixes += [g.split("_", 1)[0] + "_" for g in gold_ids if "_" in g]
    return [p for p, _ in Counter(prefixes).most_common()]


def _normalize_list_text_case(value):
    if _safe_is_missing(value):
        return value
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip().lower() for v in value if str(v).strip() and str(v).strip().lower() != "nan"]
    text = str(value).strip()
    return text.lower() if text else text


def _coerce_numeric_series(df, columns):
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].dropna()
        if len(non_null) > 0:
            try:
                if non_null.apply(float.is_integer).all():
                    df[col] = df[col].astype("Int64")
            except Exception:
                pass
    return df


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


def build_eval_view(fused_df, gold_df):
    gold_ids = set(gold_df["id"].dropna().astype(str).tolist()) if "id" in gold_df.columns else set()
    gold_prefixes = _detect_gold_prefixes(gold_ids)
    expanded_rows = []

    id_like_columns = [c for c in fused_df.columns if c in {"eval_id", "id", "_id", "yelp_id", "uber_eats_id", "kaggle380k_id"}]

    for _, row in fused_df.iterrows():
        row_dict = row.to_dict()
        cluster_id = str(row_dict.get("_id", "")).strip()
        source_ids = _parse_source_ids(row_dict.get("_fusion_sources"))

        candidates = []

        if "eval_id" in row_dict and not _safe_is_missing(row_dict.get("eval_id")):
            candidates.append(str(row_dict.get("eval_id")).strip())

        for col in id_like_columns:
            val = row_dict.get(col)
            if not _safe_is_missing(val):
                candidates.append(str(val).strip())

        candidates.extend(source_ids)

        filtered = []
        for candidate in candidates:
            if not candidate:
                continue
            if candidate in gold_ids:
                filtered.append(candidate)
                continue
            if any(candidate.startswith(pref) for pref in gold_prefixes):
                filtered.append(candidate)

        if not filtered and cluster_id:
            filtered = [cluster_id]

        seen = set()
        for eval_id in filtered:
            if not eval_id or eval_id in seen:
                continue
            seen.add(eval_id)
            enriched = dict(row_dict)
            enriched["eval_id"] = eval_id
            enriched["_eval_cluster_id"] = cluster_id
            expanded_rows.append(enriched)

    if not expanded_rows:
        fallback = fused_df.copy()
        fallback["eval_id"] = fallback["eval_id"].astype(str) if "eval_id" in fallback.columns else fallback["_id"].astype(str)
        fallback["_eval_cluster_id"] = fallback["_id"].astype(str)
        return fallback

    fused_eval = pd.DataFrame(expanded_rows)
    if "_fusion_confidence" in fused_eval.columns:
        fused_eval["_fusion_confidence"] = pd.to_numeric(fused_eval["_fusion_confidence"], errors="coerce").fillna(0.0)
        fused_eval = fused_eval.sort_values("_fusion_confidence", ascending=False)
    fused_eval = fused_eval.drop_duplicates(subset=["eval_id"], keep="first")
    return fused_eval


def _meaningful_shared_columns(fused_df, gold_df):
    shared = set(fused_df.columns) & set(gold_df.columns)
    excluded = {"_id", "_fusion_sources", "_fusion_source_datasets", "_fusion_confidence", "_fusion_metadata", "_eval_cluster_id", "eval_id"}
    cols = []
    for col in sorted(shared):
        if col in excluded:
            continue
        if "_" in col:
            parent = col.split("_", 1)[0]
            if parent in gold_df.columns and col not in gold_df.columns:
                continue
        gold_non_null = gold_df[col].notna().sum() if col in gold_df.columns else 0
        fused_non_null = fused_df[col].notna().sum() if col in fused_df.columns else 0
        if gold_non_null == 0 or fused_non_null == 0:
            continue
        cols.append(col)
    return cols


# ================================
# 1. LOAD DATA
# ================================
fused = pd.read_csv(FUSION_PATH)
gold = load_csv(GOLD_PATH, name="fusion_validation_set")

if "id" not in gold.columns and "_id" in gold.columns:
    gold["id"] = gold["_id"].astype(str)

fused_eval = build_eval_view(fused, gold)

# Always use eval_id when available
fused_id_column = "eval_id" if "eval_id" in fused_eval.columns else "_id"
gold_id_column = "id"


# ================================
# 2. HARMONIZE LIST-LIKE COLUMNS
# ================================
shared_eval_columns = _meaningful_shared_columns(fused_eval, gold)

list_eval_columns = detect_list_like_columns(
    [fused_eval[shared_eval_columns + [fused_id_column]] if fused_id_column not in shared_eval_columns else fused_eval[shared_eval_columns],
     gold[shared_eval_columns + [gold_id_column]] if gold_id_column not in shared_eval_columns else gold[shared_eval_columns]],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if c in shared_eval_columns and not c.startswith("_")]

if list_eval_columns:
    fused_eval, gold = normalize_list_like_columns([fused_eval, gold], list_eval_columns)
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
        if col in gold.columns:
            gold[col] = gold[col].apply(_normalize_list_text_case)


# ================================
# 3. COUNTRY NORMALIZATION
# ================================
country_eval_columns = [c for c in shared_eval_columns if "country" in c.lower()]
if country_eval_columns:
    country_format = _infer_country_output_format(gold, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        gold[col] = gold[col].apply(lambda x: _normalize_country_safe(x, country_format))
else:
    country_format = None


# ================================
# 4. NUMERIC TYPE COERCION
# ================================
numeric_candidates = [
    c for c in shared_eval_columns
    if any(tok in c.lower() for tok in ["latitude", "longitude", "rating", "count", "postal_code", "house_number", "phone", "year", "price", "score"])
]
fused_eval = _coerce_numeric_series(fused_eval, numeric_candidates)
gold = _coerce_numeric_series(gold, numeric_candidates)


# ================================
# 5. REGISTER EVALUATION FUNCTIONS
# ================================
strategy = DataFusionStrategy("restaurant_evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

shared_eval_columns = _meaningful_shared_columns(fused_eval, gold)

for col in shared_eval_columns:
    col_l = col.lower()

    if col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif col in {"id", "yelp_id", "uber_eats_id", "kaggle380k_id"}:
        register_eval(col, exact_match, "exact_match")
    elif col_l in {"country"} or "country" in col_l:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.95)
    elif col_l in {"name", "name_norm", "city", "state", "street", "address_line1", "address_line2", "website", "map_url", "source"}:
        threshold = 0.85
        if col_l in {"address_line1", "address_line2", "street", "website", "map_url"}:
            threshold = 0.75
        if col_l == "source":
            threshold = 0.95
        register_eval(col, tokenized_match, "tokenized_match", threshold=threshold)
    elif "date" in col_l or "founded" in col_l or "year" in col_l:
        register_eval(col, year_only_match, "year_only_match")
    elif col_l in {"latitude", "longitude"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.001)
    elif col_l in {"rating"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.15)
    elif col_l in {"rating_count"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=1.0)
    elif col_l in {"postal_code", "house_number", "phone_e164", "phone_raw"}:
        register_eval(col, exact_match, "exact_match")
    else:
        gold_non_null = gold[col].dropna()
        if len(gold_non_null) > 0 and gold_non_null.map(lambda x: isinstance(x, bool)).all():
            register_eval(col, boolean_match, "boolean_match")
        else:
            register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)


# ================================
# 6. DIAGNOSTICS
# ================================
gold_ids = set(gold[gold_id_column].dropna().astype(str).tolist()) if gold_id_column in gold.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval[fused_id_column].astype(str).tolist()) & gold_ids) if fused_id_column in fused_eval.columns else 0

print(json.dumps({
    "stage": "validation",
    "fused_rows_raw": int(len(fused)),
    "fused_rows_eval_view": int(len(fused_eval)),
    "gold_rows": int(len(gold)),
    "gold_id_count": int(len(gold_ids)),
    "id_alignment": {
        "direct__id_coverage": f"{direct_cov}/{len(gold_ids)}",
        "mapped_eval_id_coverage": f"{mapped_cov}/{len(gold_ids)}",
    },
    "list_eval_columns": list_eval_columns,
    "country_eval_columns": country_eval_columns,
    "country_normalization_target": country_format,
    "evaluated_columns": shared_eval_columns,
}, indent=2))

print("[EVAL FUNCS] " + " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items())))


# ================================
# 7. RUN EVALUATION
# ================================
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
    fused_id_column=fused_id_column,
    expected_df=gold,
    expected_id_column=gold_id_column,
)

print(json.dumps(evaluation_results, indent=2, default=str))


# ================================
# 8. WRITE RESULTS
# ================================
with open(EVAL_OUTPUT_PATH, "w") as f:
    json.dump(evaluation_results, f, indent=4, default=str)