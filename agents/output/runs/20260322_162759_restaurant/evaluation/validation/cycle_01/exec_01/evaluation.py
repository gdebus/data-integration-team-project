import pandas as pd
import json
import ast
import re
import os
from pathlib import Path
import sys

from PyDI.io import load_csv
from PyDI.normalization import normalize_country
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEvaluator,
    tokenized_match,
    set_equality_match,
    numeric_tolerance_match,
    exact_match,
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
OUTPUT_DIR = "output/runs/20260322_162759_restaurant"
FUSION_PATH = os.path.join(OUTPUT_DIR, "data_fusion", "fusion_data.csv")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_OUTPUT_PATH = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
DEBUG_PATH = os.path.join(EVAL_DIR, "debug_fusion_eval.jsonl")
GOLD_PATH = "input/datasets/restaurant/testsets/Restaurant_Fusion_Validation_Set.csv"

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
        return False


def _parse_list_like(value):
    if _safe_is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value]
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        except Exception:
            pass
    return [text]


def _flatten_nested_list(values):
    out = []
    for v in values:
        if isinstance(v, (list, tuple, set)):
            out.extend(_flatten_nested_list(list(v)))
        else:
            out.append(v)
    return out


def _canonicalize_text_token(x):
    if _safe_is_missing(x):
        return None
    s = str(x).strip().lower()
    if not s or s in {"nan", "none", "null", "n/a", "na", "[]", "{}"}:
        return None
    s = re.sub(r"\s+", " ", s)
    return s


def _canonicalize_list_value(value):
    parsed = _parse_list_like(value)
    flattened = _flatten_nested_list(parsed)

    canonical = []
    for item in flattened:
        if isinstance(item, str):
            inner = item.strip()
            reparsed = None
            for parser in (json.loads, ast.literal_eval):
                try:
                    candidate = parser(inner)
                    if isinstance(candidate, (list, tuple, set)):
                        reparsed = candidate
                        break
                except Exception:
                    pass
            if reparsed is not None:
                for sub in _flatten_nested_list(list(reparsed)):
                    token = _canonicalize_text_token(sub)
                    if token is not None:
                        canonical.append(token)
                continue
        token = _canonicalize_text_token(item)
        if token is not None:
            canonical.append(token)

    return sorted(set(canonical))


def _normalize_country_output_format(gold_df, country_columns):
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

    if alpha2 / total >= 0.7:
        return "alpha_2"
    if alpha3 / total >= 0.7:
        return "alpha_3"
    if numeric / total >= 0.7:
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


def _coerce_numeric_consistently(df, columns):
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


def _meaningful_in_gold(df, col):
    if col not in df.columns:
        return False
    non_null = df[col].dropna()
    if len(non_null) == 0:
        return False
    if non_null.astype(str).str.strip().eq("").all():
        return False
    return True


# ================================
# 2. LOAD DATA
# ================================
fused = pd.read_csv(FUSION_PATH)
gold = load_csv(GOLD_PATH, name="restaurant_validation")

# If eval_id is present, use it directly as required.
fused_eval = fused.copy()

# Standardize IDs as strings
if "eval_id" in fused_eval.columns:
    fused_eval["eval_id"] = fused_eval["eval_id"].astype(str)
if "id" in gold.columns:
    gold["id"] = gold["id"].astype(str)

# ================================
# 3. DROP DERIVED / NON-1:1 COLUMNS FROM EVALUATION SCOPE
# ================================
common_columns = set(fused_eval.columns) & set(gold.columns)
common_columns = {c for c in common_columns if not c.startswith("_")}

# Keep only columns meaningful in gold
common_columns = {c for c in common_columns if _meaningful_in_gold(gold, c)}

# ID columns are handled separately, not as evaluated attributes
common_columns.discard("id")
common_columns.discard("eval_id")

# ================================
# 4. NORMALIZE LIST-LIKE COLUMNS TO SHARED REPRESENTATION
# ================================
list_eval_columns = detect_list_like_columns(
    [fused_eval[list(common_columns | {"eval_id"})] if "eval_id" in fused_eval.columns else fused_eval,
     gold[list(common_columns | {"id"})] if "id" in gold.columns else gold],
    exclude_columns={"id", "eval_id", "_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if c in common_columns and not c.startswith("_")]

if list_eval_columns:
    fused_eval, gold = normalize_list_like_columns([fused_eval, gold], list_eval_columns)
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_canonicalize_list_value)
        if col in gold.columns:
            gold[col] = gold[col].apply(_canonicalize_list_value)

# ================================
# 5. NORMALIZE COUNTRY COLUMNS TO GOLD REPRESENTATION
# ================================
country_eval_columns = [c for c in common_columns if "country" in c.lower()]
country_format = None
if country_eval_columns:
    country_format = _normalize_country_output_format(gold, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        gold[col] = gold[col].apply(lambda x: _normalize_country_safe(x, country_format))

# ================================
# 6. NUMERIC COERCION FOR CONSISTENT COMPARISON
# ================================
numeric_like_columns = [
    c for c in common_columns
    if c in {
        "latitude", "longitude", "rating", "rating_count",
        "postal_code", "house_number", "phone_e164", "phone_raw"
    }
]
fused_eval = _coerce_numeric_consistently(fused_eval, numeric_like_columns)
gold = _coerce_numeric_consistently(gold, numeric_like_columns)

# Cast code-like numerics back to string for exact/token comparisons where appropriate
for col in ["postal_code", "house_number", "phone_e164", "phone_raw"]:
    if col in fused_eval.columns:
        fused_eval[col] = fused_eval[col].astype("string")
    if col in gold.columns:
        gold[col] = gold[col].astype("string")

# Light text normalization for shared string columns
text_columns = [c for c in common_columns if c not in list_eval_columns and c not in numeric_like_columns]
for col in text_columns:
    if col in country_eval_columns:
        continue
    if col in fused_eval.columns:
        fused_eval[col] = fused_eval[col].apply(
            lambda x: re.sub(r"\s+", " ", str(x).strip()) if not _safe_is_missing(x) else x
        )
    if col in gold.columns:
        gold[col] = gold[col].apply(
            lambda x: re.sub(r"\s+", " ", str(x).strip()) if not _safe_is_missing(x) else x
        )

# ================================
# 7. REGISTER EVALUATION FUNCTIONS
# ================================
strategy = DataFusionStrategy("restaurant_validation_evaluation")
eval_func_summary = {}

def register_eval(column, fn, label, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items())
    eval_func_summary[column] = f"{label}({args})" if args else label

for col in sorted(common_columns):
    if col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif col in {"latitude", "longitude"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.002)
    elif col in {"rating"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.25)
    elif col in {"rating_count"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=1.0)
    elif col in {"phone_e164", "phone_raw", "postal_code", "house_number"}:
        register_eval(col, exact_match, "exact_match")
    elif col in {"website", "map_url"}:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)
    elif col in {"name", "name_norm"}:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)
    elif col in {"address_line1", "address_line2", "street", "city", "state"}:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.75)
    elif col in country_eval_columns:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.95)
    elif col in {"source"}:
        register_eval(col, exact_match, "exact_match")
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.80)

# ================================
# 8. RUN EVALUATION
# ================================
evaluator = DataFusionEvaluator(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=DEBUG_PATH,
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

# ================================
# 9. PRINT STRUCTURED METRICS
# ================================
diagnostics = {
    "evaluation_stage": "validation",
    "fused_path": FUSION_PATH,
    "gold_path": GOLD_PATH,
    "fused_rows": int(len(fused_eval)),
    "gold_rows": int(len(gold)),
    "fused_id_column": "eval_id" if "eval_id" in fused_eval.columns else "_id",
    "gold_id_column": "id",
    "evaluated_columns": sorted(common_columns),
    "list_eval_columns": sorted(list_eval_columns),
    "country_eval_columns": sorted(country_eval_columns),
    "country_normalization_format": country_format,
    "evaluation_functions_one_line": " | ".join(f"{k}:{v}" for k, v in sorted(eval_func_summary.items())),
}

print(json.dumps({"diagnostics": diagnostics}, indent=2, default=str))
print(json.dumps({"evaluation_results": evaluation_results}, indent=2, default=str))

# ================================
# 10. WRITE OUTPUT
# ================================
with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "diagnostics": diagnostics,
            "evaluation_results": evaluation_results,
        },
        f,
        indent=2,
        default=str,
    )