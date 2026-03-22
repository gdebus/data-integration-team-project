import os
import sys
import json
import ast
import re
from pathlib import Path

import pandas as pd

from PyDI.io import load_csv
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEvaluator,
    tokenized_match,
    year_only_match,
    numeric_tolerance_match,
    exact_match,
    boolean_match,
)
from PyDI.normalization import normalize_country

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
OUTPUT_DIR = "output/runs/20260322_155058_books"
FUSION_PATH = os.path.join(OUTPUT_DIR, "data_fusion", "fusion_data.csv")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_OUTPUT = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
DEBUG_OUTPUT = os.path.join(EVAL_DIR, "debug_fusion_eval.jsonl")
EVAL_SET_PATH = "input/datasets/books/testsets/validation_set.csv"

os.makedirs(EVAL_DIR, exist_ok=True)


# ================================
# 1. HELPERS
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


def _parse_listish(value):
    if _safe_is_missing(value):
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
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _normalize_text_list(value, title_case=False):
    items = _parse_listish(value)
    out = []
    for item in items:
        item = re.sub(r"\s+", " ", str(item)).strip()
        if not item:
            continue
        out.append(item.title() if title_case else item.lower())
    return out


def _normalize_genres_to_eval_representation(value):
    items = _normalize_text_list(value, title_case=True)
    return ", ".join(items) if items else value


def _normalize_isbn_like(value):
    if _safe_is_missing(value):
        return value
    text = str(value).strip()
    if not text:
        return text
    text = re.sub(r"[^0-9Xx]", "", text)
    if not text:
        return text
    if re.fullmatch(r"\d+(\.0+)?", str(value).strip()):
        try:
            text = str(int(float(str(value).strip())))
        except Exception:
            pass
    if re.fullmatch(r"\d{10}", text):
        text = text.lstrip("0") or "0"
    return text


def _coerce_numeric_series(df, cols):
    for col in cols:
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


# ================================
# 2. LOAD DATA
# ================================
fused = pd.read_csv(FUSION_PATH)
gold = load_csv(EVAL_SET_PATH, name="books_validation_set")

# Use eval_id if present, as required
if "eval_id" not in fused.columns:
    raise ValueError("Fused output must contain 'eval_id' for validation evaluation.")

fused_eval = fused.copy()
gold_eval = gold.copy()

# Keep only direct 1:1 columns and exclude provenance/derived fields
gold_candidate_cols = [
    c for c in gold_eval.columns
    if not c.endswith("_provenance")
]
shared_cols = set(fused_eval.columns) & set(gold_candidate_cols)

exclude_cols = {
    "_id", "_fusion_sources", "_fusion_source_datasets", "_fusion_confidence",
    "_fusion_metadata", "_eval_cluster_id", "id", "eval_id"
}
shared_eval_cols = []
for col in sorted(shared_cols):
    if col in exclude_cols:
        continue
    if gold_eval[col].isna().all():
        continue
    shared_eval_cols.append(col)

# ================================
# 3. SHARED NORMALIZATION TO VALIDATION REPRESENTATION
# ================================
# Numeric-like columns
numeric_like_cols = [
    c for c in shared_eval_cols
    if any(tok in c.lower() for tok in ["year", "count", "rating", "score", "price", "page"])
]
fused_eval = _coerce_numeric_series(fused_eval, numeric_like_cols)
gold_eval = _coerce_numeric_series(gold_eval, numeric_like_cols)

# ISBN normalization to validation representation
if "isbn_clean" in shared_eval_cols:
    fused_eval["isbn_clean"] = fused_eval["isbn_clean"].apply(_normalize_isbn_like)
    gold_eval["isbn_clean"] = gold_eval["isbn_clean"].apply(_normalize_isbn_like)

# Genres/textual-list representation
if "genres" in shared_eval_cols:
    fused_eval["genres"] = fused_eval["genres"].apply(_normalize_genres_to_eval_representation)
    gold_eval["genres"] = gold_eval["genres"].apply(_normalize_genres_to_eval_representation)

# Generic list-like normalization if any truly shared list-like columns exist
list_eval_columns = detect_list_like_columns(
    [fused_eval[["eval_id"] + shared_eval_cols].copy(), gold_eval[["id"] + shared_eval_cols].copy()],
    exclude_columns={"id", "eval_id"},
)
list_eval_columns = [c for c in list_eval_columns if c in shared_eval_cols and c != "genres"]
if list_eval_columns:
    fused_eval, gold_eval = normalize_list_like_columns([fused_eval, gold_eval], list_eval_columns)
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_text_list(x, title_case=False))
        if col in gold_eval.columns:
            gold_eval[col] = gold_eval[col].apply(lambda x: _normalize_text_list(x, title_case=False))

# Country normalization if ever present
country_eval_columns = [c for c in shared_eval_cols if "country" in c.lower()]
if country_eval_columns:
    country_format = _infer_country_output_format(gold_eval, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        gold_eval[col] = gold_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
else:
    country_format = None

# ================================
# 4. EVALUATION STRATEGY
# ================================
strategy = DataFusionStrategy("books_validation_evaluation")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{column}:{fn_name}({args})" if args else f"{column}:{fn_name}"

# Attribute-specific registrations based on actual validation schema
if "title" in shared_eval_cols:
    register_eval("title", tokenized_match, "tokenized_match", threshold=0.85)
if "author" in shared_eval_cols:
    register_eval("author", tokenized_match, "tokenized_match", threshold=0.85)
if "language" in shared_eval_cols:
    register_eval("language", tokenized_match, "tokenized_match", threshold=0.90)
if "genres" in shared_eval_cols:
    register_eval("genres", tokenized_match, "tokenized_match", threshold=0.75)
if "publisher" in shared_eval_cols:
    register_eval("publisher", tokenized_match, "tokenized_match", threshold=0.80)
if "publish_year" in shared_eval_cols:
    register_eval("publish_year", numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.0)
if "page_count" in shared_eval_cols:
    register_eval("page_count", numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.0)
if "isbn_clean" in shared_eval_cols:
    register_eval("isbn_clean", exact_match, "exact_match")

# Sensible fallback for any remaining meaningful shared columns
registered = set(eval_funcs_summary.keys())
for col in shared_eval_cols:
    if col in registered:
        continue
    col_lower = col.lower()
    if any(tok in col_lower for tok in ["date"]):
        register_eval(col, year_only_match, "year_only_match")
    elif any(tok in col_lower for tok in ["year", "count", "rating", "score", "price", "page"]):
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.01)
    elif any(tok in col_lower for tok in ["is_", "has_", "flag", "bool"]):
        register_eval(col, boolean_match, "boolean_match")
    elif "id" == col_lower or col_lower.endswith("_id") or "isbn" in col_lower:
        register_eval(col, exact_match, "exact_match")
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.80)

# ================================
# 5. RUN EVALUATION
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

fused_eval = _strip_unevaluable_subcols(fused_eval, gold_eval)

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

_coerce_numeric_cols(fused_eval, gold_eval)

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

fused_eval = _normalize_list_separators(fused_eval, gold_eval)
evaluation_results = evaluator.evaluate(
    fused_df=fused_eval,
    fused_id_column="eval_id",
    expected_df=gold_eval,
    expected_id_column="id",
)

# ================================
# 6. PRINT STRUCTURED DIAGNOSTICS
# ================================
gold_ids = set(gold_eval["id"].dropna().astype(str))
fused_eval_ids = set(fused_eval["eval_id"].dropna().astype(str))
mapped_cov = len(fused_eval_ids & gold_ids)

diagnostics = {
    "stage": "validation",
    "fused_path": FUSION_PATH,
    "evaluation_set_path": EVAL_SET_PATH,
    "evaluation_output_path": EVAL_OUTPUT,
    "rows": {
        "fused": int(len(fused_eval)),
        "gold": int(len(gold_eval)),
    },
    "id_alignment": {
        "fused_id_column": "eval_id",
        "gold_id_column": "id",
        "mapped_eval_id_coverage": f"{mapped_cov}/{len(gold_ids)}",
    },
    "shared_evaluated_columns": shared_eval_cols,
    "list_eval_columns": list_eval_columns,
    "country_eval_columns": country_eval_columns,
    "country_normalization_format": country_format,
    "evaluation_functions_one_line": " | ".join(eval_funcs_summary[c] for c in sorted(eval_funcs_summary)),
    "metrics": evaluation_results,
}

print(json.dumps(diagnostics, indent=2, default=str))
print("[EVAL_FUNCS]", " | ".join(eval_funcs_summary[c] for c in sorted(eval_funcs_summary)))

# ================================
# 7. SAVE RESULTS
# ================================
with open(EVAL_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4, default=str)