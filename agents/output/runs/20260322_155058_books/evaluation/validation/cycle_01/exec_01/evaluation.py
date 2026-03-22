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
OUTPUT_DIR = "output/runs/20260322_155058_books"
FUSION_PATH = os.path.join(OUTPUT_DIR, "data_fusion", "fusion_data.csv")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_OUTPUT_PATH = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
EVAL_SET_PATH = "input/datasets/books/testsets/validation_set.csv"

os.makedirs(EVAL_DIR, exist_ok=True)


# ================================
# 1. HELPERS
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


def _normalize_text(value):
    if _safe_is_missing(value):
        return value
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_text_lower(value):
    if _safe_is_missing(value):
        return value
    return _normalize_text(value).lower()


def _split_text_list(value):
    if _safe_is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        if text.startswith(("[", "(", "{")):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    if isinstance(parsed, (list, tuple, set)):
                        items = list(parsed)
                        break
                except Exception:
                    pass
            else:
                items = re.split(r"\s*[,;|]\s*", text)
        else:
            items = re.split(r"\s*[,;|]\s*", text)

    cleaned = []
    for item in items:
        if _safe_is_missing(item):
            continue
        item_text = _normalize_text_lower(item)
        if item_text:
            cleaned.append(item_text)
    return cleaned


def _normalize_genres_to_list(value):
    return _split_text_list(value)


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


def _meaningful_common_columns(fused_df, gold_df, id_cols=None):
    id_cols = set(id_cols or [])
    common = []
    for col in sorted(set(fused_df.columns) & set(gold_df.columns)):
        if col in id_cols:
            continue
        if col.startswith("_"):
            continue
        if col.endswith("_provenance"):
            continue
        if gold_df[col].notna().sum() == 0:
            continue
        common.append(col)
    return common


# ================================
# 2. LOAD DATA
# ================================
fused = pd.read_csv(FUSION_PATH)
gold = load_csv(EVAL_SET_PATH, name="validation_set")

fused_eval = fused.copy()
gold_eval = gold.copy()

if "eval_id" not in fused_eval.columns:
    raise ValueError("Fused output must contain 'eval_id' for evaluation alignment.")

print(json.dumps({
    "stage": "validation",
    "fused_path": FUSION_PATH,
    "gold_path": EVAL_SET_PATH,
    "fused_rows": int(len(fused_eval)),
    "gold_rows": int(len(gold_eval)),
    "fused_columns": list(fused_eval.columns),
    "gold_columns": list(gold_eval.columns),
}, indent=2))


# ================================
# 3. COLUMN FILTERING
#    Only direct 1:1 columns present meaningfully in both datasets
# ================================
candidate_columns = _meaningful_common_columns(
    fused_eval,
    gold_eval,
    id_cols={"id", "_id", "eval_id"}
)

# Explicitly avoid evaluating provenance/meta/derived-only columns
blocked_columns = {
    "id",
    "_id",
    "eval_id",
    "_fusion_sources",
    "_fusion_source_datasets",
    "_fusion_confidence",
    "_fusion_metadata",
}
candidate_columns = [c for c in candidate_columns if c not in blocked_columns]

# For this books validation set, direct comparable columns should be:
# isbn_clean, title, author, language, genres, publisher, publish_year, page_count
print("[EVALUATION_COLUMNS]", ", ".join(candidate_columns))


# ================================
# 4. SHARED NORMALIZATION TO VALIDATION REPRESENTATION
# ================================
# Text normalization
text_columns = [
    c for c in candidate_columns
    if c not in {"isbn_clean", "publish_year", "page_count", "genres"}
]

for col in text_columns:
    fused_eval[col] = fused_eval[col].apply(_normalize_text)
    gold_eval[col] = gold_eval[col].apply(_normalize_text)

# Genres as canonical textual lists for order-independent comparison
if "genres" in candidate_columns:
    fused_eval["genres"] = fused_eval["genres"].apply(_normalize_genres_to_list)
    gold_eval["genres"] = gold_eval["genres"].apply(_normalize_genres_to_list)

# Country-like normalization if present
country_eval_columns = [
    c for c in candidate_columns
    if "country" in str(c).lower()
]
if country_eval_columns:
    country_format = _infer_country_output_format(gold_eval, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        gold_eval[col] = gold_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
    print(f"[COUNTRY NORMALIZATION] format={country_format}, columns={country_eval_columns}")

# Numeric coercion to prevent float/int mismatches
numeric_columns = [c for c in candidate_columns if c in {"isbn_clean", "publish_year", "page_count"}]
fused_eval = _coerce_numeric_consistently(fused_eval, numeric_columns)
gold_eval = _coerce_numeric_consistently(gold_eval, numeric_columns)


# ================================
# 5. ID ALIGNMENT DIAGNOSTICS
# ================================
gold_ids = set(gold_eval["id"].dropna().astype(str).tolist()) if "id" in gold_eval.columns else set()
eval_ids = set(fused_eval["eval_id"].dropna().astype(str).tolist()) if "eval_id" in fused_eval.columns else set()
mapped_cov = len(eval_ids & gold_ids)

print(json.dumps({
    "id_alignment": {
        "gold_ids": len(gold_ids),
        "fused_eval_ids": len(eval_ids),
        "mapped_eval_id_coverage": mapped_cov,
        "mapped_eval_id_coverage_ratio": round(mapped_cov / max(len(gold_ids), 1), 6),
    }
}, indent=2))


# ================================
# 6. REGISTER EVALUATION FUNCTIONS
# ================================
strategy = DataFusionStrategy("books_validation_evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

# Canonical identifiers / codes
if "isbn_clean" in candidate_columns:
    register_eval("isbn_clean", exact_match, "exact_match")

# Free text
if "title" in candidate_columns:
    register_eval("title", tokenized_match, "tokenized_match", threshold=0.85)
if "author" in candidate_columns:
    register_eval("author", tokenized_match, "tokenized_match", threshold=0.85)
if "language" in candidate_columns:
    register_eval("language", tokenized_match, "tokenized_match", threshold=0.90)
if "publisher" in candidate_columns:
    register_eval("publisher", tokenized_match, "tokenized_match", threshold=0.80)

# Textual list
if "genres" in candidate_columns:
    register_eval("genres", set_equality_match, "set_equality_match")

# Numeric columns, coerced consistently first
if "publish_year" in candidate_columns:
    register_eval("publish_year", numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.0)
if "page_count" in candidate_columns:
    register_eval("page_count", numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.0)

compact_summary = " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items()))
print(f"[EVAL_FUNCTIONS] {compact_summary}")


# ================================
# 7. RUN EVALUATION
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
    gold_df=gold_eval,
    gold_id_column="id",
)

print(json.dumps({
    "evaluation_stage": "validation",
    "evaluation_metrics": evaluation_results
}, indent=2, default=str))


# ================================
# 8. SAVE RESULTS
# ================================
with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4, default=str)

print(f"[SAVED] {EVAL_OUTPUT_PATH}")