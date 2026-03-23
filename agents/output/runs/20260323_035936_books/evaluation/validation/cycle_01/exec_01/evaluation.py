import os
import sys
import json
import ast
import re
from pathlib import Path

import pandas as pd

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
# 0. PATHS
# ================================
OUTPUT_DIR = "output/runs/20260323_035936_books"
FUSION_PATH = "output/runs/20260323_035936_books/data_fusion/fusion_data.csv"
EVAL_SET_PATH = "input/datasets/books/testsets/validation_set.csv"
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
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
        out = pd.isna(value)
        return bool(out) if not hasattr(out, "__len__") else False
    except Exception:
        return False


def _parse_maybe_list(value):
    if _safe_is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text[0] in "[({" and text[-1] in "])}":
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except Exception:
                pass
    if "," in text:
        return [p.strip() for p in text.split(",") if p.strip()]
    if ";" in text:
        return [p.strip() for p in text.split(";") if p.strip()]
    return [text]


def _normalize_text(value):
    if _safe_is_missing(value):
        return value
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def _normalize_list_text(value):
    if _safe_is_missing(value):
        return value
    items = _parse_maybe_list(value)
    items = [re.sub(r"\s+", " ", str(x).strip().lower()) for x in items if str(x).strip()]
    return sorted(set(items))


def _coerce_numeric_columns(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            non_null = df[col].dropna()
            if len(non_null) > 0:
                try:
                    if non_null.apply(lambda x: float(x).is_integer()).all():
                        df[col] = df[col].astype("Int64")
                except Exception:
                    pass
    return df


def _meaningful_shared_columns(fused_df, gold_df):
    shared = []
    for col in sorted(set(fused_df.columns) & set(gold_df.columns)):
        if col in {"id", "_id", "eval_id"}:
            continue
        if col.startswith("_") or col.endswith("_provenance"):
            continue
        if gold_df[col].dropna().empty:
            continue
        if fused_df[col].dropna().empty:
            continue
        shared.append(col)
    return shared


# ================================
# 2. LOAD DATA
# ================================
fused = pd.read_csv(FUSION_PATH)
gold = pd.read_csv(EVAL_SET_PATH)

if "eval_id" not in fused.columns:
    raise ValueError("Fused output must contain 'eval_id' for evaluation alignment.")

fused_eval = fused.copy()
gold_eval = gold.copy()

# ================================
# 3. NORMALIZE REPRESENTATION TO VALIDATION SET STYLE
# ================================
for df in (fused_eval, gold_eval):
    for col in df.columns:
        if col.endswith("_provenance") or col.startswith("_"):
            continue
        if df[col].dtype == "object" or str(df[col].dtype).startswith("string"):
            df[col] = df[col].apply(_normalize_text)

numeric_like_cols = [
    c for c in [
        "isbn_clean", "publish_year", "page_count", "rating", "score",
        "price", "count", "numratings", "year"
    ]
    if c in fused_eval.columns or c in gold_eval.columns
]
fused_eval = _coerce_numeric_columns(fused_eval, numeric_like_cols)
gold_eval = _coerce_numeric_columns(gold_eval, numeric_like_cols)

# Normalize textual list-like fields consistently before evaluation
candidate_list_cols = detect_list_like_columns(
    [fused_eval, gold_eval],
    exclude_columns={"id", "_id", "eval_id", "_fusion_confidence"},
)
candidate_list_cols = [c for c in candidate_list_cols if not c.startswith("_") and not c.endswith("_provenance")]

# Force-include known textual multi-value columns if present
for forced_col in ["genres", "categories", "tags"]:
    if forced_col in fused_eval.columns and forced_col in gold_eval.columns and forced_col not in candidate_list_cols:
        candidate_list_cols.append(forced_col)

candidate_list_cols = sorted(set(candidate_list_cols))

if candidate_list_cols:
    fused_eval, gold_eval = normalize_list_like_columns([fused_eval, gold_eval], candidate_list_cols)
    for col in candidate_list_cols:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text)
        if col in gold_eval.columns:
            gold_eval[col] = gold_eval[col].apply(_normalize_list_text)

# ================================
# 4. SELECT EVALUATION COLUMNS
# ================================
shared_columns = _meaningful_shared_columns(fused_eval, gold_eval)

# Drop structurally nested artifact-like columns if present
nested_like = []
prefix_counts = {}
for c in shared_columns:
    if c.count("_") >= 2:
        prefix = "_".join(c.split("_")[:-1])
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
for c in shared_columns:
    if c.count("_") >= 2:
        prefix = "_".join(c.split("_")[:-1])
        if prefix_counts.get(prefix, 0) >= 2:
            nested_like.append(c)

shared_columns = [c for c in shared_columns if c not in nested_like]

# ================================
# 5. REGISTER EVALUATION FUNCTIONS
# ================================
strategy = DataFusionStrategy("books_validation_evaluation")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

for col in shared_columns:
    col_l = col.lower()

    if col in candidate_list_cols:
        register_eval(col, set_equality_match, "set_equality_match")
    elif col in {"isbn", "isbn_clean"} or col.endswith("_id") or col in {"id"}:
        register_eval(col, exact_match, "exact_match")
    elif "bool" in col_l or col_l.startswith("is_") or col_l.startswith("has_"):
        register_eval(col, boolean_match, "boolean_match")
    elif col_l in {"publish_year", "page_count", "rating", "price", "numratings"}:
        if col_l == "publish_year":
            register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.0)
        elif col_l == "page_count":
            register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.0)
        elif col_l == "rating":
            register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.05)
        elif col_l == "price":
            register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.01)
        else:
            register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.0)
    elif "date" in col_l and "year" not in col_l:
        register_eval(col, year_only_match, "year_only_match")
    else:
        # Includes title, author, language, publisher, country-like text, etc.
        threshold = 0.85
        if col_l in {"publisher", "language"}:
            threshold = 0.80
        elif col_l in {"title", "author"}:
            threshold = 0.85
        register_eval(col, tokenized_match, "tokenized_match", threshold=threshold)

print("[EVAL_FUNCTIONS] " + " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items())))

# ================================
# 6. RUN EVALUATION
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
                    if numeric.dropna().apply(lambda x: float(x).is_integer()).all():
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

# ================================
# 7. PRINT STRUCTURED METRICS
# ================================
structured_output = {
    "evaluation_stage": "validation",
    "fusion_path": FUSION_PATH,
    "evaluation_set_path": EVAL_SET_PATH,
    "fused_rows": int(len(fused_eval)),
    "gold_rows": int(len(gold_eval)),
    "shared_evaluated_columns": shared_columns,
    "list_like_columns": candidate_list_cols,
    "nested_columns_dropped": nested_like,
    "evaluation_functions": eval_funcs_summary,
    "metrics": evaluation_results,
}

print(json.dumps(structured_output, indent=2, ensure_ascii=False))

# ================================
# 8. WRITE OUTPUT
# ================================
evaluation_output = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
with open(evaluation_output, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4, ensure_ascii=False)