import pandas as pd
import json
import ast
import re
import os
from collections import Counter
from pathlib import Path
import sys

from PyDI.io import load_xml, load_parquet, load_csv
from PyDI.normalization import normalize_country
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEvaluator,
    # Evaluation functions — choose based on attribute type:
    #   tokenized_match(threshold=1.0)        — fuzzy string overlap (default for most strings)
    #   exact_match                           — strict string equality (IDs, codes)
    #   year_only_match                       — compares year component only (dates)
    #   numeric_tolerance_match(tolerance=0.01) — numeric within relative tolerance
    #   set_equality_match                    — exact set comparison (list/set attributes)
    #   boolean_match                         — boolean comparison
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
    _candidates = [Path.cwd(), Path.cwd() / "agents",
                   Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent,
                   Path(__file__).resolve().parent.parent.parent]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            if str(_path.resolve()) not in sys.path:
                sys.path.append(str(_path.resolve()))
    from list_normalization import detect_list_like_columns, normalize_list_like_columns


# ================================
# HELPER: Build evaluation view by mapping fused cluster IDs to gold test set IDs
# The fused output uses cluster IDs (_id), but the gold test set uses source IDs.
# This expands fused rows via _fusion_sources to align with gold IDs.
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
    """Map fused entity IDs to gold IDs via _fusion_sources for evaluation alignment."""
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
            fused_eval["_fusion_confidence"], errors="coerce").fillna(0.0)
        fused_eval = fused_eval.sort_values("_fusion_confidence", ascending=False)
    fused_eval = fused_eval.drop_duplicates(subset=["eval_id"], keep="first")
    return fused_eval


# ================================
# HELPERS: Country normalization for evaluation alignment
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
    """Auto-detect the country format used in the gold test set."""
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
    if alpha2 / total >= 0.70:
        return "alpha_2"
    if alpha3 / total >= 0.70:
        return "alpha_3"
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


# === 0. OUTPUT DIRECTORY ===
# CRITICAL: Use OUTPUT_DIR for ALL output paths. It will be provided in the prompt.
# Do NOT hardcode "output/" — always use os.path.join(OUTPUT_DIR, ...).
OUTPUT_DIR = "output"  # Will be replaced by prompt with the actual run-scoped directory

FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

# ================================
# 1. LOAD FUSED OUTPUT AND GOLD TEST SET
# ================================

fused = pd.read_csv(os.path.join(FUSION_DIR, "fusion_data.csv"))
fusion_test_set = load_xml("<path-to-testset>", name="fusion_test_set", nested_handling="aggregate")
fused_eval = build_eval_view(fused, fusion_test_set)

# ================================
# 2. NORMALIZE LIST-LIKE COLUMNS FOR EVALUATION
# ================================

list_eval_columns = detect_list_like_columns(
    [fused_eval, fusion_test_set],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if not c.startswith("_")]
if list_eval_columns:
    fused_eval, fusion_test_set = normalize_list_like_columns(
        [fused_eval, fusion_test_set], list_eval_columns)
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
        if col in fusion_test_set.columns:
            fusion_test_set[col] = fusion_test_set[col].apply(_normalize_list_text_case)
    print(f"[LIST NORMALIZATION] columns: {', '.join(list_eval_columns)}")

# ================================
# 3. NORMALIZE COUNTRY COLUMNS FOR EVALUATION
# ================================

country_eval_columns = [
    c for c in (set(fused_eval.columns) & set(fusion_test_set.columns))
    if "country" in str(c).lower()
]
if country_eval_columns:
    country_format = _infer_country_output_format(fusion_test_set, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        fusion_test_set[col] = fusion_test_set[col].apply(lambda x: _normalize_country_safe(x, country_format))
    print(f"[COUNTRY NORMALIZATION] format={country_format}, columns={country_eval_columns}")

# ================================
# 4. ID ALIGNMENT DIAGNOSTICS
# ================================

gold_ids = set(fusion_test_set["id"].dropna().astype(str).tolist()) if "id" in fusion_test_set.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0
print(f"[ID ALIGNMENT] direct _id coverage: {direct_cov}/{len(gold_ids)}")
print(f"[ID ALIGNMENT] mapped eval_id coverage: {mapped_cov}/{len(gold_ids)}")

# ================================
# 5. REGISTER EVALUATION FUNCTIONS
#
# CHOOSING EVALUATION FUNCTIONS:
#   Default for strings:     tokenized_match(threshold=0.85)
#   Names/titles:            tokenized_match(threshold=0.85)
#   Addresses:               tokenized_match(threshold=0.75)  — lenient for formatting
#   Country fields:          tokenized_match(threshold=0.65)  — handles name/code diffs
#   Dates:                   year_only_match                  — robust for mixed formats
#   Numbers:                 numeric_tolerance_match(tolerance=0.01)
#   Lists:                   tokenized_match(threshold=0.75)
#   Booleans:                boolean_match
#   Standardized codes:      exact_match
# ================================

strategy = DataFusionStrategy("evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

# String attributes
register_eval("name", tokenized_match, "tokenized_match", threshold=0.85)
register_eval("city", tokenized_match, "tokenized_match", threshold=0.75)
register_eval("country", tokenized_match, "tokenized_match", threshold=0.65)
register_eval("street", tokenized_match, "tokenized_match", threshold=0.75)

# Date attributes
register_eval("founded", year_only_match, "year_only_match")

# Numeric attributes
register_eval("revenue", numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.01)
register_eval("latitude", numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.001)

# List attributes — auto-register any detected list columns not already registered
for col in list_eval_columns:
    if col not in eval_funcs_summary:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.75)

print(f"[EVALUATION] Registered: {json.dumps(eval_funcs_summary, indent=2)}")

# ================================
# 6. RUN EVALUATION
# ================================

evaluator = DataFusionEvaluator(
    strategy, debug=True, debug_format="json",
    debug_file=os.path.join(EVAL_DIR, "debug_fusion_eval.jsonl"),
)

evaluation_results = evaluator.evaluate(
    fused_df=fused_eval,
    fused_id_column="eval_id",
    gold_df=fusion_test_set,
    gold_id_column="id",
)

# ================================
# 7. WRITE RESULTS
# CRITICAL: Output MUST be written to this exact path as JSON.
# ================================

evaluation_output = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
with open(evaluation_output, "w") as f:
    json.dump(evaluation_results, f, indent=4)
