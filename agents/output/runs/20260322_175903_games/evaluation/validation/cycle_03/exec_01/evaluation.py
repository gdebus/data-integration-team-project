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
            resolved = str(_path.resolve())
            if resolved not in sys.path:
                sys.path.append(resolved)
    from list_normalization import detect_list_like_columns, normalize_list_like_columns


# ================================
# 0. PATHS
# ================================
OUTPUT_DIR = "output/runs/20260322_175903_games"
FUSION_PATH = os.path.join(OUTPUT_DIR, "data_fusion", "fusion_data.csv")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_OUTPUT_PATH = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
GOLD_PATH = "input/datasets/games/testsets/test_set_fusion.xml"

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
    gold_ids = set(gold_df["id"].dropna().astype(str).tolist()) if "id" in gold_df.columns else set()
    gold_prefix = _detect_gold_prefix(gold_ids)
    expanded_rows = []

    for _, row in fused_df.iterrows():
        row_dict = row.to_dict()
        cluster_id = str(row_dict.get("_id", ""))
        source_ids = _parse_source_ids(row_dict.get("_fusion_sources"))

        candidates = []
        if "eval_id" in row_dict and not _safe_is_missing(row_dict.get("eval_id")):
            candidates.append(str(row_dict.get("eval_id")))
        if gold_prefix:
            candidates.extend([sid for sid in source_ids if sid.startswith(gold_prefix)])
        candidates.extend([sid for sid in source_ids if sid in gold_ids])

        if not candidates and cluster_id:
            candidates = [cluster_id]

        seen = set()
        for eval_id in candidates:
            if not eval_id or eval_id in seen:
                continue
            seen.add(eval_id)
            enriched = dict(row_dict)
            enriched["eval_id"] = eval_id
            enriched["_eval_cluster_id"] = cluster_id
            expanded_rows.append(enriched)

    if not expanded_rows:
        fallback = fused_df.copy()
        id_col = "eval_id" if "eval_id" in fallback.columns else "_id"
        fallback["eval_id"] = fallback[id_col].astype(str)
        fallback["_eval_cluster_id"] = fallback.get("_id", fallback["eval_id"]).astype(str)
        return fallback

    fused_eval = pd.DataFrame(expanded_rows)
    if "_fusion_confidence" in fused_eval.columns:
        fused_eval["_fusion_confidence"] = pd.to_numeric(
            fused_eval["_fusion_confidence"], errors="coerce"
        ).fillna(0.0)
        fused_eval = fused_eval.sort_values("_fusion_confidence", ascending=False)
    fused_eval = fused_eval.drop_duplicates(subset=["eval_id"], keep="first")
    return fused_eval


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


def _canonicalize_scalar_text(value):
    if _safe_is_missing(value):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_canonicalize_scalar_text(v) for v in value]
    return str(value).strip()


def _coerce_numeric_like(df, columns):
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].dropna()
        if len(non_null) > 0:
            try:
                if non_null.apply(lambda x: float(x).is_integer()).all():
                    df[col] = df[col].astype("Int64")
            except Exception:
                pass
    return df


def _is_meaningful_column(df, col):
    return col in df.columns and df[col].notna().any()


# ================================
# 2. LOAD DATA
# ================================
fused = pd.read_csv(FUSION_PATH)
gold = load_xml(GOLD_PATH, name="fusion_validation_set", nested_handling="aggregate")

fused_eval = build_eval_view(fused, gold)

if "eval_id" not in fused_eval.columns:
    raise ValueError("Expected eval_id column in fused evaluation view.")

# ================================
# 3. DROP NON-DIRECT / INTERNAL COLUMNS
# ================================
internal_prefixes = ("_",)
fused_eval = fused_eval[[c for c in fused_eval.columns if not c.startswith(internal_prefixes) or c == "eval_id"]].copy()

gold_columns = set(gold.columns)
fused_columns = set(fused_eval.columns)

shared_columns = []
for col in sorted((fused_columns & gold_columns) - {"id", "eval_id"}):
    if not _is_meaningful_column(fused_eval, col):
        continue
    if not _is_meaningful_column(gold, col):
        continue
    shared_columns.append(col)

# Drop fused sub-columns when no direct 1:1 gold column exists
fused_eval = fused_eval[["eval_id"] + shared_columns].copy()
gold_eval = gold[["id"] + shared_columns].copy()

# ================================
# 4. NORMALIZE LIST-LIKE COLUMNS TO GOLD REPRESENTATION
# ================================
list_eval_columns = detect_list_like_columns(
    [fused_eval, gold_eval],
    exclude_columns={"id", "eval_id"},
)
list_eval_columns = [c for c in list_eval_columns if c in shared_columns]

if list_eval_columns:
    fused_eval, gold_eval = normalize_list_like_columns([fused_eval, gold_eval], list_eval_columns)
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
        if col in gold_eval.columns:
            gold_eval[col] = gold_eval[col].apply(_normalize_list_text_case)

# ================================
# 5. COUNTRY NORMALIZATION
# ================================
country_eval_columns = [c for c in shared_columns if "country" in str(c).lower()]
if country_eval_columns:
    country_format = _infer_country_output_format(gold_eval, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        gold_eval[col] = gold_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))

# ================================
# 6. GENERAL NORMALIZATION / TYPE ALIGNMENT
# ================================
for col in shared_columns:
    if col not in list_eval_columns:
        fused_eval[col] = fused_eval[col].apply(_canonicalize_scalar_text)
        gold_eval[col] = gold_eval[col].apply(_canonicalize_scalar_text)

numeric_like_patterns = [
    "score", "sales", "count", "rating", "price", "revenue",
    "latitude", "longitude", "duration", "amount", "number"
]
numeric_columns = [
    c for c in shared_columns
    if any(p in c.lower() for p in numeric_like_patterns)
]

fused_eval = _coerce_numeric_like(fused_eval, numeric_columns)
gold_eval = _coerce_numeric_like(gold_eval, numeric_columns)

# ================================
# 7. REGISTER EVALUATION FUNCTIONS
# ================================
strategy = DataFusionStrategy("games_evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{column}:{fn_name}({args})" if args else f"{column}:{fn_name}"

for col in shared_columns:
    col_l = col.lower()

    if col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif col == "releaseYear" or col.endswith("Date") or "date" in col_l or col.endswith("Year"):
        register_eval(col, year_only_match, "year_only_match")
    elif col in {"criticScore", "userScore", "globalSales"} or any(k in col_l for k in ["score", "sales", "rating", "count", "price", "revenue", "latitude", "longitude", "amount", "number"]):
        tolerance = 0.5 if "score" in col_l or "rating" in col_l else 0.01
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=tolerance)
    elif col in {"ESRB"} or col.endswith("_id") or col in {"id"}:
        register_eval(col, exact_match, "exact_match")
    elif "is_" in col_l or col_l.startswith("has_") or col_l.startswith("flag_") or col_l.startswith("bool_"):
        register_eval(col, boolean_match, "boolean_match")
    elif any(k in col_l for k in ["name", "title", "developer", "publisher", "platform", "genre", "series"]):
        threshold = 0.85 if col in {"name", "developer", "publisher"} else 0.75
        register_eval(col, tokenized_match, "tokenized_match", threshold=threshold)
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.80)

# ================================
# 8. DIAGNOSTICS
# ================================
gold_ids = set(gold_eval["id"].dropna().astype(str).tolist()) if "id" in gold_eval.columns else set()
direct_cov = len(set(fused.get("_id", pd.Series(dtype=str)).astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0

print(json.dumps({
    "stage": "validation",
    "fused_rows_raw": int(len(fused)),
    "fused_rows_eval_view": int(len(fused_eval)),
    "gold_rows": int(len(gold_eval)),
    "shared_columns": shared_columns,
    "list_columns": list_eval_columns,
    "country_columns": country_eval_columns,
    "numeric_columns": numeric_columns,
    "id_alignment": {
        "direct__id_coverage": f"{direct_cov}/{len(gold_ids)}",
        "mapped_eval_id_coverage": f"{mapped_cov}/{len(gold_ids)}",
    },
}, indent=2))

print("[EVAL_FUNCS] " + " | ".join(eval_funcs_summary[c] for c in sorted(eval_funcs_summary)))

# ================================
# 9. RUN EVALUATION
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

print(json.dumps(evaluation_results, indent=2))

# ================================
# 10. WRITE RESULTS
# ================================
with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4, ensure_ascii=False)