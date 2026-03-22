import os
import sys
import json
import ast
import re
from pathlib import Path
from collections import Counter

import pandas as pd

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
# 0. PATHS
# ================================
OUTPUT_DIR = "output/runs/20260322_171209_music"
FUSION_PATH = os.path.join(OUTPUT_DIR, "data_fusion", "fusion_data.csv")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_OUTPUT_PATH = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
EVAL_DEBUG_PATH = os.path.join(EVAL_DIR, "debug_fusion_eval.jsonl")
FUSION_DEBUG_PATH = os.path.join(OUTPUT_DIR, "data_fusion", "debug_fusion_data.jsonl")

EVALUATION_STAGE = "validation"
EVAL_SET_PATH = "input/datasets/music/testsets/test_set.xml"

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
        if gold_prefix:
            candidates = [sid for sid in source_ids if sid.startswith(gold_prefix)]
        if not candidates:
            candidates = [sid for sid in source_ids if sid in gold_ids]
        if not candidates and "eval_id" in row_dict and str(row_dict.get("eval_id", "")).strip():
            candidates = [str(row_dict["eval_id"])]
        if not candidates and cluster_id:
            candidates = [cluster_id]

        seen = set()
        for eval_id in candidates:
            if eval_id in seen:
                continue
            seen.add(eval_id)
            enriched = dict(row_dict)
            enriched["eval_id"] = str(eval_id)
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
    official_name_hits = sum(1 for v in values if any(tok in v.lower() for tok in ["republic", "kingdom", "states of", "federation"]))
    if official_name_hits / total >= 0.40:
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


def _normalize_text_token(value):
    if _safe_is_missing(value):
        return value
    return re.sub(r"\s+", " ", str(value).strip())


def _normalize_list_text_case(value):
    if _safe_is_missing(value):
        return value
    if isinstance(value, (list, tuple, set)):
        out = []
        for v in value:
            if _safe_is_missing(v):
                continue
            t = re.sub(r"\s+", " ", str(v).strip()).lower()
            if t:
                out.append(t)
        return out
    text = re.sub(r"\s+", " ", str(value).strip()).lower()
    return text if text else text


def _normalize_duration_list(value):
    if _safe_is_missing(value):
        return value
    items = value if isinstance(value, (list, tuple, set)) else [value]
    out = []
    for v in items:
        if _safe_is_missing(v):
            continue
        s = str(v).strip()
        if not s:
            continue
        try:
            n = pd.to_numeric(s, errors="coerce")
            if pd.notna(n):
                out.append(str(int(round(float(n)))))
            else:
                out.append(s)
        except Exception:
            out.append(s)
    return out if isinstance(value, (list, tuple, set)) else (out[0] if out else value)


def _coerce_numeric_column(df, col):
    if col not in df.columns:
        return
    df[col] = pd.to_numeric(df[col], errors="coerce")
    non_null = df[col].dropna()
    if len(non_null) and non_null.apply(lambda x: float(x).is_integer()).all():
        df[col] = df[col].astype("Int64")


def _gold_column_meaningful(df, col):
    return col in df.columns and df[col].notna().any()


# ================================
# 2. LOAD DATA
# ================================
fused = pd.read_csv(FUSION_PATH)
gold = load_xml(EVAL_SET_PATH, name=f"{EVALUATION_STAGE}_set", nested_handling="aggregate")
fused_eval = build_eval_view(fused, gold)

print(json.dumps({
    "stage": EVALUATION_STAGE,
    "fused_path": FUSION_PATH,
    "gold_path": EVAL_SET_PATH,
    "fused_rows": int(len(fused)),
    "gold_rows": int(len(gold)),
    "fused_eval_rows": int(len(fused_eval)),
}, indent=2))


# ================================
# 3. ALIGN COLUMN SPACE
#    Only evaluate direct 1:1 columns present and meaningful in both.
# ================================
shared_columns = set(fused_eval.columns) & set(gold.columns)
direct_eval_columns = []
for col in sorted(shared_columns):
    if col in {"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence", "_fusion_sources"}:
        continue
    if col.startswith("_"):
        continue
    if not _gold_column_meaningful(gold, col):
        continue
    direct_eval_columns.append(col)

# Explicitly avoid evaluating derived sub-columns without direct gold match.
fused_eval = fused_eval[[c for c in fused_eval.columns if c in set(direct_eval_columns) | {"eval_id", "_id", "_eval_cluster_id", "_fusion_confidence", "_fusion_sources"}]]
gold = gold[[c for c in gold.columns if c in set(direct_eval_columns) | {"id"}]]

print(json.dumps({
    "direct_eval_columns": direct_eval_columns
}, indent=2))


# ================================
# 4. NORMALIZE LIST-LIKE COLUMNS TO GOLD REPRESENTATION
# ================================
list_eval_columns = detect_list_like_columns(
    [fused_eval, gold],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence", "_fusion_sources"},
)
list_eval_columns = [c for c in list_eval_columns if c in direct_eval_columns and not c.startswith("_")]

if list_eval_columns:
    fused_eval, gold = normalize_list_like_columns([fused_eval, gold], list_eval_columns)
    for col in list_eval_columns:
        if col in {"tracks_track_duration"}:
            fused_eval[col] = fused_eval[col].apply(_normalize_duration_list)
            gold[col] = gold[col].apply(_normalize_duration_list)
        else:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
            gold[col] = gold[col].apply(_normalize_list_text_case)

print(json.dumps({
    "list_eval_columns": list_eval_columns
}, indent=2))


# ================================
# 5. COUNTRY NORMALIZATION TO GOLD REPRESENTATION
# ================================
country_eval_columns = [c for c in direct_eval_columns if "country" in c.lower()]
if country_eval_columns:
    country_format = _infer_country_output_format(gold, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        gold[col] = gold[col].apply(lambda x: _normalize_country_safe(x, country_format))
else:
    country_format = None

print(json.dumps({
    "country_eval_columns": country_eval_columns,
    "country_format": country_format
}, indent=2))


# ================================
# 6. GENERAL NORMALIZATION
# ================================
text_columns = [c for c in direct_eval_columns if c not in list_eval_columns]
for col in text_columns:
    if col in {"duration"}:
        continue
    if "date" in col.lower():
        continue
    if "country" in col.lower():
        continue
    fused_eval[col] = fused_eval[col].apply(_normalize_text_token)
    gold[col] = gold[col].apply(_normalize_text_token)

numeric_columns = [c for c in direct_eval_columns if c in {"duration", "year", "count", "rating", "score"} or c.lower().endswith(("_year", "_count", "_score", "_rating"))]
for col in numeric_columns:
    _coerce_numeric_column(fused_eval, col)
    _coerce_numeric_column(gold, col)


# ================================
# 7. ID ALIGNMENT DIAGNOSTICS
# ================================
gold_ids = set(gold["id"].dropna().astype(str).tolist()) if "id" in gold.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
eval_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0

print(json.dumps({
    "id_alignment": {
        "direct__id_coverage": f"{direct_cov}/{len(gold_ids)}",
        "mapped_eval_id_coverage": f"{eval_cov}/{len(gold_ids)}",
        "using_fused_id_column": "eval_id"
    }
}, indent=2))


# ================================
# 8. REGISTER EVALUATION FUNCTIONS
#    Based on actual shared columns only.
# ================================
strategy = DataFusionStrategy("music_evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

for col in direct_eval_columns:
    col_l = col.lower()

    if col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif col in {"name", "artist", "label", "genre"}:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)
    elif "country" in col_l:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.80)
    elif "date" in col_l or col_l in {"founded", "published", "release-date"}:
        register_eval(col, year_only_match, "year_only_match")
    elif col_l in {"duration", "revenue", "latitude", "longitude", "price", "rating", "score"} or col in numeric_columns:
        tolerance = 1.0 if col_l == "duration" else 0.01
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=tolerance)
    elif col_l.startswith("is_") or col_l.startswith("has_") or col_l in {"active", "available", "explicit"}:
        register_eval(col, boolean_match, "boolean_match")
    elif col_l in {"id", "code", "isbn", "issn"}:
        register_eval(col, exact_match, "exact_match")
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)

compact_summary = " | ".join(f"{col}:{spec}" for col, spec in sorted(eval_funcs_summary.items()))
print(f"[EVAL FUNCS] {compact_summary}")


# ================================
# 9. RUN EVALUATION
# ================================
evaluator = DataFusionEvaluator(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=EVAL_DEBUG_PATH,
    fusion_debug_logs=FUSION_DEBUG_PATH if os.path.exists(FUSION_DEBUG_PATH) else None,
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
                    if numeric.dropna().apply(lambda x: float(x).is_integer()).all():
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
    fused_id_column="eval_id",
    gold_df=gold,
    gold_id_column="id",
)

print(json.dumps({
    "evaluation_stage": EVALUATION_STAGE,
    "evaluation_results": evaluation_results
}, indent=2, default=str))


# ================================
# 10. WRITE OUTPUT
# ================================
with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4, ensure_ascii=False)

print(json.dumps({
    "status": "success",
    "evaluation_output": EVAL_OUTPUT_PATH
}, indent=2))