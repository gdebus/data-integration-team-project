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


# ================================
# HELPERS
# ================================

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
        if "eval_id" in row_dict and pd.notna(row_dict.get("eval_id")):
            candidates.append(str(row_dict["eval_id"]))
        if gold_prefix:
            candidates.extend([sid for sid in source_ids if sid.startswith(gold_prefix)])
        candidates.extend([sid for sid in source_ids if sid in gold_ids])

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
        return sorted({str(v).strip().lower() for v in value if str(v).strip()})
    text = str(value).strip()
    return text.lower() if text else text


def _drop_derived_subcolumns_without_gold_match(fused_df, gold_df):
    gold_cols = set(gold_df.columns)
    to_drop = []
    for col in fused_df.columns:
        if col in gold_cols:
            continue
        if col.startswith("_"):
            continue
        if "_" in col:
            parent = col.split("_", 1)[0]
            if parent in gold_cols:
                to_drop.append(col)
    if to_drop:
        fused_df = fused_df.drop(columns=to_drop, errors="ignore")
    return fused_df, to_drop


def _is_meaningful_column(df, col):
    if col not in df.columns:
        return False
    series = df[col]
    if len(series) == 0:
        return False
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    return True


def _looks_boolean(series):
    values = set(str(v).strip().lower() for v in series.dropna().head(500).tolist())
    boolean_vocab = {"true", "false", "yes", "no", "1", "0", "t", "f"}
    return len(values) > 0 and values.issubset(boolean_vocab)


def _coerce_numeric_like(df, cols):
    for col in cols:
        if col not in df.columns:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == 0:
            continue
        df[col] = converted
        non_null = converted.dropna()
        if len(non_null) > 0 and non_null.apply(lambda x: float(x).is_integer()).all():
            df[col] = converted.astype("Int64")
    return df


# ================================
# PATHS
# ================================

OUTPUT_DIR = "output/runs/20260322_175903_games"
FUSION_PATH = os.path.join(OUTPUT_DIR, "data_fusion", "fusion_data.csv")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_PATH = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
DEBUG_PATH = os.path.join(EVAL_DIR, "debug_fusion_eval.jsonl")
GOLD_PATH = "input/datasets/games/testsets/test_set_fusion.xml"

os.makedirs(EVAL_DIR, exist_ok=True)

# ================================
# LOAD DATA
# ================================

fused = pd.read_csv(FUSION_PATH)
gold = load_xml(GOLD_PATH, name="fusion_test_set", nested_handling="aggregate")
fused_eval = build_eval_view(fused, gold)

# ================================
# DROP DERIVED SUB-COLUMNS WITHOUT GOLD MATCH
# ================================

fused_eval, dropped_derived_cols = _drop_derived_subcolumns_without_gold_match(fused_eval, gold)

# ================================
# LIST NORMALIZATION
# ================================

list_eval_columns = detect_list_like_columns(
    [fused_eval, gold],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if not c.startswith("_") and c in fused_eval.columns and c in gold.columns]

if list_eval_columns:
    fused_eval, gold = normalize_list_like_columns([fused_eval, gold], list_eval_columns)
    for col in list_eval_columns:
        fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
        gold[col] = gold[col].apply(_normalize_list_text_case)

# ================================
# COUNTRY NORMALIZATION
# ================================

country_eval_columns = [
    c for c in (set(fused_eval.columns) & set(gold.columns))
    if "country" in str(c).lower()
]

if country_eval_columns:
    country_format = _infer_country_output_format(gold, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        gold[col] = gold[col].apply(lambda x: _normalize_country_safe(x, country_format))
else:
    country_format = None

# ================================
# NUMERIC COERCION
# ================================

numeric_like_candidates = []
for col in sorted(set(fused_eval.columns) & set(gold.columns)):
    name = col.lower()
    if any(tok in name for tok in ["year", "count", "score", "rating", "price", "sales", "revenue", "latitude", "longitude", "duration", "pages"]):
        numeric_like_candidates.append(col)

fused_eval = _coerce_numeric_like(fused_eval, numeric_like_candidates)
gold = _coerce_numeric_like(gold, numeric_like_candidates)

# ================================
# CHOOSE EVALUATION COLUMNS
# ================================

shared_columns = sorted(set(fused_eval.columns) & set(gold.columns))
excluded_cols = {"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence", "_fusion_sources"}

eval_columns = []
for col in shared_columns:
    if col in excluded_cols or col.startswith("_"):
        continue
    if not _is_meaningful_column(fused_eval, col):
        continue
    if not _is_meaningful_column(gold, col):
        continue
    eval_columns.append(col)

# ================================
# REGISTER EVALUATION FUNCTIONS
# ================================

strategy = DataFusionStrategy("games_evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

for col in eval_columns:
    lower = col.lower()

    if col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif lower in {"id", "isbn", "issn", "ean", "upc", "isrc", "doi", "orcid"}:
        register_eval(col, exact_match, "exact_match")
    elif _looks_boolean(gold[col].astype("object")):
        register_eval(col, boolean_match, "boolean_match")
    elif "year" in lower or "date" in lower or lower.startswith("release"):
        register_eval(col, year_only_match, "year_only_match")
    elif any(tok in lower for tok in ["score", "rating"]):
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.02)
    elif any(tok in lower for tok in ["sales", "revenue", "price", "latitude", "longitude", "count", "pages", "duration"]):
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=0.01)
    elif any(tok in lower for tok in ["name", "title", "developer", "publisher", "platform", "genre", "series", "studio", "label", "city", "street", "country"]):
        threshold = 0.85
        if "platform" in lower:
            threshold = 0.80
        elif "publisher" in lower or "developer" in lower:
            threshold = 0.80
        elif "genre" in lower:
            threshold = 0.75
        elif "country" in lower:
            threshold = 0.65
        register_eval(col, tokenized_match, "tokenized_match", threshold=threshold)
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)

# ================================
# ID ALIGNMENT DIAGNOSTICS
# ================================

gold_ids = set(gold["id"].dropna().astype(str).tolist()) if "id" in gold.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0

print(json.dumps({
    "stage": "sealed_test",
    "evaluation_set": GOLD_PATH,
    "fused_path": FUSION_PATH,
    "output_path": EVAL_PATH,
    "id_alignment": {
        "direct_id_coverage": {"matched": direct_cov, "total_gold": len(gold_ids)},
        "mapped_eval_id_coverage": {"matched": mapped_cov, "total_gold": len(gold_ids)}
    },
    "dropped_derived_subcolumns": dropped_derived_cols,
    "list_eval_columns": list_eval_columns,
    "country_eval_columns": country_eval_columns,
    "country_target_format": country_format,
    "numeric_like_columns": numeric_like_candidates,
    "evaluated_columns": eval_columns,
}, indent=2))

print("EVAL_FUNCTIONS|" + "; ".join(f"{k}={v}" for k, v in sorted(eval_funcs_summary.items())))

# ================================
# RUN EVALUATION
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
    fused_id_column="eval_id" if "eval_id" in fused_eval.columns else "_id",
    gold_df=gold,
    gold_id_column="id",
)

print(json.dumps(evaluation_results, indent=2, default=str))

# ================================
# SAVE RESULTS
# ================================

with open(EVAL_PATH, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4, ensure_ascii=False)