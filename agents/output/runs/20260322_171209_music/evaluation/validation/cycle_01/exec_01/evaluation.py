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
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
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


def _detect_gold_prefixes(gold_ids):
    prefixes = [g.split("_", 1)[0] for g in gold_ids if "_" in g]
    if not prefixes:
        return []
    ordered = [p for p, _ in Counter(prefixes).most_common()]
    return [p + "_" for p in ordered]


def build_eval_view(fused_df, gold_df):
    gold_ids = set(gold_df["id"].dropna().astype(str).tolist()) if "id" in gold_df.columns else set()
    gold_prefixes = _detect_gold_prefixes(gold_ids)
    expanded_rows = []

    for _, row in fused_df.iterrows():
        row_dict = row.to_dict()

        if "eval_id" in row_dict and not _safe_is_missing(row_dict["eval_id"]):
            eval_id = str(row_dict["eval_id"])
            enriched = dict(row_dict)
            enriched["eval_id"] = eval_id
            enriched["_eval_cluster_id"] = str(row_dict.get("_id", eval_id))
            expanded_rows.append(enriched)
            continue

        cluster_id = str(row_dict.get("_id", ""))
        source_ids = _parse_source_ids(row_dict.get("_fusion_sources"))

        candidates = []
        for prefix in gold_prefixes:
            pref_matches = [sid for sid in source_ids if sid.startswith(prefix)]
            if pref_matches:
                candidates.extend(pref_matches)

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
            enriched["eval_id"] = str(eval_id)
            enriched["_eval_cluster_id"] = cluster_id
            expanded_rows.append(enriched)

    if not expanded_rows:
        fallback = fused_df.copy()
        if "eval_id" not in fallback.columns:
            fallback["eval_id"] = fallback["_id"].astype(str)
        fallback["_eval_cluster_id"] = fallback["_id"].astype(str) if "_id" in fallback.columns else fallback["eval_id"].astype(str)
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
    officialish = sum(1 for v in values if any(tok in v.lower() for tok in ["republic", "kingdom", "states of", "federation"]))
    if alpha2 / total >= 0.70:
        return "alpha_2"
    if alpha3 / total >= 0.70:
        return "alpha_3"
    if numeric / total >= 0.70:
        return "numeric"
    if officialish / total >= 0.50:
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


def _normalize_list_text_case(value):
    if _safe_is_missing(value):
        return value
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    text = str(value).strip()
    return text.lower() if text else text


def _gold_column_meaningful(df, col):
    return col in df.columns and not df[col].isna().all()


def _canonicalize_numeric_series(series):
    s = pd.to_numeric(series, errors="coerce")
    non_null = s.dropna()
    if len(non_null) > 0 and non_null.apply(lambda x: float(x).is_integer()).all():
        return s.astype("Int64")
    return s


# ================================
# PATHS
# ================================

OUTPUT_DIR = "output/runs/20260322_171209_music"
FUSION_PATH = os.path.join(OUTPUT_DIR, "data_fusion", "fusion_data.csv")
EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
EVAL_PATH = os.path.join(EVAL_DIR, "pipeline_evaluation.json")
TESTSET_PATH = "input/datasets/music/testsets/test_set.xml"

os.makedirs(EVAL_DIR, exist_ok=True)

# ================================
# 1. LOAD DATA
# ================================

fused = pd.read_csv(FUSION_PATH)
fusion_eval_set = load_xml(TESTSET_PATH, name="fusion_validation_set", nested_handling="aggregate")
fused_eval = build_eval_view(fused, fusion_eval_set)

# ================================
# 2. DROP NON 1:1 / NON-MEANINGFUL COLUMNS
# ================================

gold_meaningful_cols = {
    c for c in fusion_eval_set.columns
    if c != "id" and _gold_column_meaningful(fusion_eval_set, c)
}

shared_cols = (set(fused_eval.columns) & gold_meaningful_cols) - {
    "_id", "id", "eval_id", "_eval_cluster_id", "_fusion_confidence", "_fusion_sources"
}

# Explicitly avoid evaluating provenance/meta columns and only allow direct matches
shared_cols = {
    c for c in shared_cols
    if not c.startswith("_") and not c.endswith("_provenance")
}

# ================================
# 3. LIST NORMALIZATION
# ================================

list_eval_columns = detect_list_like_columns(
    [fused_eval[list(shared_cols | {"eval_id"})] if "eval_id" in fused_eval.columns else fused_eval[list(shared_cols)],
     fusion_eval_set[list(shared_cols | {"id"})] if "id" in fusion_eval_set.columns else fusion_eval_set[list(shared_cols)]],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"},
)
list_eval_columns = [c for c in list_eval_columns if c in shared_cols]

if list_eval_columns:
    fused_eval, fusion_eval_set = normalize_list_like_columns([fused_eval, fusion_eval_set], list_eval_columns)
    for col in list_eval_columns:
        if col in fused_eval.columns:
            fused_eval[col] = fused_eval[col].apply(_normalize_list_text_case)
        if col in fusion_eval_set.columns:
            fusion_eval_set[col] = fusion_eval_set[col].apply(_normalize_list_text_case)
    print(f"[LIST NORMALIZATION] columns={list_eval_columns}")

# ================================
# 4. COUNTRY NORMALIZATION
# ================================

country_eval_columns = [c for c in shared_cols if "country" in c.lower()]
if country_eval_columns:
    country_format = _infer_country_output_format(fusion_eval_set, country_eval_columns)
    for col in country_eval_columns:
        fused_eval[col] = fused_eval[col].apply(lambda x: _normalize_country_safe(x, country_format))
        fusion_eval_set[col] = fusion_eval_set[col].apply(lambda x: _normalize_country_safe(x, country_format))
    print(f"[COUNTRY NORMALIZATION] format={country_format}; columns={country_eval_columns}")

# ================================
# 5. TYPE COERCION
# ================================

numeric_cols = [c for c in shared_cols if c in {"duration"}]
for col in numeric_cols:
    fused_eval[col] = _canonicalize_numeric_series(fused_eval[col])
    fusion_eval_set[col] = _canonicalize_numeric_series(fusion_eval_set[col])

date_cols = [c for c in shared_cols if c in {"release-date"}]
for col in date_cols:
    fused_eval[col] = pd.to_datetime(fused_eval[col], errors="coerce")
    fusion_eval_set[col] = pd.to_datetime(fusion_eval_set[col], errors="coerce")

# ================================
# 6. ID ALIGNMENT DIAGNOSTICS
# ================================

gold_ids = set(fusion_eval_set["id"].dropna().astype(str).tolist()) if "id" in fusion_eval_set.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0
print(f"[ID ALIGNMENT] direct _id coverage={direct_cov}/{len(gold_ids)}")
print(f"[ID ALIGNMENT] mapped eval_id coverage={mapped_cov}/{len(gold_ids)}")

# ================================
# 7. REGISTER EVALUATION FUNCTIONS
# ================================

strategy = DataFusionStrategy("music_evaluation_strategy")
eval_funcs_summary = {}

def register_eval(column, fn, fn_name, **kwargs):
    strategy.add_evaluation_function(column, fn, **kwargs)
    args = ",".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    eval_funcs_summary[column] = f"{fn_name}({args})" if args else fn_name

for col in sorted(shared_cols):
    if col in list_eval_columns:
        register_eval(col, set_equality_match, "set_equality_match")
    elif col in {"name", "artist", "label", "genre"}:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)
    elif "country" in col.lower():
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.80)
    elif col in {"release-date"}:
        register_eval(col, year_only_match, "year_only_match")
    elif col in {"duration"}:
        register_eval(col, numeric_tolerance_match, "numeric_tolerance_match", tolerance=5.0)
    elif col in {"id"}:
        register_eval(col, exact_match, "exact_match")
    elif fused_eval[col].dropna().isin([True, False]).all() and fusion_eval_set[col].dropna().isin([True, False]).all():
        register_eval(col, boolean_match, "boolean_match")
    else:
        register_eval(col, tokenized_match, "tokenized_match", threshold=0.85)

print("[EVALUATION FUNCTIONS] " + " | ".join(f"{k}:{v}" for k, v in sorted(eval_funcs_summary.items())))

# ================================
# 8. RUN EVALUATION
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

fused_eval = _strip_unevaluable_subcols(fused_eval, fusion_eval_set)

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

_coerce_numeric_cols(fused_eval, fusion_eval_set)

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

fused_eval = _normalize_list_separators(fused_eval, fusion_eval_set)
evaluation_results = evaluator.evaluate(
    fused_df=fused_eval,
    fused_id_column="eval_id",
    gold_df=fusion_eval_set,
    gold_id_column="id",
)

# ================================
# 9. PRINT STRUCTURED METRICS
# ================================

print("[EVALUATION RESULTS]")
print(json.dumps(evaluation_results, indent=2, default=str))

# ================================
# 10. SAVE RESULTS
# ================================

with open(EVAL_PATH, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4, default=str)