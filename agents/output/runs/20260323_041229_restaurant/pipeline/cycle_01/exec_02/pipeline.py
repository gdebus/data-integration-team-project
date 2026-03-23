# CRITICAL: Do not adjust output file names.
# CRITICAL: Always use OUTPUT_DIR for all output paths (correspondences, fusion, debug).

from PyDI.io import load_csv
from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
    StringComparator,
    NumericComparator,
    RuleBasedMatcher,
)
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    voting,
    most_complete,
    median,
    average,
    union,
    prefer_higher_trust,
)

import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
import sys
import ast as _ast

def _safe_scalar_isna(x):
    """Safe pd.isna() that handles list/array values without crashing."""
    if isinstance(x, (list, tuple, set)):
        return False  # a list is not null, even if it contains null elements
    try:
        result = pd.isna(x)
        if hasattr(result, '__len__') and not isinstance(result, str):
            return False  # array-like result from pd.isna on non-scalar
        return bool(result)
    except (ValueError, TypeError):
        return False

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

# === 0. OUTPUT DIRECTORY ===
OUTPUT_DIR = "output/runs/20260323_041229_restaurant/"

# === 1. LOAD DATA ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

kaggle_small = load_csv(
    "output/runs/20260323_041229_restaurant/normalization/attempt_1/kaggle_small.csv",
    name="kaggle_small",
)
uber_eats_small = load_csv(
    "output/runs/20260323_041229_restaurant/normalization/attempt_1/uber_eats_small.csv",
    name="uber_eats_small",
)
yelp_small = load_csv(
    "output/runs/20260323_041229_restaurant/normalization/attempt_1/yelp_small.csv",
    name="yelp_small",
)

# === 2. SCHEMA MATCHING ===
# Datasets are already schema-aligned from normalization.

# === 3. LIST NORMALIZATION ===
list_like_columns = detect_list_like_columns(
    [kaggle_small, uber_eats_small, yelp_small],
    exclude_columns={"id", "_id"},
)
if list_like_columns:
    kaggle_small, uber_eats_small, yelp_small = normalize_list_like_columns(
        [kaggle_small, uber_eats_small, yelp_small], list_like_columns
    )
    print(f"Normalized list-like columns: {', '.join(list_like_columns)}")

datasets = [kaggle_small, uber_eats_small, yelp_small]

# === 3b. TARGETED INLINE NORMALIZATION ===
def lower_strip(x):
    if isinstance(x, (list, tuple, set)):
        return " ".join(str(v).lower().strip() for v in x if v is not None and pd.notna(v))
    try:
        _is_na = pd.isna(x)
    except (ValueError, TypeError):
        _is_na = False
    if _is_na:
        return x
    return str(x).lower().strip()

def strip_only(x):
    if isinstance(x, (list, tuple, set)):
        return " ".join(str(v).strip() for v in x if v is not None and pd.notna(v))
    if pd.isna(x):
        return x
    return str(x).strip()

def stringify_scalar(x):
    if isinstance(x, (list, tuple, set)):
        return " ".join(str(v) for v in x if v is not None and pd.notna(v))
    if pd.isna(x):
        return x
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)

def _flatten_list_cols_for_blocking(df, text_cols):
    out = df.copy()
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: " ".join(str(v) for v in x if v is not None and not _safe_scalar_isna(v))
                if isinstance(x, (list, tuple, set))
                else ("" if pd.isna(x) else str(x))
            )
    return out

# Phone/postal columns are used in exact blocking + string comparison for kaggle/yelp.
# Normalize representation to robust string form so exact blocking is not defeated by
# float-vs-string formatting differences like 35010.0 vs "35010".
for df in [kaggle_small, yelp_small]:
    if "phone_e164" in df.columns:
        df["phone_e164"] = df["phone_e164"].apply(stringify_scalar)
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].apply(stringify_scalar)

# === 4. BLOCKING ===
print("Performing Blocking")

blocker_kaggle_small_uber_eats_small = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(kaggle_small, ["name_norm", "city", "state"]),
    _flatten_list_cols_for_blocking(uber_eats_small, ["name_norm", "city", "state"]),
    text_cols=["name_norm", "city", "state"],
    id_column="id",
    top_k=20,
)
candidates_kaggle_small_uber_eats_small = blocker_kaggle_small_uber_eats_small.materialize()

blocker_kaggle_small_yelp_small = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    id_column="id",
    batch_size=100000,
)
candidates_kaggle_small_yelp_small = blocker_kaggle_small_yelp_small.materialize()

# Fail-soft only for the problematic pair: if exact-match multi blocking yields zero
# candidates due to representation inconsistencies, back off to exact phone match only.
if candidates_kaggle_small_yelp_small.empty:
    blocker_kaggle_small_yelp_small = StandardBlocker(
        kaggle_small,
        yelp_small,
        on=["phone_e164"],
        id_column="id",
        batch_size=100000,
    )
    candidates_kaggle_small_yelp_small = blocker_kaggle_small_yelp_small.materialize()

blocker_uber_eats_small_yelp_small = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(uber_eats_small, ["name_norm", "street", "city"]),
    _flatten_list_cols_for_blocking(yelp_small, ["name_norm", "street", "city"]),
    text_cols=["name_norm", "street", "city"],
    id_column="id",
    top_k=20,
)
candidates_uber_eats_small_yelp_small = blocker_uber_eats_small_yelp_small.materialize()

# === 5. ENTITY MATCHING ===
print("Matching Entities")

threshold_kaggle_small_uber_eats_small = 0.75
threshold_kaggle_small_yelp_small = 0.75
threshold_uber_eats_small_yelp_small = 0.72

comparators_kaggle_small_uber_eats_small = [
    StringComparator(column="name_norm", similarity_function="cosine", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="street", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="city", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="state", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    NumericComparator(column="latitude", method="absolute_difference", max_difference=0.01),
    NumericComparator(column="longitude", method="absolute_difference", max_difference=0.01),
    StringComparator(column="categories", similarity_function="jaccard", preprocess=lower_strip, list_strategy="set_jaccard"),
]

comparators_kaggle_small_yelp_small = [
    StringComparator(column="phone_e164", similarity_function="jaro_winkler", preprocess=strip_only, list_strategy="concatenate"),
    StringComparator(column="name_norm", similarity_function="cosine", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="postal_code", similarity_function="jaro_winkler", preprocess=strip_only, list_strategy="concatenate"),
    StringComparator(column="city", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    NumericComparator(column="latitude", method="absolute_difference", max_difference=0.002),
    NumericComparator(column="longitude", method="absolute_difference", max_difference=0.002),
    StringComparator(column="categories", similarity_function="jaccard", preprocess=lower_strip, list_strategy="set_jaccard"),
]

comparators_uber_eats_small_yelp_small = [
    StringComparator(column="name_norm", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="street", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    StringComparator(column="city", similarity_function="jaro_winkler", preprocess=lower_strip, list_strategy="concatenate"),
    NumericComparator(column="latitude", method="absolute_difference", max_difference=0.002),
    NumericComparator(column="longitude", method="absolute_difference", max_difference=0.002),
    StringComparator(column="categories", similarity_function="jaccard", preprocess=lower_strip, list_strategy="set_jaccard"),
]

matcher = RuleBasedMatcher()

rb_correspondences_kaggle_small_uber_eats_small = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=candidates_kaggle_small_uber_eats_small,
    comparators=comparators_kaggle_small_uber_eats_small,
    weights=[0.35, 0.12, 0.08, 0.05, 0.16, 0.16, 0.08],
    threshold=threshold_kaggle_small_uber_eats_small,
    id_column="id",
)

rb_correspondences_kaggle_small_yelp_small = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=candidates_kaggle_small_yelp_small,
    comparators=comparators_kaggle_small_yelp_small,
    weights=[0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1],
    threshold=threshold_kaggle_small_yelp_small,
    id_column="id",
)

rb_correspondences_uber_eats_small_yelp_small = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=candidates_uber_eats_small_yelp_small,
    comparators=comparators_uber_eats_small_yelp_small,
    weights=[0.28, 0.16, 0.12, 0.16, 0.16, 0.12],
    threshold=threshold_uber_eats_small_yelp_small,
    id_column="id",
)

if rb_correspondences_kaggle_small_uber_eats_small.empty:
    raise ValueError("Empty correspondences for kaggle_small_uber_eats_small")
if rb_correspondences_kaggle_small_yelp_small.empty:
    raise ValueError("Empty correspondences for kaggle_small_yelp_small")
if rb_correspondences_uber_eats_small_yelp_small.empty:
    raise ValueError("Empty correspondences for uber_eats_small_yelp_small")

# === 6. SAVE CORRESPONDENCES (MANDATORY) ===
CORR_DIR = os.path.join(OUTPUT_DIR, "correspondences")
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_kaggle_small_uber_eats_small.to_csv(
    os.path.join(CORR_DIR, "correspondences_kaggle_small_uber_eats_small.csv"),
    index=False,
)
rb_correspondences_kaggle_small_yelp_small.to_csv(
    os.path.join(CORR_DIR, "correspondences_kaggle_small_yelp_small.csv"),
    index=False,
)
rb_correspondences_uber_eats_small_yelp_small.to_csv(
    os.path.join(CORR_DIR, "correspondences_uber_eats_small_yelp_small.csv"),
    index=False,
)

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_kaggle_small_uber_eats_small,
        rb_correspondences_kaggle_small_yelp_small,
        rb_correspondences_uber_eats_small_yelp_small,
    ],
    ignore_index=True,
)

# === 7. DATA FUSION ===
print("Fusing Data")

trust_map = {
    "yelp_small": 3,
    "kaggle_small": 2,
    "uber_eats_small": 1,
}

strategy = DataFusionStrategy("fusion_strategy")

strategy.add_attribute_fuser("source", prefer_higher_trust, trust_map=trust_map)

strategy.add_attribute_fuser("name", voting)
strategy.add_attribute_fuser("name_norm", voting)
strategy.add_attribute_fuser("website", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("map_url", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("address_line1", most_complete)
strategy.add_attribute_fuser("address_line2", most_complete)
strategy.add_attribute_fuser("street", most_complete)
strategy.add_attribute_fuser("house_number", voting)
strategy.add_attribute_fuser("city", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("state", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("postal_code", voting)
strategy.add_attribute_fuser("country", prefer_higher_trust, trust_map=trust_map)

strategy.add_attribute_fuser("phone_raw", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("phone_e164", prefer_higher_trust, trust_map=trust_map)

strategy.add_attribute_fuser("latitude", average)
strategy.add_attribute_fuser("longitude", average)

strategy.add_attribute_fuser("categories", union, separator="; ")

strategy.add_attribute_fuser("rating", median)
strategy.add_attribute_fuser("rating_count", median)

# === 8. RUN FUSION ===
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
os.makedirs(FUSION_DIR, exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=os.path.join(FUSION_DIR, "debug_fusion_data.jsonl"),
)

fused_result = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=True,
)

# --- Eval ID: extract source ID matching validation prefix(es) for reliable evaluation ---
_EVAL_PREFIXES = ["kaggle380k-", "uber_", "yelp-"]

def _extract_eval_id(row):
    try:
        sources = _ast.literal_eval(str(row.get("_fusion_sources", "[]")))
        if isinstance(sources, (list, tuple)):
            for sid in sources:
                s = str(sid)
                if any(s.startswith(p) for p in _EVAL_PREFIXES):
                    return s
    except Exception:
        pass
    return str(row.get("_id", row.get("id", "")))

fused_result["eval_id"] = fused_result.apply(_extract_eval_id, axis=1)
fused_result.to_csv(os.path.join(FUSION_DIR, "fusion_data.csv"), index=False)