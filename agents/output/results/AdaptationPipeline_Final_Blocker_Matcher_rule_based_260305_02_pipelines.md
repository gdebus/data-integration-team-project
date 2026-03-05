# Pipeline Snapshots

notebook_name=AdaptationPipeline_Final_Blocker_Matcher
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=67.96%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_parquet
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

kaggle_small = load_parquet(
    DATA_DIR + "kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    DATA_DIR + "uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    DATA_DIR + "yelp_small.parquet",
    name="yelp_small",
)

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# CRITICAL INSTRUCTION FOR AGENTS:
# The here implemented schema matching will match the schema of dataset2 and dataset3
# to the schema of dataset1. Therefore, the resulting columns for all datasets will
# have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True
)

# match schema of kaggle_small with uber_eats_small and rename uber_eats_small
schema_correspondences = schema_matcher.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

# match schema of kaggle_small with yelp_small and rename yelp_small
schema_correspondences = schema_matcher.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# --------------------------------
# Blocking (MUST use provided precomputed configuration)
# --------------------------------

print("Performing Blocking")

ID_COL_K = "id"
ID_COL_U = "id"
ID_COL_Y = "id"

# kaggle_small <-> uber_eats_small: exact_match_multi on ['postal_code','house_number'] -> StandardBlocker
blocker_k2u = StandardBlocker(
    kaggle_small, uber_eats_small,
    on=["postal_code", "house_number"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=ID_COL_K,
)

# kaggle_small <-> yelp_small: semantic_similarity on ['name_norm','address_line1','city'] (top_k=15) -> EmbeddingBlocker
blocker_k2y = EmbeddingBlocker(
    kaggle_small, yelp_small,
    text_cols=["name_norm", "address_line1", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=ID_COL_K,
)

# uber_eats_small <-> yelp_small: exact_match_multi on ['postal_code','house_number'] -> StandardBlocker
blocker_u2y = StandardBlocker(
    uber_eats_small, yelp_small,
    on=["postal_code", "house_number"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=ID_COL_U,
)

# --------------------------------
# Matching (MUST use provided precomputed configuration)
# --------------------------------

def _preprocess_fn(name: str):
    if name is None:
        return None
    if name == "lower":
        return lambda x: str(x).lower()
    if name == "strip":
        return lambda x: str(x).strip()
    if name == "lower_strip":
        return lambda x: str(x).lower().strip()
    raise ValueError(f"Unknown preprocess mapping: {name}")

print("Matching Entities")

rbm = RuleBasedMatcher()

# --- kaggle_small <-> uber_eats_small ---
comparators_k2u = [
    StringComparator(
        column="postal_code",
        similarity_function="levenshtein",
        preprocess=_preprocess_fn("strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="house_number",
        similarity_function="levenshtein",
        preprocess=_preprocess_fn("strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="latitude",
        max_difference=0.002,
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.002,
    ),
]

rb_correspondences_k2u = rbm.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=[0.18, 0.18, 0.30, 0.16, 0.09, 0.09],
    threshold=0.78,
    id_column=ID_COL_K,
)

# --- kaggle_small <-> yelp_small ---
comparators_k2y = [
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="address_line1",
        similarity_function="cosine",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="phone_e164",
        similarity_function="levenshtein",
        preprocess=_preprocess_fn("strip"),
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="latitude",
        max_difference=0.002,
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.002,
    ),
]

rb_correspondences_k2y = rbm.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=[0.30, 0.25, 0.10, 0.20, 0.075, 0.075],
    threshold=0.78,
    id_column=ID_COL_K,
)

# --- uber_eats_small <-> yelp_small ---
comparators_u2y = [
    StringComparator(
        column="postal_code",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="house_number",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="cosine",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="latitude",
        max_difference=0.001,
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.001,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="set_jaccard",
    ),
]

rb_correspondences_u2y = rbm.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=[0.18, 0.18, 0.14, 0.20, 0.12, 0.12, 0.06],
    threshold=0.75,
    id_column=ID_COL_U,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_k2u, rb_correspondences_k2y, rb_correspondences_u2y],
    ignore_index=True
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", longest_string)
strategy.add_attribute_fuser("website", longest_string)
strategy.add_attribute_fuser("map_url", longest_string)
strategy.add_attribute_fuser("phone_raw", longest_string)
strategy.add_attribute_fuser("phone_e164", longest_string)
strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", longest_string)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
strategy.add_attribute_fuser("postal_code", longest_string)
strategy.add_attribute_fuser("country", longest_string)
strategy.add_attribute_fuser("latitude", longest_string)
strategy.add_attribute_fuser("longitude", longest_string)
strategy.add_attribute_fuser("categories", union)
strategy.add_attribute_fuser("rating", longest_string)
strategy.add_attribute_fuser("rating_count", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl"
)

rb_fused_standard_blocker = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=11
node_name=execute_pipeline
accuracy_score=66.08%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_parquet
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    maximum,
    voting,
)
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
from dotenv import load_dotenv
import os
import re


# ----------------------------
# Helpers
# ----------------------------

def _preprocess_fn(name: str):
    if name is None:
        return None
    if name == "lower":
        return lambda x: str(x).lower()
    if name == "strip":
        return lambda x: str(x).strip()
    if name == "lower_strip":
        return lambda x: str(x).lower().strip()
    raise ValueError(f"Unknown preprocess mapping: {name}")


def _digits_only(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x)
    d = re.sub(r"\D+", "", s)
    return d if d else None


def _normalize_postal(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    s = s.split("-")[0]  # keep ZIP5 if ZIP+4
    d = re.sub(r"\D+", "", s)
    return d if d else None


def _normalize_house_number(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().lower()
    # Keep hyphens (e.g., "135-25"), remove other punctuation/spaces
    s = re.sub(r"[^\w\-]", "", s)
    return s if s else None


def _ensure_str_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string")


def _clean_for_blocking(df: pd.DataFrame):
    # Normalize critical blocking / matching keys to reduce false mismatches
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].map(_normalize_postal).astype("string")
    if "house_number" in df.columns:
        df["house_number"] = df["house_number"].map(_normalize_house_number).astype("string")
    if "phone_e164" in df.columns:
        # E.164 should be "+<digits>"; normalize to digits only for robust matching
        df["phone_e164"] = df["phone_e164"].map(_digits_only).astype("string")
    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].map(_digits_only).astype("string")
    return df


# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

kaggle_small = load_parquet(DATA_DIR + "kaggle_small.parquet", name="kaggle_small")
uber_eats_small = load_parquet(DATA_DIR + "uber_eats_small.parquet", name="uber_eats_small")
yelp_small = load_parquet(DATA_DIR + "yelp_small.parquet", name="yelp_small")

# Ensure string dtype where appropriate (stability for comparators/blockers)
_ensure_str_cols(
    kaggle_small,
    ["id", "name", "name_norm", "website", "map_url", "phone_raw", "phone_e164",
     "address_line1", "address_line2", "street", "house_number", "city", "state",
     "postal_code", "country", "categories"],
)
_ensure_str_cols(
    uber_eats_small,
    ["id", "name", "name_norm", "phone_raw", "phone_e164",
     "address_line1", "address_line2", "street", "house_number", "city", "state",
     "postal_code", "country", "categories"],
)
_ensure_str_cols(
    yelp_small,
    ["id", "name", "name_norm", "website", "map_url", "phone_raw", "phone_e164",
     "address_line1", "address_line2", "street", "house_number", "city", "state",
     "postal_code", "country", "categories"],
)

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True,
)

# match schema of kaggle_small with uber_eats_small and rename uber_eats_small
schema_correspondences = schema_matcher.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

# match schema of kaggle_small with yelp_small and rename yelp_small
schema_correspondences = schema_matcher.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# Post-schema-match normalization for blocking/matching keys
kaggle_small = _clean_for_blocking(kaggle_small)
uber_eats_small = _clean_for_blocking(uber_eats_small)
yelp_small = _clean_for_blocking(yelp_small)

# --------------------------------
# Blocking (MUST use provided precomputed configuration)
# --------------------------------

print("Performing Blocking")

ID_COL_K = "id"
ID_COL_U = "id"
ID_COL_Y = "id"

# kaggle_small <-> uber_eats_small: exact_match_multi on ['postal_code','house_number'] -> StandardBlocker
blocker_k2u = StandardBlocker(
    kaggle_small,
    uber_eats_small,
    on=["postal_code", "house_number"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=ID_COL_K,
)

# kaggle_small <-> yelp_small: semantic_similarity on ['name_norm','address_line1','city'] (top_k=15) -> EmbeddingBlocker
blocker_k2y = EmbeddingBlocker(
    kaggle_small,
    yelp_small,
    text_cols=["name_norm", "address_line1", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=ID_COL_K,
)

# uber_eats_small <-> yelp_small: exact_match_multi on ['postal_code','house_number'] -> StandardBlocker
blocker_u2y = StandardBlocker(
    uber_eats_small,
    yelp_small,
    on=["postal_code", "house_number"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=ID_COL_U,
)

# --------------------------------
# Matching (MUST use provided precomputed configuration)
# --------------------------------

print("Matching Entities")

rbm = RuleBasedMatcher()

# --- kaggle_small <-> uber_eats_small ---
comparators_k2u = [
    StringComparator(
        column="postal_code",
        similarity_function="levenshtein",
        preprocess=_preprocess_fn("strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="house_number",
        similarity_function="levenshtein",
        preprocess=_preprocess_fn("strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    NumericComparator(column="latitude", max_difference=0.002),
    NumericComparator(column="longitude", max_difference=0.002),
]

rb_correspondences_k2u = rbm.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=[0.18, 0.18, 0.30, 0.16, 0.09, 0.09],
    threshold=0.78,
    id_column=ID_COL_K,
)

# --- kaggle_small <-> yelp_small ---
comparators_k2y = [
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="address_line1",
        similarity_function="cosine",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    # phone_e164 normalized to digits-only above, so strip is enough here
    StringComparator(
        column="phone_e164",
        similarity_function="levenshtein",
        preprocess=_preprocess_fn("strip"),
        list_strategy="concatenate",
    ),
    NumericComparator(column="latitude", max_difference=0.002),
    NumericComparator(column="longitude", max_difference=0.002),
]

rb_correspondences_k2y = rbm.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=[0.30, 0.25, 0.10, 0.20, 0.075, 0.075],
    threshold=0.78,
    id_column=ID_COL_K,
)

# --- uber_eats_small <-> yelp_small ---
comparators_u2y = [
    StringComparator(
        column="postal_code",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="house_number",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="cosine",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    NumericComparator(column="latitude", max_difference=0.001),
    NumericComparator(column="longitude", max_difference=0.001),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="set_jaccard",
    ),
]

rb_correspondences_u2y = rbm.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=[0.18, 0.18, 0.14, 0.20, 0.12, 0.12, 0.06],
    threshold=0.75,
    id_column=ID_COL_U,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_k2u, rb_correspondences_k2y, rb_correspondences_u2y],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# Prefer stable canonical values via voting; keep long text via longest_string; numeric via maximum
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", voting)

strategy.add_attribute_fuser("website", longest_string)
strategy.add_attribute_fuser("map_url", longest_string)

# phone numbers: digits-only normalization already applied; vote to avoid picking malformed/rare variants
strategy.add_attribute_fuser("phone_raw", voting)
strategy.add_attribute_fuser("phone_e164", voting)

strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("street", longest_string)

# critical fixes from evaluation: avoid longest_string for house_number / postal_code
strategy.add_attribute_fuser("house_number", voting)
strategy.add_attribute_fuser("postal_code", voting)

strategy.add_attribute_fuser("city", voting)
strategy.add_attribute_fuser("state", voting)
strategy.add_attribute_fuser("country", voting)

# numeric coordinates
strategy.add_attribute_fuser("latitude", maximum)
strategy.add_attribute_fuser("longitude", maximum)

strategy.add_attribute_fuser("categories", union)

# rating fields: keep numeric maxima (often consistent) and vote for counts if present
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("rating_count", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=16
node_name=execute_pipeline
accuracy_score=67.96%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_parquet
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    maximum,
    voting,
    prefer_higher_trust,
)
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
from dotenv import load_dotenv
import os
import re


# ----------------------------
# Helpers
# ----------------------------

def _preprocess_fn(name: str):
    if name is None:
        return None
    if name == "lower":
        return lambda x: str(x).lower()
    if name == "strip":
        return lambda x: str(x).strip()
    if name == "lower_strip":
        return lambda x: str(x).lower().strip()
    raise ValueError(f"Unknown preprocess mapping: {name}")


def _digits_only(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x)
    d = re.sub(r"\D+", "", s)
    return d if d else None


def _normalize_postal(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    s = s.split("-")[0]  # keep ZIP5 if ZIP+4
    d = re.sub(r"\D+", "", s)
    return d if d else None


def _normalize_house_number(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().lower()
    # keep hyphens and alphanumerics; remove spaces/punct
    s = re.sub(r"[^\w\-]", "", s)
    return s if s else None


def _ensure_str_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string")


def _to_float_or_nan(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _canonicalize_after_fusion(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure fused output columns are in the canonical formats expected by evaluation
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].map(_normalize_postal).astype("string")
    if "house_number" in df.columns:
        df["house_number"] = df["house_number"].map(_normalize_house_number).astype("string")

    # phone_e164 expected as +<digits>; phone_raw expected as digits-only
    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].map(_digits_only).astype("string")

    if "phone_e164" in df.columns:
        def _to_e164(x):
            d = _digits_only(x)
            if d is None:
                return None
            # If already includes country code, keep; otherwise prefix '+' anyway (common eval expects '+')
            return f"+{d}"
        df["phone_e164"] = df["phone_e164"].map(_to_e164).astype("string")

    # keep numeric as float
    for c in ["latitude", "longitude", "rating", "rating_count"]:
        if c in df.columns:
            df[c] = df[c].map(_to_float_or_nan)

    return df


def _clean_for_blocking(df: pd.DataFrame):
    # Normalize critical blocking / matching keys to reduce false mismatches
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].map(_normalize_postal).astype("string")
    if "house_number" in df.columns:
        df["house_number"] = df["house_number"].map(_normalize_house_number).astype("string")

    # Keep digits-only helper cols for matching robustness without breaking expected fused formats
    if "phone_e164" in df.columns:
        df["phone_e164_digits"] = df["phone_e164"].map(_digits_only).astype("string")
    else:
        df["phone_e164_digits"] = pd.Series([None] * len(df), dtype="string")

    if "phone_raw" in df.columns:
        df["phone_raw_digits"] = df["phone_raw"].map(_digits_only).astype("string")
    else:
        df["phone_raw_digits"] = pd.Series([None] * len(df), dtype="string")

    return df


# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

kaggle_small = load_parquet(DATA_DIR + "kaggle_small.parquet", name="kaggle_small")
uber_eats_small = load_parquet(DATA_DIR + "uber_eats_small.parquet", name="uber_eats_small")
yelp_small = load_parquet(DATA_DIR + "yelp_small.parquet", name="yelp_small")

# Ensure string dtype where appropriate (stability for comparators/blockers)
_ensure_str_cols(
    kaggle_small,
    [
        "id", "source", "name", "name_norm", "website", "map_url", "phone_raw", "phone_e164",
        "address_line1", "address_line2", "street", "house_number", "city", "state",
        "postal_code", "country", "categories",
    ],
)
_ensure_str_cols(
    uber_eats_small,
    [
        "id", "source", "name", "name_norm", "website", "map_url", "phone_raw", "phone_e164",
        "address_line1", "address_line2", "street", "house_number", "city", "state",
        "postal_code", "country", "categories",
    ],
)
_ensure_str_cols(
    yelp_small,
    [
        "id", "source", "name", "name_norm", "website", "map_url", "phone_raw", "phone_e164",
        "address_line1", "address_line2", "street", "house_number", "city", "state",
        "postal_code", "country", "categories",
    ],
)

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True,
)

# match schema of kaggle_small with uber_eats_small and rename uber_eats_small
schema_correspondences = schema_matcher.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

# match schema of kaggle_small with yelp_small and rename yelp_small
schema_correspondences = schema_matcher.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# Post-schema-match normalization for blocking/matching keys
kaggle_small = _clean_for_blocking(kaggle_small)
uber_eats_small = _clean_for_blocking(uber_eats_small)
yelp_small = _clean_for_blocking(yelp_small)

# --------------------------------
# Blocking (MUST use provided precomputed configuration)
# --------------------------------

print("Performing Blocking")

ID_COL_K = "id"
ID_COL_U = "id"
ID_COL_Y = "id"

# kaggle_small <-> uber_eats_small: exact_match_multi on ['postal_code','house_number'] -> StandardBlocker
blocker_k2u = StandardBlocker(
    kaggle_small,
    uber_eats_small,
    on=["postal_code", "house_number"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=ID_COL_K,
)

# kaggle_small <-> yelp_small: semantic_similarity on ['name_norm','address_line1','city'] (top_k=15) -> EmbeddingBlocker
blocker_k2y = EmbeddingBlocker(
    kaggle_small,
    yelp_small,
    text_cols=["name_norm", "address_line1", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=ID_COL_K,
)

# uber_eats_small <-> yelp_small: exact_match_multi on ['postal_code','house_number'] -> StandardBlocker
blocker_u2y = StandardBlocker(
    uber_eats_small,
    yelp_small,
    on=["postal_code", "house_number"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=ID_COL_U,
)

# --------------------------------
# Matching (MUST use provided precomputed configuration)
# --------------------------------

print("Matching Entities")

rbm = RuleBasedMatcher()

# --- kaggle_small <-> uber_eats_small ---
comparators_k2u = [
    StringComparator(
        column="postal_code",
        similarity_function="levenshtein",
        preprocess=_preprocess_fn("strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="house_number",
        similarity_function="levenshtein",
        preprocess=_preprocess_fn("strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    NumericComparator(column="latitude", max_difference=0.002),
    NumericComparator(column="longitude", max_difference=0.002),
]

rb_correspondences_k2u = rbm.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=[0.18, 0.18, 0.30, 0.16, 0.09, 0.09],
    threshold=0.78,
    id_column=ID_COL_K,
)

# --- kaggle_small <-> yelp_small ---
comparators_k2y = [
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="address_line1",
        similarity_function="cosine",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    # Use digits-only helper column for robust matching while keeping original phone_e164 intact for fusion
    StringComparator(
        column="phone_e164_digits",
        similarity_function="levenshtein",
        preprocess=_preprocess_fn("strip"),
        list_strategy="concatenate",
    ),
    NumericComparator(column="latitude", max_difference=0.002),
    NumericComparator(column="longitude", max_difference=0.002),
]

rb_correspondences_k2y = rbm.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=[0.30, 0.25, 0.10, 0.20, 0.075, 0.075],
    threshold=0.78,
    id_column=ID_COL_K,
)

# --- uber_eats_small <-> yelp_small ---
comparators_u2y = [
    StringComparator(
        column="postal_code",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="house_number",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="cosine",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    NumericComparator(column="latitude", max_difference=0.001),
    NumericComparator(column="longitude", max_difference=0.001),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="set_jaccard",
    ),
]

rb_correspondences_u2y = rbm.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=[0.18, 0.18, 0.14, 0.20, 0.12, 0.12, 0.06],
    threshold=0.75,
    id_column=ID_COL_U,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_k2u, rb_correspondences_k2y, rb_correspondences_u2y],
    ignore_index=True,
)

# Assign trust to improve source/_id/map_url and other conflicts consistently
for df in [kaggle_small, uber_eats_small, yelp_small]:
    if "source" not in df.columns:
        df["source"] = None

# Heuristic trust: kaggle (rich web/map/phones) > yelp > uber_eats
kaggle_small["trust"] = 3.0
yelp_small["trust"] = 2.0
uber_eats_small["trust"] = 1.0

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# Prefer higher trust for identifiers/source/urls/phones to avoid wrong values from other providers
strategy.add_attribute_fuser("id", prefer_higher_trust)
strategy.add_attribute_fuser("source", prefer_higher_trust)

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", voting)

strategy.add_attribute_fuser("website", prefer_higher_trust)
strategy.add_attribute_fuser("map_url", prefer_higher_trust)

# Keep original phone formats; prefer higher trust rather than voting across different formats
strategy.add_attribute_fuser("phone_raw", prefer_higher_trust)
strategy.add_attribute_fuser("phone_e164", prefer_higher_trust)

strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("street", longest_string)

# critical: stabilize house_number / postal_code (avoid longest_string); prefer higher trust (often canonical)
strategy.add_attribute_fuser("house_number", prefer_higher_trust)
strategy.add_attribute_fuser("postal_code", prefer_higher_trust)

strategy.add_attribute_fuser("city", voting)
strategy.add_attribute_fuser("state", voting)
strategy.add_attribute_fuser("country", voting)

# numeric coordinates
strategy.add_attribute_fuser("latitude", maximum)
strategy.add_attribute_fuser("longitude", maximum)

strategy.add_attribute_fuser("categories", union)

# rating fields
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("rating_count", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# Final canonicalization to match expected evaluation formats
rb_fused_standard_blocker = _canonicalize_after_fusion(rb_fused_standard_blocker)

# write output
rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

============================================================
PIPELINE SNAPSHOT 04 START
============================================================
node_index=23
node_name=execute_pipeline
accuracy_score=67.08%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_parquet
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    maximum,
    voting,
    prefer_higher_trust,
)
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
from dotenv import load_dotenv
import os
import re


# ----------------------------
# Helpers
# ----------------------------

def _preprocess_fn(name: str):
    if name is None:
        return None
    if name == "lower":
        return lambda x: str(x).lower()
    if name == "strip":
        return lambda x: str(x).strip()
    if name == "lower_strip":
        return lambda x: str(x).lower().strip()
    raise ValueError(f"Unknown preprocess mapping: {name}")


def _is_na(x) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, pd._libs.missing.NAType))


def _digits_only(x):
    if _is_na(x):
        return None
    d = re.sub(r"\D+", "", str(x))
    return d if d else None


def _normalize_postal(x):
    if _is_na(x):
        return None
    s = str(x).strip()
    s = s.split("-")[0]  # keep ZIP5 if ZIP+4
    d = re.sub(r"\D+", "", s)
    return d if d else None


def _normalize_house_number(x):
    if _is_na(x):
        return None
    s = str(x).strip().lower()
    # keep hyphens and alphanumerics; remove spaces/punct
    s = re.sub(r"[^\w\-]", "", s)
    return s if s else None


def _ensure_str_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string")


def _to_float_or_nan(x):
    if _is_na(x):
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _choose_first_nonnull(values):
    """Conflict resolver: first non-null/non-empty string (left-to-right)"""
    for v in values:
        if _is_na(v):
            continue
        if isinstance(v, str):
            if v.strip() == "" or v.strip().lower() == "nan":
                continue
            return v
        return v
    return None


def _prefer_nonnull_then_higher_trust(values, trusts):
    """
    Conflict resolver for urls/phones/ids:
    choose highest-trust among non-null/non-empty values.
    """
    best_v = None
    best_t = float("-inf")
    for v, t in zip(values, trusts):
        if _is_na(v):
            continue
        if isinstance(v, str):
            sv = v.strip()
            if sv == "" or sv.lower() == "nan":
                continue
            v = sv
        if t is None or (isinstance(t, float) and pd.isna(t)):
            t = 0.0
        if best_v is None or float(t) > best_t:
            best_v = v
            best_t = float(t)
    return best_v


def _clean_for_blocking(df: pd.DataFrame):
    # Normalize critical blocking / matching keys
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].map(_normalize_postal).astype("string")
    if "house_number" in df.columns:
        df["house_number"] = df["house_number"].map(_normalize_house_number).astype("string")

    # Normalize phones (needed for BOTH matching + fusion accuracy)
    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].map(_digits_only).astype("string")
    else:
        df["phone_raw"] = pd.Series([None] * len(df), dtype="string")

    if "phone_e164" in df.columns:
        def _norm_e164(x):
            d = _digits_only(x)
            if d is None:
                return None
            # ensure + prefix; keep digits as-is
            return f"+{d}"
        df["phone_e164"] = df["phone_e164"].map(_norm_e164).astype("string")
    else:
        df["phone_e164"] = pd.Series([None] * len(df), dtype="string")

    # Helper for matching robustness (digits-only, no '+')
    df["phone_e164_digits"] = df["phone_e164"].map(_digits_only).astype("string")
    df["phone_raw_digits"] = df["phone_raw"].map(_digits_only).astype("string")

    return df


def _canonicalize_after_fusion(df: pd.DataFrame) -> pd.DataFrame:
    # Canonical formats expected by evaluation
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].map(_normalize_postal).astype("string")
    if "house_number" in df.columns:
        df["house_number"] = df["house_number"].map(_normalize_house_number).astype("string")

    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].map(_digits_only).astype("string")
    if "phone_e164" in df.columns:
        def _to_e164(x):
            d = _digits_only(x)
            return f"+{d}" if d else None
        df["phone_e164"] = df["phone_e164"].map(_to_e164).astype("string")

    for c in ["latitude", "longitude", "rating", "rating_count"]:
        if c in df.columns:
            df[c] = df[c].map(_to_float_or_nan)

    return df


# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

kaggle_small = load_parquet(DATA_DIR + "kaggle_small.parquet", name="kaggle_small")
uber_eats_small = load_parquet(DATA_DIR + "uber_eats_small.parquet", name="uber_eats_small")
yelp_small = load_parquet(DATA_DIR + "yelp_small.parquet", name="yelp_small")

# Ensure stable string dtype where appropriate
cols_str = [
    "id", "source", "name", "name_norm", "website", "map_url", "phone_raw", "phone_e164",
    "address_line1", "address_line2", "street", "house_number", "city", "state",
    "postal_code", "country", "categories",
]
_ensure_str_cols(kaggle_small, cols_str)
_ensure_str_cols(uber_eats_small, cols_str)
_ensure_str_cols(yelp_small, cols_str)

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True,
)

schema_correspondences = schema_matcher.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# Post-schema-match normalization for blocking/matching/fusion
kaggle_small = _clean_for_blocking(kaggle_small)
uber_eats_small = _clean_for_blocking(uber_eats_small)
yelp_small = _clean_for_blocking(yelp_small)

# --------------------------------
# Blocking (MUST use provided precomputed configuration)
# --------------------------------

print("Performing Blocking")

ID_COL_K = "id"
ID_COL_U = "id"
ID_COL_Y = "id"

blocker_k2u = StandardBlocker(
    kaggle_small,
    uber_eats_small,
    on=["postal_code", "house_number"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=ID_COL_K,
)

blocker_k2y = EmbeddingBlocker(
    kaggle_small,
    yelp_small,
    text_cols=["name_norm", "address_line1", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=ID_COL_K,
)

blocker_u2y = StandardBlocker(
    uber_eats_small,
    yelp_small,
    on=["postal_code", "house_number"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=ID_COL_U,
)

# --------------------------------
# Matching (MUST use provided precomputed configuration)
# --------------------------------

print("Matching Entities")

rbm = RuleBasedMatcher()

comparators_k2u = [
    StringComparator(
        column="postal_code",
        similarity_function="levenshtein",
        preprocess=_preprocess_fn("strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="house_number",
        similarity_function="levenshtein",
        preprocess=_preprocess_fn("strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    NumericComparator(column="latitude", max_difference=0.002),
    NumericComparator(column="longitude", max_difference=0.002),
]

rb_correspondences_k2u = rbm.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=[0.18, 0.18, 0.30, 0.16, 0.09, 0.09],
    threshold=0.78,
    id_column=ID_COL_K,
)

comparators_k2y = [
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="address_line1",
        similarity_function="cosine",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="phone_e164",
        similarity_function="levenshtein",
        preprocess=_preprocess_fn("strip"),
        list_strategy="concatenate",
    ),
    NumericComparator(column="latitude", max_difference=0.002),
    NumericComparator(column="longitude", max_difference=0.002),
]

rb_correspondences_k2y = rbm.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=[0.30, 0.25, 0.10, 0.20, 0.075, 0.075],
    threshold=0.78,
    id_column=ID_COL_K,
)

comparators_u2y = [
    StringComparator(
        column="postal_code",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="house_number",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="cosine",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    NumericComparator(column="latitude", max_difference=0.001),
    NumericComparator(column="longitude", max_difference=0.001),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="set_jaccard",
    ),
]

rb_correspondences_u2y = rbm.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=[0.18, 0.18, 0.14, 0.20, 0.12, 0.12, 0.06],
    threshold=0.75,
    id_column=ID_COL_U,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_k2u, rb_correspondences_k2y, rb_correspondences_u2y],
    ignore_index=True,
)

# Trust: kaggle > yelp > uber_eats
kaggle_small["trust"] = 3.0
yelp_small["trust"] = 2.0
uber_eats_small["trust"] = 1.0

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# Prefer higher trust for key identity / provenance
strategy.add_attribute_fuser("id", prefer_higher_trust)
strategy.add_attribute_fuser("source", prefer_higher_trust)

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", voting)

strategy.add_attribute_fuser("website", _prefer_nonnull_then_higher_trust)
strategy.add_attribute_fuser("map_url", _prefer_nonnull_then_higher_trust)

# Phones: normalize earlier; then pick highest trust non-null
strategy.add_attribute_fuser("phone_raw", _prefer_nonnull_then_higher_trust)
strategy.add_attribute_fuser("phone_e164", _prefer_nonnull_then_higher_trust)

strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("street", longest_string)

# Critical problematic fields in eval: keep canonical and stable -> highest-trust non-null
strategy.add_attribute_fuser("house_number", _prefer_nonnull_then_higher_trust)
strategy.add_attribute_fuser("postal_code", _prefer_nonnull_then_higher_trust)

strategy.add_attribute_fuser("city", voting)
strategy.add_attribute_fuser("state", voting)
strategy.add_attribute_fuser("country", voting)

# numeric coordinates (keep as before)
strategy.add_attribute_fuser("latitude", maximum)
strategy.add_attribute_fuser("longitude", maximum)

strategy.add_attribute_fuser("categories", union)

strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("rating_count", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker = _canonicalize_after_fusion(rb_fused_standard_blocker)

# write output
rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 04 END
============================================================

============================================================
PIPELINE SNAPSHOT 05 START
============================================================
node_index=30
node_name=execute_pipeline
accuracy_score=64.96%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_parquet
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    maximum,
    voting,
    prefer_higher_trust,
)
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
from dotenv import load_dotenv
import os
import re


# ----------------------------
# Helpers
# ----------------------------

def _preprocess_fn(name: str):
    if name is None:
        return None
    if name == "lower":
        return lambda x: str(x).lower()
    if name == "strip":
        return lambda x: str(x).strip()
    if name == "lower_strip":
        return lambda x: str(x).lower().strip()
    raise ValueError(f"Unknown preprocess mapping: {name}")


def _is_na(x) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _nonempty_str_or_none(x):
    if _is_na(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan" or s.lower() == "none":
        return None
    return s


def _digits_only(x):
    s = _nonempty_str_or_none(x)
    if s is None:
        return None
    d = re.sub(r"\D+", "", s)
    return d if d else None


def _normalize_postal(x):
    s = _nonempty_str_or_none(x)
    if s is None:
        return None
    s = s.split("-")[0]  # keep ZIP5 if ZIP+4
    d = re.sub(r"\D+", "", s)
    return d if d else None


def _normalize_house_number(x):
    s = _nonempty_str_or_none(x)
    if s is None:
        return None
    s = s.lower()
    # keep hyphens and alphanumerics; remove spaces/punct
    s = re.sub(r"[^\w\-]", "", s)
    return s if s else None


def _ensure_str_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string")


def _to_float_or_nan(x):
    if _is_na(x):
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _prefer_nonnull_then_higher_trust(values, trusts):
    """
    Conflict resolver: choose highest-trust among non-null/non-empty values.
    """
    best_v = None
    best_t = float("-inf")
    for v, t in zip(values, trusts):
        v = _nonempty_str_or_none(v) if isinstance(v, str) or v is None else v
        if _is_na(v):
            continue
        if t is None or (isinstance(t, float) and pd.isna(t)):
            t = 0.0
        if best_v is None or float(t) > best_t:
            best_v = v
            best_t = float(t)
    return best_v


def _prefer_longest_nonnull(values, trusts=None):
    """
    Conflict resolver: pick the most informative string (longest) among non-null values.
    (Used for addresses/URLs where longer often contains full canonical info.)
    """
    best = None
    best_len = -1
    for v in values:
        s = _nonempty_str_or_none(v)
        if s is None:
            continue
        if len(s) > best_len:
            best = s
            best_len = len(s)
    return best


def _best_phone_raw(values, trusts):
    """
    Prefer digits-only phone with plausible length; break ties by trust.
    """
    best_v = None
    best_score = (-1, float("-inf"))  # (len, trust)
    for v, t in zip(values, trusts):
        d = _digits_only(v)
        if d is None:
            continue
        # prefer 10-15 digits (typical)
        l = len(d)
        if not (7 <= l <= 15):
            continue
        if t is None or (isinstance(t, float) and pd.isna(t)):
            t = 0.0
        score = (l, float(t))
        if best_v is None or score > best_score:
            best_v = d
            best_score = score
    return best_v


def _best_phone_e164(values, trusts):
    """
    Prefer normalized E.164 with '+' prefix and plausible length; break ties by trust.
    """
    best_v = None
    best_score = (-1, float("-inf"))  # (len_digits, trust)
    for v, t in zip(values, trusts):
        d = _digits_only(v)
        if d is None:
            continue
        l = len(d)
        if not (7 <= l <= 15):
            continue
        if t is None or (isinstance(t, float) and pd.isna(t)):
            t = 0.0
        score = (l, float(t))
        if best_v is None or score > best_score:
            best_v = f"+{d}"
            best_score = score
    return best_v


def _clean_for_blocking(df: pd.DataFrame):
    # Normalize critical blocking / matching keys
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].map(_normalize_postal).astype("string")
    if "house_number" in df.columns:
        df["house_number"] = df["house_number"].map(_normalize_house_number).astype("string")

    # Normalize phones (needed for BOTH matching + fusion accuracy)
    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].map(_digits_only).astype("string")
    else:
        df["phone_raw"] = pd.Series([None] * len(df), dtype="string")

    if "phone_e164" in df.columns:
        def _norm_e164(x):
            d = _digits_only(x)
            return f"+{d}" if d else None
        df["phone_e164"] = df["phone_e164"].map(_norm_e164).astype("string")
    else:
        df["phone_e164"] = pd.Series([None] * len(df), dtype="string")

    return df


def _canonicalize_after_fusion(df: pd.DataFrame) -> pd.DataFrame:
    # Canonical formats expected by evaluation
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].map(_normalize_postal).astype("string")
    if "house_number" in df.columns:
        df["house_number"] = df["house_number"].map(_normalize_house_number).astype("string")

    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].map(_digits_only).astype("string")
    if "phone_e164" in df.columns:
        def _to_e164(x):
            d = _digits_only(x)
            return f"+{d}" if d else None
        df["phone_e164"] = df["phone_e164"].map(_to_e164).astype("string")

    for c in ["latitude", "longitude", "rating", "rating_count"]:
        if c in df.columns:
            df[c] = df[c].map(_to_float_or_nan)

    return df


# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

kaggle_small = load_parquet(DATA_DIR + "kaggle_small.parquet", name="kaggle_small")
uber_eats_small = load_parquet(DATA_DIR + "uber_eats_small.parquet", name="uber_eats_small")
yelp_small = load_parquet(DATA_DIR + "yelp_small.parquet", name="yelp_small")

# Ensure stable string dtype where appropriate
cols_str = [
    "id", "source", "name", "name_norm", "website", "map_url", "phone_raw", "phone_e164",
    "address_line1", "address_line2", "street", "house_number", "city", "state",
    "postal_code", "country", "categories",
]
_ensure_str_cols(kaggle_small, cols_str)
_ensure_str_cols(uber_eats_small, cols_str)
_ensure_str_cols(yelp_small, cols_str)

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True,
)

schema_correspondences = schema_matcher.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# Post-schema-match normalization for blocking/matching/fusion
kaggle_small = _clean_for_blocking(kaggle_small)
uber_eats_small = _clean_for_blocking(uber_eats_small)
yelp_small = _clean_for_blocking(yelp_small)

# --------------------------------
# Blocking (MUST use provided precomputed configuration)
# --------------------------------

print("Performing Blocking")

ID_COL_K = "id"
ID_COL_U = "id"
ID_COL_Y = "id"

blocker_k2u = StandardBlocker(
    kaggle_small,
    uber_eats_small,
    on=["postal_code", "house_number"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=ID_COL_K,
)

blocker_k2y = EmbeddingBlocker(
    kaggle_small,
    yelp_small,
    text_cols=["name_norm", "address_line1", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=ID_COL_K,
)

blocker_u2y = StandardBlocker(
    uber_eats_small,
    yelp_small,
    on=["postal_code", "house_number"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=ID_COL_U,
)

# --------------------------------
# Matching (MUST use provided precomputed configuration)
# --------------------------------

print("Matching Entities")

rbm = RuleBasedMatcher()

comparators_k2u = [
    StringComparator(
        column="postal_code",
        similarity_function="levenshtein",
        preprocess=_preprocess_fn("strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="house_number",
        similarity_function="levenshtein",
        preprocess=_preprocess_fn("strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    NumericComparator(column="latitude", max_difference=0.002),
    NumericComparator(column="longitude", max_difference=0.002),
]

rb_correspondences_k2u = rbm.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=[0.18, 0.18, 0.30, 0.16, 0.09, 0.09],
    threshold=0.78,
    id_column=ID_COL_K,
)

comparators_k2y = [
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="address_line1",
        similarity_function="cosine",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="phone_e164",
        similarity_function="levenshtein",
        preprocess=_preprocess_fn("strip"),
        list_strategy="concatenate",
    ),
    NumericComparator(column="latitude", max_difference=0.002),
    NumericComparator(column="longitude", max_difference=0.002),
]

rb_correspondences_k2y = rbm.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=[0.30, 0.25, 0.10, 0.20, 0.075, 0.075],
    threshold=0.78,
    id_column=ID_COL_K,
)

comparators_u2y = [
    StringComparator(
        column="postal_code",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="house_number",
        similarity_function="jaro_winkler",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="cosine",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="concatenate",
    ),
    NumericComparator(column="latitude", max_difference=0.001),
    NumericComparator(column="longitude", max_difference=0.001),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=_preprocess_fn("lower_strip"),
        list_strategy="set_jaccard",
    ),
]

rb_correspondences_u2y = rbm.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=[0.18, 0.18, 0.14, 0.20, 0.12, 0.12, 0.06],
    threshold=0.75,
    id_column=ID_COL_U,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_k2u, rb_correspondences_k2y, rb_correspondences_u2y],
    ignore_index=True,
)

# Trust: kaggle > yelp > uber_eats
kaggle_small["trust"] = 3.0
yelp_small["trust"] = 2.0
uber_eats_small["trust"] = 1.0

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# Keep fused identifier stable and meaningful: choose highest-trust id; for source, union as provenance
strategy.add_attribute_fuser("id", prefer_higher_trust)
strategy.add_attribute_fuser("source", union)

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", voting)

strategy.add_attribute_fuser("website", _prefer_nonnull_then_higher_trust)
strategy.add_attribute_fuser("map_url", _prefer_longest_nonnull)

# Phones: choose best normalized phone (fixes 0-accuracy phones)
strategy.add_attribute_fuser("phone_raw", _best_phone_raw)
strategy.add_attribute_fuser("phone_e164", _best_phone_e164)

strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("street", longest_string)

# Critical problematic fields: prefer highest-trust non-null AFTER normalization
strategy.add_attribute_fuser("house_number", _prefer_nonnull_then_higher_trust)
strategy.add_attribute_fuser("postal_code", _prefer_nonnull_then_higher_trust)

strategy.add_attribute_fuser("city", voting)
strategy.add_attribute_fuser("state", voting)
strategy.add_attribute_fuser("country", voting)

# numeric coordinates
strategy.add_attribute_fuser("latitude", maximum)
strategy.add_attribute_fuser("longitude", maximum)

strategy.add_attribute_fuser("categories", union)

strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("rating_count", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker = _canonicalize_after_fusion(rb_fused_standard_blocker)

# write output
rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 05 END
============================================================

