# Pipeline Snapshots

notebook_name=AdaptationPipeline_blocking_matching_extension_Final_Reasoning
matcher_mode=rb

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=76.88%
------------------------------------------------------------

```python
from PyDI.io import load_parquet

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
    longest_string,
    union,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
from dotenv import load_dotenv
import os


# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets with correct paths and names
kaggle_small = load_parquet(
    "output/schema-matching\\kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    "output/schema-matching\\uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    "output/schema-matching\\yelp_small.parquet",
    name="yelp_small",
)

datasets = [kaggle_small, uber_eats_small, yelp_small]


# --------------------------------
# Schema Matching (align to kaggle_small schema)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True,
)

# match kaggle_small -> uber_eats_small
schema_correspondences = matcher.match(kaggle_small, uber_eats_small)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
uber_eats_small = uber_eats_small.rename(columns=rename_map)

# match kaggle_small -> yelp_small
schema_correspondences = matcher.match(kaggle_small, yelp_small)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
yelp_small = yelp_small.rename(columns=rename_map)


# --------------------------------
# Blocking (use pre-computed optimal strategies)
# --------------------------------

print("Performing Blocking")

# ID columns as specified in config
id_column_kaggle = "kaggle380k_id"
id_column_uber = "kaggle380k_id"
id_column_yelp = "kaggle380k_id"

# Ensure the id columns exist; if original id is different, create mapped id columns
# Here we assume the original unique id column is "id" in all three and copy it
if id_column_kaggle not in kaggle_small.columns:
    kaggle_small[id_column_kaggle] = kaggle_small["id"]
if id_column_uber not in uber_eats_small.columns:
    uber_eats_small[id_column_uber] = uber_eats_small["id"]
if id_column_yelp not in yelp_small.columns:
    yelp_small[id_column_yelp] = yelp_small["id"]

# --- kaggle_small vs uber_eats_small: semantic_similarity on [name_norm, city, state] ---

blocker_k2u = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_kaggle,
)

# --- kaggle_small vs yelp_small: exact_match_multi on [phone_e164, postal_code] ---

blocker_k2y = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_kaggle,
)

# --- uber_eats_small vs yelp_small: semantic_similarity on [name_norm, street, city] ---

blocker_u2y = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_uber,
)


# --------------------------------
# Matching (RuleBasedMatcher with provided configuration)
# --------------------------------

print("Matching Entities")


def lower_strip(x):
    return str(x).lower().strip()


def strip(x):
    return str(x).strip()


# --- kaggle_small vs uber_eats_small comparators ---

comparators_k2u = [
    # name_norm, cosine, lower_strip
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # street, jaro_winkler, lower_strip
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # city, jaro_winkler, lower_strip
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # state, jaro_winkler, lower_strip
    StringComparator(
        column="state",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # latitude, numeric
    NumericComparator(
        column="latitude",
        max_difference=0.01,
    ),
    # longitude, numeric
    NumericComparator(
        column="longitude",
        max_difference=0.01,
    ),
    # categories, jaccard, lower_strip, set_jaccard
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

weights_k2u = [0.35, 0.12, 0.08, 0.05, 0.16, 0.16, 0.08]
threshold_k2u = 0.75

# --- kaggle_small vs yelp_small comparators ---

comparators_k2y = [
    # phone_e164, jaro_winkler
    StringComparator(
        column="phone_e164",
        similarity_function="jaro_winkler",
        list_strategy="concatenate",
    ),
    # name_norm, cosine, lower_strip
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # postal_code, jaro_winkler, strip
    StringComparator(
        column="postal_code",
        similarity_function="jaro_winkler",
        preprocess=strip,
        list_strategy="concatenate",
    ),
    # city, jaro_winkler, lower_strip
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # latitude, numeric
    NumericComparator(
        column="latitude",
        max_difference=0.002,
    ),
    # longitude, numeric
    NumericComparator(
        column="longitude",
        max_difference=0.002,
    ),
    # categories, jaccard, lower_strip, set_jaccard
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

weights_k2y = [0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1]
threshold_k2y = 0.75

# --- uber_eats_small vs yelp_small comparators ---

comparators_u2y = [
    # name_norm, jaro_winkler, lower_strip
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # street, jaro_winkler, lower_strip
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # city, jaro_winkler, lower_strip
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # latitude, numeric
    NumericComparator(
        column="latitude",
        max_difference=0.002,
    ),
    # longitude, numeric
    NumericComparator(
        column="longitude",
        max_difference=0.002,
    ),
    # categories, jaccard, lower_strip, set_jaccard
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

weights_u2y = [0.28, 0.16, 0.12, 0.16, 0.16, 0.12]
threshold_u2y = 0.72

# Run Rule-Based Matcher
matcher_rb = RuleBasedMatcher()

rb_correspondences_k2u = matcher_rb.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=weights_k2u,
    threshold=threshold_k2u,
    id_column=id_column_kaggle,
)

rb_correspondences_k2y = matcher_rb.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=weights_k2y,
    threshold=threshold_k2y,
    id_column=id_column_kaggle,
)

rb_correspondences_u2y = matcher_rb.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=weights_u2y,
    threshold=threshold_u2y,
    id_column=id_column_uber,
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

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", longest_string)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
strategy.add_attribute_fuser("postal_code", longest_string)
strategy.add_attribute_fuser("country", longest_string)
strategy.add_attribute_fuser("latitude", longest_string)
strategy.add_attribute_fuser("longitude", longest_string)
strategy.add_attribute_fuser("categories", union)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
    correspondences=all_rb_correspondences,
    id_column=id_column_kaggle,
    include_singletons=False,
)

# --------------------------------
# Write Output (do not change file name)
# --------------------------------

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=12
node_name=execute_pipeline
accuracy_score=78.61%
------------------------------------------------------------

```python
from PyDI.io import load_parquet

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
    longest_string,
    union,
    prefer_higher_trust,
    voting,
    maximum,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
from dotenv import load_dotenv
import os


# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets with correct paths and names
kaggle_small = load_parquet(
    "output/schema-matching\\kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    "output/schema-matching\\uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    "output/schema-matching\\yelp_small.parquet",
    name="yelp_small",
)

datasets = [kaggle_small, uber_eats_small, yelp_small]


# --------------------------------
# Schema Matching (align to kaggle_small schema)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True,
)

# match kaggle_small -> uber_eats_small
schema_correspondences = matcher.match(kaggle_small, uber_eats_small)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
uber_eats_small = uber_eats_small.rename(columns=rename_map)

# match kaggle_small -> yelp_small
schema_correspondences = matcher.match(kaggle_small, yelp_small)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
yelp_small = yelp_small.rename(columns=rename_map)


# --------------------------------
# ID Columns (as specified in config, but keep native ids separate)
# --------------------------------

print("Preparing ID columns")

# Config id columns (used for matching/fusion clusters)
id_column_kaggle = "kaggle380k_id"
id_column_uber = "kaggle380k_id"
id_column_yelp = "kaggle380k_id"

# Preserve native ids for better fusion/evaluation later
if "kaggle380k_id" not in kaggle_small.columns:
    kaggle_small["kaggle380k_id"] = kaggle_small["id"]
kaggle_small["kaggle_native_id"] = kaggle_small["id"]

if "kaggle380k_id" not in uber_eats_small.columns:
    uber_eats_small["kaggle380k_id"] = uber_eats_small["id"]
uber_eats_small["uber_eats_id"] = uber_eats_small["id"]

if "kaggle380k_id" not in yelp_small.columns:
    yelp_small["kaggle380k_id"] = yelp_small["id"]
yelp_small["yelp_id"] = yelp_small["id"]


# --------------------------------
# Blocking (use pre-computed optimal strategies)
# --------------------------------

print("Performing Blocking")

# --- kaggle_small vs uber_eats_small: semantic_similarity on [name_norm, city, state] ---

blocker_k2u = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_kaggle,
)

# --- kaggle_small vs yelp_small: exact_match_multi on [phone_e164, postal_code] ---

blocker_k2y = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_kaggle,
)

# --- uber_eats_small vs yelp_small: semantic_similarity on [name_norm, street, city] ---

blocker_u2y = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_uber,
)


# --------------------------------
# Matching (RuleBasedMatcher with provided configuration)
# --------------------------------

print("Matching Entities")


def lower_strip(x):
    return str(x).lower().strip()


def strip(x):
    return str(x).strip()


# --- kaggle_small vs uber_eats_small comparators ---

comparators_k2u = [
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="state",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="latitude",
        max_difference=0.01,
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.01,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

weights_k2u = [0.35, 0.12, 0.08, 0.05, 0.16, 0.16, 0.08]
threshold_k2u = 0.75

# --- kaggle_small vs yelp_small comparators ---

comparators_k2y = [
    StringComparator(
        column="phone_e164",
        similarity_function="jaro_winkler",
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="postal_code",
        similarity_function="jaro_winkler",
        preprocess=strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
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
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

weights_k2y = [0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1]
threshold_k2y = 0.75

# --- uber_eats_small vs yelp_small comparators ---

comparators_u2y = [
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
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
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

weights_u2y = [0.28, 0.16, 0.12, 0.16, 0.16, 0.12]
threshold_u2y = 0.72

# Run Rule-Based Matcher
matcher_rb = RuleBasedMatcher()

rb_correspondences_k2u = matcher_rb.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=weights_k2u,
    threshold=threshold_k2u,
    id_column=id_column_kaggle,
)

rb_correspondences_k2y = matcher_rb.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=weights_k2y,
    threshold=threshold_k2y,
    id_column=id_column_kaggle,
)

rb_correspondences_u2y = matcher_rb.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=weights_u2y,
    threshold=threshold_u2y,
    id_column=id_column_uber,
)


# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

# Pre-normalize postal_code to improve accuracy
def normalize_postal(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # keep only part before '-' and remove spaces
    return s.split("-")[0].replace(" ", "") or None


for df in [kaggle_small, uber_eats_small, yelp_small]:
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].map(normalize_postal)

# Merge all correspondences
all_rb_correspondences = pd.concat(
    [rb_correspondences_k2u, rb_correspondences_k2y, rb_correspondences_u2y],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# Core business attributes
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", longest_string)
strategy.add_attribute_fuser("street", longest_string)
# Prefer Kaggle for house_number to fix systemic errors
strategy.add_attribute_fuser("house_number", prefer_higher_trust)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
# Use trust-aware fuser and normalized values for postal_code
strategy.add_attribute_fuser("postal_code", prefer_higher_trust)
strategy.add_attribute_fuser("country", longest_string)

# Coordinates: treat as numeric, prefer higher-trust (Kaggle) instead of longest_string
strategy.add_attribute_fuser("latitude", prefer_higher_trust)
strategy.add_attribute_fuser("longitude", prefer_higher_trust)

# Categories: union across sources
strategy.add_attribute_fuser("categories", union)

# Contact and URLs: prefer higher trust (Kaggle, then others)
strategy.add_attribute_fuser("phone_raw", prefer_higher_trust)
strategy.add_attribute_fuser("phone_e164", prefer_higher_trust)
strategy.add_attribute_fuser("website", prefer_higher_trust)
strategy.add_attribute_fuser("map_url", prefer_higher_trust)

# Ratings: prefer Kaggle if present, otherwise maximum across sources
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("rating_count", maximum)

# ID / provenance fields: prefer Kaggle where available
strategy.add_attribute_fuser("kaggle380k_id", prefer_higher_trust)
strategy.add_attribute_fuser("kaggle_native_id", prefer_higher_trust)
strategy.add_attribute_fuser("yelp_id", prefer_higher_trust)
strategy.add_attribute_fuser("uber_eats_id", prefer_higher_trust)
strategy.add_attribute_fuser("source", voting)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
    correspondences=all_rb_correspondences,
    id_column=id_column_kaggle,
    include_singletons=True,  # include singletons to avoid missing fused values
)

# --------------------------------
# Write Output (do not change file name)
# --------------------------------

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=18
node_name=execute_pipeline
accuracy_score=71.10%
------------------------------------------------------------

```python
from PyDI.io import load_parquet

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
    longest_string,
    union,
    prefer_higher_trust,
    voting,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
from dotenv import load_dotenv
import os
import re


# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets with correct paths and names
kaggle_small = load_parquet(
    "output/schema-matching\\kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    "output/schema-matching\\uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    "output/schema-matching\\yelp_small.parquet",
    name="yelp_small",
)

datasets = [kaggle_small, uber_eats_small, yelp_small]


# --------------------------------
# Schema Matching (align to kaggle_small schema)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True,
)

# match kaggle_small -> uber_eats_small
schema_correspondences = matcher.match(kaggle_small, uber_eats_small)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
uber_eats_small = uber_eats_small.rename(columns=rename_map)

# match kaggle_small -> yelp_small
schema_correspondences = matcher.match(kaggle_small, yelp_small)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
yelp_small = yelp_small.rename(columns=rename_map)


# --------------------------------
# ID Columns and Source Labels
# --------------------------------

print("Preparing ID and source columns")

# Config id columns (used for matching/fusion clusters)
id_column_kaggle = "kaggle380k_id"
id_column_uber = "kaggle380k_id"
id_column_yelp = "kaggle380k_id"

# Explicit source labels for trust-aware fusers
kaggle_small["__source_label__"] = "kaggle"
uber_eats_small["__source_label__"] = "uber_eats"
yelp_small["__source_label__"] = "yelp"

# Preserve native ids and create canonical Kaggle id used as cluster id
if "kaggle380k_id" not in kaggle_small.columns:
    kaggle_small["kaggle380k_id"] = kaggle_small["id"]
kaggle_small["kaggle_native_id"] = kaggle_small["id"]

if "kaggle380k_id" not in uber_eats_small.columns:
    uber_eats_small["kaggle380k_id"] = uber_eats_small["id"]
uber_eats_small["uber_eats_id"] = uber_eats_small["id"]

if "kaggle380k_id" not in yelp_small.columns:
    yelp_small["kaggle380k_id"] = yelp_small["id"]
yelp_small["yelp_id"] = yelp_small["id"]


# --------------------------------
# Light preprocessing to reduce cosmetic mismatches
# --------------------------------

print("Preprocessing attributes")


def normalize_postal(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return s.split("-")[0].replace(" ", "") or None


def normalize_street_text(s):
    if s is None:
        return None
    text = str(s)
    text = re.sub(r"\bStreet\b", "St", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAvenue\b", "Ave", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDrive\b", "Dr", text, flags=re.IGNORECASE)
    text = text.replace("U.S. ", "US-").replace("US ", "US-")
    return text.strip()


def normalize_phone(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    # keep only digits and leading +
    if s.startswith("+"):
        return "+" + re.sub(r"\D", "", s)
    return "+" + re.sub(r"\D", "", s)


for df in [kaggle_small, uber_eats_small, yelp_small]:
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].map(normalize_postal)
    if "street" in df.columns:
        df["street"] = df["street"].map(normalize_street_text)
    if "address_line1" in df.columns:
        df["address_line1"] = df["address_line1"].map(normalize_street_text)
    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].map(str).map(str.strip)
    if "phone_e164" in df.columns:
        df["phone_e164"] = df["phone_e164"].map(normalize_phone)
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").round(1)


# --------------------------------
# Blocking (use pre-computed optimal strategies)
# --------------------------------

print("Performing Blocking")

# --- kaggle_small vs uber_eats_small: semantic_similarity on [name_norm, city, state] ---

blocker_k2u = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_kaggle,
)

# --- kaggle_small vs yelp_small: exact_match_multi on [phone_e164, postal_code] ---

blocker_k2y = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_kaggle,
)

# --- uber_eats_small vs yelp_small: semantic_similarity on [name_norm, street, city] ---

blocker_u2y = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_uber,
)


# --------------------------------
# Matching (RuleBasedMatcher with provided configuration)
# --------------------------------

print("Matching Entities")


def lower_strip(x):
    return str(x).lower().strip()


def strip(x):
    return str(x).strip()


# --- kaggle_small vs uber_eats_small comparators ---

comparators_k2u = [
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="state",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="latitude",
        max_difference=0.01,
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.01,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

weights_k2u = [0.35, 0.12, 0.08, 0.05, 0.16, 0.16, 0.08]
threshold_k2u = 0.75

# --- kaggle_small vs yelp_small comparators ---

comparators_k2y = [
    StringComparator(
        column="phone_e164",
        similarity_function="jaro_winkler",
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="postal_code",
        similarity_function="jaro_winkler",
        preprocess=strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
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
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

weights_k2y = [0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1]
threshold_k2y = 0.75

# --- uber_eats_small vs yelp_small comparators ---

comparators_u2y = [
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
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
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

weights_u2y = [0.28, 0.16, 0.12, 0.16, 0.16, 0.12]
threshold_u2y = 0.72

# Run Rule-Based Matcher
matcher_rb = RuleBasedMatcher()

rb_correspondences_k2u = matcher_rb.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=weights_k2u,
    threshold=threshold_k2u,
    id_column=id_column_kaggle,
)

rb_correspondences_k2y = matcher_rb.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=weights_k2y,
    threshold=threshold_k2y,
    id_column=id_column_kaggle,
)

rb_correspondences_u2y = matcher_rb.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=weights_u2y,
    threshold=threshold_u2y,
    id_column=id_column_uber,
)


# --------------------------------
# Custom fusers to improve accuracy
# --------------------------------

def _get_source_labels(cluster_rows):
    # cluster_rows is a list of dict-like rows with __source_label__ column present
    return [row.get("__source_label__") for row in cluster_rows]


def kaggle_first_fuser(values, cluster_rows):
    sources = _get_source_labels(cluster_rows)
    # prefer Kaggle value
    for v, s in zip(values, sources):
        if s == "kaggle" and pd.notna(v):
            return v
    # fallback: first non-null
    for v in values:
        if pd.notna(v):
            return v
    return None


def phone_fuser(values, cluster_rows):
    sources = _get_source_labels(cluster_rows)
    priority = ["kaggle", "yelp", "uber_eats"]
    # prefer by source priority
    for p in priority:
        for v, s in zip(values, sources):
            if s == p and pd.notna(v) and str(v).strip():
                return str(v).strip()
    # fallback: any non-empty
    for v in values:
        if pd.notna(v) and str(v).strip():
            return str(v).strip()
    return None


def rating_fuser(values, cluster_rows):
    sources = _get_source_labels(cluster_rows)
    # prefer Yelp rating (closest to gold), then Kaggle, then Uber
    for pref in ["yelp", "kaggle", "uber_eats"]:
        for v, s in zip(values, sources):
            if s == pref and pd.notna(v):
                return float(v)
    # fallback: average of non-null ratings if any
    numeric_vals = [float(v) for v in values if pd.notna(v)]
    if numeric_vals:
        return round(sum(numeric_vals) / len(numeric_vals), 1)
    return None


def canonical_name_fuser(values, cluster_rows):
    sources = _get_source_labels(cluster_rows)
    # prefer Kaggle name_norm to avoid extra tokens from other sources
    for v, s in zip(values, sources):
        if s == "kaggle" and pd.notna(v):
            return v
    # fallback: shortest non-null string (less likely to contain extra location tokens)
    non_null = [str(v) for v in values if pd.notna(v)]
    if not non_null:
        return None
    return min(non_null, key=len)


# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

# Merge all correspondences
all_rb_correspondences = pd.concat(
    [rb_correspondences_k2u, rb_correspondences_k2y, rb_correspondences_u2y],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# Core business attributes
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", canonical_name_fuser)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", kaggle_first_fuser)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
strategy.add_attribute_fuser("postal_code", kaggle_first_fuser)
strategy.add_attribute_fuser("country", longest_string)

# Coordinates: trust-based (Kaggle/Yelp preferred)
strategy.add_attribute_fuser("latitude", kaggle_first_fuser)
strategy.add_attribute_fuser("longitude", kaggle_first_fuser)

# Categories: union across sources
strategy.add_attribute_fuser("categories", union)

# Contact and URLs: trust + fallback
strategy.add_attribute_fuser("phone_raw", phone_fuser)
strategy.add_attribute_fuser("phone_e164", phone_fuser)
strategy.add_attribute_fuser("website", kaggle_first_fuser)
strategy.add_attribute_fuser("map_url", kaggle_first_fuser)

# Ratings: avoid systematic inflation
strategy.add_attribute_fuser("rating", rating_fuser)
strategy.add_attribute_fuser("rating_count", kaggle_first_fuser)

# ID / provenance fields
strategy.add_attribute_fuser("kaggle380k_id", kaggle_first_fuser)
strategy.add_attribute_fuser("kaggle_native_id", kaggle_first_fuser)
strategy.add_attribute_fuser("yelp_id", kaggle_first_fuser)
strategy.add_attribute_fuser("uber_eats_id", kaggle_first_fuser)
strategy.add_attribute_fuser("source", voting)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
    correspondences=all_rb_correspondences,
    id_column=id_column_kaggle,
    include_singletons=True,
)

# Ensure fused _id aligns with Kaggle id when available to stabilize evaluation
if "kaggle380k_id" in rb_fused_standard_blocker.columns and "_id" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["_id"] = rb_fused_standard_blocker["kaggle380k_id"].fillna(
        rb_fused_standard_blocker["_id"]
    )

# --------------------------------
# Write Output (do not change file name)
# --------------------------------

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

