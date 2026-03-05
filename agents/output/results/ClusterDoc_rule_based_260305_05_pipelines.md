# Pipeline Snapshots

notebook_name=ClusterDoc
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=63.83%
------------------------------------------------------------

```python
from PyDI.io import load_parquet
from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
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
import shutil

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
kaggle_small = load_parquet(
    "output/schema-matching/kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    "output/schema-matching/uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    "output/schema-matching/yelp_small.parquet",
    name="yelp_small",
)

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Schema Matching
# Kaggle schema is the reference; map Uber Eats and Yelp to Kaggle schema
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher_schema = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# Kaggle vs Uber Eats
schema_correspondences = matcher_schema.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

# Kaggle vs Yelp
schema_correspondences = matcher_schema.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# --------------------------------
# Blocking
# MUST use precomputed blocking strategies
# --------------------------------

print("Performing Blocking")

# id columns from config
id_kaggle = "id"
id_uber = "id"
id_yelp = "id"

# Blocking config:
# kaggle_small_uber_eats_small -> semantic_similarity on [name_norm, city, state], top_k=20
blocker_k2u = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    id_column=id_kaggle,
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    output_dir="output/blocking-evaluation",
    batch_size=1000,
)

# kaggle_small_yelp_small -> exact_match_multi on [phone_e164, postal_code]
blocker_k2y = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_kaggle,
)

# uber_eats_small_yelp_small -> semantic_similarity on [name_norm, street, city], top_k=20
blocker_u2y = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    id_column=id_uber,
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    output_dir="output/blocking-evaluation",
    batch_size=1000,
)

# --------------------------------
# Matching configuration (RuleBasedMatcher)
# MUST use provided matching_strategies
# --------------------------------

print("Matching Entities")

matcher = RuleBasedMatcher()

# Preprocess helpers
def lower_strip(x):
    return str(x).lower().strip()

def strip(x):
    return str(x).strip()

# Threshold variables as required
threshold_kaggle_small_uber_eats_small = 0.75
threshold_kaggle_small_yelp_small = 0.75
threshold_uber_eats_small_yelp_small = 0.72

# --- kaggle_small_uber_eats_small comparators ---

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
        method="relative_difference",
        max_difference=0.01,
    ),
    NumericComparator(
        column="longitude",
        method="relative_difference",
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

rb_correspondences_k2u = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=weights_k2u,
    threshold=threshold_kaggle_small_uber_eats_small,
    id_column=id_kaggle,
)

# --- kaggle_small_yelp_small comparators ---

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
        method="relative_difference",
        max_difference=0.002,
    ),
    NumericComparator(
        column="longitude",
        method="relative_difference",
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

rb_correspondences_k2y = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=weights_k2y,
    threshold=threshold_kaggle_small_yelp_small,
    id_column=id_kaggle,
)

# --- uber_eats_small_yelp_small comparators ---

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
        method="relative_difference",
        max_difference=0.002,
    ),
    NumericComparator(
        column="longitude",
        method="relative_difference",
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

rb_correspondences_u2y = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=weights_u2y,
    threshold=threshold_uber_eats_small_yelp_small,
    id_column=id_uber,
)

# --------------------------------
# Save correspondences
# DO NOT change output file names
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_k2u.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)

rb_correspondences_k2y.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)

rb_correspondences_u2y.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
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

# Use Kaggle schema attributes as reference
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
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=9
node_name=execute_pipeline
accuracy_score=45.06%
------------------------------------------------------------

```python
from PyDI.io import load_parquet
from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
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
import shutil

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
kaggle_small = load_parquet(
    "output/schema-matching/kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    "output/schema-matching/uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    "output/schema-matching/yelp_small.parquet",
    name="yelp_small",
)

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Schema Matching
# Kaggle schema is the reference; map Uber Eats and Yelp to Kaggle schema
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher_schema = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# Kaggle vs Uber Eats
schema_correspondences = matcher_schema.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

# Kaggle vs Yelp
schema_correspondences = matcher_schema.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# --------------------------------
# Blocking
# MUST use precomputed blocking strategies
# --------------------------------

print("Performing Blocking")

# id columns from config
id_kaggle = "id"
id_uber = "id"
id_yelp = "id"

# Blocking config:
# kaggle_small_uber_eats_small -> semantic_similarity on [name_norm, city, state], top_k=20
blocker_k2u = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    id_column=id_kaggle,
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    output_dir="output/blocking-evaluation",
    batch_size=1000,
)

# kaggle_small_yelp_small -> exact_match_multi on [phone_e164, postal_code]
blocker_k2y = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_kaggle,
)

# uber_eats_small_yelp_small -> semantic_similarity on [name_norm, street, city], top_k=20
blocker_u2y = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    id_column=id_uber,
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    output_dir="output/blocking-evaluation",
    batch_size=1000,
)

# --------------------------------
# Matching configuration (RuleBasedMatcher)
# MUST use provided matching_strategies (including updated weights)
# --------------------------------

print("Matching Entities")

matcher = RuleBasedMatcher()

# Preprocess helpers
def lower_strip(x):
    return str(x).lower().strip()

def strip(x):
    return str(x).strip()

# Threshold variables as required
threshold_kaggle_small_uber_eats_small = 0.75
threshold_kaggle_small_yelp_small = 0.75
threshold_uber_eats_small_yelp_small = 0.72

# --- kaggle_small_uber_eats_small comparators ---

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

# Updated weights from matching configuration / cluster analysis
weights_k2u = [0.38, 0.1, 0.06, 0.04, 0.18, 0.18, 0.06]

rb_correspondences_k2u = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=weights_k2u,
    threshold=threshold_kaggle_small_uber_eats_small,
    id_column=id_kaggle,
)

# --- kaggle_small_yelp_small comparators ---

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

rb_correspondences_k2y = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=weights_k2y,
    threshold=threshold_kaggle_small_yelp_small,
    id_column=id_kaggle,
)

# --- uber_eats_small_yelp_small comparators ---

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

# Updated weights from matching configuration / cluster analysis
weights_u2y = [0.32, 0.1, 0.06, 0.2, 0.2, 0.12]

rb_correspondences_u2y = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=weights_u2y,
    threshold=threshold_uber_eats_small_yelp_small,
    id_column=id_uber,
)

# --------------------------------
# Save correspondences
# DO NOT change output file names
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_k2u.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)

rb_correspondences_k2y.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)

rb_correspondences_u2y.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
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

# Use Kaggle schema attributes as reference
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
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=12
node_name=execute_pipeline
accuracy_score=45.06%
------------------------------------------------------------

```python
from PyDI.io import load_parquet
from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
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
import shutil

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
kaggle_small = load_parquet(
    "output/schema-matching/kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    "output/schema-matching/uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    "output/schema-matching/yelp_small.parquet",
    name="yelp_small",
)

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Schema Matching
# Kaggle schema is the reference; map Uber Eats and Yelp to Kaggle schema
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher_schema = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# Kaggle vs Uber Eats
schema_correspondences = matcher_schema.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

# Kaggle vs Yelp
schema_correspondences = matcher_schema.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# --------------------------------
# Blocking
# MUST use precomputed blocking strategies
# --------------------------------

print("Performing Blocking")

# id columns from config
id_kaggle = "id"
id_uber = "id"
id_yelp = "id"

# Blocking config:
# kaggle_small_uber_eats_small -> semantic_similarity on [name_norm, city, state], top_k=20
blocker_k2u = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    id_column=id_kaggle,
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    output_dir="output/blocking-evaluation",
    batch_size=1000,
)

# kaggle_small_yelp_small -> exact_match_multi on [phone_e164, postal_code]
blocker_k2y = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_kaggle,
)

# uber_eats_small_yelp_small -> semantic_similarity on [name_norm, street, city], top_k=20
blocker_u2y = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    id_column=id_uber,
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    output_dir="output/blocking-evaluation",
    batch_size=1000,
)

# --------------------------------
# Matching configuration (RuleBasedMatcher)
# MUST use provided matching_strategies (including updated weights)
# --------------------------------

print("Matching Entities")

matcher = RuleBasedMatcher()

# Preprocess helpers
def lower_strip(x):
    return str(x).lower().strip()


def strip(x):
    return str(x).strip()


# Threshold variables as required
threshold_kaggle_small_uber_eats_small = 0.75
threshold_kaggle_small_yelp_small = 0.75
threshold_uber_eats_small_yelp_small = 0.72

# --- kaggle_small_uber_eats_small comparators ---

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

# Updated weights from matching configuration / cluster analysis
weights_k2u = [0.38, 0.1, 0.06, 0.04, 0.18, 0.18, 0.06]

rb_correspondences_k2u = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=weights_k2u,
    threshold=threshold_kaggle_small_uber_eats_small,
    id_column=id_kaggle,
)

# --- kaggle_small_yelp_small comparators ---

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

rb_correspondences_k2y = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=weights_k2y,
    threshold=threshold_kaggle_small_yelp_small,
    id_column=id_kaggle,
)

# --- uber_eats_small_yelp_small comparators ---

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

# Updated weights from final matching configuration / cluster analysis
weights_u2y = [0.36, 0.08, 0.06, 0.2, 0.2, 0.1]

rb_correspondences_u2y = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=weights_u2y,
    threshold=threshold_uber_eats_small_yelp_small,
    id_column=id_uber,
)

# --------------------------------
# Save correspondences
# DO NOT change output file names
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_k2u.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)

rb_correspondences_k2y.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)

rb_correspondences_u2y.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
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

# Use Kaggle schema attributes as reference
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
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

============================================================
PIPELINE SNAPSHOT 04 START
============================================================
node_index=19
node_name=execute_pipeline
accuracy_score=pending
------------------------------------------------------------

```python
from PyDI.io import load_parquet
from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
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
import shutil

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
kaggle_small = load_parquet(
    "output/schema-matching/kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    "output/schema-matching/uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    "output/schema-matching/yelp_small.parquet",
    name="yelp_small",
)

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Schema Matching
# Kaggle schema is the reference; map Uber Eats and Yelp to Kaggle schema
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher_schema = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# Kaggle vs Uber Eats
schema_correspondences = matcher_schema.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

# Kaggle vs Yelp
schema_correspondences = matcher_schema.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# --------------------------------
# Blocking
# MUST use precomputed blocking strategies
# --------------------------------

print("Performing Blocking")

# id columns from config
id_kaggle = "id"
id_uber = "id"
id_yelp = "id"

# Blocking config:
# kaggle_small_uber_eats_small -> semantic_similarity on [name_norm, city, state], top_k=20
blocker_k2u = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    id_column=id_kaggle,
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    output_dir="output/blocking-evaluation",
    batch_size=1000,
)

# kaggle_small_yelp_small -> exact_match_multi on [phone_e164, postal_code]
blocker_k2y = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_kaggle,
)

# uber_eats_small_yelp_small -> semantic_similarity on [name_norm, street, city], top_k=20
blocker_u2y = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    id_column=id_uber,
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    output_dir="output/blocking-evaluation",
    batch_size=1000,
)

# --------------------------------
# Matching configuration (RuleBasedMatcher)
# MUST use provided matching_strategies (including updated weights)
# --------------------------------

print("Matching Entities")

matcher = RuleBasedMatcher()

# Preprocess helpers
def lower_strip(x):
    return str(x).lower().strip()


def strip(x):
    return str(x).strip()


# Threshold variables as required
threshold_kaggle_small_uber_eats_small = 0.75
threshold_kaggle_small_yelp_small = 0.75
threshold_uber_eats_small_yelp_small = 0.72

# --- kaggle_small_uber_eats_small comparators ---

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

# Updated weights from matching configuration / cluster analysis
weights_k2u = [0.38, 0.1, 0.06, 0.04, 0.18, 0.18, 0.06]

rb_correspondences_k2u = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=weights_k2u,
    threshold=threshold_kaggle_small_uber_eats_small,
    id_column=id_kaggle,
)

# --- kaggle_small_yelp_small comparators ---

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

rb_correspondences_k2y = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=weights_k2y,
    threshold=threshold_kaggle_small_yelp_small,
    id_column=id_kaggle,
)

# --- uber_eats_small_yelp_small comparators ---

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

# Updated weights from final matching configuration / cluster analysis
weights_u2y = [0.36, 0.08, 0.06, 0.2, 0.2, 0.1]

rb_correspondences_u2y = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=weights_u2y,
    threshold=threshold_uber_eats_small_yelp_small,
    id_column=id_uber,
)

# --------------------------------
# Save correspondences
# DO NOT change output file names
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_k2u.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)

rb_correspondences_k2y.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)

rb_correspondences_u2y.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
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

# Use Kaggle schema attributes as reference
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
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 04 END
============================================================

============================================================
PIPELINE SNAPSHOT 05 START
============================================================
node_index=26
node_name=execute_pipeline
accuracy_score=pending
------------------------------------------------------------

```python
from PyDI.io import load_parquet
from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
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
import shutil

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
kaggle_small = load_parquet(
    "output/schema-matching/kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    "output/schema-matching/uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    "output/schema-matching/yelp_small.parquet",
    name="yelp_small",
)

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Schema Matching
# Kaggle schema is the reference; map Uber Eats and Yelp to Kaggle schema
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher_schema = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# Kaggle vs Uber Eats
schema_correspondences = matcher_schema.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

# Kaggle vs Yelp
schema_correspondences = matcher_schema.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# --------------------------------
# Blocking
# MUST use precomputed blocking strategies
# --------------------------------

print("Performing Blocking")

# id columns from config
id_kaggle = "id"
id_uber = "id"
id_yelp = "id"

# Blocking config:
# kaggle_small_uber_eats_small -> semantic_similarity on [name_norm, city, state], top_k=20
blocker_k2u = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    id_column=id_kaggle,
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    output_dir="output/blocking-evaluation",
    batch_size=1000,
)

# kaggle_small_yelp_small -> exact_match_multi on [phone_e164, postal_code]
blocker_k2y = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_kaggle,
)

# uber_eats_small_yelp_small -> semantic_similarity on [name_norm, street, city], top_k=20
blocker_u2y = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    id_column=id_uber,
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    output_dir="output/blocking-evaluation",
    batch_size=1000,
)

# --------------------------------
# Matching configuration (RuleBasedMatcher)
# MUST use provided matching_strategies (including updated weights)
# --------------------------------

print("Matching Entities")

matcher = RuleBasedMatcher()

# Preprocess helpers
def lower_strip(x):
    return str(x).lower().strip()


def strip(x):
    return str(x).strip()


# Threshold variables as required
threshold_kaggle_small_uber_eats_small = 0.75
threshold_kaggle_small_yelp_small = 0.75
threshold_uber_eats_small_yelp_small = 0.72

# --- kaggle_small_uber_eats_small comparators ---

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

# Updated weights from matching configuration / cluster analysis
weights_k2u = [0.38, 0.1, 0.06, 0.04, 0.18, 0.18, 0.06]

rb_correspondences_k2u = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=weights_k2u,
    threshold=threshold_kaggle_small_uber_eats_small,
    id_column=id_kaggle,
)

# --- kaggle_small_yelp_small comparators ---

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

rb_correspondences_k2y = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=weights_k2y,
    threshold=threshold_kaggle_small_yelp_small,
    id_column=id_kaggle,
)

# --- uber_eats_small_yelp_small comparators ---

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

# Updated weights from final matching configuration / cluster analysis
weights_u2y = [0.36, 0.08, 0.06, 0.2, 0.2, 0.1]

rb_correspondences_u2y = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=weights_u2y,
    threshold=threshold_uber_eats_small_yelp_small,
    id_column=id_uber,
)

# --------------------------------
# Save correspondences
# DO NOT change output file names
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_k2u.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)

rb_correspondences_k2y.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)

rb_correspondences_u2y.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
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

# Use Kaggle schema attributes as reference
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
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 05 END
============================================================

