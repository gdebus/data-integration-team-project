# Pipeline Snapshots

notebook_name=ClusterDoc
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=46.43%
------------------------------------------------------------

```python
from PyDI.io import load_parquet

from PyDI.entitymatching import (
    EmbeddingBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    maximum,
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

DATA_DIR = "output/schema-matching/"

# Define API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
amazon_small = load_parquet(
    DATA_DIR + "amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    DATA_DIR + "goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    DATA_DIR + "metabooks_small.parquet",
    name="metabooks_small",
)

datasets = [amazon_small, goodreads_small, metabooks_small]

# --------------------------------
# Perform Schema Matching (match all to amazon_small schema)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher_schema = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# match schema of amazon_small with goodreads_small
schema_correspondences = matcher_schema.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

# match schema of amazon_small with metabooks_small
schema_correspondences = matcher_schema.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Blocking (use precomputed blocking strategies: semantic_similarity -> EmbeddingBlocker)
# --------------------------------

print("Performing Blocking")

BLOCKING_OUTPUT_DIR = "output/blocking-evaluation"
os.makedirs(BLOCKING_OUTPUT_DIR, exist_ok=True)

# Config from blocking_strategies
# goodreads_small_amazon_small: semantic_similarity on ["title", "author", "publish_year"], top_k=15
blocker_goodreads_small_amazon_small = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=15,
    output_dir=BLOCKING_OUTPUT_DIR,
)

# metabooks_small_amazon_small: semantic_similarity on ["title", "author", "publish_year"], top_k=15
blocker_metabooks_small_amazon_small = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=15,
    output_dir=BLOCKING_OUTPUT_DIR,
)

# metabooks_small_goodreads_small: semantic_similarity on ["title", "author", "publish_year"], top_k=15
blocker_metabooks_small_goodreads_small = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=15,
    output_dir=BLOCKING_OUTPUT_DIR,
)

# --------------------------------
# Matching (RuleBasedMatcher with precomputed comparators)
# --------------------------------

print("Matching Entities")

# Preprocess mapping
def lower_strip(x):
    return str(x).lower().strip()


# ---- Comparators for each pair from matching_strategies ----

# goodreads_small_amazon_small
comparators_goodreads_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]
weights_goodreads_small_amazon_small = [0.45, 0.3, 0.15, 0.1]
threshold_goodreads_small_amazon_small = 0.75

# metabooks_small_amazon_small
comparators_metabooks_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=2.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]
weights_metabooks_small_amazon_small = [0.45, 0.3, 0.05, 0.2]
threshold_metabooks_small_amazon_small = 0.75

# metabooks_small_goodreads_small
comparators_metabooks_small_goodreads_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="page_count",
        max_difference=40.0,
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=3650,
    ),
    StringComparator(
        column="genres",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]
weights_metabooks_small_goodreads_small = [0.35, 0.25, 0.15, 0.1, 0.15]
threshold_metabooks_small_goodreads_small = 0.62

# Initialize Rule-Based Matcher
matcher = RuleBasedMatcher()

rb_correspondences_goodreads_small_amazon_small = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_goodreads_small_amazon_small,
    comparators=comparators_goodreads_small_amazon_small,
    weights=weights_goodreads_small_amazon_small,
    threshold=threshold_goodreads_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_amazon_small = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_metabooks_small_amazon_small,
    comparators=comparators_metabooks_small_amazon_small,
    weights=weights_metabooks_small_amazon_small,
    threshold=threshold_metabooks_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_goodreads_small = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_metabooks_small_goodreads_small,
    comparators=comparators_metabooks_small_goodreads_small,
    weights=weights_metabooks_small_goodreads_small,
    threshold=threshold_metabooks_small_goodreads_small,
    id_column="id",
)

# --------------------------------
# Save correspondences
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_goodreads_small_amazon_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_small_amazon_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_small_goodreads_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_goodreads_small.csv",
    ),
    index=False,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

# merge rule based correspondences
all_rb_correspondences = pd.concat(
    [
        rb_correspondences_goodreads_small_amazon_small,
        rb_correspondences_metabooks_small_amazon_small,
        rb_correspondences_metabooks_small_goodreads_small,
    ],
    ignore_index=True,
)

# define data fusion strategy
strategy = DataFusionStrategy("rule_based_fusion_strategy")

# basic shared attributes
strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("publish_year", maximum)

# attributes that may exist only in some datasets, still safe to add
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("numratings", maximum)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", maximum)
strategy.add_attribute_fuser("price", maximum)

# run fusion
os.makedirs("output/data_fusion", exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
rb_fused.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=9
node_name=execute_pipeline
accuracy_score=46.43%
------------------------------------------------------------

```python
from PyDI.io import load_parquet

from PyDI.entitymatching import (
    EmbeddingBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    maximum,
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

DATA_DIR = "output/schema-matching/"

# Define API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
amazon_small = load_parquet(
    DATA_DIR + "amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    DATA_DIR + "goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    DATA_DIR + "metabooks_small.parquet",
    name="metabooks_small",
)

datasets = [amazon_small, goodreads_small, metabooks_small]

# --------------------------------
# Perform Schema Matching (match all to amazon_small schema)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher_schema = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# match schema of amazon_small with goodreads_small
schema_correspondences = matcher_schema.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

# match schema of amazon_small with metabooks_small
schema_correspondences = matcher_schema.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Blocking (use precomputed blocking strategies: semantic_similarity -> EmbeddingBlocker)
# --------------------------------

print("Performing Blocking")

BLOCKING_OUTPUT_DIR = "output/blocking-evaluation"
os.makedirs(BLOCKING_OUTPUT_DIR, exist_ok=True)

# Config from blocking_strategies
# goodreads_small_amazon_small: semantic_similarity on ["title", "author", "publish_year"], top_k=15
blocker_goodreads_small_amazon_small = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=15,
    output_dir=BLOCKING_OUTPUT_DIR,
)

# metabooks_small_amazon_small: semantic_similarity on ["title", "author", "publish_year"], top_k=15
blocker_metabooks_small_amazon_small = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=15,
    output_dir=BLOCKING_OUTPUT_DIR,
)

# metabooks_small_goodreads_small: semantic_similarity on ["title", "author", "publish_year"], top_k=15
blocker_metabooks_small_goodreads_small = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=15,
    output_dir=BLOCKING_OUTPUT_DIR,
)

# --------------------------------
# Matching (RuleBasedMatcher with updated precomputed comparators)
# --------------------------------

print("Matching Entities")

# Preprocess mapping
def lower_strip(x):
    return str(x).lower().strip()


# ---- Comparators for each pair from matching_strategies ----

# goodreads_small_amazon_small
comparators_goodreads_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]
# updated weights from matching configuration update
weights_goodreads_small_amazon_small = [0.35, 0.3, 0.2, 0.15]
threshold_goodreads_small_amazon_small = 0.75

# metabooks_small_amazon_small
comparators_metabooks_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=2.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]
# updated weights from matching configuration update
weights_metabooks_small_amazon_small = [0.2, 0.25, 0.25, 0.3]
threshold_metabooks_small_amazon_small = 0.75

# metabooks_small_goodreads_small
comparators_metabooks_small_goodreads_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="page_count",
        max_difference=40.0,
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=3650,
    ),
    StringComparator(
        column="genres",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]
# updated weights from matching configuration update
weights_metabooks_small_goodreads_small = [0.25, 0.25, 0.1, 0.25, 0.15]
threshold_metabooks_small_goodreads_small = 0.62

# Initialize Rule-Based Matcher
matcher = RuleBasedMatcher()

rb_correspondences_goodreads_small_amazon_small = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_goodreads_small_amazon_small,
    comparators=comparators_goodreads_small_amazon_small,
    weights=weights_goodreads_small_amazon_small,
    threshold=threshold_goodreads_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_amazon_small = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_metabooks_small_amazon_small,
    comparators=comparators_metabooks_small_amazon_small,
    weights=weights_metabooks_small_amazon_small,
    threshold=threshold_metabooks_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_goodreads_small = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_metabooks_small_goodreads_small,
    comparators=comparators_metabooks_small_goodreads_small,
    weights=weights_metabooks_small_goodreads_small,
    threshold=threshold_metabooks_small_goodreads_small,
    id_column="id",
)

# --------------------------------
# Save correspondences
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_goodreads_small_amazon_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_small_amazon_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_small_goodreads_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_goodreads_small.csv",
    ),
    index=False,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

# merge rule based correspondences
all_rb_correspondences = pd.concat(
    [
        rb_correspondences_goodreads_small_amazon_small,
        rb_correspondences_metabooks_small_amazon_small,
        rb_correspondences_metabooks_small_goodreads_small,
    ],
    ignore_index=True,
)

# define data fusion strategy
strategy = DataFusionStrategy("rule_based_fusion_strategy")

# basic shared attributes
strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("publish_year", maximum)

# attributes that may exist only in some datasets, still safe to add
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("numratings", maximum)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", maximum)
strategy.add_attribute_fuser("price", maximum)

# run fusion
os.makedirs("output/data_fusion", exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
rb_fused.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=12
node_name=execute_pipeline
accuracy_score=46.43%
------------------------------------------------------------

```python
from PyDI.io import load_parquet

from PyDI.entitymatching import (
    EmbeddingBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    maximum,
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

DATA_DIR = "output/schema-matching/"

# Define API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
amazon_small = load_parquet(
    DATA_DIR + "amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    DATA_DIR + "goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    DATA_DIR + "metabooks_small.parquet",
    name="metabooks_small",
)

datasets = [amazon_small, goodreads_small, metabooks_small]

# --------------------------------
# Perform Schema Matching (match all to amazon_small schema)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher_schema = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# match schema of amazon_small with goodreads_small
schema_correspondences = matcher_schema.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

# match schema of amazon_small with metabooks_small
schema_correspondences = matcher_schema.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Blocking (use precomputed blocking strategies: semantic_similarity -> EmbeddingBlocker)
# --------------------------------

print("Performing Blocking")

BLOCKING_OUTPUT_DIR = "output/blocking-evaluation"
os.makedirs(BLOCKING_OUTPUT_DIR, exist_ok=True)

# Config from blocking_strategies
# goodreads_small_amazon_small: semantic_similarity on ["title", "author", "publish_year"], top_k=15
blocker_goodreads_small_amazon_small = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=15,
    output_dir=BLOCKING_OUTPUT_DIR,
)

# metabooks_small_amazon_small: semantic_similarity on ["title", "author", "publish_year"], top_k=15
blocker_metabooks_small_amazon_small = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=15,
    output_dir=BLOCKING_OUTPUT_DIR,
)

# metabooks_small_goodreads_small: semantic_similarity on ["title", "author", "publish_year"], top_k=15
blocker_metabooks_small_goodreads_small = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=15,
    output_dir=BLOCKING_OUTPUT_DIR,
)

# --------------------------------
# Matching (RuleBasedMatcher with precomputed comparators and FINAL updated weights)
# --------------------------------

print("Matching Entities")

# Preprocess mapping
def lower_strip(x):
    return str(x).lower().strip()


# ---- Comparators for each pair from matching_strategies ----

# goodreads_small_amazon_small
comparators_goodreads_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]
# FINAL weights from matching configuration (after last update)
weights_goodreads_small_amazon_small = [0.4, 0.32, 0.16, 0.12]
threshold_goodreads_small_amazon_small = 0.75

# metabooks_small_amazon_small
comparators_metabooks_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=2.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]
# FINAL weights from matching configuration (after last update)
weights_metabooks_small_amazon_small = [0.25, 0.28, 0.17, 0.3]
threshold_metabooks_small_amazon_small = 0.75

# metabooks_small_goodreads_small
comparators_metabooks_small_goodreads_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="page_count",
        max_difference=40.0,
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=3650,
    ),
    StringComparator(
        column="genres",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]
# FINAL weights from matching configuration (after last update)
weights_metabooks_small_goodreads_small = [0.3, 0.32, 0.08, 0.17, 0.13]
threshold_metabooks_small_goodreads_small = 0.62

# Initialize Rule-Based Matcher
matcher = RuleBasedMatcher()

rb_correspondences_goodreads_small_amazon_small = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_goodreads_small_amazon_small,
    comparators=comparators_goodreads_small_amazon_small,
    weights=weights_goodreads_small_amazon_small,
    threshold=threshold_goodreads_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_amazon_small = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_metabooks_small_amazon_small,
    comparators=comparators_metabooks_small_amazon_small,
    weights=weights_metabooks_small_amazon_small,
    threshold=threshold_metabooks_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_goodreads_small = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_metabooks_small_goodreads_small,
    comparators=comparators_metabooks_small_goodreads_small,
    weights=weights_metabooks_small_goodreads_small,
    threshold=threshold_metabooks_small_goodreads_small,
    id_column="id",
)

# --------------------------------
# Save correspondences
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_goodreads_small_amazon_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_small_amazon_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_small_goodreads_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_goodreads_small.csv",
    ),
    index=False,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

# merge rule based correspondences
all_rb_correspondences = pd.concat(
    [
        rb_correspondences_goodreads_small_amazon_small,
        rb_correspondences_metabooks_small_amazon_small,
        rb_correspondences_metabooks_small_goodreads_small,
    ],
    ignore_index=True,
)

# define data fusion strategy
strategy = DataFusionStrategy("rule_based_fusion_strategy")

# basic shared attributes
strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("publish_year", maximum)

# attributes that may exist only in some datasets, still safe to add
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("numratings", maximum)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", maximum)
strategy.add_attribute_fuser("price", maximum)

# run fusion
os.makedirs("output/data_fusion", exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
rb_fused.to_csv("output/data_fusion/fusion_data.csv", index=False)
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
    EmbeddingBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    maximum,
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

DATA_DIR = "output/schema-matching/"

# Define API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
amazon_small = load_parquet(
    DATA_DIR + "amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    DATA_DIR + "goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    DATA_DIR + "metabooks_small.parquet",
    name="metabooks_small",
)

datasets = [amazon_small, goodreads_small, metabooks_small]

# --------------------------------
# Perform Schema Matching (match all to amazon_small schema)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher_schema = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# match schema of amazon_small with goodreads_small
schema_correspondences = matcher_schema.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

# match schema of amazon_small with metabooks_small
schema_correspondences = matcher_schema.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Blocking (use precomputed blocking strategies: semantic_similarity -> EmbeddingBlocker)
# --------------------------------

print("Performing Blocking")

BLOCKING_OUTPUT_DIR = "output/blocking-evaluation"
os.makedirs(BLOCKING_OUTPUT_DIR, exist_ok=True)

# Config from blocking_strategies
# goodreads_small_amazon_small: semantic_similarity on ["title", "author", "publish_year"], top_k=15
blocker_goodreads_small_amazon_small = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=15,
    output_dir=BLOCKING_OUTPUT_DIR,
)

# metabooks_small_amazon_small: semantic_similarity on ["title", "author", "publish_year"], top_k=15
blocker_metabooks_small_amazon_small = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=15,
    output_dir=BLOCKING_OUTPUT_DIR,
)

# metabooks_small_goodreads_small: semantic_similarity on ["title", "author", "publish_year"], top_k=15
blocker_metabooks_small_goodreads_small = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=15,
    output_dir=BLOCKING_OUTPUT_DIR,
)

# --------------------------------
# Matching (RuleBasedMatcher with precomputed comparators and FINAL updated weights)
# --------------------------------

print("Matching Entities")

# Preprocess mapping
def lower_strip(x):
    return str(x).lower().strip()


# ---- Comparators for each pair from matching_strategies ----

# goodreads_small_amazon_small
comparators_goodreads_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]
# FINAL weights from matching configuration (after last update)
weights_goodreads_small_amazon_small = [0.4, 0.32, 0.16, 0.12]
threshold_goodreads_small_amazon_small = 0.75

# metabooks_small_amazon_small
comparators_metabooks_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=2.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]
# FINAL weights from matching configuration (after last update)
weights_metabooks_small_amazon_small = [0.25, 0.28, 0.17, 0.3]
threshold_metabooks_small_amazon_small = 0.75

# metabooks_small_goodreads_small
comparators_metabooks_small_goodreads_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="page_count",
        max_difference=40.0,
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=3650,
    ),
    StringComparator(
        column="genres",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]
# FINAL weights from matching configuration (after last update)
weights_metabooks_small_goodreads_small = [0.3, 0.32, 0.08, 0.17, 0.13]
threshold_metabooks_small_goodreads_small = 0.62

# Initialize Rule-Based Matcher
matcher = RuleBasedMatcher()

rb_correspondences_goodreads_small_amazon_small = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_goodreads_small_amazon_small,
    comparators=comparators_goodreads_small_amazon_small,
    weights=weights_goodreads_small_amazon_small,
    threshold=threshold_goodreads_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_amazon_small = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_metabooks_small_amazon_small,
    comparators=comparators_metabooks_small_amazon_small,
    weights=weights_metabooks_small_amazon_small,
    threshold=threshold_metabooks_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_goodreads_small = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_metabooks_small_goodreads_small,
    comparators=comparators_metabooks_small_goodreads_small,
    weights=weights_metabooks_small_goodreads_small,
    threshold=threshold_metabooks_small_goodreads_small,
    id_column="id",
)

# --------------------------------
# Save correspondences
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_goodreads_small_amazon_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_small_amazon_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_small_goodreads_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_goodreads_small.csv",
    ),
    index=False,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

# merge rule based correspondences
all_rb_correspondences = pd.concat(
    [
        rb_correspondences_goodreads_small_amazon_small,
        rb_correspondences_metabooks_small_amazon_small,
        rb_correspondences_metabooks_small_goodreads_small,
    ],
    ignore_index=True,
)

# define data fusion strategy
strategy = DataFusionStrategy("rule_based_fusion_strategy")

# basic shared attributes
strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("publish_year", maximum)

# attributes that may exist only in some datasets, still safe to add
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("numratings", maximum)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", maximum)
strategy.add_attribute_fuser("price", maximum)

# run fusion
os.makedirs("output/data_fusion", exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
rb_fused.to_csv("output/data_fusion/fusion_data.csv", index=False)
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
    EmbeddingBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    maximum,
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

DATA_DIR = "output/schema-matching/"

# Define API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
amazon_small = load_parquet(
    DATA_DIR + "amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    DATA_DIR + "goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    DATA_DIR + "metabooks_small.parquet",
    name="metabooks_small",
)

datasets = [amazon_small, goodreads_small, metabooks_small]

# --------------------------------
# Perform Schema Matching (match all to amazon_small schema)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher_schema = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# match schema of amazon_small with goodreads_small
schema_correspondences = matcher_schema.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

# match schema of amazon_small with metabooks_small
schema_correspondences = matcher_schema.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Blocking (use precomputed blocking strategies: semantic_similarity -> EmbeddingBlocker)
# --------------------------------

print("Performing Blocking")

BLOCKING_OUTPUT_DIR = "output/blocking-evaluation"
os.makedirs(BLOCKING_OUTPUT_DIR, exist_ok=True)

# Config from blocking_strategies
# goodreads_small_amazon_small: semantic_similarity on ["title", "author", "publish_year"], top_k=15
blocker_goodreads_small_amazon_small = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=15,
    output_dir=BLOCKING_OUTPUT_DIR,
)

# metabooks_small_amazon_small: semantic_similarity on ["title", "author", "publish_year"], top_k=15
blocker_metabooks_small_amazon_small = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=15,
    output_dir=BLOCKING_OUTPUT_DIR,
)

# metabooks_small_goodreads_small: semantic_similarity on ["title", "author", "publish_year"], top_k=15
blocker_metabooks_small_goodreads_small = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publish_year"],
    id_column="id",
    top_k=15,
    output_dir=BLOCKING_OUTPUT_DIR,
)

# --------------------------------
# Matching (RuleBasedMatcher with precomputed comparators and FINAL updated weights)
# --------------------------------

print("Matching Entities")

# Preprocess mapping
def lower_strip(x):
    return str(x).lower().strip()


# ---- Comparators for each pair from matching_strategies ----

# goodreads_small_amazon_small
comparators_goodreads_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]
# FINAL weights from matching configuration (after last update)
weights_goodreads_small_amazon_small = [0.4, 0.32, 0.16, 0.12]
threshold_goodreads_small_amazon_small = 0.75

# metabooks_small_amazon_small
comparators_metabooks_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=2.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]
# FINAL weights from matching configuration (after last update)
weights_metabooks_small_amazon_small = [0.25, 0.28, 0.17, 0.3]
threshold_metabooks_small_amazon_small = 0.75

# metabooks_small_goodreads_small
comparators_metabooks_small_goodreads_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="page_count",
        max_difference=40.0,
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=3650,
    ),
    StringComparator(
        column="genres",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]
# FINAL weights from matching configuration (after last update)
weights_metabooks_small_goodreads_small = [0.3, 0.32, 0.08, 0.17, 0.13]
threshold_metabooks_small_goodreads_small = 0.62

# Initialize Rule-Based Matcher
matcher = RuleBasedMatcher()

rb_correspondences_goodreads_small_amazon_small = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_goodreads_small_amazon_small,
    comparators=comparators_goodreads_small_amazon_small,
    weights=weights_goodreads_small_amazon_small,
    threshold=threshold_goodreads_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_amazon_small = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_metabooks_small_amazon_small,
    comparators=comparators_metabooks_small_amazon_small,
    weights=weights_metabooks_small_amazon_small,
    threshold=threshold_metabooks_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_goodreads_small = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_metabooks_small_goodreads_small,
    comparators=comparators_metabooks_small_goodreads_small,
    weights=weights_metabooks_small_goodreads_small,
    threshold=threshold_metabooks_small_goodreads_small,
    id_column="id",
)

# --------------------------------
# Save correspondences
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_goodreads_small_amazon_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_small_amazon_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_small_goodreads_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_goodreads_small.csv",
    ),
    index=False,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

# merge rule based correspondences
all_rb_correspondences = pd.concat(
    [
        rb_correspondences_goodreads_small_amazon_small,
        rb_correspondences_metabooks_small_amazon_small,
        rb_correspondences_metabooks_small_goodreads_small,
    ],
    ignore_index=True,
)

# define data fusion strategy
strategy = DataFusionStrategy("rule_based_fusion_strategy")

# basic shared attributes
strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("publish_year", maximum)

# attributes that may exist only in some datasets, still safe to add
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("numratings", maximum)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", maximum)
strategy.add_attribute_fuser("price", maximum)

# run fusion
os.makedirs("output/data_fusion", exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
rb_fused.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 05 END
============================================================

