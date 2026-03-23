# Pipeline Snapshots

notebook_name=Agent V & VI
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=55.00%
------------------------------------------------------------

```python
from PyDI.io import load_parquet

from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
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
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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
# Perform Schema Matching (LLM-based matching)
# The resulting columns for all datasets will have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = schema_matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Prepare blocking helper columns
# EmbeddingBlocker expects text columns, so cast numeric blocking columns to string.
# --------------------------------

for df in [amazon_small, goodreads_small, metabooks_small]:
    if "publish_year" in df.columns:
        df["publish_year"] = df["publish_year"].astype("string")

# --------------------------------
# Blocking
# MUST use the precomputed blocking configuration
# semantic_similarity -> EmbeddingBlocker
# --------------------------------

print("Performing Blocking")

blocker_goodreads_small_amazon_small = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_small_amazon_small = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_small_goodreads_small = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching Configuration
# --------------------------------

lower_strip = lambda x: str(x).lower().strip()

comparators_goodreads_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
    ),
]

comparators_metabooks_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
]

comparators_metabooks_small_goodreads_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="genres",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
]

threshold_goodreads_small_amazon_small = 0.66
threshold_metabooks_small_amazon_small = 0.72
threshold_metabooks_small_goodreads_small = 0.68

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_goodreads_small_amazon_small = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_goodreads_small_amazon_small,
    comparators=comparators_goodreads_small_amazon_small,
    weights=[0.42, 0.33, 0.15, 0.1],
    threshold=threshold_goodreads_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_amazon_small = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_metabooks_small_amazon_small,
    comparators=comparators_metabooks_small_amazon_small,
    weights=[0.5, 0.25, 0.15, 0.1],
    threshold=threshold_metabooks_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_goodreads_small = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_metabooks_small_goodreads_small,
    comparators=comparators_metabooks_small_goodreads_small,
    weights=[0.42, 0.23, 0.12, 0.15, 0.08],
    threshold=threshold_metabooks_small_goodreads_small,
    id_column="id",
)

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

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_goodreads_small_amazon_small,
        rb_correspondences_metabooks_small_amazon_small,
        rb_correspondences_metabooks_small_goodreads_small,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publish_year", maximum)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("numratings", maximum)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", maximum)
strategy.add_attribute_fuser("price", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
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
accuracy_score=55.00%
------------------------------------------------------------

```python
from PyDI.io import load_parquet

from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
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
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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
# Perform Schema Matching (LLM-based matching)
# The resulting columns for all datasets will have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = schema_matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Prepare helper columns for blocking only
# EmbeddingBlocker expects text columns, so create string helper columns
# without changing the original numeric publish_year used in matching/fusion.
# --------------------------------

for df in [amazon_small, goodreads_small, metabooks_small]:
    if "publish_year" in df.columns:
        df["publish_year_blocking"] = df["publish_year"].astype("string")

# --------------------------------
# Blocking
# MUST use the precomputed blocking configuration
# semantic_similarity -> EmbeddingBlocker
# --------------------------------

print("Performing Blocking")

blocker_goodreads_small_amazon_small = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year_blocking"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_small_amazon_small = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_small_goodreads_small = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching Configuration
# Updated weights from cluster-analysis feedback
# --------------------------------

lower_strip = lambda x: str(x).lower().strip()

comparators_goodreads_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
    ),
]

comparators_metabooks_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
]

comparators_metabooks_small_goodreads_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="genres",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
]

threshold_goodreads_small_amazon_small = 0.66
threshold_metabooks_small_amazon_small = 0.72
threshold_metabooks_small_goodreads_small = 0.68

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_goodreads_small_amazon_small = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_goodreads_small_amazon_small,
    comparators=comparators_goodreads_small_amazon_small,
    weights=[0.52, 0.26, 0.08, 0.14],
    threshold=threshold_goodreads_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_amazon_small = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_metabooks_small_amazon_small,
    comparators=comparators_metabooks_small_amazon_small,
    weights=[0.58, 0.19, 0.05, 0.18],
    threshold=threshold_metabooks_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_goodreads_small = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_metabooks_small_goodreads_small,
    comparators=comparators_metabooks_small_goodreads_small,
    weights=[0.5, 0.16, 0.06, 0.18, 0.1],
    threshold=threshold_metabooks_small_goodreads_small,
    id_column="id",
)

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

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_goodreads_small_amazon_small,
        rb_correspondences_metabooks_small_amazon_small,
        rb_correspondences_metabooks_small_goodreads_small,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publish_year", maximum)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("numratings", maximum)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", maximum)
strategy.add_attribute_fuser("price", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
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
accuracy_score=55.00%
------------------------------------------------------------

```python
from PyDI.io import load_parquet

from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
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
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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
# Perform Schema Matching (LLM-based matching)
# The resulting columns for all datasets will have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = schema_matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Prepare helper columns for blocking only
# EmbeddingBlocker expects text columns, so create string helper columns
# without changing the original numeric publish_year used in matching/fusion.
# --------------------------------

for df in [amazon_small, goodreads_small, metabooks_small]:
    if "publish_year" in df.columns:
        df["publish_year_blocking"] = df["publish_year"].astype("string")

# --------------------------------
# Blocking
# MUST use the precomputed blocking configuration
# semantic_similarity -> EmbeddingBlocker
# --------------------------------

print("Performing Blocking")

blocker_goodreads_small_amazon_small = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year_blocking"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_small_amazon_small = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_small_goodreads_small = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching Configuration
# Updated weights from latest cluster-analysis feedback
# --------------------------------

lower_strip = lambda x: str(x).lower().strip()

comparators_goodreads_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
    ),
]

comparators_metabooks_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
]

comparators_metabooks_small_goodreads_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="genres",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
]

threshold_goodreads_small_amazon_small = 0.66
threshold_metabooks_small_amazon_small = 0.72
threshold_metabooks_small_goodreads_small = 0.68

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_goodreads_small_amazon_small = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_goodreads_small_amazon_small,
    comparators=comparators_goodreads_small_amazon_small,
    weights=[0.64, 0.17, 0.04, 0.15],
    threshold=threshold_goodreads_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_amazon_small = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_metabooks_small_amazon_small,
    comparators=comparators_metabooks_small_amazon_small,
    weights=[0.62, 0.14, 0.04, 0.2],
    threshold=threshold_metabooks_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_goodreads_small = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_metabooks_small_goodreads_small,
    comparators=comparators_metabooks_small_goodreads_small,
    weights=[0.5, 0.16, 0.06, 0.18, 0.1],
    threshold=threshold_metabooks_small_goodreads_small,
    id_column="id",
)

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

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_goodreads_small_amazon_small,
        rb_correspondences_metabooks_small_amazon_small,
        rb_correspondences_metabooks_small_goodreads_small,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publish_year", maximum)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("numratings", maximum)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", maximum)
strategy.add_attribute_fuser("price", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

os.makedirs("output/data_fusion", exist_ok=True)
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

from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
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
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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
# Perform Schema Matching (LLM-based matching)
# The resulting columns for all datasets will have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = schema_matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Prepare helper columns for blocking only
# EmbeddingBlocker requires the blocking text_cols to exist in both datasets.
# publish_year is numeric, so create a string helper column for the one pair
# where publish_year is part of the blocking configuration.
# --------------------------------

for df in [amazon_small, goodreads_small, metabooks_small]:
    if "publish_year" in df.columns:
        df["publish_year_blocking"] = df["publish_year"].astype("string")

# --------------------------------
# Blocking
# MUST use the precomputed blocking configuration
# semantic_similarity -> EmbeddingBlocker
# --------------------------------

print("Performing Blocking")

blocker_goodreads_small_amazon_small = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year_blocking"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_small_amazon_small = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_small_goodreads_small = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching Configuration
# Use the supplied matching configuration and the latest updated weights
# from cluster-analysis feedback.
# preprocess mapping: lower_strip -> lambda x: str(x).lower().strip()
# --------------------------------

lower_strip = lambda x: str(x).lower().strip()

comparators_goodreads_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
    ),
]

comparators_metabooks_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
]

comparators_metabooks_small_goodreads_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="genres",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
]

threshold_goodreads_small_amazon_small = 0.66
threshold_metabooks_small_amazon_small = 0.72
threshold_metabooks_small_goodreads_small = 0.68

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_goodreads_small_amazon_small = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_goodreads_small_amazon_small,
    comparators=comparators_goodreads_small_amazon_small,
    weights=[0.64, 0.17, 0.04, 0.15],
    threshold=threshold_goodreads_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_amazon_small = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_metabooks_small_amazon_small,
    comparators=comparators_metabooks_small_amazon_small,
    weights=[0.62, 0.14, 0.04, 0.2],
    threshold=threshold_metabooks_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_goodreads_small = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_metabooks_small_goodreads_small,
    comparators=comparators_metabooks_small_goodreads_small,
    weights=[0.5, 0.16, 0.06, 0.18, 0.1],
    threshold=threshold_metabooks_small_goodreads_small,
    id_column="id",
)

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

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_goodreads_small_amazon_small,
        rb_correspondences_metabooks_small_amazon_small,
        rb_correspondences_metabooks_small_goodreads_small,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publish_year", maximum)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("numratings", maximum)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", maximum)
strategy.add_attribute_fuser("price", maximum)

os.makedirs("output/data_fusion", exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
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

from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
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
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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
# Perform Schema Matching (LLM-based matching)
# The resulting columns for all datasets will have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = schema_matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Prepare helper columns for blocking only
# EmbeddingBlocker requires text columns shared across both datasets.
# publish_year is numeric, so create a string helper column for blocking.
# --------------------------------

for df in [amazon_small, goodreads_small, metabooks_small]:
    if "publish_year" in df.columns:
        df["publish_year_blocking"] = df["publish_year"].astype("string")

# --------------------------------
# Blocking
# MUST use the precomputed blocking configuration
# semantic_similarity -> EmbeddingBlocker
# --------------------------------

print("Performing Blocking")

blocker_goodreads_small_amazon_small = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year_blocking"],
    id_column="id",
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

blocker_metabooks_small_amazon_small = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publisher"],
    id_column="id",
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

blocker_metabooks_small_goodreads_small = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publisher"],
    id_column="id",
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# Matching Configuration
# Use the supplied matching configuration and latest cluster-analysis updates
# --------------------------------

lower_strip = lambda x: str(x).lower().strip()

comparators_goodreads_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
    ),
]

comparators_metabooks_small_amazon_small = [
    StringComparator(
        column="title",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
]

comparators_metabooks_small_goodreads_small = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="genres",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
]

threshold_goodreads_small_amazon_small = 0.66
threshold_metabooks_small_amazon_small = 0.72
threshold_metabooks_small_goodreads_small = 0.68

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_goodreads_small_amazon_small = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_goodreads_small_amazon_small,
    comparators=comparators_goodreads_small_amazon_small,
    weights=[0.64, 0.17, 0.04, 0.15],
    threshold=threshold_goodreads_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_amazon_small = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_metabooks_small_amazon_small,
    comparators=comparators_metabooks_small_amazon_small,
    weights=[0.62, 0.14, 0.04, 0.2],
    threshold=threshold_metabooks_small_amazon_small,
    id_column="id",
)

rb_correspondences_metabooks_small_goodreads_small = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_metabooks_small_goodreads_small,
    comparators=comparators_metabooks_small_goodreads_small,
    weights=[0.5, 0.16, 0.06, 0.18, 0.1],
    threshold=threshold_metabooks_small_goodreads_small,
    id_column="id",
)

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

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_goodreads_small_amazon_small,
        rb_correspondences_metabooks_small_amazon_small,
        rb_correspondences_metabooks_small_goodreads_small,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publish_year", maximum)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("numratings", maximum)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", maximum)
strategy.add_attribute_fuser("price", maximum)

os.makedirs("output/data_fusion", exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 05 END
============================================================

