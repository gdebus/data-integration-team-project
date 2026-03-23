# Pipeline Snapshots

notebook_name=Agent V & VI
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=75.47%
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
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union
import pandas as pd
import os
import shutil


# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

kaggle_small = load_parquet(
    DATA_DIR + "output/schema-matching/kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    DATA_DIR + "output/schema-matching/uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    DATA_DIR + "output/schema-matching/yelp_small.parquet",
    name="yelp_small",
)

datasets = [kaggle_small, uber_eats_small, yelp_small]


# --------------------------------
# Schema already aligned in provided schema-matching outputs
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()


# --------------------------------
# Perform Blocking
# MUST use the precomputed blockers provided in the blocking configuration
# --------------------------------

print("Performing Blocking")

blocker_kaggle_small_uber_eats_small = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_kaggle_small_yelp_small = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_uber_eats_small_yelp_small = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration from precomputed optimal settings
# --------------------------------

comparators_kaggle_small_uber_eats_small = [
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

comparators_kaggle_small_yelp_small = [
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
        preprocess=str.strip,
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

comparators_uber_eats_small_yelp_small = [
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

threshold_kaggle_small_uber_eats_small = 0.75
threshold_kaggle_small_yelp_small = 0.75
threshold_uber_eats_small_yelp_small = 0.72

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_kaggle_small_uber_eats_small = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_kaggle_small_uber_eats_small,
    comparators=comparators_kaggle_small_uber_eats_small,
    weights=[0.35, 0.12, 0.08, 0.05, 0.16, 0.16, 0.08],
    threshold=threshold_kaggle_small_uber_eats_small,
    id_column="id",
)

rb_correspondences_kaggle_small_yelp_small = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_kaggle_small_yelp_small,
    comparators=comparators_kaggle_small_yelp_small,
    weights=[0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1],
    threshold=threshold_kaggle_small_yelp_small,
    id_column="id",
)

rb_correspondences_uber_eats_small_yelp_small = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_uber_eats_small_yelp_small,
    comparators=comparators_uber_eats_small_yelp_small,
    weights=[0.28, 0.16, 0.12, 0.16, 0.16, 0.12],
    threshold=threshold_uber_eats_small_yelp_small,
    id_column="id",
)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_kaggle_small_uber_eats_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)

rb_correspondences_kaggle_small_yelp_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)

rb_correspondences_uber_eats_small_yelp_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_kaggle_small_uber_eats_small,
        rb_correspondences_kaggle_small_yelp_small,
        rb_correspondences_uber_eats_small_yelp_small,
    ],
    ignore_index=True,
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
strategy.add_attribute_fuser("source", longest_string)

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
accuracy_score=51.69%
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
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union
import pandas as pd
import os
import shutil


# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

kaggle_small = load_parquet(
    DATA_DIR + "output/schema-matching/kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    DATA_DIR + "output/schema-matching/uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    DATA_DIR + "output/schema-matching/yelp_small.parquet",
    name="yelp_small",
)

datasets = [kaggle_small, uber_eats_small, yelp_small]


# --------------------------------
# Schema already aligned in provided schema-matching outputs
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()


# --------------------------------
# Perform Blocking
# MUST use the precomputed blockers provided in the blocking configuration
# --------------------------------

print("Performing Blocking")

blocker_kaggle_small_uber_eats_small = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_kaggle_small_yelp_small = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_uber_eats_small_yelp_small = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration from precomputed optimal settings
# --------------------------------

comparators_kaggle_small_uber_eats_small = [
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

comparators_kaggle_small_yelp_small = [
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
        preprocess=str.strip,
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

comparators_uber_eats_small_yelp_small = [
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

threshold_kaggle_small_uber_eats_small = 0.75
threshold_kaggle_small_yelp_small = 0.75
threshold_uber_eats_small_yelp_small = 0.72

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_kaggle_small_uber_eats_small = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_kaggle_small_uber_eats_small,
    comparators=comparators_kaggle_small_uber_eats_small,
    weights=[0.35, 0.12, 0.08, 0.05, 0.16, 0.16, 0.08],
    threshold=threshold_kaggle_small_uber_eats_small,
    id_column="id",
)

rb_correspondences_kaggle_small_yelp_small = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_kaggle_small_yelp_small,
    comparators=comparators_kaggle_small_yelp_small,
    weights=[0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1],
    threshold=threshold_kaggle_small_yelp_small,
    id_column="id",
)

rb_correspondences_uber_eats_small_yelp_small = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_uber_eats_small_yelp_small,
    comparators=comparators_uber_eats_small_yelp_small,
    weights=[0.38, 0.08, 0.08, 0.18, 0.18, 0.1],
    threshold=threshold_uber_eats_small_yelp_small,
    id_column="id",
)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_kaggle_small_uber_eats_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)

rb_correspondences_kaggle_small_yelp_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)

rb_correspondences_uber_eats_small_yelp_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_kaggle_small_uber_eats_small,
        rb_correspondences_kaggle_small_yelp_small,
        rb_correspondences_uber_eats_small_yelp_small,
    ],
    ignore_index=True,
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
strategy.add_attribute_fuser("source", longest_string)

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
accuracy_score=51.69%
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
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union
import pandas as pd
import os
import shutil


# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

kaggle_small = load_parquet(
    DATA_DIR + "output/schema-matching/kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    DATA_DIR + "output/schema-matching/uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    DATA_DIR + "output/schema-matching/yelp_small.parquet",
    name="yelp_small",
)

datasets = [kaggle_small, uber_eats_small, yelp_small]


# --------------------------------
# Schema already aligned in provided schema-matching outputs
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()


# --------------------------------
# Perform Blocking
# MUST use the precomputed blockers provided in the blocking configuration
# --------------------------------

print("Performing Blocking")

blocker_kaggle_small_uber_eats_small = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_kaggle_small_yelp_small = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_uber_eats_small_yelp_small = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration from precomputed optimal settings
# --------------------------------

comparators_kaggle_small_uber_eats_small = [
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

comparators_kaggle_small_yelp_small = [
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
        preprocess=str.strip,
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

comparators_uber_eats_small_yelp_small = [
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

threshold_kaggle_small_uber_eats_small = 0.75
threshold_kaggle_small_yelp_small = 0.75
threshold_uber_eats_small_yelp_small = 0.72

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_kaggle_small_uber_eats_small = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_kaggle_small_uber_eats_small,
    comparators=comparators_kaggle_small_uber_eats_small,
    weights=[0.35, 0.12, 0.08, 0.05, 0.16, 0.16, 0.08],
    threshold=threshold_kaggle_small_uber_eats_small,
    id_column="id",
)

rb_correspondences_kaggle_small_yelp_small = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_kaggle_small_yelp_small,
    comparators=comparators_kaggle_small_yelp_small,
    weights=[0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1],
    threshold=threshold_kaggle_small_yelp_small,
    id_column="id",
)

rb_correspondences_uber_eats_small_yelp_small = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_uber_eats_small_yelp_small,
    comparators=comparators_uber_eats_small_yelp_small,
    weights=[0.3, 0.14, 0.1, 0.2, 0.2, 0.06],
    threshold=threshold_uber_eats_small_yelp_small,
    id_column="id",
)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_kaggle_small_uber_eats_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)

rb_correspondences_kaggle_small_yelp_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)

rb_correspondences_uber_eats_small_yelp_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_kaggle_small_uber_eats_small,
        rb_correspondences_kaggle_small_yelp_small,
        rb_correspondences_uber_eats_small_yelp_small,
    ],
    ignore_index=True,
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
strategy.add_attribute_fuser("source", longest_string)

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
    StringComparator,
    NumericComparator,
    RuleBasedMatcher,
)
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union
import pandas as pd
import os
import shutil


# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

kaggle_small = load_parquet(
    DATA_DIR + "output/schema-matching/kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    DATA_DIR + "output/schema-matching/uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    DATA_DIR + "output/schema-matching/yelp_small.parquet",
    name="yelp_small",
)

datasets = [kaggle_small, uber_eats_small, yelp_small]


# --------------------------------
# Schema already aligned in provided schema-matching outputs
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()


# --------------------------------
# Perform Blocking
# MUST use the precomputed blockers provided in the blocking configuration
# --------------------------------

print("Performing Blocking")

blocker_kaggle_small_uber_eats_small = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_kaggle_small_yelp_small = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_uber_eats_small_yelp_small = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration from precomputed optimal settings
# --------------------------------

comparators_kaggle_small_uber_eats_small = [
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

comparators_kaggle_small_yelp_small = [
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
        preprocess=str.strip,
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

comparators_uber_eats_small_yelp_small = [
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

threshold_kaggle_small_uber_eats_small = 0.75
threshold_kaggle_small_yelp_small = 0.75
threshold_uber_eats_small_yelp_small = 0.72

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_kaggle_small_uber_eats_small = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_kaggle_small_uber_eats_small,
    comparators=comparators_kaggle_small_uber_eats_small,
    weights=[0.35, 0.12, 0.08, 0.05, 0.16, 0.16, 0.08],
    threshold=threshold_kaggle_small_uber_eats_small,
    id_column="id",
)

rb_correspondences_kaggle_small_yelp_small = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_kaggle_small_yelp_small,
    comparators=comparators_kaggle_small_yelp_small,
    weights=[0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1],
    threshold=threshold_kaggle_small_yelp_small,
    id_column="id",
)

rb_correspondences_uber_eats_small_yelp_small = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_uber_eats_small_yelp_small,
    comparators=comparators_uber_eats_small_yelp_small,
    weights=[0.3, 0.14, 0.1, 0.2, 0.2, 0.06],
    threshold=threshold_uber_eats_small_yelp_small,
    id_column="id",
)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_kaggle_small_uber_eats_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)

rb_correspondences_kaggle_small_yelp_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)

rb_correspondences_uber_eats_small_yelp_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_kaggle_small_uber_eats_small,
        rb_correspondences_kaggle_small_yelp_small,
        rb_correspondences_uber_eats_small_yelp_small,
    ],
    ignore_index=True,
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
strategy.add_attribute_fuser("source", longest_string)

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
    StringComparator,
    NumericComparator,
    RuleBasedMatcher,
)
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union
import pandas as pd
import os
import shutil


# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

kaggle_small = load_parquet(
    DATA_DIR + "output/schema-matching/kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    DATA_DIR + "output/schema-matching/uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    DATA_DIR + "output/schema-matching/yelp_small.parquet",
    name="yelp_small",
)

datasets = [kaggle_small, uber_eats_small, yelp_small]


# --------------------------------
# Schema already aligned in provided schema-matching outputs
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()


# --------------------------------
# Perform Blocking
# MUST use the precomputed blockers provided in the blocking configuration
# --------------------------------

print("Performing Blocking")

blocker_kaggle_small_uber_eats_small = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_kaggle_small_yelp_small = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_uber_eats_small_yelp_small = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration from precomputed optimal settings
# --------------------------------

comparators_kaggle_small_uber_eats_small = [
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

comparators_kaggle_small_yelp_small = [
    StringComparator(
        column="phone_e164",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
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
        preprocess=str.strip,
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

comparators_uber_eats_small_yelp_small = [
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

threshold_kaggle_small_uber_eats_small = 0.75
threshold_kaggle_small_yelp_small = 0.75
threshold_uber_eats_small_yelp_small = 0.72

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_kaggle_small_uber_eats_small = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_kaggle_small_uber_eats_small,
    comparators=comparators_kaggle_small_uber_eats_small,
    weights=[0.35, 0.12, 0.08, 0.05, 0.16, 0.16, 0.08],
    threshold=threshold_kaggle_small_uber_eats_small,
    id_column="id",
)

rb_correspondences_kaggle_small_yelp_small = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_kaggle_small_yelp_small,
    comparators=comparators_kaggle_small_yelp_small,
    weights=[0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1],
    threshold=threshold_kaggle_small_yelp_small,
    id_column="id",
)

rb_correspondences_uber_eats_small_yelp_small = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_uber_eats_small_yelp_small,
    comparators=comparators_uber_eats_small_yelp_small,
    weights=[0.3, 0.14, 0.1, 0.2, 0.2, 0.06],
    threshold=threshold_uber_eats_small_yelp_small,
    id_column="id",
)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_kaggle_small_uber_eats_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)

rb_correspondences_kaggle_small_yelp_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)

rb_correspondences_uber_eats_small_yelp_small.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_kaggle_small_uber_eats_small,
        rb_correspondences_kaggle_small_yelp_small,
        rb_correspondences_uber_eats_small_yelp_small,
    ],
    ignore_index=True,
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
strategy.add_attribute_fuser("source", longest_string)

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

