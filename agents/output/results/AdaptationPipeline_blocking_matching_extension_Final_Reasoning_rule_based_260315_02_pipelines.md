# Pipeline Snapshots

notebook_name=AdaptationPipeline_blocking_matching_extension_Final_Reasoning
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=66.15%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv, load_xml

from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    TokenBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    shortest_string,
    union,
    prefer_higher_trust,
    voting,
    maximum,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def lower_strip(x):
    return str(x).lower().strip()


# Load datasets
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

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Perform Schema Matching
# The resulting columns for all datasets will have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

schema_correspondences = matcher.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# --------------------------------
# Ensure required blocking/matching id columns exist
# --------------------------------

for df in [kaggle_small, uber_eats_small, yelp_small]:
    if "kaggle380k_id" not in df.columns:
        df["kaggle380k_id"] = df["id"]

# --------------------------------
# Perform Blocking using precomputed blocking configuration
# --------------------------------

print("Performing Blocking")

blocker_kaggle_uber = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="kaggle380k_id",
)

blocker_kaggle_yelp = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="kaggle380k_id",
)

blocker_uber_yelp = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="kaggle380k_id",
)

# --------------------------------
# Matching configuration from precomputed optimal settings
# --------------------------------

comparators_kaggle_uber = [
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

comparators_kaggle_yelp = [
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

comparators_uber_yelp = [
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

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_kaggle_uber = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_kaggle_uber,
    comparators=comparators_kaggle_uber,
    weights=[0.35, 0.12, 0.08, 0.05, 0.16, 0.16, 0.08],
    threshold=0.75,
    id_column="kaggle380k_id",
)

rb_correspondences_kaggle_yelp = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_kaggle_yelp,
    comparators=comparators_kaggle_yelp,
    weights=[0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1],
    threshold=0.75,
    id_column="kaggle380k_id",
)

rb_correspondences_uber_yelp = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_uber_yelp,
    comparators=comparators_uber_yelp,
    weights=[0.28, 0.16, 0.12, 0.16, 0.16, 0.12],
    threshold=0.72,
    id_column="kaggle380k_id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_kaggle_uber.to_csv(
    os.path.join(CORR_DIR, "correspondences_kaggle_small_uber_eats_small.csv"),
    index=False,
)
rb_correspondences_kaggle_yelp.to_csv(
    os.path.join(CORR_DIR, "correspondences_kaggle_small_yelp_small.csv"),
    index=False,
)
rb_correspondences_uber_yelp.to_csv(
    os.path.join(CORR_DIR, "correspondences_uber_eats_small_yelp_small.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_kaggle_uber,
        rb_correspondences_kaggle_yelp,
        rb_correspondences_uber_yelp,
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
    id_column="kaggle380k_id",
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
node_index=12
node_name=execute_pipeline
accuracy_score=51.61%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv, load_xml

from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    TokenBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    shortest_string,
    union,
    prefer_higher_trust,
    voting,
    maximum,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import re

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def lower_strip(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    return s.lower()


def strip_only(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    return s


def normalize_phone(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    s = re.sub(r"\.0$", "", s)
    if s.startswith("+"):
        digits = re.sub(r"\D", "", s)
        return f"+{digits}" if digits else None
    digits = re.sub(r"\D", "", s)
    return digits if digits else None


def normalize_postal_code(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    s = re.sub(r"\.0$", "", s)
    return s


def normalize_house_number(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    s = re.sub(r"\.0$", "", s)
    return s


def normalize_text_field(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    return s


def normalize_name_norm(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"nan", "none", ""}:
        return None
    s = re.sub(r"\s+", " ", s)
    return s


def trusted_or_voting(values):
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s.lower() in {"nan", "none", ""}:
            continue
        cleaned.append(s)
    if not cleaned:
        return None
    counts = pd.Series(cleaned).value_counts()
    top_count = counts.iloc[0]
    top_values = counts[counts == top_count].index.tolist()
    if len(top_values) == 1:
        return top_values[0]
    return min(top_values, key=len)


def canonical_name_norm(values):
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        s = normalize_name_norm(v)
        if s is not None:
            cleaned.append(s)
    if not cleaned:
        return None
    counts = pd.Series(cleaned).value_counts()
    top_count = counts.iloc[0]
    top_values = counts[counts == top_count].index.tolist()
    return min(top_values, key=len)


def canonical_phone(values):
    cleaned = []
    for v in values:
        s = normalize_phone(v)
        if s is not None:
            cleaned.append(s)
    if not cleaned:
        return None
    counts = pd.Series(cleaned).value_counts()
    return counts.index[0]


def canonical_postal_code(values):
    cleaned = []
    for v in values:
        s = normalize_postal_code(v)
        if s is not None:
            cleaned.append(s)
    if not cleaned:
        return None
    counts = pd.Series(cleaned).value_counts()
    return counts.index[0]


def canonical_house_number(values):
    cleaned = []
    for v in values:
        s = normalize_house_number(v)
        if s is not None:
            cleaned.append(s)
    if not cleaned:
        return None
    counts = pd.Series(cleaned).value_counts()
    top_count = counts.iloc[0]
    top_values = counts[counts == top_count].index.tolist()
    return min(top_values, key=len)


def canonical_rating(values):
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        try:
            cleaned.append(float(v))
        except Exception:
            continue
    if not cleaned:
        return None
    rounded = [round(v, 1) for v in cleaned]
    counts = pd.Series(rounded).value_counts()
    return float(counts.index[0])


def canonical_latlon(values):
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        try:
            cleaned.append(float(v))
        except Exception:
            continue
    if not cleaned:
        return None
    return float(np.median(cleaned))


def canonical_source(values):
    priority = {"yelp": 0, "uber_eats": 1, "kaggle_380k": 2}
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s.lower() in {"nan", "none", ""}:
            continue
        cleaned.append(s)
    if not cleaned:
        return None
    return sorted(cleaned, key=lambda x: priority.get(x, 999))[0]


def canonical_cluster_id(values):
    priority = ["yelp-", "uber_eats-", "kaggle380k-"]
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s.lower() in {"nan", "none", ""}:
            continue
        cleaned.append(s)
    if not cleaned:
        return None
    for prefix in priority:
        pref = [v for v in cleaned if v.startswith(prefix)]
        if pref:
            return sorted(pref)[0]
    return sorted(cleaned)[0]


# Load datasets
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

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Perform Schema Matching
# The resulting columns for all datasets will have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

schema_correspondences = matcher.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# --------------------------------
# Normalize critical fields before blocking, matching, and fusion
# --------------------------------

for df in [kaggle_small, uber_eats_small, yelp_small]:
    if "kaggle380k_id" not in df.columns:
        df["kaggle380k_id"] = df["id"]

    if "phone_e164" in df.columns:
        df["phone_e164"] = df["phone_e164"].apply(normalize_phone)
    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].apply(normalize_phone)
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].apply(normalize_postal_code)
    if "house_number" in df.columns:
        df["house_number"] = df["house_number"].apply(normalize_house_number)
    if "name_norm" in df.columns:
        df["name_norm"] = df["name_norm"].apply(normalize_name_norm)

    for col in [
        "name",
        "website",
        "map_url",
        "address_line1",
        "address_line2",
        "street",
        "city",
        "state",
        "country",
        "categories",
        "source",
        "id",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text_field)

# --------------------------------
# Perform Blocking using precomputed blocking configuration
# --------------------------------

print("Performing Blocking")

blocker_kaggle_uber = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="kaggle380k_id",
)

blocker_kaggle_yelp = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="kaggle380k_id",
)

blocker_uber_yelp = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="kaggle380k_id",
)

# --------------------------------
# Matching configuration from precomputed optimal settings
# --------------------------------

comparators_kaggle_uber = [
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

comparators_kaggle_yelp = [
    StringComparator(
        column="phone_e164",
        similarity_function="jaro_winkler",
        preprocess=strip_only,
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
        preprocess=strip_only,
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

comparators_uber_yelp = [
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

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_kaggle_uber = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_kaggle_uber,
    comparators=comparators_kaggle_uber,
    weights=[0.35, 0.12, 0.08, 0.05, 0.16, 0.16, 0.08],
    threshold=0.75,
    id_column="kaggle380k_id",
)

rb_correspondences_kaggle_yelp = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_kaggle_yelp,
    comparators=comparators_kaggle_yelp,
    weights=[0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1],
    threshold=0.75,
    id_column="kaggle380k_id",
)

rb_correspondences_uber_yelp = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_uber_yelp,
    comparators=comparators_uber_yelp,
    weights=[0.28, 0.16, 0.12, 0.16, 0.16, 0.12],
    threshold=0.72,
    id_column="kaggle380k_id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_kaggle_uber.to_csv(
    os.path.join(CORR_DIR, "correspondences_kaggle_small_uber_eats_small.csv"),
    index=False,
)
rb_correspondences_kaggle_yelp.to_csv(
    os.path.join(CORR_DIR, "correspondences_kaggle_small_yelp_small.csv"),
    index=False,
)
rb_correspondences_uber_yelp.to_csv(
    os.path.join(CORR_DIR, "correspondences_uber_eats_small_yelp_small.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_kaggle_uber,
        rb_correspondences_kaggle_yelp,
        rb_correspondences_uber_yelp,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("kaggle380k_id", canonical_cluster_id)
strategy.add_attribute_fuser("id", canonical_cluster_id)
strategy.add_attribute_fuser("_id", canonical_cluster_id)
strategy.add_attribute_fuser("source", canonical_source)

strategy.add_attribute_fuser("name", trusted_or_voting)
strategy.add_attribute_fuser("name_norm", canonical_name_norm)
strategy.add_attribute_fuser("website", trusted_or_voting)
strategy.add_attribute_fuser("map_url", trusted_or_voting)
strategy.add_attribute_fuser("phone_raw", canonical_phone)
strategy.add_attribute_fuser("phone_e164", canonical_phone)
strategy.add_attribute_fuser("address_line1", trusted_or_voting)
strategy.add_attribute_fuser("address_line2", trusted_or_voting)
strategy.add_attribute_fuser("street", trusted_or_voting)
strategy.add_attribute_fuser("house_number", canonical_house_number)
strategy.add_attribute_fuser("city", trusted_or_voting)
strategy.add_attribute_fuser("state", trusted_or_voting)
strategy.add_attribute_fuser("postal_code", canonical_postal_code)
strategy.add_attribute_fuser("country", trusted_or_voting)
strategy.add_attribute_fuser("latitude", canonical_latlon)
strategy.add_attribute_fuser("longitude", canonical_latlon)
strategy.add_attribute_fuser("categories", union)
strategy.add_attribute_fuser("rating", canonical_rating)
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
    id_column="kaggle380k_id",
    include_singletons=True,
)

if "id" not in rb_fused_standard_blocker.columns and "kaggle380k_id" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["id"] = rb_fused_standard_blocker["kaggle380k_id"]

if "_id" not in rb_fused_standard_blocker.columns and "id" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["_id"] = rb_fused_standard_blocker["id"]

for col in ["phone_e164", "phone_raw"]:
    if col in rb_fused_standard_blocker.columns:
        rb_fused_standard_blocker[col] = rb_fused_standard_blocker[col].apply(normalize_phone)

for col in ["postal_code"]:
    if col in rb_fused_standard_blocker.columns:
        rb_fused_standard_blocker[col] = rb_fused_standard_blocker[col].apply(normalize_postal_code)

for col in ["house_number"]:
    if col in rb_fused_standard_blocker.columns:
        rb_fused_standard_blocker[col] = rb_fused_standard_blocker[col].apply(normalize_house_number)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=24
node_name=execute_pipeline
accuracy_score=46.36%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv, load_xml

from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    TokenBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    shortest_string,
    union,
    prefer_higher_trust,
    voting,
    maximum,
)

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import re
import ast

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def is_nullish(x):
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x).strip()
    return s.lower() in {"", "nan", "none", "null", "nat", "[]", "{}"}


def lower_strip(x):
    if is_nullish(x):
        return None
    return str(x).lower().strip()


def strip_only(x):
    if is_nullish(x):
        return None
    return str(x).strip()


def normalize_text_field(x):
    if is_nullish(x):
        return None
    return str(x).strip()


def normalize_name_norm(x):
    if is_nullish(x):
        return None
    s = str(x).strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None


def normalize_phone_e164(x):
    if is_nullish(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    if s.startswith("+"):
        return f"+{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    if len(digits) == 10:
        return f"+1{digits}"
    return f"+{digits}"


def normalize_phone_raw(x):
    if is_nullish(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    digits = re.sub(r"\D", "", s)
    return digits if digits else None


def normalize_postal_code(x):
    if is_nullish(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    return s


def normalize_house_number(x):
    if is_nullish(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    return s


def normalize_url(x):
    if is_nullish(x):
        return None
    s = str(x).strip()
    s = re.sub(r"^http://", "https://", s)
    s = re.sub(r"[?&](utm_[^=&]+|gclid|fbclid|y_source)=[^&]*", "", s)
    s = re.sub(r"[?&]+$", "", s)
    return s


def parse_categories(x):
    if is_nullish(x):
        return []
    if isinstance(x, list):
        vals = x
    else:
        s = str(x).strip()
        try:
            parsed = ast.literal_eval(s)
            vals = parsed if isinstance(parsed, list) else [s]
        except Exception:
            vals = [p.strip() for p in s.split(",") if p.strip()]
    cleaned = []
    for v in vals:
        t = str(v).strip().lower()
        if t and t not in {"nan", "none", "null"}:
            cleaned.append(t)
    return cleaned


def _flatten_values(values):
    if values is None:
        return []
    if isinstance(values, pd.Series):
        return values.tolist()
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, (list, tuple, set)):
        return list(values)
    return [values]


def choose_first_non_null(values):
    vals = _flatten_values(values)
    for v in vals:
        if not is_nullish(v):
            return str(v).strip()
    return None


def choose_longest_non_null(values):
    vals = _flatten_values(values)
    cleaned = [str(v).strip() for v in vals if not is_nullish(v)]
    if not cleaned:
        return None
    return max(cleaned, key=len)


def trusted_text(values):
    vals = _flatten_values(values)
    cleaned = [str(v).strip() for v in vals if not is_nullish(v)]
    if not cleaned:
        return None
    counts = pd.Series(cleaned).value_counts()
    top = counts.iloc[0]
    winners = counts[counts == top].index.tolist()
    return max(winners, key=len)


def canonical_name_norm(values):
    vals = _flatten_values(values)
    cleaned = [normalize_name_norm(v) for v in vals]
    cleaned = [v for v in cleaned if v is not None]
    if not cleaned:
        return None
    counts = pd.Series(cleaned).value_counts()
    top = counts.iloc[0]
    winners = counts[counts == top].index.tolist()
    return min(winners, key=len)


def canonical_phone_e164(values):
    vals = _flatten_values(values)
    cleaned = [normalize_phone_e164(v) for v in vals]
    cleaned = [v for v in cleaned if v is not None]
    if not cleaned:
        return None
    counts = pd.Series(cleaned).value_counts()
    return counts.index[0]


def canonical_phone_raw(values):
    vals = _flatten_values(values)
    cleaned = [normalize_phone_raw(v) for v in vals]
    cleaned = [v for v in cleaned if v is not None]
    if not cleaned:
        return None
    counts = pd.Series(cleaned).value_counts()
    return counts.index[0]


def canonical_postal_code(values):
    vals = _flatten_values(values)
    cleaned = [normalize_postal_code(v) for v in vals]
    cleaned = [v for v in cleaned if v is not None]
    if not cleaned:
        return None
    counts = pd.Series(cleaned).value_counts()
    return counts.index[0]


def canonical_house_number(values):
    vals = _flatten_values(values)
    cleaned = [normalize_house_number(v) for v in vals]
    cleaned = [v for v in cleaned if v is not None]
    if not cleaned:
        return None
    counts = pd.Series(cleaned).value_counts()
    top = counts.iloc[0]
    winners = counts[counts == top].index.tolist()
    return min(winners, key=len)


def canonical_rating(values):
    vals = _flatten_values(values)
    cleaned = []
    for v in vals:
        if is_nullish(v):
            continue
        try:
            cleaned.append(float(v))
        except Exception:
            pass
    if not cleaned:
        return None
    return float(np.median(cleaned))


def canonical_latlon(values):
    vals = _flatten_values(values)
    cleaned = []
    for v in vals:
        if is_nullish(v):
            continue
        try:
            cleaned.append(float(v))
        except Exception:
            pass
    if not cleaned:
        return None
    return float(np.median(cleaned))


# Load datasets
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

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Deterministic schema alignment for critical columns
# --------------------------------

print("Matching Schema")

for df in [kaggle_small, uber_eats_small, yelp_small]:
    if "id" not in df.columns:
        raise ValueError("Expected source id column 'id' missing.")
    df["kaggle380k_id"] = df["id"]

required_columns = [
    "id",
    "kaggle380k_id",
    "source",
    "name",
    "name_norm",
    "website",
    "map_url",
    "phone_raw",
    "phone_e164",
    "address_line1",
    "address_line2",
    "street",
    "house_number",
    "city",
    "state",
    "postal_code",
    "country",
    "latitude",
    "longitude",
    "categories",
    "rating",
    "rating_count",
]

for df in [kaggle_small, uber_eats_small, yelp_small]:
    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA

# --------------------------------
# Normalize critical fields before blocking, matching, and fusion
# --------------------------------

for df in [kaggle_small, uber_eats_small, yelp_small]:
    df["id"] = df["id"].apply(normalize_text_field)
    df["kaggle380k_id"] = df["kaggle380k_id"].apply(normalize_text_field)
    df["source"] = df["source"].apply(normalize_text_field)
    df["name"] = df["name"].apply(normalize_text_field)
    df["name_norm"] = df["name_norm"].apply(normalize_name_norm)
    df["website"] = df["website"].apply(normalize_url)
    df["map_url"] = df["map_url"].apply(normalize_url)
    df["phone_raw"] = df["phone_raw"].apply(normalize_phone_raw)
    df["phone_e164"] = df["phone_e164"].apply(normalize_phone_e164)
    df["address_line1"] = df["address_line1"].apply(normalize_text_field)
    df["address_line2"] = df["address_line2"].apply(normalize_text_field)
    df["street"] = df["street"].apply(normalize_text_field)
    df["house_number"] = df["house_number"].apply(normalize_house_number)
    df["city"] = df["city"].apply(normalize_text_field)
    df["state"] = df["state"].apply(normalize_text_field)
    df["postal_code"] = df["postal_code"].apply(normalize_postal_code)
    df["country"] = df["country"].apply(normalize_text_field)
    df["categories"] = df["categories"].apply(parse_categories)

# --------------------------------
# Perform Blocking using precomputed blocking configuration
# --------------------------------

print("Performing Blocking")

blocker_kaggle_uber = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="kaggle380k_id",
)

blocker_kaggle_yelp = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="kaggle380k_id",
)

blocker_uber_yelp = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="kaggle380k_id",
)

# --------------------------------
# Matching configuration from precomputed optimal settings
# --------------------------------

comparators_kaggle_uber = [
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

comparators_kaggle_yelp = [
    StringComparator(
        column="phone_e164",
        similarity_function="jaro_winkler",
        preprocess=strip_only,
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
        preprocess=strip_only,
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

comparators_uber_yelp = [
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

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_kaggle_uber = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_kaggle_uber,
    comparators=comparators_kaggle_uber,
    weights=[0.35, 0.12, 0.08, 0.05, 0.16, 0.16, 0.08],
    threshold=0.75,
    id_column="kaggle380k_id",
)

rb_correspondences_kaggle_yelp = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_kaggle_yelp,
    comparators=comparators_kaggle_yelp,
    weights=[0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1],
    threshold=0.75,
    id_column="kaggle380k_id",
)

rb_correspondences_uber_yelp = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_uber_yelp,
    comparators=comparators_uber_yelp,
    weights=[0.28, 0.16, 0.12, 0.16, 0.16, 0.12],
    threshold=0.72,
    id_column="kaggle380k_id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_kaggle_uber.to_csv(
    os.path.join(CORR_DIR, "correspondences_kaggle_small_uber_eats_small.csv"),
    index=False,
)
rb_correspondences_kaggle_yelp.to_csv(
    os.path.join(CORR_DIR, "correspondences_kaggle_small_yelp_small.csv"),
    index=False,
)
rb_correspondences_uber_yelp.to_csv(
    os.path.join(CORR_DIR, "correspondences_uber_eats_small_yelp_small.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_kaggle_uber,
        rb_correspondences_kaggle_yelp,
        rb_correspondences_uber_yelp,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("id", choose_first_non_null)
strategy.add_attribute_fuser("kaggle380k_id", choose_first_non_null)
strategy.add_attribute_fuser("source", choose_first_non_null)

strategy.add_attribute_fuser("name", trusted_text)
strategy.add_attribute_fuser("name_norm", canonical_name_norm)
strategy.add_attribute_fuser("website", choose_first_non_null)
strategy.add_attribute_fuser("map_url", choose_first_non_null)
strategy.add_attribute_fuser("phone_raw", canonical_phone_raw)
strategy.add_attribute_fuser("phone_e164", canonical_phone_e164)
strategy.add_attribute_fuser("address_line1", choose_longest_non_null)
strategy.add_attribute_fuser("address_line2", trusted_text)
strategy.add_attribute_fuser("street", trusted_text)
strategy.add_attribute_fuser("house_number", canonical_house_number)
strategy.add_attribute_fuser("city", trusted_text)
strategy.add_attribute_fuser("state", trusted_text)
strategy.add_attribute_fuser("postal_code", canonical_postal_code)
strategy.add_attribute_fuser("country", trusted_text)
strategy.add_attribute_fuser("latitude", canonical_latlon)
strategy.add_attribute_fuser("longitude", canonical_latlon)
strategy.add_attribute_fuser("categories", union)
strategy.add_attribute_fuser("rating", canonical_rating)
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
    id_column="kaggle380k_id",
    include_singletons=True,
)

if "kaggle380k_id" not in rb_fused_standard_blocker.columns and "id" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["kaggle380k_id"] = rb_fused_standard_blocker["id"]

if "id" not in rb_fused_standard_blocker.columns and "kaggle380k_id" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["id"] = rb_fused_standard_blocker["kaggle380k_id"]

if "_id" not in rb_fused_standard_blocker.columns:
    if "id" in rb_fused_standard_blocker.columns:
        rb_fused_standard_blocker["_id"] = rb_fused_standard_blocker["id"]
    elif "kaggle380k_id" in rb_fused_standard_blocker.columns:
        rb_fused_standard_blocker["_id"] = rb_fused_standard_blocker["kaggle380k_id"]

for col in ["id", "_id", "kaggle380k_id"]:
    if col in rb_fused_standard_blocker.columns:
        rb_fused_standard_blocker[col] = rb_fused_standard_blocker[col].apply(normalize_text_field)

if "source" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["source"] = rb_fused_standard_blocker["source"].apply(normalize_text_field)

if "source" in rb_fused_standard_blocker.columns and "id" in rb_fused_standard_blocker.columns:
    missing_source = rb_fused_standard_blocker["source"].isna() | (
        rb_fused_standard_blocker["source"].astype(str).str.strip().str.lower().isin(["", "nan", "none"])
    )
    rb_fused_standard_blocker.loc[
        missing_source & rb_fused_standard_blocker["id"].astype(str).str.startswith("yelp-"), "source"
    ] = "yelp"
    rb_fused_standard_blocker.loc[
        missing_source & rb_fused_standard_blocker["id"].astype(str).str.startswith("uber_eats-"), "source"
    ] = "uber_eats"
    rb_fused_standard_blocker.loc[
        missing_source & rb_fused_standard_blocker["id"].astype(str).str.startswith("kaggle380k-"), "source"
    ] = "kaggle_380k"

for col in ["phone_e164"]:
    if col in rb_fused_standard_blocker.columns:
        rb_fused_standard_blocker[col] = rb_fused_standard_blocker[col].apply(normalize_phone_e164)

for col in ["phone_raw"]:
    if col in rb_fused_standard_blocker.columns:
        rb_fused_standard_blocker[col] = rb_fused_standard_blocker[col].apply(normalize_phone_raw)

for col in ["postal_code"]:
    if col in rb_fused_standard_blocker.columns:
        rb_fused_standard_blocker[col] = rb_fused_standard_blocker[col].apply(normalize_postal_code)

for col in ["house_number"]:
    if col in rb_fused_standard_blocker.columns:
        rb_fused_standard_blocker[col] = rb_fused_standard_blocker[col].apply(normalize_house_number)

for col in ["name_norm"]:
    if col in rb_fused_standard_blocker.columns:
        rb_fused_standard_blocker[col] = rb_fused_standard_blocker[col].apply(normalize_name_norm)

for col in ["website", "map_url"]:
    if col in rb_fused_standard_blocker.columns:
        rb_fused_standard_blocker[col] = rb_fused_standard_blocker[col].apply(normalize_url)

for col in ["name", "address_line1", "address_line2", "street", "city", "state", "country"]:
    if col in rb_fused_standard_blocker.columns:
        rb_fused_standard_blocker[col] = rb_fused_standard_blocker[col].apply(normalize_text_field)

if "categories" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["categories"] = rb_fused_standard_blocker["categories"].apply(
        lambda x: sorted(list(set(parse_categories(x))))
    )

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

