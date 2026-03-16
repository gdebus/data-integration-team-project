# Pipeline Snapshots

notebook_name=AdaptationPipeline
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=3
node_name=execute_pipeline
accuracy_score=69.46%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

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

# Define dataset paths
DATA_DIR = "input/datasets/restaurant/"

# Define API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load the first dataset
kaggle_small = load_parquet(
    DATA_DIR + "kaggle_small.parquet",
    name="kaggle_small",
)

# Load Uber Eats dataset
uber_eats_small = load_parquet(
    DATA_DIR + "uber_eats_small.parquet",
    name="uber_eats_small",
)

# Load Yelp dataset
yelp_small = load_parquet(
    DATA_DIR + "yelp_small.parquet",
    name="yelp_small",
)

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# CRITICAL INSTRUCTION FOR AGENTS:
# The here implemented schema matching will match the schema of dataset2 and dataset3 to the schema of dataset1. Therefore, the resulting columns for all
# datasets will have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# match schema of kaggle_small with uber_eats_small and rename schema of uber_eats_small
schema_correspondences = matcher.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

# match schema of kaggle_small with yelp_small and rename schema of yelp_small
schema_correspondences = matcher.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# YOU MUST!!! USE THE PRECOMPUTED BLOCKERS PROVIEDED TO YOU UNDER "5. **BLOCKING CONFIGURATION**"!
# --------------------------------

print("Performing Blocking")

# Use normalized restaurant name as primary identity signal.
# It is highly aligned across all three datasets, fully populated, and more discriminative than city.
# Token blocking is robust to formatting differences while preserving recall.

blocker_k2u = TokenBlocker(
    kaggle_small,
    uber_eats_small,
    column="name_norm",
    min_token_len=2,
    ngram_size=1,
    ngram_type="word",
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_k2y = TokenBlocker(
    kaggle_small,
    yelp_small,
    column="name_norm",
    min_token_len=2,
    ngram_size=1,
    ngram_type="word",
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_u2y = TokenBlocker(
    uber_eats_small,
    yelp_small,
    column="name_norm",
    min_token_len=2,
    ngram_size=1,
    ngram_type="word",
    id_column="id",
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# You MUST use the matching configuration supplied to you under "6. **MATCHING CONFIGURATION**" to set the correct comparators in the following.
# --------------------------------

comparators_k2u = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="street",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    NumericComparator(
        column="house_number",
        max_difference=2,
    ),
    StringComparator(
        column="city",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=str.lower,
        list_strategy="concatenate",
    ),
]

comparators_k2y = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="street",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    NumericComparator(
        column="house_number",
        max_difference=2,
    ),
    StringComparator(
        column="city",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=str.lower,
        list_strategy="concatenate",
    ),
]

comparators_u2y = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="street",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    NumericComparator(
        column="house_number",
        max_difference=2,
    ),
    StringComparator(
        column="city",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=str.lower,
        list_strategy="concatenate",
    ),
]

print("Matching Entities")

# Initialize Rule-Based Matcher
matcher = RuleBasedMatcher()

rb_correspondences_k2u = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=[0.5, 0.15, 0.15, 0.1, 0.1],
    threshold=0.7,
    id_column="id",
)

rb_correspondences_k2y = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=[0.5, 0.15, 0.15, 0.1, 0.1],
    threshold=0.7,
    id_column="id",
)

rb_correspondences_u2y = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=[0.5, 0.15, 0.15, 0.1, 0.1],
    threshold=0.7,
    id_column="id",
)

"""
Make sure to save correspondences for each pair afer applying Matcher.
**Use proper filename with correct dataset name to save the correspondences**
"""

CORR_DIR = "output/correspondences"
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

print("Fusing Data")

# --------------------------------
# Data Fusion
# There are following conflict resolution functions available:
# For strings: longest_string, shortest_string, most_complete
# For numerics: average, median, maximum, minimum, sum_values
# For dates: most_recent, earliest
# For lists/sets: union
# --------------------------------

# merge rule based correspondences
all_rb_correspondences = pd.concat(
    [rb_correspondences_k2u, rb_correspondences_k2y, rb_correspondences_u2y],
    ignore_index=True,
)

# define data fusion strategy
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

# run fusion
engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

# fuse rule based matches
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
node_index=8
node_name=execute_pipeline
accuracy_score=56.45%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

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

DATA_DIR = "input/datasets/restaurant/"

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

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# The resulting columns for all datasets will have the schema of kaggle_small.
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
# Lightweight normalization to improve matching/fusion robustness
# --------------------------------

def normalize_postal_code(x):
    if pd.isna(x):
        return x
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    return s.split("-")[0]

def normalize_phone_e164(x):
    if pd.isna(x):
        return x
    digits = "".join(ch for ch in str(x) if ch.isdigit())
    if digits == "":
        return np.nan
    if not digits.startswith("1") and len(digits) == 10:
        digits = "1" + digits
    return "+" + digits

def normalize_phone_raw(x):
    if pd.isna(x):
        return x
    digits = "".join(ch for ch in str(x) if ch.isdigit())
    return digits if digits != "" else np.nan

for df in [kaggle_small, uber_eats_small, yelp_small]:
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].apply(normalize_postal_code)
    if "phone_e164" in df.columns:
        df["phone_e164"] = df["phone_e164"].apply(normalize_phone_e164)
    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].apply(normalize_phone_raw)

# --------------------------------
# Perform Blocking
# Use informative identity signals only: normalized restaurant name + city.
# This is stronger than city-only blocking and more precise than name-only blocking.
# --------------------------------

print("Performing Blocking")

blocker_k2u = StandardBlocker(
    kaggle_small,
    uber_eats_small,
    on=["name_norm", "city"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_k2y = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["name_norm", "city"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_u2y = StandardBlocker(
    uber_eats_small,
    yelp_small,
    on=["name_norm", "city"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching configuration
# --------------------------------

comparators_k2u = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="city",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="street",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    NumericComparator(
        column="house_number",
        max_difference=1,
    ),
    StringComparator(
        column="postal_code",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=str.lower,
        list_strategy="concatenate",
    ),
]

comparators_k2y = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="city",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="street",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    NumericComparator(
        column="house_number",
        max_difference=1,
    ),
    StringComparator(
        column="postal_code",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=str.lower,
        list_strategy="concatenate",
    ),
]

comparators_u2y = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="city",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="street",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    NumericComparator(
        column="house_number",
        max_difference=1,
    ),
    StringComparator(
        column="postal_code",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=str.lower,
        list_strategy="concatenate",
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_k2u = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=[0.45, 0.15, 0.15, 0.1, 0.1, 0.05],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_k2y = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=[0.45, 0.15, 0.15, 0.1, 0.1, 0.05],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_u2y = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=[0.45, 0.15, 0.15, 0.1, 0.1, 0.05],
    threshold=0.75,
    id_column="id",
)

CORR_DIR = "output/correspondences"
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

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_k2u, rb_correspondences_k2y, rb_correspondences_u2y],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("id", longest_string)
strategy.add_attribute_fuser("source", voting)
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", shortest_string)
strategy.add_attribute_fuser("website", longest_string)
strategy.add_attribute_fuser("map_url", shortest_string)
strategy.add_attribute_fuser("phone_raw", shortest_string)
strategy.add_attribute_fuser("phone_e164", shortest_string)
strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", longest_string)
strategy.add_attribute_fuser("city", voting)
strategy.add_attribute_fuser("state", voting)
strategy.add_attribute_fuser("postal_code", shortest_string)
strategy.add_attribute_fuser("country", voting)
strategy.add_attribute_fuser("latitude", longest_string)
strategy.add_attribute_fuser("longitude", longest_string)
strategy.add_attribute_fuser("categories", union)
strategy.add_attribute_fuser("rating", voting)
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

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=13
node_name=execute_pipeline
accuracy_score=54.69%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

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
import ast

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "input/datasets/restaurant/"

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

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# The resulting columns for all datasets will have the schema of kaggle_small.
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
# Normalization helpers
# --------------------------------

def clean_null(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    return s

def normalize_text(x):
    x = clean_null(x)
    if pd.isna(x):
        return np.nan
    s = str(x).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else np.nan

def normalize_name_for_blocking(x):
    s = normalize_text(x)
    if pd.isna(s):
        return np.nan
    s = re.sub(r"\b(the|restaurant|grill|bar|cafe|kitchen)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else np.nan

def normalize_street(x):
    s = normalize_text(x)
    if pd.isna(s):
        return np.nan
    replacements = {
        r"\bstreet\b": "st",
        r"\bavenue\b": "ave",
        r"\bboulevard\b": "blvd",
        r"\bdrive\b": "dr",
        r"\broad\b": "rd",
        r"\blane\b": "ln",
        r"\bcourt\b": "ct",
        r"\bplace\b": "pl",
        r"\bparkway\b": "pkwy",
        r"\bhighway\b": "hwy",
        r"\broute\b": "rt",
        r"\bunit\b": "",
        r"\bsuite\b": "",
    }
    for pat, repl in replacements.items():
        s = re.sub(pat, repl, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else np.nan

def normalize_city(x):
    return normalize_text(x)

def normalize_state(x):
    s = clean_null(x)
    if pd.isna(s):
        return np.nan
    return str(s).upper().strip()

def normalize_country(x):
    s = clean_null(x)
    if pd.isna(s):
        return np.nan
    s = str(s).upper().strip()
    if s in {"USA", "UNITED STATES", "UNITED STATES OF AMERICA"}:
        return "US"
    return s

def normalize_postal_code(x):
    s = clean_null(x)
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    s = s.split("-")[0]
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits if digits else np.nan

def normalize_phone_e164(x):
    s = clean_null(x)
    if pd.isna(s):
        return np.nan
    digits = "".join(ch for ch in str(s) if ch.isdigit())
    if len(digits) == 10:
        digits = "1" + digits
    if len(digits) < 11:
        return np.nan
    return "+" + digits

def normalize_phone_raw(x):
    s = clean_null(x)
    if pd.isna(s):
        return np.nan
    digits = "".join(ch for ch in str(s) if ch.isdigit())
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits if digits else np.nan

def normalize_house_number(x):
    s = clean_null(x)
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    return s if s else np.nan

def normalize_categories(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, list):
        vals = x
    else:
        s = str(x).strip()
        if s == "":
            return np.nan
        try:
            vals = ast.literal_eval(s) if s.startswith("[") else [s]
        except Exception:
            vals = [s]
    out = []
    for v in vals:
        nv = normalize_text(v)
        if pd.notna(nv):
            out.append(nv)
    return str(sorted(set(out))) if out else np.nan

for df in [kaggle_small, uber_eats_small, yelp_small]:
    if "name" in df.columns:
        df["name"] = df["name"].apply(clean_null)
    if "name_norm" in df.columns:
        df["name_norm"] = df["name_norm"].apply(normalize_text)
        df["name_block"] = df["name_norm"].apply(normalize_name_for_blocking)
    if "street" in df.columns:
        df["street"] = df["street"].apply(clean_null)
        df["street_norm"] = df["street"].apply(normalize_street)
    if "city" in df.columns:
        df["city"] = df["city"].apply(clean_null)
        df["city_norm"] = df["city"].apply(normalize_city)
    if "state" in df.columns:
        df["state"] = df["state"].apply(normalize_state)
    if "country" in df.columns:
        df["country"] = df["country"].apply(normalize_country)
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].apply(normalize_postal_code)
    if "phone_e164" in df.columns:
        df["phone_e164"] = df["phone_e164"].apply(normalize_phone_e164)
    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].apply(normalize_phone_raw)
    if "house_number" in df.columns:
        df["house_number"] = df["house_number"].apply(normalize_house_number)
    if "categories" in df.columns:
        df["categories"] = df["categories"].apply(normalize_categories)
    if "address_line1" in df.columns:
        df["address_line1"] = df["address_line1"].apply(clean_null)
    if "address_line2" in df.columns:
        df["address_line2"] = df["address_line2"].apply(clean_null)
    if "website" in df.columns:
        df["website"] = df["website"].apply(clean_null)
    if "map_url" in df.columns:
        df["map_url"] = df["map_url"].apply(clean_null)

# --------------------------------
# Perform Blocking
# Use informative identity signals only.
# Primary signal: normalized canonical name.
# Add city as disambiguator.
# --------------------------------

print("Performing Blocking")

blocker_k2u = StandardBlocker(
    kaggle_small,
    uber_eats_small,
    on=["name_block", "city_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_k2y = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["name_block", "city_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_u2y = StandardBlocker(
    uber_eats_small,
    yelp_small,
    on=["name_block", "city_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching configuration
# --------------------------------

comparators_k2u = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="city_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="street_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="postal_code",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="house_number",
        max_difference=2,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        list_strategy="concatenate",
    ),
]

comparators_k2y = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="city_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="street_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="postal_code",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="house_number",
        max_difference=2,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        list_strategy="concatenate",
    ),
]

comparators_u2y = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="city_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="street_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="postal_code",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="house_number",
        max_difference=2,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        list_strategy="concatenate",
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_k2u = matcher.match(
    df_left=kaggle_small,
    df_right=uber_eats_small,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=[0.50, 0.15, 0.15, 0.10, 0.05, 0.05],
    threshold=0.70,
    id_column="id",
)

rb_correspondences_k2y = matcher.match(
    df_left=kaggle_small,
    df_right=yelp_small,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=[0.50, 0.15, 0.15, 0.10, 0.05, 0.05],
    threshold=0.70,
    id_column="id",
)

rb_correspondences_u2y = matcher.match(
    df_left=uber_eats_small,
    df_right=yelp_small,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=[0.50, 0.15, 0.15, 0.10, 0.05, 0.05],
    threshold=0.70,
    id_column="id",
)

CORR_DIR = "output/correspondences"
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

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_k2u, rb_correspondences_k2y, rb_correspondences_u2y],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("id", shortest_string)
strategy.add_attribute_fuser("source", shortest_string)
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", shortest_string)
strategy.add_attribute_fuser("website", longest_string)
strategy.add_attribute_fuser("map_url", longest_string)
strategy.add_attribute_fuser("phone_raw", longest_string)
strategy.add_attribute_fuser("phone_e164", longest_string)
strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", shortest_string)
strategy.add_attribute_fuser("city", shortest_string)
strategy.add_attribute_fuser("state", shortest_string)
strategy.add_attribute_fuser("postal_code", longest_string)
strategy.add_attribute_fuser("country", shortest_string)
strategy.add_attribute_fuser("latitude", shortest_string)
strategy.add_attribute_fuser("longitude", shortest_string)
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

drop_cols = [c for c in ["name_block", "street_norm", "city_norm"] if c in rb_fused_standard_blocker.columns]
if drop_cols:
    rb_fused_standard_blocker = rb_fused_standard_blocker.drop(columns=drop_cols)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

