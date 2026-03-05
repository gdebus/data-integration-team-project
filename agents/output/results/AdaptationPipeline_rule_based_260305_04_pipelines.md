# Pipeline Snapshots

notebook_name=AdaptationPipeline
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=3
node_name=execute_pipeline
accuracy_score=73.22%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_parquet
from PyDI.entitymatching import StandardBlocker
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

DATA_DIR = "input/datasets/restaurant/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load Kaggle dataset (dataset1 / target schema)
good_dataset_name_1 = load_parquet(
    DATA_DIR + "kaggle_small.parquet",
    name="kaggle_small",
)

# Load Uber Eats dataset (dataset2)
good_dataset_name_2 = load_parquet(
    DATA_DIR + "uber_eats_small.parquet",
    name="uber_eats_small",
)

# Load Yelp dataset (dataset3)
good_dataset_name_3 = load_parquet(
    DATA_DIR + "yelp_small.parquet",
    name="yelp_small",
)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

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

schema_matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True
)

# match schema of kaggle with uber_eats and rename schema of uber_eats
schema_correspondences = schema_matcher.match(good_dataset_name_1, good_dataset_name_2)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_2 = good_dataset_name_2.rename(columns=rename_map)

# match schema of kaggle with yelp and rename schema of yelp
schema_correspondences = schema_matcher.match(good_dataset_name_1, good_dataset_name_3)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_3 = good_dataset_name_3.rename(columns=rename_map)

# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# YOU MUST!!! USE THE PRECOMPUTED BLOCKERS PROVIEDED TO YOU UNDER "5. **BLOCKING CONFIGURATION**"!
# --------------------------------

print("Performing Blocking")

blocker_k2u = StandardBlocker(
    good_dataset_name_1, good_dataset_name_2,
    on=['city'],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column='id'
)

blocker_k2y = StandardBlocker(
    good_dataset_name_1, good_dataset_name_3,
    on=['city'],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column='id'
)

blocker_u2y = StandardBlocker(
    good_dataset_name_2, good_dataset_name_3,
    on=['city'],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column='id'
)

# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# You MUST use the matching configuration supplied to you under "6. **MATCHING CONFIGURATION**" to set the correct comparators in the following.
# --------------------------------

comparators_k2u = [
    StringComparator(
        column='name_norm',
        similarity_function='jaccard',
    ),
    StringComparator(
        column='street',
        similarity_function='jaccard',
        preprocess=str.lower,
    ),
    NumericComparator(
        column='house_number',
        max_difference=2,
    ),
    StringComparator(
        column='categories',
        similarity_function='jaccard',
        preprocess=str.lower,
        list_strategy='concatenate'
    )
]

comparators_k2y = [
    StringComparator(
        column='name_norm',
        similarity_function='jaccard',
    ),
    StringComparator(
        column='street',
        similarity_function='jaccard',
        preprocess=str.lower,
    ),
    NumericComparator(
        column='house_number',
        max_difference=2,
    ),
    StringComparator(
        column='categories',
        similarity_function='jaccard',
        preprocess=str.lower,
        list_strategy='concatenate'
    )
]

comparators_u2y = [
    StringComparator(
        column='name_norm',
        similarity_function='jaccard',
    ),
    StringComparator(
        column='street',
        similarity_function='jaccard',
        preprocess=str.lower,
    ),
    NumericComparator(
        column='house_number',
        max_difference=2,
    ),
    StringComparator(
        column='categories',
        similarity_function='jaccard',
        preprocess=str.lower,
        list_strategy='concatenate'
    )
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_k2u = rb_matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=[0.5, 0.2, 0.2, 0.1],
    threshold=0.7,
    id_column='id'
)

rb_correspondences_k2y = rb_matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=[0.5, 0.2, 0.2, 0.1],
    threshold=0.7,
    id_column='id'
)

rb_correspondences_u2y = rb_matcher.match(
    df_left=good_dataset_name_2,
    df_right=good_dataset_name_3,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=[0.5, 0.2, 0.2, 0.1],
    threshold=0.7,
    id_column='id'
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_k2u, rb_correspondences_k2y, rb_correspondences_u2y],
    ignore_index=True
)

strategy = DataFusionStrategy('rule_based_fusion_strategy')
strategy.add_attribute_fuser('name', longest_string)
strategy.add_attribute_fuser('street', longest_string)
strategy.add_attribute_fuser('house_number', longest_string)
strategy.add_attribute_fuser('city', longest_string)
strategy.add_attribute_fuser('state', longest_string)
strategy.add_attribute_fuser('postal_code', longest_string)
strategy.add_attribute_fuser('country', longest_string)
strategy.add_attribute_fuser('latitude', longest_string)
strategy.add_attribute_fuser('longitude', longest_string)
strategy.add_attribute_fuser('categories', union)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format='json',
    debug_file="output/data_fusion/debug_fusion_data.jsonl"
)

rb_fused_standard_blocker = engine.run(
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
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
node_index=8
node_name=execute_pipeline
accuracy_score=73.22%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_parquet
from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    voting,
    prefer_higher_trust,
    maximum,
)
from PyDI.schemamatching import LLMBasedSchemaMatcher

from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
import re
from ast import literal_eval
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/restaurant/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_parquet(
    DATA_DIR + "kaggle_small.parquet",
    name="kaggle_small",
)

good_dataset_name_2 = load_parquet(
    DATA_DIR + "uber_eats_small.parquet",
    name="uber_eats_small",
)

good_dataset_name_3 = load_parquet(
    DATA_DIR + "yelp_small.parquet",
    name="yelp_small",
)

# --------------------------------
# Light preprocessing to improve fusion accuracy for problematic attributes
# (phone, postal_code, URLs, categories, source)
# --------------------------------

def _clean_phone_raw(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    # keep digits only
    digits = re.sub(r"\D+", "", s)
    return digits if digits else np.nan

def _clean_phone_e164(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    # keep digits and leading +
    digits = re.sub(r"\D+", "", s)
    if not digits:
        return np.nan
    # assume US when 10 digits; if 11 and starts with 1 => US
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    return "+" + digits

def _clean_postal_code(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    # keep digits; use first 5 if ZIP+4 present
    digits = re.sub(r"[^\d]", "", s)
    if len(digits) >= 5:
        return digits[:5]
    return np.nan

def _clean_url(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    return s

def _clean_categories(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(i).strip().lower() for i in x if str(i).strip() != ""]
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []
    # Parse stringified Python list when present
    try:
        v = literal_eval(s)
        if isinstance(v, list):
            return [str(i).strip().lower() for i in v if str(i).strip() != ""]
    except Exception:
        pass
    # Fallback: split on commas
    parts = [p.strip().lower() for p in re.split(r",|;", s) if p.strip() != ""]
    return parts

def _set_source_constant(df, value):
    if "source" in df.columns:
        df["source"] = value
    return df

def _apply_cleaning(df):
    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].apply(_clean_phone_raw)
    if "phone_e164" in df.columns:
        df["phone_e164"] = df["phone_e164"].apply(_clean_phone_e164)
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].apply(_clean_postal_code)
    if "website" in df.columns:
        df["website"] = df["website"].apply(_clean_url)
    if "map_url" in df.columns:
        df["map_url"] = df["map_url"].apply(_clean_url)
    if "categories" in df.columns:
        df["categories"] = df["categories"].apply(_clean_categories)
    return df

good_dataset_name_1 = _set_source_constant(good_dataset_name_1, "kaggle_380k")
good_dataset_name_2 = _set_source_constant(good_dataset_name_2, "uber_eats")
good_dataset_name_3 = _set_source_constant(good_dataset_name_3, "yelp")

good_dataset_name_1 = _apply_cleaning(good_dataset_name_1)
good_dataset_name_2 = _apply_cleaning(good_dataset_name_2)
good_dataset_name_3 = _apply_cleaning(good_dataset_name_3)

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

schema_correspondences = schema_matcher.match(good_dataset_name_1, good_dataset_name_2)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_2 = good_dataset_name_2.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(good_dataset_name_1, good_dataset_name_3)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_3 = good_dataset_name_3.rename(columns=rename_map)

# Re-apply cleaning in case schema renaming introduced/overwrote columns
good_dataset_name_1 = _apply_cleaning(good_dataset_name_1)
good_dataset_name_2 = _apply_cleaning(good_dataset_name_2)
good_dataset_name_3 = _apply_cleaning(good_dataset_name_3)

# --------------------------------
# Blocking (MUST use provided configuration)
# --------------------------------

print("Performing Blocking")

blocker_k2u = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_2,
    on=["city"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_k2y = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    on=["city"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_u2y = StandardBlocker(
    good_dataset_name_2,
    good_dataset_name_3,
    on=["city"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching (MUST use provided matching configuration)
# --------------------------------

comparators_k2u = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="street", similarity_function="jaccard", preprocess=str.lower),
    NumericComparator(column="house_number", max_difference=2),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=str.lower,
        list_strategy="concatenate",
    ),
]

comparators_k2y = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="street", similarity_function="jaccard", preprocess=str.lower),
    NumericComparator(column="house_number", max_difference=2),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=str.lower,
        list_strategy="concatenate",
    ),
]

comparators_u2y = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="street", similarity_function="jaccard", preprocess=str.lower),
    NumericComparator(column="house_number", max_difference=2),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=str.lower,
        list_strategy="concatenate",
    ),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_k2u = rb_matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=[0.5, 0.2, 0.2, 0.1],
    threshold=0.7,
    id_column="id",
)

rb_correspondences_k2y = rb_matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=[0.5, 0.2, 0.2, 0.1],
    threshold=0.7,
    id_column="id",
)

rb_correspondences_u2y = rb_matcher.match(
    df_left=good_dataset_name_2,
    df_right=good_dataset_name_3,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=[0.5, 0.2, 0.2, 0.1],
    threshold=0.7,
    id_column="id",
)

# --------------------------------
# Fusion
# --------------------------------

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_k2u, rb_correspondences_k2y, rb_correspondences_u2y],
    ignore_index=True,
)

# Prefer Kaggle values over others where possible to stabilize attributes
trust_map = {"kaggle_380k": 3, "yelp": 2, "uber_eats": 1}
def _trust(row):
    return trust_map.get(row.get("source", None), 0)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# identity / provenance
strategy.add_attribute_fuser("id", prefer_higher_trust, trust_function=_trust)
strategy.add_attribute_fuser("source", prefer_higher_trust, trust_function=_trust)

# core strings
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", longest_string)
strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", longest_string)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
strategy.add_attribute_fuser("postal_code", prefer_higher_trust, trust_function=_trust)
strategy.add_attribute_fuser("country", longest_string)

# urls / phones
strategy.add_attribute_fuser("website", prefer_higher_trust, trust_function=_trust)
strategy.add_attribute_fuser("map_url", prefer_higher_trust, trust_function=_trust)
strategy.add_attribute_fuser("phone_raw", prefer_higher_trust, trust_function=_trust)
strategy.add_attribute_fuser("phone_e164", prefer_higher_trust, trust_function=_trust)

# geo
strategy.add_attribute_fuser("latitude", prefer_higher_trust, trust_function=_trust)
strategy.add_attribute_fuser("longitude", prefer_higher_trust, trust_function=_trust)

# categories
strategy.add_attribute_fuser("categories", union)

# ratings
strategy.add_attribute_fuser("rating", prefer_higher_trust, trust_function=_trust)
strategy.add_attribute_fuser("rating_count", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
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
accuracy_score=77.97%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_parquet
from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    prefer_higher_trust,
    maximum,
)
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
import re
from ast import literal_eval
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/restaurant/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_parquet(
    DATA_DIR + "kaggle_small.parquet",
    name="kaggle_small",
)

good_dataset_name_2 = load_parquet(
    DATA_DIR + "uber_eats_small.parquet",
    name="uber_eats_small",
)

good_dataset_name_3 = load_parquet(
    DATA_DIR + "yelp_small.parquet",
    name="yelp_small",
)

# --------------------------------
# Cleaning / normalization helpers (focus: postal_code + phones + urls)
# --------------------------------

_NA_STRINGS = {"", "nan", "none", "null", "na", "n/a"}


def _is_na(x):
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    s = str(x).strip().lower()
    return s in _NA_STRINGS


def _clean_phone_digits(x):
    if _is_na(x):
        return np.nan
    digits = re.sub(r"\D+", "", str(x))
    return digits if digits else np.nan


def _to_e164_us(x):
    """
    Normalize to E.164 assuming US/CA when length indicates it.
    If already includes country code -> keep.
    """
    digits = _clean_phone_digits(x)
    if _is_na(digits):
        return np.nan
    digits = str(digits)

    # US number forms
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits

    # otherwise keep with +
    return "+" + digits


def _clean_postal_code(x):
    if _is_na(x):
        return np.nan
    s = str(x).strip()
    # keep digits; zip+4 -> first 5
    digits = re.sub(r"[^\d]", "", s)
    if len(digits) >= 5:
        return digits[:5]
    return np.nan


def _canonical_url(x):
    if _is_na(x):
        return np.nan
    s = str(x).strip()
    # drop tracking query for stability
    s = re.sub(r"[?#].*$", "", s)
    return s if s else np.nan


def _clean_categories(x):
    if _is_na(x):
        return []
    if isinstance(x, list):
        return [str(i).strip().lower() for i in x if not _is_na(i)]
    s = str(x).strip()
    try:
        v = literal_eval(s)
        if isinstance(v, list):
            return [str(i).strip().lower() for i in v if not _is_na(i)]
    except Exception:
        pass
    parts = [p.strip().lower() for p in re.split(r",|;", s) if p.strip()]
    return parts


def _set_source_constant(df, value):
    if "source" in df.columns:
        df["source"] = value
    return df


def _apply_cleaning(df):
    # phones: store digits in phone_raw; store E.164 in phone_e164
    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].apply(_clean_phone_digits)
    if "phone_e164" in df.columns:
        df["phone_e164"] = df["phone_e164"].apply(_to_e164_us)

    # postal code
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].apply(_clean_postal_code)

    # urls
    if "website" in df.columns:
        df["website"] = df["website"].apply(_canonical_url)
    if "map_url" in df.columns:
        df["map_url"] = df["map_url"].apply(_canonical_url)

    # categories to list
    if "categories" in df.columns:
        df["categories"] = df["categories"].apply(_clean_categories)

    return df


good_dataset_name_1 = _set_source_constant(good_dataset_name_1, "kaggle_380k")
good_dataset_name_2 = _set_source_constant(good_dataset_name_2, "uber_eats")
good_dataset_name_3 = _set_source_constant(good_dataset_name_3, "yelp")

good_dataset_name_1 = _apply_cleaning(good_dataset_name_1)
good_dataset_name_2 = _apply_cleaning(good_dataset_name_2)
good_dataset_name_3 = _apply_cleaning(good_dataset_name_3)

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# Resulting columns for all datasets will have the schema of dataset1
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

schema_correspondences = schema_matcher.match(good_dataset_name_1, good_dataset_name_2)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_2 = good_dataset_name_2.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(good_dataset_name_1, good_dataset_name_3)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_3 = good_dataset_name_3.rename(columns=rename_map)

# Re-apply cleaning after renaming
good_dataset_name_1 = _apply_cleaning(good_dataset_name_1)
good_dataset_name_2 = _apply_cleaning(good_dataset_name_2)
good_dataset_name_3 = _apply_cleaning(good_dataset_name_3)

# --------------------------------
# Blocking (MUST use provided configuration)
# --------------------------------

print("Performing Blocking")

blocker_k2u = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_2,
    on=["city"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_k2y = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    on=["city"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_u2y = StandardBlocker(
    good_dataset_name_2,
    good_dataset_name_3,
    on=["city"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching (MUST use provided matching configuration)
# --------------------------------

comparators_k2u = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="street", similarity_function="jaccard", preprocess=str.lower),
    NumericComparator(column="house_number", max_difference=2),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=str.lower,
        list_strategy="concatenate",
    ),
]

comparators_k2y = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="street", similarity_function="jaccard", preprocess=str.lower),
    NumericComparator(column="house_number", max_difference=2),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=str.lower,
        list_strategy="concatenate",
    ),
]

comparators_u2y = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="street", similarity_function="jaccard", preprocess=str.lower),
    NumericComparator(column="house_number", max_difference=2),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=str.lower,
        list_strategy="concatenate",
    ),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_k2u = rb_matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2,
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=[0.5, 0.2, 0.2, 0.1],
    threshold=0.7,
    id_column="id",
)

rb_correspondences_k2y = rb_matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3,
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=[0.5, 0.2, 0.2, 0.1],
    threshold=0.7,
    id_column="id",
)

rb_correspondences_u2y = rb_matcher.match(
    df_left=good_dataset_name_2,
    df_right=good_dataset_name_3,
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=[0.5, 0.2, 0.2, 0.1],
    threshold=0.7,
    id_column="id",
)

# --------------------------------
# Fusion (fix postal_code + phone_* accuracy by choosing best non-null, consistent values)
# --------------------------------

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_k2u, rb_correspondences_k2y, rb_correspondences_u2y],
    ignore_index=True,
)

# Trust: prefer more complete / canonical sources; Yelp often has correct phones; Kaggle often has website/map.
trust_map = {"kaggle_380k": 3, "yelp": 3, "uber_eats": 2}


def _trust(row):
    return trust_map.get(row.get("source", None), 0)


# Prefer value with highest trust; if tie, pick longest (helps avoid empty-ish strings)
def _prefer_trust_then_longest(values, rows=None):
    best_v = np.nan
    best_score = -1
    best_len = -1
    if rows is None:
        # fallback: longest non-null
        for v in values:
            if _is_na(v):
                continue
            l = len(str(v))
            if l > best_len:
                best_len = l
                best_v = v
        return best_v

    for v, r in zip(values, rows):
        if _is_na(v):
            continue
        score = _trust(r)
        l = len(str(v))
        if score > best_score or (score == best_score and l > best_len):
            best_score = score
            best_len = l
            best_v = v
    return best_v if best_score >= 0 else np.nan


def _best_postal(values, rows=None):
    # choose first valid 5-digit ZIP by trust
    cleaned = []
    for v in values:
        cleaned.append(_clean_postal_code(v))
    return _prefer_trust_then_longest(cleaned, rows=rows)


def _best_phone_raw(values, rows=None):
    cleaned = [(_clean_phone_digits(v)) for v in values]
    return _prefer_trust_then_longest(cleaned, rows=rows)


def _best_phone_e164(values, rows=None):
    cleaned = [(_to_e164_us(v)) for v in values]
    return _prefer_trust_then_longest(cleaned, rows=rows)


def _best_map_url(values, rows=None):
    cleaned = [(_canonical_url(v)) for v in values]
    return _prefer_trust_then_longest(cleaned, rows=rows)


def _best_website(values, rows=None):
    cleaned = [(_canonical_url(v)) for v in values]
    return _prefer_trust_then_longest(cleaned, rows=rows)


strategy = DataFusionStrategy("rule_based_fusion_strategy")

# keep original cluster id/source stable by trust
strategy.add_attribute_fuser("id", prefer_higher_trust, trust_function=_trust)
strategy.add_attribute_fuser("source", prefer_higher_trust, trust_function=_trust)

# core strings
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", longest_string)
strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", longest_string)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
strategy.add_attribute_fuser("country", longest_string)

# fixed: postal_code + phones + urls (custom)
strategy.add_attribute_fuser("postal_code", _best_postal)
strategy.add_attribute_fuser("phone_raw", _best_phone_raw)
strategy.add_attribute_fuser("phone_e164", _best_phone_e164)
strategy.add_attribute_fuser("map_url", _best_map_url)
strategy.add_attribute_fuser("website", _best_website)

# geo: trust
strategy.add_attribute_fuser("latitude", prefer_higher_trust, trust_function=_trust)
strategy.add_attribute_fuser("longitude", prefer_higher_trust, trust_function=_trust)

# categories
strategy.add_attribute_fuser("categories", union)

# ratings
strategy.add_attribute_fuser("rating", prefer_higher_trust, trust_function=_trust)
strategy.add_attribute_fuser("rating_count", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# Final pass to ensure fused outputs are in canonical format (postal/phones/urls/categories)
rb_fused_standard_blocker = _apply_cleaning(rb_fused_standard_blocker)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

