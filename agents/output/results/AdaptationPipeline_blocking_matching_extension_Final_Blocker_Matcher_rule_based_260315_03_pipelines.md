# Pipeline Snapshots

notebook_name=AdaptationPipeline_blocking_matching_extension_Final_Blocker_Matcher
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=45.89%
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

schema_correspondences = schema_matcher.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# Ensure blocking/matching id column exists as required by provided configuration
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
# Matching using precomputed matching configuration
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
strategy.add_attribute_fuser("rating", longest_string)

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
node_index=11
node_name=execute_pipeline
accuracy_score=40.52%
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


def strip_only(x):
    return str(x).strip()


def normalize_phone(x):
    if pd.isna(x):
        return x
    s = str(x).strip()
    if s.lower() == "nan":
        return np.nan
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return np.nan
    if s.startswith("+"):
        return "+" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    if len(digits) == 10:
        return "+1" + digits
    return "+" + digits


def normalize_postal_code(x):
    if pd.isna(x):
        return x
    s = str(x).strip()
    if s.lower() == "nan" or s == "":
        return np.nan
    return s


def parse_categories(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(v).strip().lower() for v in x if pd.notna(v)]
    s = str(x).strip()
    if s.lower() == "nan" or s == "":
        return []
    s = s.strip("[]")
    if not s:
        return []
    parts = [p.strip().strip("'").strip('"').lower() for p in s.split(",")]
    return [p for p in parts if p]


def coalesce_columns(df, preferred_order, target_col):
    existing = [c for c in preferred_order if c in df.columns]
    if not existing:
        return
    series = df[existing[0]]
    for c in existing[1:]:
        series = series.combine_first(df[c])
    df[target_col] = series


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

schema_correspondences = schema_matcher.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# Restore required id column explicitly and retain original record ids
for df in [kaggle_small, uber_eats_small, yelp_small]:
    df["kaggle380k_id"] = df["id"]
    df["_id"] = df["id"]

# Repair important columns after schema matching by coalescing semantically similar fields if duplicates exist
for df in [kaggle_small, uber_eats_small, yelp_small]:
    coalesce_columns(df, ["phone_e164"], "phone_e164")
    coalesce_columns(df, ["phone_raw"], "phone_raw")
    coalesce_columns(df, ["postal_code"], "postal_code")
    coalesce_columns(df, ["house_number"], "house_number")
    coalesce_columns(df, ["map_url"], "map_url")
    coalesce_columns(df, ["website"], "website")
    coalesce_columns(df, ["rating"], "rating")
    coalesce_columns(df, ["rating_count"], "rating_count")
    coalesce_columns(df, ["source"], "source")

# Normalize fields used in blocking/matching/fusion
for df in [kaggle_small, uber_eats_small, yelp_small]:
    if "phone_e164" in df.columns:
        df["phone_e164"] = df["phone_e164"].apply(normalize_phone)
    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].apply(normalize_phone)
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].apply(normalize_postal_code)
    if "categories" in df.columns:
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
# Matching using precomputed matching configuration
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

strategy.add_attribute_fuser("id", longest_string)
strategy.add_attribute_fuser("_id", longest_string)
strategy.add_attribute_fuser("kaggle380k_id", longest_string)
strategy.add_attribute_fuser("source", longest_string)
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
    id_column="kaggle380k_id",
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
node_index=16
node_name=execute_pipeline
accuracy_score=42.55%
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
    maximum,
)
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
import ast
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
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


def strip_only(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_phone(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        return np.nan
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    if s.startswith("+"):
        return "+" + digits
    return "+" + digits


def normalize_postal_code(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    return s


def parse_categories(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(v).strip().lower() for v in x if pd.notna(v) and str(v).strip() != ""]
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(v).strip().lower() for v in parsed if pd.notna(v) and str(v).strip() != ""]
    except Exception:
        pass
    s = s.strip("[]")
    parts = [p.strip().strip("'").strip('"').lower() for p in s.split(",")]
    return [p for p in parts if p]


def first_non_null(series):
    vals = series.dropna()
    if len(vals) == 0:
        return np.nan
    return vals.iloc[0]


def prefer_source(series, source_series, preferred_sources):
    tmp = pd.DataFrame({"value": series, "source": source_series})
    tmp = tmp[tmp["value"].notna()]
    if tmp.empty:
        return np.nan
    for src in preferred_sources:
        sub = tmp[tmp["source"] == src]
        if not sub.empty:
            return sub.iloc[0]["value"]
    return tmp.iloc[0]["value"]


def fuse_categories(series):
    values = []
    for v in series.dropna():
        if isinstance(v, list):
            values.extend(v)
        else:
            parsed = parse_categories(v)
            values.extend(parsed)
    seen = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


def fuse_source(series):
    vals = [str(v) for v in series.dropna() if str(v).strip() != ""]
    if not vals:
        return np.nan
    priority = ["kaggle_380k", "yelp", "uber_eats"]
    for p in priority:
        if p in vals:
            return p
    return vals[0]


def fuse_house_number(series):
    vals = [str(v).strip() for v in series.dropna() if str(v).strip() != ""]
    if not vals:
        return np.nan
    exact_numeric = [v for v in vals if v.replace("-", "").isdigit()]
    if exact_numeric:
        exact_numeric = sorted(exact_numeric, key=lambda x: (-len(x), x))
        return exact_numeric[0]
    return max(vals, key=len)


def fuse_postal_code(series):
    vals = [str(v).strip() for v in series.dropna() if str(v).strip() != ""]
    if not vals:
        return np.nan
    five_digit = [v for v in vals if len(v) == 5 and v.isdigit()]
    if five_digit:
        return five_digit[0]
    return vals[0]


def fuse_phone(series):
    vals = [normalize_phone(v) for v in series if pd.notna(v)]
    vals = [v for v in vals if pd.notna(v)]
    if not vals:
        return np.nan
    vals = sorted(vals, key=len, reverse=True)
    return vals[0]


def fuse_lat(series):
    vals = [v for v in series.dropna()]
    if not vals:
        return np.nan
    return vals[0]


def fuse_lon(series):
    vals = [v for v in series.dropna()]
    if not vals:
        return np.nan
    return vals[0]


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

schema_correspondences = schema_matcher.match(kaggle_small, uber_eats_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
uber_eats_small = uber_eats_small.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(kaggle_small, yelp_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
yelp_small = yelp_small.rename(columns=rename_map)

# Ensure shared id column required by config exists and is populated
for df in [kaggle_small, uber_eats_small, yelp_small]:
    df["kaggle380k_id"] = df["id"]

# Normalize important columns
for df in [kaggle_small, uber_eats_small, yelp_small]:
    if "phone_e164" in df.columns:
        df["phone_e164"] = df["phone_e164"].apply(normalize_phone)
    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].apply(normalize_phone)
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].apply(normalize_postal_code)
    if "categories" in df.columns:
        df["categories"] = df["categories"].apply(parse_categories)

datasets = [kaggle_small, uber_eats_small, yelp_small]

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
# Matching using precomputed matching configuration
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

# Keep only the canonical id required by evaluation/config and avoid introducing wrong extra ids
strategy.add_attribute_fuser("kaggle380k_id", longest_string)
strategy.add_attribute_fuser("source", fuse_source)
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", longest_string)
strategy.add_attribute_fuser("website", longest_string)
strategy.add_attribute_fuser("map_url", longest_string)
strategy.add_attribute_fuser("phone_raw", fuse_phone)
strategy.add_attribute_fuser("phone_e164", fuse_phone)
strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", fuse_house_number)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
strategy.add_attribute_fuser("postal_code", fuse_postal_code)
strategy.add_attribute_fuser("country", longest_string)
strategy.add_attribute_fuser("latitude", fuse_lat)
strategy.add_attribute_fuser("longitude", fuse_lon)
strategy.add_attribute_fuser("categories", fuse_categories)
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

# Ensure output schema does not contain known harmful extra id columns
for col in ["id", "_id"]:
    if col in rb_fused_standard_blocker.columns:
        rb_fused_standard_blocker = rb_fused_standard_blocker.drop(columns=[col])

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

