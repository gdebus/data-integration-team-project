# Pipeline Snapshots

notebook_name=Agent II
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=29.41%
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import (
    EmbeddingBlocker,
    StringComparator,
    NumericComparator,
    DateComparator,
    RuleBasedMatcher,
)
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    maximum,
)
import pandas as pd
import os


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

dbpedia = load_csv(
    DATA_DIR + "dbpedia.csv",
    name="dbpedia",
)

metacritic = load_csv(
    DATA_DIR + "metacritic.csv",
    name="metacritic",
)

sales = load_csv(
    DATA_DIR + "sales.csv",
    name="sales",
)

datasets = [dbpedia, metacritic, sales]


# --------------------------------
# Schema already aligned from schema-matching output
# The resulting columns should follow the schema of dataset1 in the example,
# but here the provided files are already schema-matched.
# --------------------------------

print("Schema already matched")


# --------------------------------
# Perform Blocking
# YOU MUST USE THE PRECOMPUTED BLOCKERS PROVIDED
# semantic_similarity -> EmbeddingBlocker
# --------------------------------

print("Performing Blocking")

blocker_dbpedia_sales = EmbeddingBlocker(
    dbpedia,
    sales,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metacritic_dbpedia = EmbeddingBlocker(
    metacritic,
    dbpedia,
    text_cols=["name", "developer", "platform"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metacritic_sales = EmbeddingBlocker(
    metacritic,
    sales,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration from precomputed settings
# --------------------------------

lower_strip = lambda x: str(x).lower().strip() if pd.notnull(x) else x

comparators_dbpedia_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
]

comparators_metacritic_dbpedia = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=366,
    ),
]

comparators_metacritic_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="criticScore",
        max_difference=5.0,
    ),
    NumericComparator(
        column="userScore",
        max_difference=1.0,
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_dbpedia_sales = matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=blocker_dbpedia_sales,
    comparators=comparators_dbpedia_sales,
    weights=[0.45, 0.2, 0.15, 0.2],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_metacritic_dbpedia = matcher.match(
    df_left=metacritic,
    df_right=dbpedia,
    candidates=blocker_metacritic_dbpedia,
    comparators=comparators_metacritic_dbpedia,
    weights=[0.45, 0.2, 0.2, 0.15],
    threshold=0.8,
    id_column="id",
)

rb_correspondences_metacritic_sales = matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=blocker_metacritic_sales,
    comparators=comparators_metacritic_sales,
    weights=[0.35, 0.2, 0.15, 0.12, 0.08, 0.1],
    threshold=0.78,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_dbpedia_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_dbpedia_sales.csv"),
    index=False,
)

rb_correspondences_metacritic_dbpedia.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_dbpedia.csv"),
    index=False,
)

rb_correspondences_metacritic_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_sales.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_dbpedia_sales,
        rb_correspondences_metacritic_dbpedia,
        rb_correspondences_metacritic_sales,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("releaseYear", longest_string)
strategy.add_attribute_fuser("developer", longest_string)
strategy.add_attribute_fuser("platform", longest_string)
strategy.add_attribute_fuser("series", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("criticScore", maximum)
strategy.add_attribute_fuser("userScore", maximum)
strategy.add_attribute_fuser("ESRB", longest_string)
strategy.add_attribute_fuser("globalSales", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[dbpedia, metacritic, sales],
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
node_index=11
node_name=execute_pipeline
accuracy_score=35.29%
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import (
    EmbeddingBlocker,
    StringComparator,
    NumericComparator,
    DateComparator,
    RuleBasedMatcher,
)
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    maximum,
)
import pandas as pd
import numpy as np
import os


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

dbpedia = load_csv(
    DATA_DIR + "dbpedia.csv",
    name="dbpedia",
)

metacritic = load_csv(
    DATA_DIR + "metacritic.csv",
    name="metacritic",
)

sales = load_csv(
    DATA_DIR + "sales.csv",
    name="sales",
)

# --------------------------------
# Data cleaning / normalization to improve matching and fusion quality
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip() if pd.notnull(x) else x


def normalize_text(x):
    if pd.isnull(x):
        return x
    x = str(x).strip()
    if x.lower() in {"nan", "none", "null", ""}:
        return np.nan
    return x


def normalize_year_date(x):
    if pd.isnull(x):
        return x
    dt = pd.to_datetime(x, errors="coerce")
    return dt.strftime("%Y-%m-%d") if pd.notnull(dt) else x


def normalize_platform(x):
    if pd.isnull(x):
        return x
    s = str(x).strip().lower()
    replacements = {
        "ps": "playstation",
        "ps1": "playstation",
        "ps2": "playstation 2",
        "ps3": "playstation 3",
        "ps4": "playstation 4",
        "ps5": "playstation 5",
        "xb": "xbox",
        "xbone": "xbox one",
        "x360": "xbox 360",
    }
    s = replacements.get(s, s)
    return s


for df in [dbpedia, metacritic, sales]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(normalize_text)

    if "releaseYear" in df.columns:
        df["releaseYear"] = df["releaseYear"].apply(normalize_year_date)

    if "platform" in df.columns:
        df["platform"] = df["platform"].apply(normalize_platform)

    if "name" in df.columns:
        df["name"] = df["name"].astype("object")
    if "developer" in df.columns:
        df["developer"] = df["developer"].astype("object")


datasets = [dbpedia, metacritic, sales]

print("Schema already matched")


# --------------------------------
# Perform Blocking
# YOU MUST USE THE PRECOMPUTED BLOCKERS PROVIDED
# semantic_similarity -> EmbeddingBlocker
# --------------------------------

print("Performing Blocking")

blocker_dbpedia_sales = EmbeddingBlocker(
    dbpedia,
    sales,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metacritic_dbpedia = EmbeddingBlocker(
    metacritic,
    dbpedia,
    text_cols=["name", "developer", "platform"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metacritic_sales = EmbeddingBlocker(
    metacritic,
    sales,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration from precomputed settings
# --------------------------------

comparators_dbpedia_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
]

comparators_metacritic_dbpedia = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=366,
    ),
]

comparators_metacritic_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="criticScore",
        max_difference=5.0,
    ),
    NumericComparator(
        column="userScore",
        max_difference=1.0,
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_dbpedia_sales = matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=blocker_dbpedia_sales,
    comparators=comparators_dbpedia_sales,
    weights=[0.45, 0.2, 0.15, 0.2],
    threshold=0.78,
    id_column="id",
)

rb_correspondences_metacritic_dbpedia = matcher.match(
    df_left=metacritic,
    df_right=dbpedia,
    candidates=blocker_metacritic_dbpedia,
    comparators=comparators_metacritic_dbpedia,
    weights=[0.45, 0.2, 0.2, 0.15],
    threshold=0.82,
    id_column="id",
)

rb_correspondences_metacritic_sales = matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=blocker_metacritic_sales,
    comparators=comparators_metacritic_sales,
    weights=[0.35, 0.2, 0.15, 0.12, 0.08, 0.1],
    threshold=0.82,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_dbpedia_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_dbpedia_sales.csv"),
    index=False,
)

rb_correspondences_metacritic_dbpedia.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_dbpedia.csv"),
    index=False,
)

rb_correspondences_metacritic_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_sales.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_dbpedia_sales,
        rb_correspondences_metacritic_dbpedia,
        rb_correspondences_metacritic_sales,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# Prefer richer schema from sales/metacritic over dbpedia where possible
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("releaseYear", longest_string)
strategy.add_attribute_fuser("developer", longest_string)
strategy.add_attribute_fuser("platform", longest_string)
strategy.add_attribute_fuser("series", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("criticScore", maximum)
strategy.add_attribute_fuser("userScore", maximum)
strategy.add_attribute_fuser("ESRB", longest_string)
strategy.add_attribute_fuser("globalSales", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[dbpedia, metacritic, sales],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=True,
)

# Post-fusion cleanup to improve attribute accuracy
if "releaseYear" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["releaseYear"] = rb_fused_standard_blocker["releaseYear"].apply(normalize_year_date)

if "platform" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["platform"] = rb_fused_standard_blocker["platform"].apply(normalize_text)

if "name" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["name"] = rb_fused_standard_blocker["name"].apply(normalize_text)

if "developer" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["developer"] = rb_fused_standard_blocker["developer"].apply(normalize_text)

if "publisher" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["publisher"] = rb_fused_standard_blocker["publisher"].apply(normalize_text)

if "ESRB" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["ESRB"] = rb_fused_standard_blocker["ESRB"].apply(normalize_text)

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
accuracy_score=73.95%
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import (
    EmbeddingBlocker,
    StringComparator,
    NumericComparator,
    DateComparator,
    RuleBasedMatcher,
)
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    maximum,
)
import pandas as pd
import numpy as np
import os


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

dbpedia = load_csv(
    DATA_DIR + "dbpedia.csv",
    name="dbpedia",
)

metacritic = load_csv(
    DATA_DIR + "metacritic.csv",
    name="metacritic",
)

sales = load_csv(
    DATA_DIR + "sales.csv",
    name="sales",
)


# --------------------------------
# Data cleaning / normalization
# --------------------------------

def lower_strip(x):
    if pd.isnull(x):
        return x
    return str(x).lower().strip()


def normalize_text(x):
    if pd.isnull(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    return s


def normalize_year_date(x):
    if pd.isnull(x):
        return np.nan
    dt = pd.to_datetime(x, errors="coerce")
    if pd.isnull(dt):
        return np.nan
    return dt.strftime("%Y-%m-%d")


def normalize_platform_value(x):
    if pd.isnull(x):
        return np.nan
    s = str(x).strip().lower()
    replacements = {
        "ps": "playstation",
        "ps1": "playstation",
        "ps2": "playstation 2",
        "ps3": "playstation 3",
        "ps4": "playstation 4",
        "ps5": "playstation 5",
        "xb": "xbox",
        "xbone": "xbox one",
        "x360": "xbox 360",
        "x-box 360": "xbox 360",
        "x-box one": "xbox one",
        "gba": "game boy advance",
        "gbc": "game boy color",
        "gc": "gamecube",
        "gcn": "gamecube",
    }
    return replacements.get(s, s)


def normalize_esrb_value(x):
    if pd.isnull(x):
        return np.nan
    s = str(x).strip().upper()
    replacements = {
        "E 10+": "E10+",
        "EVERYONE": "E",
        "EVERYONE 10+": "E10+",
        "TEEN": "T",
        "MATURE": "M",
        "MATURE 17+": "M",
        "ADULTS ONLY": "AO",
        "RATING PENDING": "RP",
        "EARLY CHILDHOOD": "EC",
    }
    return replacements.get(s, s)


def normalize_name_value(x):
    if pd.isnull(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    return s


for df in [dbpedia, metacritic, sales]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(normalize_text)

    if "name" in df.columns:
        df["name"] = df["name"].apply(normalize_name_value)

    if "releaseYear" in df.columns:
        df["releaseYear"] = df["releaseYear"].apply(normalize_year_date)

    if "platform" in df.columns:
        df["platform"] = df["platform"].apply(normalize_platform_value)

    if "ESRB" in df.columns:
        df["ESRB"] = df["ESRB"].apply(normalize_esrb_value)

    for num_col in ["criticScore", "userScore", "globalSales"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")


datasets = [dbpedia, metacritic, sales]

print("Schema already matched")


# --------------------------------
# Perform Blocking
# MUST use precomputed blocking configuration
# semantic_similarity -> EmbeddingBlocker
# --------------------------------

print("Performing Blocking")

blocker_dbpedia_sales = EmbeddingBlocker(
    dbpedia,
    sales,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metacritic_dbpedia = EmbeddingBlocker(
    metacritic,
    dbpedia,
    text_cols=["name", "developer", "platform"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metacritic_sales = EmbeddingBlocker(
    metacritic,
    sales,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration from precomputed settings
# --------------------------------

comparators_dbpedia_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
]

comparators_metacritic_dbpedia = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=366,
    ),
]

comparators_metacritic_sales = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="criticScore",
        max_difference=5.0,
    ),
    NumericComparator(
        column="userScore",
        max_difference=1.0,
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_dbpedia_sales = matcher.match(
    df_left=dbpedia,
    df_right=sales,
    candidates=blocker_dbpedia_sales,
    comparators=comparators_dbpedia_sales,
    weights=[0.45, 0.2, 0.15, 0.2],
    threshold=0.82,
    id_column="id",
)

rb_correspondences_metacritic_dbpedia = matcher.match(
    df_left=metacritic,
    df_right=dbpedia,
    candidates=blocker_metacritic_dbpedia,
    comparators=comparators_metacritic_dbpedia,
    weights=[0.45, 0.2, 0.2, 0.15],
    threshold=0.85,
    id_column="id",
)

rb_correspondences_metacritic_sales = matcher.match(
    df_left=metacritic,
    df_right=sales,
    candidates=blocker_metacritic_sales,
    comparators=comparators_metacritic_sales,
    weights=[0.35, 0.2, 0.15, 0.12, 0.08, 0.1],
    threshold=0.83,
    id_column="id",
)


# --------------------------------
# Reduce many-to-many links before fusion to avoid oversized cyclic groups
# --------------------------------

def deduplicate_pairwise_correspondences(corr_df):
    if corr_df is None or len(corr_df) == 0:
        return corr_df

    score_col = None
    for candidate_col in ["similarity", "score", "confidence", "final_score"]:
        if candidate_col in corr_df.columns:
            score_col = candidate_col
            break

    if score_col is None:
        return corr_df.drop_duplicates()

    left_col = None
    right_col = None
    for c in corr_df.columns:
        cl = c.lower()
        if cl in {"id_left", "left_id", "source_id", "ltable_id"}:
            left_col = c
        if cl in {"id_right", "right_id", "target_id", "rtable_id"}:
            right_col = c

    if left_col is None or right_col is None:
        id_cols = [c for c in corr_df.columns if c.lower() != score_col.lower() and "id" in c.lower()]
        if len(id_cols) >= 2:
            left_col, right_col = id_cols[0], id_cols[1]

    if left_col is None or right_col is None:
        return corr_df.drop_duplicates()

    corr_df = corr_df.sort_values(score_col, ascending=False).drop_duplicates(subset=[left_col], keep="first")
    corr_df = corr_df.sort_values(score_col, ascending=False).drop_duplicates(subset=[right_col], keep="first")
    corr_df = corr_df.drop_duplicates(subset=[left_col, right_col])

    return corr_df


rb_correspondences_dbpedia_sales = deduplicate_pairwise_correspondences(rb_correspondences_dbpedia_sales)
rb_correspondences_metacritic_dbpedia = deduplicate_pairwise_correspondences(rb_correspondences_metacritic_dbpedia)
rb_correspondences_metacritic_sales = deduplicate_pairwise_correspondences(rb_correspondences_metacritic_sales)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_dbpedia_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_dbpedia_sales.csv"),
    index=False,
)

rb_correspondences_metacritic_dbpedia.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_dbpedia.csv"),
    index=False,
)

rb_correspondences_metacritic_sales.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_sales.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_dbpedia_sales,
        rb_correspondences_metacritic_dbpedia,
        rb_correspondences_metacritic_sales,
    ],
    ignore_index=True,
).drop_duplicates()


# --------------------------------
# Data Fusion
# Prefer precise source-specific attributes by fusion choice
# --------------------------------

def first_non_null(values):
    vals = [v for v in values if pd.notnull(v)]
    return vals[0] if len(vals) > 0 else np.nan


def most_frequent_non_null(values):
    vals = [v for v in values if pd.notnull(v)]
    if len(vals) == 0:
        return np.nan
    return pd.Series(vals).mode().iloc[0]


def best_name(values):
    vals = [v for v in values if pd.notnull(v)]
    if len(vals) == 0:
        return np.nan
    vals = sorted(vals, key=lambda x: len(str(x).strip()), reverse=True)
    return vals[0]


strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", best_name)
strategy.add_attribute_fuser("releaseYear", most_frequent_non_null)
strategy.add_attribute_fuser("developer", most_frequent_non_null)
strategy.add_attribute_fuser("platform", most_frequent_non_null)
strategy.add_attribute_fuser("series", union)
strategy.add_attribute_fuser("publisher", first_non_null)
strategy.add_attribute_fuser("criticScore", maximum)
strategy.add_attribute_fuser("userScore", maximum)
strategy.add_attribute_fuser("ESRB", most_frequent_non_null)
strategy.add_attribute_fuser("globalSales", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[dbpedia, metacritic, sales],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)


# --------------------------------
# Post-fusion cleanup
# --------------------------------

if "releaseYear" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["releaseYear"] = rb_fused_standard_blocker["releaseYear"].apply(normalize_year_date)

if "platform" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["platform"] = rb_fused_standard_blocker["platform"].apply(normalize_platform_value)

if "name" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["name"] = rb_fused_standard_blocker["name"].apply(normalize_name_value)

if "developer" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["developer"] = rb_fused_standard_blocker["developer"].apply(normalize_text)

if "publisher" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["publisher"] = rb_fused_standard_blocker["publisher"].apply(normalize_text)

if "ESRB" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["ESRB"] = rb_fused_standard_blocker["ESRB"].apply(normalize_esrb_value)

for num_col in ["criticScore", "userScore", "globalSales"]:
    if num_col in rb_fused_standard_blocker.columns:
        rb_fused_standard_blocker[num_col] = pd.to_numeric(
            rb_fused_standard_blocker[num_col], errors="coerce"
        )

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

