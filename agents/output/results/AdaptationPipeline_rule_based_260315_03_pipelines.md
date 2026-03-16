# Pipeline Snapshots

notebook_name=AdaptationPipeline
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=3
node_name=execute_pipeline
accuracy_score=59.29%
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

DATA_DIR = "input/datasets/books/"

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
# The resulting columns for all datasets will have the schema of amazon_small.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

schema_correspondences = matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Normalize blocking/matching columns
# Use strong identity signals: title + author + publish_year
# --------------------------------

def normalize_text(series):
    return (
        series.astype("string")
        .fillna("")
        .str.lower()
        .str.replace(r"&amp;", "and", regex=True)
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

def normalize_author(series):
    return (
        series.astype("string")
        .fillna("")
        .str.replace(r"\(.*?\)", "", regex=True)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

for df in [amazon_small, goodreads_small, metabooks_small]:
    df["title_norm"] = normalize_text(df["title"])
    df["author_norm"] = normalize_author(df["author"])
    if "publish_year" in df.columns:
        df["publish_year_num"] = pd.to_numeric(df["publish_year"], errors="coerce")

# --------------------------------
# Perform Blocking
# Use precomputed-style configuration adapted to the datasets.
# Title is the strongest shared identity signal with high coverage.
# --------------------------------

print("Performing Blocking")

blocker_a2g = TokenBlocker(
    amazon_small,
    goodreads_small,
    column="title_norm",
    min_token_len=3,
    ngram_size=2,
    ngram_type="word",
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_a2m = TokenBlocker(
    amazon_small,
    metabooks_small,
    column="title_norm",
    min_token_len=3,
    ngram_size=2,
    ngram_type="word",
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_g2m = TokenBlocker(
    goodreads_small,
    metabooks_small,
    column="title_norm",
    min_token_len=3,
    ngram_size=2,
    ngram_type="word",
    id_column="id",
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# Matching configuration
# Use title, author, publish_year, publisher as aligned attributes.
# --------------------------------

comparators_a2g = [
    StringComparator(
        column="title_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="author_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="publish_year_num",
        max_difference=1,
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
]

comparators_a2m = [
    StringComparator(
        column="title_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="author_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="publish_year_num",
        max_difference=1,
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
]

comparators_g2m = [
    StringComparator(
        column="title_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="author_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="publish_year_num",
        max_difference=1,
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_a2g = matcher.match(
    df_left=amazon_small,
    df_right=goodreads_small,
    candidates=blocker_a2g,
    comparators=comparators_a2g,
    weights=[0.5, 0.3, 0.1, 0.1],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_a2m = matcher.match(
    df_left=amazon_small,
    df_right=metabooks_small,
    candidates=blocker_a2m,
    comparators=comparators_a2m,
    weights=[0.5, 0.3, 0.1, 0.1],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_g2m = matcher.match(
    df_left=goodreads_small,
    df_right=metabooks_small,
    candidates=blocker_g2m,
    comparators=comparators_g2m,
    weights=[0.5, 0.3, 0.1, 0.1],
    threshold=0.75,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_a2g.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_amazon_small_goodreads_small.csv",
    ),
    index=False,
)

rb_correspondences_a2m.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_amazon_small_metabooks_small.csv",
    ),
    index=False,
)

rb_correspondences_g2m.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_metabooks_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_a2g, rb_correspondences_a2m, rb_correspondences_g2m],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publish_year", maximum)
strategy.add_attribute_fuser("publisher", longest_string)

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
node_index=8
node_name=execute_pipeline
accuracy_score=53.57%
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
import html
import re

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "input/datasets/books/"

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
# The resulting columns for all datasets will have the schema of amazon_small.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

schema_correspondences = matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Normalize columns for blocking and matching
# Strong entity identity signal: title + author (+ year as disambiguator)
# Also normalize fusion-relevant attributes to improve output quality.
# --------------------------------

def _safe_string(series):
    return series.astype("string").fillna("")

def normalize_text_value(x):
    if pd.isna(x):
        return ""
    x = html.unescape(str(x))
    x = x.lower()
    x = re.sub(r"&", " and ", x)
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def normalize_author_value(x):
    if pd.isna(x):
        return ""
    x = html.unescape(str(x))
    x = re.sub(r"\(.*?\)", "", x)
    x = x.lower()
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def normalize_publisher_value(x):
    if pd.isna(x):
        return ""
    x = html.unescape(str(x))
    x = x.lower()
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\b(co|inc|ltd|llc|press|books|book|publishing|publishers|publisher)\b", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def normalize_language_value(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return pd.NA
    return x.title()

def normalize_genres_value(x):
    if pd.isna(x):
        return []
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return []
    parts = [p.strip().lower() for p in x.split(",")]
    parts = [re.sub(r"\s+", " ", p) for p in parts if p.strip()]
    return sorted(list(set(parts)))

def choose_longest_nonempty(values):
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s != "" and s.lower() != "nan":
            cleaned.append(s)
    if not cleaned:
        return pd.NA
    return max(cleaned, key=len)

def choose_best_author(values):
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s != "" and s.lower() != "nan":
            cleaned.append(s)
    if not cleaned:
        return pd.NA
    cleaned_sorted = sorted(cleaned, key=lambda s: (("(" in s or ")" in s), -len(s)))
    return cleaned_sorted[0]

def choose_best_title(values):
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s != "" and s.lower() != "nan":
            cleaned.append(s)
    if not cleaned:
        return pd.NA
    return max(cleaned, key=len)

def choose_best_publisher(values):
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s != "" and s.lower() != "nan":
            cleaned.append(s)
    if not cleaned:
        return pd.NA
    return max(cleaned, key=len)

def choose_publish_year(values):
    nums = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if len(nums) == 0:
        return pd.NA
    counts = nums.value_counts()
    max_count = counts.max()
    candidates = counts[counts == max_count].index.tolist()
    return int(min(candidates))

def choose_page_count(values):
    nums = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if len(nums) == 0:
        return pd.NA
    counts = nums.value_counts()
    max_count = counts.max()
    candidates = counts[counts == max_count].index.tolist()
    return int(max(candidates))

def choose_language(values):
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s != "" and s.lower() != "nan":
            cleaned.append(s.title())
    if not cleaned:
        return pd.NA
    return pd.Series(cleaned).mode().iloc[0]

for df in [amazon_small, goodreads_small, metabooks_small]:
    df["title"] = _safe_string(df["title"]).apply(lambda x: html.unescape(str(x)).strip())
    df["author"] = _safe_string(df["author"]).apply(lambda x: html.unescape(str(x)).strip())

    if "publisher" in df.columns:
        df["publisher"] = _safe_string(df["publisher"]).replace({"nan": pd.NA, "": pd.NA})

    if "language" not in df.columns:
        df["language"] = pd.NA
    df["language"] = df["language"].apply(normalize_language_value)

    if "genres" not in df.columns:
        df["genres"] = [[] for _ in range(len(df))]
    else:
        df["genres"] = df["genres"].apply(normalize_genres_value)

    if "page_count" not in df.columns:
        df["page_count"] = pd.NA
    df["page_count_num"] = pd.to_numeric(df["page_count"], errors="coerce")

    if "publish_year" not in df.columns:
        df["publish_year"] = pd.NA
    df["publish_year_num"] = pd.to_numeric(df["publish_year"], errors="coerce")

    df["title_norm"] = df["title"].apply(normalize_text_value)
    df["author_norm"] = df["author"].apply(normalize_author_value)
    df["publisher_norm"] = df["publisher"].apply(normalize_publisher_value) if "publisher" in df.columns else ""

# --------------------------------
# Perform Blocking
# Use title as primary blocker and author as disambiguating signal.
# --------------------------------

print("Performing Blocking")

blocker_a2g = TokenBlocker(
    amazon_small,
    goodreads_small,
    column="title_norm",
    min_token_len=3,
    ngram_size=2,
    ngram_type="word",
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_a2m = TokenBlocker(
    amazon_small,
    metabooks_small,
    column="title_norm",
    min_token_len=3,
    ngram_size=2,
    ngram_type="word",
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_g2m = TokenBlocker(
    goodreads_small,
    metabooks_small,
    column="title_norm",
    min_token_len=3,
    ngram_size=2,
    ngram_type="word",
    id_column="id",
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# Matching configuration
# Increase precision to reduce incorrect merges that hurt fusion accuracy.
# --------------------------------

comparators_a2g = [
    StringComparator(
        column="title_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="author_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="publish_year_num",
        max_difference=1,
    ),
    StringComparator(
        column="publisher_norm",
        similarity_function="jaccard",
    ),
]

comparators_a2m = [
    StringComparator(
        column="title_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="author_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="publish_year_num",
        max_difference=1,
    ),
    StringComparator(
        column="publisher_norm",
        similarity_function="jaccard",
    ),
]

comparators_g2m = [
    StringComparator(
        column="title_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="author_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="publish_year_num",
        max_difference=1,
    ),
    StringComparator(
        column="publisher_norm",
        similarity_function="jaccard",
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_a2g = matcher.match(
    df_left=amazon_small,
    df_right=goodreads_small,
    candidates=blocker_a2g,
    comparators=comparators_a2g,
    weights=[0.6, 0.25, 0.1, 0.05],
    threshold=0.82,
    id_column="id",
)

rb_correspondences_a2m = matcher.match(
    df_left=amazon_small,
    df_right=metabooks_small,
    candidates=blocker_a2m,
    comparators=comparators_a2m,
    weights=[0.6, 0.25, 0.1, 0.05],
    threshold=0.82,
    id_column="id",
)

rb_correspondences_g2m = matcher.match(
    df_left=goodreads_small,
    df_right=metabooks_small,
    candidates=blocker_g2m,
    comparators=comparators_g2m,
    weights=[0.6, 0.25, 0.1, 0.05],
    threshold=0.82,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_a2g.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_amazon_small_goodreads_small.csv",
    ),
    index=False,
)

rb_correspondences_a2m.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_amazon_small_metabooks_small.csv",
    ),
    index=False,
)

rb_correspondences_g2m.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_metabooks_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_a2g, rb_correspondences_a2m, rb_correspondences_g2m],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("title", choose_best_title)
strategy.add_attribute_fuser("author", choose_best_author)
strategy.add_attribute_fuser("publish_year", choose_publish_year)
strategy.add_attribute_fuser("publisher", choose_best_publisher)
strategy.add_attribute_fuser("language", choose_language)
strategy.add_attribute_fuser("page_count", choose_page_count)
strategy.add_attribute_fuser("genres", union)

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

if "genres" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["genres"] = rb_fused_standard_blocker["genres"].apply(
        lambda x: ", ".join(x) if isinstance(x, (list, set, tuple)) else x
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
accuracy_score=53.57%
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
import html
import re

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "input/datasets/books/"

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
# The resulting columns for all datasets will have the schema of amazon_small.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

schema_correspondences = matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Normalize columns for blocking, matching and fusion
# Use strong identity signals: title + author + publish_year
# --------------------------------

def _safe_string(series):
    return series.astype("string")

def normalize_text_value(x):
    if pd.isna(x):
        return ""
    x = html.unescape(str(x))
    x = x.lower()
    x = x.replace("&", " and ")
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\b(the|a|an)\b", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def normalize_author_value(x):
    if pd.isna(x):
        return ""
    x = html.unescape(str(x))
    x = re.sub(r"\(.*?\)", "", x)
    x = x.lower()
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def normalize_publisher_value(x):
    if pd.isna(x):
        return ""
    x = html.unescape(str(x))
    x = x.lower()
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\b(company|co|inc|ltd|llc)\b", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def normalize_language_value(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return pd.NA
    x = x.lower()
    mapping = {
        "english": "English",
        "eng": "English",
        "en": "English",
    }
    return mapping.get(x, x.title())

def normalize_genres_value(x):
    if pd.isna(x):
        return []
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return []
    parts = [p.strip() for p in x.split(",") if p.strip()]
    cleaned = []
    for p in parts:
        p = html.unescape(p).lower()
        p = re.sub(r"[^a-z0-9\s]", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            cleaned.append(p)
    return list(dict.fromkeys(cleaned))

def choose_best_title(values):
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        s = html.unescape(str(v)).strip()
        if s and s.lower() != "nan":
            cleaned.append(s)
    if not cleaned:
        return pd.NA
    cleaned = sorted(cleaned, key=lambda s: (-len(s), s))
    return cleaned[0]

def choose_best_author(values):
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        s = html.unescape(str(v)).strip()
        if s and s.lower() != "nan":
            cleaned.append(s)
    if not cleaned:
        return pd.NA
    cleaned = sorted(
        cleaned,
        key=lambda s: (
            "(" in s or ")" in s,
            len(re.sub(r"\(.*?\)", "", s).strip()) == 0,
            -len(re.sub(r"\(.*?\)", "", s).strip()),
        ),
    )
    return re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", cleaned[0])).strip()

def choose_publish_year(values):
    nums = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    nums = nums[(nums >= 1400) & (nums <= 2100)]
    if len(nums) == 0:
        return pd.NA
    counts = nums.astype(int).value_counts()
    max_count = counts.max()
    candidates = sorted(counts[counts == max_count].index.tolist())
    return int(candidates[0])

def choose_page_count(values):
    nums = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    nums = nums[(nums > 0) & (nums < 5000)]
    if len(nums) == 0:
        return pd.NA
    counts = nums.astype(int).value_counts()
    max_count = counts.max()
    candidates = sorted(counts[counts == max_count].index.tolist())
    return int(candidates[0])

def choose_best_publisher(values):
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        s = html.unescape(str(v)).strip()
        if s and s.lower() != "nan":
            cleaned.append(s)
    if not cleaned:
        return pd.NA
    cleaned = sorted(cleaned, key=lambda s: (-len(normalize_publisher_value(s)), -len(s), s))
    return cleaned[0]

def choose_language(values):
    cleaned = []
    for v in values:
        val = normalize_language_value(v)
        if not pd.isna(val):
            cleaned.append(val)
    if not cleaned:
        return pd.NA
    return pd.Series(cleaned).mode().iloc[0]

def choose_genres(values):
    all_items = []
    for v in values:
        if pd.isna(v):
            continue
        if isinstance(v, (list, tuple, set)):
            items = list(v)
        else:
            items = normalize_genres_value(v)
        for item in items:
            item = str(item).strip()
            if item:
                all_items.append(item)
    if not all_items:
        return pd.NA
    counts = pd.Series(all_items).value_counts()
    selected = counts[counts >= 2].index.tolist()
    if not selected:
        selected = counts.head(3).index.tolist()
    return ", ".join(selected)

for df in [amazon_small, goodreads_small, metabooks_small]:
    df["title"] = _safe_string(df["title"]).apply(lambda x: html.unescape(str(x)).strip() if not pd.isna(x) else pd.NA)
    df["author"] = _safe_string(df["author"]).apply(lambda x: html.unescape(str(x)).strip() if not pd.isna(x) else pd.NA)

    if "publisher" not in df.columns:
        df["publisher"] = pd.NA
    df["publisher"] = _safe_string(df["publisher"]).apply(
        lambda x: pd.NA if pd.isna(x) or str(x).strip() == "" or str(x).strip().lower() == "nan" else html.unescape(str(x)).strip()
    )

    if "language" not in df.columns:
        df["language"] = pd.NA
    df["language"] = df["language"].apply(normalize_language_value)

    if "genres" not in df.columns:
        df["genres"] = [[] for _ in range(len(df))]
    else:
        df["genres"] = df["genres"].apply(normalize_genres_value)

    if "page_count" not in df.columns:
        df["page_count"] = pd.NA
    df["page_count_num"] = pd.to_numeric(df["page_count"], errors="coerce")

    if "publish_year" not in df.columns:
        df["publish_year"] = pd.NA
    df["publish_year_num"] = pd.to_numeric(df["publish_year"], errors="coerce")

    df["title_norm"] = df["title"].apply(normalize_text_value)
    df["author_norm"] = df["author"].apply(normalize_author_value)
    df["publisher_norm"] = df["publisher"].apply(normalize_publisher_value)
    df["title_author_key"] = (
        df["title_norm"].fillna("").astype(str).str[:40] + " " + df["author_norm"].fillna("").astype(str).str[:20]
    ).str.strip()

# --------------------------------
# Perform Blocking
# Strong blocker based on canonical title, with title+author key for better precision
# --------------------------------

print("Performing Blocking")

blocker_a2g = SortedNeighbourhoodBlocker(
    amazon_small,
    goodreads_small,
    key="title_norm",
    window=15,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_a2m = SortedNeighbourhoodBlocker(
    amazon_small,
    metabooks_small,
    key="title_norm",
    window=15,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_g2m = SortedNeighbourhoodBlocker(
    goodreads_small,
    metabooks_small,
    key="title_norm",
    window=15,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# Matching configuration
# Improve precision to avoid incorrect cluster merges that degrade fusion.
# --------------------------------

comparators_a2g = [
    StringComparator(
        column="title_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="author_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="publish_year_num",
        max_difference=1,
    ),
    StringComparator(
        column="publisher_norm",
        similarity_function="jaccard",
    ),
]

comparators_a2m = [
    StringComparator(
        column="title_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="author_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="publish_year_num",
        max_difference=1,
    ),
    StringComparator(
        column="publisher_norm",
        similarity_function="jaccard",
    ),
]

comparators_g2m = [
    StringComparator(
        column="title_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="author_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="publish_year_num",
        max_difference=1,
    ),
    StringComparator(
        column="publisher_norm",
        similarity_function="jaccard",
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_a2g = matcher.match(
    df_left=amazon_small,
    df_right=goodreads_small,
    candidates=blocker_a2g,
    comparators=comparators_a2g,
    weights=[0.58, 0.27, 0.1, 0.05],
    threshold=0.86,
    id_column="id",
)

rb_correspondences_a2m = matcher.match(
    df_left=amazon_small,
    df_right=metabooks_small,
    candidates=blocker_a2m,
    comparators=comparators_a2m,
    weights=[0.58, 0.27, 0.1, 0.05],
    threshold=0.86,
    id_column="id",
)

rb_correspondences_g2m = matcher.match(
    df_left=goodreads_small,
    df_right=metabooks_small,
    candidates=blocker_g2m,
    comparators=comparators_g2m,
    weights=[0.58, 0.27, 0.1, 0.05],
    threshold=0.86,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_a2g.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_amazon_small_goodreads_small.csv",
    ),
    index=False,
)

rb_correspondences_a2m.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_amazon_small_metabooks_small.csv",
    ),
    index=False,
)

rb_correspondences_g2m.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_metabooks_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_a2g, rb_correspondences_a2m, rb_correspondences_g2m],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("title", choose_best_title)
strategy.add_attribute_fuser("author", choose_best_author)
strategy.add_attribute_fuser("publish_year", choose_publish_year)
strategy.add_attribute_fuser("publisher", choose_best_publisher)
strategy.add_attribute_fuser("language", choose_language)
strategy.add_attribute_fuser("page_count", choose_page_count)
strategy.add_attribute_fuser("genres", choose_genres)

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
PIPELINE SNAPSHOT 03 END
============================================================

