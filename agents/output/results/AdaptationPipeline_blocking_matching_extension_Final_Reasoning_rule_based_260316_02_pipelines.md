# Pipeline Snapshots

notebook_name=AdaptationPipeline_blocking_matching_extension_Final_Reasoning
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=62.14%
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

DATA_DIR = ""

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

amazon_small = load_parquet(
    DATA_DIR + "output/schema-matching/amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    DATA_DIR + "output/schema-matching/goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    DATA_DIR + "output/schema-matching/metabooks_small.parquet",
    name="metabooks_small",
)

datasets = [amazon_small, goodreads_small, metabooks_small]

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

schema_matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = schema_matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# YOU MUST!!! USE THE PRECOMPUTED BLOCKERS PROVIEDED TO YOU UNDER "5. **BLOCKING CONFIGURATION**"!
# --------------------------------

print("Performing Blocking")

blocker_g2a = EmbeddingBlocker(
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

blocker_m2a = EmbeddingBlocker(
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

blocker_m2g = EmbeddingBlocker(
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
# CRITICAL INSTRUCTION FOR AGENTS:
# You MUST use the matching configuration supplied to you under "6. **MATCHING CONFIGURATION**" to set the correct comparators in the following.
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()

comparators_g2a = [
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

comparators_m2a = [
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

comparators_m2g = [
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

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_g2a = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_g2a,
    comparators=comparators_g2a,
    weights=[0.42, 0.33, 0.15, 0.10],
    threshold=0.66,
    id_column="id",
)

rb_correspondences_m2a = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_m2a,
    comparators=comparators_m2a,
    weights=[0.50, 0.25, 0.15, 0.10],
    threshold=0.72,
    id_column="id",
)

rb_correspondences_m2g = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_m2g,
    comparators=comparators_m2g,
    weights=[0.42, 0.23, 0.12, 0.15, 0.08],
    threshold=0.68,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_g2a.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_m2a.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_m2g.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_goodreads_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_g2a, rb_correspondences_m2a, rb_correspondences_m2g],
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
node_index=12
node_name=execute_pipeline
accuracy_score=65.71%
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
import html

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

amazon_small = load_parquet(
    DATA_DIR + "output/schema-matching/amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    DATA_DIR + "output/schema-matching/goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    DATA_DIR + "output/schema-matching/metabooks_small.parquet",
    name="metabooks_small",
)

datasets = [amazon_small, goodreads_small, metabooks_small]

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

schema_matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = schema_matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Normalization / preprocessing before blocking, matching, and fusion
# --------------------------------

def is_missing(x):
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x).strip()
    return s == "" or s.lower() == "nan" or s.lower() == "none"

def clean_text(x):
    if is_missing(x):
        return None
    s = html.unescape(str(x))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def lower_strip(x):
    if is_missing(x):
        return ""
    return clean_text(x).lower().strip()

def normalize_title(x):
    if is_missing(x):
        return None
    s = clean_text(x)
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip(" -:")
    return s if s else None

def normalize_author(x):
    if is_missing(x):
        return None
    s = clean_text(x)
    s = re.sub(r"\([^)]*(author|editor|illustrator|foreword|introduction|translator|adaptation)[^)]*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" ,;")
    if " and " in s.lower():
        parts = re.split(r"\band\b", s, flags=re.IGNORECASE)
        s = parts[0].strip(" ,;")
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) > 0:
            s = parts[0]
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

def normalize_publisher(x):
    if is_missing(x):
        return None
    s = clean_text(x)
    s = re.sub(r"\s*\([^)]*\)", "", s).strip()
    replacements = {
        "Vintage Books USA": "Vintage",
        "Vintage Books": "Vintage",
        "Simon & Schuster Trade Division": "Simon & Schuster",
        "Simon & Schuster": "Simon & Schuster",
    }
    for old, new in replacements.items():
        if s.lower() == old.lower():
            s = new
    s = re.sub(r"\s+", " ", s).strip(" ,;")
    return s if s else None

def normalize_language(x):
    if is_missing(x):
        return None
    s = clean_text(x)
    return s.title()

def normalize_genres_to_string(x):
    if is_missing(x):
        return None
    values = x if isinstance(x, list) else [x]
    tokens = []
    seen = set()
    for value in values:
        if is_missing(value):
            continue
        parts = str(value).split(",")
        for part in parts:
            token = clean_text(part)
            if not token:
                continue
            key = token.lower()
            if key not in seen:
                seen.add(key)
                tokens.append(token)
    return ", ".join(tokens) if tokens else None

def prepare_dataframe(df):
    df = df.copy()

    if "title" in df.columns:
        df["title"] = df["title"].apply(normalize_title)
    if "author" in df.columns:
        df["author"] = df["author"].apply(normalize_author)
    if "publisher" in df.columns:
        df["publisher"] = df["publisher"].apply(normalize_publisher)
    if "language" in df.columns:
        df["language"] = df["language"].apply(normalize_language)
    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(normalize_genres_to_string)

    return df

amazon_small = prepare_dataframe(amazon_small)
goodreads_small = prepare_dataframe(goodreads_small)
metabooks_small = prepare_dataframe(metabooks_small)

# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# YOU MUST!!! USE THE PRECOMPUTED BLOCKERS PROVIEDED TO YOU UNDER "5. **BLOCKING CONFIGURATION**"!
# --------------------------------

print("Performing Blocking")

blocker_g2a = EmbeddingBlocker(
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

blocker_m2a = EmbeddingBlocker(
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

blocker_m2g = EmbeddingBlocker(
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
# CRITICAL INSTRUCTION FOR AGENTS:
# You MUST use the matching configuration supplied to you under "6. **MATCHING CONFIGURATION**" to set the correct comparators in the following.
# --------------------------------

comparators_g2a = [
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

comparators_m2a = [
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

comparators_m2g = [
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

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_g2a = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_g2a,
    comparators=comparators_g2a,
    weights=[0.42, 0.33, 0.15, 0.10],
    threshold=0.66,
    id_column="id",
)

rb_correspondences_m2a = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_m2a,
    comparators=comparators_m2a,
    weights=[0.50, 0.25, 0.15, 0.10],
    threshold=0.72,
    id_column="id",
)

rb_correspondences_m2g = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_m2g,
    comparators=comparators_m2g,
    weights=[0.42, 0.23, 0.12, 0.15, 0.08],
    threshold=0.68,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_g2a.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_m2a.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_m2g.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_goodreads_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_g2a, rb_correspondences_m2a, rb_correspondences_m2g],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("title", voting)
strategy.add_attribute_fuser("author", voting)
strategy.add_attribute_fuser("publish_year", voting)
strategy.add_attribute_fuser("publisher", voting)
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("numratings", maximum)
strategy.add_attribute_fuser("language", voting)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", voting)
strategy.add_attribute_fuser("edition", shortest_string)
strategy.add_attribute_fuser("page_count", voting)
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
    include_singletons=True,
)

if "genres" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["genres"] = rb_fused_standard_blocker["genres"].apply(normalize_genres_to_string)
if "title" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["title"] = rb_fused_standard_blocker["title"].apply(normalize_title)
if "author" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["author"] = rb_fused_standard_blocker["author"].apply(normalize_author)
if "publisher" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["publisher"] = rb_fused_standard_blocker["publisher"].apply(normalize_publisher)
if "language" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["language"] = rb_fused_standard_blocker["language"].apply(normalize_language)

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
accuracy_score=59.29%
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
import html

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

amazon_small = load_parquet(
    DATA_DIR + "output/schema-matching/amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    DATA_DIR + "output/schema-matching/goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    DATA_DIR + "output/schema-matching/metabooks_small.parquet",
    name="metabooks_small",
)

datasets = [amazon_small, goodreads_small, metabooks_small]

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

schema_matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = schema_matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Normalization / preprocessing before blocking, matching, and fusion
# --------------------------------

def is_missing(x):
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}

def clean_text(x):
    if is_missing(x):
        return None
    s = html.unescape(str(x))
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

def lower_strip(x):
    if is_missing(x):
        return ""
    return clean_text(x).lower().strip()

def normalize_title_for_match(x):
    if is_missing(x):
        return None
    s = clean_text(x)
    s = re.sub(r"\s+", " ", s).strip(" -:")
    return s if s else None

def normalize_title_core(x):
    if is_missing(x):
        return None
    s = clean_text(x)
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip(" -:")
    return s if s else None

def normalize_author_for_match(x):
    if is_missing(x):
        return None
    s = clean_text(x)
    s = re.sub(
        r"\((?:[^)]*)(author|editor|illustrator|foreword|introduction|translator|adaptation)(?:[^)]*)\)",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\s+", " ", s).strip(" ,;")
    return s if s else None

def normalize_author_for_output(x):
    if is_missing(x):
        return None
    s = normalize_author_for_match(x)
    return s

def normalize_publisher(x):
    if is_missing(x):
        return None
    s = clean_text(x)
    s = re.sub(r"\s*\([^)]*\)", "", s).strip()
    s = re.sub(r"\bincorporated\b", "inc", s, flags=re.IGNORECASE)
    s = re.sub(r"\bcompany\b", "co", s, flags=re.IGNORECASE)
    s = re.sub(r"\blimited\b", "ltd", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" ,;")

    exact_map = {
        "vintage books": "Vintage",
        "vintage books usa": "Vintage",
        "vintage": "Vintage",
        "simon & schuster trade division": "Simon & Schuster",
        "simon and schuster": "Simon & Schuster",
        "simon & schuster": "Simon & Schuster",
        "ace books": "Ace Books",
        "ace": "Ace Books",
        "algonquin books": "Algonquin Books",
        "penguin books": "Penguin Books",
        "penguin classics": "Penguin Classics",
        "henry holt and co.": "Henry Holt and Co.",
        "henry holt and co": "Henry Holt and Co.",
    }
    key = s.lower()
    return exact_map.get(key, s)

def normalize_language(x):
    if is_missing(x):
        return None
    s = clean_text(x)
    return s.title() if s else None

def genres_to_list(x):
    if is_missing(x):
        return []
    raw_values = x if isinstance(x, list) else [x]
    tokens = []
    seen = set()
    for value in raw_values:
        if is_missing(value):
            continue
        parts = re.split(r"\s*,\s*|\s*\|\s*|\s*;\s*", str(value))
        for part in parts:
            token = clean_text(part)
            if not token:
                continue
            token = re.sub(r"\s+", " ", token).strip()
            key = token.lower()
            if key and key not in seen:
                seen.add(key)
                tokens.append(token)
    return tokens

def normalize_genres_to_string(x):
    tokens = genres_to_list(x)
    return ", ".join(tokens) if tokens else None

def safe_int(x):
    if is_missing(x):
        return None
    try:
        return int(round(float(x)))
    except Exception:
        return None

def prepare_dataframe(df, source_name):
    df = df.copy()

    df["_source"] = source_name

    if "title" in df.columns:
        df["title"] = df["title"].apply(normalize_title_for_match)
        df["title_core"] = df["title"].apply(normalize_title_core)
    if "author" in df.columns:
        df["author"] = df["author"].apply(normalize_author_for_match)
    if "publisher" in df.columns:
        df["publisher"] = df["publisher"].apply(normalize_publisher)
    if "language" in df.columns:
        df["language"] = df["language"].apply(normalize_language)
    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(genres_to_list)
    if "publish_year" in df.columns:
        df["publish_year"] = df["publish_year"].apply(safe_int)
    if "page_count" in df.columns:
        df["page_count"] = df["page_count"].apply(safe_int)

    return df

amazon_small = prepare_dataframe(amazon_small, "amazon_small")
goodreads_small = prepare_dataframe(goodreads_small, "goodreads_small")
metabooks_small = prepare_dataframe(metabooks_small, "metabooks_small")

# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# YOU MUST!!! USE THE PRECOMPUTED BLOCKERS PROVIEDED TO YOU UNDER "5. **BLOCKING CONFIGURATION**"!
# --------------------------------

print("Performing Blocking")

blocker_g2a = EmbeddingBlocker(
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

blocker_m2a = EmbeddingBlocker(
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

blocker_m2g = EmbeddingBlocker(
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
# CRITICAL INSTRUCTION FOR AGENTS:
# You MUST use the matching configuration supplied to you under "6. **MATCHING CONFIGURATION**" to set the correct comparators in the following.
# --------------------------------

comparators_g2a = [
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

comparators_m2a = [
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

comparators_m2g = [
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

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_g2a = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_g2a,
    comparators=comparators_g2a,
    weights=[0.42, 0.33, 0.15, 0.10],
    threshold=0.66,
    id_column="id",
)

rb_correspondences_m2a = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_m2a,
    comparators=comparators_m2a,
    weights=[0.50, 0.25, 0.15, 0.10],
    threshold=0.72,
    id_column="id",
)

rb_correspondences_m2g = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_m2g,
    comparators=comparators_m2g,
    weights=[0.42, 0.23, 0.12, 0.15, 0.08],
    threshold=0.68,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_g2a.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_m2a.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_m2g.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_goodreads_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_g2a, rb_correspondences_m2a, rb_correspondences_m2g],
    ignore_index=True,
)

source_priority = {
    "amazon_small": 1,
    "goodreads_small": 3,
    "metabooks_small": 2,
}

def flatten_values(values):
    if values is None:
        return []
    if isinstance(values, pd.Series):
        values = values.tolist()
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, list):
        values = [values]
    flat = []
    for v in values:
        if isinstance(v, (list, tuple, set)):
            flat.extend(list(v))
        else:
            flat.append(v)
    return flat

def choose_best_string(values, prefer_longest=True):
    candidates = []
    for v in flatten_values(values):
        if is_missing(v):
            continue
        s = clean_text(v)
        if s:
            candidates.append(s)
    if not candidates:
        return None
    candidates = list(dict.fromkeys(candidates))
    if prefer_longest:
        candidates = sorted(candidates, key=lambda s: (len(s), len(s.split())), reverse=True)
    else:
        candidates = sorted(candidates, key=lambda s: (len(s), len(s.split())))
    return candidates[0]

def choose_best_language(values):
    candidates = []
    for v in flatten_values(values):
        if is_missing(v):
            continue
        s = normalize_language(v)
        if s:
            candidates.append(s)
    if not candidates:
        return None
    counts = pd.Series(candidates).value_counts()
    return counts.index[0]

def choose_consensus_numeric(values):
    nums = []
    for v in flatten_values(values):
        if is_missing(v):
            continue
        try:
            nums.append(float(v))
        except Exception:
            continue
    if not nums:
        return None
    if len(nums) == 1:
        return int(round(nums[0]))
    rounded = [int(round(x)) for x in nums]
    counts = pd.Series(rounded).value_counts()
    top_value = counts.index[0]
    if counts.iloc[0] >= 2:
        return int(top_value)
    return int(round(float(np.median(rounded))))

def choose_max_numeric(values):
    nums = []
    for v in flatten_values(values):
        if is_missing(v):
            continue
        try:
            nums.append(float(v))
        except Exception:
            continue
    if not nums:
        return None
    return max(nums)

def choose_best_genres(values):
    all_lists = []
    for v in flatten_values(values):
        tokens = genres_to_list(v)
        if tokens:
            all_lists.append(tokens)
    if not all_lists:
        return None

    counts = {}
    first_seen = {}
    for lst in all_lists:
        for idx, token in enumerate(lst):
            key = token.lower()
            counts[key] = counts.get(key, 0) + 1
            if key not in first_seen:
                first_seen[key] = token

    min_support = 2 if len(all_lists) >= 2 else 1
    selected = [
        first_seen[key]
        for key, cnt in counts.items()
        if cnt >= min_support
    ]

    if not selected:
        best_list = sorted(all_lists, key=lambda x: len(x), reverse=True)[0]
        selected = best_list

    return ", ".join(selected) if selected else None

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publish_year", choose_consensus_numeric)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("rating", choose_max_numeric)
strategy.add_attribute_fuser("numratings", choose_max_numeric)
strategy.add_attribute_fuser("language", choose_best_language)
strategy.add_attribute_fuser("genres", choose_best_genres)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", choose_consensus_numeric)
strategy.add_attribute_fuser("price", choose_max_numeric)

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
    include_singletons=True,
)

if "title" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["title"] = rb_fused_standard_blocker["title"].apply(normalize_title_for_match)
if "author" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["author"] = rb_fused_standard_blocker["author"].apply(normalize_author_for_output)
if "publisher" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["publisher"] = rb_fused_standard_blocker["publisher"].apply(normalize_publisher)
if "language" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["language"] = rb_fused_standard_blocker["language"].apply(normalize_language)
if "genres" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["genres"] = rb_fused_standard_blocker["genres"].apply(normalize_genres_to_string)
if "publish_year" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["publish_year"] = rb_fused_standard_blocker["publish_year"].apply(safe_int)
if "page_count" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["page_count"] = rb_fused_standard_blocker["page_count"].apply(safe_int)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

