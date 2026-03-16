# Pipeline Snapshots

notebook_name=AdaptationPipeline_blocking_matching_extension_Final_Blocker_Matcher
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=69.29%
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
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

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

blocker_goodreads_amazon = EmbeddingBlocker(
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

blocker_metabooks_amazon = EmbeddingBlocker(
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

blocker_metabooks_goodreads = EmbeddingBlocker(
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

lower_strip = lambda x: str(x).lower().strip()

comparators_goodreads_amazon = [
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

comparators_metabooks_amazon = [
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

comparators_metabooks_goodreads = [
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

rb_correspondences_goodreads_amazon = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_goodreads_amazon,
    comparators=comparators_goodreads_amazon,
    weights=[0.42, 0.33, 0.15, 0.1],
    threshold=0.66,
    id_column="id",
)

rb_correspondences_metabooks_amazon = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_metabooks_amazon,
    comparators=comparators_metabooks_amazon,
    weights=[0.5, 0.25, 0.15, 0.1],
    threshold=0.72,
    id_column="id",
)

rb_correspondences_metabooks_goodreads = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_metabooks_goodreads,
    comparators=comparators_metabooks_goodreads,
    weights=[0.42, 0.23, 0.12, 0.15, 0.08],
    threshold=0.68,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_goodreads_amazon.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_amazon.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_goodreads.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_goodreads_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_goodreads_amazon,
        rb_correspondences_metabooks_amazon,
        rb_correspondences_metabooks_goodreads,
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
node_index=13
node_name=execute_pipeline
accuracy_score=66.43%
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
import html

# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

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
# Normalization helpers
# --------------------------------

def is_missing(x):
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict, np.ndarray)):
        return False
    try:
        return pd.isna(x)
    except Exception:
        return False

def lower_strip(x):
    if is_missing(x):
        return ""
    return str(x).lower().strip()

def clean_text(x):
    if is_missing(x):
        return ""
    s = html.unescape(str(x))
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() == "nan":
        return ""
    return s

def clean_title(x):
    s = clean_text(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_author(x):
    s = clean_text(x)
    s = re.sub(r"\s*\(.*?goodreads author.*?\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_publisher(x):
    s = clean_text(x)
    s = re.sub(r"[^\w\s&'\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_genres(x):
    if is_missing(x):
        return []
    if isinstance(x, np.ndarray):
        vals = x.tolist()
    elif isinstance(x, (list, tuple, set)):
        vals = list(x)
    else:
        s = str(x).strip()
        if not s or s.lower() == "nan":
            return []
        vals = [p.strip() for p in s.split(",")]
    cleaned = []
    for v in vals:
        sv = clean_text(v).lower()
        if sv and sv != "nan":
            cleaned.append(sv)
    seen = set()
    result = []
    for g in cleaned:
        if g not in seen:
            seen.add(g)
            result.append(g)
    return result

def format_genres_output(x):
    vals = split_genres(x)
    if not vals:
        return ""
    return ", ".join(vals)

def pick_best_title(values):
    vals = []
    for v in values:
        s = clean_title(v)
        if s:
            vals.append(s)
    if not vals:
        return ""
    vals = sorted(set(vals), key=lambda s: (-len(s), s))
    return vals[0]

def pick_best_author(values):
    vals = []
    for v in values:
        s = clean_author(v)
        if s:
            vals.append(s)
    if not vals:
        return ""
    vals = sorted(set(vals), key=lambda s: (("(" in s or ")" in s), -len(s), s))
    return vals[0]

def pick_best_publisher(values):
    vals = []
    for v in values:
        s = clean_publisher(v)
        if s:
            vals.append(s)
    if not vals:
        return ""
    vals = sorted(set(vals), key=lambda s: (-len(s), s))
    return vals[0]

def pick_best_language(values):
    vals = []
    for v in values:
        s = clean_text(v)
        if s:
            vals.append(s)
    if not vals:
        return ""
    counts = pd.Series(vals).value_counts()
    return counts.index[0]

# Pre-normalize source datasets
for df in [amazon_small, goodreads_small, metabooks_small]:
    if "title" in df.columns:
        df["title"] = df["title"].apply(clean_title)
    if "author" in df.columns:
        df["author"] = df["author"].apply(clean_author)
    if "publisher" in df.columns:
        df["publisher"] = df["publisher"].apply(clean_publisher)
    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(split_genres)

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

# Re-apply normalization after schema matching
for df in [amazon_small, goodreads_small, metabooks_small]:
    if "title" in df.columns:
        df["title"] = df["title"].apply(clean_title)
    if "author" in df.columns:
        df["author"] = df["author"].apply(clean_author)
    if "publisher" in df.columns:
        df["publisher"] = df["publisher"].apply(clean_publisher)
    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(split_genres)

# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# YOU MUST!!! USE THE PRECOMPUTED BLOCKERS PROVIEDED TO YOU UNDER "5. **BLOCKING CONFIGURATION**"!
# --------------------------------

print("Performing Blocking")

blocker_goodreads_amazon = EmbeddingBlocker(
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

blocker_metabooks_amazon = EmbeddingBlocker(
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

blocker_metabooks_goodreads = EmbeddingBlocker(
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

comparators_goodreads_amazon = [
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

comparators_metabooks_amazon = [
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

comparators_metabooks_goodreads = [
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

rb_correspondences_goodreads_amazon = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_goodreads_amazon,
    comparators=comparators_goodreads_amazon,
    weights=[0.42, 0.33, 0.15, 0.1],
    threshold=0.66,
    id_column="id",
)

rb_correspondences_metabooks_amazon = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_metabooks_amazon,
    comparators=comparators_metabooks_amazon,
    weights=[0.5, 0.25, 0.15, 0.1],
    threshold=0.72,
    id_column="id",
)

rb_correspondences_metabooks_goodreads = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_metabooks_goodreads,
    comparators=comparators_metabooks_goodreads,
    weights=[0.42, 0.23, 0.12, 0.15, 0.08],
    threshold=0.68,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_goodreads_amazon.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_amazon.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_goodreads.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_goodreads_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_goodreads_amazon,
        rb_correspondences_metabooks_amazon,
        rb_correspondences_metabooks_goodreads,
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
strategy.add_attribute_fuser("language", voting)
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

# Post-fusion cleanup to improve attribute-level accuracy
if "title" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["title"] = rb_fused_standard_blocker["title"].apply(clean_title)

if "author" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["author"] = rb_fused_standard_blocker["author"].apply(clean_author)

if "publisher" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["publisher"] = rb_fused_standard_blocker["publisher"].apply(clean_publisher)

if "language" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["language"] = rb_fused_standard_blocker["language"].apply(
        lambda x: clean_text(x)
    )

if "genres" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["genres"] = rb_fused_standard_blocker["genres"].apply(format_genres_output)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=20
node_name=execute_pipeline
accuracy_score=60.00%
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
import html

# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

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
# Normalization helpers
# --------------------------------

def is_missing(x):
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict, np.ndarray)):
        return False
    try:
        return pd.isna(x)
    except Exception:
        return False

def ensure_iterable_values(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return list(x)
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]

def clean_text(x):
    if is_missing(x):
        return ""
    s = html.unescape(str(x))
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return s

def lower_strip(x):
    return clean_text(x).lower().strip()

def clean_title(x):
    s = clean_text(x)
    s = re.sub(r"\s*:\s*", ": ", s)
    s = re.sub(r"\s*;\s*", "; ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_title_for_compare(x):
    s = clean_title(x).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_author(x):
    s = clean_text(x)
    s = re.sub(r"\s*\(.*?goodreads author.*?\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_author_for_compare(x):
    s = clean_author(x).lower()
    s = re.sub(r"[^\w\s'\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_publisher(x):
    s = clean_text(x)
    s = re.sub(r"[^\w\s&'\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_publisher_for_compare(x):
    s = clean_publisher(x).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\b(co|corp|corporation|inc|ltd|llc)\b\.?", " ", s)
    s = re.sub(r"\b(publisher|publishers|publishing)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_genres(x):
    if is_missing(x):
        return []
    if isinstance(x, np.ndarray):
        vals = x.tolist()
    elif isinstance(x, (list, tuple, set)):
        vals = list(x)
    else:
        s = str(x).strip()
        if not s or s.lower() == "nan":
            return []
        vals = [p.strip() for p in s.split(",")]
    cleaned = []
    for v in vals:
        sv = clean_text(v).lower()
        if sv and sv != "nan":
            cleaned.append(sv)
    seen = set()
    result = []
    for g in cleaned:
        if g not in seen:
            seen.add(g)
            result.append(g)
    return result

def canonicalize_genre_token(g):
    s = clean_text(g).lower()
    mappings = {
        "novels": "fiction",
        "adult fiction": "fiction",
        "contemporary": "fiction",
        "historical": "historical fiction",
        "memoirs": "memoir",
        "biographies": "biography",
        "professionals": "",
        "academics": "",
        "book club": "",
        "adult": "",
    }
    return mappings.get(s, s)

def normalize_genres(x):
    vals = split_genres(x)
    cleaned = []
    for v in vals:
        c = canonicalize_genre_token(v)
        if c:
            cleaned.append(c)
    seen = set()
    result = []
    for g in cleaned:
        if g not in seen:
            seen.add(g)
            result.append(g)
    return result

def safe_int(x):
    if is_missing(x):
        return pd.NA
    try:
        return int(float(x))
    except Exception:
        return pd.NA

def choose_best_title(values):
    vals = []
    for v in ensure_iterable_values(values):
        s = clean_title(v)
        if s:
            vals.append(s)
    if not vals:
        return ""
    grouped = {}
    for s in vals:
        key = normalize_title_for_compare(s)
        grouped.setdefault(key, []).append(s)
    best_key = sorted(
        grouped.keys(),
        key=lambda k: (-len(grouped[k]), -max(len(v) for v in grouped[k]), k)
    )[0]
    candidates = grouped[best_key]
    return sorted(candidates, key=lambda s: (-len(s), s))[0]

def choose_best_author(values):
    vals = []
    for v in ensure_iterable_values(values):
        s = clean_author(v)
        if s:
            vals.append(s)
    if not vals:
        return ""
    grouped = {}
    for s in vals:
        key = normalize_author_for_compare(s)
        grouped.setdefault(key, []).append(s)
    best_key = sorted(
        grouped.keys(),
        key=lambda k: (-len(grouped[k]), -max(len(v) for v in grouped[k]), k)
    )[0]
    candidates = grouped[best_key]
    return sorted(candidates, key=lambda s: (("(" in s or ")" in s), -len(s), s))[0]

def choose_best_publisher(values):
    vals = []
    for v in ensure_iterable_values(values):
        s = clean_publisher(v)
        if s:
            vals.append(s)
    if not vals:
        return ""
    grouped = {}
    for s in vals:
        key = normalize_publisher_for_compare(s)
        grouped.setdefault(key, []).append(s)
    best_key = sorted(
        grouped.keys(),
        key=lambda k: (-len(grouped[k]), -max(len(v) for v in grouped[k]), k)
    )[0]
    candidates = grouped[best_key]
    return sorted(candidates, key=lambda s: (-len(s), s))[0]

def choose_best_language(values):
    vals = []
    for v in ensure_iterable_values(values):
        s = clean_text(v)
        if s:
            vals.append(s)
    if not vals:
        return ""
    counts = pd.Series(vals).value_counts()
    return counts.index[0]

def choose_best_publish_year(values):
    vals = []
    for v in ensure_iterable_values(values):
        iv = safe_int(v)
        if not pd.isna(iv):
            vals.append(iv)
    if not vals:
        return pd.NA
    counts = pd.Series(vals).value_counts()
    return counts.index[0]

def choose_best_page_count(values):
    vals = []
    for v in ensure_iterable_values(values):
        iv = safe_int(v)
        if not pd.isna(iv) and iv > 0:
            vals.append(iv)
    if not vals:
        return pd.NA
    counts = pd.Series(vals).value_counts()
    return counts.index[0]

def choose_best_genres(values):
    all_lists = []
    for v in ensure_iterable_values(values):
        current = normalize_genres(v)
        if current:
            all_lists.append(current)
    if not all_lists:
        return ""
    counts = {}
    for lst in all_lists:
        for g in lst:
            counts[g] = counts.get(g, 0) + 1
    selected = [g for g, c in counts.items() if c >= 2]
    if not selected:
        max_count = max(counts.values())
        selected = [g for g, c in counts.items() if c == max_count]
    selected = sorted(selected)
    return ", ".join(selected)

def preprocess_books(df):
    if "title" in df.columns:
        df["title"] = df["title"].apply(clean_title)
    if "author" in df.columns:
        df["author"] = df["author"].apply(clean_author)
    if "publisher" in df.columns:
        df["publisher"] = df["publisher"].apply(clean_publisher)
    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(normalize_genres)
    if "publish_year" in df.columns:
        df["publish_year"] = df["publish_year"].apply(safe_int)
    if "page_count" in df.columns:
        df["page_count"] = df["page_count"].apply(safe_int)
    if "language" in df.columns:
        df["language"] = df["language"].apply(clean_text)

# Pre-normalize source datasets
for df in [amazon_small, goodreads_small, metabooks_small]:
    preprocess_books(df)

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

for df in [amazon_small, goodreads_small, metabooks_small]:
    preprocess_books(df)

# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# YOU MUST!!! USE THE PRECOMPUTED BLOCKERS PROVIEDED TO YOU UNDER "5. **BLOCKING CONFIGURATION**"!
# --------------------------------

print("Performing Blocking")

blocker_goodreads_amazon = EmbeddingBlocker(
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

blocker_metabooks_amazon = EmbeddingBlocker(
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

blocker_metabooks_goodreads = EmbeddingBlocker(
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

comparators_goodreads_amazon = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lambda x: str(x).lower().strip(),
    ),
]

comparators_metabooks_amazon = [
    StringComparator(
        column="title",
        similarity_function="cosine",
        preprocess=lambda x: str(x).lower().strip(),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
]

comparators_metabooks_goodreads = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    StringComparator(
        column="genres",
        similarity_function="jaccard",
        preprocess=lambda x: str(x).lower().strip(),
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1.0,
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_goodreads_amazon = matcher.match(
    df_left=goodreads_small,
    df_right=amazon_small,
    candidates=blocker_goodreads_amazon,
    comparators=comparators_goodreads_amazon,
    weights=[0.42, 0.33, 0.15, 0.1],
    threshold=0.66,
    id_column="id",
)

rb_correspondences_metabooks_amazon = matcher.match(
    df_left=metabooks_small,
    df_right=amazon_small,
    candidates=blocker_metabooks_amazon,
    comparators=comparators_metabooks_amazon,
    weights=[0.5, 0.25, 0.15, 0.1],
    threshold=0.72,
    id_column="id",
)

rb_correspondences_metabooks_goodreads = matcher.match(
    df_left=metabooks_small,
    df_right=goodreads_small,
    candidates=blocker_metabooks_goodreads,
    comparators=comparators_metabooks_goodreads,
    weights=[0.42, 0.23, 0.12, 0.15, 0.08],
    threshold=0.68,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_goodreads_amazon.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_amazon.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_amazon_small.csv",
    ),
    index=False,
)

rb_correspondences_metabooks_goodreads.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_goodreads_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_goodreads_amazon,
        rb_correspondences_metabooks_amazon,
        rb_correspondences_metabooks_goodreads,
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

if "title" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["title"] = rb_fused_standard_blocker["title"].apply(choose_best_title)

if "author" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["author"] = rb_fused_standard_blocker["author"].apply(choose_best_author)

if "publisher" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["publisher"] = rb_fused_standard_blocker["publisher"].apply(choose_best_publisher)

if "language" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["language"] = rb_fused_standard_blocker["language"].apply(choose_best_language)

if "genres" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["genres"] = rb_fused_standard_blocker["genres"].apply(choose_best_genres)

if "publish_year" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["publish_year"] = rb_fused_standard_blocker["publish_year"].apply(choose_best_publish_year)

if "page_count" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["page_count"] = rb_fused_standard_blocker["page_count"].apply(choose_best_page_count)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

