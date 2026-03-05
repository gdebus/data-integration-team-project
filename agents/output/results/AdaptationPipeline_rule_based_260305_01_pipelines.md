# Pipeline Snapshots

notebook_name=AdaptationPipeline
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=3
node_name=execute_pipeline
accuracy_score=46.43%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_parquet
from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union, maximum
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "input/datasets/books/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
good_dataset_name_1 = load_parquet(
    DATA_DIR + "amazon_small.parquet",
    name="amazon_small",
)

good_dataset_name_2 = load_parquet(
    DATA_DIR + "goodreads_small.parquet",
    name="goodreads_small",
)

good_dataset_name_3 = load_parquet(
    DATA_DIR + "metabooks_small.parquet",
    name="metabooks_small",
)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# The resulting columns for all datasets will have the schema of dataset1 (amazon_small).
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

# Match schema of amazon_small with goodreads_small and rename schema of goodreads_small
schema_correspondences = schema_matcher.match(good_dataset_name_1, good_dataset_name_2)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
good_dataset_name_2 = good_dataset_name_2.rename(columns=rename_map)

# Match schema of amazon_small with metabooks_small and rename schema of metabooks_small
schema_correspondences = schema_matcher.match(good_dataset_name_1, good_dataset_name_3)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
good_dataset_name_3 = good_dataset_name_3.rename(columns=rename_map)

# --------------------------------
# Blocking
# Policy: Use informative identity signals -> title + author (+ year as disambiguator).
# Do NOT block on ratings/counts or internal ids.
# --------------------------------

print("Performing Blocking")

blocker_a2g = StandardBlocker(
    good_dataset_name_1, good_dataset_name_2,
    on=["title", "author"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id"
)

blocker_a2m = StandardBlocker(
    good_dataset_name_1, good_dataset_name_3,
    on=["title", "author"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id"
)

blocker_g2m = StandardBlocker(
    good_dataset_name_2, good_dataset_name_3,
    on=["title", "author"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id"
)

# --------------------------------
# Matching Configuration (comparators)
# Use title/author strongly; publish_year as a soft numeric disambiguator.
# --------------------------------

comparators_a2g = [
    StringComparator(
        column="title",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="author",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1,
    ),
]

comparators_a2m = [
    StringComparator(
        column="title",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="author",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1,
    ),
]

comparators_g2m = [
    StringComparator(
        column="title",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="author",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1,
    ),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_a2g = rb_matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2,
    candidates=blocker_a2g,
    comparators=comparators_a2g,
    weights=[0.55, 0.35, 0.10],
    threshold=0.75,
    id_column="id"
)

rb_correspondences_a2m = rb_matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3,
    candidates=blocker_a2m,
    comparators=comparators_a2m,
    weights=[0.55, 0.35, 0.10],
    threshold=0.75,
    id_column="id"
)

rb_correspondences_g2m = rb_matcher.match(
    df_left=good_dataset_name_2,
    df_right=good_dataset_name_3,
    candidates=blocker_g2m,
    comparators=comparators_g2m,
    weights=[0.55, 0.35, 0.10],
    threshold=0.75,
    id_column="id"
)

print("Fusing Data")

# Merge rule based correspondences
all_rb_correspondences = pd.concat(
    [rb_correspondences_a2g, rb_correspondences_a2m, rb_correspondences_g2m],
    ignore_index=True
)

# --------------------------------
# Data Fusion
# Keep amazon schema (dataset1). Only fuse columns that exist in amazon after schema matching.
# --------------------------------

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("publish_year", maximum)

# If schema matching mapped any extra attributes onto amazon columns (rare), fuse safely when present.
# (Union is safe for list-like fields; it will simply not be used if the column doesn't exist.)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", maximum)
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("numratings", maximum)
strategy.add_attribute_fuser("price", maximum)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl"
)

rb_fused_standard_blocker = engine.run(
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
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
accuracy_score=56.43%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_parquet
from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union, maximum
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
import re
import html
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/books/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_parquet(
    DATA_DIR + "amazon_small.parquet",
    name="amazon_small",
)

good_dataset_name_2 = load_parquet(
    DATA_DIR + "goodreads_small.parquet",
    name="goodreads_small",
)

good_dataset_name_3 = load_parquet(
    DATA_DIR + "metabooks_small.parquet",
    name="metabooks_small",
)

# --------------------------------
# Lightweight, deterministic standardization to improve matching + fusion accuracy
# - Fix HTML entities in Amazon titles
# - Normalize title/author into *_norm columns used for matching
# - Clean Goodreads author suffix "(Goodreads Author)"
# - Normalize genres into lists for union() to work well
# - Normalize publisher (trim/space collapse)
# - Coerce publish_year/page_count to numeric (where available)
# --------------------------------

def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def normalize_title(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x)
    s = html.unescape(s)
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[\u2010-\u2015]", "-", s)  # hyphen variants
    s = re.sub(r"[^\w\s-]", " ", s)         # drop punctuation
    s = _collapse_ws(s)
    return s

def normalize_author(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x)
    s = html.unescape(s)
    s = re.sub(r"\(goodreads author\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\(.*?\)", "", s)  # drop remaining parenthetical qualifiers
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = _collapse_ws(s)
    return s

def normalize_publisher(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x)
    s = html.unescape(s)
    s = _collapse_ws(s)
    return s

def parse_genres(x):
    # Expect strings like "Fiction, Historical Fiction, Romance"
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [g.strip() for g in x if isinstance(g, str) and g.strip()]
    s = str(x).strip()
    if s.lower() == "nan" or s == "":
        return []
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]
    # de-duplicate while preserving order, normalize casing lightly
    seen = set()
    out = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out

def coerce_int(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64")

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    # Ensure expected text columns exist before applying
    if "title" in df.columns:
        df["title"] = df["title"].astype("string")
        df["title_norm"] = df["title"].map(normalize_title).astype("string")
    if "author" in df.columns:
        df["author"] = df["author"].astype("string")
        df["author_norm"] = df["author"].map(normalize_author).astype("string")
    if "publisher" in df.columns:
        df["publisher"] = df["publisher"].astype("string")
        df["publisher_norm"] = df["publisher"].map(normalize_publisher).astype("string")

    if "genres" in df.columns:
        df["genres"] = df["genres"].map(parse_genres)

    if "publish_year" in df.columns:
        df["publish_year"] = coerce_int(df["publish_year"])

    if "page_count" in df.columns:
        df["page_count"] = coerce_int(df["page_count"])

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# Keep amazon_small schema (dataset1) as target.
# Important: do NOT remap away our *_norm columns in amazon; they help matching.
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

# Re-apply key normalizations after schema rename (in case columns were newly aligned to amazon schema)
for df in [good_dataset_name_2, good_dataset_name_3]:
    if "title" in df.columns and "title_norm" not in df.columns:
        df["title"] = df["title"].astype("string")
        df["title_norm"] = df["title"].map(normalize_title).astype("string")
    if "author" in df.columns and "author_norm" not in df.columns:
        df["author"] = df["author"].astype("string")
        df["author_norm"] = df["author"].map(normalize_author).astype("string")
    if "publisher" in df.columns and "publisher_norm" not in df.columns:
        df["publisher"] = df["publisher"].astype("string")
        df["publisher_norm"] = df["publisher"].map(normalize_publisher).astype("string")
    if "genres" in df.columns:
        df["genres"] = df["genres"].map(parse_genres)
    if "publish_year" in df.columns:
        df["publish_year"] = coerce_int(df["publish_year"])
    if "page_count" in df.columns:
        df["page_count"] = coerce_int(df["page_count"])

# --------------------------------
# Blocking
# Use strong identity signals: title_norm + author_norm (robust to punctuation/casing/HTML).
# Avoid popularity/rating/count fields.
# --------------------------------

print("Performing Blocking")

blocker_a2g = StandardBlocker(
    good_dataset_name_1, good_dataset_name_2,
    on=["title_norm", "author_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_a2m = StandardBlocker(
    good_dataset_name_1, good_dataset_name_3,
    on=["title_norm", "author_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_g2m = StandardBlocker(
    good_dataset_name_2, good_dataset_name_3,
    on=["title_norm", "author_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching Configuration (comparators)
# - Title_norm + author_norm drive identity
# - publish_year + page_count act as disambiguators (soft)
# --------------------------------

comparators_a2g = [
    StringComparator(
        column="title_norm",
        similarity_function="jaccard",
        preprocess=None,
    ),
    StringComparator(
        column="author_norm",
        similarity_function="jaccard",
        preprocess=None,
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1,
    ),
    NumericComparator(
        column="page_count",
        max_difference=10,
    ),
]

comparators_a2m = [
    StringComparator(
        column="title_norm",
        similarity_function="jaccard",
        preprocess=None,
    ),
    StringComparator(
        column="author_norm",
        similarity_function="jaccard",
        preprocess=None,
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1,
    ),
    NumericComparator(
        column="page_count",
        max_difference=10,
    ),
]

comparators_g2m = [
    StringComparator(
        column="title_norm",
        similarity_function="jaccard",
        preprocess=None,
    ),
    StringComparator(
        column="author_norm",
        similarity_function="jaccard",
        preprocess=None,
    ),
    NumericComparator(
        column="publish_year",
        max_difference=1,
    ),
    NumericComparator(
        column="page_count",
        max_difference=10,
    ),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_a2g = rb_matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2,
    candidates=blocker_a2g,
    comparators=comparators_a2g,
    weights=[0.55, 0.30, 0.10, 0.05],
    threshold=0.72,
    id_column="id",
)

rb_correspondences_a2m = rb_matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3,
    candidates=blocker_a2m,
    comparators=comparators_a2m,
    weights=[0.55, 0.30, 0.10, 0.05],
    threshold=0.72,
    id_column="id",
)

rb_correspondences_g2m = rb_matcher.match(
    df_left=good_dataset_name_2,
    df_right=good_dataset_name_3,
    candidates=blocker_g2m,
    comparators=comparators_g2m,
    weights=[0.55, 0.30, 0.10, 0.05],
    threshold=0.72,
    id_column="id",
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_a2g, rb_correspondences_a2m, rb_correspondences_g2m],
    ignore_index=True,
)

# --------------------------------
# Data Fusion
# - Genres: union of list values (after parsing) to fix 0% genres accuracy
# - page_count: maximum (often missing in amazon; take available value)
# - publish_year: maximum (edition reprints sometimes later; tends to be more present)
# - publisher: longest_string over normalized/cleaned values helps remove truncation
# - title/author: prefer longest_string to keep fuller names; norms kept for matching
# --------------------------------

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("publish_year", maximum)
strategy.add_attribute_fuser("page_count", maximum)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("numratings", maximum)
strategy.add_attribute_fuser("price", maximum)

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

# write output
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
accuracy_score=56.43%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_parquet
from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union, maximum
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
import re
import html
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/books/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_parquet(
    DATA_DIR + "amazon_small.parquet",
    name="amazon_small",
)

good_dataset_name_2 = load_parquet(
    DATA_DIR + "goodreads_small.parquet",
    name="goodreads_small",
)

good_dataset_name_3 = load_parquet(
    DATA_DIR + "metabooks_small.parquet",
    name="metabooks_small",
)

# --------------------------------
# Deterministic normalization to improve matching + fusion accuracy
# Key fixes vs previous version:
# - Improve blocking recall: block on title_norm only (author varies/noisy)
# - Improve title/author similarity: use token-jaccard-friendly normalization
# - Fix genres 0%: normalize genres to canonical lowercase tokens; union() then works better
# - Keep genres as list for fusion, but also normalize publishers/authors for comparators
# --------------------------------

def _is_null(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(x, pd._libs.missing.NAType))

def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

_STOPWORDS_TITLE = {
    "a", "an", "the",
    "and", "or",
    "of", "to", "in", "on", "for", "with", "from", "by",
    "volume", "vol", "edition", "ed", "revised", "rev",
    "paperback", "hardcover", "kindle", "audiobook",
    "novel", "novels",
}

def normalize_title(x):
    if _is_null(x):
        return np.nan
    s = html.unescape(str(x))
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[\u2010-\u2015]", "-", s)      # hyphen variants
    s = re.sub(r"[:;/|]", " ", s)
    s = re.sub(r"[^\w\s-]", " ", s)            # drop punctuation
    s = s.replace("-", " ")
    s = _collapse_ws(s)
    toks = [t for t in s.split(" ") if t and t not in _STOPWORDS_TITLE]
    if not toks:
        return np.nan
    return " ".join(toks)

def normalize_author(x):
    if _is_null(x):
        return np.nan
    s = html.unescape(str(x))
    s = re.sub(r"\(goodreads author\)", "", s, flags=re.IGNORECASE)
    # remove parenthetical qualifiers but keep initials and apostrophes
    s = re.sub(r"\(.*?\)", "", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s'.-]", " ", s)
    s = s.replace("-", " ")
    s = s.replace(".", " ")
    s = _collapse_ws(s)

    # collapse single-letter initials: "a s byatt" -> "as byatt"
    parts = s.split()
    merged = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and len(parts[i]) == 1 and len(parts[i + 1]) == 1:
            merged.append(parts[i] + parts[i + 1])
            i += 2
        else:
            merged.append(parts[i])
            i += 1
    s = " ".join(merged)
    s = _collapse_ws(s)
    return s if s else np.nan

def normalize_publisher(x):
    if _is_null(x):
        return np.nan
    s = html.unescape(str(x)).lower()
    s = _collapse_ws(s)
    # light canonicalization of common publisher suffixes
    s = re.sub(r"\b(publishing|publishers|press|books|book|inc|ltd|llc|co)\b\.?", "", s)
    s = _collapse_ws(s)
    return s if s else np.nan

def normalize_genre_token(g: str) -> str:
    g = html.unescape(str(g)).strip().lower()
    g = re.sub(r"[^\w\s/&-]", " ", g)
    g = g.replace("&", " and ")
    g = g.replace("-", " ")
    g = _collapse_ws(g)

    # canonicalize frequent variants
    mapping = {
        "sci fi": "science fiction",
        "scifi": "science fiction",
        "ya": "young adult",
        "non fiction": "nonfiction",
        "bio": "biography",
        "biographies": "biography",
        "memoirs": "memoir",
        "graphic novels": "graphic novel",
        "classics": "classic",
        "historical fiction": "historical",
        "adult fiction": "adult",
        "childrens": "children",
    }
    return mapping.get(g, g)

def parse_genres_to_list(x):
    # Ensure list[str] with canonical lowercase tokens
    if _is_null(x):
        return []
    if isinstance(x, list):
        raw = x
    else:
        s = str(x).strip()
        if s.lower() == "nan" or s == "":
            return []
        raw = [p.strip() for p in s.split(",")]

    out = []
    seen = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        tok = normalize_genre_token(item)
        if not tok:
            continue
        key = tok
        if key not in seen:
            seen.add(key)
            out.append(tok)
    return out

def coerce_int(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64")

def ensure_text(df, col):
    if col in df.columns:
        df[col] = df[col].astype("string")

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    ensure_text(df, "title")
    ensure_text(df, "author")
    ensure_text(df, "publisher")

    if "title" in df.columns:
        df["title_norm"] = df["title"].map(normalize_title).astype("string")
    if "author" in df.columns:
        df["author_norm"] = df["author"].map(normalize_author).astype("string")
    if "publisher" in df.columns:
        df["publisher_norm"] = df["publisher"].map(normalize_publisher).astype("string")

    if "genres" in df.columns:
        df["genres"] = df["genres"].map(parse_genres_to_list)

    if "publish_year" in df.columns:
        df["publish_year"] = coerce_int(df["publish_year"])
    if "page_count" in df.columns:
        df["page_count"] = coerce_int(df["page_count"])

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# Keep amazon_small schema (dataset1) as target.
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

# Re-apply key normalizations after schema rename (ensures *_norm and list-genres exist)
for df in [good_dataset_name_2, good_dataset_name_3]:
    ensure_text(df, "title")
    ensure_text(df, "author")
    ensure_text(df, "publisher")

    if "title" in df.columns:
        df["title_norm"] = df["title"].map(normalize_title).astype("string")
    if "author" in df.columns:
        df["author_norm"] = df["author"].map(normalize_author).astype("string")
    if "publisher" in df.columns:
        df["publisher_norm"] = df["publisher"].map(normalize_publisher).astype("string")

    if "genres" in df.columns:
        df["genres"] = df["genres"].map(parse_genres_to_list)

    if "publish_year" in df.columns:
        df["publish_year"] = coerce_int(df["publish_year"])
    if "page_count" in df.columns:
        df["page_count"] = coerce_int(df["page_count"])

# --------------------------------
# Blocking
# Use strong identity signal with high coverage: normalized title.
# (Author is noisy/variable; using it in blocking hurt recall and downstream fusion accuracy.)
# --------------------------------

print("Performing Blocking")

blocker_a2g = StandardBlocker(
    good_dataset_name_1, good_dataset_name_2,
    on=["title_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_a2m = StandardBlocker(
    good_dataset_name_1, good_dataset_name_3,
    on=["title_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_g2m = StandardBlocker(
    good_dataset_name_2, good_dataset_name_3,
    on=["title_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching Configuration (comparators)
# Add publisher_norm as a disambiguator to reduce false positives on same/similar titles.
# Keep publish_year/page_count soft.
# --------------------------------

comparators_a2g = [
    StringComparator(column="title_norm", similarity_function="jaccard", preprocess=None),
    StringComparator(column="author_norm", similarity_function="jaccard", preprocess=None),
    StringComparator(column="publisher_norm", similarity_function="jaccard", preprocess=None),
    NumericComparator(column="publish_year", max_difference=1),
    NumericComparator(column="page_count", max_difference=10),
]

comparators_a2m = [
    StringComparator(column="title_norm", similarity_function="jaccard", preprocess=None),
    StringComparator(column="author_norm", similarity_function="jaccard", preprocess=None),
    StringComparator(column="publisher_norm", similarity_function="jaccard", preprocess=None),
    NumericComparator(column="publish_year", max_difference=1),
    NumericComparator(column="page_count", max_difference=10),
]

comparators_g2m = [
    StringComparator(column="title_norm", similarity_function="jaccard", preprocess=None),
    StringComparator(column="author_norm", similarity_function="jaccard", preprocess=None),
    StringComparator(column="publisher_norm", similarity_function="jaccard", preprocess=None),
    NumericComparator(column="publish_year", max_difference=1),
    NumericComparator(column="page_count", max_difference=10),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_a2g = rb_matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2,
    candidates=blocker_a2g,
    comparators=comparators_a2g,
    weights=[0.55, 0.25, 0.10, 0.07, 0.03],
    threshold=0.74,
    id_column="id",
)

rb_correspondences_a2m = rb_matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3,
    candidates=blocker_a2m,
    comparators=comparators_a2m,
    weights=[0.55, 0.25, 0.10, 0.07, 0.03],
    threshold=0.74,
    id_column="id",
)

rb_correspondences_g2m = rb_matcher.match(
    df_left=good_dataset_name_2,
    df_right=good_dataset_name_3,
    candidates=blocker_g2m,
    comparators=comparators_g2m,
    weights=[0.55, 0.25, 0.10, 0.07, 0.03],
    threshold=0.74,
    id_column="id",
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_a2g, rb_correspondences_a2m, rb_correspondences_g2m],
    ignore_index=True,
)

# --------------------------------
# Data Fusion
# Key fix for genres accuracy:
# - genres already normalized to canonical lowercase tokens; union() yields stable merged lists.
# Other attributes:
# - publish_year/page_count: maximum (fills missing from amazon; keeps plausible later reprints)
# - title/author/publisher/language: longest_string tends to keep most informative value
# --------------------------------

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("publish_year", maximum)
strategy.add_attribute_fuser("page_count", maximum)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("rating", maximum)
strategy.add_attribute_fuser("numratings", maximum)
strategy.add_attribute_fuser("price", maximum)

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

# write output
rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

