# Pipeline Snapshots

notebook_name=AdaptationPipeline_blocking_matching_extension_Final_Reasoning
matcher_mode=rb

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=55.84%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv, load_xml

from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    TokenBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
discogs = load_csv(
    DATA_DIR + "discogs.csv",
    name="discogs",
)

lastfm = load_csv(
    DATA_DIR + "lastfm.csv",
    name="lastfm",
)

musicbrainz = load_csv(
    DATA_DIR + "musicbrainz.csv",
    name="musicbrainz",
)

datasets = [discogs, lastfm, musicbrainz]

# --------------------------------
# Schema Matching (LLM-based)
# Schema of lastfm and musicbrainz will be matched/renamed to the schema of discogs
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True,
)

# match schema of discogs with lastfm and rename schema of lastfm
schema_correspondences = matcher.match(discogs, lastfm)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
lastfm = lastfm.rename(columns=rename_map)

# match schema of discogs with musicbrainz and rename schema of musicbrainz
schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
musicbrainz = musicbrainz.rename(columns=rename_map)

# --------------------------------
# Blocking (use precomputed strategies)
# --------------------------------

print("Performing Blocking")

# id columns from config
discogs_id_col = "id"
lastfm_id_col = "id"
musicbrainz_id_col = "id"

# discogs_lastfm: semantic_similarity on ["name", "artist"], top_k=10
blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=discogs_id_col,
)

# discogs_musicbrainz: sorted_neighbourhood on ["name"], window=15
blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    window=15,
    id_column=discogs_id_col,
    output_dir="output/blocking-evaluation",
)

# musicbrainz_lastfm: semantic_similarity on ["name", "artist", "duration"], top_k=10
blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=musicbrainz_id_col,
)

# --------------------------------
# Matching (Rule-based, using provided configuration)
# --------------------------------

print("Matching Entities")

def lower_strip(x):
    return str(x).lower().strip() if pd.notnull(x) else ""

# discogs_lastfm comparators
comparators_discogs_lastfm = [
    # artist
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # name
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # tracks_track_name
    StringComparator(
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    # duration
    NumericComparator(
        column="duration",
        max_difference=10.0,
    ),
]

weights_discogs_lastfm = [0.3, 0.35, 0.25, 0.1]
threshold_discogs_lastfm = 0.6

# discogs_musicbrainz comparators
comparators_discogs_musicbrainz = [
    # name
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # artist
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # release-date
    DateComparator(
        column="release-date",
        max_days_difference=365,
    ),
    # release-country
    StringComparator(
        column="release-country",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # duration
    NumericComparator(
        column="duration",
        max_difference=60.0,
    ),
]

weights_discogs_musicbrainz = [0.35, 0.25, 0.2, 0.1, 0.1]
threshold_discogs_musicbrainz = 0.7

# musicbrainz_lastfm comparators
comparators_musicbrainz_lastfm = [
    # artist
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # name
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # duration (treated as string similarity as in config)
    StringComparator(
        column="duration",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

weights_musicbrainz_lastfm = [0.35, 0.45, 0.2]
threshold_musicbrainz_lastfm = 0.75

# Initialize Rule-Based Matcher
rb_matcher = RuleBasedMatcher()

# discogs vs lastfm
rb_correspondences_discogs_lastfm = rb_matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=weights_discogs_lastfm,
    threshold=threshold_discogs_lastfm,
    id_column=discogs_id_col,
)

# discogs vs musicbrainz
rb_correspondences_discogs_musicbrainz = rb_matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=weights_discogs_musicbrainz,
    threshold=threshold_discogs_musicbrainz,
    id_column=discogs_id_col,
)

# musicbrainz vs lastfm
rb_correspondences_musicbrainz_lastfm = rb_matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_musicbrainz_lastfm,
    comparators=comparators_musicbrainz_lastfm,
    weights=weights_musicbrainz_lastfm,
    threshold=threshold_musicbrainz_lastfm,
    id_column=musicbrainz_id_col,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

# merge rule based correspondences
all_rb_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_musicbrainz_lastfm,
    ],
    ignore_index=True,
)

# define data fusion strategy
strategy = DataFusionStrategy("music_fusion_strategy")

# basic attribute fusers (discogs schema as reference)
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("duration", longest_string)
strategy.add_attribute_fuser("label", longest_string)
strategy.add_attribute_fuser("genre", longest_string)
strategy.add_attribute_fuser("tracks_track_name", union)
strategy.add_attribute_fuser("tracks_track_position", union)
strategy.add_attribute_fuser("tracks_track_duration", union)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output (do NOT change this file name)
rb_fused.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=12
node_name=execute_pipeline
accuracy_score=70.56%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv, load_xml

from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    TokenBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    union,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import re
import unicodedata
from collections import Counter

# --------------------------------
# Helper functions for preprocessing and fusion
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip() if pd.notnull(x) else ""

def normalize_unicode(x):
    if pd.isnull(x):
        return x
    return unicodedata.normalize("NFKC", str(x))

def canonical_string(values):
    """
    Pick the most frequent normalized value; break ties by shortest length.
    """
    cleaned = []
    for v in values:
        if pd.isnull(v):
            continue
        s = lower_strip(normalize_unicode(v))
        if s:
            cleaned.append(s)
    if not cleaned:
        return None
    counts = Counter(cleaned)
    best_norm, _ = max(counts.items(), key=lambda x: (x[1], -len(x[0])))
    # Return one of the original values that maps to best_norm, preferring shorter
    candidates = [
        v for v in values
        if not pd.isnull(v) and lower_strip(normalize_unicode(v)) == best_norm
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda v: len(str(v)))

def numeric_median(values):
    """
    Parse all values as floats, return median as int (seconds) if possible.
    """
    nums = []
    for v in values:
        if pd.isnull(v):
            continue
        s = str(v).strip()
        if not s:
            continue
        try:
            nums.append(float(s))
        except (TypeError, ValueError):
            # try to extract digits from mixed strings
            m = re.search(r"\d+", s)
            if m:
                try:
                    nums.append(float(m.group(0)))
                except ValueError:
                    continue
            continue
    if not nums:
        return None
    return int(round(np.median(nums)))

def normalize_track_title(t):
    if pd.isnull(t):
        return ""
    t = normalize_unicode(t)
    t = str(t).strip()
    # remove leading track numbers and punctuation, e.g. "01 ", "1. ", "01-"
    t = re.sub(r"^\d+\s*[-.)]*\s*", "", t)
    return t.strip()

def dedup_track_list(values):
    """
    Deduplicate and normalize track titles across sources.
    Keep one canonical form per semantic track.
    """
    # Flatten possible list-like strings: assume values are already list-like strings or lists
    tracks = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, (list, tuple, set)):
            tracks.extend(list(v))
        else:
            # strings like "['A', 'B']" -> try eval-safe literal_eval
            s = str(v)
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = eval(s, {"__builtins__": {}}, {})
                    if isinstance(parsed, (list, tuple, set)):
                        tracks.extend(list(parsed))
                        continue
                except Exception:
                    pass
            tracks.append(s)

    if not tracks:
        return None

    # Normalize titles
    normalized = [(orig, normalize_track_title(orig).lower()) for orig in tracks]

    # Simple inner deduplication by normalized string
    by_norm = {}
    for orig, norm in normalized:
        if not norm:
            continue
        # keep shortest original variant for each normalized form
        if norm not in by_norm or len(str(orig)) < len(str(by_norm[norm])):
            by_norm[norm] = orig

    # Return canonical originals in a stable order
    canon = list(by_norm.values())
    return canon if canon else None

def list_union(values):
    """
    Safer union for list-like attributes: flatten and unique while preserving order.
    """
    items = []
    seen = set()
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, (list, tuple, set)):
            it = list(v)
        else:
            s = str(v)
            if s.startswith("[") and s.endswith("]"):
                try:
                    it = eval(s, {"__builtins__": {}}, {})
                    if not isinstance(it, (list, tuple, set)):
                        it = [s]
                    else:
                        it = list(it)
                except Exception:
                    it = [s]
            else:
                it = [s]
        for x in it:
            if x not in seen:
                seen.add(x)
                items.append(x)
    return items if items else None

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
discogs = load_csv(
    DATA_DIR + "discogs.csv",
    name="discogs",
)

lastfm = load_csv(
    DATA_DIR + "lastfm.csv",
    name="lastfm",
)

musicbrainz = load_csv(
    DATA_DIR + "musicbrainz.csv",
    name="musicbrainz",
)

# Basic text normalization on key columns before matching/fusion
for df in [discogs, lastfm, musicbrainz]:
    for col in ["name", "artist", "label", "genre", "release-country", "tracks_track_name"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_unicode)

datasets = [discogs, lastfm, musicbrainz]

# --------------------------------
# Schema Matching (LLM-based)
# Schema of lastfm and musicbrainz will be matched/renamed to the schema of discogs
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True,
)

# match schema of discogs with lastfm and rename schema of lastfm
schema_correspondences = matcher.match(discogs, lastfm)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
lastfm = lastfm.rename(columns=rename_map)

# match schema of discogs with musicbrainz and rename schema of musicbrainz
schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
musicbrainz = musicbrainz.rename(columns=rename_map)

# --------------------------------
# Blocking (use precomputed strategies)
# --------------------------------

print("Performing Blocking")

# id columns from config
discogs_id_col = "id"
lastfm_id_col = "id"
musicbrainz_id_col = "id"

# discogs_lastfm: semantic_similarity on ["name", "artist"], top_k=10
blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=discogs_id_col,
)

# discogs_musicbrainz: sorted_neighbourhood on ["name"], window=15
blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    window=15,
    id_column=discogs_id_col,
    output_dir="output/blocking-evaluation",
)

# musicbrainz_lastfm: semantic_similarity on ["name", "artist", "duration"], top_k=10
blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=musicbrainz_id_col,
)

# --------------------------------
# Matching (Rule-based, using provided configuration)
# --------------------------------

print("Matching Entities")

# discogs_lastfm comparators
comparators_discogs_lastfm = [
    # artist
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # name
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # tracks_track_name
    StringComparator(
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    # duration
    NumericComparator(
        column="duration",
        max_difference=10.0,
    ),
]

weights_discogs_lastfm = [0.3, 0.35, 0.25, 0.1]
threshold_discogs_lastfm = 0.6

# discogs_musicbrainz comparators
comparators_discogs_musicbrainz = [
    # name
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # artist
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # release-date
    DateComparator(
        column="release-date",
        max_days_difference=365,
    ),
    # release-country
    StringComparator(
        column="release-country",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # duration
    NumericComparator(
        column="duration",
        max_difference=60.0,
    ),
]

weights_discogs_musicbrainz = [0.35, 0.25, 0.2, 0.1, 0.1]
threshold_discogs_musicbrainz = 0.7

# musicbrainz_lastfm comparators
comparators_musicbrainz_lastfm = [
    # artist
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # name
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # duration (treated as string similarity as in config)
    StringComparator(
        column="duration",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

weights_musicbrainz_lastfm = [0.35, 0.45, 0.2]
threshold_musicbrainz_lastfm = 0.75

# Initialize Rule-Based Matcher
rb_matcher = RuleBasedMatcher()

# discogs vs lastfm
rb_correspondences_discogs_lastfm = rb_matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=weights_discogs_lastfm,
    threshold=threshold_discogs_lastfm,
    id_column=discogs_id_col,
)

# discogs vs musicbrainz
rb_correspondences_discogs_musicbrainz = rb_matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=weights_discogs_musicbrainz,
    threshold=threshold_discogs_musicbrainz,
    id_column=discogs_id_col,
)

# musicbrainz vs lastfm
rb_correspondences_musicbrainz_lastfm = rb_matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_musicbrainz_lastfm,
    comparators=comparators_musicbrainz_lastfm,
    weights=weights_musicbrainz_lastfm,
    threshold=threshold_musicbrainz_lastfm,
    id_column=musicbrainz_id_col,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

# merge rule based correspondences
all_rb_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_musicbrainz_lastfm,
    ],
    ignore_index=True,
)

# define data fusion strategy
strategy = DataFusionStrategy("music_fusion_strategy")

# Attribute-specific fusers (discogs schema as reference)
strategy.add_attribute_fuser("name", canonical_string)
strategy.add_attribute_fuser("artist", canonical_string)
strategy.add_attribute_fuser("release-date", canonical_string)
strategy.add_attribute_fuser("release-country", canonical_string)
strategy.add_attribute_fuser("duration", numeric_median)
strategy.add_attribute_fuser("label", canonical_string)
strategy.add_attribute_fuser("genre", canonical_string)

# List-like attributes with better handling
strategy.add_attribute_fuser("tracks_track_name", dedup_track_list)
strategy.add_attribute_fuser("tracks_track_position", list_union)
strategy.add_attribute_fuser("tracks_track_duration", list_union)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=True,
)

# write output (do NOT change this file name)
rb_fused.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=20
node_name=execute_pipeline
accuracy_score=pending
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv, load_xml

from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    TokenBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    union,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import re
import unicodedata
from collections import Counter

# --------------------------------
# Helper functions for preprocessing and fusion
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip() if pd.notnull(x) else ""


def normalize_unicode(x):
    if pd.isnull(x):
        return x
    return unicodedata.normalize("NFKC", str(x))


# --- Country normalization for better, consistent fusion ---
_COUNTRY_MAP = {
    "uk": "United Kingdom of Great Britain and Northern Ireland",
    "u.k.": "United Kingdom of Great Britain and Northern Ireland",
    "united kingdom": "United Kingdom of Great Britain and Northern Ireland",
    "gb": "United Kingdom of Great Britain and Northern Ireland",
    "great britain": "United Kingdom of Great Britain and Northern Ireland",
    "england": "United Kingdom of Great Britain and Northern Ireland",
    "usa": "United States of America",
    "u.s.a.": "United States of America",
    "united states": "United States of America",
    "united states of america": "United States of America",
    "us": "United States of America",
    "u.s.": "United States of America",
}


def normalize_country_value(v):
    if pd.isnull(v):
        return None
    s = lower_strip(normalize_unicode(v))
    if not s:
        return None
    mapped = _COUNTRY_MAP.get(s)
    return mapped if mapped is not None else v


def canonical_string(values):
    cleaned = []
    for v in values:
        if pd.isnull(v):
            continue
        s = lower_strip(normalize_unicode(v))
        if s:
            cleaned.append(s)
    if not cleaned:
        return None
    counts = Counter(cleaned)
    best_norm, _ = max(counts.items(), key=lambda x: (x[1], -len(x[0])))
    candidates = [
        v
        for v in values
        if not pd.isnull(v) and lower_strip(normalize_unicode(v)) == best_norm
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda v: len(str(v)))


def canonical_country(values):
    norm_vals = []
    for v in values:
        nv = normalize_country_value(v)
        if nv is not None:
            norm_vals.append(nv)
    if not norm_vals:
        return None
    return canonical_string(norm_vals)


def numeric_median(values):
    nums = []
    for v in values:
        if pd.isnull(v):
            continue
        s = str(v).strip()
        if not s:
            continue
        try:
            x = float(s)
        except (TypeError, ValueError):
            m = re.search(r"\d+", s)
            if not m:
                continue
            try:
                x = float(m.group(0))
            except ValueError:
                continue

        if x == 0:
            continue

        if 1800 < x < 180000:
            x = x / 1000.0

        if 5 <= x <= 10800:
            nums.append(x)

    if not nums:
        return None

    nums = sorted(nums)
    k = len(nums)
    if k > 4:
        lo = int(0.1 * k)
        hi = max(lo + 1, int(0.9 * k))
        trimmed = nums[lo:hi]
    else:
        trimmed = nums

    if not trimmed:
        trimmed = nums
    return int(round(np.median(trimmed)))


def normalize_track_title(t):
    if pd.isnull(t):
        return ""
    t = normalize_unicode(t)
    t = str(t).strip()

    t = re.sub(r"^(disc\s*\d+\s*[:.\-]\s*)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\d+\s*[-.)]*\s*", "", t)

    t = re.sub(r"\s+", " ", t).strip()

    base = t.lower()

    base = re.sub(r"\s*\(bonus[^\)]*\)", "", base)
    base = re.sub(r"\s*\(album version\)", "", base)
    base = re.sub(r"\s*\(album\)", "", base)
    base = re.sub(r"\s*\(original mix\)", "", base)
    base = re.sub(r"\s*\(radio edit\)", "", base)
    base = re.sub(r"\s*\(remaster.*?\)", "", base)
    base = re.sub(r"\s*\(live[^\)]]*\)", "", base)

    base = base.replace("&", "and")
    base = re.sub(r"[^\w\s]", "", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base


def _safe_parse_list(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    s = str(value).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = eval(s, {"__builtins__": {}}, {})
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
            return [s]
        except Exception:
            return [s]
    if s == "":
        return []
    return [s]


def dedup_track_list(values):
    tracks = []
    for v in values:
        tracks.extend(_safe_parse_list(v))

    if not tracks:
        return None

    normalized = [(orig, normalize_track_title(orig)) for orig in tracks]

    by_norm = {}
    for orig, norm in normalized:
        if not norm:
            continue
        if norm not in by_norm or len(str(orig)) < len(str(by_norm[norm])):
            by_norm[norm] = orig

    canon = list(by_norm.values())
    return canon if canon else None


def list_union(values):
    items = []
    seen = set()
    for v in values:
        for x in _safe_parse_list(v):
            key = str(x)
            try:
                num = float(key)
                if num <= 0:
                    continue
            except Exception:
                pass
            if key not in seen:
                seen.add(key)
                items.append(x)
    return items if items else None


def align_tracks_fuser(all_values):
    """
    Joint fuser for track names, positions, and durations.
    all_values is a dict-like structure provided by the engine:
    {
      'tracks_track_name': [...],
      'tracks_track_position': [...],
      'tracks_track_duration': [...]
    }
    We:
    - deduplicate/normalize names
    - align positions/durations by index
    """
    names_vals = all_values.get("tracks_track_name", [])
    pos_vals = all_values.get("tracks_track_position", [])
    dur_vals = all_values.get("tracks_track_duration", [])

    names = []
    for v in names_vals:
        names.extend(_safe_parse_list(v))
    if not names:
        return {
            "tracks_track_name": None,
            "tracks_track_position": None,
            "tracks_track_duration": None,
        }

    norm_to_name = {}
    ordered_norms = []
    for n in names:
        norm = normalize_track_title(n)
        if not norm:
            continue
        if norm not in norm_to_name or len(str(n)) < len(str(norm_to_name[norm])):
            norm_to_name[norm] = n
        if norm not in ordered_norms:
            ordered_norms.append(norm)

    fused_names = [norm_to_name[n] for n in ordered_norms]

    fused_positions = list(range(1, len(fused_names) + 1))

    dur_candidates = []
    for v in dur_vals:
        dur_candidates.append(_safe_parse_list(v))

    fused_durations = []
    for _ in fused_names:
        track_durs = []
        for src_list in dur_candidates:
            if not src_list:
                continue
            if len(src_list) == len(fused_names):
                track_durs.append(src_list[len(fused_durations)])
        if not track_durs:
            fused_durations.append(None)
        else:
            fused_durations.append(numeric_median(track_durs))

    return {
        "tracks_track_name": fused_names,
        "tracks_track_position": fused_positions,
        "tracks_track_duration": fused_durations,
    }


# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

discogs = load_csv(
    DATA_DIR + "discogs.csv",
    name="discogs",
)

lastfm = load_csv(
    DATA_DIR + "lastfm.csv",
    name="lastfm",
)

musicbrainz = load_csv(
    DATA_DIR + "musicbrainz.csv",
    name="musicbrainz",
)

for df in [discogs, lastfm, musicbrainz]:
    for col in ["name", "artist", "label", "genre", "release-country", "tracks_track_name"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_unicode)

datasets = [discogs, lastfm, musicbrainz]

# --------------------------------
# Schema Matching (LLM-based)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True,
)

schema_correspondences = matcher.match(discogs, lastfm)
rename_map = (
    schema_correspondences.set_index("target_column")["source_column"].to_dict()
)
lastfm = lastfm.rename(columns=rename_map)

schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = (
    schema_correspondences.set_index("target_column")["source_column"].to_dict()
)
musicbrainz = musicbrainz.rename(columns=rename_map)

# --------------------------------
# Blocking (use precomputed strategies)
# --------------------------------

print("Performing Blocking")

discogs_id_col = "id"
lastfm_id_col = "id"
musicbrainz_id_col = "id"

blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=discogs_id_col,
)

blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    window=15,
    id_column=discogs_id_col,
    output_dir="output/blocking-evaluation",
)

blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=musicbrainz_id_col,
)

# --------------------------------
# Matching (Rule-based, using provided configuration)
# --------------------------------

print("Matching Entities")

comparators_discogs_lastfm = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=10.0,
    ),
]

weights_discogs_lastfm = [0.3, 0.35, 0.25, 0.1]
threshold_discogs_lastfm = 0.6

comparators_discogs_musicbrainz = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="release-date",
        max_days_difference=365,
    ),
    StringComparator(
        column="release-country",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=60.0,
    ),
]

weights_discogs_musicbrainz = [0.35, 0.25, 0.2, 0.1, 0.1]
threshold_discogs_musicbrainz = 0.7

comparators_musicbrainz_lastfm = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="duration",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

weights_musicbrainz_lastfm = [0.35, 0.45, 0.2]
threshold_musicbrainz_lastfm = 0.75

rb_matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = rb_matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=weights_discogs_lastfm,
    threshold=threshold_discogs_lastfm,
    id_column=discogs_id_col,
)

rb_correspondences_discogs_musicbrainz = rb_matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=weights_discogs_musicbrainz,
    threshold=threshold_discogs_musicbrainz,
    id_column=discogs_id_col,
)

rb_correspondences_musicbrainz_lastfm = rb_matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_musicbrainz_lastfm,
    comparators=comparators_musicbrainz_lastfm,
    weights=weights_musicbrainz_lastfm,
    threshold=threshold_musicbrainz_lastfm,
    id_column=musicbrainz_id_col,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_musicbrainz_lastfm,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("music_fusion_strategy")

strategy.add_attribute_fuser("name", canonical_string)
strategy.add_attribute_fuser("artist", canonical_string)
strategy.add_attribute_fuser("release-date", canonical_string)
strategy.add_attribute_fuser("release-country", canonical_country)
strategy.add_attribute_fuser("duration", numeric_median)
strategy.add_attribute_fuser("label", canonical_string)
strategy.add_attribute_fuser("genre", canonical_string)

strategy.add_attribute_fuser("tracks_track_name", dedup_track_list)
strategy.add_attribute_fuser("tracks_track_position", list_union)
strategy.add_attribute_fuser("tracks_track_duration", list_union)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=True,
)

rb_fused.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

