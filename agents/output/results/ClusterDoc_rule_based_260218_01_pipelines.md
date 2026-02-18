# Pipeline Snapshots

notebook_name=ClusterDoc
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=10.66%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_csv
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union

import pandas as pd

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

# Define dataset paths
DISCogs_PATH = "output/schema-matching/discogs.csv"
LASTFM_PATH = "output/schema-matching/lastfm.csv"
MUSICBRAINZ_PATH = "output/schema-matching/musicbrainz.csv"

# Load datasets (already schema-matched outputs)
discogs = load_csv(DISCogs_PATH, name="discogs")
lastfm = load_csv(LASTFM_PATH, name="lastfm")
musicbrainz = load_csv(MUSICBRAINZ_PATH, name="musicbrainz")

datasets = [discogs, lastfm, musicbrainz]

# --------------------------------
# Blocking
# NOTE: No precomputed blocking config was provided for these music datasets.
# Use reasonable blockers based on available columns.
# --------------------------------

print("Performing Blocking")

# discogs <-> lastfm: exact block on artist (reduces candidate pairs strongly)
blocker_discogs_lastfm = StandardBlocker(
    discogs, lastfm,
    on=["artist"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id"
)

# discogs <-> musicbrainz: exact block on artist
blocker_discogs_musicbrainz = StandardBlocker(
    discogs, musicbrainz,
    on=["artist"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id"
)

# lastfm <-> musicbrainz: embedding block on name+artist (lastfm "name" often contains artist prefix)
# Use concatenated text columns for semantic blocking
blocker_lastfm_musicbrainz = EmbeddingBlocker(
    lastfm, musicbrainz,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id"
)

# --------------------------------
# Matching
# NOTE: No precomputed matching config was provided for these music datasets.
# Use a robust rule-based setup with sensible weights/thresholds.
# --------------------------------

print("Matching Entities")

# Required threshold variables (pair keys)
threshold_discogs_lastfm = 0.78
threshold_discogs_musicbrainz = 0.82
threshold_lastfm_musicbrainz = 0.78

comparators_discogs_lastfm = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    StringComparator(
        column="release-date",
        similarity_function="levenshtein",
        preprocess=lambda x: str(x).strip(),
    ),
    NumericComparator(
        column="duration",
        max_difference=30,  # seconds tolerance
    ),
]

comparators_discogs_musicbrainz = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    StringComparator(
        column="release-date",
        similarity_function="levenshtein",
        preprocess=lambda x: str(x).strip(),
    ),
    StringComparator(
        column="release-country",
        similarity_function="cosine",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    NumericComparator(
        column="duration",
        max_difference=30,
    ),
]

comparators_lastfm_musicbrainz = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    NumericComparator(
        column="duration",
        max_difference=30,
    ),
]

matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=[0.30, 0.45, 0.10, 0.15],
    threshold=threshold_discogs_lastfm,
    id_column="id",
)

rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=[0.25, 0.40, 0.12, 0.08, 0.15],
    threshold=threshold_discogs_musicbrainz,
    id_column="id",
)

rb_correspondences_lastfm_musicbrainz = matcher.match(
    df_left=lastfm,
    df_right=musicbrainz,
    candidates=blocker_lastfm_musicbrainz,
    comparators=comparators_lastfm_musicbrainz,
    weights=[0.25, 0.55, 0.20],
    threshold=threshold_lastfm_musicbrainz,
    id_column="id",
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_lastfm_musicbrainz,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

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

rb_fused_standard_blocker = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
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
node_index=13
node_name=execute_pipeline
accuracy_score=12.69%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_csv
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union

import pandas as pd
import numpy as np

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATASETS = {
    "discogs": "output/schema-matching/discogs.csv",
    "lastfm": "output/schema-matching/lastfm.csv",
    "musicbrainz": "output/schema-matching/musicbrainz.csv",
}

discogs = load_csv(DATASETS["discogs"], name="discogs")
lastfm = load_csv(DATASETS["lastfm"], name="lastfm")
musicbrainz = load_csv(DATASETS["musicbrainz"], name="musicbrainz")

# --------------------------------
# Blocking
# (No precomputed blocking config provided for these datasets)
# --------------------------------

print("Performing Blocking")

blocker_discogs_lastfm = StandardBlocker(
    discogs, lastfm,
    on=["artist"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_discogs_musicbrainz = StandardBlocker(
    discogs, musicbrainz,
    on=["artist"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_lastfm_musicbrainz = EmbeddingBlocker(
    lastfm, musicbrainz,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching
# (No precomputed matching config provided for these datasets)
# IMPORTANT: define threshold_[pair_key] variables and use them
# --------------------------------

print("Matching Entities")

threshold_discogs_lastfm = 0.78
threshold_discogs_musicbrainz = 0.82
threshold_lastfm_musicbrainz = 0.78

lower_strip = lambda x: str(x).lower().strip()
strip_ = lambda x: str(x).strip()

comparators_discogs_lastfm = [
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
        column="release-date",
        similarity_function="levenshtein",
        preprocess=strip_,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=30,
    ),
]

comparators_discogs_musicbrainz = [
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
        column="release-date",
        similarity_function="levenshtein",
        preprocess=strip_,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="release-country",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=30,
    ),
]

comparators_lastfm_musicbrainz = [
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
    NumericComparator(
        column="duration",
        max_difference=30,
    ),
]

matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=[0.30, 0.45, 0.10, 0.15],
    threshold=threshold_discogs_lastfm,
    id_column="id",
)

rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=[0.25, 0.40, 0.12, 0.08, 0.15],
    threshold=threshold_discogs_musicbrainz,
    id_column="id",
)

rb_correspondences_lastfm_musicbrainz = matcher.match(
    df_left=lastfm,
    df_right=musicbrainz,
    candidates=blocker_lastfm_musicbrainz,
    comparators=comparators_lastfm_musicbrainz,
    weights=[0.25, 0.55, 0.20],
    threshold=threshold_lastfm_musicbrainz,
    id_column="id",
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_lastfm_musicbrainz,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# Use schema from discogs as reference (already schema-matched inputs, but some cols may be missing in some sources)
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

rb_fused_standard_blocker = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
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
node_index=20
node_name=execute_pipeline
accuracy_score=8.12%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_csv
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union

import pandas as pd

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

discogs = load_csv("output/schema-matching/discogs.csv", name="discogs")
lastfm = load_csv("output/schema-matching/lastfm.csv", name="lastfm")
musicbrainz = load_csv("output/schema-matching/musicbrainz.csv", name="musicbrainz")

datasets = [discogs, lastfm, musicbrainz]

# --------------------------------
# Blocking
# --------------------------------

print("Performing Blocking")

# Discogs <-> LastFM: exact block on artist
blocker_discogs_lastfm = StandardBlocker(
    discogs, lastfm,
    on=["artist"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# Discogs <-> MusicBrainz: exact block on artist
blocker_discogs_musicbrainz = StandardBlocker(
    discogs, musicbrainz,
    on=["artist"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# LastFM <-> MusicBrainz: semantic block on (name, artist)
blocker_lastfm_musicbrainz = EmbeddingBlocker(
    lastfm, musicbrainz,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching
# IMPORTANT: define threshold_[pair_key] variables and use them
# --------------------------------

print("Matching Entities")

lower_strip = lambda x: str(x).lower().strip()
strip_ = lambda x: str(x).strip()

threshold_discogs_lastfm = 0.78
threshold_discogs_musicbrainz = 0.82
threshold_lastfm_musicbrainz = 0.78

comparators_discogs_lastfm = [
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
        column="release-date",
        similarity_function="levenshtein",
        preprocess=strip_,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=30,
    ),
]

comparators_discogs_musicbrainz = [
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
        column="release-date",
        similarity_function="levenshtein",
        preprocess=strip_,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="release-country",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=30,
    ),
]

comparators_lastfm_musicbrainz = [
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
    NumericComparator(
        column="duration",
        max_difference=30,
    ),
]

matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=[0.30, 0.45, 0.10, 0.15],
    threshold=threshold_discogs_lastfm,
    id_column="id",
)

rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=[0.25, 0.40, 0.12, 0.08, 0.15],
    threshold=threshold_discogs_musicbrainz,
    id_column="id",
)

rb_correspondences_lastfm_musicbrainz = matcher.match(
    df_left=lastfm,
    df_right=musicbrainz,
    candidates=blocker_lastfm_musicbrainz,
    comparators=comparators_lastfm_musicbrainz,
    weights=[0.25, 0.55, 0.20],
    threshold=threshold_lastfm_musicbrainz,
    id_column="id",
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_lastfm_musicbrainz,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

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

rb_fused_standard_blocker = engine.run(
    datasets=datasets,
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

