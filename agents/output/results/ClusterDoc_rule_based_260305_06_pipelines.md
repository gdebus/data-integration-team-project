# Pipeline Snapshots

notebook_name=ClusterDoc
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=54.82%
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
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
from dotenv import load_dotenv
import os
import shutil

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
# Schema Matching
# Match schema of lastfm and musicbrainz to discogs
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# lastfm -> discogs schema
schema_correspondences = matcher.match(discogs, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map)

# musicbrainz -> discogs schema
schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
musicbrainz = musicbrainz.rename(columns=rename_map)

# --------------------------------
# Blocking
# Use precomputed optimal blocking strategies
# --------------------------------

print("Performing Blocking")

ID_COLUMNS = {
    "discogs": "id",
    "lastfm": "id",
    "musicbrainz": "id",
}

# discogs_lastfm -> semantic_similarity on ["name", "artist"], top_k=10
blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    id_column=ID_COLUMNS["discogs"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

# discogs_musicbrainz -> sorted_neighbourhood on ["name"], window=15
blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    id_column=ID_COLUMNS["discogs"],
    window=15,
    batch_size=100000,
    output_dir="output/blocking-evaluation",
)

# musicbrainz_lastfm -> semantic_similarity on ["name", "artist", "duration"], top_k=10
blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    id_column=ID_COLUMNS["musicbrainz"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# Matching
# Use precomputed matching strategies
# --------------------------------

print("Matching Entities")

def lower_strip(x):
    return str(x).lower().strip() if pd.notnull(x) else x

# Threshold variables as required
threshold_discogs_lastfm = 0.6
threshold_discogs_musicbrainz = 0.7
threshold_musicbrainz_lastfm = 0.75

# discogs_lastfm comparators
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

# discogs_musicbrainz comparators
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

# musicbrainz_lastfm comparators
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

matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=weights_discogs_lastfm,
    threshold=threshold_discogs_lastfm,
    id_column=ID_COLUMNS["discogs"],
)

rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=weights_discogs_musicbrainz,
    threshold=threshold_discogs_musicbrainz,
    id_column=ID_COLUMNS["discogs"],
)

rb_correspondences_musicbrainz_lastfm = matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_musicbrainz_lastfm,
    comparators=comparators_musicbrainz_lastfm,
    weights=weights_musicbrainz_lastfm,
    threshold=threshold_musicbrainz_lastfm,
    id_column=ID_COLUMNS["musicbrainz"],
)

# --------------------------------
# Save Correspondences
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_discogs_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_lastfm.csv",
    ),
    index=False,
)
rb_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_musicbrainz.csv",
    ),
    index=False,
)
rb_correspondences_musicbrainz_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_musicbrainz_lastfm.csv",
    ),
    index=False,
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

# Choose reasonable fusers based on columns seen in schemas
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("duration", maximum)
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

fused_data = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
os.makedirs("output/data_fusion", exist_ok=True)
fused_data.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=9
node_name=execute_pipeline
accuracy_score=54.82%
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
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
from dotenv import load_dotenv
import os
import shutil

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
# Schema Matching
# Match schema of lastfm and musicbrainz to discogs
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# lastfm -> discogs schema
schema_correspondences = matcher.match(discogs, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map)

# musicbrainz -> discogs schema
schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
musicbrainz = musicbrainz.rename(columns=rename_map)

# --------------------------------
# Blocking
# Use precomputed optimal blocking strategies
# --------------------------------

print("Performing Blocking")

ID_COLUMNS = {
    "discogs": "id",
    "lastfm": "id",
    "musicbrainz": "id",
}

# discogs_lastfm -> semantic_similarity on ["name", "artist"], top_k=10
blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    id_column=ID_COLUMNS["discogs"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

# discogs_musicbrainz -> sorted_neighbourhood on ["name"], window=15
blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    id_column=ID_COLUMNS["discogs"],
    window=15,
    batch_size=100000,
    output_dir="output/blocking-evaluation",
)

# musicbrainz_lastfm -> semantic_similarity on ["name", "artist", "duration"], top_k=10
blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    id_column=ID_COLUMNS["musicbrainz"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# Matching
# Use precomputed (updated) matching strategies
# --------------------------------

print("Matching Entities")

def lower_strip(x):
    return str(x).lower().strip() if pd.notnull(x) else x

# Threshold variables as required
threshold_discogs_lastfm = 0.6
threshold_discogs_musicbrainz = 0.7
threshold_musicbrainz_lastfm = 0.75

# discogs_lastfm comparators
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

# updated weights from matching configuration
weights_discogs_lastfm = [0.35, 0.4, 0.15, 0.1]

# discogs_musicbrainz comparators
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

# updated weights from matching configuration
weights_discogs_musicbrainz = [0.32, 0.32, 0.16, 0.1, 0.1]

# musicbrainz_lastfm comparators
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

# updated weights from matching configuration
weights_musicbrainz_lastfm = [0.32, 0.5, 0.18]

matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=weights_discogs_lastfm,
    threshold=threshold_discogs_lastfm,
    id_column=ID_COLUMNS["discogs"],
)

rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=weights_discogs_musicbrainz,
    threshold=threshold_discogs_musicbrainz,
    id_column=ID_COLUMNS["discogs"],
)

rb_correspondences_musicbrainz_lastfm = matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_musicbrainz_lastfm,
    comparators=comparators_musicbrainz_lastfm,
    weights=weights_musicbrainz_lastfm,
    threshold=threshold_musicbrainz_lastfm,
    id_column=ID_COLUMNS["musicbrainz"],
)

# --------------------------------
# Save Correspondences
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_discogs_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_lastfm.csv",
    ),
    index=False,
)
rb_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_musicbrainz.csv",
    ),
    index=False,
)
rb_correspondences_musicbrainz_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_musicbrainz_lastfm.csv",
    ),
    index=False,
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

# Choose reasonable fusers based on columns seen in schemas
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("duration", maximum)
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

fused_data = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
os.makedirs("output/data_fusion", exist_ok=True)
fused_data.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=12
node_name=execute_pipeline
accuracy_score=54.82%
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
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
from dotenv import load_dotenv
import os
import shutil

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
# Schema Matching
# Match schema of lastfm and musicbrainz to discogs
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# lastfm -> discogs schema
schema_correspondences = matcher.match(discogs, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map)

# musicbrainz -> discogs schema
schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
musicbrainz = musicbrainz.rename(columns=rename_map)

# --------------------------------
# Blocking
# Use precomputed optimal blocking strategies
# --------------------------------

print("Performing Blocking")

ID_COLUMNS = {
    "discogs": "id",
    "lastfm": "id",
    "musicbrainz": "id",
}

# discogs_lastfm -> semantic_similarity on ["name", "artist"], top_k=10
blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    id_column=ID_COLUMNS["discogs"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

# discogs_musicbrainz -> sorted_neighbourhood on ["name"], window=15
blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    id_column=ID_COLUMNS["discogs"],
    window=15,
    batch_size=100000,
    output_dir="output/blocking-evaluation",
)

# musicbrainz_lastfm -> semantic_similarity on ["name", "artist", "duration"], top_k=10
blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    id_column=ID_COLUMNS["musicbrainz"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# Matching
# Use precomputed (updated) matching strategies
# --------------------------------

print("Matching Entities")

def lower_strip(x):
    return str(x).lower().strip() if pd.notnull(x) else x

# Threshold variables as required
threshold_discogs_lastfm = 0.6
threshold_discogs_musicbrainz = 0.7
threshold_musicbrainz_lastfm = 0.75

# discogs_lastfm comparators
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

# weights from final matching configuration
weights_discogs_lastfm = [0.2, 0.25, 0.35, 0.2]

# discogs_musicbrainz comparators
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

# weights from final matching configuration
weights_discogs_musicbrainz = [0.22, 0.22, 0.26, 0.15, 0.15]

# musicbrainz_lastfm comparators
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

# weights from final matching configuration
weights_musicbrainz_lastfm = [0.4, 0.4, 0.2]

matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=weights_discogs_lastfm,
    threshold=threshold_discogs_lastfm,
    id_column=ID_COLUMNS["discogs"],
)

rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=weights_discogs_musicbrainz,
    threshold=threshold_discogs_musicbrainz,
    id_column=ID_COLUMNS["discogs"],
)

rb_correspondences_musicbrainz_lastfm = matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_musicbrainz_lastfm,
    comparators=comparators_musicbrainz_lastfm,
    weights=weights_musicbrainz_lastfm,
    threshold=threshold_musicbrainz_lastfm,
    id_column=ID_COLUMNS["musicbrainz"],
)

# --------------------------------
# Save Correspondences
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_discogs_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_lastfm.csv",
    ),
    index=False,
)
rb_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_musicbrainz.csv",
    ),
    index=False,
)
rb_correspondences_musicbrainz_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_musicbrainz_lastfm.csv",
    ),
    index=False,
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

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("duration", maximum)
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

fused_data = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
os.makedirs("output/data_fusion", exist_ok=True)
fused_data.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

============================================================
PIPELINE SNAPSHOT 04 START
============================================================
node_index=19
node_name=execute_pipeline
accuracy_score=pending
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
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
from dotenv import load_dotenv
import os
import shutil

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
# Schema Matching
# Match schema of lastfm and musicbrainz to discogs
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# lastfm -> discogs schema
schema_correspondences = matcher.match(discogs, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map)

# musicbrainz -> discogs schema
schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
musicbrainz = musicbrainz.rename(columns=rename_map)

# --------------------------------
# Blocking
# Use precomputed optimal blocking strategies
# --------------------------------

print("Performing Blocking")

ID_COLUMNS = {
    "discogs": "id",
    "lastfm": "id",
    "musicbrainz": "id",
}

# discogs_lastfm -> semantic_similarity on ["name", "artist"], top_k=10
blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    id_column=ID_COLUMNS["discogs"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

# discogs_musicbrainz -> sorted_neighbourhood on ["name"], window=15
blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    id_column=ID_COLUMNS["discogs"],
    window=15,
    batch_size=100000,
    output_dir="output/blocking-evaluation",
)

# musicbrainz_lastfm -> semantic_similarity on ["name", "artist", "duration"], top_k=10
blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    id_column=ID_COLUMNS["musicbrainz"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# Matching
# Use precomputed (updated) matching strategies
# --------------------------------

print("Matching Entities")

def lower_strip(x):
    return str(x).lower().strip() if pd.notnull(x) else x

# Threshold variables as required
threshold_discogs_lastfm = 0.6
threshold_discogs_musicbrainz = 0.7
threshold_musicbrainz_lastfm = 0.75

# discogs_lastfm comparators
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

# weights from final matching configuration
weights_discogs_lastfm = [0.2, 0.25, 0.35, 0.2]

# discogs_musicbrainz comparators
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

# weights from final matching configuration
weights_discogs_musicbrainz = [0.22, 0.22, 0.26, 0.15, 0.15]

# musicbrainz_lastfm comparators
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

# weights from final matching configuration
weights_musicbrainz_lastfm = [0.4, 0.4, 0.2]

matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=weights_discogs_lastfm,
    threshold=threshold_discogs_lastfm,
    id_column=ID_COLUMNS["discogs"],
)

rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=weights_discogs_musicbrainz,
    threshold=threshold_discogs_musicbrainz,
    id_column=ID_COLUMNS["discogs"],
)

rb_correspondences_musicbrainz_lastfm = matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_musicbrainz_lastfm,
    comparators=comparators_musicbrainz_lastfm,
    weights=weights_musicbrainz_lastfm,
    threshold=threshold_musicbrainz_lastfm,
    id_column=ID_COLUMNS["musicbrainz"],
)

# --------------------------------
# Save Correspondences
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_discogs_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_lastfm.csv",
    ),
    index=False,
)
rb_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_musicbrainz.csv",
    ),
    index=False,
)
rb_correspondences_musicbrainz_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_musicbrainz_lastfm.csv",
    ),
    index=False,
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

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("duration", maximum)
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

fused_data = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
os.makedirs("output/data_fusion", exist_ok=True)
fused_data.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 04 END
============================================================

============================================================
PIPELINE SNAPSHOT 05 START
============================================================
node_index=26
node_name=execute_pipeline
accuracy_score=pending
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
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
from dotenv import load_dotenv
import os
import shutil

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
# Schema Matching
# Match schema of lastfm and musicbrainz to discogs
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# lastfm -> discogs schema
schema_correspondences = matcher.match(discogs, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map)

# musicbrainz -> discogs schema
schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
musicbrainz = musicbrainz.rename(columns=rename_map)

# --------------------------------
# Blocking
# Use precomputed optimal blocking strategies
# --------------------------------

print("Performing Blocking")

ID_COLUMNS = {
    "discogs": "id",
    "lastfm": "id",
    "musicbrainz": "id",
}

# discogs_lastfm -> semantic_similarity on ["name", "artist"], top_k=10
blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    id_column=ID_COLUMNS["discogs"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

# discogs_musicbrainz -> sorted_neighbourhood on ["name"], window=15
blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    id_column=ID_COLUMNS["discogs"],
    window=15,
    batch_size=100000,
    output_dir="output/blocking-evaluation",
)

# musicbrainz_lastfm -> semantic_similarity on ["name", "artist", "duration"], top_k=10
blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    id_column=ID_COLUMNS["musicbrainz"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# Matching
# Use precomputed (updated) matching strategies
# --------------------------------

print("Matching Entities")

def lower_strip(x):
    return str(x).lower().strip() if pd.notnull(x) else x

# Threshold variables as required
threshold_discogs_lastfm = 0.6
threshold_discogs_musicbrainz = 0.7
threshold_musicbrainz_lastfm = 0.75

# discogs_lastfm comparators
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

# weights from final matching configuration
weights_discogs_lastfm = [0.2, 0.25, 0.35, 0.2]

# discogs_musicbrainz comparators
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

# weights from final matching configuration
weights_discogs_musicbrainz = [0.22, 0.22, 0.26, 0.15, 0.15]

# musicbrainz_lastfm comparators
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

# weights from final matching configuration
weights_musicbrainz_lastfm = [0.4, 0.4, 0.2]

matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=weights_discogs_lastfm,
    threshold=threshold_discogs_lastfm,
    id_column=ID_COLUMNS["discogs"],
)

rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=weights_discogs_musicbrainz,
    threshold=threshold_discogs_musicbrainz,
    id_column=ID_COLUMNS["discogs"],
)

rb_correspondences_musicbrainz_lastfm = matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_musicbrainz_lastfm,
    comparators=comparators_musicbrainz_lastfm,
    weights=weights_musicbrainz_lastfm,
    threshold=threshold_musicbrainz_lastfm,
    id_column=ID_COLUMNS["musicbrainz"],
)

# --------------------------------
# Save Correspondences
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_discogs_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_lastfm.csv",
    ),
    index=False,
)
rb_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_musicbrainz.csv",
    ),
    index=False,
)
rb_correspondences_musicbrainz_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_musicbrainz_lastfm.csv",
    ),
    index=False,
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

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("duration", maximum)
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

fused_data = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
os.makedirs("output/data_fusion", exist_ok=True)
fused_data.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 05 END
============================================================

