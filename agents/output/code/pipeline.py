from PyDI.io import load_csv

from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
    TokenBlocker,
    RuleBasedMatcher,
    StringComparator,
    NumericComparator,
    DateComparator,
)

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"

# Load API key
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
# Match lastfm and musicbrainz schema to discogs schema
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

# lastfm -> discogs schema
schema_correspondences = matcher.match(discogs, lastfm)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
lastfm = lastfm.rename(columns=rename_map)

# musicbrainz -> discogs schema
schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
musicbrainz = musicbrainz.rename(columns=rename_map)

# --------------------------------
# Blocking (use provided blocking configuration)
# --------------------------------

print("Performing Blocking")

# id columns from config
discogs_id_col = "id"
lastfm_id_col = "id"
musicbrainz_id_col = "id"

# discogs - lastfm: semantic_similarity on [artist, name, tracks_track_name], top_k=20
blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["artist", "name", "tracks_track_name"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=discogs_id_col,
)

# discogs - musicbrainz: semantic_similarity on [name, artist, release-date], top_k=20
blocker_discogs_musicbrainz = EmbeddingBlocker(
    discogs,
    musicbrainz,
    text_cols=["name", "artist", "release-date"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=discogs_id_col,
)

# musicbrainz - lastfm: token_blocking on [name], min_token_len=4
blocker_musicbrainz_lastfm = TokenBlocker(
    musicbrainz,
    lastfm,
    column="name",
    min_token_len=4,
    ngram_size=1,
    ngram_type="word",
    id_column=musicbrainz_id_col,
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# Matching (use provided matching configuration)
# --------------------------------

print("Matching Entities")

def lower_strip(x):
    return str(x).lower().strip() if x is not None else ""

# discogs - lastfm comparators
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

weights_discogs_lastfm = [0.3, 0.3, 0.3, 0.1]
threshold_discogs_lastfm = 0.72

# discogs - musicbrainz comparators
comparators_discogs_musicbrainz = [
    StringComparator(
        column="name",
        similarity_function="cosine",
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

weights_discogs_musicbrainz = [0.3, 0.25, 0.2, 0.15, 0.1]
threshold_discogs_musicbrainz = 0.75

# musicbrainz - lastfm comparators
comparators_musicbrainz_lastfm = [
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
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

weights_musicbrainz_lastfm = [0.4, 0.25, 0.25, 0.1]
threshold_musicbrainz_lastfm = 0.75

# Initialize Rule-Based Matcher
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

# Core attributes
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("duration", longest_string)
strategy.add_attribute_fuser("label", longest_string)
strategy.add_attribute_fuser("genre", longest_string)

# Track-level attributes as list-like
strategy.add_attribute_fuser("tracks_track_name", union)
strategy.add_attribute_fuser("tracks_track_position", union)
strategy.add_attribute_fuser("tracks_track_duration", union)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_rb_standard_blocker.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# --------------------------------
# Write Output
# --------------------------------

rb_fused_standard_blocker.to_csv(
    "output/data_fusion/fusion_rb_standard_blocker.csv",
    index=False,
)