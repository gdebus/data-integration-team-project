from PyDI.io import load_parquet, load_csv, load_xml

from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
)
from PyDI.fusion import DataFusionEvaluator, tokenized_match

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
discogs = load_xml(
    DATA_DIR + "discogs.xml",
    name="discogs",
)

lastfm = load_xml(
    DATA_DIR + "lastfm.xml",
    name="lastfm",
)

musicbrainz = load_xml(
    DATA_DIR + "musicbrainz.xml",
    name="musicbrainz",
)

datasets = [discogs, lastfm, musicbrainz]

# --------------------------------
# Schema Matching (map lastfm & musicbrainz to discogs schema)
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

# discogs <-> lastfm
schema_correspondences = matcher.match(discogs, lastfm)
rename_map = (
    schema_correspondences.set_index("target_column")["source_column"].to_dict()
)
lastfm = lastfm.rename(columns=rename_map)

# discogs <-> musicbrainz
schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = (
    schema_correspondences.set_index("target_column")["source_column"].to_dict()
)
musicbrainz = musicbrainz.rename(columns=rename_map)

# Ensure required columns exist in all datasets
required_cols = [
    "id",
    "name",
    "artist",
    "duration",
    "release-date",
    "release-country",
    "label",
    "genre",
    "tracks_track_name",
    "tracks_track_duration",
    "tracks_track_position",
]

for col in required_cols:
    if col not in discogs.columns:
        discogs[col] = np.nan
    if col not in lastfm.columns:
        lastfm[col] = np.nan
    if col not in musicbrainz.columns:
        musicbrainz[col] = np.nan

# --------------------------------
# Preprocessing helpers
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip() if pd.notnull(x) else ""

# Normalize release-country strings to improve fusion consistency
def normalize_country(col):
    mapping = {
        "united kingdom of great britain and northern ireland": "UK",
        "united kingdom": "UK",
        "england": "UK",
    }
    def _norm(v):
        if pd.isna(v):
            return v
        s = str(v).lower().strip()
        return mapping.get(s, v)
    return col.apply(_norm)

discogs["release-country"] = normalize_country(discogs["release-country"])
musicbrainz["release-country"] = normalize_country(musicbrainz["release-country"])
if "release-country" in lastfm.columns:
    lastfm["release-country"] = normalize_country(lastfm["release-country"])

# Coerce durations to numeric (seconds) where possible
for df in [discogs, lastfm, musicbrainz]:
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

# --------------------------------
# Blocking (use semantic_similarity config: name + artist, top_k=15)
# --------------------------------

print("Performing Blocking")

embedding_blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=500,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

embedding_blocker_discogs_musicbrainz = EmbeddingBlocker(
    discogs,
    musicbrainz,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=500,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

embedding_blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=500,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Comparators (from matching configuration)
# --------------------------------

print("Matching Entities")

# discogs_lastfm
comparators_discogs_lastfm = [
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
]
weights_discogs_lastfm = [0.45, 0.3, 0.25]
threshold_discogs_lastfm = 0.75

# discogs_musicbrainz
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
    StringComparator(
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    DateComparator(
        column="release-date",
        max_days_difference=365,
    ),
    NumericComparator(
        column="duration",
        max_difference=30.0,
    ),
]
weights_discogs_musicbrainz = [0.3, 0.25, 0.25, 0.15, 0.05]
threshold_discogs_musicbrainz = 0.75

# musicbrainz_lastfm
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
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=120.0,
    ),
]
weights_musicbrainz_lastfm = [0.25, 0.35, 0.25, 0.15]
threshold_musicbrainz_lastfm = 0.68

# --------------------------------
# Rule-Based Matching
# --------------------------------

rb_matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = rb_matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=embedding_blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=weights_discogs_lastfm,
    threshold=threshold_discogs_lastfm,
    id_column="id",
)

rb_correspondences_discogs_musicbrainz = rb_matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=embedding_blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=weights_discogs_musicbrainz,
    threshold=threshold_discogs_musicbrainz,
    id_column="id",
)

rb_correspondences_musicbrainz_lastfm = rb_matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=embedding_blocker_musicbrainz_lastfm,
    comparators=comparators_musicbrainz_lastfm,
    weights=weights_musicbrainz_lastfm,
    threshold=threshold_musicbrainz_lastfm,
    id_column="id",
)

print("Fusing Data")

# --------------------------------
# Data Fusion
# --------------------------------

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_musicbrainz_lastfm,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("music_release_fusion_strategy")

# Favor "canonical" discogs/musicbrainz style values by using longest_string,
# which tends to keep more complete titles and artist names
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)

# Use longest_string for release-date and release-country after normalization
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)

# Durations were previously fused with union which likely produced
# incorrect list-like values; use longest_string to keep a single,
# most complete duration representation
strategy.add_attribute_fuser("duration", longest_string)

strategy.add_attribute_fuser("label", longest_string)
strategy.add_attribute_fuser("genre", union)

# Tracks: use longest_string instead of union so each attribute
# is a clean, single value rather than a noisy aggregate
strategy.add_attribute_fuser("tracks_track_name", longest_string)
strategy.add_attribute_fuser("tracks_track_position", longest_string)
strategy.add_attribute_fuser("tracks_track_duration", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_rb_music.xml.jsonl",
)

rb_fused_music = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# --------------------------------
# Write output
# --------------------------------

rb_fused_music.to_csv("output/data_fusion/fusion_rb_standard_blocker.csv", index=False)