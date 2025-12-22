from PyDI.io import load_xml
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    prefer_higher_trust,
)
from PyDI.fusion import DataFusionEvaluator
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

# Load datasets (XML!)
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
# Schema Matching
# Schema of lastfm and musicbrainz will be mapped to discogs
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

# match discogs with lastfm
schema_correspondences = matcher.match(discogs, lastfm)
rename_map = (
    schema_correspondences.set_index("target_column")["source_column"].to_dict()
)
lastfm = lastfm.rename(columns=rename_map)

# match discogs with musicbrainz
schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = (
    schema_correspondences.set_index("target_column")["source_column"].to_dict()
)
musicbrainz = musicbrainz.rename(columns=rename_map)

# Rebuild datasets list after renaming
datasets = [discogs, lastfm, musicbrainz]

# --------------------------------
# Preprocessing / Normalization
# --------------------------------

for df in datasets:
    # normalize name and artist
    if "name" in df.columns:
        df["name_norm"] = (
            df["name"]
            .astype(str)
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    else:
        df["name_norm"] = ""

    if "artist" in df.columns:
        df["artist_norm"] = (
            df["artist"]
            .astype(str)
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    else:
        df["artist_norm"] = ""

    # normalize label, release-country, genre
    for col in ["label", "release-country", "genre"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

    # normalize release-date to string (no heavy parsing to avoid errors)
    if "release-date" in df.columns:
        df["release-date"] = (
            df["release-date"]
            .astype(str)
            .str.strip()
        )

    # duration and track durations to numeric where possible
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    for col in [
        "tracks_track_name",
        "tracks_track_position",
        "tracks_track_duration",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x if isinstance(x, (list, tuple)) else ([] if pd.isna(x) else [x])
            )

# --------------------------------
# Entity Blocking (using provided optimal strategies)
# --------------------------------

print("Performing Blocking")

# discogs - lastfm: exact_match_single on 'artist'
blocker_discogs_lastfm = StandardBlocker(
    discogs,
    lastfm,
    on=["artist"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# discogs - musicbrainz: exact_match_multi on ['artist', 'name']
blocker_discogs_musicbrainz = StandardBlocker(
    discogs,
    musicbrainz,
    on=["artist", "name"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# musicbrainz - lastfm: semantic_similarity on ['name'] -> use EmbeddingBlocker
embedding_blocker_mb_lfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=256,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Comparators
# Use similarity functions supported by PyDI (no token_sort_ratio).
# Emphasize jaro_winkler for title/artist; allow tolerant numeric duration.
# --------------------------------

comparators_discogs_lastfm = [
    # Title similarity
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    # Artist similarity
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    # Duration numeric similarity (allow some difference)
    NumericComparator(
        column="duration",
        max_difference=90,  # seconds tolerance
    ),
]

comparators_discogs_musicbrainz = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    NumericComparator(
        column="duration",
        max_difference=120,
    ),
    # Release date similarity (string-based)
    StringComparator(
        column="release-date",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).strip(),
    ),
    # Release country (string similarity to correct UK vs full name)
    StringComparator(
        column="release-country",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
    ),
]

comparators_musicbrainz_lastfm = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lambda x: str(x).lower().strip(),
    ),
    NumericComparator(
        column="duration",
        max_difference=120,
    ),
]

# --------------------------------
# Rule-Based Matching
# --------------------------------

print("Matching Entities")

matcher = RuleBasedMatcher()

# Discogs - LastFM
rb_correspondences_d_l = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    # Put higher weight on name and artist; keep threshold moderate to improve recall
    weights=[0.45, 0.45, 0.10],
    threshold=0.7,
    id_column="id",
)

# Discogs - MusicBrainz
rb_correspondences_d_m = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=[0.35, 0.35, 0.10, 0.10, 0.10],
    threshold=0.7,
    id_column="id",
)

# MusicBrainz - LastFM
rb_correspondences_m_l = matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=embedding_blocker_mb_lfm,
    comparators=comparators_musicbrainz_lastfm,
    weights=[0.45, 0.35, 0.20],
    threshold=0.7,
    id_column="id",
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_d_l, rb_correspondences_d_m, rb_correspondences_m_l],
    ignore_index=True,
)

# Assign trust scores: prefer Discogs and MusicBrainz over LastFM
trust_scores = {
    "discogs": 0.9,
    "musicbrainz": 0.85,
    "lastfm": 0.7,
}

strategy = DataFusionStrategy("music_release_fusion_strategy")

# Helper fuser to prefer higher trust, then longest string
def trusted_string_fuser(values, sources):
    # Delegate to prefer_higher_trust, then longest_string as fallback
    return prefer_higher_trust(values, sources, trust_scores, fallback=longest_string)

# Basic string attributes
for attr in ["name", "artist", "label", "release-country", "release-date", "genre"]:
    if attr in discogs.columns:
        strategy.add_attribute_fuser(attr, trusted_string_fuser)

# Duration: prefer higher trust then median as fallback
def duration_fuser(values, sources):
    numeric_vals = [v for v in values if pd.notna(v)]
    if not numeric_vals:
        return np.nan
    # first try trust-based resolution
    return prefer_higher_trust(numeric_vals, sources, trust_scores)

strategy.add_attribute_fuser("duration", duration_fuser)

# Track-level attributes:
# - names and durations: union to improve recall
# - positions: union as well (keeps all seen positions)
for attr in ["tracks_track_name", "tracks_track_position", "tracks_track_duration"]:
    if attr in discogs.columns:
        strategy.add_attribute_fuser(attr, union)

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
# Write output
# --------------------------------

rb_fused_standard_blocker.to_csv(
    "output/data_fusion/fusion_rb_standard_blocker.csv", index=False
)