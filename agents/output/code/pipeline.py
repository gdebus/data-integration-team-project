from PyDI.io import load_xml
from PyDI.entitymatching import EmbeddingBlocker, StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    most_recent,
    prefer_higher_trust,
)
from PyDI.fusion import DataFusionEvaluator, tokenized_match
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import re

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
# Match lastfm & musicbrainz to discogs schema
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=25,
    debug=True,
)

# Match discogs (source) -> lastfm (target), rename lastfm to discogs schema
schema_correspondences = matcher.match(discogs, lastfm)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
lastfm = lastfm.rename(columns=rename_map)

# Match discogs (source) -> musicbrainz (target), rename musicbrainz to discogs schema
schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
musicbrainz = musicbrainz.rename(columns=rename_map)

# Ensure critical columns exist across datasets
for df in [discogs, lastfm, musicbrainz]:
    for col in [
        "id",
        "name",
        "artist",
        "release-date",
        "release-country",
        "duration",
        "label",
        "genre",
        "tracks_track_name",
        "tracks_track_position",
        "tracks_track_duration",
    ]:
        if col not in df.columns:
            df[col] = np.nan

# --------------------------------
# Preprocessing for better matching & fusion
# --------------------------------

def normalize_text(s):
    if isinstance(s, (list, tuple, np.ndarray)):
        # Join lists (e.g. track names) into a single string
        s = " | ".join(map(str, s))
    if pd.isna(s):
        return s
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_country(c):
    if pd.isna(c):
        return c
    c = str(c).strip()
    mapping = {
        "uk": "United Kingdom of Great Britain and Northern Ireland",
        "u.k.": "United Kingdom of Great Britain and Northern Ireland",
        "united kingdom": "United Kingdom of Great Britain and Northern Ireland",
        "england": "United Kingdom of Great Britain and Northern Ireland",
    }
    key = c.lower()
    return mapping.get(key, c)

# Helper to safely convert durations to numeric seconds
def to_seconds(x):
    # Handle list-like durations (e.g. per-track) by ignoring them for release-level duration
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.nan
    if pd.isna(x):
        return np.nan
    try:
        return float(x)
    except Exception:
        # Try to parse formats like "mm:ss"
        s = str(x)
        if re.match(r"^\d+:\d{2}$", s):
            m, s_sec = s.split(":")
            try:
                return int(m) * 60 + int(s_sec)
            except Exception:
                return np.nan
        return np.nan

for df in [discogs, lastfm, musicbrainz]:
    # Normalize title and artist
    df["name_norm"] = (
        df["name"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )
    df["artist_norm"] = df["artist"].astype(str).str.strip().str.lower()

    # Create a "clean" name that drops leading artist prefixes like "Artist - "
    df["name_clean"] = df["name"].astype(str)

    def strip_artist_prefix(row):
        name = str(row["name"])
        artist = str(row["artist"])
        name_l = name.lower()
        artist_l = artist.lower()
        # patterns like "artist - title" or "artist: title"
        if name_l.startswith(artist_l + " - "):
            return name[len(artist) + 3 :].strip()
        if name_l.startswith(artist_l + " : "):
            return name[len(artist) + 3 :].strip()
        if name_l.startswith(artist_l + " – "):  # en dash
            return name[len(artist) + 3 :].strip()
        return name

    df["name_clean"] = df.apply(strip_artist_prefix, axis=1)
    df["name_clean_norm"] = df["name_clean"].astype(str).str.lower().str.replace(
        r"\s+", " ", regex=True
    ).str.strip()

    # Track concatenation for similarity; keep ordering stable
    def concat_tracks(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return " | ".join(map(str, x))
        if pd.isna(x):
            return ""
        return str(x)

    df["tracks_concat"] = df["tracks_track_name"].apply(concat_tracks).str.lower()

    # Numeric duration in seconds (release-level)
    df["duration_sec"] = df["duration"].apply(to_seconds)

    # Normalize country text for better fusion consistency
    df["release-country-norm"] = df["release-country"].apply(normalize_country)

# --------------------------------
# Blocking (use provided configuration)
# --------------------------------

print("Performing Blocking")

id_columns = {
    "discogs": "id",
    "lastfm": "id",
    "musicbrainz": "id",
}

blocking_strategies = {
    "discogs_lastfm": {
        "strategy": "semantic_similarity",
        "columns": ["name", "artist"],
        "top_k": 20,
        "threshold": 0.3,
    },
    "discogs_musicbrainz": {
        "strategy": "semantic_similarity",
        "columns": ["name", "artist"],
        "top_k": 20,
        "threshold": 0.3,
    },
    "musicbrainz_lastfm": {
        "strategy": "exact_match_multi",
        "columns": ["artist", "duration"],
        "top_k": 20,
        "threshold": 0.3,
    },
}

# For 20k+ rows, prefer StandardBlocker as per instructions
use_standard_for_discogs = len(discogs) > 20000 or len(lastfm) > 20000
use_standard_for_discogs_mb = len(discogs) > 20000 or len(musicbrainz) > 20000

if use_standard_for_discogs:
    embedding_blocker_discogs_lastfm = StandardBlocker(
        discogs,
        lastfm,
        on=blocking_strategies["discogs_lastfm"]["columns"],
        batch_size=1000,
        output_dir="output/blocking-evaluation",
        id_column=id_columns["discogs"],
    )
else:
    embedding_blocker_discogs_lastfm = EmbeddingBlocker(
        discogs,
        lastfm,
        text_cols=blocking_strategies["discogs_lastfm"]["columns"],
        model="sentence-transformers/all-MiniLM-L6-v2",
        index_backend="sklearn",
        top_k=blocking_strategies["discogs_lastfm"]["top_k"],
        batch_size=500,
        output_dir="output/blocking-evaluation",
        id_column=id_columns["discogs"],
    )

if use_standard_for_discogs_mb:
    embedding_blocker_discogs_musicbrainz = StandardBlocker(
        discogs,
        musicbrainz,
        on=blocking_strategies["discogs_musicbrainz"]["columns"],
        batch_size=1000,
        output_dir="output/blocking-evaluation",
        id_column=id_columns["discogs"],
    )
else:
    embedding_blocker_discogs_musicbrainz = EmbeddingBlocker(
        discogs,
        musicbrainz,
        text_cols=blocking_strategies["discogs_musicbrainz"]["columns"],
        model="sentence-transformers/all-MiniLM-L6-v2",
        index_backend="sklearn",
        top_k=blocking_strategies["discogs_musicbrainz"]["top_k"],
        batch_size=500,
        output_dir="output/blocking-evaluation",
        id_column=id_columns["discogs"],
    )

# StandardBlocker for exact_match_multi (artist + duration)
standard_blocker_mbrainz_lastfm = StandardBlocker(
    musicbrainz,
    lastfm,
    on=blocking_strategies["musicbrainz_lastfm"]["columns"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_columns["musicbrainz"],
)

# --------------------------------
# Comparators
# --------------------------------

# Emphasize name/artist, use tracks more strongly to improve track-related accuracy

comparators_discogs_lastfm = [
    StringComparator(
        column="name_clean_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="artist_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="tracks_concat",
        similarity_function="cosine",
    ),
    NumericComparator(
        column="duration_sec",
        max_difference=60,  # allow more noise across sources
    ),
]

comparators_discogs_musicbrainz = [
    StringComparator(
        column="name_clean_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="artist_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="tracks_concat",
        similarity_function="cosine",
    ),
    NumericComparator(
        column="duration_sec",
        max_difference=60,
    ),
]

comparators_mbrainz_lastfm = [
    StringComparator(
        column="name_clean_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="artist_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="tracks_concat",
        similarity_function="cosine",
    ),
    NumericComparator(
        column="duration_sec",
        max_difference=60,
    ),
]

# --------------------------------
# Entity Matching
# --------------------------------

print("Matching Entities")

matcher = RuleBasedMatcher()

# Raise weight of tracks to help track-related attributes; keep reasonable threshold
weights = [0.4, 0.35, 0.2, 0.05]
threshold = 0.78

# Discogs - LastFM
rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=embedding_blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=weights,
    threshold=threshold,
    id_column=id_columns["discogs"],
)

# Discogs - MusicBrainz
rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=embedding_blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=weights,
    threshold=threshold,
    id_column=id_columns["discogs"],
)

# MusicBrainz - LastFM
rb_correspondences_mbrainz_lastfm = matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=standard_blocker_mbrainz_lastfm,
    comparators=comparators_mbrainz_lastfm,
    weights=weights,
    threshold=threshold,
    id_column=id_columns["musicbrainz"],
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

# Merge correspondences
all_rb_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_mbrainz_lastfm,
    ],
    ignore_index=True,
)

# Source trust scores: musicbrainz > discogs > lastfm
source_trust = {
    "musicbrainz": 0.9,
    "discogs": 0.8,
    "lastfm": 0.6,
}

def prefer_musicbrainz_then_discogs(values, provenance):
    return prefer_higher_trust(values, provenance, trust_scores=source_trust)

strategy = DataFusionStrategy("music_release_fusion_strategy")

# Core attributes
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", prefer_musicbrainz_then_discogs)
strategy.add_attribute_fuser("release-date", most_recent)

# Use normalized country to reduce variation; then output canonical country field
def fuse_release_country(values, provenance):
    fused = prefer_higher_trust(values, provenance, trust_scores=source_trust)
    return normalize_country(fused)

strategy.add_attribute_fuser("release-country", fuse_release_country)

# For duration, prefer higher-trust source instead of averaging
def fuse_duration(values, provenance):
    chosen = prefer_higher_trust(values, provenance, trust_scores=source_trust)
    return chosen

strategy.add_attribute_fuser("duration", fuse_duration)

strategy.add_attribute_fuser("label", prefer_musicbrainz_then_discogs)
strategy.add_attribute_fuser("genre", union)

# Track-level attributes:
# Union but evaluation will use tokenized_match for robustness.
strategy.add_attribute_fuser("tracks_track_name", union)
strategy.add_attribute_fuser("tracks_track_position", union)
strategy.add_attribute_fuser("tracks_track_duration", union)

# Run fusion
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

# Write output (file name must remain unchanged)
rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_rb_standard_blocker.csv", index=False)