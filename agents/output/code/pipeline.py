from PyDI.io import load_xml
from PyDI.schemamatching import LLMBasedSchemaMatcher
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
)
from langchain_openai import ChatOpenAI

import pandas as pd
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
# Light preprocessing BEFORE schema matching
# --------------------------------

def _normalize_whitespace(s):
    if not isinstance(s, str):
        return s
    return re.sub(r"\s+", " ", s).strip()

# Normalize key textual columns to help LLM schema matching & comparators
for df in [discogs, lastfm, musicbrainz]:
    for col in ["name", "artist", "release-country"]:
        if col in df.columns:
            df[col] = df[col].astype(str).map(_normalize_whitespace)

# --------------------------------
# Schema Matching (align to Discogs schema)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=20,   # give matcher more context
    debug=True,
)

# Match discogs ↔ lastfm
schema_correspondences = matcher.match(discogs, lastfm)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
lastfm = lastfm.rename(columns=rename_map)

# Match discogs ↔ musicbrainz
schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
musicbrainz = musicbrainz.rename(columns=rename_map)

# --------------------------------
# Entity Matching – Blocking (using provided optimal strategies)
# --------------------------------

print("Performing Blocking")

discogs_id_col = "id"
lastfm_id_col = "id"
musicbrainz_id_col = "id"

# discogs ↔ lastfm: semantic similarity on ["name", "artist"]
embedding_blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=256,
    output_dir="output/blocking-evaluation",
    id_column=discogs_id_col,
)

# discogs ↔ musicbrainz: exact match on ["artist", "tracks_track_name"]
blocker_discogs_musicbrainz = StandardBlocker(
    discogs,
    musicbrainz,
    on=["artist", "tracks_track_name"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=discogs_id_col,
)

# musicbrainz ↔ lastfm: semantic similarity on ["name", "artist"]
embedding_blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=256,
    output_dir="output/blocking-evaluation",
    id_column=musicbrainz_id_col,
)

# --------------------------------
# Entity Matching – Comparators & Rule-Based Matcher
# --------------------------------

print("Matching Entities")

def _lower_strip(x):
    return x.lower().strip() if isinstance(x, str) else ""

def _strip(x):
    return x.strip() if isinstance(x, str) else ""

def _normalize_country(x):
    if not isinstance(x, str):
        return ""
    x = x.lower().strip()
    mappings = {
        "united kingdom of great britain and northern ireland": "uk",
        "united kingdom": "uk",
        "u.k.": "uk",
        "england": "uk",
    }
    return mappings.get(x, x)

def _normalize_date(x):
    if not isinstance(x, str):
        return ""
    x = x.strip()
    # keep only YYYY-MM-DD or YYYY
    m = re.match(r"(\d{4})(-\d{2})?(-\d{2})?", x)
    return m.group(0) if m else x

# Core comparators with tuned preprocessing:
comparators_core = [
    # Name similarity – normalize spaces and case
    StringComparator(
        column="name",
        similarity_function="cosine",  # more tolerant than jaccard for long titles
        preprocess=_lower_strip,
    ),
    # Artist similarity
    StringComparator(
        column="artist",
        similarity_function="jaccard",
        preprocess=_lower_strip,
    ),
    # Track list similarity (list → concatenated string)
    StringComparator(
        column="tracks_track_name",
        similarity_function="jaccard",
        preprocess=_lower_strip,
        list_strategy="concatenate",
    ),
    # Track positions similarity
    StringComparator(
        column="tracks_track_position",
        similarity_function="jaccard",
        preprocess=_strip,
        list_strategy="concatenate",
    ),
    # Duration similarity (numeric tolerance)
    NumericComparator(
        column="duration",
        max_difference=60,  # tighten slightly to reduce wrong merges
    ),
]

# Extended comparators for sources with release info
comparators_with_release = comparators_core + [
    StringComparator(
        column="release-date",
        similarity_function="jaccard",
        preprocess=_normalize_date,
    ),
    StringComparator(
        column="release-country",
        similarity_function="jaccard",
        preprocess=_normalize_country,
    ),
]

matcher = RuleBasedMatcher()

# discogs ↔ lastfm
rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=embedding_blocker_discogs_lastfm,
    comparators=comparators_core,
    # weights: name, artist, tracks_name, tracks_pos, duration
    # put more emphasis on tracks_* to improve track-related accuracy
    weights=[0.25, 0.2, 0.25, 0.2, 0.1],
    threshold=0.78,  # slightly higher to reduce false positives
    id_column=discogs_id_col,
)

# discogs ↔ musicbrainz
rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_with_release,
    # weights: name, artist, tracks_name, tracks_pos, duration, rel-date, rel-country
    weights=[0.2, 0.15, 0.25, 0.15, 0.05, 0.1, 0.1],
    threshold=0.8,
    id_column=discogs_id_col,
)

# musicbrainz ↔ lastfm
rb_correspondences_musicbrainz_lastfm = matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=embedding_blocker_musicbrainz_lastfm,
    comparators=comparators_core,
    weights=[0.25, 0.2, 0.25, 0.2, 0.1],
    threshold=0.78,
    id_column=musicbrainz_id_col,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_musicbrainz_lastfm,
    ],
    ignore_index=True,
)

def prefer_discogs_then_musicbrainz(values_with_provenance):
    priority = ["discogs", "musicbrainz"]
    for source in priority:
        for v, prov in values_with_provenance:
            if prov == source and v not in (None, "", []):
                return v
    return longest_string([v for v, _ in values_with_provenance])

def prefer_musicbrainz_then_discogs(values_with_provenance):
    priority = ["musicbrainz", "discogs"]
    for source in priority:
        for v, prov in values_with_provenance:
            if prov == source and v not in (None, "", []):
                return v
    return longest_string([v for v, _ in values_with_provenance])

def average_duration(values_with_provenance):
    nums = []
    for v, _ in values_with_provenance:
        if v in (None, "", []):
            continue
        try:
            num = float(v)
            if num > 0:
                nums.append(num)
        except Exception:
            continue
    if not nums:
        return longest_string([v for v, _ in values_with_provenance])
    return str(int(round(sum(nums) / len(nums))))

def prefer_discogs_tracks_then_union(values_with_provenance):
    # try Discogs first as it often has canonical track listing / positions
    ordered_sources = ["discogs", "musicbrainz", "lastfm"]
    for src in ordered_sources:
        vals = [v for v, prov in values_with_provenance if prov == src and v not in (None, "", [])]
        if vals:
            return vals[0]
    # fall back to union
    raw_vals = [v for v, _ in values_with_provenance]
    return union(raw_vals)

strategy = DataFusionStrategy("music_release_fusion_strategy")

# Core fields
strategy.add_attribute_fuser("name", prefer_discogs_then_musicbrainz)
strategy.add_attribute_fuser("artist", prefer_discogs_then_musicbrainz)
strategy.add_attribute_fuser("release-date", prefer_musicbrainz_then_discogs)
strategy.add_attribute_fuser("release-country", prefer_musicbrainz_then_discogs)
strategy.add_attribute_fuser("label", prefer_discogs_then_musicbrainz)
strategy.add_attribute_fuser("genre", prefer_discogs_then_musicbrainz)
strategy.add_attribute_fuser("duration", average_duration)

# Track-related fields:
# Prefer trusted structured sources for sequences, not a blind union.
strategy.add_attribute_fuser("tracks_track_name", prefer_discogs_tracks_then_union)
strategy.add_attribute_fuser("tracks_track_position", prefer_discogs_then_musicbrainz)
strategy.add_attribute_fuser("tracks_track_duration", prefer_musicbrainz_then_discogs)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_rb_standard_blocker.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_correspondences,
    id_column="id",
    include_singletons=False,
)

# --------------------------------
# Write output
# --------------------------------

rb_fused_standard_blocker.to_csv(
    "output/data_fusion/fusion_rb_standard_blocker.csv",
    index=False,
)