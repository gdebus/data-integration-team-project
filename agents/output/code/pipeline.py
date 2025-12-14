from PyDI.io import load_xml

from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
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

# Load datasets (XML)
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
# Schema Matching (align schemas of lastfm & musicbrainz to discogs)
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
    system_prompt=(
        "You match music release schemas. Map semantically identical attributes:\n"
        "- 'id' is the record identifier.\n"
        "- 'name' is the release / album / single title.\n"
        "- 'artist' is the main artist.\n"
        "- 'duration' is total duration in seconds when it is a positive integer. "
        "If there are multiple duration-like fields, map the total release duration "
        "to 'duration'.\n"
        "- 'release-date' is the release date.\n"
        "- 'release-country' is the country of release (prefer ISO-like or short names).\n"
        "- Track-level columns: 'tracks_track_name', 'tracks_track_position', "
        "'tracks_track_duration' represent, respectively, per-track title, index, "
        "and duration in seconds. Map any equivalent track fields to these.\n"
        "Do not invent new column names. Use Discogs column names as target."
    ),
)

# match schema of discogs with lastfm and rename lastfm
schema_correspondences = matcher.match(discogs, lastfm)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
lastfm = lastfm.rename(columns=rename_map)

# match schema of discogs with musicbrainz and rename musicbrainz
schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
musicbrainz = musicbrainz.rename(columns=rename_map)

# --------------------------------
# Preprocessing before matching
# --------------------------------

def to_numeric_duration(series):
    s = pd.to_numeric(series, errors="coerce")
    # Treat 0 and negative values as missing (placeholder / invalid)
    s = s.mask(s <= 0)
    return s

def normalize_text(col: pd.Series) -> pd.Series:
    return (
        col.astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

for df in [discogs, lastfm, musicbrainz]:
    if "duration" in df.columns:
        df["duration"] = to_numeric_duration(df["duration"])

    # Normalize artist / name for better blocking & similarity
    if "artist" in df.columns:
        df["artist_norm"] = normalize_text(df["artist"])
    if "name" in df.columns:
        df["name_norm"] = normalize_text(df["name"])

    # Normalize track names for better fusion & evaluation
    if "tracks_track_name" in df.columns:
        df["tracks_track_name_norm"] = df["tracks_track_name"].apply(
            lambda x: [t.strip().lower() for t in x] if isinstance(x, (list, tuple)) else x
        )

# --------------------------------
# Entity Matching / Blocking
# --------------------------------

print("Performing Blocking")

# discogs has > 20k rows, so use StandardBlocker per instructions
# Block on artist first, then on name; intersect candidate sets implicitly by
# requiring good similarity on both in the rule-based matcher.

blocker_discogs_lastfm_artist = StandardBlocker(
    discogs,
    lastfm,
    on=["artist_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_discogs_musicbrainz_artist = StandardBlocker(
    discogs,
    musicbrainz,
    on=["artist_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_discogs_lastfm_name = StandardBlocker(
    discogs,
    lastfm,
    on=["name_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_discogs_musicbrainz_name = StandardBlocker(
    discogs,
    musicbrainz,
    on=["name_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Define Comparators (only columns present in all datasets after schema matching)
# --------------------------------

comparators = [
    # Title similarity (releases often differ only slightly in formatting)
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
        preprocess=None,
    ),

    # Artist similarity (highly reliable)
    StringComparator(
        column="artist_norm",
        similarity_function="jaccard",
        preprocess=None,
    ),

    # Duration similarity with a moderate tolerance
    NumericComparator(
        column="duration",
        max_difference=45,  # slightly stricter than before
    ),
]

print("Matching Entities")

matcher_rb = RuleBasedMatcher()

# Use the stricter blocking on both artist and name:
rb_correspondences_discogs_lastfm = matcher_rb.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm_artist,
    comparators=comparators,
    # stronger weight on exact text fields, slightly reduced duration effect
    weights=[0.50, 0.45, 0.05],
    threshold=0.86,
    id_column="id",
)

rb_correspondences_discogs_musicbrainz = matcher_rb.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz_artist,
    comparators=comparators,
    weights=[0.50, 0.45, 0.05],
    threshold=0.86,
    id_column="id",
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_discogs_lastfm, rb_correspondences_discogs_musicbrainz],
    ignore_index=True,
)

strategy = DataFusionStrategy("music_release_fusion_strategy")

# Core attributes
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)

# Duration: prefer the longest non-null value (usually most complete)
strategy.add_attribute_fuser("duration", longest_string)

# Release info
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)

# Additional metadata
strategy.add_attribute_fuser("label", longest_string)
strategy.add_attribute_fuser("genre", union)

# Track-level attributes: keep union across sources
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
# Optional: Evaluate fusion quality if gold standard is available
# (expects a CSV with a 'id' column matching fused ids)
# --------------------------------

gold_path = "input/gold/music_gold.csv"
if os.path.exists(gold_path):
    gold_df = pd.read_csv(gold_path)

    evaluator = DataFusionEvaluator(
        fused_dataset=rb_fused_standard_blocker,
        gold_dataset=gold_df,
        id_column="id",
        match_function=tokenized_match,
    )

    eval_results = evaluator.evaluate()
    eval_results.to_json(
        "output/data_fusion/fusion_rb_standard_blocker_evaluation.json",
        orient="records",
        lines=True,
    )

# --------------------------------
# Write Output
# --------------------------------

rb_fused_standard_blocker.to_csv(
    "output/data_fusion/fusion_rb_standard_blocker.csv",
    index=False,
)