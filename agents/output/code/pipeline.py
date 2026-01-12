# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_xml

from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import DataFusionStrategy, DataFusionEngine
from PyDI.fusion import longest_string, union

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
import ast
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

discogs = load_xml(DATA_DIR + "discogs.xml", name="discogs")
lastfm = load_xml(DATA_DIR + "lastfm.xml", name="lastfm")
musicbrainz = load_xml(DATA_DIR + "musicbrainz.xml", name="musicbrainz")

# --------------------------------
# Schema Matching (discogs as canonical schema)
# IMPORTANT: Ensure ID column stays intact after renaming.
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

schema_correspondences = schema_matcher.match(discogs, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
# Protect ID column from accidental remap
rename_map.pop("id", None)
lastfm = lastfm.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(discogs, musicbrainz)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
rename_map.pop("id", None)
musicbrainz = musicbrainz.rename(columns=rename_map)

# --------------------------------
# ID integrity checks (addresses "fused_id: NaN" / missing fused entities)
# --------------------------------

def assert_id_ok(df, df_name, id_col="id"):
    assert id_col in df.columns, f"[{df_name}] missing id column after schema matching"
    assert df[id_col].notna().all(), f"[{df_name}] id contains nulls"
    # Not all datasets guarantee uniqueness, but empties are always bad
    assert df[id_col].astype(str).str.strip().ne("").all(), f"[{df_name}] id contains empty strings"

assert_id_ok(discogs, "discogs")
assert_id_ok(lastfm, "lastfm")
assert_id_ok(musicbrainz, "musicbrainz")

# --------------------------------
# Normalization / Type Fixes
# - ensure track columns are real Python lists (not stringified lists)
# - normalize durations to numeric seconds (float) to improve numeric matching + fusion
# - keep track positions as strings (often compared/fused as categorical)
# --------------------------------

def _is_nan(v):
    return v is None or (isinstance(v, float) and np.isnan(v))

def _to_list_maybe(v):
    if _is_nan(v):
        return np.nan
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return np.nan
        # Parse Python-literal list representations
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                pass
        return [s]
    return [str(v)]

def _normalize_list_column(df, col):
    if col in df.columns:
        df[col] = df[col].apply(_to_list_maybe)

def _to_float_or_nan(v):
    if _is_nan(v):
        return np.nan
    s = str(v).strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def _normalize_duration_seconds(v):
    """
    Heuristic to harmonize duration units.
    Commonly: seconds (<= ~20k) vs milliseconds (>= ~20k).
    """
    x = _to_float_or_nan(v)
    if _is_nan(x):
        return np.nan
    if x >= 20000:  # likely milliseconds
        return x / 1000.0
    return x

def normalize_dataset(df):
    for c in ["tracks_track_name", "tracks_track_position", "tracks_track_duration"]:
        _normalize_list_column(df, c)

    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(_normalize_duration_seconds)

    return df

discogs = normalize_dataset(discogs)
lastfm = normalize_dataset(lastfm)
musicbrainz = normalize_dataset(musicbrainz)

# --------------------------------
# Entity Matching
# Use pre-computed optimal blocking strategy: semantic_similarity on ['artist','name'], top_k=10.
# In PyDI, the closest implementation available here is EmbeddingBlocker on the same columns and top_k.
# --------------------------------

print("Performing Blocking")

# IMPORTANT: id_column must match df_left for each blocker to avoid corrupt pair IDs.
blocker_discogs_lastfm = EmbeddingBlocker(
    discogs, lastfm,
    text_cols=["artist", "name"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=500,
    output_dir="output/blocking-evaluation",
    id_column="id"
)

blocker_discogs_musicbrainz = EmbeddingBlocker(
    discogs, musicbrainz,
    text_cols=["artist", "name"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=500,
    output_dir="output/blocking-evaluation",
    id_column="id"
)

blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz, lastfm,
    text_cols=["artist", "name"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=500,
    output_dir="output/blocking-evaluation",
    id_column="id"
)

# --------------------------------
# Comparators + Rule-Based Matching (use provided matching configuration)
# Slightly relax ONLY musicbrainz_lastfm threshold to reduce over-fragmentation
# (cluster report showed almost exclusively size-2 clusters; low risk of runaway FP).
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()

print("Matching Entities")

rbm = RuleBasedMatcher()

# discogs_lastfm (config)
comparators_discogs_lastfm = [
    StringComparator(column="artist", similarity_function="jaro_winkler", preprocess=lower_strip),
    StringComparator(column="name", similarity_function="cosine", preprocess=lower_strip),
    StringComparator(
        column="tracks_track_name",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard"
    ),
    NumericComparator(column="duration", max_difference=30.0),
]

correspondences_discogs_lastfm = rbm.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=[0.3, 0.35, 0.25, 0.1],
    threshold=0.6,
    id_column="id"
)

# discogs_musicbrainz (config)
comparators_discogs_musicbrainz = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate"
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate"
    ),
    StringComparator(
        column="tracks_track_name",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard"
    ),
    DateComparator(column="release-date", max_days_difference=365),
    NumericComparator(column="duration", max_difference=60.0),
]

correspondences_discogs_musicbrainz = rbm.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=[0.25, 0.3, 0.25, 0.1, 0.1],
    threshold=0.78,
    id_column="id"
)

# musicbrainz_lastfm (config; slightly relaxed threshold to reduce under-linking)
comparators_musicbrainz_lastfm = [
    StringComparator(column="artist", similarity_function="jaro_winkler", preprocess=lower_strip),
    StringComparator(column="name", similarity_function="cosine", preprocess=lower_strip),
    StringComparator(
        column="tracks_track_name",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard"
    ),
    NumericComparator(column="duration", max_difference=300.0),
]

correspondences_musicbrainz_lastfm = rbm.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_musicbrainz_lastfm,
    comparators=comparators_musicbrainz_lastfm,
    weights=[0.25, 0.35, 0.3, 0.1],
    threshold=0.68,  # was 0.72; slight relaxation to encourage transitive linking
    id_column="id"
)

# --------------------------------
# Save correspondences for each pair (CRITICAL: keep filenames)
# --------------------------------

if not os.path.exists("output/correspondences"):
    os.makedirs("output/correspondences", exist_ok=True)

correspondences_discogs_lastfm.to_csv(
    "output/correspondences/correspondences_discogs_lastfm.csv",
    index=False
)
correspondences_discogs_musicbrainz.to_csv(
    "output/correspondences/correspondences_discogs_musicbrainz.csv",
    index=False
)
correspondences_musicbrainz_lastfm.to_csv(
    "output/correspondences/correspondences_musicbrainz_lastfm.csv",
    index=False
)

# --------------------------------
# Data Fusion
# Key improvements for evaluation issues:
# 1) include_singletons=True so every input record yields a fused row (coverage / "fused_id: NaN").
# 2) Avoid set-like "union" for ordered track fields to prevent 0% exact-match from reordering.
#    Use longest_string on these columns: with real Python lists, longest_string tends to keep the
#    most complete list and preserve order from one source.
# 3) Duration is numeric-ish; longest_string can pick wrong representation. Use a custom fuser that
#    prefers the median of available numeric seconds and returns a clean string (stable for evaluation).
# --------------------------------

print("Fusing Data")

all_correspondences = pd.concat(
    [
        correspondences_discogs_lastfm,
        correspondences_discogs_musicbrainz,
        correspondences_musicbrainz_lastfm,
    ],
    ignore_index=True
)

def duration_median_seconds(values, ctx=None):
    """
    Conflict resolution for duration:
    - input values may be float/int/str/nan
    - return a normalized seconds string (rounded to nearest integer) for stable comparisons
    """
    cleaned = []
    for v in values:
        if _is_nan(v):
            continue
        x = _to_float_or_nan(v)
        if _is_nan(x):
            continue
        cleaned.append(float(x))
    if not cleaned:
        return np.nan
    med = float(np.median(cleaned))
    # return as integer-like string when close to int
    return str(int(round(med)))

strategy = DataFusionStrategy("music_release_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("duration", duration_median_seconds)
strategy.add_attribute_fuser("label", longest_string)
strategy.add_attribute_fuser("genre", longest_string)

# Track attributes: keep ordered lists from the most complete source to improve exact_match
strategy.add_attribute_fuser("tracks_track_name", longest_string)
strategy.add_attribute_fuser("tracks_track_position", longest_string)
strategy.add_attribute_fuser("tracks_track_duration", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_rb_standard_blocker.jsonl"
)

fused = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_correspondences,
    id_column="id",
    include_singletons=True,
)

if not os.path.exists("output/data_fusion"):
    os.makedirs("output/data_fusion", exist_ok=True)

fused.to_csv("output/data_fusion/fusion_rb_standard_blocker.csv", index=False)