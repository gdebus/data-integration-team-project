# Pipeline Snapshots

notebook_name=AdaptationPipeline_Final_Blocker_Matcher
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=64.96%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_csv

from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker, SortedNeighbourhoodBlocker, TokenBlocker
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""  # paths provided are already relative

# Define API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets (already schema-matched exports, but we still run matching as in template)
discogs = load_csv(
    DATA_DIR + "output/schema-matching/discogs.csv",
    name="discogs",
)

lastfm = load_csv(
    DATA_DIR + "output/schema-matching/lastfm.csv",
    name="lastfm",
)

musicbrainz = load_csv(
    DATA_DIR + "output/schema-matching/musicbrainz.csv",
    name="musicbrainz",
)

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# CRITICAL INSTRUCTION FOR AGENTS:
# The here implemented schema matching will match the schema of dataset2 and dataset3 to the schema of dataset1.
# Therefore, the resulting columns for all datasets will have the schema of dataset1.
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

# match schema of discogs with lastfm and rename schema of lastfm
schema_correspondences = schema_matcher.match(discogs, lastfm)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
lastfm = lastfm.rename(columns=rename_map)

# match schema of discogs with musicbrainz and rename schema of musicbrainz
schema_correspondences = schema_matcher.match(discogs, musicbrainz)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
musicbrainz = musicbrainz.rename(columns=rename_map)

# --------------------------------
# Blocking
# CRITICAL INSTRUCTION FOR AGENTS:
# YOU MUST!!! USE THE PRECOMPUTED BLOCKERS PROVIEDED TO YOU UNDER "5. **BLOCKING CONFIGURATION**"!
# --------------------------------

print("Performing Blocking")

# From blocking configuration:
# discogs_lastfm: semantic_similarity on ["name","artist"] top_k=20
# discogs_musicbrainz: semantic_similarity on ["name","artist","release-date"] top_k=20
# musicbrainz_lastfm: semantic_similarity on ["name","artist","duration"] top_k=20
#
# Strategy mapping: semantic_similarity -> EmbeddingBlocker (top_k)

blocker_d2l = EmbeddingBlocker(
    discogs, lastfm,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id"
)

blocker_d2m = EmbeddingBlocker(
    discogs, musicbrainz,
    text_cols=["name", "artist", "release-date"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id"
)

blocker_m2l = EmbeddingBlocker(
    musicbrainz, lastfm,
    text_cols=["name", "artist", "duration"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id"
)

# --------------------------------
# Matching
# CRITICAL INSTRUCTION FOR AGENTS:
# You MUST use the matching configuration supplied to you under "6. **MATCHING CONFIGURATION**"
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()

comparators_d2l = [
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
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=120.0,
    ),
]

comparators_d2m = [
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
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=120.0,
    ),
]

comparators_m2l = [
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
        max_difference=15.0,
    ),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_d2l = rb_matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_d2l,
    comparators=comparators_d2l,
    weights=[0.35, 0.35, 0.2, 0.1],
    threshold=0.55,
    id_column="id",
)

rb_correspondences_d2m = rb_matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_d2m,
    comparators=comparators_d2m,
    weights=[0.32, 0.28, 0.15, 0.15, 0.1],
    threshold=0.78,
    id_column="id",
)

rb_correspondences_m2l = rb_matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_m2l,
    comparators=comparators_m2l,
    weights=[0.35, 0.45, 0.2],
    threshold=0.75,
    id_column="id",
)

print("Fusing Data")

# merge rule based correspondences
all_rb_correspondences = pd.concat(
    [rb_correspondences_d2l, rb_correspondences_d2m, rb_correspondences_m2l],
    ignore_index=True
)

# --------------------------------
# Data Fusion
# --------------------------------

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# Prefer more complete textual attributes via longest_string; union for track lists
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
accuracy_score=42.64%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_csv

from PyDI.entitymatching import (
    EmbeddingBlocker,
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
import numpy as np
from dotenv import load_dotenv
import os
import ast
import re

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""  # paths provided are already relative

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

discogs = load_csv(
    DATA_DIR + "output/schema-matching/discogs.csv",
    name="discogs",
)
lastfm = load_csv(
    DATA_DIR + "output/schema-matching/lastfm.csv",
    name="lastfm",
)
musicbrainz = load_csv(
    DATA_DIR + "output/schema-matching/musicbrainz.csv",
    name="musicbrainz",
)

# --------------------------------
# Light cleaning / normalization (improves matching robustness)
# --------------------------------

def lower_strip(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        # pd.isna can raise on array-like; treat as non-missing and stringify
        pass
    return str(x).lower().strip()


def is_missing_scalar(x) -> bool:
    """Safe missing check that won't error on list/array-like."""
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict, np.ndarray)):
        return False
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def parse_listish(x):
    """
    Parse stringified python lists like "['a','b']" -> ['a','b'].
    Return NaN if missing/empty. Keep list/tuple/set as list.
    """
    if is_missing_scalar(x):
        return np.nan

    if isinstance(x, (list, tuple, set, np.ndarray)):
        return list(x)

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan

    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, set, np.ndarray)):
            return list(v)
        if v is None:
            return np.nan
        return [v]
    except Exception:
        # fallback: if it's a delimited string, split; else single item
        if " / " in s:
            parts = [p.strip() for p in s.split(" / ") if p.strip()]
            return parts if parts else np.nan
        return [s]


def to_numeric_seconds(x):
    if is_missing_scalar(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        m = re.search(r"[-+]?\d*\.?\d+", s)
        return float(m.group(0)) if m else np.nan


def ensure_columns(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def normalize_for_matching(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply normalization consistently across datasets to improve blocking/matching.
    - Parse list-like track columns
    - Coerce duration to numeric seconds
    - Add helper columns for more robust semantic blocking
    """
    df = df.copy()

    for c in ["tracks_track_name", "tracks_track_position", "tracks_track_duration"]:
        if c in df.columns:
            df[c] = df[c].apply(parse_listish)

    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(to_numeric_seconds)

    # A stable text used only for embedding blocking (keeps required config columns intact)
    def tracks_to_text(v):
        v = parse_listish(v)
        if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, list):
            return " ".join([lower_strip(t) for t in v if lower_strip(t)])
        return lower_strip(v)

    name = df["name"].map(lower_strip) if "name" in df.columns else ""
    artist = df["artist"].map(lower_strip) if "artist" in df.columns else ""
    tracks_txt = df["tracks_track_name"].apply(tracks_to_text) if "tracks_track_name" in df.columns else ""

    df["_block_text_name_artist"] = (name + " " + artist).str.strip()
    df["_block_text_name_artist_tracks"] = (name + " " + artist + " " + tracks_txt).str.strip()

    # Ensure release-date is a clean string for embedding and date comparator
    if "release-date" in df.columns:
        df["release-date"] = df["release-date"].apply(lambda x: "" if is_missing_scalar(x) else str(x).strip())

    return df


needed_cols = [
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
]

discogs = ensure_columns(discogs, needed_cols)
lastfm = ensure_columns(lastfm, needed_cols)
musicbrainz = ensure_columns(musicbrainz, needed_cols)

discogs = normalize_for_matching(discogs)
lastfm = normalize_for_matching(lastfm)
musicbrainz = normalize_for_matching(musicbrainz)

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# CRITICAL INSTRUCTION FOR AGENTS:
# The here implemented schema matching will match the schema of dataset2 and dataset3 to the schema of dataset1.
# Therefore, the resulting columns for all datasets will have the schema of dataset1.
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
    debug=True,
)

schema_correspondences = schema_matcher.match(discogs, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(discogs, musicbrainz)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
musicbrainz = musicbrainz.rename(columns=rename_map)

# Ensure required columns still exist after renaming
discogs = ensure_columns(discogs, needed_cols)
lastfm = ensure_columns(lastfm, needed_cols)
musicbrainz = ensure_columns(musicbrainz, needed_cols)

# Re-apply normalization post schema-rename (idempotent)
discogs = normalize_for_matching(discogs)
lastfm = normalize_for_matching(lastfm)
musicbrainz = normalize_for_matching(musicbrainz)

# --------------------------------
# Blocking
# CRITICAL INSTRUCTION FOR AGENTS:
# YOU MUST!!! USE THE PRECOMPUTED BLOCKERS PROVIEDED TO YOU UNDER "5. **BLOCKING CONFIGURATION**"!
# --------------------------------

print("Performing Blocking")

# discogs_lastfm: semantic_similarity on ["name","artist"] top_k=20
blocker_d2l = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# discogs_musicbrainz: semantic_similarity on ["name","artist","release-date"] top_k=20
blocker_d2m = EmbeddingBlocker(
    discogs,
    musicbrainz,
    text_cols=["name", "artist", "release-date"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# musicbrainz_lastfm: semantic_similarity on ["name","artist","duration"] top_k=20
blocker_m2l = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching
# CRITICAL INSTRUCTION FOR AGENTS:
# You MUST use the matching configuration supplied to you under "6. **MATCHING CONFIGURATION**"
# --------------------------------

comparators_d2l = [
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
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=120.0,
    ),
]

comparators_d2m = [
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
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=120.0,
    ),
]

comparators_m2l = [
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
        max_difference=15.0,
    ),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_d2l = rb_matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_d2l,
    comparators=comparators_d2l,
    weights=[0.35, 0.35, 0.2, 0.1],
    threshold=0.55,
    id_column="id",
)

rb_correspondences_d2m = rb_matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_d2m,
    comparators=comparators_d2m,
    weights=[0.32, 0.28, 0.15, 0.15, 0.1],
    threshold=0.78,
    id_column="id",
)

rb_correspondences_m2l = rb_matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_m2l,
    comparators=comparators_m2l,
    weights=[0.35, 0.45, 0.2],
    threshold=0.75,
    id_column="id",
)

# Save correspondences with correct filenames (do not change names)
CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_d2l.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)
rb_correspondences_d2m.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)
rb_correspondences_m2l.to_csv(
    os.path.join(CORR_DIR, "correspondences_musicbrainz_lastfm.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_d2l, rb_correspondences_d2m, rb_correspondences_m2l],
    ignore_index=True,
)

# --------------------------------
# Data Fusion
# --------------------------------

strategy = DataFusionStrategy("rule_based_fusion_strategy")

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

rb_fused_standard_blocker = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=18
node_name=execute_pipeline
accuracy_score=34.52%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_csv

from PyDI.entitymatching import EmbeddingBlocker
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
import numpy as np
from dotenv import load_dotenv
import os
import ast
import re

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = ""  # paths provided are already relative

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

discogs = load_csv(
    DATA_DIR + "output/schema-matching/discogs.csv",
    name="discogs",
)
lastfm = load_csv(
    DATA_DIR + "output/schema-matching/lastfm.csv",
    name="lastfm",
)
musicbrainz = load_csv(
    DATA_DIR + "output/schema-matching/musicbrainz.csv",
    name="musicbrainz",
)

# --------------------------------
# Cleaning / normalization helpers
# --------------------------------


def is_missing_scalar(x) -> bool:
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict, np.ndarray)):
        return False
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def lower_strip(x):
    if is_missing_scalar(x):
        return ""
    return str(x).lower().strip()


def parse_listish(x):
    """
    Parse stringified python lists like "['a','b']" -> ['a','b'].
    Return NaN if missing/empty. Keep list/tuple/set/np.ndarray as list.
    """
    if is_missing_scalar(x):
        return np.nan

    if isinstance(x, (list, tuple, set, np.ndarray)):
        v = list(x)
        return v if len(v) > 0 else np.nan

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan

    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, set, np.ndarray)):
            v = list(v)
            return v if len(v) > 0 else np.nan
        if v is None:
            return np.nan
        return [v]
    except Exception:
        # fallback: split common delimiters
        if " / " in s:
            parts = [p.strip() for p in s.split(" / ") if p.strip()]
            return parts if parts else np.nan
        if ";" in s:
            parts = [p.strip() for p in s.split(";") if p.strip()]
            return parts if parts else np.nan
        return [s]


def to_numeric_seconds(x):
    if is_missing_scalar(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        m = re.search(r"[-+]?\d*\.?\d+", s)
        return float(m.group(0)) if m else np.nan


def ensure_columns(df, cols):
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def normalize_release_date(x):
    """
    Normalize to YYYY-MM-DD string when possible (helps DateComparator + fusion).
    """
    if is_missing_scalar(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if pd.isna(dt):
        # keep original string if unparsable
        return s
    return dt.date().isoformat()


def canonicalize_track_names(v):
    """
    Lower/strip, remove bracketed info, normalize separators, and deduplicate.
    Improves both matching and fusion accuracy for tracks_track_name.
    """
    v = parse_listish(v)
    if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    out = []
    for t in v:
        s = lower_strip(t)
        if not s:
            continue
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s*[-–—]\s*", " - ", s)
        s = re.sub(r"\s*/\s*", " / ", s)
        s = re.sub(r"\(.*?\)", "", s).strip()
        s = re.sub(r"\[.*?\]", "", s).strip()
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            out.append(s)
    if not out:
        return np.nan
    seen = set()
    dedup = []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


def normalize_track_positions(v):
    v = parse_listish(v)
    if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    out = []
    for p in v:
        s = str(p).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            continue
        # keep only leading digits if present (A1 -> 1, "01" -> 1)
        m = re.search(r"\d+", s)
        out.append(str(int(m.group(0))) if m else s)
    return out if out else np.nan


def normalize_track_durations(v):
    v = parse_listish(v)
    if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    out = []
    for d in v:
        sec = to_numeric_seconds(d)
        if not (isinstance(sec, float) and np.isnan(sec)):
            out.append(int(round(sec)))
    return out if out else np.nan


def extract_artist_from_name(name, artist):
    """
    Some rows (e.g., lastfm) embed artist in the name as "Artist - Title".
    Create a helper for blocking/matching without changing configured columns.
    """
    a = lower_strip(artist)
    n = str(name) if not is_missing_scalar(name) else ""
    if a:
        return a
    if " - " in n:
        left = n.split(" - ", 1)[0]
        return lower_strip(left)
    return ""


def extract_title_from_name(name):
    """
    Create helper to remove embedded artist prefix from name.
    """
    n = str(name) if not is_missing_scalar(name) else ""
    if " - " in n:
        return lower_strip(n.split(" - ", 1)[1])
    return lower_strip(n)


def normalize_for_matching(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Track columns: normalize aggressively to improve matching + fusion
    if "tracks_track_name" in df.columns:
        df["tracks_track_name"] = df["tracks_track_name"].apply(canonicalize_track_names)
    if "tracks_track_position" in df.columns:
        df["tracks_track_position"] = df["tracks_track_position"].apply(normalize_track_positions)
    if "tracks_track_duration" in df.columns:
        df["tracks_track_duration"] = df["tracks_track_duration"].apply(normalize_track_durations)

    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(to_numeric_seconds)

    if "release-date" in df.columns:
        df["release-date"] = df["release-date"].apply(normalize_release_date)

    # Helpers for embedding blocking: keep required config columns intact
    # (EmbeddingBlocker will use these helpers via text_cols)
    df["_artist_from_name"] = [
        extract_artist_from_name(n, a) for n, a in zip(df.get("name", []), df.get("artist", []))
    ]
    df["_title_from_name"] = df.get("name", pd.Series([np.nan] * len(df))).apply(extract_title_from_name)

    def tracks_to_text(v):
        v = parse_listish(v)
        if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, list):
            return " ".join([lower_strip(t) for t in v if lower_strip(t)])
        return lower_strip(v)

    tracks_txt = df["tracks_track_name"].apply(tracks_to_text) if "tracks_track_name" in df.columns else ""
    df["_block_text_name_artist_tracks"] = (
        df["_title_from_name"].fillna("").astype(str).map(lower_strip)
        + " "
        + df["_artist_from_name"].fillna("").astype(str).map(lower_strip)
        + " "
        + tracks_txt.astype(str)
    ).str.strip()

    return df


needed_cols = [
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
]

discogs = ensure_columns(discogs, needed_cols)
lastfm = ensure_columns(lastfm, needed_cols)
musicbrainz = ensure_columns(musicbrainz, needed_cols)

discogs = normalize_for_matching(discogs)
lastfm = normalize_for_matching(lastfm)
musicbrainz = normalize_for_matching(musicbrainz)

# --------------------------------
# Schema Matching (LLM)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = schema_matcher.match(discogs, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(discogs, musicbrainz)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
musicbrainz = musicbrainz.rename(columns=rename_map)

discogs = ensure_columns(discogs, needed_cols)
lastfm = ensure_columns(lastfm, needed_cols)
musicbrainz = ensure_columns(musicbrainz, needed_cols)

discogs = normalize_for_matching(discogs)
lastfm = normalize_for_matching(lastfm)
musicbrainz = normalize_for_matching(musicbrainz)

# --------------------------------
# Blocking (MUST use precomputed strategies)
# --------------------------------

print("Performing Blocking")

# discogs_lastfm: semantic_similarity on ["name","artist"] top_k=20
# Use normalized helper columns by overwriting those text fields ONLY for blocking input
# while preserving original columns for matching/fusion (helpers already exist).
discogs_block_d2l = discogs.copy()
lastfm_block_d2l = lastfm.copy()
discogs_block_d2l["name"] = discogs_block_d2l["_title_from_name"]
lastfm_block_d2l["name"] = lastfm_block_d2l["_title_from_name"]
discogs_block_d2l["artist"] = discogs_block_d2l["_artist_from_name"]
lastfm_block_d2l["artist"] = lastfm_block_d2l["_artist_from_name"]

blocker_d2l = EmbeddingBlocker(
    discogs_block_d2l,
    lastfm_block_d2l,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# discogs_musicbrainz: semantic_similarity on ["name","artist","release-date"] top_k=20
discogs_block_d2m = discogs.copy()
musicbrainz_block_d2m = musicbrainz.copy()
discogs_block_d2m["name"] = discogs_block_d2m["_title_from_name"]
musicbrainz_block_d2m["name"] = musicbrainz_block_d2m["_title_from_name"]
discogs_block_d2m["artist"] = discogs_block_d2m["_artist_from_name"]
musicbrainz_block_d2m["artist"] = musicbrainz_block_d2m["_artist_from_name"]

blocker_d2m = EmbeddingBlocker(
    discogs_block_d2m,
    musicbrainz_block_d2m,
    text_cols=["name", "artist", "release-date"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# musicbrainz_lastfm: semantic_similarity on ["name","artist","duration"] top_k=20
musicbrainz_block_m2l = musicbrainz.copy()
lastfm_block_m2l = lastfm.copy()
musicbrainz_block_m2l["name"] = musicbrainz_block_m2l["_title_from_name"]
lastfm_block_m2l["name"] = lastfm_block_m2l["_title_from_name"]
musicbrainz_block_m2l["artist"] = musicbrainz_block_m2l["_artist_from_name"]
lastfm_block_m2l["artist"] = lastfm_block_m2l["_artist_from_name"]

blocker_m2l = EmbeddingBlocker(
    musicbrainz_block_m2l,
    lastfm_block_m2l,
    text_cols=["name", "artist", "duration"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching (MUST use provided comparator config)
# --------------------------------

comparators_d2l = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lambda x: lower_strip(x),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lambda x: lower_strip(extract_title_from_name(x)),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="tracks_track_name",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=120.0,
    ),
]

comparators_d2m = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lambda x: lower_strip(extract_title_from_name(x)),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lambda x: lower_strip(x),
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
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=120.0,
    ),
]

comparators_m2l = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lambda x: lower_strip(x),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lambda x: lower_strip(extract_title_from_name(x)),
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=15.0,
    ),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_d2l = rb_matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_d2l,
    comparators=comparators_d2l,
    weights=[0.35, 0.35, 0.2, 0.1],
    threshold=0.55,
    id_column="id",
)

rb_correspondences_d2m = rb_matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_d2m,
    comparators=comparators_d2m,
    weights=[0.32, 0.28, 0.15, 0.15, 0.1],
    threshold=0.78,
    id_column="id",
)

rb_correspondences_m2l = rb_matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_m2l,
    comparators=comparators_m2l,
    weights=[0.35, 0.45, 0.2],
    threshold=0.75,
    id_column="id",
)

# Save correspondences with correct filenames (do not change names)
CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_d2l.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)
rb_correspondences_d2m.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)
rb_correspondences_m2l.to_csv(
    os.path.join(CORR_DIR, "correspondences_musicbrainz_lastfm.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_d2l, rb_correspondences_d2m, rb_correspondences_m2l],
    ignore_index=True,
)

# --------------------------------
# Data Fusion (improve track-related fusion)
# --------------------------------

def prefer_non_empty_string(series: pd.Series):
    vals = []
    for v in series.tolist():
        if is_missing_scalar(v):
            continue
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", "null"}:
            vals.append(s)
    if not vals:
        return np.nan
    # prefer longer (more informative), tie-breaker by first
    vals_sorted = sorted(vals, key=lambda x: (len(x), x), reverse=True)
    return vals_sorted[0]


def fuse_track_names(series: pd.Series):
    # union but keep deterministic order by appearance
    items = []
    seen = set()
    for v in series.tolist():
        v = canonicalize_track_names(v)
        if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
            continue
        for t in v:
            tt = lower_strip(t)
            if not tt:
                continue
            if tt not in seen:
                seen.add(tt)
                items.append(t)  # keep original normalized token
    return items if items else np.nan


def fuse_track_positions(series: pd.Series):
    # prefer an ordered complete sequence if available; else union
    candidates = []
    for v in series.tolist():
        v = normalize_track_positions(v)
        if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
            continue
        # keep only numeric positions for ordering
        nums = []
        ok = True
        for p in v:
            try:
                nums.append(int(p))
            except Exception:
                ok = False
                break
        if ok and nums:
            candidates.append(nums)

    if candidates:
        # prefer the longest list (most complete); tie-breaker: smallest max position
        best = sorted(candidates, key=lambda x: (len(x), -max(x)), reverse=True)[0]
        return [str(x) for x in best]

    # fallback union
    u = []
    seen = set()
    for v in series.tolist():
        v = normalize_track_positions(v)
        if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
            continue
        for p in v:
            if p not in seen:
                seen.add(p)
                u.append(p)
    return u if u else np.nan


def fuse_track_durations(series: pd.Series):
    # prefer list that aligns with longest available track list length (from same record often)
    parsed = []
    for v in series.tolist():
        v = normalize_track_durations(v)
        if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
            continue
        parsed.append(v)

    if not parsed:
        return np.nan

    # pick the most informative: longest list, then highest sum (often indicates full durations present)
    best = sorted(parsed, key=lambda x: (len(x), sum(x)), reverse=True)[0]
    return best


strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", prefer_non_empty_string)
strategy.add_attribute_fuser("release-country", prefer_non_empty_string)
strategy.add_attribute_fuser("duration", maximum)
strategy.add_attribute_fuser("label", prefer_non_empty_string)
strategy.add_attribute_fuser("genre", prefer_non_empty_string)

# Custom fusers to address 0% tracks_track_name accuracy and low track duration accuracy
strategy.add_attribute_fuser("tracks_track_name", fuse_track_names)
strategy.add_attribute_fuser("tracks_track_position", fuse_track_positions)
strategy.add_attribute_fuser("tracks_track_duration", fuse_track_durations)

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

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

============================================================
PIPELINE SNAPSHOT 04 START
============================================================
node_index=23
node_name=execute_pipeline
accuracy_score=33.50%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_csv

from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    maximum,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import ast
import re

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = ""  # paths provided are already relative

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

discogs = load_csv(
    DATA_DIR + "output/schema-matching/discogs.csv",
    name="discogs",
)
lastfm = load_csv(
    DATA_DIR + "output/schema-matching/lastfm.csv",
    name="lastfm",
)
musicbrainz = load_csv(
    DATA_DIR + "output/schema-matching/musicbrainz.csv",
    name="musicbrainz",
)

# --------------------------------
# Cleaning / normalization helpers
# --------------------------------


def is_missing_scalar(x) -> bool:
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict, np.ndarray)):
        return False
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def lower_strip(x):
    if is_missing_scalar(x):
        return ""
    return str(x).lower().strip()


def parse_listish(x):
    """
    Parse stringified python lists like "['a','b']" -> ['a','b'].
    Return NaN if missing/empty. Keep list/tuple/set/np.ndarray as list.
    """
    if is_missing_scalar(x):
        return np.nan

    if isinstance(x, (list, tuple, set, np.ndarray)):
        v = list(x)
        return v if len(v) > 0 else np.nan

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan

    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, set, np.ndarray)):
            v = list(v)
            return v if len(v) > 0 else np.nan
        if v is None:
            return np.nan
        return [v]
    except Exception:
        # fallback: split common delimiters
        if " / " in s:
            parts = [p.strip() for p in s.split(" / ") if p.strip()]
            return parts if parts else np.nan
        if ";" in s:
            parts = [p.strip() for p in s.split(";") if p.strip()]
            return parts if parts else np.nan
        return [s]


def to_numeric_seconds(x):
    if is_missing_scalar(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    # handle mm:ss or hh:mm:ss
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", s):
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 2:
            mm, ss = parts
            return float(mm * 60 + ss)
        hh, mm, ss = parts
        return float(hh * 3600 + mm * 60 + ss)
    try:
        return float(s)
    except Exception:
        m = re.search(r"[-+]?\d*\.?\d+", s)
        return float(m.group(0)) if m else np.nan


def ensure_columns(df, cols):
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def normalize_release_date(x):
    """
    Normalize to YYYY-MM-DD string when possible (helps DateComparator + fusion).
    """
    if is_missing_scalar(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if pd.isna(dt):
        return s
    return dt.date().isoformat()


def canonicalize_track_names(v):
    """
    Return a list of canonical track names in a stable order.
    """
    v = parse_listish(v)
    if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    out = []
    for t in v:
        s = lower_strip(t)
        if not s:
            continue
        s = re.sub(r"\(.*?\)", "", s).strip()
        s = re.sub(r"\[.*?\]", "", s).strip()
        s = re.sub(r"[_]+", " ", s)
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s*[-–—]\s*", " - ", s)
        s = re.sub(r"\s*/\s*", " / ", s)
        s = s.strip()
        if s:
            out.append(s)
    if not out:
        return np.nan
    seen = set()
    dedup = []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


def normalize_track_positions(v):
    v = parse_listish(v)
    if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    out = []
    for p in v:
        s = str(p).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            continue
        m = re.search(r"\d+", s)
        out.append(str(int(m.group(0))) if m else s)
    return out if out else np.nan


def normalize_track_durations(v):
    v = parse_listish(v)
    if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    out = []
    for d in v:
        sec = to_numeric_seconds(d)
        if not (isinstance(sec, float) and np.isnan(sec)):
            out.append(int(round(sec)))
    return out if out else np.nan


def extract_artist_from_name(name, artist):
    """
    Some rows (e.g., lastfm) embed artist in the name as "Artist - Title".
    """
    a = lower_strip(artist)
    n = str(name) if not is_missing_scalar(name) else ""
    if a:
        return a
    if " - " in n:
        left = n.split(" - ", 1)[0]
        return lower_strip(left)
    return ""


def extract_title_from_name(name):
    """
    Remove embedded artist prefix from name.
    """
    n = str(name) if not is_missing_scalar(name) else ""
    if " - " in n:
        return lower_strip(n.split(" - ", 1)[1])
    return lower_strip(n)


def normalize_for_matching(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalize core fields
    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(to_numeric_seconds)
        # treat 0 duration as missing for these music datasets
        df.loc[df["duration"].fillna(np.nan) == 0, "duration"] = np.nan

    if "release-date" in df.columns:
        df["release-date"] = df["release-date"].apply(normalize_release_date)

    # normalize track columns
    if "tracks_track_name" in df.columns:
        df["tracks_track_name"] = df["tracks_track_name"].apply(canonicalize_track_names)
    if "tracks_track_position" in df.columns:
        df["tracks_track_position"] = df["tracks_track_position"].apply(normalize_track_positions)
    if "tracks_track_duration" in df.columns:
        df["tracks_track_duration"] = df["tracks_track_duration"].apply(normalize_track_durations)

    # helpers for blocking
    df["_artist_from_name"] = [
        extract_artist_from_name(n, a) for n, a in zip(df.get("name", []), df.get("artist", []))
    ]
    df["_title_from_name"] = df.get("name", pd.Series([np.nan] * len(df))).apply(extract_title_from_name)

    def tracks_to_text(v):
        v = canonicalize_track_names(v)
        if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
            return ""
        return " ".join([lower_strip(t) for t in v if lower_strip(t)])

    tracks_txt = df["tracks_track_name"].apply(tracks_to_text) if "tracks_track_name" in df.columns else ""
    df["_block_text_name_artist_tracks"] = (
        df["_title_from_name"].fillna("").astype(str).map(lower_strip)
        + " "
        + df["_artist_from_name"].fillna("").astype(str).map(lower_strip)
        + " "
        + tracks_txt.astype(str)
    ).str.strip()

    return df


needed_cols = [
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
]

discogs = ensure_columns(discogs, needed_cols)
lastfm = ensure_columns(lastfm, needed_cols)
musicbrainz = ensure_columns(musicbrainz, needed_cols)

discogs = normalize_for_matching(discogs)
lastfm = normalize_for_matching(lastfm)
musicbrainz = normalize_for_matching(musicbrainz)

# --------------------------------
# Schema Matching (LLM)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = schema_matcher.match(discogs, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(discogs, musicbrainz)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
musicbrainz = musicbrainz.rename(columns=rename_map)

discogs = ensure_columns(discogs, needed_cols)
lastfm = ensure_columns(lastfm, needed_cols)
musicbrainz = ensure_columns(musicbrainz, needed_cols)

discogs = normalize_for_matching(discogs)
lastfm = normalize_for_matching(lastfm)
musicbrainz = normalize_for_matching(musicbrainz)

# --------------------------------
# Blocking (MUST use precomputed strategies)
# --------------------------------

print("Performing Blocking")

# discogs_lastfm: semantic_similarity on ["name","artist"] top_k=20
discogs_block_d2l = discogs.copy()
lastfm_block_d2l = lastfm.copy()
discogs_block_d2l["name"] = discogs_block_d2l["_title_from_name"]
lastfm_block_d2l["name"] = lastfm_block_d2l["_title_from_name"]
discogs_block_d2l["artist"] = discogs_block_d2l["_artist_from_name"]
lastfm_block_d2l["artist"] = lastfm_block_d2l["_artist_from_name"]

blocker_d2l = EmbeddingBlocker(
    discogs_block_d2l,
    lastfm_block_d2l,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# discogs_musicbrainz: semantic_similarity on ["name","artist","release-date"] top_k=20
discogs_block_d2m = discogs.copy()
musicbrainz_block_d2m = musicbrainz.copy()
discogs_block_d2m["name"] = discogs_block_d2m["_title_from_name"]
musicbrainz_block_d2m["name"] = musicbrainz_block_d2m["_title_from_name"]
discogs_block_d2m["artist"] = discogs_block_d2m["_artist_from_name"]
musicbrainz_block_d2m["artist"] = musicbrainz_block_d2m["_artist_from_name"]

blocker_d2m = EmbeddingBlocker(
    discogs_block_d2m,
    musicbrainz_block_d2m,
    text_cols=["name", "artist", "release-date"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# musicbrainz_lastfm: semantic_similarity on ["name","artist","duration"] top_k=20
musicbrainz_block_m2l = musicbrainz.copy()
lastfm_block_m2l = lastfm.copy()
musicbrainz_block_m2l["name"] = musicbrainz_block_m2l["_title_from_name"]
lastfm_block_m2l["name"] = lastfm_block_m2l["_title_from_name"]
musicbrainz_block_m2l["artist"] = musicbrainz_block_m2l["_artist_from_name"]
lastfm_block_m2l["artist"] = lastfm_block_m2l["_artist_from_name"]

# duration is numeric; cast to string for embedding text usage without changing column name
musicbrainz_block_m2l["duration"] = musicbrainz_block_m2l["duration"].apply(
    lambda x: "" if is_missing_scalar(x) else str(int(round(float(x)))) if not pd.isna(x) else ""
)
lastfm_block_m2l["duration"] = lastfm_block_m2l["duration"].apply(
    lambda x: "" if is_missing_scalar(x) else str(int(round(float(x)))) if not pd.isna(x) else ""
)

blocker_m2l = EmbeddingBlocker(
    musicbrainz_block_m2l,
    lastfm_block_m2l,
    text_cols=["name", "artist", "duration"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching (MUST use provided comparator config)
# --------------------------------

comparators_d2l = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lambda x: lower_strip(x),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lambda x: lower_strip(extract_title_from_name(x)),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="tracks_track_name",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=120.0,
    ),
]

comparators_d2m = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lambda x: lower_strip(extract_title_from_name(x)),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lambda x: lower_strip(x),
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
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=120.0,
    ),
]

comparators_m2l = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lambda x: lower_strip(x),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lambda x: lower_strip(extract_title_from_name(x)),
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=15.0,
    ),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_d2l = rb_matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_d2l,
    comparators=comparators_d2l,
    weights=[0.35, 0.35, 0.2, 0.1],
    threshold=0.55,
    id_column="id",
)

rb_correspondences_d2m = rb_matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_d2m,
    comparators=comparators_d2m,
    weights=[0.32, 0.28, 0.15, 0.15, 0.1],
    threshold=0.78,
    id_column="id",
)

rb_correspondences_m2l = rb_matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_m2l,
    comparators=comparators_m2l,
    weights=[0.35, 0.45, 0.2],
    threshold=0.75,
    id_column="id",
)

# Save correspondences with correct filenames (do not change names)
CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_d2l.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)
rb_correspondences_d2m.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)
rb_correspondences_m2l.to_csv(
    os.path.join(CORR_DIR, "correspondences_musicbrainz_lastfm.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_d2l, rb_correspondences_d2m, rb_correspondences_m2l],
    ignore_index=True,
)

# --------------------------------
# Data Fusion (fix track fields; keep canonical form)
# --------------------------------

def prefer_non_empty_string(series: pd.Series):
    vals = []
    for v in series.tolist():
        if is_missing_scalar(v):
            continue
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", "null"}:
            vals.append(s)
    if not vals:
        return np.nan
    vals_sorted = sorted(vals, key=lambda x: (len(x), x), reverse=True)
    return vals_sorted[0]


def fuse_track_names(series: pd.Series):
    # choose the most complete (longest) canonical list; tie-breaker by total char length
    candidates = []
    for v in series.tolist():
        v = canonicalize_track_names(v)
        if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, list) and len(v) > 0:
            candidates.append(v)
    if not candidates:
        return np.nan
    best = sorted(candidates, key=lambda x: (len(x), sum(len(t) for t in x)), reverse=True)[0]
    return best


def fuse_track_positions(series: pd.Series):
    candidates = []
    for v in series.tolist():
        v = normalize_track_positions(v)
        if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, list) and len(v) > 0:
            # normalize to numeric strings when possible
            out = []
            ok = True
            for p in v:
                try:
                    out.append(str(int(p)))
                except Exception:
                    ok = False
                    break
            if ok:
                candidates.append(out)
    if not candidates:
        return np.nan
    best = sorted(candidates, key=lambda x: len(x), reverse=True)[0]
    return best


def fuse_track_durations(series: pd.Series):
    candidates = []
    for v in series.tolist():
        v = normalize_track_durations(v)
        if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, list) and len(v) > 0:
            candidates.append(v)
    if not candidates:
        return np.nan
    best = sorted(candidates, key=lambda x: (len(x), sum(x)), reverse=True)[0]
    return best


strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", prefer_non_empty_string)
strategy.add_attribute_fuser("artist", prefer_non_empty_string)
strategy.add_attribute_fuser("release-date", prefer_non_empty_string)
strategy.add_attribute_fuser("release-country", prefer_non_empty_string)
strategy.add_attribute_fuser("duration", maximum)
strategy.add_attribute_fuser("label", prefer_non_empty_string)
strategy.add_attribute_fuser("genre", prefer_non_empty_string)

strategy.add_attribute_fuser("tracks_track_name", fuse_track_names)
strategy.add_attribute_fuser("tracks_track_position", fuse_track_positions)
strategy.add_attribute_fuser("tracks_track_duration", fuse_track_durations)

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

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 04 END
============================================================

============================================================
PIPELINE SNAPSHOT 05 START
============================================================
node_index=28
node_name=execute_pipeline
accuracy_score=33.50%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_csv

from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    maximum,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import ast
import re

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = ""  # paths provided are already relative

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

discogs = load_csv(
    DATA_DIR + "output/schema-matching/discogs.csv",
    name="discogs",
)
lastfm = load_csv(
    DATA_DIR + "output/schema-matching/lastfm.csv",
    name="lastfm",
)
musicbrainz = load_csv(
    DATA_DIR + "output/schema-matching/musicbrainz.csv",
    name="musicbrainz",
)

# --------------------------------
# Cleaning / normalization helpers
# --------------------------------


def is_missing_scalar(x) -> bool:
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict, np.ndarray)):
        return False
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def lower_strip(x):
    if is_missing_scalar(x):
        return ""
    return str(x).lower().strip()


def parse_listish(x):
    """
    Parse stringified python lists like "['a','b']" -> ['a','b'].
    Return NaN if missing/empty. Keep list/tuple/set/np.ndarray as list.
    """
    if is_missing_scalar(x):
        return np.nan

    if isinstance(x, (list, tuple, set, np.ndarray)):
        v = list(x)
        return v if len(v) > 0 else np.nan

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan

    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, set, np.ndarray)):
            v = list(v)
            return v if len(v) > 0 else np.nan
        if v is None:
            return np.nan
        return [v]
    except Exception:
        # fallback: split common delimiters
        if " / " in s:
            parts = [p.strip() for p in s.split(" / ") if p.strip()]
            return parts if parts else np.nan
        if ";" in s:
            parts = [p.strip() for p in s.split(";") if p.strip()]
            return parts if parts else np.nan
        return [s]


def to_numeric_seconds(x):
    if is_missing_scalar(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    # handle mm:ss or hh:mm:ss
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", s):
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 2:
            mm, ss = parts
            return float(mm * 60 + ss)
        hh, mm, ss = parts
        return float(hh * 3600 + mm * 60 + ss)
    try:
        return float(s)
    except Exception:
        m = re.search(r"[-+]?\d*\.?\d+", s)
        return float(m.group(0)) if m else np.nan


def ensure_columns(df, cols):
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def normalize_release_date(x):
    """
    Normalize to YYYY-MM-DD string when possible (helps DateComparator + fusion).
    """
    if is_missing_scalar(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if pd.isna(dt):
        return s
    return dt.date().isoformat()


def normalize_release_country(x):
    """
    Canonicalize common country variants to improve fusion accuracy.
    """
    if is_missing_scalar(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan

    sl = s.lower().strip()
    mapping = {
        "uk": "UK",
        "u.k.": "UK",
        "united kingdom": "UK",
        "great britain": "UK",
        "united kingdom of great britain and northern ireland": "UK",
        "england": "UK",
        "scotland": "UK",
        "wales": "UK",
        "northern ireland": "UK",
        "u.s.": "US",
        "usa": "US",
        "united states": "US",
        "united states of america": "US",
    }
    if sl in mapping:
        return mapping[sl]

    # keep short all-caps country codes as-is
    if re.fullmatch(r"[A-Z]{2,3}", s):
        return s

    # Title case fallback
    return " ".join([w.capitalize() for w in re.split(r"\s+", sl) if w])


def clean_track_token(s: str) -> str:
    s = lower_strip(s)
    if not s:
        return ""
    # remove featuring parts that often differ by source
    s = re.sub(r"\b(feat\.?|featuring|ft\.)\b.*$", "", s).strip()
    s = re.sub(r"\(.*?\)", "", s).strip()
    s = re.sub(r"\[.*?\]", "", s).strip()
    s = re.sub(r"[_]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*[-–—]\s*", " - ", s)
    s = re.sub(r"\s*/\s*", " / ", s)
    return s.strip()


def canonicalize_track_names(v):
    """
    Return a list of canonical track names in a stable order.
    """
    v = parse_listish(v)
    if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    out = []
    for t in v:
        s = clean_track_token(t)
        if s:
            out.append(s)
    if not out:
        return np.nan
    seen = set()
    dedup = []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


def normalize_track_positions(v):
    v = parse_listish(v)
    if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    out = []
    for p in v:
        s = str(p).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            continue
        m = re.search(r"\d+", s)
        out.append(str(int(m.group(0))) if m else s)
    return out if out else np.nan


def normalize_track_durations(v):
    v = parse_listish(v)
    if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    out = []
    for d in v:
        sec = to_numeric_seconds(d)
        if not (isinstance(sec, float) and np.isnan(sec)):
            out.append(int(round(sec)))
    return out if out else np.nan


def extract_artist_from_name(name, artist):
    """
    Some rows (e.g., lastfm) embed artist in the name as "Artist - Title".
    If an explicit artist exists, keep it (normalized).
    """
    a = lower_strip(artist)
    if a:
        return a

    n = str(name) if not is_missing_scalar(name) else ""
    if " - " in n:
        left = n.split(" - ", 1)[0]
        return lower_strip(left)
    return ""


def extract_title_from_name(name):
    """
    Remove embedded artist prefix from name.
    """
    n = str(name) if not is_missing_scalar(name) else ""
    if " - " in n:
        return lower_strip(n.split(" - ", 1)[1])
    return lower_strip(n)


def safe_list_to_string(v):
    v = parse_listish(v)
    if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
        return ""
    return " ".join([lower_strip(x) for x in v if lower_strip(x)]).strip()


def normalize_for_matching(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalize core fields
    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(to_numeric_seconds)
        # treat 0 duration as missing for these music datasets
        df.loc[df["duration"].fillna(np.nan) == 0, "duration"] = np.nan

    if "release-date" in df.columns:
        df["release-date"] = df["release-date"].apply(normalize_release_date)

    if "release-country" in df.columns:
        df["release-country"] = df["release-country"].apply(normalize_release_country)

    # normalize track columns to lists (not strings)
    if "tracks_track_name" in df.columns:
        df["tracks_track_name"] = df["tracks_track_name"].apply(canonicalize_track_names)
    if "tracks_track_position" in df.columns:
        df["tracks_track_position"] = df["tracks_track_position"].apply(normalize_track_positions)
    if "tracks_track_duration" in df.columns:
        df["tracks_track_duration"] = df["tracks_track_duration"].apply(normalize_track_durations)

    # helpers for blocking
    df["_artist_from_name"] = [
        extract_artist_from_name(n, a) for n, a in zip(df.get("name", []), df.get("artist", []))
    ]
    df["_title_from_name"] = df.get("name", pd.Series([np.nan] * len(df))).apply(extract_title_from_name)

    # additional composite text for embedding blocking stability (keeps config columns, but improves content)
    df["_tracks_text"] = df.get("tracks_track_name", pd.Series([np.nan] * len(df))).apply(safe_list_to_string)

    return df


needed_cols = [
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
]

discogs = ensure_columns(discogs, needed_cols)
lastfm = ensure_columns(lastfm, needed_cols)
musicbrainz = ensure_columns(musicbrainz, needed_cols)

discogs = normalize_for_matching(discogs)
lastfm = normalize_for_matching(lastfm)
musicbrainz = normalize_for_matching(musicbrainz)

# --------------------------------
# Schema Matching (LLM)
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

schema_matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = schema_matcher.match(discogs, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(discogs, musicbrainz)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
musicbrainz = musicbrainz.rename(columns=rename_map)

discogs = ensure_columns(discogs, needed_cols)
lastfm = ensure_columns(lastfm, needed_cols)
musicbrainz = ensure_columns(musicbrainz, needed_cols)

discogs = normalize_for_matching(discogs)
lastfm = normalize_for_matching(lastfm)
musicbrainz = normalize_for_matching(musicbrainz)

# --------------------------------
# Blocking (MUST use precomputed strategies)
# --------------------------------

print("Performing Blocking")

# NOTE: Keep the configured blocking columns, but enrich their content using normalized title/artist
# and (for name) append tracks text to better separate releases with same title.

# discogs_lastfm: semantic_similarity on ["name","artist"] top_k=20
discogs_block_d2l = discogs.copy()
lastfm_block_d2l = lastfm.copy()

discogs_block_d2l["artist"] = discogs_block_d2l["_artist_from_name"]
lastfm_block_d2l["artist"] = lastfm_block_d2l["_artist_from_name"]

discogs_block_d2l["name"] = (
    discogs_block_d2l["_title_from_name"].fillna("").astype(str).map(lower_strip)
    + " "
    + discogs_block_d2l["_tracks_text"].fillna("").astype(str).map(lower_strip)
).str.strip()
lastfm_block_d2l["name"] = (
    lastfm_block_d2l["_title_from_name"].fillna("").astype(str).map(lower_strip)
    + " "
    + lastfm_block_d2l["_tracks_text"].fillna("").astype(str).map(lower_strip)
).str.strip()

blocker_d2l = EmbeddingBlocker(
    discogs_block_d2l,
    lastfm_block_d2l,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# discogs_musicbrainz: semantic_similarity on ["name","artist","release-date"] top_k=20
discogs_block_d2m = discogs.copy()
musicbrainz_block_d2m = musicbrainz.copy()

discogs_block_d2m["artist"] = discogs_block_d2m["_artist_from_name"]
musicbrainz_block_d2m["artist"] = musicbrainz_block_d2m["_artist_from_name"]

discogs_block_d2m["name"] = (
    discogs_block_d2m["_title_from_name"].fillna("").astype(str).map(lower_strip)
    + " "
    + discogs_block_d2m["_tracks_text"].fillna("").astype(str).map(lower_strip)
).str.strip()
musicbrainz_block_d2m["name"] = (
    musicbrainz_block_d2m["_title_from_name"].fillna("").astype(str).map(lower_strip)
    + " "
    + musicbrainz_block_d2m["_tracks_text"].fillna("").astype(str).map(lower_strip)
).str.strip()

blocker_d2m = EmbeddingBlocker(
    discogs_block_d2m,
    musicbrainz_block_d2m,
    text_cols=["name", "artist", "release-date"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# musicbrainz_lastfm: semantic_similarity on ["name","artist","duration"] top_k=20
musicbrainz_block_m2l = musicbrainz.copy()
lastfm_block_m2l = lastfm.copy()

musicbrainz_block_m2l["artist"] = musicbrainz_block_m2l["_artist_from_name"]
lastfm_block_m2l["artist"] = lastfm_block_m2l["_artist_from_name"]

musicbrainz_block_m2l["name"] = (
    musicbrainz_block_m2l["_title_from_name"].fillna("").astype(str).map(lower_strip)
    + " "
    + musicbrainz_block_m2l["_tracks_text"].fillna("").astype(str).map(lower_strip)
).str.strip()
lastfm_block_m2l["name"] = (
    lastfm_block_m2l["_title_from_name"].fillna("").astype(str).map(lower_strip)
    + " "
    + lastfm_block_m2l["_tracks_text"].fillna("").astype(str).map(lower_strip)
).str.strip()

# duration is numeric; cast to string for embedding text usage without changing column name
def duration_to_embed_text(x):
    if is_missing_scalar(x) or pd.isna(x):
        return ""
    try:
        return str(int(round(float(x))))
    except Exception:
        return ""


musicbrainz_block_m2l["duration"] = musicbrainz_block_m2l["duration"].apply(duration_to_embed_text)
lastfm_block_m2l["duration"] = lastfm_block_m2l["duration"].apply(duration_to_embed_text)

blocker_m2l = EmbeddingBlocker(
    musicbrainz_block_m2l,
    lastfm_block_m2l,
    text_cols=["name", "artist", "duration"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching (MUST use provided comparator config)
# --------------------------------

comparators_d2l = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lambda x: lower_strip(x),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lambda x: lower_strip(extract_title_from_name(x)),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="tracks_track_name",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=120.0,
    ),
]

comparators_d2m = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lambda x: lower_strip(extract_title_from_name(x)),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lambda x: lower_strip(x),
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
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=120.0,
    ),
]

comparators_m2l = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lambda x: lower_strip(x),
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lambda x: lower_strip(extract_title_from_name(x)),
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=15.0,
    ),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_d2l = rb_matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_d2l,
    comparators=comparators_d2l,
    weights=[0.35, 0.35, 0.2, 0.1],
    threshold=0.55,
    id_column="id",
)

rb_correspondences_d2m = rb_matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_d2m,
    comparators=comparators_d2m,
    weights=[0.32, 0.28, 0.15, 0.15, 0.1],
    threshold=0.78,
    id_column="id",
)

rb_correspondences_m2l = rb_matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_m2l,
    comparators=comparators_m2l,
    weights=[0.35, 0.45, 0.2],
    threshold=0.75,
    id_column="id",
)

# Save correspondences with correct filenames (do not change names)
CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_d2l.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)
rb_correspondences_d2m.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)
rb_correspondences_m2l.to_csv(
    os.path.join(CORR_DIR, "correspondences_musicbrainz_lastfm.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_d2l, rb_correspondences_d2m, rb_correspondences_m2l],
    ignore_index=True,
)

# --------------------------------
# Data Fusion (improve track fields + name/artist/country normalization)
# --------------------------------


def prefer_non_empty_string(series: pd.Series):
    vals = []
    for v in series.tolist():
        if is_missing_scalar(v):
            continue
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", "null"}:
            vals.append(s)
    if not vals:
        return np.nan
    # prefer longer, but stable
    return sorted(vals, key=lambda x: (len(x), x), reverse=True)[0]


def prefer_title_without_artist_prefix(series: pd.Series):
    """
    For 'name' attribute: prefer a title-like value (without 'Artist - ' prefix).
    """
    candidates = []
    for v in series.tolist():
        if is_missing_scalar(v):
            continue
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            continue
        title = extract_title_from_name(s)
        # keep original capitalization if no prefix was present
        if " - " in s:
            candidates.append(title)
        else:
            candidates.append(s)
    if not candidates:
        return np.nan
    return sorted(candidates, key=lambda x: (len(str(x)), str(x)), reverse=True)[0]


def prefer_artist_field(series: pd.Series):
    """
    For 'artist' attribute: prefer explicit artist; fallback to extracted from 'name' if needed.
    """
    vals = []
    for v in series.tolist():
        if is_missing_scalar(v):
            continue
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", "null"}:
            vals.append(s)
    if not vals:
        return np.nan
    return sorted(vals, key=lambda x: (len(x), x), reverse=True)[0]


def fuse_release_country(series: pd.Series):
    vals = []
    for v in series.tolist():
        v2 = normalize_release_country(v)
        if is_missing_scalar(v2) or (isinstance(v2, float) and np.isnan(v2)):
            continue
        s = str(v2).strip()
        if s:
            vals.append(s)
    if not vals:
        return np.nan
    # voting then longest
    counts = {}
    for s in vals:
        counts[s] = counts.get(s, 0) + 1
    best_count = max(counts.values())
    tied = [k for k, c in counts.items() if c == best_count]
    return sorted(tied, key=lambda x: (len(x), x), reverse=True)[0]


def fuse_track_names(series: pd.Series):
    # choose the most complete canonical list; tie-breaker by total char length
    candidates = []
    for v in series.tolist():
        v = canonicalize_track_names(v)
        if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, list) and len(v) > 0:
            candidates.append(v)
    if not candidates:
        return np.nan
    best = sorted(candidates, key=lambda x: (len(x), sum(len(t) for t in x)), reverse=True)[0]
    # return as a stringified list to match typical CSV expectation and improve evaluator compatibility
    return str(best)


def fuse_track_positions(series: pd.Series):
    candidates = []
    for v in series.tolist():
        v = normalize_track_positions(v)
        if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, list) and len(v) > 0:
            out = []
            for p in v:
                try:
                    out.append(str(int(str(p).strip())))
                except Exception:
                    out.append(str(p).strip())
            candidates.append(out)
    if not candidates:
        return np.nan
    best = sorted(candidates, key=lambda x: len(x), reverse=True)[0]
    return str(best)


def fuse_track_durations(series: pd.Series):
    candidates = []
    for v in series.tolist():
        v = normalize_track_durations(v)
        if is_missing_scalar(v) or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, list) and len(v) > 0:
            candidates.append(v)
    if not candidates:
        return np.nan
    best = sorted(candidates, key=lambda x: (len(x), sum(x)), reverse=True)[0]
    return str(best)


strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", prefer_title_without_artist_prefix)
strategy.add_attribute_fuser("artist", prefer_artist_field)
strategy.add_attribute_fuser("release-date", prefer_non_empty_string)
strategy.add_attribute_fuser("release-country", fuse_release_country)
strategy.add_attribute_fuser("duration", maximum)
strategy.add_attribute_fuser("label", prefer_non_empty_string)
strategy.add_attribute_fuser("genre", prefer_non_empty_string)

strategy.add_attribute_fuser("tracks_track_name", fuse_track_names)
strategy.add_attribute_fuser("tracks_track_position", fuse_track_positions)
strategy.add_attribute_fuser("tracks_track_duration", fuse_track_durations)

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

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 05 END
============================================================

