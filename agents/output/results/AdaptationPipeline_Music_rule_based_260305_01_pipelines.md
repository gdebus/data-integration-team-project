# Pipeline Snapshots

notebook_name=AdaptationPipeline_Music
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=3
node_name=execute_pipeline
accuracy_score=40.10%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_xml

from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union, maximum

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "input/datasets/music/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets (XML)
musicbrainz = load_xml(
    DATA_DIR + "musicbrainz.xml",
    name="musicbrainz",
)

discogs = load_xml(
    DATA_DIR + "discogs.xml",
    name="discogs",
)

lastfm = load_xml(
    DATA_DIR + "lastfm.xml",
    name="lastfm",
)

datasets = [musicbrainz, discogs, lastfm]

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# The resulting columns for all datasets will have the schema of dataset1 (musicbrainz).
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

# Match schema of musicbrainz with discogs, rename discogs to musicbrainz schema
schema_correspondences = schema_matcher.match(musicbrainz, discogs)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
discogs = discogs.rename(columns=rename_map)

# Match schema of musicbrainz with lastfm, rename lastfm to musicbrainz schema
schema_correspondences = schema_matcher.match(musicbrainz, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map)

# --------------------------------
# Blocking (MANDATORY POLICY)
# Use informative identity signals: release "name" + "artist" (strong cross-source identity keys).
# Do NOT block on internal ids. Duration/release-date can be missing/noisy, so avoid as blocker.
# --------------------------------

print("Performing Blocking")

blocker_mb_dg = StandardBlocker(
    musicbrainz, discogs,
    on=["name", "artist"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_mb_lf = StandardBlocker(
    musicbrainz, lastfm,
    on=["name", "artist"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_dg_lf = StandardBlocker(
    discogs, lastfm,
    on=["name", "artist"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching Configuration (rule-based)
# --------------------------------

comparators_mb_dg = [
    StringComparator(
        column="name",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="artist",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="release-date",
        similarity_function="jaccard",
        preprocess=lambda x: str(x)[:10].lower(),  # robust for YYYY-MM-DD or missing
    ),
    NumericComparator(
        column="duration",
        max_difference=120,  # seconds tolerance
    ),
]

comparators_mb_lf = [
    StringComparator(
        column="name",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="artist",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    NumericComparator(
        column="duration",
        max_difference=180,  # lastfm can differ; allow wider tolerance
    ),
]

comparators_dg_lf = [
    StringComparator(
        column="name",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    StringComparator(
        column="artist",
        similarity_function="jaccard",
        preprocess=str.lower,
    ),
    NumericComparator(
        column="duration",
        max_difference=180,
    ),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_mb_dg = rb_matcher.match(
    df_left=musicbrainz,
    df_right=discogs,
    candidates=blocker_mb_dg,
    comparators=comparators_mb_dg,
    weights=[0.5, 0.3, 0.1, 0.1],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_mb_lf = rb_matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_mb_lf,
    comparators=comparators_mb_lf,
    weights=[0.55, 0.35, 0.10],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_dg_lf = rb_matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_dg_lf,
    comparators=comparators_dg_lf,
    weights=[0.55, 0.35, 0.10],
    threshold=0.75,
    id_column="id",
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_mb_dg, rb_correspondences_mb_lf, rb_correspondences_dg_lf],
    ignore_index=True
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# Prefer more complete strings for descriptive attributes
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)

# Dates/country often differ in formatting; keep the longest string (usually most explicit)
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)

# Duration: choose maximum to avoid zeros (discogs has 0 in sample) and truncated values
strategy.add_attribute_fuser("duration", maximum)

# Lists: union for track-related arrays and multi-valued attributes
strategy.add_attribute_fuser("tracks_track_name", union)
strategy.add_attribute_fuser("tracks_track_duration", union)
strategy.add_attribute_fuser("tracks_track_position", union)

# Extra discogs attributes may exist after schema match depending on mapping; fuse if present
strategy.add_attribute_fuser("label", longest_string)
strategy.add_attribute_fuser("genre", union)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[musicbrainz, discogs, lastfm],
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
node_index=8
node_name=execute_pipeline
accuracy_score=46.70%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_xml

from PyDI.entitymatching import SortedNeighbourhoodBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
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
import re
import ast
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/music/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

musicbrainz = load_xml(DATA_DIR + "musicbrainz.xml", name="musicbrainz")
discogs = load_xml(DATA_DIR + "discogs.xml", name="discogs")
lastfm = load_xml(DATA_DIR + "lastfm.xml", name="lastfm")

# --------------------------------
# Helpers: normalization & safe parsing
# --------------------------------

def _is_null(x):
    return x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip().lower() in {"", "nan", "none", "null"}

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def normalize_title(s):
    if _is_null(s):
        return ""
    s = str(s)
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = normalize_whitespace(s)
    s = s.lower()
    s = re.sub(r"[^\w\s/-]", " ", s)   # keep / and - which are common in releases
    s = normalize_whitespace(s)
    return s

def normalize_artist(s):
    if _is_null(s):
        return ""
    s = str(s)
    s = normalize_whitespace(s)
    s = s.lower()
    s = re.sub(r"[^\w\s&+']", " ", s)
    s = normalize_whitespace(s)
    return s

def normalize_release_name_with_artist(row_name, row_artist):
    """
    Fix Last.fm pattern: 'Artist - Title' present in name field.
    If artist exists, strip leading 'artist - ' from name.
    """
    name = "" if _is_null(row_name) else str(row_name)
    artist = "" if _is_null(row_artist) else str(row_artist)

    n = normalize_whitespace(name)
    a = normalize_whitespace(artist)
    if n and a:
        # remove "artist - " prefix (allow multiple spaces)
        pattern = r"^\s*" + re.escape(a) + r"\s*-\s*"
        n2 = re.sub(pattern, "", n, flags=re.IGNORECASE)
        if n2 and n2 != n:
            return n2
    return n

def parse_maybe_list(x):
    """
    XML loader often yields python lists, but sometimes strings like "['a','b']" or "a;b".
    """
    if _is_null(x):
        return []
    if isinstance(x, list):
        return [str(v) for v in x if not _is_null(v)]
    if isinstance(x, tuple):
        return [str(v) for v in x if not _is_null(v)]
    s = str(x).strip()
    # try python literal list
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(i) for i in v if not _is_null(i)]
        except Exception:
            pass
    # fallback: split on common separators
    parts = re.split(r"\s*;\s*|\s*,\s*|\s*\|\s*", s)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def tracklist_signature(names, positions):
    """
    Produce a stable signature for blocking/matching: '1:fermats theorem|2:sight beyond'
    (normalized, position-aware). Helps fix tracks_* accuracy (was 0.0 due to union/ordering).
    """
    n_list = parse_maybe_list(names)
    p_list = parse_maybe_list(positions)
    # if positions missing, use sequential
    if len(p_list) != len(n_list) or len(p_list) == 0:
        p_list = [str(i + 1) for i in range(len(n_list))]
    pairs = []
    for p, n in zip(p_list, n_list):
        p = re.sub(r"[^\d]+", "", str(p)) or str(p)
        nn = normalize_title(n)
        if nn:
            pairs.append((int(p) if p.isdigit() else 999999, f"{p}:{nn}"))
    pairs.sort(key=lambda t: t[0])
    return "|".join([t[1] for t in pairs])

def normalize_country(s):
    if _is_null(s):
        return ""
    s = str(s).strip()
    mapping = {
        "uk": "united kingdom",
        "u.k.": "united kingdom",
        "united kingdom of great britain and northern ireland": "united kingdom",
        "united states": "united states",
        "usa": "united states",
        "u.s.a.": "united states",
        "us": "united states",
    }
    k = normalize_whitespace(s).lower()
    return mapping.get(k, k)

def normalize_date(s):
    if _is_null(s):
        return ""
    s = str(s).strip()
    # keep YYYY-MM-DD if present, else YYYY-MM, else YYYY
    m = re.match(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", s)
    if not m:
        return ""
    y, mo, d = m.group(1), m.group(2), m.group(3)
    if y and mo and d:
        return f"{y}-{mo}-{d}"
    if y and mo:
        return f"{y}-{mo}"
    return y

def to_int_safe(x):
    if _is_null(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r"[^\d\-\.]", "", s)
    if s == "":
        return np.nan
    try:
        return int(float(s))
    except Exception:
        return np.nan

# --------------------------------
# Schema Matching (LLM-based matching)
# Target schema: musicbrainz
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(model="gpt-5.1", temperature=0, max_tokens=None)

schema_matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = schema_matcher.match(musicbrainz, discogs)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
discogs = discogs.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(musicbrainz, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map)

# --------------------------------
# Feature engineering (improve matching + fusion correctness)
# - name_norm / artist_norm for robust comparison
# - track_sig for ordering-aware track fusion (fixes 0.0 track name accuracy)
# - date_norm / country_norm for stable formatting
# - duration_int numeric handling (discogs has "0")
# --------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # fix lastfm-like "Artist - Title" in name using artist column when present
    df["name_clean"] = df.apply(lambda r: normalize_release_name_with_artist(r.get("name", ""), r.get("artist", "")), axis=1)

    df["name_norm"] = df["name_clean"].apply(normalize_title)
    df["artist_norm"] = df["artist"].apply(normalize_artist) if "artist" in df.columns else ""

    if "release-date" in df.columns:
        df["release_date_norm"] = df["release-date"].apply(normalize_date)
    else:
        df["release_date_norm"] = ""

    if "release-country" in df.columns:
        df["release_country_norm"] = df["release-country"].apply(normalize_country)
    else:
        df["release_country_norm"] = ""

    if "duration" in df.columns:
        df["duration_int"] = df["duration"].apply(to_int_safe)
        # treat 0 as missing (discogs sample shows 0 often)
        df.loc[df["duration_int"] == 0, "duration_int"] = np.nan
    else:
        df["duration_int"] = np.nan

    # track signature for better matching + later ordered fusion
    df["track_sig"] = df.apply(
        lambda r: tracklist_signature(r.get("tracks_track_name", np.nan), r.get("tracks_track_position", np.nan)),
        axis=1,
    )

    return df

musicbrainz = add_features(musicbrainz)
discogs = add_features(discogs)
lastfm = add_features(lastfm)

# --------------------------------
# Blocking (MANDATORY POLICY)
# Use canonical title/name + artist as identity signal.
# SortedNeighborhood is more robust than exact Standard blocking given formatting noise.
# --------------------------------

print("Performing Blocking")

# key uses normalized name+artist; keep it deterministic and compact
for df in (musicbrainz, discogs, lastfm):
    df["block_key"] = (df["artist_norm"].fillna("") + " " + df["name_norm"].fillna("")).apply(normalize_whitespace)

blocker_mb_dg = SortedNeighbourhoodBlocker(
    musicbrainz, discogs,
    key="block_key",
    window=25,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_mb_lf = SortedNeighbourhoodBlocker(
    musicbrainz, lastfm,
    key="block_key",
    window=25,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_dg_lf = SortedNeighbourhoodBlocker(
    discogs, lastfm,
    key="block_key",
    window=25,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# Matching Configuration (rule-based)
# - compare normalized name/artist
# - include track signature similarity to improve alignment and track attribute correctness
# - include date/country (normalized) lightly
# - duration numeric with tolerance (seconds)
# --------------------------------

comparators_mb_dg = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="artist_norm", similarity_function="jaccard"),
    StringComparator(column="track_sig", similarity_function="jaccard"),
    StringComparator(column="release_date_norm", similarity_function="jaccard"),
    StringComparator(column="release_country_norm", similarity_function="jaccard"),
    NumericComparator(column="duration_int", max_difference=180),
]

comparators_mb_lf = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="artist_norm", similarity_function="jaccard"),
    StringComparator(column="track_sig", similarity_function="jaccard"),
    NumericComparator(column="duration_int", max_difference=240),
]

comparators_dg_lf = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="artist_norm", similarity_function="jaccard"),
    StringComparator(column="track_sig", similarity_function="jaccard"),
    NumericComparator(column="duration_int", max_difference=240),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_mb_dg = rb_matcher.match(
    df_left=musicbrainz,
    df_right=discogs,
    candidates=blocker_mb_dg,
    comparators=comparators_mb_dg,
    weights=[0.38, 0.28, 0.18, 0.06, 0.05, 0.05],
    threshold=0.72,
    id_column="id",
)

rb_correspondences_mb_lf = rb_matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_mb_lf,
    comparators=comparators_mb_lf,
    weights=[0.40, 0.30, 0.20, 0.10],
    threshold=0.72,
    id_column="id",
)

rb_correspondences_dg_lf = rb_matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_dg_lf,
    comparators=comparators_dg_lf,
    weights=[0.40, 0.30, 0.20, 0.10],
    threshold=0.72,
    id_column="id",
)

# --------------------------------
# Data Fusion
# Fix track attribute errors by fusing *ordered* track lists from the best available source.
# Approach:
# 1) Keep union fusers as baseline
# 2) Post-process fused output to choose track list from the record with most complete track_sig
#    within each fused cluster is not directly accessible here, so we approximate by:
#    - build "tracks_track_*" from track_sig if present in fused row
#    - else keep union result
# --------------------------------

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_mb_dg, rb_correspondences_mb_lf, rb_correspondences_dg_lf],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# Use cleaned/original display fields but prefer more informative strings
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)

strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)

# duration: avoid discogs 0, maximize among non-null; we already mapped 0->nan in duration_int
strategy.add_attribute_fuser("duration", maximum)

# track lists: do NOT union (union destroys order and can introduce duplicates)
# Instead, keep the longest string representation when available after reconstruction.
# We still keep union as fallback if present as real lists.
strategy.add_attribute_fuser("tracks_track_name", longest_string)
strategy.add_attribute_fuser("tracks_track_duration", longest_string)
strategy.add_attribute_fuser("tracks_track_position", longest_string)

# discogs extras
strategy.add_attribute_fuser("label", longest_string)
strategy.add_attribute_fuser("genre", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[musicbrainz, discogs, lastfm],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# --------------------------------
# Post-fusion cleanup:
# - normalize name for Last.fm style if still present
# - reconstruct ordered track lists from track_sig when it exists in any source-derived fields
#   (we kept engineered cols only in source dfs; they won't be in fused output unless present in schema.
#   So we reconstruct using existing tracks_* columns: if they are strings/lists, normalize to lists.)
# - de-duplicate track names while preserving order
# --------------------------------

def ensure_list_column(df, col):
    if col not in df.columns:
        df[col] = [[] for _ in range(len(df))]
        return df
    out = []
    for v in df[col].values:
        lst = parse_maybe_list(v)
        out.append(lst)
    df[col] = out
    return df

rb_fused_standard_blocker = rb_fused_standard_blocker.copy()

# Clean possible "Artist - Title" leftovers in fused name
if "name" in rb_fused_standard_blocker.columns and "artist" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["name"] = rb_fused_standard_blocker.apply(
        lambda r: normalize_release_name_with_artist(r.get("name", ""), r.get("artist", "")),
        axis=1,
    )

# Normalize track columns to proper lists and align lengths if possible
rb_fused_standard_blocker = ensure_list_column(rb_fused_standard_blocker, "tracks_track_name")
rb_fused_standard_blocker = ensure_list_column(rb_fused_standard_blocker, "tracks_track_position")
rb_fused_standard_blocker = ensure_list_column(rb_fused_standard_blocker, "tracks_track_duration")

def dedup_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        k = normalize_title(x)
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out

# If positions exist, sort by numeric position to preserve original order
def sort_by_position(names, positions, durations):
    if not names:
        return names, positions, durations
    if not positions or len(positions) != len(names):
        positions = [str(i + 1) for i in range(len(names))]
    # pad durations
    if not durations or len(durations) != len(names):
        durations = (durations or []) + [""] * (len(names) - len(durations or []))
    triples = []
    for n, p, d in zip(names, positions, durations):
        pnum = re.sub(r"[^\d]+", "", str(p))
        pval = int(pnum) if pnum.isdigit() else 999999
        triples.append((pval, str(p), n, d))
    triples.sort(key=lambda t: t[0])
    return [t[2] for t in triples], [t[1] for t in triples], [t[3] for t in triples]

for idx, row in rb_fused_standard_blocker.iterrows():
    names = parse_maybe_list(row.get("tracks_track_name", []))
    pos = parse_maybe_list(row.get("tracks_track_position", []))
    dur = parse_maybe_list(row.get("tracks_track_duration", []))

    names = dedup_preserve_order(names)
    names, pos, dur = sort_by_position(names, pos, dur)

    rb_fused_standard_blocker.at[idx, "tracks_track_name"] = names
    rb_fused_standard_blocker.at[idx, "tracks_track_position"] = pos
    rb_fused_standard_blocker.at[idx, "tracks_track_duration"] = dur

# write output
rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=13
node_name=execute_pipeline
accuracy_score=42.13%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_xml

from PyDI.entitymatching import SortedNeighbourhoodBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    shortest_string,
    union,
    maximum,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
import re
import ast
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "input/datasets/music/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

musicbrainz = load_xml(DATA_DIR + "musicbrainz.xml", name="musicbrainz")
discogs = load_xml(DATA_DIR + "discogs.xml", name="discogs")
lastfm = load_xml(DATA_DIR + "lastfm.xml", name="lastfm")

# --------------------------------
# Helpers
# --------------------------------

def _is_null(x):
    return (
        x is None
        or (isinstance(x, float) and np.isnan(x))
        or str(x).strip().lower() in {"", "nan", "none", "null"}
    )

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def normalize_title(s):
    if _is_null(s):
        return ""
    s = str(s).replace("\u2013", "-").replace("\u2014", "-")
    s = normalize_whitespace(s).lower()
    # drop bracketed qualifiers like "(remastered)", "[live]" to reduce noise for identity match
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"[^\w\s/-]", " ", s)  # keep / and -
    s = normalize_whitespace(s)
    return s

def normalize_artist(s):
    if _is_null(s):
        return ""
    s = normalize_whitespace(s).lower()
    # normalize common joiners
    s = s.replace(" feat. ", " ft ").replace(" featuring ", " ft ").replace(" & ", " and ")
    s = re.sub(r"[^\w\s&+']", " ", s)
    s = normalize_whitespace(s)
    return s

def normalize_release_name_with_artist(row_name, row_artist):
    """
    Fix Last.fm pattern: 'Artist - Title' in name.
    Also handle variants like 'Artist — Title' and multiple spaces.
    """
    name = "" if _is_null(row_name) else str(row_name)
    artist = "" if _is_null(row_artist) else str(row_artist)

    n = normalize_whitespace(name)
    a = normalize_whitespace(artist)

    if n and a:
        # allow -, – or — as separator
        pattern = r"^\s*" + re.escape(a) + r"\s*[-\u2013\u2014]\s*"
        n2 = re.sub(pattern, "", n, flags=re.IGNORECASE)
        if n2 and n2 != n:
            return n2
    return n

def parse_maybe_list(x):
    if _is_null(x):
        return []
    if isinstance(x, list):
        return [str(v) for v in x if not _is_null(v)]
    if isinstance(x, tuple):
        return [str(v) for v in x if not _is_null(v)]
    s = str(x).strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(i) for i in v if not _is_null(i)]
        except Exception:
            pass
    parts = re.split(r"\s*;\s*|\s*\|\s*", s)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def normalize_date(s):
    if _is_null(s):
        return ""
    s = str(s).strip()
    m = re.match(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", s)
    if not m:
        return ""
    y, mo, d = m.group(1), m.group(2), m.group(3)
    if y and mo and d:
        return f"{y}-{mo}-{d}"
    if y and mo:
        return f"{y}-{mo}"
    return y

def normalize_country(s):
    if _is_null(s):
        return ""
    k = normalize_whitespace(s).lower()
    mapping = {
        "uk": "united kingdom",
        "u.k.": "united kingdom",
        "united kingdom of great britain and northern ireland": "united kingdom",
        "gb": "united kingdom",
        "great britain": "united kingdom",
        "united states": "united states",
        "usa": "united states",
        "u.s.a.": "united states",
        "us": "united states",
    }
    return mapping.get(k, k)

def to_int_safe(x):
    if _is_null(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r"[^\d\-\.]", "", s)
    if s == "":
        return np.nan
    try:
        return int(float(s))
    except Exception:
        return np.nan

def safe_jaccard_token_set(s):
    s = normalize_title(s)
    if not s:
        return set()
    toks = [t for t in re.split(r"\s+|/|-", s) if t]
    # drop very short tokens (noise)
    toks = [t for t in toks if len(t) >= 2]
    return set(toks)

def tracklist_signature(names, positions):
    """
    Stable, order-aware signature for matching:
    '1:fermats theorem|2:sight beyond'
    """
    n_list = parse_maybe_list(names)
    p_list = parse_maybe_list(positions)

    if len(p_list) != len(n_list) or len(p_list) == 0:
        p_list = [str(i + 1) for i in range(len(n_list))]

    pairs = []
    for p, n in zip(p_list, n_list):
        p_raw = str(p)
        p_num = re.sub(r"[^\d]+", "", p_raw)
        p_val = int(p_num) if p_num.isdigit() else 999999
        nn = normalize_title(n)
        if nn:
            pairs.append((p_val, f"{p_num if p_num else p_raw}:{nn}"))
    pairs.sort(key=lambda t: t[0])
    return "|".join([t[1] for t in pairs])

def ensure_list_column(df, col):
    if col not in df.columns:
        df[col] = [[] for _ in range(len(df))]
        return df
    df[col] = [parse_maybe_list(v) for v in df[col].values]
    return df

def dedup_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        k = normalize_title(x)
        if k and k not in seen:
            seen.add(k)
            out.append(str(x))
    return out

def sort_by_position(names, positions, durations):
    if not names:
        return names, positions, durations
    if not positions or len(positions) != len(names):
        positions = [str(i + 1) for i in range(len(names))]
    if not durations or len(durations) != len(names):
        durations = (durations or []) + [""] * (len(names) - len(durations or []))
    triples = []
    for n, p, d in zip(names, positions, durations):
        pnum = re.sub(r"[^\d]+", "", str(p))
        pval = int(pnum) if pnum.isdigit() else 999999
        triples.append((pval, str(p), str(n), str(d)))
    triples.sort(key=lambda t: t[0])
    return [t[2] for t in triples], [t[1] for t in triples], [t[3] for t in triples]

def build_best_tracklists(names, positions, durations):
    """
    Choose a coherent tracklist:
    - prefer source with aligned (name, position) lengths and more items
    - ensure order by numeric positions
    - de-duplicate names
    """
    n = dedup_preserve_order(parse_maybe_list(names))
    p = parse_maybe_list(positions)
    d = parse_maybe_list(durations)

    n, p, d = sort_by_position(n, p, d)

    # If positions length still mismatched, rebuild sequential to avoid evaluation mismatches
    if n and (len(p) != len(n)):
        p = [str(i + 1) for i in range(len(n))]
    if n and (len(d) != len(n)):
        d = (d or []) + [""] * (len(n) - len(d or []))
    return n, p, d

def clean_release_name(name, artist):
    return normalize_release_name_with_artist(name, artist)

# --------------------------------
# Schema Matching (LLM-based matching)
# Target schema: musicbrainz
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(model="gpt-5.1", temperature=0, max_tokens=None)
schema_matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = schema_matcher.match(musicbrainz, discogs)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
discogs = discogs.rename(columns=rename_map)

schema_correspondences = schema_matcher.match(musicbrainz, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map)

# --------------------------------
# Feature engineering for better matching
# --------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["name_clean"] = df.apply(
        lambda r: clean_release_name(r.get("name", ""), r.get("artist", "")),
        axis=1,
    )
    df["name_norm"] = df["name_clean"].apply(normalize_title)
    df["artist_norm"] = df["artist"].apply(normalize_artist) if "artist" in df.columns else ""

    df["release_date_norm"] = df["release-date"].apply(normalize_date) if "release-date" in df.columns else ""
    df["release_country_norm"] = df["release-country"].apply(normalize_country) if "release-country" in df.columns else ""

    df["duration_int"] = df["duration"].apply(to_int_safe) if "duration" in df.columns else np.nan
    # treat 0 as missing
    df.loc[df["duration_int"] == 0, "duration_int"] = np.nan

    df["track_sig"] = df.apply(
        lambda r: tracklist_signature(r.get("tracks_track_name", np.nan), r.get("tracks_track_position", np.nan)),
        axis=1,
    )

    # for blocking: compact key with reduced noise, keep only a few tokens from title to avoid overly long keys
    def block_key_row(r):
        a = r.get("artist_norm", "")
        t = r.get("name_norm", "")
        t_toks = list(safe_jaccard_token_set(t))
        t_toks = sorted(t_toks)[:8]
        return normalize_whitespace(a + " " + " ".join(t_toks))

    df["block_key"] = df.apply(block_key_row, axis=1)
    return df

musicbrainz = add_features(musicbrainz)
discogs = add_features(discogs)
lastfm = add_features(lastfm)

# --------------------------------
# Blocking (MANDATORY POLICY)
# Use canonical identity signals: artist + title tokens
# --------------------------------

print("Performing Blocking")

blocker_mb_dg = SortedNeighbourhoodBlocker(
    musicbrainz,
    discogs,
    key="block_key",
    window=40,  # increase recall (prior accuracy suggests missed/incorrect links)
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_mb_lf = SortedNeighbourhoodBlocker(
    musicbrainz,
    lastfm,
    key="block_key",
    window=40,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_dg_lf = SortedNeighbourhoodBlocker(
    discogs,
    lastfm,
    key="block_key",
    window=40,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# Matching (rule-based)
# Key improvements:
# - strengthen name+artist (higher weights)
# - keep track_sig but moderate weight (missingness in lastfm tracks columns)
# - use release_date/country only as light signals (formatting differences)
# - duration tolerance larger due to dataset disagreement
# --------------------------------

comparators_mb_dg = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="artist_norm", similarity_function="jaccard"),
    StringComparator(column="track_sig", similarity_function="jaccard"),
    StringComparator(column="release_date_norm", similarity_function="jaccard"),
    StringComparator(column="release_country_norm", similarity_function="jaccard"),
    NumericComparator(column="duration_int", max_difference=300),
]

comparators_mb_lf = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="artist_norm", similarity_function="jaccard"),
    StringComparator(column="track_sig", similarity_function="jaccard"),
    NumericComparator(column="duration_int", max_difference=360),
]

comparators_dg_lf = [
    StringComparator(column="name_norm", similarity_function="jaccard"),
    StringComparator(column="artist_norm", similarity_function="jaccard"),
    StringComparator(column="track_sig", similarity_function="jaccard"),
    NumericComparator(column="duration_int", max_difference=360),
]

print("Matching Entities")

rb_matcher = RuleBasedMatcher()

rb_correspondences_mb_dg = rb_matcher.match(
    df_left=musicbrainz,
    df_right=discogs,
    candidates=blocker_mb_dg,
    comparators=comparators_mb_dg,
    weights=[0.44, 0.32, 0.12, 0.05, 0.03, 0.04],
    threshold=0.75,  # reduce false positives -> improves attribute accuracies
    id_column="id",
)

rb_correspondences_mb_lf = rb_matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_mb_lf,
    comparators=comparators_mb_lf,
    weights=[0.48, 0.34, 0.10, 0.08],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_dg_lf = rb_matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_dg_lf,
    comparators=comparators_dg_lf,
    weights=[0.48, 0.34, 0.10, 0.08],
    threshold=0.75,
    id_column="id",
)

# --------------------------------
# Data Fusion
# Fix observed issues:
# - track_* accuracies low: avoid union; reconstruct consistent ordered lists post-fusion
# - duration errors: avoid picking "0" / missing; use maximum (with later cleanup)
# - release-country formatting: normalize post-fusion
# - name errors: post-fusion remove "Artist - " pattern again
# --------------------------------

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_mb_dg, rb_correspondences_mb_lf, rb_correspondences_dg_lf],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

# Prefer shorter name to avoid "Artist - Title" variants; then clean again post-fusion
strategy.add_attribute_fuser("name", shortest_string)
strategy.add_attribute_fuser("artist", longest_string)

strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)

strategy.add_attribute_fuser("duration", maximum)

# Keep as strings during fusion; fix as lists post-fusion deterministically
strategy.add_attribute_fuser("tracks_track_name", longest_string)
strategy.add_attribute_fuser("tracks_track_duration", longest_string)
strategy.add_attribute_fuser("tracks_track_position", longest_string)

# discogs-only-ish
strategy.add_attribute_fuser("label", longest_string)
strategy.add_attribute_fuser("genre", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[musicbrainz, discogs, lastfm],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# --------------------------------
# Post-fusion cleanup (deterministic, improves attribute accuracies)
# --------------------------------

rb_fused_standard_blocker = rb_fused_standard_blocker.copy()

# Clean name and normalize country/date formats
if "name" in rb_fused_standard_blocker.columns and "artist" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["name"] = rb_fused_standard_blocker.apply(
        lambda r: clean_release_name(r.get("name", ""), r.get("artist", "")),
        axis=1,
    )

if "release-country" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["release-country"] = rb_fused_standard_blocker["release-country"].apply(
        lambda x: normalize_country(x) if not _is_null(x) else x
    )

if "release-date" in rb_fused_standard_blocker.columns:
    rb_fused_standard_blocker["release-date"] = rb_fused_standard_blocker["release-date"].apply(
        lambda x: normalize_date(x) if not _is_null(x) else x
    )

# Fix duration: if it's "0" keep as missing
if "duration" in rb_fused_standard_blocker.columns:
    dur_int = rb_fused_standard_blocker["duration"].apply(to_int_safe)
    dur_int = dur_int.replace(0, np.nan)
    # write back as string seconds when present (keeps original schema as object)
    rb_fused_standard_blocker["duration"] = dur_int.apply(lambda v: "" if _is_null(v) else str(int(v)))

# Track list reconstruction
rb_fused_standard_blocker = ensure_list_column(rb_fused_standard_blocker, "tracks_track_name")
rb_fused_standard_blocker = ensure_list_column(rb_fused_standard_blocker, "tracks_track_position")
rb_fused_standard_blocker = ensure_list_column(rb_fused_standard_blocker, "tracks_track_duration")

for idx, row in rb_fused_standard_blocker.iterrows():
    n, p, d = build_best_tracklists(
        row.get("tracks_track_name", []),
        row.get("tracks_track_position", []),
        row.get("tracks_track_duration", []),
    )
    rb_fused_standard_blocker.at[idx, "tracks_track_name"] = n
    rb_fused_standard_blocker.at[idx, "tracks_track_position"] = p
    rb_fused_standard_blocker.at[idx, "tracks_track_duration"] = d

# write output
rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

