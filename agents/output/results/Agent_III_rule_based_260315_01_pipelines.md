# Pipeline Snapshots

notebook_name=Agent III & IV
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=8
node_name=execute_pipeline
accuracy_score=55.84%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.schemamatching import LLMBasedSchemaMatcher

from PyDI.entitymatching import (
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    StringComparator,
    NumericComparator,
    DateComparator,
    RuleBasedMatcher,
)

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
)

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def lower_strip(x):
    return str(x).lower().strip()


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
# Perform Schema Matching (LLM-based matching)
# The resulting columns for all datasets will have the schema of discogs.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(discogs, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map)

schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
musicbrainz = musicbrainz.rename(columns=rename_map)

# --------------------------------
# Perform Blocking using precomputed blocking configuration
# --------------------------------

print("Performing Blocking")

blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    window=15,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching using precomputed matching configuration
# --------------------------------

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

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=[0.3, 0.35, 0.25, 0.1],
    threshold=0.6,
    id_column="id",
)

rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=[0.35, 0.25, 0.2, 0.1, 0.1],
    threshold=0.7,
    id_column="id",
)

rb_correspondences_musicbrainz_lastfm = matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_musicbrainz_lastfm,
    comparators=comparators_musicbrainz_lastfm,
    weights=[0.35, 0.45, 0.2],
    threshold=0.75,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_discogs_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)

rb_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)

rb_correspondences_musicbrainz_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_musicbrainz_lastfm.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_musicbrainz_lastfm,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

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

os.makedirs("output/data_fusion", exist_ok=True)
rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=14
node_name=execute_pipeline
accuracy_score=59.39%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.schemamatching import LLMBasedSchemaMatcher

from PyDI.entitymatching import (
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    StringComparator,
    NumericComparator,
    DateComparator,
    RuleBasedMatcher,
)

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
)

import pandas as pd
import numpy as np
import ast
import re
from collections import Counter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def lower_strip(x):
    return str(x).lower().strip()


def is_missing(x):
    return pd.isna(x) or str(x).strip().lower() in {"", "nan", "none", "null"}


def parse_list_value(x):
    if is_missing(x):
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if not is_missing(v)]
    s = str(x).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if not is_missing(v)]
    except Exception:
        pass
    return [v.strip() for v in s.split("|") if v.strip()]


def normalize_whitespace(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def normalize_name_for_fusion(s):
    if is_missing(s):
        return None
    s = normalize_whitespace(s)
    s = re.sub(r"^\s*album\s*:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    if " - " in s:
        left, right = s.split(" - ", 1)
        if len(left.strip()) < len(right.strip()):
            s = right.strip()
    return s


def normalize_artist_for_fusion(s):
    if is_missing(s):
        return None
    s = normalize_whitespace(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


COUNTRY_MAP = {
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "united kingdom": "United Kingdom",
    "united kingdom of great britain and northern ireland": "United Kingdom",
    "great britain": "United Kingdom",
    "gb": "United Kingdom",
    "us": "United States",
    "u.s.": "United States",
    "usa": "United States",
    "u.s.a.": "United States",
    "united states": "United States",
    "united states of america": "United States",
}


def normalize_country(s):
    if is_missing(s):
        return None
    s = normalize_whitespace(s)
    key = s.lower()
    return COUNTRY_MAP.get(key, s)


def parse_duration_scalar(x):
    if is_missing(x):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        val = float(x)
        if pd.isna(val) or val <= 0:
            return None
        return int(round(val))
    s = str(x).strip()
    s = re.sub(r"[^\d.]", "", s)
    if not s:
        return None
    try:
        val = float(s)
        if val <= 0:
            return None
        return int(round(val))
    except Exception:
        return None


def canonical_string(s):
    if is_missing(s):
        return None
    return normalize_whitespace(str(s)).lower()


def vote_best(values, normalizer=None, prefer_longest=False):
    cleaned = []
    for v in values:
        if is_missing(v):
            continue
        vv = normalizer(v) if normalizer else v
        if is_missing(vv):
            continue
        cleaned.append(vv)
    if not cleaned:
        return None
    counts = Counter(cleaned)
    max_count = max(counts.values())
    candidates = [k for k, c in counts.items() if c == max_count]
    if len(candidates) == 1:
        return candidates[0]
    if prefer_longest:
        return sorted(candidates, key=lambda x: (-len(str(x)), str(x)))[0]
    return sorted(candidates, key=lambda x: (len(str(x)), str(x)))[0]


def choose_name(values):
    normalized_pairs = []
    for v in values:
        nv = normalize_name_for_fusion(v)
        if nv is not None:
            normalized_pairs.append((nv.lower(), nv))
    if not normalized_pairs:
        return None
    counts = Counter([k for k, _ in normalized_pairs])
    max_count = max(counts.values())
    candidate_keys = [k for k, c in counts.items() if c == max_count]
    if len(candidate_keys) == 1:
        key = candidate_keys[0]
    else:
        key = sorted(candidate_keys, key=lambda x: len(x))[0]
    originals = [orig for k, orig in normalized_pairs if k == key]
    return sorted(originals, key=lambda x: len(x))[0]


def choose_artist(values):
    normalized_pairs = []
    for v in values:
        nv = normalize_artist_for_fusion(v)
        if nv is not None:
            normalized_pairs.append((nv.lower(), nv))
    if not normalized_pairs:
        return None
    counts = Counter([k for k, _ in normalized_pairs])
    max_count = max(counts.values())
    candidate_keys = [k for k, c in counts.items() if c == max_count]
    key = sorted(candidate_keys, key=lambda x: len(x))[0]
    originals = [orig for k, orig in normalized_pairs if k == key]
    return sorted(originals, key=lambda x: len(x))[0]


def choose_release_country(values):
    return vote_best(values, normalizer=normalize_country, prefer_longest=False)


def choose_release_date(values):
    valid = []
    for v in values:
        if is_missing(v):
            continue
        dt = pd.to_datetime(v, errors="coerce")
        if pd.isna(dt):
            continue
        valid.append(dt.strftime("%Y-%m-%d"))
    if not valid:
        return None
    counts = Counter(valid)
    max_count = max(counts.values())
    candidates = [k for k, c in counts.items() if c == max_count]
    return sorted(candidates)[0]


def choose_duration(values):
    parsed = [parse_duration_scalar(v) for v in values]
    parsed = [v for v in parsed if v is not None]
    if not parsed:
        return None
    counts = Counter(parsed)
    max_count = max(counts.values())
    candidates = [k for k, c in counts.items() if c == max_count]
    if len(candidates) == 1:
        return candidates[0]
    return int(round(float(np.median(sorted(candidates)))))


def choose_label(values):
    return vote_best(values, normalizer=normalize_whitespace, prefer_longest=False)


def choose_genre(values):
    genre_values = []
    for v in values:
        if is_missing(v):
            continue
        parts = parse_list_value(v)
        if parts:
            genre_values.extend(parts)
        else:
            genre_values.append(normalize_whitespace(v))
    if not genre_values:
        return None
    counts = Counter([g.lower() for g in genre_values])
    best = sorted([k for k, c in counts.items() if c == max(counts.values())])[0]
    originals = [g for g in genre_values if g.lower() == best]
    return sorted(originals, key=lambda x: len(x))[0]


def score_tracklist(names, positions, durations):
    score = 0
    if names:
        score += len(names) * 3
    if positions:
        score += len(positions) * 2
    if durations:
        score += len([d for d in durations if parse_duration_scalar(d) is not None])
    return score


def choose_best_track_record(rows):
    best = None
    best_score = -1
    best_source_priority = -1
    source_priority = {"musicbrainz": 3, "discogs": 2, "lastfm": 1}

    for row in rows:
        names = parse_list_value(row.get("tracks_track_name"))
        positions = parse_list_value(row.get("tracks_track_position"))
        durations = parse_list_value(row.get("tracks_track_duration"))
        score = score_tracklist(names, positions, durations)
        src = row.get("_source_dataset", "")
        priority = source_priority.get(src, 0)
        if score > best_score or (score == best_score and priority > best_source_priority):
            best = {
                "tracks_track_name": names,
                "tracks_track_position": positions,
                "tracks_track_duration": durations,
            }
            best_score = score
            best_source_priority = priority
    return best


def format_list_output(lst):
    if not lst:
        return None
    return str(lst)


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
# Perform Schema Matching (LLM-based matching)
# The resulting columns for all datasets will have the schema of discogs.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(discogs, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map)

schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
musicbrainz = musicbrainz.rename(columns=rename_map)

# keep source provenance for post-fusion refinement
discogs = discogs.copy()
lastfm = lastfm.copy()
musicbrainz = musicbrainz.copy()

discogs["_source_dataset"] = "discogs"
lastfm["_source_dataset"] = "lastfm"
musicbrainz["_source_dataset"] = "musicbrainz"

# --------------------------------
# Perform Blocking using precomputed blocking configuration
# --------------------------------

print("Performing Blocking")

blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    window=15,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching using precomputed matching configuration
# --------------------------------

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

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=[0.3, 0.35, 0.25, 0.1],
    threshold=0.6,
    id_column="id",
)

rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=[0.35, 0.25, 0.2, 0.1, 0.1],
    threshold=0.7,
    id_column="id",
)

rb_correspondences_musicbrainz_lastfm = matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_musicbrainz_lastfm,
    comparators=comparators_musicbrainz_lastfm,
    weights=[0.35, 0.45, 0.2],
    threshold=0.75,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_discogs_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)

rb_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)

rb_correspondences_musicbrainz_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_musicbrainz_lastfm.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_musicbrainz_lastfm,
    ],
    ignore_index=True,
)

# Initial fusion strategy with conservative, vote-oriented attribute fusers
strategy = DataFusionStrategy("rule_based_fusion_strategy")
strategy.add_attribute_fuser("name", choose_name)
strategy.add_attribute_fuser("artist", choose_artist)
strategy.add_attribute_fuser("release-date", choose_release_date)
strategy.add_attribute_fuser("release-country", choose_release_country)
strategy.add_attribute_fuser("duration", choose_duration)
strategy.add_attribute_fuser("label", choose_label)
strategy.add_attribute_fuser("genre", choose_genre)
strategy.add_attribute_fuser("tracks_track_name", lambda values: None)
strategy.add_attribute_fuser("tracks_track_position", lambda values: None)
strategy.add_attribute_fuser("tracks_track_duration", lambda values: None)

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
    include_singletons=True,
)

# --------------------------------
# Post-fusion refinement:
# choose coherent track metadata from a single best source record per fused cluster
# and apply source-aware corrections favoring musicbrainz when available
# --------------------------------

all_records = pd.concat([discogs, lastfm, musicbrainz], ignore_index=True, sort=False)

fused = rb_fused_standard_blocker.copy()

cluster_member_cols = [c for c in fused.columns if c.endswith("_ids") or c.endswith("_id")]
source_id_columns = ["discogs_id", "lastfm_id", "musicbrainz_id"]

for col in source_id_columns:
    if col not in fused.columns:
        fused[col] = None

record_lookup = {}
for _, row in all_records.iterrows():
    record_lookup[row["id"]] = row.to_dict()

def collect_cluster_rows(fused_row):
    rows = []
    for src_col in source_id_columns:
        val = fused_row.get(src_col)
        if is_missing(val):
            continue
        values = parse_list_value(val) if str(val).startswith("[") else [val]
        for rid in values:
            if rid in record_lookup:
                rows.append(record_lookup[rid])
    if rows:
        return rows
    for col in cluster_member_cols:
        val = fused_row.get(col)
        if is_missing(val):
            continue
        values = parse_list_value(val) if str(val).startswith("[") else [val]
        for rid in values:
            if rid in record_lookup:
                rows.append(record_lookup[rid])
    return rows

for idx, row in fused.iterrows():
    cluster_rows = collect_cluster_rows(row)
    if not cluster_rows:
        continue

    mb_rows = [r for r in cluster_rows if r.get("_source_dataset") == "musicbrainz"]
    all_name_values = [r.get("name") for r in cluster_rows]
    all_artist_values = [r.get("artist") for r in cluster_rows]
    all_date_values = [r.get("release-date") for r in cluster_rows]
    all_country_values = [r.get("release-country") for r in cluster_rows]
    all_duration_values = [r.get("duration") for r in cluster_rows]
    all_label_values = [r.get("label") for r in cluster_rows if "label" in r]
    all_genre_values = [r.get("genre") for r in cluster_rows if "genre" in r]

    # Prefer musicbrainz-backed consensus when present
    if mb_rows:
        fused.at[idx, "name"] = choose_name([r.get("name") for r in mb_rows] + all_name_values)
        fused.at[idx, "artist"] = choose_artist([r.get("artist") for r in mb_rows] + all_artist_values)
        fused.at[idx, "release-date"] = choose_release_date([r.get("release-date") for r in mb_rows] + all_date_values)
        fused.at[idx, "release-country"] = choose_release_country([r.get("release-country") for r in mb_rows] + all_country_values)
        fused.at[idx, "duration"] = choose_duration([r.get("duration") for r in mb_rows] + all_duration_values)
    else:
        fused.at[idx, "name"] = choose_name(all_name_values)
        fused.at[idx, "artist"] = choose_artist(all_artist_values)
        fused.at[idx, "release-date"] = choose_release_date(all_date_values)
        fused.at[idx, "release-country"] = choose_release_country(all_country_values)
        fused.at[idx, "duration"] = choose_duration(all_duration_values)

    fused.at[idx, "label"] = choose_label(all_label_values)
    fused.at[idx, "genre"] = choose_genre(all_genre_values)

    best_track_record = choose_best_track_record(cluster_rows)
    if best_track_record:
        fused.at[idx, "tracks_track_name"] = format_list_output(best_track_record["tracks_track_name"])
        fused.at[idx, "tracks_track_position"] = format_list_output(best_track_record["tracks_track_position"])
        fused.at[idx, "tracks_track_duration"] = format_list_output(best_track_record["tracks_track_duration"])

os.makedirs("output/data_fusion", exist_ok=True)
fused.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=22
node_name=execute_pipeline
accuracy_score=62.94%
------------------------------------------------------------

```python
# -*- coding: utf-8 -*-
from PyDI.io import load_csv
from PyDI.schemamatching import LLMBasedSchemaMatcher

from PyDI.entitymatching import (
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    StringComparator,
    NumericComparator,
    DateComparator,
    RuleBasedMatcher,
)

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
)

import pandas as pd
import numpy as np
import ast
import re
from collections import Counter, defaultdict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def lower_strip(x):
    return str(x).lower().strip()


def is_missing(x):
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    return str(x).strip().lower() in {"", "nan", "none", "null"}


def parse_list_value(x):
    if is_missing(x):
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if not is_missing(v)]
    s = str(x).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if not is_missing(v)]
    except Exception:
        pass
    if "|" in s:
        return [v.strip() for v in s.split("|") if v.strip()]
    return [s] if s else []


def normalize_whitespace(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def normalize_text_key(s):
    if is_missing(s):
        return None
    s = normalize_whitespace(s).lower()
    s = s.replace("`", "'")
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"\bfeat\.?\b|\bft\.?\b|\bfeaturing\b", " feat ", s)
    s = re.sub(r"\bthe\s+([a-z0-9])", r"\1", s)
    s = re.sub(r"[^\w\s/&+\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_name_for_fusion(s):
    if is_missing(s):
        return None
    s = normalize_whitespace(s)
    s = re.sub(r"^\s*album\s*:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    m = re.match(r"^(.*?)\s+-\s+(.*)$", s)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        left_key = normalize_text_key(left)
        right_key = normalize_text_key(right)
        if left_key and right_key and len(left_key.split()) <= 4 and len(right_key.split()) > 1:
            s = right
    return s


def normalize_artist_for_fusion(s):
    if is_missing(s):
        return None
    s = normalize_whitespace(s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r",\s*the$", "", s, flags=re.IGNORECASE)
    return s


COUNTRY_CANONICAL_LONG = {
    "uk": "United Kingdom of Great Britain and Northern Ireland",
    "u.k.": "United Kingdom of Great Britain and Northern Ireland",
    "united kingdom": "United Kingdom of Great Britain and Northern Ireland",
    "united kingdom of great britain and northern ireland": "United Kingdom of Great Britain and Northern Ireland",
    "great britain": "United Kingdom of Great Britain and Northern Ireland",
    "gb": "United Kingdom of Great Britain and Northern Ireland",
    "england": "United Kingdom of Great Britain and Northern Ireland",
    "scotland": "United Kingdom of Great Britain and Northern Ireland",
    "wales": "United Kingdom of Great Britain and Northern Ireland",
    "northern ireland": "United Kingdom of Great Britain and Northern Ireland",
    "us": "United States of America",
    "u.s.": "United States of America",
    "usa": "United States of America",
    "u.s.a.": "United States of America",
    "united states": "United States of America",
    "united states of america": "United States of America",
}


def normalize_country_key(s):
    if is_missing(s):
        return None
    s = normalize_whitespace(s)
    key = s.lower()
    return COUNTRY_CANONICAL_LONG.get(key, s)


def parse_duration_scalar(x):
    if is_missing(x):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        val = float(x)
        if pd.isna(val) or val <= 0:
            return None
        return int(round(val))
    s = str(x).strip()
    if not s:
        return None
    if re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", s):
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    s_clean = re.sub(r"[^\d.]", "", s)
    if not s_clean:
        return None
    try:
        val = float(s_clean)
        if val <= 0:
            return None
        return int(round(val))
    except Exception:
        return None


SOURCE_WEIGHTS = {
    "musicbrainz": {
        "name": 3.0,
        "artist": 3.0,
        "release-date": 3.0,
        "release-country": 2.0,
        "duration": 2.5,
        "tracks_track_name": 3.0,
        "tracks_track_position": 3.0,
        "tracks_track_duration": 3.0,
        "label": 1.0,
        "genre": 1.0,
    },
    "discogs": {
        "name": 2.5,
        "artist": 2.0,
        "release-date": 2.0,
        "release-country": 3.5,
        "duration": 2.0,
        "tracks_track_name": 2.0,
        "tracks_track_position": 2.0,
        "tracks_track_duration": 1.5,
        "label": 3.5,
        "genre": 2.5,
    },
    "lastfm": {
        "name": 1.5,
        "artist": 1.5,
        "release-date": 0.5,
        "release-country": 0.5,
        "duration": 1.5,
        "tracks_track_name": 1.5,
        "tracks_track_position": 1.5,
        "tracks_track_duration": 1.5,
        "label": 0.5,
        "genre": 0.5,
    },
}


def source_weight(row, attr):
    src = row.get("_source_dataset", "")
    return SOURCE_WEIGHTS.get(src, {}).get(attr, 1.0)


def weighted_vote_original(values, weights, key_func, prefer_longest=True):
    scored = defaultdict(float)
    originals_by_key = defaultdict(list)

    for v, w in zip(values, weights):
        if is_missing(v):
            continue
        original = normalize_whitespace(v)
        key = key_func(original)
        if is_missing(key):
            continue
        scored[key] += w
        originals_by_key[key].append(original)

    if not scored:
        return None

    best_score = max(scored.values())
    best_keys = [k for k, s in scored.items() if s == best_score]

    def rank_key(k):
        originals = originals_by_key[k]
        candidate = sorted(
            originals,
            key=lambda x: (-len(x), x) if prefer_longest else (len(x), x),
        )[0]
        return (-scored[k], -len(candidate), candidate.lower())

    winner_key = sorted(best_keys, key=rank_key)[0]
    originals = originals_by_key[winner_key]
    return sorted(originals, key=lambda x: (-len(x), x))[0] if prefer_longest else sorted(originals, key=lambda x: (len(x), x))[0]


def choose_scalar_placeholder(values):
    non_null = [normalize_whitespace(v) for v in values if not is_missing(v)]
    return non_null[0] if non_null else None


def choose_most_coherent_rows(rows):
    if len(rows) <= 1:
        return rows

    row_infos = []
    for r in rows:
        name_key = normalize_text_key(normalize_name_for_fusion(r.get("name")))
        artist_key = normalize_text_key(normalize_artist_for_fusion(r.get("artist")))
        row_infos.append((r, name_key, artist_key))

    name_counts = Counter([nk for _, nk, _ in row_infos if nk])
    artist_counts = Counter([ak for _, _, ak in row_infos if ak])

    best_name = name_counts.most_common(1)[0][0] if name_counts else None
    best_artist = artist_counts.most_common(1)[0][0] if artist_counts else None

    coherent = []
    for r, nk, ak in row_infos:
        score = 0
        if best_name and nk == best_name:
            score += 1
        if best_artist and ak == best_artist:
            score += 1
        if score >= 1:
            coherent.append(r)

    return coherent if coherent else rows


def weighted_choose_name(rows):
    values, weights = [], []
    for r in rows:
        values.append(normalize_name_for_fusion(r.get("name")))
        weights.append(source_weight(r, "name"))
    return weighted_vote_original(values, weights, normalize_text_key, prefer_longest=True)


def weighted_choose_artist(rows):
    values, weights = [], []
    for r in rows:
        values.append(normalize_artist_for_fusion(r.get("artist")))
        weights.append(source_weight(r, "artist"))
    return weighted_vote_original(values, weights, normalize_text_key, prefer_longest=True)


def weighted_choose_release_country(rows):
    values, weights = [], []
    for r in rows:
        v = r.get("release-country")
        if not is_missing(v):
            v = normalize_country_key(v)
        values.append(v)
        weights.append(source_weight(r, "release-country"))
    return weighted_vote_original(values, weights, normalize_country_key, prefer_longest=True)


def weighted_choose_release_date(rows):
    weighted_dates = defaultdict(float)
    for r in rows:
        v = r.get("release-date")
        if is_missing(v):
            continue
        dt = pd.to_datetime(v, errors="coerce")
        if pd.isna(dt):
            continue
        key = dt.strftime("%Y-%m-%d")
        weighted_dates[key] += source_weight(r, "release-date")
    if not weighted_dates:
        return None
    best_score = max(weighted_dates.values())
    candidates = [k for k, s in weighted_dates.items() if s == best_score]
    return sorted(candidates)[0]


def weighted_choose_duration(rows, fused_track_durations=None):
    grouped = defaultdict(list)
    weighted_values = defaultdict(float)

    for r in rows:
        d = parse_duration_scalar(r.get("duration"))
        if d is None or d <= 0:
            continue
        matched_key = None
        for existing in list(grouped.keys()):
            if abs(existing - d) <= 15:
                matched_key = existing
                break
        if matched_key is None:
            matched_key = d
        grouped[matched_key].append(d)
        weighted_values[matched_key] += source_weight(r, "duration")

    track_sum = None
    if fused_track_durations:
        parsed_track_durations = [parse_duration_scalar(v) for v in parse_list_value(fused_track_durations)]
        parsed_track_durations = [v for v in parsed_track_durations if v is not None and v > 0]
        if parsed_track_durations:
            track_sum = int(sum(parsed_track_durations))

    if not grouped:
        return track_sum

    best_key = sorted(
        grouped.keys(),
        key=lambda k: (-weighted_values[k], -len(grouped[k]), abs(np.median(grouped[k]) - k), k),
    )[0]
    best_duration = int(round(float(np.median(grouped[best_key]))))

    if track_sum is not None:
        if abs(best_duration - track_sum) <= 20:
            return track_sum
        if abs(best_duration - track_sum) <= 60:
            return int(round((best_duration + track_sum) / 2.0))

    return best_duration


def weighted_choose_label(rows):
    values, weights = [], []
    for r in rows:
        values.append(r.get("label"))
        weights.append(source_weight(r, "label"))
    return weighted_vote_original(values, weights, normalize_text_key, prefer_longest=True)


def weighted_choose_genre(rows):
    scores = defaultdict(float)
    originals = defaultdict(list)
    for r in rows:
        v = r.get("genre")
        if is_missing(v):
            continue
        parts = parse_list_value(v)
        if not parts:
            parts = [v]
        for p in parts:
            key = normalize_text_key(p)
            if is_missing(key):
                continue
            scores[key] += source_weight(r, "genre")
            originals[key].append(normalize_whitespace(p))
    if not scores:
        return None
    best_score = max(scores.values())
    candidates = [k for k, s in scores.items() if s == best_score]
    winner = sorted(candidates)[0]
    return sorted(originals[winner], key=lambda x: (-len(x), x))[0]


def parse_track_record(row):
    names = parse_list_value(row.get("tracks_track_name"))
    positions = parse_list_value(row.get("tracks_track_position"))
    durations_raw = parse_list_value(row.get("tracks_track_duration"))
    max_len = max(len(names), len(positions), len(durations_raw), 0)

    names = names + [None] * (max_len - len(names))
    positions = positions + [None] * (max_len - len(positions))
    durations_raw = durations_raw + [None] * (max_len - len(durations_raw))

    tracks = []
    for i in range(max_len):
        name = names[i] if i < len(names) else None
        pos = positions[i] if i < len(positions) else None
        dur = durations_raw[i] if i < len(durations_raw) else None
        dur_parsed = parse_duration_scalar(dur)

        pos_norm = None
        if not is_missing(pos):
            pos_str = str(pos).strip()
            numeric_match = re.search(r"(\d+)", pos_str)
            pos_norm = numeric_match.group(1) if numeric_match else pos_str

        tracks.append(
            {
                "name": normalize_whitespace(name) if not is_missing(name) else None,
                "name_key": normalize_text_key(name) if not is_missing(name) else None,
                "position": pos_norm,
                "duration": dur_parsed,
            }
        )
    return tracks


def track_record_score(row):
    tracks = parse_track_record(row)
    if not tracks:
        return 0
    score = 0
    score += sum(3 for t in tracks if not is_missing(t["name"]))
    score += sum(2 for t in tracks if not is_missing(t["position"]))
    score += sum(1 for t in tracks if t["duration"] is not None and t["duration"] > 0)
    score += source_weight(row, "tracks_track_name")
    return score


def assign_track_slot(track, next_slot, name_to_slot):
    if not is_missing(track.get("position")):
        return str(track["position"])

    name_key = track.get("name_key")
    if name_key and name_key in name_to_slot:
        return name_to_slot[name_key]

    slot = str(next_slot[0])
    next_slot[0] += 1
    if name_key:
        name_to_slot[name_key] = slot
    return slot


def fuse_tracklists(rows):
    rows = sorted(rows, key=lambda r: (-track_record_score(r), r.get("_source_dataset", "")))
    slots = defaultdict(list)
    name_to_slot = {}
    next_slot = [1]

    for r in rows:
        tracks = parse_track_record(r)
        row_weight_name = source_weight(r, "tracks_track_name")
        row_weight_pos = source_weight(r, "tracks_track_position")
        row_weight_dur = source_weight(r, "tracks_track_duration")

        for t in tracks:
            slot_key = assign_track_slot(t, next_slot, name_to_slot)
            slots[slot_key].append(
                {
                    "name": t["name"],
                    "name_key": t["name_key"],
                    "position": t["position"] if not is_missing(t["position"]) else slot_key,
                    "duration": t["duration"],
                    "name_weight": row_weight_name,
                    "position_weight": row_weight_pos,
                    "duration_weight": row_weight_dur,
                }
            )

    if not slots:
        return None, None, None

    def sort_slot_key(k):
        m = re.search(r"(\d+)", str(k))
        return (int(m.group(1)) if m else 10**9, str(k))

    fused_names = []
    fused_positions = []
    fused_durations = []

    for slot_key in sorted(slots.keys(), key=sort_slot_key):
        entries = slots[slot_key]

        name_scores = defaultdict(float)
        name_originals = defaultdict(list)
        position_scores = defaultdict(float)
        duration_scores = defaultdict(float)
        duration_groups = defaultdict(list)

        for e in entries:
            if not is_missing(e["name"]) and e["name_key"]:
                name_scores[e["name_key"]] += e["name_weight"]
                name_originals[e["name_key"]].append(e["name"])

            if not is_missing(e["position"]):
                pk = str(e["position"])
                position_scores[pk] += e["position_weight"]

            if e["duration"] is not None and e["duration"] > 0:
                matched = None
                for existing in list(duration_groups.keys()):
                    if abs(existing - e["duration"]) <= 10:
                        matched = existing
                        break
                if matched is None:
                    matched = e["duration"]
                duration_groups[matched].append(e["duration"])
                duration_scores[matched] += e["duration_weight"]

        fused_name = None
        fused_position = None
        fused_duration = None

        if name_scores:
            best_name_key = sorted(
                name_scores.keys(),
                key=lambda k: (-name_scores[k], -len(sorted(name_originals[k], key=lambda x: (-len(x), x))[0]), k),
            )[0]
            fused_name = sorted(name_originals[best_name_key], key=lambda x: (-len(x), x))[0]

        if position_scores:
            fused_position = sorted(position_scores.keys(), key=lambda k: (-position_scores[k], sort_slot_key(k)))[0]
        else:
            fused_position = str(slot_key)

        if duration_groups:
            best_duration_key = sorted(
                duration_groups.keys(),
                key=lambda k: (-duration_scores[k], -len(duration_groups[k]), abs(np.median(duration_groups[k]) - k), k),
            )[0]
            fused_duration = str(int(round(float(np.median(duration_groups[best_duration_key])))))

        if fused_name is not None or fused_position is not None or fused_duration is not None:
            fused_names.append(fused_name if fused_name is not None else "")
            fused_positions.append(fused_position if fused_position is not None else "")
            fused_durations.append(fused_duration if fused_duration is not None else "")

    while fused_names and fused_names[-1] == "" and fused_durations[-1] == "":
        fused_names.pop()
        fused_positions.pop()
        fused_durations.pop()

    if not fused_names and not fused_positions and not fused_durations:
        return None, None, None

    return (
        str(fused_names) if fused_names else None,
        str(fused_positions) if fused_positions else None,
        str(fused_durations) if fused_durations else None,
    )


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
# Perform Schema Matching
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(discogs, lastfm)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map)

schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
musicbrainz = musicbrainz.rename(columns=rename_map)

discogs = discogs.copy()
lastfm = lastfm.copy()
musicbrainz = musicbrainz.copy()

discogs["_source_dataset"] = "discogs"
lastfm["_source_dataset"] = "lastfm"
musicbrainz["_source_dataset"] = "musicbrainz"

# --------------------------------
# Perform Blocking using precomputed blocking configuration
# --------------------------------

print("Performing Blocking")

blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    window=15,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching using precomputed matching configuration
# --------------------------------

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

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=blocker_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=[0.3, 0.35, 0.25, 0.1],
    threshold=0.6,
    id_column="id",
)

rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=[0.35, 0.25, 0.2, 0.1, 0.1],
    threshold=0.7,
    id_column="id",
)

rb_correspondences_musicbrainz_lastfm = matcher.match(
    df_left=musicbrainz,
    df_right=lastfm,
    candidates=blocker_musicbrainz_lastfm,
    comparators=comparators_musicbrainz_lastfm,
    weights=[0.35, 0.45, 0.2],
    threshold=0.75,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_discogs_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)

rb_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)

rb_correspondences_musicbrainz_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_musicbrainz_lastfm.csv"),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_musicbrainz_lastfm,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")
strategy.add_attribute_fuser("name", choose_scalar_placeholder)
strategy.add_attribute_fuser("artist", choose_scalar_placeholder)
strategy.add_attribute_fuser("release-date", choose_scalar_placeholder)
strategy.add_attribute_fuser("release-country", choose_scalar_placeholder)
strategy.add_attribute_fuser("duration", choose_scalar_placeholder)
strategy.add_attribute_fuser("label", choose_scalar_placeholder)
strategy.add_attribute_fuser("genre", choose_scalar_placeholder)
strategy.add_attribute_fuser("tracks_track_name", choose_scalar_placeholder)
strategy.add_attribute_fuser("tracks_track_position", choose_scalar_placeholder)
strategy.add_attribute_fuser("tracks_track_duration", choose_scalar_placeholder)

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
    include_singletons=True,
)

all_records = pd.concat([discogs, lastfm, musicbrainz], ignore_index=True, sort=False)
fused = rb_fused_standard_blocker.copy()

cluster_member_cols = [c for c in fused.columns if c.endswith("_ids") or c.endswith("_id")]
source_id_columns = ["discogs_id", "lastfm_id", "musicbrainz_id"]

for col in source_id_columns:
    if col not in fused.columns:
        fused[col] = None

record_lookup = {}
for _, row in all_records.iterrows():
    record_lookup[row["id"]] = row.to_dict()


def collect_cluster_rows(fused_row):
    rows = []
    seen = set()

    for src_col in source_id_columns:
        val = fused_row.get(src_col)
        if is_missing(val):
            continue
        values = parse_list_value(val)
        for rid in values:
            if rid in record_lookup and rid not in seen:
                rows.append(record_lookup[rid])
                seen.add(rid)

    if rows:
        return rows

    for col in cluster_member_cols:
        val = fused_row.get(col)
        if is_missing(val):
            continue
        values = parse_list_value(val)
        for rid in values:
            if rid in record_lookup and rid not in seen:
                rows.append(record_lookup[rid])
                seen.add(rid)

    if rows:
        return rows

    rid = fused_row.get("id")
    if rid in record_lookup:
        return [record_lookup[rid]]

    return []


for idx, row in fused.iterrows():
    cluster_rows = collect_cluster_rows(row)
    if not cluster_rows:
        continue

    cluster_rows = choose_most_coherent_rows(cluster_rows)

    fused.at[idx, "name"] = weighted_choose_name(cluster_rows)
    fused.at[idx, "artist"] = weighted_choose_artist(cluster_rows)
    fused.at[idx, "release-date"] = weighted_choose_release_date(cluster_rows)
    fused.at[idx, "release-country"] = weighted_choose_release_country(cluster_rows)
    fused.at[idx, "label"] = weighted_choose_label(cluster_rows)
    fused.at[idx, "genre"] = weighted_choose_genre(cluster_rows)

    track_names, track_positions, track_durations = fuse_tracklists(cluster_rows)
    fused.at[idx, "tracks_track_name"] = track_names
    fused.at[idx, "tracks_track_position"] = track_positions
    fused.at[idx, "tracks_track_duration"] = track_durations

    fused_duration = weighted_choose_duration(cluster_rows, fused_track_durations=track_durations)
    fused.at[idx, "duration"] = fused_duration

os.makedirs("output/data_fusion", exist_ok=True)
fused.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

