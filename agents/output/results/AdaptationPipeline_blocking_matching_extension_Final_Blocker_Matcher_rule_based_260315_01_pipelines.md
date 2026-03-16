# Pipeline Snapshots

notebook_name=AdaptationPipeline_blocking_matching_extension_Final_Blocker_Matcher
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=50.76%
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import (
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    StringComparator,
    NumericComparator,
    RuleBasedMatcher,
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
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_csv(
    DATA_DIR + "output/schema-matching/discogs.csv",
    name="discogs",
)

good_dataset_name_2 = load_csv(
    DATA_DIR + "output/schema-matching/lastfm.csv",
    name="lastfm",
)

good_dataset_name_3 = load_csv(
    DATA_DIR + "output/schema-matching/musicbrainz.csv",
    name="musicbrainz",
)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# --------------------------------
# Perform Schema Matching
# The resulting columns for all datasets will have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_2)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_2 = good_dataset_name_2.rename(columns=rename_map)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_3)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_3 = good_dataset_name_3.rename(columns=rename_map)

# --------------------------------
# Blocking
# Use the precomputed blocking configuration.
# --------------------------------

print("Performing Blocking")

blocker_d2l = EmbeddingBlocker(
    good_dataset_name_1,
    good_dataset_name_2,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_d2m = SortedNeighbourhoodBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    key="name",
    window=15,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_m2l = EmbeddingBlocker(
    good_dataset_name_3,
    good_dataset_name_2,
    text_cols=["name", "artist", "duration"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching
# Use the precomputed matching configuration.
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
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=10.0,
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
    StringComparator(
        column="release-date",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
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
    StringComparator(
        column="duration",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_d2l = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2,
    candidates=blocker_d2l,
    comparators=comparators_d2l,
    weights=[0.3, 0.35, 0.25, 0.1],
    threshold=0.6,
    id_column="id",
)

rb_correspondences_d2m = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3,
    candidates=blocker_d2m,
    comparators=comparators_d2m,
    weights=[0.35, 0.25, 0.2, 0.1, 0.1],
    threshold=0.7,
    id_column="id",
)

rb_correspondences_m2l = matcher.match(
    df_left=good_dataset_name_3,
    df_right=good_dataset_name_2,
    candidates=blocker_m2l,
    comparators=comparators_m2l,
    weights=[0.35, 0.45, 0.2],
    threshold=0.75,
    id_column="id",
)

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
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=17
node_name=execute_pipeline
accuracy_score=34.52%
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import (
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    StringComparator,
    NumericComparator,
    DateComparator,
    RuleBasedMatcher,
)
from PyDI.fusion import DataFusionStrategy, DataFusionEngine
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import ast
import re
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = ""

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_csv(
    DATA_DIR + "output/schema-matching/discogs.csv",
    name="discogs",
)

good_dataset_name_2 = load_csv(
    DATA_DIR + "output/schema-matching/lastfm.csv",
    name="lastfm",
)

good_dataset_name_3 = load_csv(
    DATA_DIR + "output/schema-matching/musicbrainz.csv",
    name="musicbrainz",
)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# --------------------------------
# Perform Schema Matching
# The resulting columns for all datasets will have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_2)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_2 = good_dataset_name_2.rename(columns=rename_map)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_3)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_3 = good_dataset_name_3.rename(columns=rename_map)

# --------------------------------
# Helpers
# --------------------------------

def is_missing(x):
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    if isinstance(x, str) and x.strip().lower() in {"", "nan", "none", "null", "[]"}:
        return True
    return False

def safe_str(x):
    if is_missing(x):
        return ""
    return str(x).strip()

def lower_strip(x):
    return safe_str(x).lower().strip()

def parse_list_like(x):
    if is_missing(x):
        return []
    if isinstance(x, (list, tuple, set)):
        return [safe_str(v) for v in x if not is_missing(v) and safe_str(v)]
    s = safe_str(x)
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple, set)):
            return [safe_str(v) for v in parsed if not is_missing(v) and safe_str(v)]
    except Exception:
        pass
    if "|" in s:
        return [p.strip() for p in s.split("|") if p.strip()]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]

def normalize_whitespace_punct(s):
    s = safe_str(s).lower()
    s = s.replace("&", " and ")
    s = s.replace("_", " ")
    s = re.sub(r"\s*[-]+\s*", " ", s)
    s = re.sub(r"\s*/\s*", " / ", s)
    s = re.sub(r"[^\w\s/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_duration_value(x):
    if is_missing(x):
        return pd.NA
    s = safe_str(x)
    try:
        return int(round(float(s)))
    except Exception:
        return pd.NA

def normalize_name_for_match(x):
    s = normalize_whitespace_punct(x)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def remove_artist_prefix_from_name(name, artist):
    n = safe_str(name)
    a = safe_str(artist)
    if not n:
        return ""
    n_l = lower_strip(n)
    a_l = lower_strip(a)

    if a_l:
        artist_patterns = [
            rf"^{re.escape(a_l)}\s*[-:]\s*",
            rf"^{re.escape(a_l)}\s+",
        ]
        for pattern in artist_patterns:
            if re.match(pattern, n_l):
                stripped = re.sub(pattern, "", n_l, count=1)
                return normalize_name_for_match(stripped)

    return normalize_name_for_match(n)

def normalize_country(x):
    s = lower_strip(x)
    mapping = {
        "uk": "united kingdom",
        "u.k.": "united kingdom",
        "great britain": "united kingdom",
        "england": "united kingdom",
        "gb": "united kingdom",
        "united kingdom of great britain and northern ireland": "united kingdom",
        "scotland": "united kingdom",
        "wales": "united kingdom",
        "northern ireland": "united kingdom",
        "us": "united states",
        "u.s.": "united states",
        "usa": "united states",
        "u.s.a.": "united states",
    }
    return mapping.get(s, s)

def normalize_track_names(x):
    vals = parse_list_like(x)
    out = []
    for v in vals:
        s = normalize_name_for_match(v)
        if s:
            out.append(s)
    return out

def normalize_track_positions(x):
    vals = parse_list_like(x)
    out = []
    for v in vals:
        try:
            out.append(str(int(float(v))))
        except Exception:
            s = safe_str(v)
            if s:
                out.append(s)
    return out

def normalize_track_durations(x):
    vals = parse_list_like(x)
    out = []
    for v in vals:
        try:
            out.append(str(int(round(float(v)))))
        except Exception:
            continue
    return out

def canonical_list_string(vals):
    vals = [safe_str(v) for v in vals if safe_str(v)]
    return " | ".join(vals)

def canonical_set_string(vals):
    vals = sorted(set([safe_str(v) for v in vals if safe_str(v)]))
    return " | ".join(vals)

def choose_most_complete(values):
    cleaned = [safe_str(v) for v in values if not is_missing(v) and safe_str(v)]
    if not cleaned:
        return ""
    cleaned = sorted(cleaned, key=lambda x: (len(x), sum(ch.isalnum() for ch in x)), reverse=True)
    return cleaned[0]

def choose_best_name(values):
    cleaned = [safe_str(v) for v in values if not is_missing(v) and safe_str(v)]
    if not cleaned:
        return ""
    scored = []
    for v in cleaned:
        score = 0
        if " - " not in v and re.search(r"\b[a-zA-Z0-9]", v):
            score += 5
        if "/" in v:
            score += 2
        score += min(len(v), 200) / 1000.0
        scored.append((score, v))
    scored.sort(reverse=True)
    return scored[0][1]

def fuse_duration(values):
    nums = []
    for v in values:
        if is_missing(v):
            continue
        try:
            nums.append(int(round(float(str(v)))))
        except Exception:
            continue
    if not nums:
        return ""
    nums = sorted(nums)
    if len(nums) == 1:
        return str(nums[0])
    if len(nums) == 2:
        if abs(nums[0] - nums[1]) <= 10:
            return str(int(round(sum(nums) / 2.0)))
        return str(nums[0])
    if abs(nums[0] - nums[1]) <= abs(nums[1] - nums[2]):
        return str(int(round((nums[0] + nums[1]) / 2.0)))
    return str(int(round((nums[1] + nums[2]) / 2.0)))

def fuse_release_date(values):
    cleaned = [safe_str(v) for v in values if safe_str(v)]
    if not cleaned:
        return ""
    exact_dates = [v for v in cleaned if re.match(r"^\d{4}-\d{2}-\d{2}$", v)]
    if exact_dates:
        return sorted(exact_dates)[0]
    year_month = [v for v in cleaned if re.match(r"^\d{4}-\d{2}$", v)]
    if year_month:
        return sorted(year_month)[0]
    years = [v for v in cleaned if re.match(r"^\d{4}$", v)]
    if years:
        return sorted(years)[0]
    return cleaned[0]

def fuse_release_country(values):
    cleaned = [normalize_country(v) for v in values if safe_str(v)]
    if not cleaned:
        return ""
    freq = {}
    for v in cleaned:
        freq[v] = freq.get(v, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    return ranked[0][0]

def fuse_track_name(values):
    candidates = []
    for v in values:
        vals = normalize_track_names(v)
        if vals:
            candidates.append(vals)
    if not candidates:
        return ""
    candidates.sort(key=lambda x: (len(x), sum(len(t) for t in x)), reverse=True)
    return canonical_list_string(candidates[0])

def fuse_track_position(values):
    candidates = []
    for v in values:
        vals = normalize_track_positions(v)
        if vals:
            candidates.append(vals)
    if not candidates:
        return ""
    candidates.sort(key=lambda x: (len(x), sum(len(t) for t in x)), reverse=True)
    return canonical_list_string(candidates[0])

def fuse_track_duration(values):
    candidates = []
    for v in values:
        vals = normalize_track_durations(v)
        if vals:
            candidates.append(vals)
    if not candidates:
        return ""
    exact_multi = [c for c in candidates if len(c) > 1]
    if exact_multi:
        exact_multi.sort(key=lambda x: (len(x), sum(int(v) for v in x if safe_str(v).isdigit())), reverse=True)
        return canonical_list_string(exact_multi[0])
    candidates.sort(key=lambda x: len(x), reverse=True)
    return canonical_list_string(candidates[0])

# --------------------------------
# Data normalization
# --------------------------------

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    if "artist" in df.columns:
        df["artist"] = df["artist"].apply(safe_str)

    if "name" in df.columns:
        df["name"] = df["name"].apply(safe_str)
        df["name_norm"] = df.apply(
            lambda row: remove_artist_prefix_from_name(
                row["name"],
                row["artist"] if "artist" in row.index else "",
            ),
            axis=1,
        )

    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(normalize_duration_value)
        df["duration_str"] = df["duration"].apply(lambda x: "" if is_missing(x) else str(int(x)))

    if "release-country" in df.columns:
        df["release-country"] = df["release-country"].apply(normalize_country)

    if "release-date" in df.columns:
        df["release-date"] = df["release-date"].apply(safe_str)

    if "tracks_track_name" in df.columns:
        df["tracks_track_name"] = df["tracks_track_name"].apply(normalize_track_names)
        df["tracks_track_name_match"] = df["tracks_track_name"].apply(canonical_set_string)

    if "tracks_track_position" in df.columns:
        df["tracks_track_position"] = df["tracks_track_position"].apply(normalize_track_positions)
        df["tracks_track_position_match"] = df["tracks_track_position"].apply(canonical_list_string)

    if "tracks_track_duration" in df.columns:
        df["tracks_track_duration"] = df["tracks_track_duration"].apply(normalize_track_durations)
        df["tracks_track_duration_match"] = df["tracks_track_duration"].apply(canonical_list_string)

# --------------------------------
# Blocking
# Use the precomputed blocking configuration.
# --------------------------------

print("Performing Blocking")

blocker_d2l = EmbeddingBlocker(
    good_dataset_name_1,
    good_dataset_name_2,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_d2m = SortedNeighbourhoodBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    key="name",
    window=15,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_m2l = EmbeddingBlocker(
    good_dataset_name_3,
    good_dataset_name_2,
    text_cols=["name", "artist", "duration"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching
# Use the precomputed matching configuration.
# --------------------------------

comparators_d2l = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="tracks_track_name_match",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=10.0,
    ),
]

comparators_d2m = [
    StringComparator(
        column="name_norm",
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

comparators_m2l = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="duration_str",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_d2l = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2,
    candidates=blocker_d2l,
    comparators=comparators_d2l,
    weights=[0.3, 0.35, 0.25, 0.1],
    threshold=0.6,
    id_column="id",
)

rb_correspondences_d2m = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3,
    candidates=blocker_d2m,
    comparators=comparators_d2m,
    weights=[0.35, 0.25, 0.2, 0.1, 0.1],
    threshold=0.7,
    id_column="id",
)

rb_correspondences_m2l = matcher.match(
    df_left=good_dataset_name_3,
    df_right=good_dataset_name_2,
    candidates=blocker_m2l,
    comparators=comparators_m2l,
    weights=[0.35, 0.45, 0.2],
    threshold=0.75,
    id_column="id",
)

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

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    if "tracks_track_name" in df.columns:
        df["tracks_track_name"] = df["tracks_track_name"].apply(canonical_list_string)
    if "tracks_track_position" in df.columns:
        df["tracks_track_position"] = df["tracks_track_position"].apply(canonical_list_string)
    if "tracks_track_duration" in df.columns:
        df["tracks_track_duration"] = df["tracks_track_duration"].apply(canonical_list_string)
    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(lambda x: "" if is_missing(x) else str(int(x)))
    if "artist" in df.columns:
        df["artist"] = df["artist"].apply(safe_str)
    if "name" in df.columns:
        df["name"] = df["name"].apply(safe_str)
    if "release-date" in df.columns:
        df["release-date"] = df["release-date"].apply(safe_str)
    if "release-country" in df.columns:
        df["release-country"] = df["release-country"].apply(normalize_country)

    drop_cols = [
        "name_norm",
        "duration_str",
        "tracks_track_name_match",
        "tracks_track_position_match",
        "tracks_track_duration_match",
    ]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    if existing_drop_cols:
        df.drop(columns=existing_drop_cols, inplace=True)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", choose_best_name)
strategy.add_attribute_fuser("artist", choose_most_complete)
strategy.add_attribute_fuser("release-date", fuse_release_date)
strategy.add_attribute_fuser("release-country", fuse_release_country)
strategy.add_attribute_fuser("duration", fuse_duration)
strategy.add_attribute_fuser("label", choose_most_complete)
strategy.add_attribute_fuser("genre", choose_most_complete)
strategy.add_attribute_fuser("tracks_track_name", fuse_track_name)
strategy.add_attribute_fuser("tracks_track_position", fuse_track_position)
strategy.add_attribute_fuser("tracks_track_duration", fuse_track_duration)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
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
node_index=22
node_name=execute_pipeline
accuracy_score=51.27%
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import (
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    StringComparator,
    NumericComparator,
    DateComparator,
    RuleBasedMatcher,
)
from PyDI.fusion import DataFusionStrategy, DataFusionEngine
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import ast
import re
from collections import Counter
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = ""

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_csv(
    DATA_DIR + "output/schema-matching/discogs.csv",
    name="discogs",
)

good_dataset_name_2 = load_csv(
    DATA_DIR + "output/schema-matching/lastfm.csv",
    name="lastfm",
)

good_dataset_name_3 = load_csv(
    DATA_DIR + "output/schema-matching/musicbrainz.csv",
    name="musicbrainz",
)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# --------------------------------
# Perform Schema Matching
# The resulting columns for all datasets will have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_2)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_2 = good_dataset_name_2.rename(columns=rename_map)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_3)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
good_dataset_name_3 = good_dataset_name_3.rename(columns=rename_map)

# --------------------------------
# Helpers
# --------------------------------

def is_missing(x):
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    if isinstance(x, str) and x.strip().lower() in {"", "nan", "none", "null", "[]", "{}", "na", "n/a"}:
        return True
    return False

def safe_str(x):
    if is_missing(x):
        return ""
    return str(x).strip()

def lower_strip(x):
    return safe_str(x).lower().strip()

def parse_list_like(x):
    if is_missing(x):
        return []
    if isinstance(x, (list, tuple, set)):
        return [safe_str(v) for v in x if not is_missing(v) and safe_str(v)]
    s = safe_str(x)
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple, set)):
            return [safe_str(v) for v in parsed if not is_missing(v) and safe_str(v)]
    except Exception:
        pass
    if "|" in s:
        return [p.strip() for p in s.split("|") if p.strip()]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]

def normalize_whitespace_punct(s):
    s = safe_str(s).lower()
    s = s.replace("&", " and ")
    s = s.replace("_", " ")
    s = re.sub(r"\s*[-]+\s*", " ", s)
    s = re.sub(r"\s*/\s*", " / ", s)
    s = re.sub(r"[^\w\s/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_name_for_match(x):
    s = normalize_whitespace_punct(x)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_duration_value(x):
    if is_missing(x):
        return pd.NA
    s = safe_str(x)
    try:
        return int(round(float(s)))
    except Exception:
        return pd.NA

def remove_artist_prefix_from_name(name, artist):
    n = safe_str(name)
    a = safe_str(artist)
    if not n:
        return ""
    n_l = lower_strip(n)
    a_l = lower_strip(a)

    if a_l:
        artist_patterns = [
            rf"^{re.escape(a_l)}\s*[-:]\s*",
            rf"^{re.escape(a_l)}\s+[-:]\s+",
        ]
        for pattern in artist_patterns:
            if re.match(pattern, n_l):
                stripped = re.sub(pattern, "", n_l, count=1)
                return normalize_name_for_match(stripped)

    return normalize_name_for_match(n)

def normalize_country(x):
    s = lower_strip(x)
    mapping = {
        "uk": "united kingdom",
        "u.k.": "united kingdom",
        "great britain": "united kingdom",
        "england": "united kingdom",
        "gb": "united kingdom",
        "united kingdom of great britain and northern ireland": "united kingdom",
        "scotland": "united kingdom",
        "wales": "united kingdom",
        "northern ireland": "united kingdom",
        "us": "united states",
        "u.s.": "united states",
        "usa": "united states",
        "u.s.a.": "united states",
    }
    return mapping.get(s, s)

def normalize_track_names(x):
    vals = parse_list_like(x)
    out = []
    for v in vals:
        s = normalize_name_for_match(v)
        if s:
            out.append(s)
    return out

def normalize_track_positions(x):
    vals = parse_list_like(x)
    out = []
    for v in vals:
        try:
            out.append(str(int(float(v))))
        except Exception:
            s = safe_str(v)
            if s:
                out.append(s)
    return out

def normalize_track_durations(x):
    vals = parse_list_like(x)
    out = []
    for v in vals:
        try:
            out.append(str(int(round(float(v)))))
        except Exception:
            continue
    return out

def canonical_list_string(vals):
    vals = [safe_str(v) for v in vals if safe_str(v)]
    return str(vals)

def canonical_set_string(vals):
    vals = sorted(set([safe_str(v) for v in vals if safe_str(v)]))
    return " | ".join(vals)

def prefer_nonempty(values):
    cleaned = [safe_str(v) for v in values if safe_str(v)]
    return cleaned[0] if cleaned else ""

def choose_best_artist(values):
    cleaned = [safe_str(v) for v in values if safe_str(v)]
    if not cleaned:
        return ""
    cleaned = sorted(cleaned, key=lambda x: (len(x), x), reverse=True)
    return cleaned[0]

def choose_best_name(values):
    cleaned = [safe_str(v) for v in values if safe_str(v)]
    if not cleaned:
        return ""
    exact_album_like = []
    prefixed = []
    for v in cleaned:
        score = 0
        if re.match(r".+\s-\s+.+", v):
            prefixed.append(v)
        else:
            exact_album_like.append(v)
        if "/" in v:
            score += 2
        if re.search(r"[A-Za-z0-9]", v):
            score += 1
        score += min(len(v), 200) / 1000.0
    target = exact_album_like if exact_album_like else cleaned
    target = sorted(target, key=lambda x: ("/" in x, len(x), x), reverse=True)
    return target[0]

def fuse_duration(values):
    nums = []
    for v in values:
        if is_missing(v):
            continue
        try:
            nums.append(int(round(float(str(v)))))
        except Exception:
            continue
    if not nums:
        return ""
    counts = Counter(nums)
    most_common = counts.most_common()
    if most_common[0][1] >= 2:
        return str(most_common[0][0])
    nums_sorted = sorted(nums)
    if len(nums_sorted) == 1:
        return str(nums_sorted[0])
    if len(nums_sorted) == 2:
        if abs(nums_sorted[0] - nums_sorted[1]) <= 10:
            return str(int(round(sum(nums_sorted) / 2.0)))
        return str(min(nums_sorted))
    median_val = nums_sorted[len(nums_sorted) // 2]
    return str(int(median_val))

def fuse_release_date(values):
    cleaned = [safe_str(v) for v in values if safe_str(v)]
    if not cleaned:
        return ""
    exact_dates = [v for v in cleaned if re.match(r"^\d{4}-\d{2}-\d{2}$", v)]
    if exact_dates:
        counts = Counter(exact_dates)
        return sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)[0][0]
    year_month = [v for v in cleaned if re.match(r"^\d{4}-\d{2}$", v)]
    if year_month:
        counts = Counter(year_month)
        return sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)[0][0]
    years = [v for v in cleaned if re.match(r"^\d{4}$", v)]
    if years:
        counts = Counter(years)
        return sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)[0][0]
    return cleaned[0]

def fuse_release_country(values):
    cleaned = [normalize_country(v) for v in values if safe_str(v)]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    return sorted(counts.items(), key=lambda x: (x[1], len(x[0]), x[0]), reverse=True)[0][0]

def select_best_track_list(candidates, prefer_longer=True):
    if not candidates:
        return []
    count_map = Counter(tuple(c) for c in candidates if c)
    if count_map:
        best_tuple = sorted(
            count_map.items(),
            key=lambda x: (x[1], len(x[0]), sum(len(i) for i in x[0])),
            reverse=True,
        )[0][0]
        return list(best_tuple)
    candidates = [c for c in candidates if c]
    if not candidates:
        return []
    if prefer_longer:
        candidates.sort(key=lambda x: (len(x), sum(len(i) for i in x)), reverse=True)
    else:
        candidates.sort(key=lambda x: (sum(len(i) for i in x), len(x)), reverse=True)
    return candidates[0]

def fuse_track_name(values):
    candidates = []
    for v in values:
        vals = normalize_track_names(v)
        if vals:
            candidates.append(vals)
    best = select_best_track_list(candidates, prefer_longer=True)
    return canonical_list_string(best) if best else ""

def fuse_track_position(values):
    candidates = []
    for v in values:
        vals = normalize_track_positions(v)
        if vals:
            candidates.append(vals)
    best = select_best_track_list(candidates, prefer_longer=True)
    return canonical_list_string(best) if best else ""

def fuse_track_duration(values):
    candidates = []
    for v in values:
        vals = normalize_track_durations(v)
        if vals:
            candidates.append(vals)
    non_empty = [c for c in candidates if c]
    if not non_empty:
        return ""
    exact_multi = [c for c in non_empty if len(c) > 1]
    chosen = select_best_track_list(exact_multi if exact_multi else non_empty, prefer_longer=True)
    return canonical_list_string(chosen) if chosen else ""

# --------------------------------
# Data normalization
# --------------------------------

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    if "artist" in df.columns:
        df["artist"] = df["artist"].apply(safe_str)

    if "name" in df.columns:
        df["name"] = df["name"].apply(safe_str)
        df["name_norm"] = df.apply(
            lambda row: remove_artist_prefix_from_name(
                row["name"],
                row["artist"] if "artist" in row.index else "",
            ),
            axis=1,
        )

    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(normalize_duration_value)
        df["duration_str"] = df["duration"].apply(lambda x: "" if is_missing(x) else str(int(x)))

    if "release-country" in df.columns:
        df["release-country"] = df["release-country"].apply(normalize_country)

    if "release-date" in df.columns:
        df["release-date"] = df["release-date"].apply(safe_str)

    if "tracks_track_name" in df.columns:
        df["tracks_track_name"] = df["tracks_track_name"].apply(normalize_track_names)
        df["tracks_track_name_match"] = df["tracks_track_name"].apply(canonical_set_string)

    if "tracks_track_position" in df.columns:
        df["tracks_track_position"] = df["tracks_track_position"].apply(normalize_track_positions)
        df["tracks_track_position_match"] = df["tracks_track_position"].apply(canonical_list_string)

    if "tracks_track_duration" in df.columns:
        df["tracks_track_duration"] = df["tracks_track_duration"].apply(normalize_track_durations)
        df["tracks_track_duration_match"] = df["tracks_track_duration"].apply(canonical_list_string)

# --------------------------------
# Blocking
# Use the precomputed blocking configuration.
# --------------------------------

print("Performing Blocking")

blocker_d2l = EmbeddingBlocker(
    good_dataset_name_1,
    good_dataset_name_2,
    text_cols=["name", "artist"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_d2m = SortedNeighbourhoodBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    key="name",
    window=15,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_m2l = EmbeddingBlocker(
    good_dataset_name_3,
    good_dataset_name_2,
    text_cols=["name", "artist", "duration"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching
# Use the precomputed matching configuration.
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
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=10.0,
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
    StringComparator(
        column="duration_str",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_d2l = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2,
    candidates=blocker_d2l,
    comparators=comparators_d2l,
    weights=[0.3, 0.35, 0.25, 0.1],
    threshold=0.6,
    id_column="id",
)

rb_correspondences_d2m = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3,
    candidates=blocker_d2m,
    comparators=comparators_d2m,
    weights=[0.35, 0.25, 0.2, 0.1, 0.1],
    threshold=0.7,
    id_column="id",
)

rb_correspondences_m2l = matcher.match(
    df_left=good_dataset_name_3,
    df_right=good_dataset_name_2,
    candidates=blocker_m2l,
    comparators=comparators_m2l,
    weights=[0.35, 0.45, 0.2],
    threshold=0.75,
    id_column="id",
)

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

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(lambda x: "" if is_missing(x) else str(int(x)))
    if "artist" in df.columns:
        df["artist"] = df["artist"].apply(safe_str)
    if "name" in df.columns:
        df["name"] = df["name"].apply(safe_str)
    if "release-date" in df.columns:
        df["release-date"] = df["release-date"].apply(safe_str)
    if "release-country" in df.columns:
        df["release-country"] = df["release-country"].apply(normalize_country)
    if "tracks_track_name" in df.columns:
        df["tracks_track_name"] = df["tracks_track_name"].apply(canonical_list_string)
    if "tracks_track_position" in df.columns:
        df["tracks_track_position"] = df["tracks_track_position"].apply(canonical_list_string)
    if "tracks_track_duration" in df.columns:
        df["tracks_track_duration"] = df["tracks_track_duration"].apply(canonical_list_string)

    drop_cols = [
        "name_norm",
        "duration_str",
        "tracks_track_name_match",
        "tracks_track_position_match",
        "tracks_track_duration_match",
    ]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    if existing_drop_cols:
        df.drop(columns=existing_drop_cols, inplace=True)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", choose_best_name)
strategy.add_attribute_fuser("artist", choose_best_artist)
strategy.add_attribute_fuser("release-date", fuse_release_date)
strategy.add_attribute_fuser("release-country", fuse_release_country)
strategy.add_attribute_fuser("duration", fuse_duration)
strategy.add_attribute_fuser("label", prefer_nonempty)
strategy.add_attribute_fuser("genre", prefer_nonempty)
strategy.add_attribute_fuser("tracks_track_name", fuse_track_name)
strategy.add_attribute_fuser("tracks_track_position", fuse_track_position)
strategy.add_attribute_fuser("tracks_track_duration", fuse_track_duration)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

rb_fused_standard_blocker = engine.run(
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

