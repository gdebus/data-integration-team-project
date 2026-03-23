# Pipeline Snapshots

notebook_name=Agent I
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=5
node_name=execute_pipeline
accuracy_score=36.04%
------------------------------------------------------------

```python
from PyDI.io import load_xml

from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

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
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "input/datasets/music/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_xml(
    DATA_DIR + "musicbrainz.xml",
    name="musicbrainz",
)

good_dataset_name_2 = load_xml(
    DATA_DIR + "discogs.xml",
    name="discogs",
)

good_dataset_name_3 = load_xml(
    DATA_DIR + "lastfm.xml",
    name="lastfm",
)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# --------------------------------
# Perform Schema Matching (LLM-based matching)
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
# Lightweight normalization for blocking/matching robustness
# --------------------------------

def is_missing(value):
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return False
    try:
        return pd.isna(value)
    except Exception:
        return False

def normalize_text(value):
    if is_missing(value):
        return value
    if isinstance(value, (list, tuple, set)):
        value = " ".join([str(v) for v in value if not is_missing(v)])
    value = str(value).lower().strip()
    value = " ".join(value.split())
    return value

def normalize_name(value):
    if is_missing(value):
        return value
    value = normalize_text(value)
    return value

def to_numeric(value):
    if is_missing(value):
        return None
    if isinstance(value, (list, tuple, set)):
        if len(value) == 0:
            return None
        numeric_values = []
        for v in value:
            if is_missing(v):
                continue
            parsed = pd.to_numeric(v, errors="coerce")
            if not pd.isna(parsed):
                numeric_values.append(float(parsed))
        if len(numeric_values) == 0:
            return None
        return float(sum(numeric_values))
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    if "name" in df.columns:
        df["name_norm"] = df["name"].apply(normalize_name)
    if "artist" in df.columns:
        df["artist_norm"] = df["artist"].apply(normalize_text)
    if "duration" in df.columns:
        df["duration_num"] = df["duration"].apply(to_numeric)
    if "tracks_track_name" in df.columns:
        df["tracks_track_name_norm"] = df["tracks_track_name"].apply(normalize_text)

# --------------------------------
# Blocking
# Use informative identity signals only: artist is shared, high-coverage, and discriminative
# --------------------------------

print("Performing Blocking")

blocker_m2d = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_2,
    on=["artist_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_m2l = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    on=["artist_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_d2l = StandardBlocker(
    good_dataset_name_2,
    good_dataset_name_3,
    on=["artist_norm"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching configuration
# --------------------------------

comparators_m2d = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="artist_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="duration_num",
        max_difference=180,
    ),
    StringComparator(
        column="tracks_track_name_norm",
        similarity_function="jaccard",
    ),
]

comparators_m2l = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="artist_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="duration_num",
        max_difference=180,
    ),
    StringComparator(
        column="tracks_track_name_norm",
        similarity_function="jaccard",
    ),
]

comparators_d2l = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="artist_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="duration_num",
        max_difference=180,
    ),
    StringComparator(
        column="tracks_track_name_norm",
        similarity_function="jaccard",
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_m2d = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2,
    candidates=blocker_m2d,
    comparators=comparators_m2d,
    weights=[0.4, 0.3, 0.1, 0.2],
    threshold=0.65,
    id_column="id",
)

rb_correspondences_m2l = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3,
    candidates=blocker_m2l,
    comparators=comparators_m2l,
    weights=[0.45, 0.3, 0.1, 0.15],
    threshold=0.65,
    id_column="id",
)

rb_correspondences_d2l = matcher.match(
    df_left=good_dataset_name_2,
    df_right=good_dataset_name_3,
    candidates=blocker_d2l,
    comparators=comparators_d2l,
    weights=[0.45, 0.3, 0.1, 0.15],
    threshold=0.65,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_m2d.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_good_dataset_name_1_good_dataset_name_2.csv",
    ),
    index=False,
)

rb_correspondences_m2l.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_good_dataset_name_1_good_dataset_name_3.csv",
    ),
    index=False,
)

rb_correspondences_d2l.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_good_dataset_name_2_good_dataset_name_3.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_m2d, rb_correspondences_m2l, rb_correspondences_d2l],
    ignore_index=True,
)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("duration", longest_string)
strategy.add_attribute_fuser("tracks_track_name", union)
strategy.add_attribute_fuser("tracks_track_duration", union)
strategy.add_attribute_fuser("tracks_track_position", union)

if "label" in good_dataset_name_1.columns or "label" in good_dataset_name_2.columns or "label" in good_dataset_name_3.columns:
    strategy.add_attribute_fuser("label", longest_string)

if "genre" in good_dataset_name_1.columns or "genre" in good_dataset_name_2.columns or "genre" in good_dataset_name_3.columns:
    strategy.add_attribute_fuser("genre", longest_string)

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
node_index=10
node_name=execute_pipeline
accuracy_score=26.90%
------------------------------------------------------------

```python
from PyDI.io import load_xml

from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import re
from dotenv import load_dotenv
import os

# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "input/datasets/music/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_xml(
    DATA_DIR + "musicbrainz.xml",
    name="musicbrainz",
)

good_dataset_name_2 = load_xml(
    DATA_DIR + "discogs.xml",
    name="discogs",
)

good_dataset_name_3 = load_xml(
    DATA_DIR + "lastfm.xml",
    name="lastfm",
)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# --------------------------------
# Perform Schema Matching (LLM-based matching)
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
# Normalization helpers
# --------------------------------

def is_missing(value):
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return False
    try:
        return pd.isna(value)
    except Exception:
        return False

def normalize_text(value):
    if is_missing(value):
        return None
    if isinstance(value, (list, tuple, set)):
        value = " ".join([str(v) for v in value if not is_missing(v)])
    value = str(value).lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value

def normalize_text_loose(value):
    value = normalize_text(value)
    if value is None:
        return None
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

def normalize_name(value):
    value = normalize_text_loose(value)
    if value is None:
        return None
    return value

def normalize_artist(value):
    return normalize_text_loose(value)

def normalize_release_country(value):
    value = normalize_text(value)
    if value is None:
        return None
    country_map = {
        "uk": "united kingdom",
        "u.k.": "united kingdom",
        "united kingdom of great britain and northern ireland": "united kingdom",
        "great britain": "united kingdom",
        "england": "united kingdom",
        "us": "united states",
        "u.s.": "united states",
        "usa": "united states",
        "u.s.a.": "united states",
    }
    return country_map.get(value, value)

def to_numeric(value):
    if is_missing(value):
        return None
    if isinstance(value, (list, tuple, set)):
        nums = []
        for v in value:
            if is_missing(v):
                continue
            parsed = pd.to_numeric(v, errors="coerce")
            if not pd.isna(parsed):
                nums.append(float(parsed))
        if len(nums) == 0:
            return None
        return float(sum(nums))
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)

def normalize_duration(value):
    num = to_numeric(value)
    if num is None:
        return None
    if float(num) == 0.0:
        return None
    return str(int(round(num)))

def normalize_track_names(value):
    if is_missing(value):
        return None
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    cleaned = []
    for v in value:
        nv = normalize_text_loose(v)
        if nv:
            cleaned.append(nv)
    if len(cleaned) == 0:
        return None
    cleaned = sorted(dict.fromkeys(cleaned))
    return " | ".join(cleaned)

def normalize_track_positions(value):
    if is_missing(value):
        return None
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    cleaned = []
    for v in value:
        parsed = pd.to_numeric(v, errors="coerce")
        if not pd.isna(parsed):
            cleaned.append(str(int(parsed)))
    if len(cleaned) == 0:
        return None
    return " | ".join(cleaned)

def normalize_track_durations(value):
    if is_missing(value):
        return None
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    cleaned = []
    for v in value:
        parsed = pd.to_numeric(v, errors="coerce")
        if not pd.isna(parsed) and float(parsed) > 0:
            cleaned.append(str(int(parsed)))
    if len(cleaned) == 0:
        return None
    return " | ".join(cleaned)

def build_block_key(row):
    artist = row.get("artist_norm")
    name = row.get("name_key")
    if is_missing(artist) or is_missing(name):
        return None
    return f"{artist}||{name}"

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    if "name" in df.columns:
        df["name_norm"] = df["name"].apply(normalize_name)
        df["name_key"] = df["name_norm"].apply(
            lambda x: None if is_missing(x) else re.sub(r"\s+", " ", x).strip()
        )
    if "artist" in df.columns:
        df["artist_norm"] = df["artist"].apply(normalize_artist)
    if "release-country" in df.columns:
        df["release_country_norm"] = df["release-country"].apply(normalize_release_country)
    if "duration" in df.columns:
        df["duration_num"] = df["duration"].apply(to_numeric)
        df["duration_clean"] = df["duration"].apply(normalize_duration)
    if "tracks_track_name" in df.columns:
        df["tracks_track_name_norm"] = df["tracks_track_name"].apply(normalize_track_names)
    if "tracks_track_position" in df.columns:
        df["tracks_track_position_norm"] = df["tracks_track_position"].apply(normalize_track_positions)
    if "tracks_track_duration" in df.columns:
        df["tracks_track_duration_norm"] = df["tracks_track_duration"].apply(normalize_track_durations)

    df["artist_name_block"] = df.apply(build_block_key, axis=1)

# --------------------------------
# Blocking
# Use strong entity identity signal: artist + normalized release name
# --------------------------------

print("Performing Blocking")

blocker_m2d = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_2,
    on=["artist_name_block"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_m2l = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    on=["artist_name_block"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_d2l = StandardBlocker(
    good_dataset_name_2,
    good_dataset_name_3,
    on=["artist_name_block"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching configuration
# --------------------------------

comparators_m2d = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="artist_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="tracks_track_name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="tracks_track_position_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="duration_num",
        max_difference=240,
    ),
]

comparators_m2l = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="artist_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="tracks_track_name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="tracks_track_position_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="duration_num",
        max_difference=240,
    ),
]

comparators_d2l = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="artist_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="tracks_track_name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="tracks_track_position_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="duration_num",
        max_difference=240,
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_m2d = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2,
    candidates=blocker_m2d,
    comparators=comparators_m2d,
    weights=[0.35, 0.25, 0.25, 0.10, 0.05],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_m2l = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3,
    candidates=blocker_m2l,
    comparators=comparators_m2l,
    weights=[0.35, 0.25, 0.25, 0.10, 0.05],
    threshold=0.75,
    id_column="id",
)

rb_correspondences_d2l = matcher.match(
    df_left=good_dataset_name_2,
    df_right=good_dataset_name_3,
    candidates=blocker_d2l,
    comparators=comparators_d2l,
    weights=[0.35, 0.25, 0.25, 0.10, 0.05],
    threshold=0.75,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_m2d.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_good_dataset_name_1_good_dataset_name_2.csv",
    ),
    index=False,
)

rb_correspondences_m2l.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_good_dataset_name_1_good_dataset_name_3.csv",
    ),
    index=False,
)

rb_correspondences_d2l.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_good_dataset_name_2_good_dataset_name_3.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_m2d, rb_correspondences_m2l, rb_correspondences_d2l],
    ignore_index=True,
)

# improve fusion inputs before running engine
for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    if "name" in df.columns:
        df["name"] = df["name_norm"].where(df["name_norm"].notna(), df["name"])
    if "artist" in df.columns:
        df["artist"] = df["artist_norm"].where(df["artist_norm"].notna(), df["artist"])
    if "release-country" in df.columns:
        df["release-country"] = df["release_country_norm"].where(
            df["release_country_norm"].notna(), df["release-country"]
        )
    if "duration" in df.columns:
        df["duration"] = df["duration_clean"].where(df["duration_clean"].notna(), df["duration"])
    if "tracks_track_name" in df.columns:
        df["tracks_track_name"] = df["tracks_track_name_norm"].where(
            df["tracks_track_name_norm"].notna(), df["tracks_track_name"]
        )
    if "tracks_track_position" in df.columns:
        df["tracks_track_position"] = df["tracks_track_position_norm"].where(
            df["tracks_track_position_norm"].notna(), df["tracks_track_position"]
        )
    if "tracks_track_duration" in df.columns:
        df["tracks_track_duration"] = df["tracks_track_duration_norm"].where(
            df["tracks_track_duration_norm"].notna(), df["tracks_track_duration"]
        )

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("duration", longest_string)
strategy.add_attribute_fuser("tracks_track_name", longest_string)
strategy.add_attribute_fuser("tracks_track_duration", longest_string)
strategy.add_attribute_fuser("tracks_track_position", longest_string)

if "label" in good_dataset_name_1.columns or "label" in good_dataset_name_2.columns or "label" in good_dataset_name_3.columns:
    strategy.add_attribute_fuser("label", longest_string)

if "genre" in good_dataset_name_1.columns or "genre" in good_dataset_name_2.columns or "genre" in good_dataset_name_3.columns:
    strategy.add_attribute_fuser("genre", union)

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
node_index=15
node_name=execute_pipeline
accuracy_score=26.90%
------------------------------------------------------------

```python
from PyDI.io import load_xml

from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
)

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import re
from dotenv import load_dotenv
import os

# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

DATA_DIR = "input/datasets/music/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_xml(
    DATA_DIR + "musicbrainz.xml",
    name="musicbrainz",
)

good_dataset_name_2 = load_xml(
    DATA_DIR + "discogs.xml",
    name="discogs",
)

good_dataset_name_3 = load_xml(
    DATA_DIR + "lastfm.xml",
    name="lastfm",
)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

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

# Ensure optional columns exist in all datasets for stable fusion output
optional_columns = ["release-date", "release-country", "label", "genre", "tracks_track_duration"]
for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    for col in optional_columns:
        if col not in df.columns:
            df[col] = None

# --------------------------------
# Normalization helpers
# --------------------------------

def is_missing(value):
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return False
    try:
        return pd.isna(value)
    except Exception:
        return False

def normalize_text(value):
    if is_missing(value):
        return None
    if isinstance(value, (list, tuple, set)):
        value = " ".join([str(v) for v in value if not is_missing(v)])
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value

def normalize_text_loose(value):
    value = normalize_text(value)
    if value is None:
        return None
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

def normalize_artist(value):
    return normalize_text_loose(value)

def remove_artist_prefix_from_name(name_value, artist_value):
    name_norm = normalize_text_loose(name_value)
    artist_norm = normalize_text_loose(artist_value)
    if name_norm is None:
        return None
    if artist_norm is None:
        return name_norm
    patterns = [
        rf"^{re.escape(artist_norm)}\s+\-\s+",
        rf"^{re.escape(artist_norm)}\s*[:/]\s*",
        rf"^{re.escape(artist_norm)}\s+",
    ]
    cleaned = name_norm
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else name_norm

def normalize_name(value):
    return normalize_text_loose(value)

def normalize_release_country(value):
    value = normalize_text(value)
    if value is None:
        return None
    country_map = {
        "uk": "united kingdom",
        "u k": "united kingdom",
        "u.k.": "united kingdom",
        "great britain": "united kingdom",
        "england": "united kingdom",
        "united kingdom of great britain and northern ireland": "united kingdom",
        "us": "united states",
        "u s": "united states",
        "u.s.": "united states",
        "usa": "united states",
        "u.s.a.": "united states",
    }
    cleaned = re.sub(r"[^a-z0-9\s]", " ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return country_map.get(value, country_map.get(cleaned, cleaned))

def to_numeric_scalar(value):
    if is_missing(value):
        return None
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)

def to_numeric(value):
    if is_missing(value):
        return None
    if isinstance(value, (list, tuple, set)):
        nums = []
        for v in value:
            parsed = to_numeric_scalar(v)
            if parsed is not None:
                nums.append(parsed)
        if len(nums) == 0:
            return None
        return float(sum(nums))
    return to_numeric_scalar(value)

def normalize_duration(value):
    num = to_numeric(value)
    if num is None or float(num) <= 0:
        return None
    return str(int(round(num)))

def normalize_release_date(value):
    if is_missing(value):
        return None
    text = normalize_text(value)
    if text is None:
        return None
    match = re.match(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$", text)
    if match:
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)
        if month is None:
            return year
        if day is None:
            return f"{year}-{month}"
        return f"{year}-{month}-{day}"
    year_match = re.search(r"\b(19|20)\d{2}\b", text)
    if year_match:
        return year_match.group(0)
    return text

def normalize_track_names_list(value):
    if is_missing(value):
        return None
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    cleaned = []
    for v in value:
        nv = normalize_text_loose(v)
        if nv:
            cleaned.append(nv)
    if len(cleaned) == 0:
        return None
    return cleaned

def normalize_track_positions_list(value):
    if is_missing(value):
        return None
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    cleaned = []
    for v in value:
        parsed = pd.to_numeric(v, errors="coerce")
        if not pd.isna(parsed):
            cleaned.append(str(int(parsed)))
    if len(cleaned) == 0:
        return None
    return cleaned

def normalize_track_durations_list(value):
    if is_missing(value):
        return None
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    cleaned = []
    for v in value:
        parsed = pd.to_numeric(v, errors="coerce")
        if not pd.isna(parsed) and float(parsed) > 0:
            cleaned.append(str(int(parsed)))
    if len(cleaned) == 0:
        return None
    return cleaned

def join_track_list(values):
    if values is None:
        return None
    unique_vals = []
    seen = set()
    for v in values:
        if v not in seen:
            seen.add(v)
            unique_vals.append(v)
    if len(unique_vals) == 0:
        return None
    return " | ".join(unique_vals)

def build_block_key(row):
    artist = row.get("artist_norm")
    name = row.get("name_block")
    if is_missing(artist) or is_missing(name):
        return None
    return f"{artist}||{name}"

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    df["artist_norm"] = df["artist"].apply(normalize_artist)
    df["name_norm_raw"] = df["name"].apply(normalize_name)
    df["name_norm"] = df.apply(
        lambda r: remove_artist_prefix_from_name(r.get("name"), r.get("artist")),
        axis=1,
    )
    df["name_block"] = df["name_norm"].apply(
        lambda x: None if is_missing(x) else re.sub(r"\s+", " ", x).strip()
    )
    df["release_country_norm"] = df["release-country"].apply(normalize_release_country)
    df["duration_num"] = df["duration"].apply(to_numeric)
    df["duration_clean"] = df["duration"].apply(normalize_duration)
    df["release_date_norm"] = df["release-date"].apply(normalize_release_date)

    df["tracks_track_name_list"] = df["tracks_track_name"].apply(normalize_track_names_list)
    df["tracks_track_position_list"] = df["tracks_track_position"].apply(normalize_track_positions_list)
    df["tracks_track_duration_list"] = df["tracks_track_duration"].apply(normalize_track_durations_list)

    df["tracks_track_name_norm"] = df["tracks_track_name_list"].apply(join_track_list)
    df["tracks_track_position_norm"] = df["tracks_track_position_list"].apply(join_track_list)
    df["tracks_track_duration_norm"] = df["tracks_track_duration_list"].apply(join_track_list)

    df["track_count"] = df["tracks_track_position_list"].apply(
        lambda x: len(x) if isinstance(x, list) else (
            len(df["tracks_track_name_list"]) if isinstance(df.get("tracks_track_name_list"), list) else None
        )
    )
    df["artist_name_block"] = df.apply(build_block_key, axis=1)

print("Performing Blocking")

blocker_m2d = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_2,
    on=["artist_name_block"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_m2l = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    on=["artist_name_block"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_d2l = StandardBlocker(
    good_dataset_name_2,
    good_dataset_name_3,
    on=["artist_name_block"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

comparators_m2d = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="artist_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="tracks_track_name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="tracks_track_position_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="duration_num",
        max_difference=180,
    ),
]

comparators_m2l = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="artist_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="tracks_track_name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="tracks_track_position_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="duration_num",
        max_difference=180,
    ),
]

comparators_d2l = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="artist_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="tracks_track_name_norm",
        similarity_function="jaccard",
    ),
    StringComparator(
        column="tracks_track_position_norm",
        similarity_function="jaccard",
    ),
    NumericComparator(
        column="duration_num",
        max_difference=180,
    ),
]

print("Matching Entities")

matcher = RuleBasedMatcher()

rb_correspondences_m2d = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2,
    candidates=blocker_m2d,
    comparators=comparators_m2d,
    weights=[0.40, 0.25, 0.20, 0.10, 0.05],
    threshold=0.68,
    id_column="id",
)

rb_correspondences_m2l = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3,
    candidates=blocker_m2l,
    comparators=comparators_m2l,
    weights=[0.45, 0.25, 0.20, 0.05, 0.05],
    threshold=0.65,
    id_column="id",
)

rb_correspondences_d2l = matcher.match(
    df_left=good_dataset_name_2,
    df_right=good_dataset_name_3,
    candidates=blocker_d2l,
    comparators=comparators_d2l,
    weights=[0.45, 0.25, 0.20, 0.05, 0.05],
    threshold=0.65,
    id_column="id",
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_m2d.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_good_dataset_name_1_good_dataset_name_2.csv",
    ),
    index=False,
)

rb_correspondences_m2l.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_good_dataset_name_1_good_dataset_name_3.csv",
    ),
    index=False,
)

rb_correspondences_d2l.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_good_dataset_name_2_good_dataset_name_3.csv",
    ),
    index=False,
)

print("Fusing Data")

all_rb_correspondences = pd.concat(
    [rb_correspondences_m2d, rb_correspondences_m2l, rb_correspondences_d2l],
    ignore_index=True,
)

# Prepare normalized values for fusion
for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    df["name"] = df["name_norm"].where(df["name_norm"].notna(), df["name"])
    df["artist"] = df["artist_norm"].where(df["artist_norm"].notna(), df["artist"])
    df["release-country"] = df["release_country_norm"].where(
        df["release_country_norm"].notna(), df["release-country"]
    )
    df["release-date"] = df["release_date_norm"].where(
        df["release_date_norm"].notna(), df["release-date"]
    )
    df["tracks_track_name"] = df["tracks_track_name_norm"].where(
        df["tracks_track_name_norm"].notna(), df["tracks_track_name"]
    )
    df["tracks_track_position"] = df["tracks_track_position_norm"].where(
        df["tracks_track_position_norm"].notna(), df["tracks_track_position"]
    )
    df["tracks_track_duration"] = df["tracks_track_duration_norm"].where(
        df["tracks_track_duration_norm"].notna(), df["tracks_track_duration"]
    )
    df["duration"] = df["duration_clean"].where(df["duration_clean"].notna(), df["duration"])
    df["duration"] = df["duration"].replace("0", None)

strategy = DataFusionStrategy("rule_based_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("duration", longest_string)
strategy.add_attribute_fuser("tracks_track_name", longest_string)
strategy.add_attribute_fuser("tracks_track_duration", longest_string)
strategy.add_attribute_fuser("tracks_track_position", longest_string)
strategy.add_attribute_fuser("label", longest_string)
strategy.add_attribute_fuser("genre", union)

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

