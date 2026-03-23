# Pipeline Snapshots

notebook_name=Agent VII
matcher_mode=rule_based

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=7
node_name=execute_pipeline
accuracy_score=pending
------------------------------------------------------------

```python
# CRITICAL: Do not adjust output file names.
# CRITICAL: Always use OUTPUT_DIR for all output paths (correspondences, fusion, debug).

from PyDI.io import load_csv
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
    shortest_string,
    longest_string,
    most_complete,
    maximum,
    union,
    prefer_higher_trust,
    favour_sources,
)

import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

def _safe_scalar_isna(x):
    """Safe pd.isna() that handles list/array values without crashing."""
    if isinstance(x, (list, tuple, set)):
        return False  # a list is not null, even if it contains null elements
    try:
        result = pd.isna(x)
        if hasattr(result, '__len__') and not isinstance(result, str):
            return False  # array-like result from pd.isna on non-scalar
        return bool(result)
    except (ValueError, TypeError):
        return False

try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _candidates = [
        Path.cwd(),
        Path.cwd() / "agents",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
    ]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            if str(_path.resolve()) not in sys.path:
                sys.path.append(str(_path.resolve()))
    from list_normalization import detect_list_like_columns, normalize_list_like_columns

# === 0. OUTPUT DIRECTORY ===
OUTPUT_DIR = "output/runs/20260323_005631_music/"

# === 1. LOAD DATA ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

discogs = load_csv(
    "output/runs/20260323_005631_music/normalization/attempt_1/discogs.csv",
    name="discogs",
)
lastfm = load_csv(
    "output/runs/20260323_005631_music/normalization/attempt_1/lastfm.csv",
    name="lastfm",
)
musicbrainz = load_csv(
    "output/runs/20260323_005631_music/normalization/attempt_1/musicbrainz.csv",
    name="musicbrainz",
)

# === 2. TARGETED INLINE NORMALIZATION ===
# Keep IDs unchanged. Only normalize scalar text columns and safely coerce duration/date fields.

def lower_strip(x):
    if isinstance(x, (list, tuple, set)):
        return " ".join(str(v).lower().strip() for v in x if v is not None)
    return str(x).lower().strip()

for df in [discogs, lastfm, musicbrainz]:
    for col in ["name", "artist", "release-country", "label", "genre"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

for df in [discogs, lastfm, musicbrainz]:
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

for df in [discogs, musicbrainz]:
    if "release-date" in df.columns:
        df["release-date"] = pd.to_datetime(df["release-date"], errors="coerce")

datasets = [discogs, lastfm, musicbrainz]

# === 3. LIST NORMALIZATION ===
list_like_columns = detect_list_like_columns(
    [discogs, lastfm, musicbrainz],
    exclude_columns={"id", "_id"},
)
if list_like_columns:
    discogs, lastfm, musicbrainz = normalize_list_like_columns(
        [discogs, lastfm, musicbrainz],
        list_like_columns,
    )
    print(f"Normalized list-like columns: {', '.join(list_like_columns)}")

datasets = [discogs, lastfm, musicbrainz]

# === 4. BLOCKING ===
print("Performing Blocking")

def _flatten_list_cols_for_blocking(df, text_cols):
    """Flatten list-valued cells to strings so EmbeddingBlocker can embed them."""
    out = df.copy()
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: " ".join(str(v) for v in x) if isinstance(x, (list, tuple, set))
                else ("" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))
            )
    return out

blocker_discogs_lastfm = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(discogs, ["name", "artist"]),
    _flatten_list_cols_for_blocking(lastfm, ["name", "artist"]),
    text_cols=["name", "artist"],
    id_column="id",
    top_k=10,
)
candidates_discogs_lastfm = blocker_discogs_lastfm.materialize()

blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    id_column="id",
    window=15,
)
candidates_discogs_musicbrainz = blocker_discogs_musicbrainz.materialize()

blocker_lastfm_musicbrainz = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(lastfm, ["name", "artist", "duration"]),
    _flatten_list_cols_for_blocking(musicbrainz, ["name", "artist", "duration"]),
    text_cols=["name", "artist", "duration"],
    id_column="id",
    top_k=10,
)
candidates_lastfm_musicbrainz = blocker_lastfm_musicbrainz.materialize()

# === 5. ENTITY MATCHING ===
print("Matching Entities")

threshold_discogs_lastfm = 0.6
threshold_discogs_musicbrainz = 0.7
threshold_musicbrainz_lastfm = 0.75

comparators_discogs_lastfm = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        method="absolute_difference",
        max_difference=10.0,
        list_strategy="average",
    ),
]

comparators_discogs_musicbrainz = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    DateComparator(
        column="release-date",
        max_days_difference=365,
        list_strategy="closest_dates",
    ),
    StringComparator(
        column="release-country",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    NumericComparator(
        column="duration",
        method="absolute_difference",
        max_difference=60.0,
        list_strategy="average",
    ),
]

comparators_musicbrainz_lastfm = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="duration",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
]

matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=candidates_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=[0.3, 0.35, 0.25, 0.1],
    threshold=threshold_discogs_lastfm,
    id_column="id",
)

rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=candidates_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=[0.35, 0.25, 0.2, 0.1, 0.1],
    threshold=threshold_discogs_musicbrainz,
    id_column="id",
)

rb_correspondences_lastfm_musicbrainz = matcher.match(
    df_left=lastfm,
    df_right=musicbrainz,
    candidates=candidates_lastfm_musicbrainz,
    comparators=comparators_musicbrainz_lastfm,
    weights=[0.35, 0.45, 0.2],
    threshold=threshold_musicbrainz_lastfm,
    id_column="id",
)

if rb_correspondences_discogs_lastfm.empty:
    raise ValueError("No correspondences found for discogs_lastfm; aborting before fusion.")
if rb_correspondences_discogs_musicbrainz.empty:
    raise ValueError("No correspondences found for discogs_musicbrainz; aborting before fusion.")
if rb_correspondences_lastfm_musicbrainz.empty:
    raise ValueError("No correspondences found for lastfm_musicbrainz; aborting before fusion.")

# === 6. SAVE CORRESPONDENCES (MANDATORY) ===
CORR_DIR = os.path.join(OUTPUT_DIR, "correspondences")
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_discogs_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)
rb_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)
rb_correspondences_lastfm_musicbrainz.to_csv(
    os.path.join(CORR_DIR, "correspondences_lastfm_musicbrainz.csv"),
    index=False,
)

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_lastfm_musicbrainz,
    ],
    ignore_index=True,
)

# === 7. DATA FUSION ===
print("Fusing Data")

trust_map = {
    "musicbrainz": 3,
    "discogs": 2,
    "lastfm": 1,
}

strategy = DataFusionStrategy("fusion_strategy")

strategy.add_attribute_fuser("name", shortest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("release-country", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("duration", maximum)
strategy.add_attribute_fuser("label", most_complete)
strategy.add_attribute_fuser("genre", most_complete)

strategy.add_attribute_fuser(
    "tracks_track_name",
    favour_sources,
    source_preferences=["musicbrainz", "discogs", "lastfm"],
)
strategy.add_attribute_fuser(
    "tracks_track_position",
    favour_sources,
    source_preferences=["musicbrainz", "discogs", "lastfm"],
)
strategy.add_attribute_fuser(
    "tracks_track_duration",
    favour_sources,
    source_preferences=["musicbrainz", "lastfm", "discogs"],
)

# === 8. RUN FUSION ===
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
os.makedirs(FUSION_DIR, exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=os.path.join(FUSION_DIR, "debug_fusion_data.jsonl"),
)

fused_result = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=True,
)


# --- Eval ID: extract source ID matching validation prefix(es) for reliable evaluation ---
import ast as _ast
_EVAL_PREFIXES = ['mbrainz_']
def _extract_eval_id(row):
    try:
        sources = _ast.literal_eval(str(row.get("_fusion_sources", "[]")))
        if isinstance(sources, (list, tuple)):
            for sid in sources:
                s = str(sid)
                if any(s.startswith(p) for p in _EVAL_PREFIXES):
                    return s
    except Exception:
        pass
    return str(row.get("_id", row.get("id", "")))
fused_result["eval_id"] = fused_result.apply(_extract_eval_id, axis=1)
fused_result.to_csv(os.path.join(FUSION_DIR, "fusion_data.csv"), index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=11
node_name=execute_pipeline
accuracy_score=pending
------------------------------------------------------------

```python
# CRITICAL: Do not adjust output file names.
# CRITICAL: Always use OUTPUT_DIR for all output paths (correspondences, fusion, debug).

from PyDI.io import load_csv
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
    shortest_string,
    longest_string,
    most_complete,
    maximum,
    union,
    prefer_higher_trust,
    favour_sources,
)

import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

def _safe_scalar_isna(x):
    """Safe pd.isna() that handles list/array values without crashing."""
    if isinstance(x, (list, tuple, set)):
        return False  # a list is not null, even if it contains null elements
    try:
        result = pd.isna(x)
        if hasattr(result, '__len__') and not isinstance(result, str):
            return False  # array-like result from pd.isna on non-scalar
        return bool(result)
    except (ValueError, TypeError):
        return False

try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _candidates = [
        Path.cwd(),
        Path.cwd() / "agents",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
    ]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            if str(_path.resolve()) not in sys.path:
                sys.path.append(str(_path.resolve()))
    from list_normalization import detect_list_like_columns, normalize_list_like_columns

# === 0. OUTPUT DIRECTORY ===
OUTPUT_DIR = "output/runs/20260323_005631_music/"

# === 1. LOAD DATA ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

discogs = load_csv(
    "output/runs/20260323_005631_music/normalization/attempt_1/discogs.csv",
    name="discogs",
)
lastfm = load_csv(
    "output/runs/20260323_005631_music/normalization/attempt_1/lastfm.csv",
    name="lastfm",
)
musicbrainz = load_csv(
    "output/runs/20260323_005631_music/normalization/attempt_1/musicbrainz.csv",
    name="musicbrainz",
)

# === 2. TARGETED INLINE NORMALIZATION ===
# Keep IDs unchanged. Only normalize scalar text columns and safely coerce duration/date fields.

def lower_strip(x):
    if isinstance(x, (list, tuple, set)):
        return " ".join(str(v).lower().strip() for v in x if v is not None)
    return str(x).lower().strip()

for df in [discogs, lastfm, musicbrainz]:
    for col in ["name", "artist", "release-country", "label", "genre"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

for df in [discogs, lastfm, musicbrainz]:
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

for df in [discogs, musicbrainz]:
    if "release-date" in df.columns:
        df["release-date"] = pd.to_datetime(df["release-date"], errors="coerce")

datasets = [discogs, lastfm, musicbrainz]

# === 3. LIST NORMALIZATION ===
list_like_columns = detect_list_like_columns(
    [discogs, lastfm, musicbrainz],
    exclude_columns={"id", "_id"},
)
if list_like_columns:
    discogs, lastfm, musicbrainz = normalize_list_like_columns(
        [discogs, lastfm, musicbrainz],
        list_like_columns,
    )
    print(f"Normalized list-like columns: {', '.join(list_like_columns)}")

datasets = [discogs, lastfm, musicbrainz]

# === 4. BLOCKING ===
print("Performing Blocking")

def _flatten_list_cols_for_blocking(df, text_cols):
    """Flatten list-valued cells to strings so EmbeddingBlocker can embed them."""
    out = df.copy()
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: " ".join(str(v) for v in x) if isinstance(x, (list, tuple, set))
                else ("" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))
            )
    return out

blocker_discogs_lastfm = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(discogs, ["name", "artist"]),
    _flatten_list_cols_for_blocking(lastfm, ["name", "artist"]),
    text_cols=["name", "artist"],
    id_column="id",
    top_k=10,
)
candidates_discogs_lastfm = blocker_discogs_lastfm.materialize()

blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    id_column="id",
    window=15,
)
candidates_discogs_musicbrainz = blocker_discogs_musicbrainz.materialize()

blocker_lastfm_musicbrainz = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(lastfm, ["name", "artist", "duration"]),
    _flatten_list_cols_for_blocking(musicbrainz, ["name", "artist", "duration"]),
    text_cols=["name", "artist", "duration"],
    id_column="id",
    top_k=10,
)
candidates_lastfm_musicbrainz = blocker_lastfm_musicbrainz.materialize()

# === 5. ENTITY MATCHING ===
print("Matching Entities")

threshold_discogs_lastfm = 0.6
threshold_discogs_musicbrainz = 0.7
threshold_musicbrainz_lastfm = 0.75

comparators_discogs_lastfm = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        method="absolute_difference",
        max_difference=10.0,
        list_strategy="average",
    ),
]

comparators_discogs_musicbrainz = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    DateComparator(
        column="release-date",
        max_days_difference=365,
        list_strategy="closest_dates",
    ),
    StringComparator(
        column="release-country",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    NumericComparator(
        column="duration",
        method="absolute_difference",
        max_difference=60.0,
        list_strategy="average",
    ),
]

comparators_musicbrainz_lastfm = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="duration",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
]

matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=candidates_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=[0.3, 0.35, 0.25, 0.1],
    threshold=threshold_discogs_lastfm,
    id_column="id",
)

rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=candidates_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=[0.35, 0.25, 0.2, 0.1, 0.1],
    threshold=threshold_discogs_musicbrainz,
    id_column="id",
)

rb_correspondences_lastfm_musicbrainz = matcher.match(
    df_left=lastfm,
    df_right=musicbrainz,
    candidates=candidates_lastfm_musicbrainz,
    comparators=comparators_musicbrainz_lastfm,
    weights=[0.35, 0.45, 0.2],
    threshold=threshold_musicbrainz_lastfm,
    id_column="id",
)

if rb_correspondences_discogs_lastfm.empty:
    raise ValueError("No correspondences found for discogs_lastfm; aborting before fusion.")
if rb_correspondences_discogs_musicbrainz.empty:
    raise ValueError("No correspondences found for discogs_musicbrainz; aborting before fusion.")
if rb_correspondences_lastfm_musicbrainz.empty:
    raise ValueError("No correspondences found for lastfm_musicbrainz; aborting before fusion.")

# === 6. SAVE CORRESPONDENCES (MANDATORY) ===
CORR_DIR = os.path.join(OUTPUT_DIR, "correspondences")
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_discogs_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)
rb_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)
rb_correspondences_lastfm_musicbrainz.to_csv(
    os.path.join(CORR_DIR, "correspondences_lastfm_musicbrainz.csv"),
    index=False,
)

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_lastfm_musicbrainz,
    ],
    ignore_index=True,
)
from PyDI.entitymatching import MaximumBipartiteMatching



# === 7. POST-CLUSTERING ===
clusterer = MaximumBipartiteMatching()
refined_discogs_lastfm = clusterer.cluster(rb_correspondences_discogs_lastfm)
refined_discogs_musicbrainz = clusterer.cluster(rb_correspondences_discogs_musicbrainz)
refined_lastfm_musicbrainz = clusterer.cluster(rb_correspondences_lastfm_musicbrainz)

all_rb_correspondences = pd.concat(
    [
        refined_discogs_lastfm,
        refined_discogs_musicbrainz,
        refined_lastfm_musicbrainz,
    ],
    ignore_index=True,
)

# === 8. DATA FUSION ===
print("Fusing Data")

trust_map = {
    "musicbrainz": 3,
    "discogs": 2,
    "lastfm": 1,
}

strategy = DataFusionStrategy("fusion_strategy")

trust_map_label = {"discogs": 3, "musicbrainz": 2, "lastfm": 1}
trust_map_artist = {"musicbrainz": 3, "discogs": 2, "lastfm": 1}
trust_map_duration = {"discogs": 3, "musicbrainz": 2, "lastfm": 1}
strategy.add_attribute_fuser("name", shortest_string)
strategy.add_attribute_fuser("artist", prefer_higher_trust, trust_map=trust_map_artist)
strategy.add_attribute_fuser(
    "release-date",
    favour_sources,
    source_preferences=["discogs", "musicbrainz", "lastfm"],
)
strategy.add_attribute_fuser("release-country", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser(
    "duration",
    prefer_higher_trust,
    trust_map={"discogs": 3, "musicbrainz": 2, "lastfm": 1},
)
strategy.add_attribute_fuser(
    "label",
    prefer_higher_trust,
    trust_map={"discogs": 3, "musicbrainz": 2, "lastfm": 1},
)
strategy.add_attribute_fuser("genre", most_complete)

strategy.add_attribute_fuser(
    "tracks_track_name",
    favour_sources,
    source_preferences=["lastfm", "musicbrainz", "discogs"],
)
strategy.add_attribute_fuser(
    "tracks_track_position",
    favour_sources,
    source_preferences=["musicbrainz", "discogs", "lastfm"],
)
strategy.add_attribute_fuser(
    "tracks_track_duration",
    favour_sources,
    source_preferences=["discogs", "musicbrainz", "lastfm"],
)




# === 8. RUN FUSION ===
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
os.makedirs(FUSION_DIR, exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=os.path.join(FUSION_DIR, "debug_fusion_data.jsonl"),
)

fused_result = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=True,
)


# --- Eval ID: extract source ID matching validation prefix(es) for reliable evaluation ---
import ast as _ast
_EVAL_PREFIXES = ['mbrainz_']
def _extract_eval_id(row):
    try:
        sources = _ast.literal_eval(str(row.get("_fusion_sources", "[]")))
        if isinstance(sources, (list, tuple)):
            for sid in sources:
                s = str(sid)
                if any(s.startswith(p) for p in _EVAL_PREFIXES):
                    return s
    except Exception:
        pass
    return str(row.get("_id", row.get("id", "")))
fused_result["eval_id"] = fused_result.apply(_extract_eval_id, axis=1)
fused_result.to_csv(os.path.join(FUSION_DIR, "fusion_data.csv"), index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=15
node_name=execute_pipeline
accuracy_score=pending
------------------------------------------------------------

```python
# CRITICAL: Do not adjust output file names.
# CRITICAL: Always use OUTPUT_DIR for all output paths (correspondences, fusion, debug).

from PyDI.io import load_csv
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
    shortest_string,
    longest_string,
    most_complete,
    maximum,
    union,
    prefer_higher_trust,
    favour_sources,
)

import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

def _safe_scalar_isna(x):
    """Safe pd.isna() that handles list/array values without crashing."""
    if isinstance(x, (list, tuple, set)):
        return False  # a list is not null, even if it contains null elements
    try:
        result = pd.isna(x)
        if hasattr(result, '__len__') and not isinstance(result, str):
            return False  # array-like result from pd.isna on non-scalar
        return bool(result)
    except (ValueError, TypeError):
        return False

try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _candidates = [
        Path.cwd(),
        Path.cwd() / "agents",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
    ]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            if str(_path.resolve()) not in sys.path:
                sys.path.append(str(_path.resolve()))
    from list_normalization import detect_list_like_columns, normalize_list_like_columns

# === 0. OUTPUT DIRECTORY ===
OUTPUT_DIR = "output/runs/20260323_005631_music/"

# === 1. LOAD DATA ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

discogs = load_csv(
    "output/runs/20260323_005631_music/normalization/attempt_1/discogs.csv",
    name="discogs",
)
lastfm = load_csv(
    "output/runs/20260323_005631_music/normalization/attempt_1/lastfm.csv",
    name="lastfm",
)
musicbrainz = load_csv(
    "output/runs/20260323_005631_music/normalization/attempt_1/musicbrainz.csv",
    name="musicbrainz",
)

# === 2. TARGETED INLINE NORMALIZATION ===
# Keep IDs unchanged. Only normalize scalar text columns and safely coerce duration/date fields.

def lower_strip(x):
    if isinstance(x, (list, tuple, set)):
        return " ".join(str(v).lower().strip() for v in x if v is not None)
    return str(x).lower().strip()

for df in [discogs, lastfm, musicbrainz]:
    for col in ["name", "artist", "release-country", "label", "genre"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

for df in [discogs, lastfm, musicbrainz]:
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

for df in [discogs, musicbrainz]:
    if "release-date" in df.columns:
        df["release-date"] = pd.to_datetime(df["release-date"], errors="coerce")

datasets = [discogs, lastfm, musicbrainz]

# === 3. LIST NORMALIZATION ===
list_like_columns = detect_list_like_columns(
    [discogs, lastfm, musicbrainz],
    exclude_columns={"id", "_id"},
)
if list_like_columns:
    discogs, lastfm, musicbrainz = normalize_list_like_columns(
        [discogs, lastfm, musicbrainz],
        list_like_columns,
    )
    print(f"Normalized list-like columns: {', '.join(list_like_columns)}")

datasets = [discogs, lastfm, musicbrainz]

# === 4. BLOCKING ===
print("Performing Blocking")

def _flatten_list_cols_for_blocking(df, text_cols):
    """Flatten list-valued cells to strings so EmbeddingBlocker can embed them."""
    out = df.copy()
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: " ".join(str(v) for v in x) if isinstance(x, (list, tuple, set))
                else ("" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))
            )
    return out

blocker_discogs_lastfm = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(discogs, ["name", "artist"]),
    _flatten_list_cols_for_blocking(lastfm, ["name", "artist"]),
    text_cols=["name", "artist"],
    id_column="id",
    top_k=10,
)
candidates_discogs_lastfm = blocker_discogs_lastfm.materialize()

blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    id_column="id",
    window=15,
)
candidates_discogs_musicbrainz = blocker_discogs_musicbrainz.materialize()

blocker_lastfm_musicbrainz = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(lastfm, ["name", "artist", "duration"]),
    _flatten_list_cols_for_blocking(musicbrainz, ["name", "artist", "duration"]),
    text_cols=["name", "artist", "duration"],
    id_column="id",
    top_k=10,
)
candidates_lastfm_musicbrainz = blocker_lastfm_musicbrainz.materialize()

# === 5. ENTITY MATCHING ===
print("Matching Entities")

threshold_discogs_lastfm = 0.6
threshold_discogs_musicbrainz = 0.7
threshold_musicbrainz_lastfm = 0.75

comparators_discogs_lastfm = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        method="absolute_difference",
        max_difference=10.0,
        list_strategy="average",
    ),
]

comparators_discogs_musicbrainz = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    DateComparator(
        column="release-date",
        max_days_difference=365,
        list_strategy="closest_dates",
    ),
    StringComparator(
        column="release-country",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    NumericComparator(
        column="duration",
        method="absolute_difference",
        max_difference=60.0,
        list_strategy="average",
    ),
]

comparators_musicbrainz_lastfm = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="duration",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
]

matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=candidates_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=[0.3, 0.35, 0.25, 0.1],
    threshold=threshold_discogs_lastfm,
    id_column="id",
)

rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=candidates_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=[0.35, 0.25, 0.2, 0.1, 0.1],
    threshold=threshold_discogs_musicbrainz,
    id_column="id",
)

rb_correspondences_lastfm_musicbrainz = matcher.match(
    df_left=lastfm,
    df_right=musicbrainz,
    candidates=candidates_lastfm_musicbrainz,
    comparators=comparators_musicbrainz_lastfm,
    weights=[0.35, 0.45, 0.2],
    threshold=threshold_musicbrainz_lastfm,
    id_column="id",
)

if rb_correspondences_discogs_lastfm.empty:
    raise ValueError("No correspondences found for discogs_lastfm; aborting before fusion.")
if rb_correspondences_discogs_musicbrainz.empty:
    raise ValueError("No correspondences found for discogs_musicbrainz; aborting before fusion.")
if rb_correspondences_lastfm_musicbrainz.empty:
    raise ValueError("No correspondences found for lastfm_musicbrainz; aborting before fusion.")

# === 6. SAVE CORRESPONDENCES (MANDATORY) ===
CORR_DIR = os.path.join(OUTPUT_DIR, "correspondences")
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_discogs_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)
rb_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)
rb_correspondences_lastfm_musicbrainz.to_csv(
    os.path.join(CORR_DIR, "correspondences_lastfm_musicbrainz.csv"),
    index=False,
)

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_lastfm_musicbrainz,
    ],
    ignore_index=True,
)
from PyDI.entitymatching import MaximumBipartiteMatching



# === 7. POST-CLUSTERING ===
clusterer = MaximumBipartiteMatching()
refined_discogs_lastfm = clusterer.cluster(rb_correspondences_discogs_lastfm)
refined_discogs_musicbrainz = clusterer.cluster(rb_correspondences_discogs_musicbrainz)
refined_lastfm_musicbrainz = clusterer.cluster(rb_correspondences_lastfm_musicbrainz)

all_rb_correspondences = pd.concat(
    [
        refined_discogs_lastfm,
        refined_discogs_musicbrainz,
        refined_lastfm_musicbrainz,
    ],
    ignore_index=True,
)

# === 8. DATA FUSION ===
print("Fusing Data")

trust_map = {
    "discogs": 3.000,
    "musicbrainz": 2.000,
    "lastfm": 1.000,
}

strategy = DataFusionStrategy("fusion_strategy")

strategy.add_attribute_fuser("name", shortest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser(
    "release-date",
    favour_sources,
    source_preferences=["discogs", "lastfm", "musicbrainz"],
)
strategy.add_attribute_fuser("release-country", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("duration", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("label", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("genre", most_complete)

strategy.add_attribute_fuser(
    "tracks_track_name",
    favour_sources,
    source_preferences=["musicbrainz", "discogs", "lastfm"],
)
strategy.add_attribute_fuser(
    "tracks_track_position",
    favour_sources,
    source_preferences=["musicbrainz", "discogs", "lastfm"],
)
strategy.add_attribute_fuser(
    "tracks_track_duration",
    favour_sources,
    source_preferences=["musicbrainz", "lastfm", "discogs"],
)




# === 8. RUN FUSION ===
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
os.makedirs(FUSION_DIR, exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=os.path.join(FUSION_DIR, "debug_fusion_data.jsonl"),
)

fused_result = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=True,
)


# --- Eval ID: extract source ID matching validation prefix(es) for reliable evaluation ---
import ast as _ast
_EVAL_PREFIXES = ['mbrainz_']
def _extract_eval_id(row):
    try:
        sources = _ast.literal_eval(str(row.get("_fusion_sources", "[]")))
        if isinstance(sources, (list, tuple)):
            for sid in sources:
                s = str(sid)
                if any(s.startswith(p) for p in _EVAL_PREFIXES):
                    return s
    except Exception:
        pass
    return str(row.get("_id", row.get("id", "")))
fused_result["eval_id"] = fused_result.apply(_extract_eval_id, axis=1)
fused_result.to_csv(os.path.join(FUSION_DIR, "fusion_data.csv"), index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

============================================================
PIPELINE SNAPSHOT 04 START
============================================================
node_index=19
node_name=execute_pipeline
accuracy_score=pending
------------------------------------------------------------

```python
# CRITICAL: Do not adjust output file names.
# CRITICAL: Always use OUTPUT_DIR for all output paths (correspondences, fusion, debug).

from PyDI.io import load_csv
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
    shortest_string,
    longest_string,
    most_complete,
    maximum,
    union,
    prefer_higher_trust,
    favour_sources,
)

import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

def _safe_scalar_isna(x):
    """Safe pd.isna() that handles list/array values without crashing."""
    if isinstance(x, (list, tuple, set)):
        return False  # a list is not null, even if it contains null elements
    try:
        result = pd.isna(x)
        if hasattr(result, '__len__') and not isinstance(result, str):
            return False  # array-like result from pd.isna on non-scalar
        return bool(result)
    except (ValueError, TypeError):
        return False

try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _candidates = [
        Path.cwd(),
        Path.cwd() / "agents",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
    ]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            if str(_path.resolve()) not in sys.path:
                sys.path.append(str(_path.resolve()))
    from list_normalization import detect_list_like_columns, normalize_list_like_columns

# === 0. OUTPUT DIRECTORY ===
OUTPUT_DIR = "output/runs/20260323_005631_music/"

# === 1. LOAD DATA ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

discogs = load_csv(
    "output/runs/20260323_005631_music/normalization/attempt_1/discogs.csv",
    name="discogs",
)
lastfm = load_csv(
    "output/runs/20260323_005631_music/normalization/attempt_1/lastfm.csv",
    name="lastfm",
)
musicbrainz = load_csv(
    "output/runs/20260323_005631_music/normalization/attempt_1/musicbrainz.csv",
    name="musicbrainz",
)

# === 2. TARGETED INLINE NORMALIZATION ===
# Keep IDs unchanged. Only normalize scalar text columns and safely coerce duration/date fields.

def lower_strip(x):
    if isinstance(x, (list, tuple, set)):
        return " ".join(str(v).lower().strip() for v in x if v is not None)
    return str(x).lower().strip()

for df in [discogs, lastfm, musicbrainz]:
    for col in ["name", "artist", "release-country", "label", "genre"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

for df in [discogs, lastfm, musicbrainz]:
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

for df in [discogs, musicbrainz]:
    if "release-date" in df.columns:
        df["release-date"] = pd.to_datetime(df["release-date"], errors="coerce")

datasets = [discogs, lastfm, musicbrainz]

# === 3. LIST NORMALIZATION ===
list_like_columns = detect_list_like_columns(
    [discogs, lastfm, musicbrainz],
    exclude_columns={"id", "_id"},
)
if list_like_columns:
    discogs, lastfm, musicbrainz = normalize_list_like_columns(
        [discogs, lastfm, musicbrainz],
        list_like_columns,
    )
    print(f"Normalized list-like columns: {', '.join(list_like_columns)}")

datasets = [discogs, lastfm, musicbrainz]

# === 4. BLOCKING ===
print("Performing Blocking")

def _flatten_list_cols_for_blocking(df, text_cols):
    """Flatten list-valued cells to strings so EmbeddingBlocker can embed them."""
    out = df.copy()
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: " ".join(str(v) for v in x) if isinstance(x, (list, tuple, set))
                else ("" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))
            )
    return out

blocker_discogs_lastfm = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(discogs, ["name", "artist"]),
    _flatten_list_cols_for_blocking(lastfm, ["name", "artist"]),
    text_cols=["name", "artist"],
    id_column="id",
    top_k=10,
)
candidates_discogs_lastfm = blocker_discogs_lastfm.materialize()

blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    id_column="id",
    window=15,
)
candidates_discogs_musicbrainz = blocker_discogs_musicbrainz.materialize()

blocker_lastfm_musicbrainz = EmbeddingBlocker(
    _flatten_list_cols_for_blocking(lastfm, ["name", "artist", "duration"]),
    _flatten_list_cols_for_blocking(musicbrainz, ["name", "artist", "duration"]),
    text_cols=["name", "artist", "duration"],
    id_column="id",
    top_k=10,
)
candidates_lastfm_musicbrainz = blocker_lastfm_musicbrainz.materialize()

# === 5. ENTITY MATCHING ===
print("Matching Entities")

threshold_discogs_lastfm = 0.6
threshold_discogs_musicbrainz = 0.7
threshold_musicbrainz_lastfm = 0.75

comparators_discogs_lastfm = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        method="absolute_difference",
        max_difference=10.0,
        list_strategy="average",
    ),
]

comparators_discogs_musicbrainz = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    DateComparator(
        column="release-date",
        max_days_difference=365,
        list_strategy="closest_dates",
    ),
    StringComparator(
        column="release-country",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    NumericComparator(
        column="duration",
        method="absolute_difference",
        max_difference=60.0,
        list_strategy="average",
    ),
]

comparators_musicbrainz_lastfm = [
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
    StringComparator(
        column="duration",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="best_match",
    ),
]

matcher = RuleBasedMatcher()

rb_correspondences_discogs_lastfm = matcher.match(
    df_left=discogs,
    df_right=lastfm,
    candidates=candidates_discogs_lastfm,
    comparators=comparators_discogs_lastfm,
    weights=[0.3, 0.35, 0.25, 0.1],
    threshold=threshold_discogs_lastfm,
    id_column="id",
)

rb_correspondences_discogs_musicbrainz = matcher.match(
    df_left=discogs,
    df_right=musicbrainz,
    candidates=candidates_discogs_musicbrainz,
    comparators=comparators_discogs_musicbrainz,
    weights=[0.35, 0.25, 0.2, 0.1, 0.1],
    threshold=threshold_discogs_musicbrainz,
    id_column="id",
)

rb_correspondences_lastfm_musicbrainz = matcher.match(
    df_left=lastfm,
    df_right=musicbrainz,
    candidates=candidates_lastfm_musicbrainz,
    comparators=comparators_musicbrainz_lastfm,
    weights=[0.35, 0.45, 0.2],
    threshold=threshold_musicbrainz_lastfm,
    id_column="id",
)

if rb_correspondences_discogs_lastfm.empty:
    raise ValueError("No correspondences found for discogs_lastfm; aborting before fusion.")
if rb_correspondences_discogs_musicbrainz.empty:
    raise ValueError("No correspondences found for discogs_musicbrainz; aborting before fusion.")
if rb_correspondences_lastfm_musicbrainz.empty:
    raise ValueError("No correspondences found for lastfm_musicbrainz; aborting before fusion.")

# === 6. SAVE CORRESPONDENCES (MANDATORY) ===
CORR_DIR = os.path.join(OUTPUT_DIR, "correspondences")
os.makedirs(CORR_DIR, exist_ok=True)

rb_correspondences_discogs_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)
rb_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)
rb_correspondences_lastfm_musicbrainz.to_csv(
    os.path.join(CORR_DIR, "correspondences_lastfm_musicbrainz.csv"),
    index=False,
)

all_rb_correspondences = pd.concat(
    [
        rb_correspondences_discogs_lastfm,
        rb_correspondences_discogs_musicbrainz,
        rb_correspondences_lastfm_musicbrainz,
    ],
    ignore_index=True,
)
from PyDI.entitymatching import MaximumBipartiteMatching



# === 7. POST-CLUSTERING ===
clusterer = MaximumBipartiteMatching()
refined_discogs_lastfm = clusterer.cluster(rb_correspondences_discogs_lastfm)
refined_discogs_musicbrainz = clusterer.cluster(rb_correspondences_discogs_musicbrainz)
refined_lastfm_musicbrainz = clusterer.cluster(rb_correspondences_lastfm_musicbrainz)

all_rb_correspondences = pd.concat(
    [
        refined_discogs_lastfm,
        refined_discogs_musicbrainz,
        refined_lastfm_musicbrainz,
    ],
    ignore_index=True,
)

# === 8. DATA FUSION ===
print("Fusing Data")

trust_map = {
    "musicbrainz": 3,
    "discogs": 2,
    "lastfm": 1,
}

label_trust_map = {
    "discogs": 3,
    "musicbrainz": 2,
    "lastfm": 1,
}

strategy = DataFusionStrategy("fusion_strategy")

trust_map_release_country = {"musicbrainz": 3, "discogs": 2, "lastfm": 1}
trust_map_label = {"discogs": 3, "musicbrainz": 2, "lastfm": 1}
trust_map_duration = {"discogs": 3, "lastfm": 2, "musicbrainz": 1}
strategy.add_attribute_fuser("name", shortest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser(
    "release-date",
    favour_sources,
    source_preferences=["discogs", "lastfm", "musicbrainz"],
)
strategy.add_attribute_fuser("release-country", prefer_higher_trust, trust_map=trust_map_release_country)
strategy.add_attribute_fuser("duration", maximum)
strategy.add_attribute_fuser("label", prefer_higher_trust, trust_map=trust_map_label)
strategy.add_attribute_fuser("genre", most_complete)

strategy.add_attribute_fuser(
    "tracks_track_name",
    favour_sources,
    source_preferences=["musicbrainz", "discogs", "lastfm"],
)
strategy.add_attribute_fuser(
    "tracks_track_position",
    favour_sources,
    source_preferences=["musicbrainz", "discogs", "lastfm"],
)
strategy.add_attribute_fuser(
    "tracks_track_duration",
    favour_sources,
    source_preferences=["musicbrainz", "lastfm", "discogs"],
)




# === 8. RUN FUSION ===
FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
os.makedirs(FUSION_DIR, exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file=os.path.join(FUSION_DIR, "debug_fusion_data.jsonl"),
)

fused_result = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=True,
)


# --- Eval ID: extract source ID matching validation prefix(es) for reliable evaluation ---
import ast as _ast
_EVAL_PREFIXES = ['mbrainz_']
def _extract_eval_id(row):
    try:
        sources = _ast.literal_eval(str(row.get("_fusion_sources", "[]")))
        if isinstance(sources, (list, tuple)):
            for sid in sources:
                s = str(sid)
                if any(s.startswith(p) for p in _EVAL_PREFIXES):
                    return s
    except Exception:
        pass
    return str(row.get("_id", row.get("id", "")))
fused_result["eval_id"] = fused_result.apply(_extract_eval_id, axis=1)
fused_result.to_csv(os.path.join(FUSION_DIR, "fusion_data.csv"), index=False)
```

============================================================
PIPELINE SNAPSHOT 04 END
============================================================

