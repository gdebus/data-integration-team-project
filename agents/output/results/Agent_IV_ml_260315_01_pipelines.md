# Pipeline Snapshots

notebook_name=Agent III & IV
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=65.48%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    MLBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import numpy as np
import os


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

discogs = load_csv(
    "output/schema-matching/discogs.csv",
    name="discogs",
)

lastfm = load_csv(
    "output/schema-matching/lastfm.csv",
    name="lastfm",
)

musicbrainz = load_csv(
    "output/schema-matching/musicbrainz.csv",
    name="musicbrainz",
)

discogs["id"] = discogs["id"]
lastfm["id"] = lastfm["id"]
musicbrainz["id"] = musicbrainz["id"]

datasets = [discogs, lastfm, musicbrainz]


# --------------------------------
# Schema already matched in provided files
# --------------------------------

print("Matching Schema")
print("Schema already aligned in provided schema-matching output files")


# --------------------------------
# Perform Blocking
# MUST use provided blocking configuration
# --------------------------------

print("Performing Blocking")

blocker_1_2 = EmbeddingBlocker(
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

blocker_1_3 = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    window=15,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_2_3 = EmbeddingBlocker(
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
# Matching configuration
# --------------------------------

def lower_strip(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


comparators_1_2 = [
    StringComparator(
        column="name",
        similarity_function="cosine",
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
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=15.0,
    ),
]

comparators_1_3 = [
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
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=10.0,
    ),
    DateComparator(
        column="release-date",
        max_days_difference=365,
    ),
    StringComparator(
        column="release-country",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_2_3 = [
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
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=5.0,
    ),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

# Load ground truth correspondences (ML training/test)
train_1_2 = load_csv(
    "input/datasets/music/testsets/discogs_lastfm_goldstandard_blocking.csv",
    name="ground_truth_discogs_lastfm_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/datasets/music/testsets/discogs_musicbrainz_goldstandard_blocking.csv",
    name="ground_truth_discogs_musicbrainz_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/music/testsets/musicbrainz_lastfm_goldstandard_blocking.csv",
    name="ground_truth_musicbrainz_lastfm_train",
    add_index=False,
)

train_1_2_features = feature_extractor_1_2.create_features(
    discogs,
    lastfm,
    train_1_2[["id1", "id2"]],
    labels=train_1_2["label"],
    id_column="id",
)

train_1_3_features = feature_extractor_1_3.create_features(
    discogs,
    musicbrainz,
    train_1_3[["id1", "id2"]],
    labels=train_1_3["label"],
    id_column="id",
)

train_2_3_features = feature_extractor_2_3.create_features(
    musicbrainz,
    lastfm,
    train_2_3[["id1", "id2"]],
    labels=train_2_3["label"],
    id_column="id",
)

feat_cols_1_2 = [
    col for col in train_1_2_features.columns if col not in ["id1", "id2", "label"]
]
X_train_1_2 = train_1_2_features[feat_cols_1_2]
y_train_1_2 = train_1_2_features["label"]

feat_cols_1_3 = [
    col for col in train_1_3_features.columns if col not in ["id1", "id2", "label"]
]
X_train_1_3 = train_1_3_features[feat_cols_1_3]
y_train_1_3 = train_1_3_features["label"]

feat_cols_2_3 = [
    col for col in train_2_3_features.columns if col not in ["id1", "id2", "label"]
]
X_train_2_3 = train_2_3_features[feat_cols_2_3]
y_train_2_3 = train_2_3_features["label"]

training_datasets = [
    (X_train_1_2, y_train_1_2),
    (X_train_1_3, y_train_1_3),
    (X_train_2_3, y_train_2_3),
]


# --------------------------------
# Select Best Model
# --------------------------------

param_grids = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced", None],
        },
    },
    "LogisticRegression": {
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "class_weight": ["balanced", None],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "learning_rate": [0.1, 0.2],
        },
    },
    "SVM": {
        "model": SVC(random_state=42, probability=True),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "class_weight": ["balanced", None],
        },
    },
}

scorer = make_scorer(f1_score)
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_models = []

for dataset in training_datasets:
    best_overall_score = -1
    best_overall_model = None

    for model_name, config in param_grids.items():
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring=scorer,
            cv=cv_folds,
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(dataset[0], dataset[1])
        if grid_search.best_score_ > best_overall_score:
            best_overall_model = grid_search.best_estimator_
            best_overall_score = grid_search.best_score_

    best_models.append(best_overall_model)

print("Matching Entities")

ml_matcher_1_2 = MLBasedMatcher(feature_extractor_1_2)
ml_matcher_1_3 = MLBasedMatcher(feature_extractor_1_3)
ml_matcher_2_3 = MLBasedMatcher(feature_extractor_2_3)

ml_correspondences_1_2 = ml_matcher_1_2.match(
    discogs,
    lastfm,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    discogs,
    musicbrainz,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    musicbrainz,
    lastfm,
    candidates=blocker_2_3,
    id_column="id",
    trained_classifier=best_models[2],
)

print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_1_2 = clusterer.cluster(ml_correspondences_1_2)
ml_correspondences_1_3 = clusterer.cluster(ml_correspondences_1_3)
ml_correspondences_2_3 = clusterer.cluster(ml_correspondences_2_3)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_1_2.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)
ml_correspondences_1_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)
ml_correspondences_2_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_musicbrainz_lastfm.csv"),
    index=False,
)

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

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

ml_fused_standard_blocker = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=False,
)

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=14
node_name=execute_pipeline
accuracy_score=33.50%
------------------------------------------------------------

```python
# -*- coding: utf-8 -*-
from PyDI.io import load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    MLBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.fusion import DataFusionStrategy, DataFusionEngine
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import numpy as np
import os
import ast
import re
import unicodedata
from collections import Counter


# --------------------------------
# Prepare Data
# --------------------------------

discogs = load_csv(
    "output/schema-matching/discogs.csv",
    name="discogs",
)

lastfm = load_csv(
    "output/schema-matching/lastfm.csv",
    name="lastfm",
)

musicbrainz = load_csv(
    "output/schema-matching/musicbrainz.csv",
    name="musicbrainz",
)

discogs["id"] = discogs["id"]
lastfm["id"] = lastfm["id"]
musicbrainz["id"] = musicbrainz["id"]

datasets = [discogs, lastfm, musicbrainz]

print("Matching Schema")
print("Schema already aligned in provided schema-matching output files")


# --------------------------------
# Normalization helpers
# --------------------------------

def is_null_like(x):
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x).strip().lower()
    return s in {"", "nan", "none", "null", "nat", "[]", "[nan]"}


def safe_str(x):
    if is_null_like(x):
        return ""
    return str(x)


def lower_strip(x):
    if is_null_like(x):
        return ""
    return str(x).lower().strip()


def normalize_text_basic(x):
    s = safe_str(x)
    if not s:
        return ""
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u00b4", "'").replace("`", "'")
    try:
        s = s.encode("latin1").decode("utf-8")
    except Exception:
        pass
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("utf-8", errors="ignore")
    s = s.replace("&", " and ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_for_compare(x):
    s = normalize_text_basic(x).lower().strip()
    s = re.sub(r"^\s*album\s*:\s*", "", s)
    s = re.sub(r"\((album|ep|single)\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" -_/")
    return s


def clean_title_value(x):
    s = normalize_text_basic(x)
    if not s:
        return ""
    s = re.sub(r"^\s*album\s*:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\((album)\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" -_/")
    return s


def parse_numeric(x):
    if is_null_like(x):
        return np.nan
    s = str(x).strip()
    if not s or (s.startswith("[") and s.endswith("]")):
        return np.nan
    s = re.sub(r"[^\d\.\-]", "", s)
    if s in {"", "-", ".", "-."}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def numeric_to_string(x):
    if pd.isna(x):
        return ""
    if abs(float(x) - round(float(x))) < 1e-9:
        return str(int(round(float(x))))
    return str(round(float(x), 3)).rstrip("0").rstrip(".")


def parse_date_value(x):
    if is_null_like(x):
        return pd.NaT
    return pd.to_datetime(x, errors="coerce")


def parse_list_field(x):
    if is_null_like(x):
        return []
    if isinstance(x, list):
        return [str(v) for v in x if not is_null_like(v)]
    s = str(x).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if not is_null_like(v)]
    except Exception:
        pass
    return [s]


def clean_track_name(x):
    s = normalize_text_basic(x)
    s = re.sub(r"^\s*\d+\s*[\.\-\)]\s*", "", s)
    s = re.sub(r"^\s*track\s*\d+\s*[:\-\)]\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" -_/")
    return s


def normalize_track_name_key(x):
    s = clean_track_name(x).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_country_value(x):
    s = normalize_text_basic(x)
    if not s:
        return ""
    mapping = {
        "united kingdom of great britain and northern ireland": "UK",
        "united kingdom": "UK",
        "great britain": "UK",
        "england": "UK",
        "united states of america": "US",
        "united states": "US",
    }
    return mapping.get(s.lower(), s)


def majority_vote_scalar(values, normalizer=normalize_for_compare, cleaner=normalize_text_basic, prefer_shortest=True):
    cleaned = []
    for v in values:
        if is_null_like(v):
            continue
        c = cleaner(v)
        if c:
            cleaned.append(c)
    if not cleaned:
        return ""

    groups = {}
    counts = Counter()
    for v in cleaned:
        key = normalizer(v)
        if not key:
            continue
        counts[key] += 1
        groups.setdefault(key, []).append(v)

    if not counts:
        return cleaned[0]

    best_key = sorted(
        counts.keys(),
        key=lambda k: (-counts[k], len(min(groups[k], key=len)), min(groups[k], key=len).lower()),
    )[0]
    candidates = groups[best_key]
    if prefer_shortest:
        return min(candidates, key=lambda x: (len(x), x.lower()))
    return max(candidates, key=lambda x: (len(x), x.lower()))


def fuse_release_date(values):
    valid_dates = [parse_date_value(v) for v in values if not is_null_like(v)]
    valid_dates = [d for d in valid_dates if not pd.isna(d)]
    if not valid_dates:
        return ""
    date_counts = Counter([d.strftime("%Y-%m-%d") for d in valid_dates])
    best = sorted(date_counts.keys(), key=lambda k: (-date_counts[k], k))[0]
    return best


def fuse_duration(values):
    nums = [parse_numeric(v) for v in values]
    nums = [v for v in nums if not pd.isna(v) and v > 0]
    if not nums:
        return ""
    rounded_counts = Counter([int(round(v)) for v in nums])
    best_rounded = sorted(rounded_counts.keys(), key=lambda k: (-rounded_counts[k], k))[0]
    close_vals = [v for v in nums if abs(v - best_rounded) <= 5]
    if close_vals:
        return numeric_to_string(float(np.median(close_vals)))
    return numeric_to_string(float(np.median(nums)))


def canonicalize_track_positions(lst):
    out = []
    for v in lst:
        s = safe_str(v).strip()
        if not s:
            continue
        m = re.search(r"\d+", s)
        out.append(m.group(0) if m else s)
    return out


def canonicalize_track_names(lst):
    return [clean_track_name(v) for v in lst if clean_track_name(v)]


def canonicalize_track_durations(lst):
    out = []
    for v in lst:
        n = parse_numeric(v)
        if pd.isna(n) or n <= 0:
            out.append("")
        else:
            out.append(numeric_to_string(n))
    return out


def extract_tracklists(values_names, values_positions, values_durations):
    tracklists = []
    max_len = max(len(values_names), len(values_positions), len(values_durations))
    values_names = list(values_names) + [""] * (max_len - len(values_names))
    values_positions = list(values_positions) + [""] * (max_len - len(values_positions))
    values_durations = list(values_durations) + [""] * (max_len - len(values_durations))

    for n_val, p_val, d_val in zip(values_names, values_positions, values_durations):
        names = canonicalize_track_names(parse_list_field(n_val))
        positions = canonicalize_track_positions(parse_list_field(p_val))
        durations = canonicalize_track_durations(parse_list_field(d_val))
        length = max(len(names), len(positions), len(durations))
        if length == 0:
            continue
        names = names + [""] * (length - len(names))
        positions = positions + [""] * (length - len(positions))
        durations = durations + [""] * (length - len(durations))
        triples = [(positions[i], names[i], durations[i]) for i in range(length)]
        tracklists.append(triples)
    return tracklists


def choose_best_tracklist(tracklists):
    if not tracklists:
        return [], [], []

    keyed = Counter()
    variants = {}
    for tl in tracklists:
        key = tuple((p, normalize_track_name_key(n), d) for p, n, d in tl)
        keyed[key] += 1
        variants.setdefault(key, []).append(tl)

    best_key = sorted(
        keyed.keys(),
        key=lambda k: (-keyed[k], len(k), sum(len(t[1]) for t in k)),
    )[0]
    candidate_lists = variants[best_key]
    best_variant = sorted(
        candidate_lists,
        key=lambda tl: (
            len(tl),
            sum(1 for p, n, d in tl if not n),
            sum(1 for p, n, d in tl if not d),
            sum(len(n) for p, n, d in tl),
        ),
    )[0]

    positions = [p for p, n, d in best_variant if p or n or d]
    names = [n for p, n, d in best_variant if p or n or d]
    durations = [d for p, n, d in best_variant if p or n or d]
    return positions, names, durations


def build_cluster_tracklist_fusers():
    state = {"positions": None, "names": None, "durations": None}

    def choose(values, field):
        tracklists = extract_tracklists(
            state["names"] if state["names"] is not None else values,
            state["positions"] if state["positions"] is not None else values,
            state["durations"] if state["durations"] is not None else values,
        )
        pos, nam, dur = choose_best_tracklist(tracklists)
        state["positions"], state["names"], state["durations"] = pos, nam, dur
        if field == "positions":
            return str(pos) if pos else ""
        if field == "names":
            return str(nam) if nam else ""
        return str(dur) if dur else ""

    def fuse_track_positions(values):
        return choose(values, "positions")

    def fuse_track_names(values):
        return choose(values, "names")

    def fuse_track_durations(values):
        return choose(values, "durations")

    return fuse_track_positions, fuse_track_names, fuse_track_durations


# --------------------------------
# Light preprocessing
# --------------------------------

for df in [discogs, lastfm, musicbrainz]:
    for col in ["name", "artist", "release-country", "label", "genre", "release-date"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "" if is_null_like(x) else safe_str(x))

    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(parse_numeric)

    if "release-country" in df.columns:
        df["release-country"] = df["release-country"].apply(clean_country_value)

    if "name" in df.columns:
        df["name"] = df["name"].apply(clean_title_value)

    if "artist" in df.columns:
        df["artist"] = df["artist"].apply(normalize_text_basic)

    if "label" in df.columns:
        df["label"] = df["label"].apply(normalize_text_basic)

    if "tracks_track_name" in df.columns:
        df["tracks_track_name"] = df["tracks_track_name"].apply(
            lambda x: str(canonicalize_track_names(parse_list_field(x))) if not is_null_like(x) else ""
        )

    if "tracks_track_position" in df.columns:
        df["tracks_track_position"] = df["tracks_track_position"].apply(
            lambda x: str(canonicalize_track_positions(parse_list_field(x))) if not is_null_like(x) else ""
        )

    if "tracks_track_duration" in df.columns:
        df["tracks_track_duration"] = df["tracks_track_duration"].apply(
            lambda x: str(canonicalize_track_durations(parse_list_field(x))) if not is_null_like(x) else ""
        )


# --------------------------------
# Perform Blocking
# --------------------------------

print("Performing Blocking")

blocker_1_2 = EmbeddingBlocker(
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

blocker_1_3 = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    window=15,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_2_3 = EmbeddingBlocker(
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
# Matching configuration
# --------------------------------

comparators_1_2 = [
    StringComparator(
        column="name",
        similarity_function="cosine",
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
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=15.0,
    ),
]

comparators_1_3 = [
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
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=10.0,
    ),
    DateComparator(
        column="release-date",
        max_days_difference=365,
    ),
    StringComparator(
        column="release-country",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_2_3 = [
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
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=5.0,
    ),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

train_1_2 = load_csv(
    "input/datasets/music/testsets/discogs_lastfm_goldstandard_blocking.csv",
    name="ground_truth_discogs_lastfm_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/datasets/music/testsets/discogs_musicbrainz_goldstandard_blocking.csv",
    name="ground_truth_discogs_musicbrainz_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/music/testsets/musicbrainz_lastfm_goldstandard_blocking.csv",
    name="ground_truth_musicbrainz_lastfm_train",
    add_index=False,
)

train_1_2_features = feature_extractor_1_2.create_features(
    discogs,
    lastfm,
    train_1_2[["id1", "id2"]],
    labels=train_1_2["label"],
    id_column="id",
)

train_1_3_features = feature_extractor_1_3.create_features(
    discogs,
    musicbrainz,
    train_1_3[["id1", "id2"]],
    labels=train_1_3["label"],
    id_column="id",
)

train_2_3_features = feature_extractor_2_3.create_features(
    musicbrainz,
    lastfm,
    train_2_3[["id1", "id2"]],
    labels=train_2_3["label"],
    id_column="id",
)

feat_cols_1_2 = [
    col for col in train_1_2_features.columns if col not in ["id1", "id2", "label"]
]
X_train_1_2 = train_1_2_features[feat_cols_1_2]
y_train_1_2 = train_1_2_features["label"]

feat_cols_1_3 = [
    col for col in train_1_3_features.columns if col not in ["id1", "id2", "label"]
]
X_train_1_3 = train_1_3_features[feat_cols_1_3]
y_train_1_3 = train_1_3_features["label"]

feat_cols_2_3 = [
    col for col in train_2_3_features.columns if col not in ["id1", "id2", "label"]
]
X_train_2_3 = train_2_3_features[feat_cols_2_3]
y_train_2_3 = train_2_3_features["label"]

training_datasets = [
    (X_train_1_2, y_train_1_2),
    (X_train_1_3, y_train_1_3),
    (X_train_2_3, y_train_2_3),
]


# --------------------------------
# Select Best Model
# --------------------------------

param_grids = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced", None],
        },
    },
    "LogisticRegression": {
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "class_weight": ["balanced", None],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "learning_rate": [0.1, 0.2],
        },
    },
    "SVM": {
        "model": SVC(random_state=42, probability=True),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "class_weight": ["balanced", None],
        },
    },
}

scorer = make_scorer(f1_score)
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_models = []

for dataset in training_datasets:
    best_overall_score = -1
    best_overall_model = None

    for _, config in param_grids.items():
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring=scorer,
            cv=cv_folds,
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(dataset[0], dataset[1])
        if grid_search.best_score_ > best_overall_score:
            best_overall_model = grid_search.best_estimator_
            best_overall_score = grid_search.best_score_

    best_models.append(best_overall_model)

print("Matching Entities")

ml_matcher_1_2 = MLBasedMatcher(feature_extractor_1_2)
ml_matcher_1_3 = MLBasedMatcher(feature_extractor_1_3)
ml_matcher_2_3 = MLBasedMatcher(feature_extractor_2_3)

ml_correspondences_1_2 = ml_matcher_1_2.match(
    discogs,
    lastfm,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    discogs,
    musicbrainz,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    musicbrainz,
    lastfm,
    candidates=blocker_2_3,
    id_column="id",
    trained_classifier=best_models[2],
)

print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_1_2 = clusterer.cluster(ml_correspondences_1_2)
ml_correspondences_1_3 = clusterer.cluster(ml_correspondences_1_3)
ml_correspondences_2_3 = clusterer.cluster(ml_correspondences_2_3)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_1_2.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)
ml_correspondences_1_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)
ml_correspondences_2_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_musicbrainz_lastfm.csv"),
    index=False,
)

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
)

fuse_track_positions, fuse_track_names, fuse_track_durations = build_cluster_tracklist_fusers()

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser(
    "name",
    lambda values: majority_vote_scalar(
        values,
        normalizer=normalize_for_compare,
        cleaner=clean_title_value,
        prefer_shortest=True,
    ),
)
strategy.add_attribute_fuser(
    "artist",
    lambda values: majority_vote_scalar(
        values,
        normalizer=normalize_for_compare,
        cleaner=normalize_text_basic,
        prefer_shortest=True,
    ),
)
strategy.add_attribute_fuser("release-date", fuse_release_date)
strategy.add_attribute_fuser(
    "release-country",
    lambda values: majority_vote_scalar(
        [clean_country_value(v) for v in values],
        normalizer=normalize_for_compare,
        cleaner=clean_country_value,
        prefer_shortest=True,
    ),
)
strategy.add_attribute_fuser("duration", fuse_duration)
strategy.add_attribute_fuser(
    "label",
    lambda values: majority_vote_scalar(
        values,
        normalizer=normalize_for_compare,
        cleaner=normalize_text_basic,
        prefer_shortest=True,
    ),
)
strategy.add_attribute_fuser("tracks_track_position", fuse_track_positions)
strategy.add_attribute_fuser("tracks_track_name", fuse_track_names)
strategy.add_attribute_fuser("tracks_track_duration", fuse_track_durations)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused_standard_blocker = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=False,
)

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=20
node_name=execute_pipeline
accuracy_score=36.55%
------------------------------------------------------------

```python
# -*- coding: utf-8 -*-
from PyDI.io import load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    MLBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.fusion import DataFusionStrategy, DataFusionEngine
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import numpy as np
import os
import ast
import re
import unicodedata
from collections import Counter


# --------------------------------
# Prepare Data
# --------------------------------

discogs = load_csv(
    "output/schema-matching/discogs.csv",
    name="discogs",
)

lastfm = load_csv(
    "output/schema-matching/lastfm.csv",
    name="lastfm",
)

musicbrainz = load_csv(
    "output/schema-matching/musicbrainz.csv",
    name="musicbrainz",
)

discogs["id"] = discogs["id"]
lastfm["id"] = lastfm["id"]
musicbrainz["id"] = musicbrainz["id"]

datasets = [discogs, lastfm, musicbrainz]

print("Matching Schema")
print("Schema already aligned in provided schema-matching output files")


# --------------------------------
# Normalization helpers
# --------------------------------

def is_null_like(x):
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x).strip().lower()
    return s in {"", "nan", "none", "null", "nat", "[]", "[nan]"}


def safe_str(x):
    if is_null_like(x):
        return ""
    return str(x)


def lower_strip(x):
    if is_null_like(x):
        return ""
    return str(x).lower().strip()


def normalize_text_basic(x):
    s = safe_str(x)
    if not s:
        return ""
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u00b4", "'").replace("`", "'")
    try:
        s = s.encode("latin1").decode("utf-8")
    except Exception:
        pass
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("utf-8", errors="ignore")
    s = s.replace("&", " and ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_for_compare(x):
    s = normalize_text_basic(x).lower().strip()
    s = re.sub(r"^\s*album\s*:\s*", "", s)
    s = re.sub(r"\((album|ep|single)\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" -_/")
    return s


def title_variant_penalty(s):
    t = normalize_for_compare(s)
    penalty = 0
    if " / " in safe_str(s):
        penalty += 2
    if re.search(r"\b(ep|single|version|mix|remix|edit|live|demo|instrumental)\b", t, flags=re.IGNORECASE):
        penalty += 1
    return penalty


def clean_title_value(x):
    s = normalize_text_basic(x)
    if not s:
        return ""
    s = re.sub(r"^\s*album\s*:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\((album)\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" -_/")
    return s


def parse_numeric(x):
    if is_null_like(x):
        return np.nan
    s = str(x).strip()
    if not s or (s.startswith("[") and s.endswith("]")):
        return np.nan
    s = re.sub(r"[^\d\.\-]", "", s)
    if s in {"", "-", ".", "-."}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def numeric_to_string(x):
    if pd.isna(x):
        return ""
    if abs(float(x) - round(float(x))) < 1e-9:
        return str(int(round(float(x))))
    return str(round(float(x), 3)).rstrip("0").rstrip(".")


def parse_date_value(x):
    if is_null_like(x):
        return pd.NaT
    return pd.to_datetime(x, errors="coerce")


def parse_list_field(x):
    if is_null_like(x):
        return []
    if isinstance(x, list):
        return [str(v) for v in x if not is_null_like(v)]
    s = str(x).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if not is_null_like(v)]
    except Exception:
        pass
    return [s]


def clean_track_name(x):
    s = normalize_text_basic(x)
    s = re.sub(r"^\s*\d+\s*[\.\-\)]\s*", "", s)
    s = re.sub(r"^\s*track\s*\d+\s*[:\-\)]\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" -_/")
    return s


def normalize_track_name_key(x):
    s = clean_track_name(x).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


COUNTRY_TO_CANONICAL = {
    "uk": "United Kingdom of Great Britain and Northern Ireland",
    "united kingdom": "United Kingdom of Great Britain and Northern Ireland",
    "great britain": "United Kingdom of Great Britain and Northern Ireland",
    "england": "United Kingdom of Great Britain and Northern Ireland",
    "united kingdom of great britain and northern ireland": "United Kingdom of Great Britain and Northern Ireland",
    "us": "United States of America",
    "u.s.": "United States of America",
    "usa": "United States of America",
    "united states": "United States of America",
    "united states of america": "United States of America",
}


def clean_country_value(x):
    s = normalize_text_basic(x)
    if not s:
        return ""
    return COUNTRY_TO_CANONICAL.get(s.lower(), s)


def completeness_score(v):
    s = safe_str(v).strip()
    if not s:
        return 0
    score = len(s)
    score += len(re.findall(r"[A-Za-z0-9]+", s))
    if " / " in s:
        score -= 3
    return score


def majority_vote_scalar(values, normalizer=normalize_for_compare, cleaner=normalize_text_basic, prefer_longest=True, penalize_title_variants=False):
    cleaned = []
    for v in values:
        if is_null_like(v):
            continue
        c = cleaner(v)
        if c:
            cleaned.append(c)
    if not cleaned:
        return ""

    groups = {}
    counts = Counter()
    for v in cleaned:
        key = normalizer(v)
        if not key:
            continue
        counts[key] += 1
        groups.setdefault(key, []).append(v)

    if not counts:
        return cleaned[0]

    def group_score(k):
        vals = groups[k]
        best_completeness = max(completeness_score(v) for v in vals)
        min_penalty = min(title_variant_penalty(v) for v in vals) if penalize_title_variants else 0
        return (-counts[k], min_penalty, -best_completeness, -len(k), k)

    best_key = sorted(counts.keys(), key=group_score)[0]
    candidates = groups[best_key]

    if prefer_longest:
        return sorted(
            candidates,
            key=lambda x: (
                title_variant_penalty(x) if penalize_title_variants else 0,
                -completeness_score(x),
                -len(x),
                x.lower(),
            ),
        )[0]

    return sorted(
        candidates,
        key=lambda x: (
            title_variant_penalty(x) if penalize_title_variants else 0,
            len(x),
            x.lower(),
        ),
    )[0]


def fuse_release_date(values):
    valid_dates = [parse_date_value(v) for v in values if not is_null_like(v)]
    valid_dates = [d for d in valid_dates if not pd.isna(d)]
    if not valid_dates:
        return ""
    date_counts = Counter([d.strftime("%Y-%m-%d") for d in valid_dates])
    best = sorted(date_counts.keys(), key=lambda k: (-date_counts[k], k))[0]
    return best


def canonicalize_track_positions(lst):
    out = []
    for idx, v in enumerate(lst, start=1):
        s = safe_str(v).strip()
        if not s:
            out.append(str(idx))
            continue
        m = re.search(r"\d+", s)
        out.append(m.group(0) if m else str(idx))
    return out


def canonicalize_track_names(lst):
    return [clean_track_name(v) for v in lst if clean_track_name(v)]


def canonicalize_track_durations(lst):
    out = []
    for v in lst:
        n = parse_numeric(v)
        if pd.isna(n) or n <= 0:
            out.append("")
        else:
            out.append(numeric_to_string(n))
    return out


def extract_tracklists(values_names, values_positions, values_durations):
    tracklists = []
    max_len = max(len(values_names), len(values_positions), len(values_durations))
    values_names = list(values_names) + [""] * (max_len - len(values_names))
    values_positions = list(values_positions) + [""] * (max_len - len(values_positions))
    values_durations = list(values_durations) + [""] * (max_len - len(values_durations))

    for n_val, p_val, d_val in zip(values_names, values_positions, values_durations):
        names = canonicalize_track_names(parse_list_field(n_val))
        positions = canonicalize_track_positions(parse_list_field(p_val)) if not is_null_like(p_val) else []
        durations = canonicalize_track_durations(parse_list_field(d_val))

        length = max(len(names), len(positions), len(durations))
        if length == 0:
            continue

        if len(names) < length:
            names = names + [""] * (length - len(names))
        if len(positions) < length:
            start = len(positions) + 1
            positions = positions + [str(i) for i in range(start, length + 1)]
        if len(durations) < length:
            durations = durations + [""] * (length - len(durations))

        triples = []
        for i in range(length):
            triples.append((positions[i], names[i], durations[i]))
        tracklists.append(triples)

    return tracklists


def _tracklist_similarity(tl_a, tl_b):
    a_names = {normalize_track_name_key(n) for _, n, _ in tl_a if normalize_track_name_key(n)}
    b_names = {normalize_track_name_key(n) for _, n, _ in tl_b if normalize_track_name_key(n)}
    if not a_names and not b_names:
        name_score = 0.0
    else:
        name_score = len(a_names & b_names) / max(1, len(a_names | b_names))
    len_score = 1.0 / (1.0 + abs(len(tl_a) - len(tl_b)))
    return 0.7 * name_score + 0.3 * len_score


def choose_best_tracklist(tracklists):
    if not tracklists:
        return [], [], []

    support_scores = []
    for i, tl in enumerate(tracklists):
        support = 0.0
        for j, other in enumerate(tracklists):
            if i == j:
                continue
            support += _tracklist_similarity(tl, other)
        completeness = (
            sum(1 for p, n, d in tl if n) * 2
            + sum(1 for p, n, d in tl if d)
            + sum(1 for p, n, d in tl if p)
        )
        support_scores.append((support, completeness, len(tl), i))

    best_idx = sorted(
        support_scores,
        key=lambda x: (-x[0], -x[1], -x[2], x[3]),
    )[0][3]
    best_variant = tracklists[best_idx]

    positions = [p for p, n, d in best_variant if p or n or d]
    names = [n for p, n, d in best_variant if p or n or d]
    durations = [d for p, n, d in best_variant if p or n or d]
    return positions, names, durations


def build_tracklist_cluster_map(cluster_df):
    names_values = cluster_df["tracks_track_name"].tolist() if "tracks_track_name" in cluster_df.columns else []
    positions_values = cluster_df["tracks_track_position"].tolist() if "tracks_track_position" in cluster_df.columns else []
    durations_values = cluster_df["tracks_track_duration"].tolist() if "tracks_track_duration" in cluster_df.columns else []

    tracklists = extract_tracklists(names_values, positions_values, durations_values)
    pos, nam, dur = choose_best_tracklist(tracklists)

    if not nam and pos:
        nam = [""] * len(pos)
    if not pos and nam:
        pos = [str(i) for i in range(1, len(nam) + 1)]
    if len(dur) < len(nam):
        dur = dur + [""] * (len(nam) - len(dur))

    track_duration_nums = [parse_numeric(v) for v in dur if not is_null_like(v)]
    track_duration_nums = [v for v in track_duration_nums if not pd.isna(v) and v > 0]
    fused_total_duration = numeric_to_string(sum(track_duration_nums)) if track_duration_nums else ""

    return {
        "tracks_track_position": str(pos) if pos else "",
        "tracks_track_name": str(nam) if nam else "",
        "tracks_track_duration": str(dur) if dur else "",
        "duration": fused_total_duration,
    }


def fuse_duration(values, cluster_track_duration=""):
    if cluster_track_duration:
        return cluster_track_duration
    nums = [parse_numeric(v) for v in values]
    nums = [v for v in nums if not pd.isna(v) and v > 0]
    if not nums:
        return ""
    rounded_counts = Counter([int(round(v)) for v in nums])
    best_rounded = sorted(rounded_counts.keys(), key=lambda k: (-rounded_counts[k], -k))[0]
    close_vals = [v for v in nums if abs(v - best_rounded) <= 10]
    if close_vals:
        return numeric_to_string(float(np.median(close_vals)))
    return numeric_to_string(float(np.median(nums)))


# --------------------------------
# Light preprocessing
# --------------------------------

for df in [discogs, lastfm, musicbrainz]:
    for col in ["name", "artist", "release-country", "label", "genre", "release-date"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "" if is_null_like(x) else safe_str(x))

    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(parse_numeric)

    if "release-country" in df.columns:
        df["release-country"] = df["release-country"].apply(clean_country_value)

    if "name" in df.columns:
        df["name"] = df["name"].apply(clean_title_value)

    if "artist" in df.columns:
        df["artist"] = df["artist"].apply(normalize_text_basic)

    if "label" in df.columns:
        df["label"] = df["label"].apply(normalize_text_basic)

    if "tracks_track_name" in df.columns:
        df["tracks_track_name"] = df["tracks_track_name"].apply(
            lambda x: str(canonicalize_track_names(parse_list_field(x))) if not is_null_like(x) else ""
        )

    if "tracks_track_position" in df.columns:
        df["tracks_track_position"] = df["tracks_track_position"].apply(
            lambda x: str(canonicalize_track_positions(parse_list_field(x))) if not is_null_like(x) else ""
        )

    if "tracks_track_duration" in df.columns:
        df["tracks_track_duration"] = df["tracks_track_duration"].apply(
            lambda x: str(canonicalize_track_durations(parse_list_field(x))) if not is_null_like(x) else ""
        )


# --------------------------------
# Perform Blocking
# --------------------------------

print("Performing Blocking")

blocker_1_2 = EmbeddingBlocker(
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

blocker_1_3 = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    window=15,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

blocker_2_3 = EmbeddingBlocker(
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
# Matching configuration
# --------------------------------

comparators_1_2 = [
    StringComparator(
        column="name",
        similarity_function="cosine",
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
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=15.0,
    ),
]

comparators_1_3 = [
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
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=10.0,
    ),
    DateComparator(
        column="release-date",
        max_days_difference=365,
    ),
    StringComparator(
        column="release-country",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_2_3 = [
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
        column="tracks_track_name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="duration",
        max_difference=5.0,
    ),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

train_1_2 = load_csv(
    "input/datasets/music/testsets/discogs_lastfm_goldstandard_blocking.csv",
    name="ground_truth_discogs_lastfm_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/datasets/music/testsets/discogs_musicbrainz_goldstandard_blocking.csv",
    name="ground_truth_discogs_musicbrainz_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/music/testsets/musicbrainz_lastfm_goldstandard_blocking.csv",
    name="ground_truth_musicbrainz_lastfm_train",
    add_index=False,
)

train_1_2_features = feature_extractor_1_2.create_features(
    discogs,
    lastfm,
    train_1_2[["id1", "id2"]],
    labels=train_1_2["label"],
    id_column="id",
)

train_1_3_features = feature_extractor_1_3.create_features(
    discogs,
    musicbrainz,
    train_1_3[["id1", "id2"]],
    labels=train_1_3["label"],
    id_column="id",
)

train_2_3_features = feature_extractor_2_3.create_features(
    musicbrainz,
    lastfm,
    train_2_3[["id1", "id2"]],
    labels=train_2_3["label"],
    id_column="id",
)

feat_cols_1_2 = [
    col for col in train_1_2_features.columns if col not in ["id1", "id2", "label"]
]
X_train_1_2 = train_1_2_features[feat_cols_1_2]
y_train_1_2 = train_1_2_features["label"]

feat_cols_1_3 = [
    col for col in train_1_3_features.columns if col not in ["id1", "id2", "label"]
]
X_train_1_3 = train_1_3_features[feat_cols_1_3]
y_train_1_3 = train_1_3_features["label"]

feat_cols_2_3 = [
    col for col in train_2_3_features.columns if col not in ["id1", "id2", "label"]
]
X_train_2_3 = train_2_3_features[feat_cols_2_3]
y_train_2_3 = train_2_3_features["label"]

training_datasets = [
    (X_train_1_2, y_train_1_2),
    (X_train_1_3, y_train_1_3),
    (X_train_2_3, y_train_2_3),
]


# --------------------------------
# Select Best Model
# --------------------------------

param_grids = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced", None],
        },
    },
    "LogisticRegression": {
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "class_weight": ["balanced", None],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "learning_rate": [0.1, 0.2],
        },
    },
    "SVM": {
        "model": SVC(random_state=42, probability=True),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "class_weight": ["balanced", None],
        },
    },
}

scorer = make_scorer(f1_score)
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_models = []

for dataset in training_datasets:
    best_overall_score = -1
    best_overall_model = None

    for _, config in param_grids.items():
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring=scorer,
            cv=cv_folds,
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(dataset[0], dataset[1])
        if grid_search.best_score_ > best_overall_score:
            best_overall_model = grid_search.best_estimator_
            best_overall_score = grid_search.best_score_

    best_models.append(best_overall_model)

print("Matching Entities")

ml_matcher_1_2 = MLBasedMatcher(feature_extractor_1_2)
ml_matcher_1_3 = MLBasedMatcher(feature_extractor_1_3)
ml_matcher_2_3 = MLBasedMatcher(feature_extractor_2_3)

ml_correspondences_1_2 = ml_matcher_1_2.match(
    discogs,
    lastfm,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    discogs,
    musicbrainz,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    musicbrainz,
    lastfm,
    candidates=blocker_2_3,
    id_column="id",
    trained_classifier=best_models[2],
)

print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_1_2 = clusterer.cluster(ml_correspondences_1_2)
ml_correspondences_1_3 = clusterer.cluster(ml_correspondences_1_3)
ml_correspondences_2_3 = clusterer.cluster(ml_correspondences_2_3)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_1_2.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)
ml_correspondences_1_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)
ml_correspondences_2_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_musicbrainz_lastfm.csv"),
    index=False,
)

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser(
    "name",
    lambda values: majority_vote_scalar(
        values,
        normalizer=normalize_for_compare,
        cleaner=clean_title_value,
        prefer_longest=True,
        penalize_title_variants=True,
    ),
)
strategy.add_attribute_fuser(
    "artist",
    lambda values: majority_vote_scalar(
        values,
        normalizer=normalize_for_compare,
        cleaner=normalize_text_basic,
        prefer_longest=True,
        penalize_title_variants=False,
    ),
)
strategy.add_attribute_fuser("release-date", fuse_release_date)
strategy.add_attribute_fuser(
    "release-country",
    lambda values: majority_vote_scalar(
        [clean_country_value(v) for v in values],
        normalizer=normalize_for_compare,
        cleaner=clean_country_value,
        prefer_longest=True,
        penalize_title_variants=False,
    ),
)
strategy.add_attribute_fuser(
    "label",
    lambda values: majority_vote_scalar(
        values,
        normalizer=normalize_for_compare,
        cleaner=normalize_text_basic,
        prefer_longest=True,
        penalize_title_variants=False,
    ),
)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused_standard_blocker = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=False,
)

source_union = pd.concat(
    [
        discogs.assign(__source_dataset="discogs"),
        lastfm.assign(__source_dataset="lastfm"),
        musicbrainz.assign(__source_dataset="musicbrainz"),
    ],
    ignore_index=True,
)

cluster_rows = []

for _, fused_row in ml_fused_standard_blocker.iterrows():
    cluster_ids = set()
    for col in ["id", "ids", "member_ids", "cluster_ids"]:
        if col in fused_row.index and not is_null_like(fused_row[col]):
            value = fused_row[col]
            if isinstance(value, list):
                cluster_ids.update([str(v) for v in value if not is_null_like(v)])
            else:
                parsed = parse_list_field(value)
                if parsed:
                    cluster_ids.update([str(v) for v in parsed if not is_null_like(v)])
                else:
                    cluster_ids.add(str(value))

    if not cluster_ids and "id" in fused_row.index and not is_null_like(fused_row["id"]):
        cluster_ids.add(str(fused_row["id"]))

    cluster_df = source_union[source_union["id"].astype(str).isin(cluster_ids)].copy()

    if cluster_df.empty:
        cluster_rows.append(fused_row.to_dict())
        continue

    track_map = build_tracklist_cluster_map(cluster_df)

    row = fused_row.to_dict()
    row["tracks_track_position"] = track_map["tracks_track_position"]
    row["tracks_track_name"] = track_map["tracks_track_name"]
    row["tracks_track_duration"] = track_map["tracks_track_duration"]
    row["duration"] = fuse_duration(cluster_df["duration"].tolist(), cluster_track_duration=track_map["duration"])

    if "name" in cluster_df.columns:
        row["name"] = majority_vote_scalar(
            cluster_df["name"].tolist(),
            normalizer=normalize_for_compare,
            cleaner=clean_title_value,
            prefer_longest=True,
            penalize_title_variants=True,
        )
    if "artist" in cluster_df.columns:
        row["artist"] = majority_vote_scalar(
            cluster_df["artist"].tolist(),
            normalizer=normalize_for_compare,
            cleaner=normalize_text_basic,
            prefer_longest=True,
            penalize_title_variants=False,
        )
    if "label" in cluster_df.columns:
        row["label"] = majority_vote_scalar(
            cluster_df["label"].tolist(),
            normalizer=normalize_for_compare,
            cleaner=normalize_text_basic,
            prefer_longest=True,
            penalize_title_variants=False,
        )
    if "release-date" in cluster_df.columns:
        row["release-date"] = fuse_release_date(cluster_df["release-date"].tolist())
    if "release-country" in cluster_df.columns:
        row["release-country"] = majority_vote_scalar(
            [clean_country_value(v) for v in cluster_df["release-country"].tolist()],
            normalizer=normalize_for_compare,
            cleaner=clean_country_value,
            prefer_longest=True,
            penalize_title_variants=False,
        )

    cluster_rows.append(row)

ml_fused_standard_blocker = pd.DataFrame(cluster_rows)

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

