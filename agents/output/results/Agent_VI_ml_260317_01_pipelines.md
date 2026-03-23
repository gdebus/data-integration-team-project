# Pipeline Snapshots

notebook_name=Agent V & VI
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=44.67%
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
import shutil


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

discogs = load_csv(
    "output/schema-matching/discogs.csv",
    name="discogs",
    add_index=False,
)

lastfm = load_csv(
    "output/schema-matching/lastfm.csv",
    name="lastfm",
    add_index=False,
)

musicbrainz = load_csv(
    "output/schema-matching/musicbrainz.csv",
    name="musicbrainz",
    add_index=False,
)

# create id columns
discogs["id"] = discogs["id"]
lastfm["id"] = lastfm["id"]
musicbrainz["id"] = musicbrainz["id"]

datasets = [discogs, lastfm, musicbrainz]


# --------------------------------
# Helper preprocessing
# --------------------------------

def lower_strip(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


# normalize numeric/date columns used in matching
for df in [discogs, lastfm, musicbrainz]:
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

for df in [discogs, musicbrainz]:
    if "release-date" in df.columns:
        df["release-date"] = pd.to_datetime(df["release-date"], errors="coerce")


# --------------------------------
# Perform Blocking
# Use precomputed blocker types and parameter settings
# --------------------------------

print("Performing Blocking")

# discogs_lastfm -> semantic_similarity on ["name", "artist"], top_k=10
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

# discogs_musicbrainz -> sorted_neighbourhood on ["name"], window=15
blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    window=15,
    id_column="id",
    output_dir="output/blocking",
)

# musicbrainz_lastfm -> semantic_similarity on ["name", "artist", "duration"], top_k=10
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
# Matching configuration
# Use the supplied comparator settings
# --------------------------------

comparators_discogs_lastfm = [
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

feature_extractor_discogs_lastfm = FeatureExtractor(comparators_discogs_lastfm)
feature_extractor_discogs_musicbrainz = FeatureExtractor(comparators_discogs_musicbrainz)
feature_extractor_musicbrainz_lastfm = FeatureExtractor(comparators_musicbrainz_lastfm)


# Load ground truth correspondences (ML training/test)
train_discogs_lastfm = load_csv(
    "input/datasets/music/testsets/discogs_lastfm_goldstandard_blocking.csv",
    name="ground_truth_discogs_lastfm_train",
    add_index=False,
)

train_discogs_musicbrainz = load_csv(
    "input/datasets/music/testsets/discogs_musicbrainz_goldstandard_blocking.csv",
    name="ground_truth_discogs_musicbrainz_train",
    add_index=False,
)

train_musicbrainz_lastfm = load_csv(
    "input/datasets/music/testsets/musicbrainz_lastfm_goldstandard_blocking.csv",
    name="ground_truth_musicbrainz_lastfm_train",
    add_index=False,
)

# Extract features
train_discogs_lastfm_features = feature_extractor_discogs_lastfm.create_features(
    discogs,
    lastfm,
    train_discogs_lastfm[["id1", "id2"]],
    labels=train_discogs_lastfm["label"],
    id_column="id",
)

train_discogs_musicbrainz_features = feature_extractor_discogs_musicbrainz.create_features(
    discogs,
    musicbrainz,
    train_discogs_musicbrainz[["id1", "id2"]],
    labels=train_discogs_musicbrainz["label"],
    id_column="id",
)

train_musicbrainz_lastfm_features = feature_extractor_musicbrainz_lastfm.create_features(
    musicbrainz,
    lastfm,
    train_musicbrainz_lastfm[["id1", "id2"]],
    labels=train_musicbrainz_lastfm["label"],
    id_column="id",
)

# Prepare data for ML training
feat_cols_discogs_lastfm = [
    col for col in train_discogs_lastfm_features.columns if col not in ["id1", "id2", "label"]
]
X_train_discogs_lastfm = train_discogs_lastfm_features[feat_cols_discogs_lastfm]
y_train_discogs_lastfm = train_discogs_lastfm_features["label"]

feat_cols_discogs_musicbrainz = [
    col for col in train_discogs_musicbrainz_features.columns if col not in ["id1", "id2", "label"]
]
X_train_discogs_musicbrainz = train_discogs_musicbrainz_features[feat_cols_discogs_musicbrainz]
y_train_discogs_musicbrainz = train_discogs_musicbrainz_features["label"]

feat_cols_musicbrainz_lastfm = [
    col for col in train_musicbrainz_lastfm_features.columns if col not in ["id1", "id2", "label"]
]
X_train_musicbrainz_lastfm = train_musicbrainz_lastfm_features[feat_cols_musicbrainz_lastfm]
y_train_musicbrainz_lastfm = train_musicbrainz_lastfm_features["label"]

training_datasets = [
    (X_train_discogs_lastfm, y_train_discogs_lastfm),
    (X_train_discogs_musicbrainz, y_train_discogs_musicbrainz),
    (X_train_musicbrainz_lastfm, y_train_musicbrainz_lastfm),
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

ml_matcher_discogs_lastfm = MLBasedMatcher(feature_extractor_discogs_lastfm)
ml_matcher_discogs_musicbrainz = MLBasedMatcher(feature_extractor_discogs_musicbrainz)
ml_matcher_musicbrainz_lastfm = MLBasedMatcher(feature_extractor_musicbrainz_lastfm)

ml_correspondences_discogs_lastfm = ml_matcher_discogs_lastfm.match(
    discogs,
    lastfm,
    candidates=blocker_discogs_lastfm,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_discogs_musicbrainz = ml_matcher_discogs_musicbrainz.match(
    discogs,
    musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_musicbrainz_lastfm = ml_matcher_musicbrainz_lastfm.match(
    musicbrainz,
    lastfm,
    candidates=blocker_musicbrainz_lastfm,
    id_column="id",
    trained_classifier=best_models[2],
)

print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_discogs_lastfm = clusterer.cluster(ml_correspondences_discogs_lastfm)
ml_correspondences_discogs_musicbrainz = clusterer.cluster(ml_correspondences_discogs_musicbrainz)
ml_correspondences_musicbrainz_lastfm = clusterer.cluster(ml_correspondences_musicbrainz_lastfm)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_discogs_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)
ml_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)
ml_correspondences_musicbrainz_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_musicbrainz_lastfm.csv"),
    index=False,
)

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_discogs_lastfm,
        ml_correspondences_discogs_musicbrainz,
        ml_correspondences_musicbrainz_lastfm,
    ],
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
node_index=13
node_name=execute_pipeline
accuracy_score=48.22%
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
import os
import shutil


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

discogs = load_csv(
    "output/schema-matching/discogs.csv",
    name="discogs",
    add_index=False,
)

lastfm = load_csv(
    "output/schema-matching/lastfm.csv",
    name="lastfm",
    add_index=False,
)

musicbrainz = load_csv(
    "output/schema-matching/musicbrainz.csv",
    name="musicbrainz",
    add_index=False,
)

# create id columns
discogs["id"] = discogs["id"]
lastfm["id"] = lastfm["id"]
musicbrainz["id"] = musicbrainz["id"]

datasets = [discogs, lastfm, musicbrainz]


# --------------------------------
# Helper preprocessing
# --------------------------------

def lower_strip(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


# normalize columns used by comparators / blockers
for df in [discogs, lastfm, musicbrainz]:
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

for df in [discogs, musicbrainz]:
    if "release-date" in df.columns:
        df["release-date"] = pd.to_datetime(df["release-date"], errors="coerce")

# create text versions for embedding blocker columns that may be numeric
musicbrainz["duration_text"] = musicbrainz["duration"].fillna("").astype(str)
lastfm["duration_text"] = lastfm["duration"].fillna("").astype(str)


# --------------------------------
# Perform Blocking
# Use precomputed blocker types and parameter settings
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
    output_dir="output/blocking",
)

blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration_text"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration
# Use the supplied comparator settings
# --------------------------------

comparators_discogs_lastfm = [
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

feature_extractor_discogs_lastfm = FeatureExtractor(comparators_discogs_lastfm)
feature_extractor_discogs_musicbrainz = FeatureExtractor(comparators_discogs_musicbrainz)
feature_extractor_musicbrainz_lastfm = FeatureExtractor(comparators_musicbrainz_lastfm)


# --------------------------------
# Load ground truth correspondences (ML training/test)
# --------------------------------

train_discogs_lastfm = load_csv(
    "input/datasets/music/testsets/discogs_lastfm_goldstandard_blocking.csv",
    name="ground_truth_discogs_lastfm_train",
    add_index=False,
)

train_discogs_musicbrainz = load_csv(
    "input/datasets/music/testsets/discogs_musicbrainz_goldstandard_blocking.csv",
    name="ground_truth_discogs_musicbrainz_train",
    add_index=False,
)

train_musicbrainz_lastfm = load_csv(
    "input/datasets/music/testsets/musicbrainz_lastfm_goldstandard_blocking.csv",
    name="ground_truth_musicbrainz_lastfm_train",
    add_index=False,
)

# Extract features
train_discogs_lastfm_features = feature_extractor_discogs_lastfm.create_features(
    discogs,
    lastfm,
    train_discogs_lastfm[["id1", "id2"]],
    labels=train_discogs_lastfm["label"],
    id_column="id",
)

train_discogs_musicbrainz_features = feature_extractor_discogs_musicbrainz.create_features(
    discogs,
    musicbrainz,
    train_discogs_musicbrainz[["id1", "id2"]],
    labels=train_discogs_musicbrainz["label"],
    id_column="id",
)

train_musicbrainz_lastfm_features = feature_extractor_musicbrainz_lastfm.create_features(
    musicbrainz,
    lastfm,
    train_musicbrainz_lastfm[["id1", "id2"]],
    labels=train_musicbrainz_lastfm["label"],
    id_column="id",
)

# Prepare data for ML training
feat_cols_discogs_lastfm = [
    col for col in train_discogs_lastfm_features.columns if col not in ["id1", "id2", "label"]
]
X_train_discogs_lastfm = train_discogs_lastfm_features[feat_cols_discogs_lastfm]
y_train_discogs_lastfm = train_discogs_lastfm_features["label"]

feat_cols_discogs_musicbrainz = [
    col for col in train_discogs_musicbrainz_features.columns if col not in ["id1", "id2", "label"]
]
X_train_discogs_musicbrainz = train_discogs_musicbrainz_features[feat_cols_discogs_musicbrainz]
y_train_discogs_musicbrainz = train_discogs_musicbrainz_features["label"]

feat_cols_musicbrainz_lastfm = [
    col for col in train_musicbrainz_lastfm_features.columns if col not in ["id1", "id2", "label"]
]
X_train_musicbrainz_lastfm = train_musicbrainz_lastfm_features[feat_cols_musicbrainz_lastfm]
y_train_musicbrainz_lastfm = train_musicbrainz_lastfm_features["label"]

training_datasets = [
    (X_train_discogs_lastfm, y_train_discogs_lastfm),
    (X_train_discogs_musicbrainz, y_train_discogs_musicbrainz),
    (X_train_musicbrainz_lastfm, y_train_musicbrainz_lastfm),
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

ml_matcher_discogs_lastfm = MLBasedMatcher(feature_extractor_discogs_lastfm)
ml_matcher_discogs_musicbrainz = MLBasedMatcher(feature_extractor_discogs_musicbrainz)
ml_matcher_musicbrainz_lastfm = MLBasedMatcher(feature_extractor_musicbrainz_lastfm)

ml_correspondences_discogs_lastfm = ml_matcher_discogs_lastfm.match(
    discogs,
    lastfm,
    candidates=blocker_discogs_lastfm,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_discogs_musicbrainz = ml_matcher_discogs_musicbrainz.match(
    discogs,
    musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_musicbrainz_lastfm = ml_matcher_musicbrainz_lastfm.match(
    musicbrainz,
    lastfm,
    candidates=blocker_musicbrainz_lastfm,
    id_column="id",
    trained_classifier=best_models[2],
)

print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_discogs_lastfm = clusterer.cluster(ml_correspondences_discogs_lastfm)
ml_correspondences_discogs_musicbrainz = clusterer.cluster(ml_correspondences_discogs_musicbrainz)
ml_correspondences_musicbrainz_lastfm = clusterer.cluster(ml_correspondences_musicbrainz_lastfm)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_discogs_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)
ml_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)
ml_correspondences_musicbrainz_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_musicbrainz_lastfm.csv"),
    index=False,
)

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_discogs_lastfm,
        ml_correspondences_discogs_musicbrainz,
        ml_correspondences_musicbrainz_lastfm,
    ],
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
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=20
node_name=execute_pipeline
accuracy_score=50.25%
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
import os
import shutil


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

discogs = load_csv(
    "output/schema-matching/discogs.csv",
    name="discogs",
    add_index=False,
)

lastfm = load_csv(
    "output/schema-matching/lastfm.csv",
    name="lastfm",
    add_index=False,
)

musicbrainz = load_csv(
    "output/schema-matching/musicbrainz.csv",
    name="musicbrainz",
    add_index=False,
)

# create id columns
discogs["id"] = discogs["id"]
lastfm["id"] = lastfm["id"]
musicbrainz["id"] = musicbrainz["id"]

datasets = [discogs, lastfm, musicbrainz]


# --------------------------------
# Helper preprocessing
# --------------------------------

def lower_strip(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


for df in [discogs, lastfm, musicbrainz]:
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

for df in [discogs, musicbrainz]:
    if "release-date" in df.columns:
        df["release-date"] = pd.to_datetime(df["release-date"], errors="coerce")

musicbrainz["duration_text"] = musicbrainz["duration"].fillna("").astype(str)
lastfm["duration_text"] = lastfm["duration"].fillna("").astype(str)


# --------------------------------
# Perform Blocking
# Use precomputed blocker types and parameter settings
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
    output_dir="output/blocking",
)

blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration_text"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration
# Use the supplied comparator settings
# --------------------------------

comparators_discogs_lastfm = [
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

feature_extractor_discogs_lastfm = FeatureExtractor(comparators_discogs_lastfm)
feature_extractor_discogs_musicbrainz = FeatureExtractor(comparators_discogs_musicbrainz)
feature_extractor_musicbrainz_lastfm = FeatureExtractor(comparators_musicbrainz_lastfm)


# --------------------------------
# Load ground truth correspondences (ML training/test)
# --------------------------------

train_discogs_lastfm = load_csv(
    "input/datasets/music/testsets/discogs_lastfm_goldstandard_blocking.csv",
    name="ground_truth_discogs_lastfm_train",
    add_index=False,
)

train_discogs_musicbrainz = load_csv(
    "input/datasets/music/testsets/discogs_musicbrainz_goldstandard_blocking.csv",
    name="ground_truth_discogs_musicbrainz_train",
    add_index=False,
)

train_musicbrainz_lastfm = load_csv(
    "input/datasets/music/testsets/musicbrainz_lastfm_goldstandard_blocking.csv",
    name="ground_truth_musicbrainz_lastfm_train",
    add_index=False,
)

train_discogs_lastfm_features = feature_extractor_discogs_lastfm.create_features(
    discogs,
    lastfm,
    train_discogs_lastfm[["id1", "id2"]],
    labels=train_discogs_lastfm["label"],
    id_column="id",
)

train_discogs_musicbrainz_features = feature_extractor_discogs_musicbrainz.create_features(
    discogs,
    musicbrainz,
    train_discogs_musicbrainz[["id1", "id2"]],
    labels=train_discogs_musicbrainz["label"],
    id_column="id",
)

train_musicbrainz_lastfm_features = feature_extractor_musicbrainz_lastfm.create_features(
    musicbrainz,
    lastfm,
    train_musicbrainz_lastfm[["id1", "id2"]],
    labels=train_musicbrainz_lastfm["label"],
    id_column="id",
)

feat_cols_discogs_lastfm = [
    col for col in train_discogs_lastfm_features.columns if col not in ["id1", "id2", "label"]
]
X_train_discogs_lastfm = train_discogs_lastfm_features[feat_cols_discogs_lastfm]
y_train_discogs_lastfm = train_discogs_lastfm_features["label"]

feat_cols_discogs_musicbrainz = [
    col for col in train_discogs_musicbrainz_features.columns if col not in ["id1", "id2", "label"]
]
X_train_discogs_musicbrainz = train_discogs_musicbrainz_features[feat_cols_discogs_musicbrainz]
y_train_discogs_musicbrainz = train_discogs_musicbrainz_features["label"]

feat_cols_musicbrainz_lastfm = [
    col for col in train_musicbrainz_lastfm_features.columns if col not in ["id1", "id2", "label"]
]
X_train_musicbrainz_lastfm = train_musicbrainz_lastfm_features[feat_cols_musicbrainz_lastfm]
y_train_musicbrainz_lastfm = train_musicbrainz_lastfm_features["label"]

training_datasets = [
    (X_train_discogs_lastfm, y_train_discogs_lastfm),
    (X_train_discogs_musicbrainz, y_train_discogs_musicbrainz),
    (X_train_musicbrainz_lastfm, y_train_musicbrainz_lastfm),
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

ml_matcher_discogs_lastfm = MLBasedMatcher(feature_extractor_discogs_lastfm)
ml_matcher_discogs_musicbrainz = MLBasedMatcher(feature_extractor_discogs_musicbrainz)
ml_matcher_musicbrainz_lastfm = MLBasedMatcher(feature_extractor_musicbrainz_lastfm)

ml_correspondences_discogs_lastfm = ml_matcher_discogs_lastfm.match(
    discogs,
    lastfm,
    candidates=blocker_discogs_lastfm,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_discogs_musicbrainz = ml_matcher_discogs_musicbrainz.match(
    discogs,
    musicbrainz,
    candidates=blocker_discogs_musicbrainz,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_musicbrainz_lastfm = ml_matcher_musicbrainz_lastfm.match(
    musicbrainz,
    lastfm,
    candidates=blocker_musicbrainz_lastfm,
    id_column="id",
    trained_classifier=best_models[2],
)

print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_discogs_lastfm = clusterer.cluster(ml_correspondences_discogs_lastfm)
ml_correspondences_discogs_musicbrainz = clusterer.cluster(ml_correspondences_discogs_musicbrainz)
ml_correspondences_musicbrainz_lastfm = clusterer.cluster(ml_correspondences_musicbrainz_lastfm)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_discogs_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_lastfm.csv"),
    index=False,
)
ml_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(CORR_DIR, "correspondences_discogs_musicbrainz.csv"),
    index=False,
)
ml_correspondences_musicbrainz_lastfm.to_csv(
    os.path.join(CORR_DIR, "correspondences_musicbrainz_lastfm.csv"),
    index=False,
)

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_discogs_lastfm,
        ml_correspondences_discogs_musicbrainz,
        ml_correspondences_musicbrainz_lastfm,
    ],
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
PIPELINE SNAPSHOT 03 END
============================================================

