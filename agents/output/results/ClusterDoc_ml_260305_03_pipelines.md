# Pipeline Snapshots

notebook_name=ClusterDoc
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=53.81%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    StandardBlocker,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    MLBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
)
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
# --------------------------------

DATA_DIR = "output/schema-matching/"
TESTSET_DIR = "input/datasets/music/testsets/"

# Load datasets (already schema-matched externally, so no LLMBasedSchemaMatcher here)
discogs = load_csv(
    os.path.join(DATA_DIR, "discogs.csv"),
    name="discogs",
    add_index=False,
)
lastfm = load_csv(
    os.path.join(DATA_DIR, "lastfm.csv"),
    name="lastfm",
    add_index=False,
)
musicbrainz = load_csv(
    os.path.join(DATA_DIR, "musicbrainz.csv"),
    name="musicbrainz",
    add_index=False,
)

# Create id columns according to configuration (already named "id", just ensure existence)
discogs["id"] = discogs["id"]
lastfm["id"] = lastfm["id"]
musicbrainz["id"] = musicbrainz["id"]

datasets = [discogs, lastfm, musicbrainz]

# --------------------------------
# Blocking (use pre-computed configuration)
# --------------------------------

print("Performing Blocking")

# discogs <-> lastfm: semantic_similarity on ["name", "artist"], top_k=10
blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    id_column="id",
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    output_dir="output/blocking-evaluation",
)

# discogs <-> musicbrainz: sorted_neighbourhood on "name", window=15
blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    id_column="id",
    window=15,
    batch_size=100000,
    output_dir="output/blocking-evaluation",
)

# musicbrainz <-> lastfm: semantic_similarity on ["name", "artist", "duration"], top_k=10
blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    id_column="id",
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# Matching (use pre-computed comparator configuration)
# --------------------------------

print("Setting up Comparators and Feature Extractors")

# Helper preprocess function
def lower_strip(x):
    return str(x).lower().strip()


# discogs <-> lastfm comparators
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

# discogs <-> musicbrainz comparators
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

# musicbrainz <-> lastfm comparators
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
# Load ground truth pairs (for ML training)
# --------------------------------

print("Loading Ground Truth Pairs")

train_discogs_lastfm = load_csv(
    os.path.join(TESTSET_DIR, "discogs_lastfm_goldstandard_blocking.csv"),
    name="discogs_lastfm_goldstandard_blocking",
    add_index=False,
)
train_discogs_musicbrainz = load_csv(
    os.path.join(TESTSET_DIR, "discogs_musicbrainz_goldstandard_blocking.csv"),
    name="discogs_musicbrainz_goldstandard_blocking",
    add_index=False,
)
train_musicbrainz_lastfm = load_csv(
    os.path.join(TESTSET_DIR, "musicbrainz_lastfm_goldstandard_blocking.csv"),
    name="musicbrainz_lastfm_goldstandard_blocking",
    add_index=False,
)

# Ensure label column is present and correctly typed
for df in [train_discogs_lastfm, train_discogs_musicbrainz, train_musicbrainz_lastfm]:
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)

# --------------------------------
# Feature Extraction for Training
# --------------------------------

print("Extracting Training Features")

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
    col
    for col in train_discogs_lastfm_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_discogs_lastfm = train_discogs_lastfm_features[feat_cols_discogs_lastfm]
y_train_discogs_lastfm = train_discogs_lastfm_features["label"]

feat_cols_discogs_musicbrainz = [
    col
    for col in train_discogs_musicbrainz_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_discogs_musicbrainz = train_discogs_musicbrainz_features[
    feat_cols_discogs_musicbrainz
]
y_train_discogs_musicbrainz = train_discogs_musicbrainz_features["label"]

feat_cols_musicbrainz_lastfm = [
    col
    for col in train_musicbrainz_lastfm_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_musicbrainz_lastfm = train_musicbrainz_lastfm_features[
    feat_cols_musicbrainz_lastfm
]
y_train_musicbrainz_lastfm = train_musicbrainz_lastfm_features["label"]

training_datasets = [
    (X_train_discogs_lastfm, y_train_discogs_lastfm),
    (X_train_discogs_musicbrainz, y_train_discogs_musicbrainz),
    (X_train_musicbrainz_lastfm, y_train_musicbrainz_lastfm),
]

# --------------------------------
# Model Selection
# --------------------------------

print("Selecting Best Models")

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
            "max_depth": [3, 5],
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

for X_train, y_train in training_datasets:
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
        grid_search.fit(X_train, y_train)
        if grid_search.best_score_ > best_overall_score:
            best_overall_model = grid_search.best_estimator_
            best_overall_score = grid_search.best_score_

    best_models.append(best_overall_model)

# --------------------------------
# Matching Entities
# --------------------------------

print("Matching Entities")

ml_matcher_discogs_lastfm = MLBasedMatcher(feature_extractor_discogs_lastfm)
ml_matcher_discogs_musicbrainz = MLBasedMatcher(feature_extractor_discogs_musicbrainz)
ml_matcher_musicbrainz_lastfm = MLBasedMatcher(feature_extractor_musicbrainz_lastfm)

# Order of datasets must correspond to testset file naming:
# discogs_lastfm, discogs_musicbrainz, musicbrainz_lastfm

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

print("Applying Maximum Bipartite Matching")

clusterer = MaximumBipartiteMatching()
ml_correspondences_discogs_lastfm = clusterer.cluster(
    ml_correspondences_discogs_lastfm
)
ml_correspondences_discogs_musicbrainz = clusterer.cluster(
    ml_correspondences_discogs_musicbrainz
)
ml_correspondences_musicbrainz_lastfm = clusterer.cluster(
    ml_correspondences_musicbrainz_lastfm
)

# --------------------------------
# Save Correspondences
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_discogs_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_lastfm.csv",
    ),
    index=False,
)
ml_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_musicbrainz.csv",
    ),
    index=False,
)
ml_correspondences_musicbrainz_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_musicbrainz_lastfm.csv",
    ),
    index=False,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_discogs_lastfm,
        ml_correspondences_discogs_musicbrainz,
        ml_correspondences_musicbrainz_lastfm,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy_music")

# Basic textual attributes
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("label", longest_string)
strategy.add_attribute_fuser("genre", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("release-date", longest_string)

# Numeric-ish attributes (keep as string/longest representation)
strategy.add_attribute_fuser("duration", longest_string)

# Track list attributes
strategy.add_attribute_fuser("tracks_track_name", union)
strategy.add_attribute_fuser("tracks_track_position", union)
strategy.add_attribute_fuser("tracks_track_duration", union)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=False,
)

os.makedirs("output/data_fusion", exist_ok=True)
ml_fused.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=13
node_name=execute_pipeline
accuracy_score=53.81%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    StandardBlocker,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    MLBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
)
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
# --------------------------------

DATA_DIR = "output/schema-matching/"
TESTSET_DIR = "input/datasets/music/testsets/"

# Load datasets (already schema-matched)
discogs = load_csv(
    os.path.join(DATA_DIR, "discogs.csv"),
    name="discogs",
    add_index=False,
)
lastfm = load_csv(
    os.path.join(DATA_DIR, "lastfm.csv"),
    name="lastfm",
    add_index=False,
)
musicbrainz = load_csv(
    os.path.join(DATA_DIR, "musicbrainz.csv"),
    name="musicbrainz",
    add_index=False,
)

# Ensure id columns as required by configuration
discogs["id"] = discogs["id"]
lastfm["id"] = lastfm["id"]
musicbrainz["id"] = musicbrainz["id"]

datasets = [discogs, lastfm, musicbrainz]

# --------------------------------
# Blocking (use pre-computed configuration)
# --------------------------------

print("Performing Blocking")

# discogs <-> lastfm: semantic_similarity on ["name", "artist"], top_k=10
blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    id_column="id",
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    output_dir="output/blocking-evaluation",
)

# discogs <-> musicbrainz: sorted_neighbourhood on "name", window=15
blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    id_column="id",
    window=15,
    batch_size=100000,
    output_dir="output/blocking-evaluation",
)

# musicbrainz <-> lastfm: semantic_similarity on ["name", "artist", "duration"], top_k=10
blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    id_column="id",
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# Matching (use pre-computed comparator configuration)
# --------------------------------

print("Setting up Comparators and Feature Extractors")


def lower_strip(x):
    return str(x).lower().strip()


# discogs <-> lastfm comparators
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

# discogs <-> musicbrainz comparators
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

# musicbrainz <-> lastfm comparators
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
# Load ground truth pairs (for ML training)
# --------------------------------

print("Loading Ground Truth Pairs")

train_discogs_lastfm = load_csv(
    os.path.join(TESTSET_DIR, "discogs_lastfm_goldstandard_blocking.csv"),
    name="discogs_lastfm_goldstandard_blocking",
    add_index=False,
)
train_discogs_musicbrainz = load_csv(
    os.path.join(TESTSET_DIR, "discogs_musicbrainz_goldstandard_blocking.csv"),
    name="discogs_musicbrainz_goldstandard_blocking",
    add_index=False,
)
train_musicbrainz_lastfm = load_csv(
    os.path.join(TESTSET_DIR, "musicbrainz_lastfm_goldstandard_blocking.csv"),
    name="musicbrainz_lastfm_goldstandard_blocking",
    add_index=False,
)

for df in [train_discogs_lastfm, train_discogs_musicbrainz, train_musicbrainz_lastfm]:
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)

# --------------------------------
# Feature Extraction for Training
# --------------------------------

print("Extracting Training Features")

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
    col
    for col in train_discogs_lastfm_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_discogs_lastfm = train_discogs_lastfm_features[feat_cols_discogs_lastfm]
y_train_discogs_lastfm = train_discogs_lastfm_features["label"]

feat_cols_discogs_musicbrainz = [
    col
    for col in train_discogs_musicbrainz_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_discogs_musicbrainz = train_discogs_musicbrainz_features[
    feat_cols_discogs_musicbrainz
]
y_train_discogs_musicbrainz = train_discogs_musicbrainz_features["label"]

feat_cols_musicbrainz_lastfm = [
    col
    for col in train_musicbrainz_lastfm_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_musicbrainz_lastfm = train_musicbrainz_lastfm_features[
    feat_cols_musicbrainz_lastfm
]
y_train_musicbrainz_lastfm = train_musicbrainz_lastfm_features["label"]

training_datasets = [
    (X_train_discogs_lastfm, y_train_discogs_lastfm),
    (X_train_discogs_musicbrainz, y_train_discogs_musicbrainz),
    (X_train_musicbrainz_lastfm, y_train_musicbrainz_lastfm),
]

# --------------------------------
# Model Selection
# --------------------------------

print("Selecting Best Models")

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
            "max_depth": [3, 5],
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

for X_train, y_train in training_datasets:
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
        grid_search.fit(X_train, y_train)
        if grid_search.best_score_ > best_overall_score:
            best_overall_model = grid_search.best_estimator_
            best_overall_score = grid_search.best_score_

    best_models.append(best_overall_model)

# --------------------------------
# Matching Entities
# --------------------------------

print("Matching Entities")

ml_matcher_discogs_lastfm = MLBasedMatcher(feature_extractor_discogs_lastfm)
ml_matcher_discogs_musicbrainz = MLBasedMatcher(feature_extractor_discogs_musicbrainz)
ml_matcher_musicbrainz_lastfm = MLBasedMatcher(feature_extractor_musicbrainz_lastfm)

# Order of datasets must correspond to testset file naming:
# discogs_lastfm, discogs_musicbrainz, musicbrainz_lastfm

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

print("Applying Maximum Bipartite Matching")

clusterer = MaximumBipartiteMatching()
ml_correspondences_discogs_lastfm = clusterer.cluster(
    ml_correspondences_discogs_lastfm
)
ml_correspondences_discogs_musicbrainz = clusterer.cluster(
    ml_correspondences_discogs_musicbrainz
)
ml_correspondences_musicbrainz_lastfm = clusterer.cluster(
    ml_correspondences_musicbrainz_lastfm
)

# --------------------------------
# Save Correspondences
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_discogs_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_lastfm.csv",
    ),
    index=False,
)
ml_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_musicbrainz.csv",
    ),
    index=False,
)
ml_correspondences_musicbrainz_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_musicbrainz_lastfm.csv",
    ),
    index=False,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_discogs_lastfm,
        ml_correspondences_discogs_musicbrainz,
        ml_correspondences_musicbrainz_lastfm,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy_music")

# Textual attributes
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("label", longest_string)
strategy.add_attribute_fuser("genre", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("release-date", longest_string)

# Duration (keep the longest representation)
strategy.add_attribute_fuser("duration", longest_string)

# Track list attributes
strategy.add_attribute_fuser("tracks_track_name", union)
strategy.add_attribute_fuser("tracks_track_position", union)
strategy.add_attribute_fuser("tracks_track_duration", union)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=False,
)

os.makedirs("output/data_fusion", exist_ok=True)
ml_fused.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=20
node_name=execute_pipeline
accuracy_score=47.21%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    StandardBlocker,
    EmbeddingBlocker,
    SortedNeighbourhoodBlocker,
    MLBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
)
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
# --------------------------------

DATA_DIR = "output/schema-matching/"
TESTSET_DIR = "input/datasets/music/testsets/"

# Load datasets (already schema-matched)
discogs = load_csv(
    os.path.join(DATA_DIR, "discogs.csv"),
    name="discogs",
    add_index=False,
)
lastfm = load_csv(
    os.path.join(DATA_DIR, "lastfm.csv"),
    name="lastfm",
    add_index=False,
)
musicbrainz = load_csv(
    os.path.join(DATA_DIR, "musicbrainz.csv"),
    name="musicbrainz",
    add_index=False,
)

# Ensure id columns as required by configuration
discogs["id"] = discogs["id"]
lastfm["id"] = lastfm["id"]
musicbrainz["id"] = musicbrainz["id"]

datasets = [discogs, lastfm, musicbrainz]

# --------------------------------
# Blocking (use pre-computed configuration)
# --------------------------------

print("Performing Blocking")

# discogs <-> lastfm: semantic_similarity on ["name", "artist"], top_k=10
blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    id_column="id",
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    output_dir="output/blocking-evaluation",
)

# discogs <-> musicbrainz: sorted_neighbourhood on "name", window=15
blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    id_column="id",
    window=15,
    batch_size=100000,
    output_dir="output/blocking-evaluation",
)

# musicbrainz <-> lastfm: semantic_similarity on ["name", "artist", "duration"], top_k=10
blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    id_column="id",
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    output_dir="output/blocking-evaluation",
)

# --------------------------------
# Matching (use pre-computed comparator configuration)
# --------------------------------

print("Setting up Comparators and Feature Extractors")


def lower_strip(x):
    return str(x).lower().strip()


# discogs <-> lastfm comparators
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

# discogs <-> musicbrainz comparators
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

# musicbrainz <-> lastfm comparators
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
# Load ground truth pairs (for ML training)
# --------------------------------

print("Loading Ground Truth Pairs")

train_discogs_lastfm = load_csv(
    os.path.join(TESTSET_DIR, "discogs_lastfm_goldstandard_blocking.csv"),
    name="discogs_lastfm_goldstandard_blocking",
    add_index=False,
)
train_discogs_musicbrainz = load_csv(
    os.path.join(TESTSET_DIR, "discogs_musicbrainz_goldstandard_blocking.csv"),
    name="discogs_musicbrainz_goldstandard_blocking",
    add_index=False,
)
train_musicbrainz_lastfm = load_csv(
    os.path.join(TESTSET_DIR, "musicbrainz_lastfm_goldstandard_blocking.csv"),
    name="musicbrainz_lastfm_goldstandard_blocking",
    add_index=False,
)

for df in [train_discogs_lastfm, train_discogs_musicbrainz, train_musicbrainz_lastfm]:
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)

# --------------------------------
# Feature Extraction for Training
# --------------------------------

print("Extracting Training Features")

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
    col
    for col in train_discogs_lastfm_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_discogs_lastfm = train_discogs_lastfm_features[feat_cols_discogs_lastfm]
y_train_discogs_lastfm = train_discogs_lastfm_features["label"]

feat_cols_discogs_musicbrainz = [
    col
    for col in train_discogs_musicbrainz_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_discogs_musicbrainz = train_discogs_musicbrainz_features[
    feat_cols_discogs_musicbrainz
]
y_train_discogs_musicbrainz = train_discogs_musicbrainz_features["label"]

feat_cols_musicbrainz_lastfm = [
    col
    for col in train_musicbrainz_lastfm_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_musicbrainz_lastfm = train_musicbrainz_lastfm_features[
    feat_cols_musicbrainz_lastfm
]
y_train_musicbrainz_lastfm = train_musicbrainz_lastfm_features["label"]

training_datasets = [
    (X_train_discogs_lastfm, y_train_discogs_lastfm),
    (X_train_discogs_musicbrainz, y_train_discogs_musicbrainz),
    (X_train_musicbrainz_lastfm, y_train_musicbrainz_lastfm),
]

# --------------------------------
# Model Selection
# --------------------------------

print("Selecting Best Models")

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
            "max_depth": [3, 5],
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

for X_train, y_train in training_datasets:
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
        grid_search.fit(X_train, y_train)
        if grid_search.best_score_ > best_overall_score:
            best_overall_model = grid_search.best_estimator_
            best_overall_score = grid_search.best_score_

    best_models.append(best_overall_model)

# --------------------------------
# Matching Entities
# --------------------------------

print("Matching Entities")

ml_matcher_discogs_lastfm = MLBasedMatcher(feature_extractor_discogs_lastfm)
ml_matcher_discogs_musicbrainz = MLBasedMatcher(feature_extractor_discogs_musicbrainz)
ml_matcher_musicbrainz_lastfm = MLBasedMatcher(feature_extractor_musicbrainz_lastfm)

# Order of datasets must correspond to testset file naming:
# discogs_lastfm, discogs_musicbrainz, musicbrainz_lastfm

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

print("Applying Maximum Bipartite Matching")

clusterer = MaximumBipartiteMatching()
ml_correspondences_discogs_lastfm = clusterer.cluster(
    ml_correspondences_discogs_lastfm
)
ml_correspondences_discogs_musicbrainz = clusterer.cluster(
    ml_correspondences_discogs_musicbrainz
)
ml_correspondences_musicbrainz_lastfm = clusterer.cluster(
    ml_correspondences_musicbrainz_lastfm
)

# --------------------------------
# Save Correspondences
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_discogs_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_lastfm.csv",
    ),
    index=False,
)
ml_correspondences_discogs_musicbrainz.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_discogs_musicbrainz.csv",
    ),
    index=False,
)
ml_correspondences_musicbrainz_lastfm.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_musicbrainz_lastfm.csv",
    ),
    index=False,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_discogs_lastfm,
        ml_correspondences_discogs_musicbrainz,
        ml_correspondences_musicbrainz_lastfm,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy_music")

# Textual attributes
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("label", longest_string)
strategy.add_attribute_fuser("genre", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("release-date", longest_string)

# Duration (keep the longest representation)
strategy.add_attribute_fuser("duration", longest_string)

# Track list attributes
strategy.add_attribute_fuser("tracks_track_name", union)
strategy.add_attribute_fuser("tracks_track_position", union)
strategy.add_attribute_fuser("tracks_track_duration", union)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=False,
)

os.makedirs("output/data_fusion", exist_ok=True)
ml_fused.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

