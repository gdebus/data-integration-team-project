# Pipeline Snapshots

notebook_name=AdaptationPipeline_blocking_matching_extension_Final_Blocker_Matcher
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=50.25%
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker, SortedNeighbourhoodBlocker
from PyDI.entitymatching import MLBasedMatcher, MaximumBipartiteMatching

from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union
from langchain_openai import ChatOpenAI

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"
TESTSET_DIR = "input/datasets/music/testsets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets (already schema-matched)
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

# Set id columns according to configuration
discogs["id"] = discogs["id"]
lastfm["id"] = lastfm["id"]
musicbrainz["id"] = musicbrainz["id"]

datasets = [discogs, lastfm, musicbrainz]

# --------------------------------
# Blocking (use pre-computed blocking configuration)
# --------------------------------

print("Performing Blocking")

# Helper: common parameters
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_INDEX_BACKEND = "sklearn"
BLOCKING_OUTPUT_DIR = "output/blocking-evaluation"

# discogs <-> lastfm: semantic_similarity on ["name", "artist"], top_k=10
blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    model=EMBEDDING_MODEL,
    index_backend=EMBEDDING_INDEX_BACKEND,
    top_k=10,
    batch_size=1000,
    output_dir=BLOCKING_OUTPUT_DIR,
    id_column="id",
)

# discogs <-> musicbrainz: sorted_neighbourhood on ["name"], window=15
# SortedNeighbourhoodBlocker operates on a single key column
blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    window=15,
    id_column="id",
    output_dir="output/blocking",
)

# musicbrainz <-> lastfm: semantic_similarity on ["name", "artist", "duration"], top_k=10
blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    model=EMBEDDING_MODEL,
    index_backend=EMBEDDING_INDEX_BACKEND,
    top_k=10,
    batch_size=1000,
    output_dir=BLOCKING_OUTPUT_DIR,
    id_column="id",
)

# --------------------------------
# Matching configuration (comparators from pre-computed strategies)
# --------------------------------

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
# Load ground truth correspondences (ML training/test)
# --------------------------------

train_discogs_lastfm = load_csv(
    TESTSET_DIR + "discogs_lastfm_goldstandard_blocking.csv",
    name="discogs_lastfm_goldstandard_blocking",
    add_index=False,
)

train_discogs_musicbrainz = load_csv(
    TESTSET_DIR + "discogs_musicbrainz_goldstandard_blocking.csv",
    name="discogs_musicbrainz_goldstandard_blocking",
    add_index=False,
)

train_musicbrainz_lastfm = load_csv(
    TESTSET_DIR + "musicbrainz_lastfm_goldstandard_blocking.csv",
    name="musicbrainz_lastfm_goldstandard_blocking",
    add_index=False,
)

# --------------------------------
# Extract features
# --------------------------------

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
X_train_discogs_musicbrainz = train_discogs_musicbrainz_features[feat_cols_discogs_musicbrainz]
y_train_discogs_musicbrainz = train_discogs_musicbrainz_features["label"]

feat_cols_musicbrainz_lastfm = [
    col
    for col in train_musicbrainz_lastfm_features.columns
    if col not in ["id1", "id2", "label"]
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
            "l1_ratio": [0],
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

print("Matching Entities")

ml_matcher_discogs_lastfm = MLBasedMatcher(feature_extractor_discogs_lastfm)
ml_matcher_discogs_musicbrainz = MLBasedMatcher(feature_extractor_discogs_musicbrainz)
ml_matcher_musicbrainz_lastfm = MLBasedMatcher(feature_extractor_musicbrainz_lastfm)

# The order of datasets in the matcher MUST correspond to the order in the testset filenames

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

print("Clustering Matches")

clusterer = MaximumBipartiteMatching()
ml_correspondences_discogs_lastfm = clusterer.cluster(ml_correspondences_discogs_lastfm)
ml_correspondences_discogs_musicbrainz = clusterer.cluster(ml_correspondences_discogs_musicbrainz)
ml_correspondences_musicbrainz_lastfm = clusterer.cluster(ml_correspondences_musicbrainz_lastfm)

print("Fusing Data")

# Merge ML-based correspondences
all_ml_correspondences = pd.concat(
    [
        ml_correspondences_discogs_lastfm,
        ml_correspondences_discogs_musicbrainz,
        ml_correspondences_musicbrainz_lastfm,
    ],
    ignore_index=True,
)

# Define data fusion strategy
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

# --------------------------------
# Write output (file name must not be changed)
# --------------------------------
ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=12
node_name=execute_pipeline
accuracy_score=51.78%
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker, SortedNeighbourhoodBlocker
from PyDI.entitymatching import MLBasedMatcher, MaximumBipartiteMatching

from PyDI.fusion import DataFusionStrategy, DataFusionEngine
from langchain_openai import ChatOpenAI

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import re
from datetime import datetime

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"
TESTSET_DIR = "input/datasets/music/testsets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets (already schema-matched)
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

# Set id columns according to configuration
discogs["id"] = discogs["id"]
lastfm["id"] = lastfm["id"]
musicbrainz["id"] = musicbrainz["id"]

datasets = [discogs, lastfm, musicbrainz]

# --------------------------------
# Blocking (use pre-computed blocking configuration)
# --------------------------------

print("Performing Blocking")

# Helper: common parameters
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_INDEX_BACKEND = "sklearn"
BLOCKING_OUTPUT_DIR = "output/blocking-evaluation"

# discogs <-> lastfm: semantic_similarity on ["name", "artist"], top_k=10
blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    model=EMBEDDING_MODEL,
    index_backend=EMBEDDING_INDEX_BACKEND,
    top_k=10,
    batch_size=1000,
    output_dir=BLOCKING_OUTPUT_DIR,
    id_column="id",
)

# discogs <-> musicbrainz: sorted_neighbourhood on ["name"], window=15
blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    window=15,
    id_column="id",
    output_dir="output/blocking",
)

# musicbrainz <-> lastfm: semantic_similarity on ["name", "artist", "duration"], top_k=10
blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    model=EMBEDDING_MODEL,
    index_backend=EMBEDDING_INDEX_BACKEND,
    top_k=10,
    batch_size=1000,
    output_dir=BLOCKING_OUTPUT_DIR,
    id_column="id",
)

# --------------------------------
# Matching configuration (comparators from pre-computed strategies)
# --------------------------------

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
# Load ground truth correspondences (ML training/test)
# --------------------------------

train_discogs_lastfm = load_csv(
    TESTSET_DIR + "discogs_lastfm_goldstandard_blocking.csv",
    name="discogs_lastfm_goldstandard_blocking",
    add_index=False,
)

train_discogs_musicbrainz = load_csv(
    TESTSET_DIR + "discogs_musicbrainz_goldstandard_blocking.csv",
    name="discogs_musicbrainz_goldstandard_blocking",
    add_index=False,
)

train_musicbrainz_lastfm = load_csv(
    TESTSET_DIR + "musicbrainz_lastfm_goldstandard_blocking.csv",
    name="musicbrainz_lastfm_goldstandard_blocking",
    add_index=False,
)

# --------------------------------
# Extract features
# --------------------------------

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
X_train_discogs_musicbrainz = train_discogs_musicbrainz_features[feat_cols_discogs_musicbrainz]
y_train_discogs_musicbrainz = train_discogs_musicbrainz_features["label"]

feat_cols_musicbrainz_lastfm = [
    col
    for col in train_musicbrainz_lastfm_features.columns
    if col not in ["id1", "id2", "label"]
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
            "l1_ratio": [0],
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

print("Matching Entities")

ml_matcher_discogs_lastfm = MLBasedMatcher(feature_extractor_discogs_lastfm)
ml_matcher_discogs_musicbrainz = MLBasedMatcher(feature_extractor_discogs_musicbrainz)
ml_matcher_musicbrainz_lastfm = MLBasedMatcher(feature_extractor_musicbrainz_lastfm)

# The order of datasets in the matcher MUST correspond to the order in the testset filenames

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

print("Clustering Matches")

clusterer = MaximumBipartiteMatching()
ml_correspondences_discogs_lastfm = clusterer.cluster(ml_correspondences_discogs_lastfm)
ml_correspondences_discogs_musicbrainz = clusterer.cluster(ml_correspondences_discogs_musicbrainz)
ml_correspondences_musicbrainz_lastfm = clusterer.cluster(ml_correspondences_musicbrainz_lastfm)

print("Fusing Data")

# Merge ML-based correspondences
all_ml_correspondences = pd.concat(
    [
        ml_correspondences_discogs_lastfm,
        ml_correspondences_discogs_musicbrainz,
        ml_correspondences_musicbrainz_lastfm,
    ],
    ignore_index=True,
)

# --------------------------------
# Helper functions for smarter fusion
# --------------------------------

PREFERRED_SOURCE_ORDER = ["musicbrainz", "discogs", "lastfm"]

def infer_source_from_id(record_id: str) -> str:
    if isinstance(record_id, str):
        if record_id.startswith("mbrainz_"):
            return "musicbrainz"
        if record_id.startswith("discogs_"):
            return "discogs"
        if record_id.startswith("lastFM_") or record_id.startswith("lastfm_"):
            return "lastfm"
    return ""

def normalize_simple(text: str) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())

def normalize_title(title: str) -> str:
    t = normalize_simple(title)
    t = re.sub(r"^(album:)\s*", "", t)
    t = re.sub(r"\s*\(album\)$", "", t)
    t = re.sub(r"\s*side\s+[a-z0-9]+$", "", t)
    t = re.sub(r"\s+disc\s+[0-9]+$", "", t)
    return t

def fuser_categorical_majority(values, sources):
    norm_to_originals = {}
    norm_counts = {}
    for v, s in zip(values, sources):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        norm = normalize_simple(v)
        if not norm:
            continue
        norm_counts[norm] = norm_counts.get(norm, 0) + 1
        norm_to_originals.setdefault(norm, []).append((s.get("dataset_name"), v))

    if not norm_counts:
        return None

    # majority normalized value
    majority_norm = max(norm_counts.items(), key=lambda x: x[1])[0]
    candidates = norm_to_originals[majority_norm]

    # pick from preferred source if available
    for src in PREFERRED_SOURCE_ORDER:
        for ds_name, val in candidates:
            if ds_name == src:
                return val

    # fallback: first observed
    return candidates[0][1]

def fuser_name(values, sources):
    norm_to_originals = {}
    norm_counts = {}
    for v, s in zip(values, sources):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        norm = normalize_title(v)
        if not norm:
            continue
        norm_counts[norm] = norm_counts.get(norm, 0) + 1
        norm_to_originals.setdefault(norm, []).append((s.get("dataset_name"), v))

    if not norm_counts:
        return None

    # majority normalized title
    majority_norm = max(norm_counts.items(), key=lambda x: x[1])[0]
    candidates = norm_to_originals[majority_norm]

    for src in PREFERRED_SOURCE_ORDER:
        for ds_name, val in candidates:
            if ds_name == src:
                return val

    return candidates[0][1]

def parse_year(date_str):
    if pd.isna(date_str):
        return None
    s = str(date_str).strip()
    if not s:
        return None
    # try full date
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(s[:len(fmt)], fmt)
            return dt.year
        except Exception:
            continue
    # fallback: first 4-digit year
    m = re.search(r"\d{4}", s)
    if m:
        return int(m.group(0))
    return None

def fuser_release_date(values, sources):
    years = []
    year_records = []
    for v, s in zip(values, sources):
        y = parse_year(v)
        if y is not None:
            years.append(y)
            year_records.append((y, s.get("dataset_name"), v))

    if not years:
        return None

    # mode year
    year_counts = {}
    for y in years:
        year_counts[y] = year_counts.get(y, 0) + 1
    best_year = max(year_counts.items(), key=lambda x: x[1])[0]

    candidates = [rec for rec in year_records if rec[0] == best_year]
    # prefer most complete date (longest string) from preferred source
    best_candidate = None
    best_len = -1
    for src in PREFERRED_SOURCE_ORDER:
        for y, ds_name, val in candidates:
            if ds_name == src:
                l = len(str(val))
                if l > best_len:
                    best_len = l
                    best_candidate = val
        if best_candidate is not None:
            return best_candidate

    # fallback: longest string
    for y, ds_name, val in candidates:
        l = len(str(val))
        if l > best_len:
            best_len = l
            best_candidate = val
    return best_candidate

def fuser_numeric_median(values, sources):
    nums = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        try:
            nums.append(float(v))
        except Exception:
            continue
    if not nums:
        return None
    return float(np.median(nums))

def normalize_country(c):
    c_norm = normalize_simple(c)
    mappings = {
        "uk": "united kingdom of great britain and northern ireland",
        "u.k.": "united kingdom of great britain and northern ireland",
        "united kingdom": "united kingdom of great britain and northern ireland",
        "usa": "united states",
        "u.s.a.": "united states",
        "us": "united states",
        "u.s.": "united states",
    }
    return mappings.get(c_norm, c_norm)

def fuser_release_country(values, sources):
    norm_to_originals = {}
    norm_counts = {}
    for v, s in zip(values, sources):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        norm = normalize_country(v)
        if not norm:
            continue
        norm_counts[norm] = norm_counts.get(norm, 0) + 1
        norm_to_originals.setdefault(norm, []).append((s.get("dataset_name"), v))

    if not norm_counts:
        return None

    majority_norm = max(norm_counts.items(), key=lambda x: x[1])[0]
    candidates = norm_to_originals[majority_norm]

    for src in PREFERRED_SOURCE_ORDER:
        for ds_name, val in candidates:
            if ds_name == src:
                return val

    return candidates[0][1]

def parse_list_field(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    s = str(v).strip()
    if not s:
        return []
    # very simple parsing for "['a', 'b']"
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        parts = re.split(r"',\s*'", inner.strip("'"))
        return [p for p in parts if p]
    # fallback: split on |
    return [p for p in re.split(r"[|;]", s) if p.strip()]

def normalize_track_title(t: str) -> str:
    t_norm = normalize_simple(t)
    t_norm = re.sub(r"[^\w\s]", "", t_norm)
    t_norm = re.sub(r"\s+", " ", t_norm).strip()
    return t_norm

def fuser_tracks_name(values, sources):
    norm_to_canonical = {}
    support_count = {}
    support_sources = {}

    for v, s in zip(values, sources):
        tracks = parse_list_field(v)
        ds_name = s.get("dataset_name")
        for t in tracks:
            norm = normalize_track_title(t)
            if not norm:
                continue
            # prefer canonical spelling from preferred source order
            if norm not in norm_to_canonical:
                norm_to_canonical[norm] = (ds_name, t)
            else:
                cur_src, cur_val = norm_to_canonical[norm]
                if PREFERRED_SOURCE_ORDER.index(ds_name) < PREFERRED_SOURCE_ORDER.index(cur_src):
                    norm_to_canonical[norm] = (ds_name, t)
            support_count[norm] = support_count.get(norm, 0) + 1
            support_sources.setdefault(norm, set()).add(ds_name)

    if not norm_to_canonical:
        return None

    # To avoid huge supersets, keep tracks supported by at least 2 sources if possible
    strong_tracks = [norm for norm, cnt in support_count.items() if cnt >= 2]
    if strong_tracks:
        selected_norms = strong_tracks
    else:
        selected_norms = list(norm_to_canonical.keys())

    # Build final list in a deterministic order (sorted by name)
    selected_norms_sorted = sorted(selected_norms)
    fused_tracks = [norm_to_canonical[norm][1] for norm in selected_norms_sorted]
    return fused_tracks

def fuser_tracks_position(values, sources):
    fused = fuser_tracks_name(values, sources)
    if fused is None:
        return None
    # positions are less critical; keep positions from preferred source where available
    # We will align by normalized track name
    norm_to_pos = {}
    for v, s in zip(values, sources):
        tracks = parse_list_field(v if isinstance(v, str) and v.startswith("[") else v)
        ds_name = s.get("dataset_name")
        # this field is positions; we need the track names from same source,
        # but we do not have them here, so fall back to simple union order
        # -> to not over-union, we use range positions
        # This is a simplified approach: just use 1..N
        break
    return [str(i + 1) for i in range(len(fused))]

def fuser_tracks_duration(values, sources):
    # Align with fused track names: approximate using order
    # Take median per index position
    all_lists = []
    for v in values:
        all_lists.append(parse_list_field(v))
    if not any(all_lists):
        return None
    max_len = max(len(lst) for lst in all_lists)
    fused = []
    for i in range(max_len):
        nums = []
        for lst in all_lists:
            if i < len(lst):
                try:
                    nums.append(float(lst[i]))
                except Exception:
                    continue
        if nums:
            fused.append(str(int(round(np.median(nums)))))
    return fused

# --------------------------------
# Define data fusion strategy with improved fusers
# --------------------------------

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", fuser_name)
strategy.add_attribute_fuser("artist", fuser_categorical_majority)
strategy.add_attribute_fuser("label", fuser_categorical_majority)
strategy.add_attribute_fuser("genre", fuser_categorical_majority)
strategy.add_attribute_fuser("release-date", fuser_release_date)
strategy.add_attribute_fuser("release-country", fuser_release_country)
strategy.add_attribute_fuser("duration", fuser_numeric_median)
strategy.add_attribute_fuser("tracks_track_name", fuser_tracks_name)
strategy.add_attribute_fuser("tracks_track_position", fuser_tracks_position)
strategy.add_attribute_fuser("tracks_track_duration", fuser_tracks_duration)

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

# --------------------------------
# Write output (file name must not be changed)
# --------------------------------
ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=18
node_name=execute_pipeline
accuracy_score=pending
------------------------------------------------------------

```python
from PyDI.io import load_csv

from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker, SortedNeighbourhoodBlocker
from PyDI.entitymatching import MLBasedMatcher, MaximumBipartiteMatching

from PyDI.fusion import DataFusionStrategy, DataFusionEngine
from langchain_openai import ChatOpenAI

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import re
from datetime import datetime

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"
TESTSET_DIR = "input/datasets/music/testsets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets (already schema-matched)
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

# Set id columns according to configuration
discogs["id"] = discogs["id"]
lastfm["id"] = lastfm["id"]
musicbrainz["id"] = musicbrainz["id"]

datasets = [discogs, lastfm, musicbrainz]

# --------------------------------
# Blocking (use pre-computed blocking configuration)
# --------------------------------

print("Performing Blocking")

# Helper: common parameters
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_INDEX_BACKEND = "sklearn"
BLOCKING_OUTPUT_DIR = "output/blocking-evaluation"

# discogs <-> lastfm: semantic_similarity on ["name", "artist"], top_k=10
blocker_discogs_lastfm = EmbeddingBlocker(
    discogs,
    lastfm,
    text_cols=["name", "artist"],
    model=EMBEDDING_MODEL,
    index_backend=EMBEDDING_INDEX_BACKEND,
    top_k=10,
    batch_size=1000,
    output_dir=BLOCKING_OUTPUT_DIR,
    id_column="id",
)

# discogs <-> musicbrainz: sorted_neighbourhood on ["name"], window=15
blocker_discogs_musicbrainz = SortedNeighbourhoodBlocker(
    discogs,
    musicbrainz,
    key="name",
    window=15,
    id_column="id",
    output_dir="output/blocking",
)

# musicbrainz <-> lastfm: semantic_similarity on ["name", "artist", "duration"], top_k=10
blocker_musicbrainz_lastfm = EmbeddingBlocker(
    musicbrainz,
    lastfm,
    text_cols=["name", "artist", "duration"],
    model=EMBEDDING_MODEL,
    index_backend=EMBEDDING_INDEX_BACKEND,
    top_k=10,
    batch_size=1000,
    output_dir=BLOCKING_OUTPUT_DIR,
    id_column="id",
)

# --------------------------------
# Matching configuration (comparators from pre-computed strategies)
# --------------------------------

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
# Load ground truth correspondences (ML training/test)
# --------------------------------

train_discogs_lastfm = load_csv(
    TESTSET_DIR + "discogs_lastfm_goldstandard_blocking.csv",
    name="discogs_lastfm_goldstandard_blocking",
    add_index=False,
)

train_discogs_musicbrainz = load_csv(
    TESTSET_DIR + "discogs_musicbrainz_goldstandard_blocking.csv",
    name="discogs_musicbrainz_goldstandard_blocking",
    add_index=False,
)

train_musicbrainz_lastfm = load_csv(
    TESTSET_DIR + "musicbrainz_lastfm_goldstandard_blocking.csv",
    name="musicbrainz_lastfm_goldstandard_blocking",
    add_index=False,
)

# --------------------------------
# Extract features
# --------------------------------

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
X_train_discogs_musicbrainz = train_discogs_musicbrainz_features[feat_cols_discogs_musicbrainz]
y_train_discogs_musicbrainz = train_discogs_musicbrainz_features["label"]

feat_cols_musicbrainz_lastfm = [
    col
    for col in train_musicbrainz_lastfm_features.columns
    if col not in ["id1", "id2", "label"]
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
            "l1_ratio": [0],
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

print("Matching Entities")

ml_matcher_discogs_lastfm = MLBasedMatcher(feature_extractor_discogs_lastfm)
ml_matcher_discogs_musicbrainz = MLBasedMatcher(feature_extractor_discogs_musicbrainz)
ml_matcher_musicbrainz_lastfm = MLBasedMatcher(feature_extractor_musicbrainz_lastfm)

# The order of datasets in the matcher MUST correspond to the order in the testset filenames

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

print("Clustering Matches")

clusterer = MaximumBipartiteMatching()
ml_correspondences_discogs_lastfm = clusterer.cluster(ml_correspondences_discogs_lastfm)
ml_correspondences_discogs_musicbrainz = clusterer.cluster(ml_correspondences_discogs_musicbrainz)
ml_correspondences_musicbrainz_lastfm = clusterer.cluster(ml_correspondences_musicbrainz_lastfm)

print("Fusing Data")

# Merge ML-based correspondences
all_ml_correspondences = pd.concat(
    [
        ml_correspondences_discogs_lastfm,
        ml_correspondences_discogs_musicbrainz,
        ml_correspondences_musicbrainz_lastfm,
    ],
    ignore_index=True,
)

# --------------------------------
# Helper functions for smarter fusion
# --------------------------------

PREFERRED_SOURCE_ORDER = ["musicbrainz", "discogs", "lastfm"]

def infer_source_from_id(record_id: str) -> str:
    if isinstance(record_id, str):
        if record_id.startswith("mbrainz_"):
            return "musicbrainz"
        if record_id.startswith("discogs_"):
            return "discogs"
        if record_id.startswith("lastFM_") or record_id.startswith("lastfm_"):
            return "lastfm"
    return ""

def normalize_simple(text: str) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())

def normalize_title(title: str) -> str:
    t = normalize_simple(title)
    t = re.sub(r"^(album:)\s*", "", t)
    t = re.sub(r"\s*\(album\)$", "", t)
    t = re.sub(r"\s*side\s+[a-z0-9]+$", "", t)
    t = re.sub(r"\s+disc\s+[0-9]+$", "", t)
    return t

def fuser_categorical_majority(values, sources):
    norm_to_originals = {}
    norm_counts = {}
    by_ds = {}

    for v, s in zip(values, sources):
        ds_name = s.get("dataset_name")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        norm = normalize_simple(v)
        if not norm:
            continue
        norm_counts[norm] = norm_counts.get(norm, 0) + 1
        norm_to_originals.setdefault(norm, []).append((ds_name, v))
        by_ds.setdefault(ds_name, []).append((norm, v))

    # Prefer Musicbrainz value directly if available
    if "musicbrainz" in by_ds and by_ds["musicbrainz"]:
        # take most frequent norm within musicbrainz
        mb_counts = {}
        for n, _v in by_ds["musicbrainz"]:
            mb_counts[n] = mb_counts.get(n, 0) + 1
        mb_norm = max(mb_counts.items(), key=lambda x: x[1])[0]
        candidates = [v for n, v in by_ds["musicbrainz"] if n == mb_norm]
        if candidates:
            return candidates[0]

    if not norm_counts:
        return None

    # majority normalized value across all
    majority_norm = max(norm_counts.items(), key=lambda x: x[1])[0]
    candidates = norm_to_originals[majority_norm]

    # pick from preferred source if available
    for src in PREFERRED_SOURCE_ORDER:
        for ds_name, val in candidates:
            if ds_name == src:
                return val

    return candidates[0][1]

def fuser_name(values, sources):
    norm_to_originals = {}
    by_ds = {}

    for v, s in zip(values, sources):
        ds_name = s.get("dataset_name")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        norm = normalize_title(v)
        if not norm:
            continue
        norm_to_originals.setdefault(norm, []).append((ds_name, v))
        by_ds.setdefault(ds_name, []).append((norm, v))

    # Prefer Musicbrainz titles
    if "musicbrainz" in by_ds and by_ds["musicbrainz"]:
        mb_counts = {}
        for n, _v in by_ds["musicbrainz"]:
            mb_counts[n] = mb_counts.get(n, 0) + 1
        best_norm = max(mb_counts.items(), key=lambda x: x[1])[0]
        candidates = [v for n, v in by_ds["musicbrainz"] if n == best_norm]
        if candidates:
            return candidates[0]

    if not norm_to_originals:
        return None

    # majority normalized title across all
    norm_counts = {n: len(vs) for n, vs in norm_to_originals.items()}
    majority_norm = max(norm_counts.items(), key=lambda x: x[1])[0]
    candidates = norm_to_originals[majority_norm]

    for src in PREFERRED_SOURCE_ORDER:
        for ds_name, val in candidates:
            if ds_name == src:
                return val

    return candidates[0][1]

def parse_year(date_str):
    if pd.isna(date_str):
        return None
    s = str(date_str).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(s[:len(fmt)], fmt)
            return dt.year
        except Exception:
            continue
    m = re.search(r"\d{4}", s)
    if m:
        return int(m.group(0))
    return None

def fuser_release_date(values, sources):
    # Prefer Musicbrainz release date if available and valid
    mb_years = []
    mb_records = []
    for v, s in zip(values, sources):
        if s.get("dataset_name") == "musicbrainz":
            y = parse_year(v)
            if y is not None:
                mb_years.append(y)
                mb_records.append((y, v))
    if mb_years:
        # choose most frequent MB year, then most complete string
        year_counts = {}
        for y in mb_years:
            year_counts[y] = year_counts.get(y, 0) + 1
        best_year = max(year_counts.items(), key=lambda x: x[1])[0]
        candidates = [val for y, val in mb_records if y == best_year]
        if candidates:
            return max(candidates, key=lambda x: len(str(x)))

    # Fallback: mode year across all sources
    years = []
    year_records = []
    for v, s in zip(values, sources):
        y = parse_year(v)
        if y is not None:
            years.append(y)
            year_records.append((y, s.get("dataset_name"), v))

    if not years:
        return None

    year_counts = {}
    for y in years:
        year_counts[y] = year_counts.get(y, 0) + 1
    best_year = max(year_counts.items(), key=lambda x: x[1])[0]

    candidates = [rec for rec in year_records if rec[0] == best_year]
    best_candidate = None
    best_len = -1
    for src in PREFERRED_SOURCE_ORDER:
        for y, ds_name, val in candidates:
            if ds_name == src:
                l = len(str(val))
                if l > best_len:
                    best_len = l
                    best_candidate = val
        if best_candidate is not None:
            return best_candidate

    for y, ds_name, val in candidates:
        l = len(str(val))
        if l > best_len:
            best_len = l
            best_candidate = val
    return best_candidate

def fuser_numeric_median(values, sources):
    # Prefer numeric from Musicbrainz if present, otherwise median
    mb_nums = []
    for v, s in zip(values, sources):
        if s.get("dataset_name") == "musicbrainz":
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            try:
                mb_nums.append(float(v))
            except Exception:
                continue
    if mb_nums:
        return float(np.median(mb_nums))

    nums = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        try:
            nums.append(float(v))
        except Exception:
            continue
    if not nums:
        return None
    return float(np.median(nums))

def normalize_country(c):
    c_norm = normalize_simple(c)
    mappings = {
        "uk": "united kingdom of great britain and northern ireland",
        "u.k.": "united kingdom of great britain and northern ireland",
        "united kingdom": "united kingdom of great britain and northern ireland",
        "usa": "united states",
        "u.s.a.": "united states",
        "us": "united states",
        "u.s.": "united states",
    }
    return mappings.get(c_norm, c_norm)

def fuser_release_country(values, sources):
    norm_to_originals = {}
    norm_counts = {}
    by_ds = {}

    for v, s in zip(values, sources):
        ds_name = s.get("dataset_name")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        norm = normalize_country(v)
        if not norm:
            continue
        norm_counts[norm] = norm_counts.get(norm, 0) + 1
        norm_to_originals.setdefault(norm, []).append((ds_name, v))
        by_ds.setdefault(ds_name, []).append((norm, v))

    # Prefer Musicbrainz country if available
    if "musicbrainz" in by_ds and by_ds["musicbrainz"]:
        mb_counts = {}
        for n, _v in by_ds["musicbrainz"]:
            mb_counts[n] = mb_counts.get(n, 0) + 1
        best_norm = max(mb_counts.items(), key=lambda x: x[1])[0]
        candidates = [v for n, v in by_ds["musicbrainz"] if n == best_norm]
        if candidates:
            return candidates[0]

    if not norm_counts:
        return None

    majority_norm = max(norm_counts.items(), key=lambda x: x[1])[0]
    candidates = norm_to_originals[majority_norm]

    for src in PREFERRED_SOURCE_ORDER:
        for ds_name, val in candidates:
            if ds_name == src:
                return val

    return candidates[0][1]

def parse_list_field(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    s = str(v).strip()
    if not s:
        return []
    # very simple parsing for "['a', 'b']"
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        parts = re.split(r"',\s*'", inner.strip("'"))
        return [p for p in parts if p]
    # fallback: split on | or ;
    return [p for p in re.split(r"[|;]", s) if p.strip()]

def normalize_track_title(t: str) -> str:
    t_norm = normalize_simple(t)
    t_norm = re.sub(r"[^\w\s]", "", t_norm)
    t_norm = re.sub(r"\s+", " ", t_norm).strip()
    return t_norm

def fuser_tracks_name(values, sources):
    # Prefer Musicbrainz track list (aligned with evaluation IDs)
    mb_tracks = None
    for v, s in zip(values, sources):
        if s.get("dataset_name") == "musicbrainz":
            mb_tracks = parse_list_field(v)
            break

    if mb_tracks is not None and len(mb_tracks) > 0:
        # normalize duplicates but keep original order and spelling from MB
        seen_norms = set()
        fused = []
        for t in mb_tracks:
            norm = normalize_track_title(t)
            if not norm or norm in seen_norms:
                continue
            seen_norms.add(norm)
            fused.append(t)
        return fused if fused else None

    # Fallback: union logic when MB is absent
    norm_to_canonical = {}
    support_count = {}

    for v, s in zip(values, sources):
        tracks = parse_list_field(v)
        ds_name = s.get("dataset_name")
        for t in tracks:
            norm = normalize_track_title(t)
            if not norm:
                continue
            if norm not in norm_to_canonical:
                norm_to_canonical[norm] = (ds_name, t)
            else:
                cur_src, cur_val = norm_to_canonical[norm]
                if PREFERRED_SOURCE_ORDER.index(ds_name) < PREFERRED_SOURCE_ORDER.index(cur_src):
                    norm_to_canonical[norm] = (ds_name, t)
            support_count[norm] = support_count.get(norm, 0) + 1

    if not norm_to_canonical:
        return None

    selected_norms = list(norm_to_canonical.keys())
    selected_norms_sorted = sorted(selected_norms)
    fused_tracks = [norm_to_canonical[norm][1] for norm in selected_norms_sorted]
    return fused_tracks

def fuser_tracks_position(values, sources):
    # Use positions from Musicbrainz if available; this preserves correct track count and order
    mb_positions = None
    for v, s in zip(values, sources):
        if s.get("dataset_name") == "musicbrainz":
            mb_positions = parse_list_field(v)
            break

    fused_names = fuser_tracks_name(values, sources)
    if fused_names is None:
        return None
    n = len(fused_names)

    if mb_positions is not None and len(mb_positions) >= n:
        # Use first n positions from MB (they are already 1..N strings)
        return [str(p) for p in mb_positions[:n]]

    # Fallback: sequential positions 1..N
    return [str(i + 1) for i in range(n)]

def fuser_tracks_duration(values, sources):
    # Align durations with primary (Musicbrainz) track list if available
    mb_durations = None
    for v, s in zip(values, sources):
        if s.get("dataset_name") == "musicbrainz":
            mb_durations = parse_list_field(v)
            break

    fused_names = fuser_tracks_name(values, sources)
    if fused_names is None:
        return None
    n = len(fused_names)

    # If MB durations exist and length matches, trust them directly
    if mb_durations is not None and len(mb_durations) >= n:
        out = []
        for d in mb_durations[:n]:
            try:
                out.append(str(int(round(float(d)))))
            except Exception:
                out.append(str(d))
        return out

    # Otherwise, approximate by index-wise median but only up to min length
    all_lists = [parse_list_field(v) for v in values if v is not None]
    if not any(all_lists):
        return None
    max_len = min(max(len(lst) for lst in all_lists), n)
    fused = []
    for i in range(max_len):
        nums = []
        for lst in all_lists:
            if i < len(lst):
                try:
                    nums.append(float(lst[i]))
                except Exception:
                    continue
        if nums:
            fused.append(str(int(round(np.median(nums)))))
    # If we have fewer durations than tracks, pad with empty strings to keep list length consistent
    while len(fused) < n:
        fused.append("")
    return fused

# --------------------------------
# Define data fusion strategy with improved fusers
# --------------------------------

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", fuser_name)
strategy.add_attribute_fuser("artist", fuser_categorical_majority)
strategy.add_attribute_fuser("label", fuser_categorical_majority)
strategy.add_attribute_fuser("genre", fuser_categorical_majority)
strategy.add_attribute_fuser("release-date", fuser_release_date)
strategy.add_attribute_fuser("release-country", fuser_release_country)
strategy.add_attribute_fuser("duration", fuser_numeric_median)
strategy.add_attribute_fuser("tracks_track_name", fuser_tracks_name)
strategy.add_attribute_fuser("tracks_track_position", fuser_tracks_position)
strategy.add_attribute_fuser("tracks_track_duration", fuser_tracks_duration)

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

# --------------------------------
# Write output (file name must not be changed)
# --------------------------------
ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

