# Pipeline Snapshots

notebook_name=ClusterDoc
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=81.23%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv
from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import MLBasedMatcher, MaximumBipartiteMatching
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
from dotenv import load_dotenv
import os
import shutil


# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"
TESTSET_DIR = "input/datasets/restaurant/testsets/"

# Define API Key (if needed elsewhere, kept for parity with template)
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets (already schema-matched externally)
kaggle_small = load_parquet(
    DATA_DIR + "kaggle_small.parquet",
    name="kaggle_small",
)
uber_eats_small = load_parquet(
    DATA_DIR + "uber_eats_small.parquet",
    name="uber_eats_small",
)
yelp_small = load_parquet(
    DATA_DIR + "yelp_small.parquet",
    name="yelp_small",
)

# Set id columns according to configuration
kaggle_small["id"] = kaggle_small["id"]
uber_eats_small["id"] = uber_eats_small["id"]
yelp_small["id"] = yelp_small["id"]

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Blocking (use pre-computed optimal strategies)
# --------------------------------

print("Performing Blocking")

# Blocking config:
# kaggle_small <-> uber_eats_small: semantic_similarity on ["name_norm","city","state"], top_k=20
blocker_kaggle_uber = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=25000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# kaggle_small <-> yelp_small: exact_match_multi on ["phone_e164","postal_code"]
blocker_kaggle_yelp = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    id_column="id",
    batch_size=100000,
    output_dir="output/blocking-evaluation",
)

# uber_eats_small <-> yelp_small: semantic_similarity on ["name_norm","street","city"], top_k=20
blocker_uber_yelp = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=25000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching Configuration (comparators and feature extraction)
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()


def strip(x):
    return str(x).strip()


print("Setting up comparators and feature extractors")

# kaggle_small <-> uber_eats_small comparators
comparators_kaggle_uber = [
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="state",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="latitude",
        max_difference=0.01,
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.01,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

# kaggle_small <-> yelp_small comparators
comparators_kaggle_yelp = [
    StringComparator(
        column="phone_e164",
        similarity_function="jaro_winkler",
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="postal_code",
        similarity_function="jaro_winkler",
        preprocess=strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="latitude",
        max_difference=0.002,
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.002,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

# uber_eats_small <-> yelp_small comparators
comparators_uber_yelp = [
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="latitude",
        max_difference=0.002,
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.002,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

feature_extractor_kaggle_uber = FeatureExtractor(comparators_kaggle_uber)
feature_extractor_kaggle_yelp = FeatureExtractor(comparators_kaggle_yelp)
feature_extractor_uber_yelp = FeatureExtractor(comparators_uber_yelp)

# --------------------------------
# Load ground truth correspondences (ML training/test)
# --------------------------------

print("Loading ground truth testsets")

train_kaggle_uber = load_csv(
    TESTSET_DIR + "kaggle_uber_eats_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_uber_eats_small",
    add_index=False,
)

train_kaggle_yelp = load_csv(
    TESTSET_DIR + "kaggle_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_yelp_small",
    add_index=False,
)

train_uber_yelp = load_csv(
    TESTSET_DIR + "uber_eats_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_uber_eats_yelp_small",
    add_index=False,
)

# --------------------------------
# Feature extraction for training
# --------------------------------

print("Extracting features for training")

train_kaggle_uber_features = feature_extractor_kaggle_uber.create_features(
    kaggle_small,
    uber_eats_small,
    train_kaggle_uber[["id1", "id2"]],
    labels=train_kaggle_uber["label"],
    id_column="id",
)

train_kaggle_yelp_features = feature_extractor_kaggle_yelp.create_features(
    kaggle_small,
    yelp_small,
    train_kaggle_yelp[["id1", "id2"]],
    labels=train_kaggle_yelp["label"],
    id_column="id",
)

train_uber_yelp_features = feature_extractor_uber_yelp.create_features(
    uber_eats_small,
    yelp_small,
    train_uber_yelp[["id1", "id2"]],
    labels=train_uber_yelp["label"],
    id_column="id",
)

feat_cols_kaggle_uber = [
    col
    for col in train_kaggle_uber_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_kaggle_uber = train_kaggle_uber_features[feat_cols_kaggle_uber]
y_train_kaggle_uber = train_kaggle_uber_features["label"]

feat_cols_kaggle_yelp = [
    col
    for col in train_kaggle_yelp_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_kaggle_yelp = train_kaggle_yelp_features[feat_cols_kaggle_yelp]
y_train_kaggle_yelp = train_kaggle_yelp_features["label"]

feat_cols_uber_yelp = [
    col
    for col in train_uber_yelp_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_uber_yelp = train_uber_yelp_features[feat_cols_uber_yelp]
y_train_uber_yelp = train_uber_yelp_features["label"]

training_datasets = [
    (X_train_kaggle_uber, y_train_kaggle_uber),
    (X_train_kaggle_yelp, y_train_kaggle_yelp),
    (X_train_uber_yelp, y_train_uber_yelp),
]

# --------------------------------
# Select Best Model (per pair)
# --------------------------------

print("Selecting best models via GridSearchCV")

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
        "model": LogisticRegression(random_state=42, max_iter=1000, solver="saga"),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "l1_ratio": [0],
            "class_weight": ["balanced", None],
            "penalty": ["l2"],
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

ml_matcher_kaggle_uber = MLBasedMatcher(feature_extractor_kaggle_uber)
ml_matcher_kaggle_yelp = MLBasedMatcher(feature_extractor_kaggle_yelp)
ml_matcher_uber_yelp = MLBasedMatcher(feature_extractor_uber_yelp)

# Order of datasets must correspond to order in testsets:
# kaggle_small <-> uber_eats_small
ml_correspondences_kaggle_uber = ml_matcher_kaggle_uber.match(
    kaggle_small,
    uber_eats_small,
    candidates=blocker_kaggle_uber,
    id_column="id",
    trained_classifier=best_models[0],
)

# kaggle_small <-> yelp_small
ml_correspondences_kaggle_yelp = ml_matcher_kaggle_yelp.match(
    kaggle_small,
    yelp_small,
    candidates=blocker_kaggle_yelp,
    id_column="id",
    trained_classifier=best_models[1],
)

# uber_eats_small <-> yelp_small
ml_correspondences_uber_yelp = ml_matcher_uber_yelp.match(
    uber_eats_small,
    yelp_small,
    candidates=blocker_uber_yelp,
    id_column="id",
    trained_classifier=best_models[2],
)

print("Applying Maximum Bipartite Matching")

clusterer = MaximumBipartiteMatching()
ml_correspondences_kaggle_uber = clusterer.cluster(ml_correspondences_kaggle_uber)
ml_correspondences_kaggle_yelp = clusterer.cluster(ml_correspondences_kaggle_yelp)
ml_correspondences_uber_yelp = clusterer.cluster(ml_correspondences_uber_yelp)

# --------------------------------
# Save correspondences
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_kaggle_uber.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)
ml_correspondences_kaggle_yelp.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)
ml_correspondences_uber_yelp.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_kaggle_uber,
        ml_correspondences_kaggle_yelp,
        ml_correspondences_uber_yelp,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", longest_string)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
strategy.add_attribute_fuser("postal_code", longest_string)
strategy.add_attribute_fuser("country", longest_string)
strategy.add_attribute_fuser("latitude", longest_string)
strategy.add_attribute_fuser("longitude", longest_string)
strategy.add_attribute_fuser("categories", union)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
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
accuracy_score=72.09%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv
from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import MLBasedMatcher, MaximumBipartiteMatching
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
from dotenv import load_dotenv
import os
import shutil


# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"
TESTSET_DIR = "input/datasets/restaurant/testsets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

kaggle_small = load_parquet(
    DATA_DIR + "kaggle_small.parquet",
    name="kaggle_small",
)
uber_eats_small = load_parquet(
    DATA_DIR + "uber_eats_small.parquet",
    name="uber_eats_small",
)
yelp_small = load_parquet(
    DATA_DIR + "yelp_small.parquet",
    name="yelp_small",
)

# Ensure explicit id columns (already named 'id' in data)
kaggle_small["id"] = kaggle_small["id"]
uber_eats_small["id"] = uber_eats_small["id"]
yelp_small["id"] = yelp_small["id"]

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Blocking (use pre-computed optimal strategies)
# --------------------------------

print("Performing Blocking")

# kaggle_small <-> uber_eats_small: semantic_similarity on ["name_norm","city","state"], top_k=20
blocker_kaggle_uber = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=25000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# kaggle_small <-> yelp_small: exact_match_multi on ["phone_e164","postal_code"]
blocker_kaggle_yelp = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    id_column="id",
    batch_size=100000,
    output_dir="output/blocking-evaluation",
)

# uber_eats_small <-> yelp_small: semantic_similarity on ["name_norm","street","city"], top_k=20
blocker_uber_yelp = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=25000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching Configuration (comparators & feature extraction)
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()


def strip(x):
    return str(x).strip()


print("Setting up comparators and feature extractors")

# kaggle_small <-> uber_eats_small comparators
comparators_kaggle_uber = [
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="state",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="latitude",
        max_difference=0.01,
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.01,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

# kaggle_small <-> yelp_small comparators
comparators_kaggle_yelp = [
    StringComparator(
        column="phone_e164",
        similarity_function="jaro_winkler",
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="postal_code",
        similarity_function="jaro_winkler",
        preprocess=strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="latitude",
        max_difference=0.002,
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.002,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

# uber_eats_small <-> yelp_small comparators
comparators_uber_yelp = [
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="latitude",
        max_difference=0.002,
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.002,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

feature_extractor_kaggle_uber = FeatureExtractor(comparators_kaggle_uber)
feature_extractor_kaggle_yelp = FeatureExtractor(comparators_kaggle_yelp)
feature_extractor_uber_yelp = FeatureExtractor(comparators_uber_yelp)

# --------------------------------
# Load ground truth correspondences (ML training/test)
# --------------------------------

print("Loading ground truth testsets")

train_kaggle_uber = load_csv(
    TESTSET_DIR + "kaggle_uber_eats_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_uber_eats_small",
    add_index=False,
)

train_kaggle_yelp = load_csv(
    TESTSET_DIR + "kaggle_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_yelp_small",
    add_index=False,
)

train_uber_yelp = load_csv(
    TESTSET_DIR + "uber_eats_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_uber_eats_yelp_small",
    add_index=False,
)

# --------------------------------
# Feature extraction for training
# --------------------------------

print("Extracting features for training")

train_kaggle_uber_features = feature_extractor_kaggle_uber.create_features(
    kaggle_small,
    uber_eats_small,
    train_kaggle_uber[["id1", "id2"]],
    labels=train_kaggle_uber["label"],
    id_column="id",
)

train_kaggle_yelp_features = feature_extractor_kaggle_yelp.create_features(
    kaggle_small,
    yelp_small,
    train_kaggle_yelp[["id1", "id2"]],
    labels=train_kaggle_yelp["label"],
    id_column="id",
)

train_uber_yelp_features = feature_extractor_uber_yelp.create_features(
    uber_eats_small,
    yelp_small,
    train_uber_yelp[["id1", "id2"]],
    labels=train_uber_yelp["label"],
    id_column="id",
)

feat_cols_kaggle_uber = [
    col
    for col in train_kaggle_uber_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_kaggle_uber = train_kaggle_uber_features[feat_cols_kaggle_uber]
y_train_kaggle_uber = train_kaggle_uber_features["label"]

feat_cols_kaggle_yelp = [
    col
    for col in train_kaggle_yelp_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_kaggle_yelp = train_kaggle_yelp_features[feat_cols_kaggle_yelp]
y_train_kaggle_yelp = train_kaggle_yelp_features["label"]

feat_cols_uber_yelp = [
    col
    for col in train_uber_yelp_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_uber_yelp = train_uber_yelp_features[feat_cols_uber_yelp]
y_train_uber_yelp = train_uber_yelp_features["label"]

training_datasets = [
    (X_train_kaggle_uber, y_train_kaggle_uber),
    (X_train_kaggle_yelp, y_train_kaggle_yelp),
    (X_train_uber_yelp, y_train_uber_yelp),
]

# --------------------------------
# Select Best Model (per pair)
# --------------------------------

print("Selecting best models via GridSearchCV")

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
        "model": LogisticRegression(random_state=42, max_iter=1000, solver="saga"),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "l1_ratio": [0],
            "class_weight": ["balanced", None],
            "penalty": ["l2"],
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

ml_matcher_kaggle_uber = MLBasedMatcher(feature_extractor_kaggle_uber)
ml_matcher_kaggle_yelp = MLBasedMatcher(feature_extractor_kaggle_yelp)
ml_matcher_uber_yelp = MLBasedMatcher(feature_extractor_uber_yelp)

# kaggle_small <-> uber_eats_small
ml_correspondences_kaggle_uber = ml_matcher_kaggle_uber.match(
    kaggle_small,
    uber_eats_small,
    candidates=blocker_kaggle_uber,
    id_column="id",
    trained_classifier=best_models[0],
)

# kaggle_small <-> yelp_small
ml_correspondences_kaggle_yelp = ml_matcher_kaggle_yelp.match(
    kaggle_small,
    yelp_small,
    candidates=blocker_kaggle_yelp,
    id_column="id",
    trained_classifier=best_models[1],
)

# uber_eats_small <-> yelp_small
ml_correspondences_uber_yelp = ml_matcher_uber_yelp.match(
    uber_eats_small,
    yelp_small,
    candidates=blocker_uber_yelp,
    id_column="id",
    trained_classifier=best_models[2],
)

print("Applying Maximum Bipartite Matching")

clusterer = MaximumBipartiteMatching()
ml_correspondences_kaggle_uber = clusterer.cluster(ml_correspondences_kaggle_uber)
ml_correspondences_kaggle_yelp = clusterer.cluster(ml_correspondences_kaggle_yelp)
ml_correspondences_uber_yelp = clusterer.cluster(ml_correspondences_uber_yelp)

# --------------------------------
# Save correspondences (do not change filenames)
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_kaggle_uber.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)
ml_correspondences_kaggle_yelp.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)
ml_correspondences_uber_yelp.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_kaggle_uber,
        ml_correspondences_kaggle_yelp,
        ml_correspondences_uber_yelp,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", longest_string)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
strategy.add_attribute_fuser("postal_code", longest_string)
strategy.add_attribute_fuser("country", longest_string)
strategy.add_attribute_fuser("latitude", longest_string)
strategy.add_attribute_fuser("longitude", longest_string)
strategy.add_attribute_fuser("categories", union)

os.makedirs("output/data_fusion", exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=False,
)

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
accuracy_score=71.96%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv
from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import MLBasedMatcher, MaximumBipartiteMatching
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
from dotenv import load_dotenv
import os
import shutil


# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"
TESTSET_DIR = "input/datasets/restaurant/testsets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

kaggle_small = load_parquet(
    DATA_DIR + "kaggle_small.parquet",
    name="kaggle_small",
)
uber_eats_small = load_parquet(
    DATA_DIR + "uber_eats_small.parquet",
    name="uber_eats_small",
)
yelp_small = load_parquet(
    DATA_DIR + "yelp_small.parquet",
    name="yelp_small",
)

# Explicit id columns (already named 'id')
kaggle_small["id"] = kaggle_small["id"]
uber_eats_small["id"] = uber_eats_small["id"]
yelp_small["id"] = yelp_small["id"]

datasets = [kaggle_small, uber_eats_small, yelp_small]


# --------------------------------
# Blocking (use pre-computed optimal strategies)
# --------------------------------

print("Performing Blocking")

# kaggle_small <-> uber_eats_small: semantic_similarity on ["name_norm","city","state"], top_k=20
blocker_kaggle_uber = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=25000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# kaggle_small <-> yelp_small: exact_match_multi on ["phone_e164","postal_code"]
blocker_kaggle_yelp = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    id_column="id",
    batch_size=100000,
    output_dir="output/blocking-evaluation",
)

# uber_eats_small <-> yelp_small: semantic_similarity on ["name_norm","street","city"], top_k=20
blocker_uber_yelp = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=25000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching Configuration (comparators & feature extraction)
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()


def strip(x):
    return str(x).strip()


print("Setting up comparators and feature extractors")

# kaggle_small <-> uber_eats_small comparators
comparators_kaggle_uber = [
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="state",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="latitude",
        max_difference=0.01,
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.01,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

# kaggle_small <-> yelp_small comparators
comparators_kaggle_yelp = [
    StringComparator(
        column="phone_e164",
        similarity_function="jaro_winkler",
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="postal_code",
        similarity_function="jaro_winkler",
        preprocess=strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="latitude",
        max_difference=0.002,
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.002,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

# uber_eats_small <-> yelp_small comparators
comparators_uber_yelp = [
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="latitude",
        max_difference=0.002,
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.002,
    ),
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

feature_extractor_kaggle_uber = FeatureExtractor(comparators_kaggle_uber)
feature_extractor_kaggle_yelp = FeatureExtractor(comparators_kaggle_yelp)
feature_extractor_uber_yelp = FeatureExtractor(comparators_uber_yelp)


# --------------------------------
# Load ground truth correspondences (ML training/test)
# --------------------------------

print("Loading ground truth testsets")

train_kaggle_uber = load_csv(
    TESTSET_DIR + "kaggle_uber_eats_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_uber_eats_small",
    add_index=False,
)

train_kaggle_yelp = load_csv(
    TESTSET_DIR + "kaggle_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_yelp_small",
    add_index=False,
)

train_uber_yelp = load_csv(
    TESTSET_DIR + "uber_eats_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_uber_eats_yelp_small",
    add_index=False,
)


# --------------------------------
# Feature extraction for training
# --------------------------------

print("Extracting features for training")

train_kaggle_uber_features = feature_extractor_kaggle_uber.create_features(
    kaggle_small,
    uber_eats_small,
    train_kaggle_uber[["id1", "id2"]],
    labels=train_kaggle_uber["label"],
    id_column="id",
)

train_kaggle_yelp_features = feature_extractor_kaggle_yelp.create_features(
    kaggle_small,
    yelp_small,
    train_kaggle_yelp[["id1", "id2"]],
    labels=train_kaggle_yelp["label"],
    id_column="id",
)

train_uber_yelp_features = feature_extractor_uber_yelp.create_features(
    uber_eats_small,
    yelp_small,
    train_uber_yelp[["id1", "id2"]],
    labels=train_uber_yelp["label"],
    id_column="id",
)

feat_cols_kaggle_uber = [
    col
    for col in train_kaggle_uber_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_kaggle_uber = train_kaggle_uber_features[feat_cols_kaggle_uber]
y_train_kaggle_uber = train_kaggle_uber_features["label"]

feat_cols_kaggle_yelp = [
    col
    for col in train_kaggle_yelp_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_kaggle_yelp = train_kaggle_yelp_features[feat_cols_kaggle_yelp]
y_train_kaggle_yelp = train_kaggle_yelp_features["label"]

feat_cols_uber_yelp = [
    col
    for col in train_uber_yelp_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_uber_yelp = train_uber_yelp_features[feat_cols_uber_yelp]
y_train_uber_yelp = train_uber_yelp_features["label"]

training_datasets = [
    (X_train_kaggle_uber, y_train_kaggle_uber),
    (X_train_kaggle_yelp, y_train_kaggle_yelp),
    (X_train_uber_yelp, y_train_uber_yelp),
]


# --------------------------------
# Select Best Model (per pair)
# --------------------------------

print("Selecting best models via GridSearchCV")

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
        "model": LogisticRegression(random_state=42, max_iter=1000, solver="saga"),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "l1_ratio": [0],
            "class_weight": ["balanced", None],
            "penalty": ["l2"],
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

ml_matcher_kaggle_uber = MLBasedMatcher(feature_extractor_kaggle_uber)
ml_matcher_kaggle_yelp = MLBasedMatcher(feature_extractor_kaggle_yelp)
ml_matcher_uber_yelp = MLBasedMatcher(feature_extractor_uber_yelp)

# kaggle_small <-> uber_eats_small
ml_correspondences_kaggle_uber = ml_matcher_kaggle_uber.match(
    kaggle_small,
    uber_eats_small,
    candidates=blocker_kaggle_uber,
    id_column="id",
    trained_classifier=best_models[0],
)

# kaggle_small <-> yelp_small
ml_correspondences_kaggle_yelp = ml_matcher_kaggle_yelp.match(
    kaggle_small,
    yelp_small,
    candidates=blocker_kaggle_yelp,
    id_column="id",
    trained_classifier=best_models[1],
)

# uber_eats_small <-> yelp_small
ml_correspondences_uber_yelp = ml_matcher_uber_yelp.match(
    uber_eats_small,
    yelp_small,
    candidates=blocker_uber_yelp,
    id_column="id",
    trained_classifier=best_models[2],
)

print("Applying Maximum Bipartite Matching")

clusterer = MaximumBipartiteMatching()
ml_correspondences_kaggle_uber = clusterer.cluster(ml_correspondences_kaggle_uber)
ml_correspondences_kaggle_yelp = clusterer.cluster(ml_correspondences_kaggle_yelp)
ml_correspondences_uber_yelp = clusterer.cluster(ml_correspondences_uber_yelp)


# --------------------------------
# Save correspondences (file names must not be changed)
# --------------------------------

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_kaggle_uber.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)
ml_correspondences_kaggle_yelp.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)
ml_correspondences_uber_yelp.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
)


# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_kaggle_uber,
        ml_correspondences_kaggle_yelp,
        ml_correspondences_uber_yelp,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", longest_string)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
strategy.add_attribute_fuser("postal_code", longest_string)
strategy.add_attribute_fuser("country", longest_string)
strategy.add_attribute_fuser("latitude", longest_string)
strategy.add_attribute_fuser("longitude", longest_string)
strategy.add_attribute_fuser("categories", union)

os.makedirs("output/data_fusion", exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=False,
)

ml_fused.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

