# Pipeline Snapshots

notebook_name=AdaptationPipeline_blocking_matching_extension_Final_Reasoning
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=75.92%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv

from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import MLBasedMatcher, MaximumBipartiteMatching

from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import numpy as np

# --------------------------------
# Prepare Data
# --------------------------------

# Define dataset paths
DATA_DIR = ""

# Load the datasets
kaggle_small = load_parquet(
    "output/schema-matching/kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    "output/schema-matching/uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    "output/schema-matching/yelp_small.parquet",
    name="yelp_small",
)

# --------------------------------
# Set ID columns according to config (no renaming of files, only columns if needed)
# --------------------------------

# Config id_columns:
# "kaggle_small": "kaggle380k_id",
# "uber_eats_small": "kaggle380k_id",
# "yelp_small": "kaggle380k_id"

# Create the required id columns from existing "id"
kaggle_small["kaggle380k_id"] = kaggle_small["id"]
uber_eats_small["kaggle380k_id"] = uber_eats_small["id"]
yelp_small["kaggle380k_id"] = yelp_small["id"]

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Blocking (use pre-computed blocking configuration)
# --------------------------------

print("Performing Blocking")

# Blocking configuration:
# "kaggle_small_uber_eats_small": strategy "semantic_similarity", columns ["name_norm", "city", "state"], params {"top_k": 20}
# "kaggle_small_yelp_small": strategy "exact_match_multi", columns ["phone_e164", "postal_code"], params {}
# "uber_eats_small_yelp_small": strategy "semantic_similarity", columns ["name_norm", "street", "city"], params {"top_k": 20}

# Map strategy to correct blocker:
# semantic_similarity -> EmbeddingBlocker
# exact_match_multi -> StandardBlocker (on multiple columns)

# kaggle_small <-> uber_eats_small (semantic_similarity on name_norm, city, state)
blocker_kaggle_uber = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="kaggle380k_id",
)

# kaggle_small <-> yelp_small (exact_match_multi on phone_e164, postal_code)
blocker_kaggle_yelp = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="kaggle380k_id",
)

# uber_eats_small <-> yelp_small (semantic_similarity on name_norm, street, city)
blocker_uber_yelp = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="kaggle380k_id",
)

# --------------------------------
# Matching (use pre-computed matching configuration)
# --------------------------------

print("Setting up comparators and feature extractors")

# Helper preprocess functions according to mapping:
# "lower" -> str.lower
# "strip" -> str.strip
# "lower_strip" -> lambda x: str(x).lower().strip()

def lower_strip(x):
    return str(x).lower().strip()

def strip_only(x):
    return str(x).strip()

# Matching configuration:
# For each pair, build the list of comparators exactly as specified.

# -------- kaggle_small <-> uber_eats_small comparators --------
comparators_kaggle_uber = [
    # name_norm: string, jaro_winkler, lower_strip
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # city: string, jaro_winkler, lower_strip
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # state: string, jaro_winkler, lower_strip
    StringComparator(
        column="state",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # latitude: numeric, max_difference 0.01
    NumericComparator(
        column="latitude",
        max_difference=0.01,
    ),
    # longitude: numeric, max_difference 0.01
    NumericComparator(
        column="longitude",
        max_difference=0.01,
    ),
    # categories: string, jaccard, lower_strip, set_jaccard
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

# -------- kaggle_small <-> yelp_small comparators --------
comparators_kaggle_yelp = [
    # phone_e164: string, jaro_winkler
    StringComparator(
        column="phone_e164",
        similarity_function="jaro_winkler",
        list_strategy="concatenate",
    ),
    # name_norm: string, cosine, lower_strip
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # latitude: numeric, max_difference 0.001
    NumericComparator(
        column="latitude",
        max_difference=0.001,
    ),
    # longitude: numeric, max_difference 0.001
    NumericComparator(
        column="longitude",
        max_difference=0.001,
    ),
    # postal_code: string, jaro_winkler, strip
    StringComparator(
        column="postal_code",
        similarity_function="jaro_winkler",
        preprocess=strip_only,
        list_strategy="concatenate",
    ),
    # city: string, jaro_winkler, lower_strip
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

# -------- uber_eats_small <-> yelp_small comparators --------
comparators_uber_yelp = [
    # name_norm: string, jaro_winkler, lower_strip
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # street: string, jaro_winkler, lower_strip
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # city: string, jaro_winkler, lower_strip
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # postal_code: string, jaro_winkler, lower_strip
    StringComparator(
        column="postal_code",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    # latitude: numeric, max_difference 0.01
    NumericComparator(
        column="latitude",
        max_difference=0.01,
    ),
    # longitude: numeric, max_difference 0.01
    NumericComparator(
        column="longitude",
        max_difference=0.01,
    ),
    # categories: string, jaccard, lower_strip, set_jaccard
    StringComparator(
        column="categories",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

# Create feature extractors
feature_extractor_kaggle_uber = FeatureExtractor(comparators_kaggle_uber)
feature_extractor_kaggle_yelp = FeatureExtractor(comparators_kaggle_yelp)
feature_extractor_uber_yelp = FeatureExtractor(comparators_uber_yelp)

# --------------------------------
# Load ground truth correspondences for ML training
# --------------------------------

train_kaggle_uber = load_csv(
    "input/datasets/restaurant/testsets/kaggle_uber_eats_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_uber",
    add_index=False,
)

train_kaggle_yelp = load_csv(
    "input/datasets/restaurant/testsets/kaggle_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_yelp",
    add_index=False,
)

train_uber_yelp = load_csv(
    "input/datasets/restaurant/testsets/uber_eats_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_uber_yelp",
    add_index=False,
)

# --------------------------------
# Extract features for training
# Note: id_column must be "kaggle380k_id" for all datasets as per config
# --------------------------------

train_kaggle_uber_features = feature_extractor_kaggle_uber.create_features(
    kaggle_small,
    uber_eats_small,
    train_kaggle_uber[["id1", "id2"]],
    labels=train_kaggle_uber["label"],
    id_column="kaggle380k_id",
)

train_kaggle_yelp_features = feature_extractor_kaggle_yelp.create_features(
    kaggle_small,
    yelp_small,
    train_kaggle_yelp[["id1", "id2"]],
    labels=train_kaggle_yelp["label"],
    id_column="kaggle380k_id",
)

train_uber_yelp_features = feature_extractor_uber_yelp.create_features(
    uber_eats_small,
    yelp_small,
    train_uber_yelp[["id1", "id2"]],
    labels=train_uber_yelp["label"],
    id_column="kaggle380k_id",
)

# --------------------------------
# Prepare data for ML training
# --------------------------------

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
# Select Best Model via GridSearchCV for each pair
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

# --------------------------------
# ML-Based Matching
# --------------------------------

ml_matcher_kaggle_uber = MLBasedMatcher(feature_extractor_kaggle_uber)
ml_matcher_kaggle_yelp = MLBasedMatcher(feature_extractor_kaggle_yelp)
ml_matcher_uber_yelp = MLBasedMatcher(feature_extractor_uber_yelp)

# Order of datasets in match() must correspond to order in testsets filenames:
# kaggle_uber_eats_goldstandard_blocking_small.csv -> (kaggle_small, uber_eats_small)
# kaggle_yelp_goldstandard_blocking_small.csv -> (kaggle_small, yelp_small)
# uber_eats_yelp_goldstandard_blocking_small.csv -> (uber_eats_small, yelp_small)

ml_correspondences_kaggle_uber = ml_matcher_kaggle_uber.match(
    kaggle_small,
    uber_eats_small,
    candidates=blocker_kaggle_uber,
    id_column="kaggle380k_id",
    trained_classifier=best_models[0],
)

ml_correspondences_kaggle_yelp = ml_matcher_kaggle_yelp.match(
    kaggle_small,
    yelp_small,
    candidates=blocker_kaggle_yelp,
    id_column="kaggle380k_id",
    trained_classifier=best_models[1],
)

ml_correspondences_uber_yelp = ml_matcher_uber_yelp.match(
    uber_eats_small,
    yelp_small,
    candidates=blocker_uber_yelp,
    id_column="kaggle380k_id",
    trained_classifier=best_models[2],
)

# --------------------------------
# Enforce one-to-one matches via Maximum Bipartite Matching
# --------------------------------

print("Post-processing correspondences with Maximum Bipartite Matching")

clusterer = MaximumBipartiteMatching()
ml_correspondences_kaggle_uber = clusterer.cluster(ml_correspondences_kaggle_uber)
ml_correspondences_kaggle_yelp = clusterer.cluster(ml_correspondences_kaggle_yelp)
ml_correspondences_uber_yelp = clusterer.cluster(ml_correspondences_uber_yelp)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

# Merge correspondences
all_ml_correspondences = pd.concat(
    [
        ml_correspondences_kaggle_uber,
        ml_correspondences_kaggle_yelp,
        ml_correspondences_uber_yelp,
    ],
    ignore_index=True,
)

# Define data fusion strategy (on common attributes)
strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", longest_string)
strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
strategy.add_attribute_fuser("postal_code", longest_string)
strategy.add_attribute_fuser("country", longest_string)
strategy.add_attribute_fuser("latitude", longest_string)
strategy.add_attribute_fuser("longitude", longest_string)
strategy.add_attribute_fuser("categories", union)
strategy.add_attribute_fuser("phone_raw", longest_string)
strategy.add_attribute_fuser("phone_e164", longest_string)
strategy.add_attribute_fuser("map_url", longest_string)
strategy.add_attribute_fuser("website", longest_string)
strategy.add_attribute_fuser("rating", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused_standard_blocker = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
    correspondences=all_ml_correspondences,
    id_column="kaggle380k_id",
    include_singletons=False,
)

# --------------------------------
# Write output (must keep this file name)
# --------------------------------

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 01 END
============================================================

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=16
node_name=execute_pipeline
accuracy_score=pending
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv

from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import MLBasedMatcher, MaximumBipartiteMatching

from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import numpy as np
import re

# --------------------------------
# Prepare Data
# --------------------------------

# Define dataset paths
DATA_DIR = ""

# Load the datasets
kaggle_small = load_parquet(
    "output/schema-matching/kaggle_small.parquet",
    name="kaggle_small",
)

uber_eats_small = load_parquet(
    "output/schema-matching/uber_eats_small.parquet",
    name="uber_eats_small",
)

yelp_small = load_parquet(
    "output/schema-matching/yelp_small.parquet",
    name="yelp_small",
)

# --------------------------------
# ID handling
# --------------------------------
# IMPORTANT: Avoid overloading kaggle380k_id across sources for fusion.
# Keep original IDs and create a technical id_column for blocking/matching/fusion.

# Preserve source-specific IDs
kaggle_small["kaggle380k_id"] = kaggle_small["id"]
uber_eats_small["uber_eats_id"] = uber_eats_small["id"]
yelp_small["yelp_id"] = yelp_small["id"]

# Create a unified technical id column used by PyDI (no change to file names)
kaggle_small["_source_id"] = kaggle_small["id"]
uber_eats_small["_source_id"] = uber_eats_small["id"]
yelp_small["_source_id"] = yelp_small["id"]

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Blocking (use pre-computed blocking configuration)
# --------------------------------

print("Performing Blocking")

# kaggle_small <-> uber_eats_small (semantic_similarity on name_norm, city, state)
blocker_kaggle_uber = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="_source_id",
)

# kaggle_small <-> yelp_small (exact_match_multi on phone_e164, postal_code)
blocker_kaggle_yelp = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="_source_id",
)

# uber_eats_small <-> yelp_small (semantic_similarity on name_norm, street, city)
blocker_uber_yelp = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="_source_id",
)

# --------------------------------
# Matching (use pre-computed matching configuration)
# --------------------------------

print("Setting up comparators and feature extractors")

def lower_strip(x):
    return str(x).lower().strip()

def strip_only(x):
    return str(x).strip()

# -------- kaggle_small <-> uber_eats_small comparators --------
comparators_kaggle_uber = [
    StringComparator(
        column="name_norm",
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

# -------- kaggle_small <-> yelp_small comparators --------
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
    NumericComparator(
        column="latitude",
        max_difference=0.001,
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.001,
    ),
    StringComparator(
        column="postal_code",
        similarity_function="jaro_winkler",
        preprocess=strip_only,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="city",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

# -------- uber_eats_small <-> yelp_small comparators --------
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
    StringComparator(
        column="postal_code",
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

# Create feature extractors
feature_extractor_kaggle_uber = FeatureExtractor(comparators_kaggle_uber)
feature_extractor_kaggle_yelp = FeatureExtractor(comparators_kaggle_yelp)
feature_extractor_uber_yelp = FeatureExtractor(comparators_uber_yelp)

# --------------------------------
# Load ground truth correspondences for ML training
# --------------------------------

train_kaggle_uber = load_csv(
    "input/datasets/restaurant/testsets/kaggle_uber_eats_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_uber",
    add_index=False,
)

train_kaggle_yelp = load_csv(
    "input/datasets/restaurant/testsets/kaggle_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_yelp",
    add_index=False,
)

train_uber_yelp = load_csv(
    "input/datasets/restaurant/testsets/uber_eats_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_uber_yelp",
    add_index=False,
)

# --------------------------------
# Extract features for training
# Note: use the technical id_column "_source_id" consistently
# --------------------------------

train_kaggle_uber_features = feature_extractor_kaggle_uber.create_features(
    kaggle_small,
    uber_eats_small,
    train_kaggle_uber[["id1", "id2"]],
    labels=train_kaggle_uber["label"],
    id_column="_source_id",
)

train_kaggle_yelp_features = feature_extractor_kaggle_yelp.create_features(
    kaggle_small,
    yelp_small,
    train_kaggle_yelp[["id1", "id2"]],
    labels=train_kaggle_yelp["label"],
    id_column="_source_id",
)

train_uber_yelp_features = feature_extractor_uber_yelp.create_features(
    uber_eats_small,
    yelp_small,
    train_uber_yelp[["id1", "id2"]],
    labels=train_uber_yelp["label"],
    id_column="_source_id",
)

# --------------------------------
# Prepare data for ML training
# --------------------------------

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
# Select Best Model via GridSearchCV for each pair
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

# --------------------------------
# ML-Based Matching
# --------------------------------

ml_matcher_kaggle_uber = MLBasedMatcher(feature_extractor_kaggle_uber)
ml_matcher_kaggle_yelp = MLBasedMatcher(feature_extractor_kaggle_yelp)
ml_matcher_uber_yelp = MLBasedMatcher(feature_extractor_uber_yelp)

ml_correspondences_kaggle_uber = ml_matcher_kaggle_uber.match(
    kaggle_small,
    uber_eats_small,
    candidates=blocker_kaggle_uber,
    id_column="_source_id",
    trained_classifier=best_models[0],
)

ml_correspondences_kaggle_yelp = ml_matcher_kaggle_yelp.match(
    kaggle_small,
    yelp_small,
    candidates=blocker_kaggle_yelp,
    id_column="_source_id",
    trained_classifier=best_models[1],
)

ml_correspondences_uber_yelp = ml_matcher_uber_yelp.match(
    uber_eats_small,
    yelp_small,
    candidates=blocker_uber_yelp,
    id_column="_source_id",
    trained_classifier=best_models[2],
)

# --------------------------------
# Enforce one-to-one matches via Maximum Bipartite Matching
# --------------------------------

print("Post-processing correspondences with Maximum Bipartite Matching")

clusterer = MaximumBipartiteMatching()
ml_correspondences_kaggle_uber = clusterer.cluster(ml_correspondences_kaggle_uber)
ml_correspondences_kaggle_yelp = clusterer.cluster(ml_correspondences_kaggle_yelp)
ml_correspondences_uber_yelp = clusterer.cluster(ml_correspondences_uber_yelp)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

# Merge correspondences
all_ml_correspondences = pd.concat(
    [
        ml_correspondences_kaggle_uber,
        ml_correspondences_kaggle_yelp,
        ml_correspondences_uber_yelp,
    ],
    ignore_index=True,
)

# Custom fusers for IDs and phones to fix evaluation issues

def kaggle_id_fuser(values):
    """Prefer real Kaggle IDs (kaggle380k- prefix) as canonical ID."""
    cleaned = [v for v in values if pd.notna(v)]
    if not cleaned:
        return None
    as_str = [str(v) for v in cleaned]
    kaggle_like = [v for v in as_str if v.startswith("kaggle380k-")]
    if kaggle_like:
        # if multiple, arbitrarily take the first; gold uses Kaggle IDs
        return kaggle_like[0]
    # No Kaggle record in this cluster -> no Kaggle ID
    return None

def _normalize_phone_value(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v)
    # remove trailing ".0" coming from float representation
    s = re.sub(r"\.0$", "", s)
    # keep only digits and optional leading +
    s = re.sub(r"[^\d+]", "", s)
    s = s.strip()
    return s if s else None

def phone_fuser(values):
    """Normalize phone numbers and choose the most frequent normalized value."""
    normed = []
    for v in values:
        nv = _normalize_phone_value(v)
        if nv:
            normed.append(nv)
    if not normed:
        return None
    # pick most frequent normalized phone
    return pd.Series(normed).mode().iloc[0]

# Define data fusion strategy (on common attributes)
strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", longest_string)
strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
strategy.add_attribute_fuser("postal_code", longest_string)
strategy.add_attribute_fuser("country", longest_string)
strategy.add_attribute_fuser("latitude", longest_string)
strategy.add_attribute_fuser("longitude", longest_string)
strategy.add_attribute_fuser("categories", union)
strategy.add_attribute_fuser("map_url", longest_string)
strategy.add_attribute_fuser("website", longest_string)
strategy.add_attribute_fuser("rating", longest_string)

# ID and phone fusers (to improve kaggle380k_id, phone_raw, phone_e164 accuracy)
strategy.add_attribute_fuser("kaggle380k_id", kaggle_id_fuser)
strategy.add_attribute_fuser("phone_raw", phone_fuser)
strategy.add_attribute_fuser("phone_e164", phone_fuser)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused_standard_blocker = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
    correspondences=all_ml_correspondences,
    id_column="_source_id",
    include_singletons=False,
)

# --------------------------------
# Post-process fused phone columns to ensure clean string format
# --------------------------------

for col in ["phone_e164", "phone_raw"]:
    if col in ml_fused_standard_blocker.columns:
        ml_fused_standard_blocker[col] = (
            ml_fused_standard_blocker[col]
            .apply(_normalize_phone_value)
            .astype(object)
        )

# --------------------------------
# Write output (must keep this file name)
# --------------------------------

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

