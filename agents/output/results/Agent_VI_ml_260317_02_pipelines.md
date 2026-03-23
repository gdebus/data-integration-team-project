# Pipeline Snapshots

notebook_name=Agent V & VI
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=74.72%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    StandardBlocker,
    EmbeddingBlocker,
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
import os
import shutil


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

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

# create id columns
kaggle_small["id"] = kaggle_small["id"]
uber_eats_small["id"] = uber_eats_small["id"]
yelp_small["id"] = yelp_small["id"]

datasets = [kaggle_small, uber_eats_small, yelp_small]


# --------------------------------
# Perform Blocking
# Use the precomputed blocker types and parameter settings
# --------------------------------

print("Performing Blocking")

blocker_1_2 = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    id_column="id",
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

blocker_1_3 = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    id_column="id",
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

blocker_2_3 = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    id_column="id",
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)


# --------------------------------
# Matching configuration
# Use the precomputed comparator settings
# --------------------------------

lower_strip = lambda x: str(x).lower().strip()
strip_only = lambda x: str(x).strip()

comparators_1_2 = [
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

comparators_1_3 = [
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

comparators_2_3 = [
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

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)


# --------------------------------
# Load ground truth correspondences (ML training/test)
# use the entity matching testsets provided
# --------------------------------

train_1_2 = load_csv(
    "input/datasets/restaurant/testsets/kaggle_uber_eats_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_uber_eats_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/datasets/restaurant/testsets/kaggle_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_yelp_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/restaurant/testsets/uber_eats_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_uber_eats_yelp_train",
    add_index=False,
)


# --------------------------------
# Extract features
# --------------------------------

train_1_2_features = feature_extractor_1_2.create_features(
    kaggle_small,
    uber_eats_small,
    train_1_2[["id1", "id2"]],
    labels=train_1_2["label"],
    id_column="id",
)

train_1_3_features = feature_extractor_1_3.create_features(
    kaggle_small,
    yelp_small,
    train_1_3[["id1", "id2"]],
    labels=train_1_3["label"],
    id_column="id",
)

train_2_3_features = feature_extractor_2_3.create_features(
    uber_eats_small,
    yelp_small,
    train_2_3[["id1", "id2"]],
    labels=train_2_3["label"],
    id_column="id",
)


# --------------------------------
# Prepare data for ML training
# --------------------------------

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

# --------------------------------
# the order of datasets in correspondences is important and must
# correspond to the order of columns within the testsets
# --------------------------------

ml_matcher_1_2 = MLBasedMatcher(feature_extractor_1_2)
ml_matcher_1_3 = MLBasedMatcher(feature_extractor_1_3)
ml_matcher_2_3 = MLBasedMatcher(feature_extractor_2_3)

ml_correspondences_1_2 = ml_matcher_1_2.match(
    kaggle_small,
    uber_eats_small,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    kaggle_small,
    yelp_small,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    uber_eats_small,
    yelp_small,
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
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_1_2.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)
ml_correspondences_1_3.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)
ml_correspondences_2_3.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", longest_string)
strategy.add_attribute_fuser("website", longest_string)
strategy.add_attribute_fuser("map_url", longest_string)
strategy.add_attribute_fuser("phone_raw", longest_string)
strategy.add_attribute_fuser("phone_e164", longest_string)
strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", longest_string)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
strategy.add_attribute_fuser("postal_code", longest_string)
strategy.add_attribute_fuser("country", longest_string)
strategy.add_attribute_fuser("latitude", longest_string)
strategy.add_attribute_fuser("longitude", longest_string)
strategy.add_attribute_fuser("categories", union)
strategy.add_attribute_fuser("rating", longest_string)
strategy.add_attribute_fuser("rating_count", longest_string)
strategy.add_attribute_fuser("source", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused_standard_blocker = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
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
accuracy_score=57.07%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    StandardBlocker,
    EmbeddingBlocker,
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
import os
import shutil


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

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

# create id columns
kaggle_small["id"] = kaggle_small["id"]
uber_eats_small["id"] = uber_eats_small["id"]
yelp_small["id"] = yelp_small["id"]

datasets = [kaggle_small, uber_eats_small, yelp_small]


# --------------------------------
# Perform Blocking
# Use the precomputed blocker types and parameter settings
# --------------------------------

print("Performing Blocking")

blocker_1_2 = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    id_column="id",
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

blocker_1_3 = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    id_column="id",
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

blocker_2_3 = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    id_column="id",
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)


# --------------------------------
# Matching configuration
# Use the precomputed comparator settings
# --------------------------------

lower_strip = lambda x: str(x).lower().strip()
strip_only = lambda x: str(x).strip()

comparators_1_2 = [
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

comparators_1_3 = [
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

comparators_2_3 = [
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

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)


# --------------------------------
# Load ground truth correspondences (ML training/test)
# use the entity matching testsets provided
# --------------------------------

train_1_2 = load_csv(
    "input/datasets/restaurant/testsets/kaggle_uber_eats_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_uber_eats_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/datasets/restaurant/testsets/kaggle_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_yelp_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/restaurant/testsets/uber_eats_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_uber_eats_yelp_train",
    add_index=False,
)


# --------------------------------
# Extract features
# --------------------------------

train_1_2_features = feature_extractor_1_2.create_features(
    kaggle_small,
    uber_eats_small,
    train_1_2[["id1", "id2"]],
    labels=train_1_2["label"],
    id_column="id",
)

train_1_3_features = feature_extractor_1_3.create_features(
    kaggle_small,
    yelp_small,
    train_1_3[["id1", "id2"]],
    labels=train_1_3["label"],
    id_column="id",
)

train_2_3_features = feature_extractor_2_3.create_features(
    uber_eats_small,
    yelp_small,
    train_2_3[["id1", "id2"]],
    labels=train_2_3["label"],
    id_column="id",
)


# --------------------------------
# Prepare data for ML training
# --------------------------------

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

# --------------------------------
# the order of datasets in correspondences is important and must
# correspond to the order of columns within the testsets
# --------------------------------

ml_matcher_1_2 = MLBasedMatcher(feature_extractor_1_2)
ml_matcher_1_3 = MLBasedMatcher(feature_extractor_1_3)
ml_matcher_2_3 = MLBasedMatcher(feature_extractor_2_3)

ml_correspondences_1_2 = ml_matcher_1_2.match(
    kaggle_small,
    uber_eats_small,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    kaggle_small,
    yelp_small,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    uber_eats_small,
    yelp_small,
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
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_1_2.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)
ml_correspondences_1_3.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)
ml_correspondences_2_3.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", longest_string)
strategy.add_attribute_fuser("website", longest_string)
strategy.add_attribute_fuser("map_url", longest_string)
strategy.add_attribute_fuser("phone_raw", longest_string)
strategy.add_attribute_fuser("phone_e164", longest_string)
strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", longest_string)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
strategy.add_attribute_fuser("postal_code", longest_string)
strategy.add_attribute_fuser("country", longest_string)
strategy.add_attribute_fuser("latitude", longest_string)
strategy.add_attribute_fuser("longitude", longest_string)
strategy.add_attribute_fuser("categories", union)
strategy.add_attribute_fuser("rating", longest_string)
strategy.add_attribute_fuser("rating_count", longest_string)
strategy.add_attribute_fuser("source", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused_standard_blocker = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
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
accuracy_score=72.59%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    StandardBlocker,
    EmbeddingBlocker,
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
import os
import shutil


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

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

# create id columns
kaggle_small["id"] = kaggle_small["id"]
uber_eats_small["id"] = uber_eats_small["id"]
yelp_small["id"] = yelp_small["id"]

datasets = [kaggle_small, uber_eats_small, yelp_small]


# --------------------------------
# Perform Blocking
# Use the precomputed blocker types and parameter settings
# --------------------------------

print("Performing Blocking")

blocker_1_2 = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    id_column="id",
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

blocker_1_3 = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    id_column="id",
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)

blocker_2_3 = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    id_column="id",
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
)


# --------------------------------
# Matching configuration
# Use the precomputed comparator settings
# --------------------------------

lower_strip = lambda x: str(x).lower().strip()
strip_only = lambda x: str(x).strip()

comparators_1_2 = [
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

comparators_1_3 = [
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

comparators_2_3 = [
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

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)


# --------------------------------
# Load ground truth correspondences (ML training/test)
# use the entity matching testsets provided
# --------------------------------

train_1_2 = load_csv(
    "input/datasets/restaurant/testsets/kaggle_uber_eats_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_uber_eats_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/datasets/restaurant/testsets/kaggle_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_yelp_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/restaurant/testsets/uber_eats_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_uber_eats_yelp_train",
    add_index=False,
)


# --------------------------------
# Extract features
# --------------------------------

train_1_2_features = feature_extractor_1_2.create_features(
    kaggle_small,
    uber_eats_small,
    train_1_2[["id1", "id2"]],
    labels=train_1_2["label"],
    id_column="id",
)

train_1_3_features = feature_extractor_1_3.create_features(
    kaggle_small,
    yelp_small,
    train_1_3[["id1", "id2"]],
    labels=train_1_3["label"],
    id_column="id",
)

train_2_3_features = feature_extractor_2_3.create_features(
    uber_eats_small,
    yelp_small,
    train_2_3[["id1", "id2"]],
    labels=train_2_3["label"],
    id_column="id",
)


# --------------------------------
# Prepare data for ML training
# --------------------------------

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

# --------------------------------
# the order of datasets in correspondences is important and must
# correspond to the order of columns within the testsets
# --------------------------------

ml_matcher_1_2 = MLBasedMatcher(feature_extractor_1_2)
ml_matcher_1_3 = MLBasedMatcher(feature_extractor_1_3)
ml_matcher_2_3 = MLBasedMatcher(feature_extractor_2_3)

ml_correspondences_1_2 = ml_matcher_1_2.match(
    kaggle_small,
    uber_eats_small,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    kaggle_small,
    yelp_small,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    uber_eats_small,
    yelp_small,
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
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_1_2.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_uber_eats_small.csv",
    ),
    index=False,
)
ml_correspondences_1_3.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_kaggle_small_yelp_small.csv",
    ),
    index=False,
)
ml_correspondences_2_3.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_uber_eats_small_yelp_small.csv",
    ),
    index=False,
)

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", longest_string)
strategy.add_attribute_fuser("website", longest_string)
strategy.add_attribute_fuser("map_url", longest_string)
strategy.add_attribute_fuser("phone_raw", longest_string)
strategy.add_attribute_fuser("phone_e164", longest_string)
strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", longest_string)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
strategy.add_attribute_fuser("postal_code", longest_string)
strategy.add_attribute_fuser("country", longest_string)
strategy.add_attribute_fuser("latitude", longest_string)
strategy.add_attribute_fuser("longitude", longest_string)
strategy.add_attribute_fuser("categories", union)
strategy.add_attribute_fuser("rating", longest_string)
strategy.add_attribute_fuser("rating_count", longest_string)
strategy.add_attribute_fuser("source", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused_standard_blocker = engine.run(
    datasets=[kaggle_small, uber_eats_small, yelp_small],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=False,
)

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

