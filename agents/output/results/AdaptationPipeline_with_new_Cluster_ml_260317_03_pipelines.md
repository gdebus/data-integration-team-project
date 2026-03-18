# Pipeline Snapshots

notebook_name=AdaptationPipeline_with_new_Cluster
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=59.29%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
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

DATA_DIR = ""

amazon_small = load_parquet(
    DATA_DIR + "output/schema-matching/amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    DATA_DIR + "output/schema-matching/goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    DATA_DIR + "output/schema-matching/metabooks_small.parquet",
    name="metabooks_small",
)

# create id columns
amazon_small["id"] = amazon_small["id"]
goodreads_small["id"] = goodreads_small["id"]
metabooks_small["id"] = metabooks_small["id"]

datasets = [amazon_small, goodreads_small, metabooks_small]


# --------------------------------
# Perform Blocking
# MUST use provided precomputed blocking configuration
# --------------------------------

print("Performing Blocking")

# goodreads_small <-> amazon_small
blocker_goodreads_amazon = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# metabooks_small <-> amazon_small
blocker_metabooks_amazon = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# metabooks_small <-> goodreads_small
blocker_metabooks_goodreads = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration
# MUST use provided precomputed comparator settings
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()


comparators_goodreads_amazon = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=365,
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_metabooks_amazon = [
    StringComparator(
        column="title",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_metabooks_goodreads = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
    ),
    NumericComparator(
        column="page_count",
        max_difference=10.0,
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=366,
    ),
]

feature_extractor_goodreads_amazon = FeatureExtractor(comparators_goodreads_amazon)
feature_extractor_metabooks_amazon = FeatureExtractor(comparators_metabooks_amazon)
feature_extractor_metabooks_goodreads = FeatureExtractor(comparators_metabooks_goodreads)


# --------------------------------
# Load ground truth correspondences (ML training/test)
# use the entity matching testsets provided
# --------------------------------

train_goodreads_amazon = load_csv(
    "input/datasets/books/testsets/goodreads_2_amazon.csv",
    name="ground_truth_goodreads_amazon_train",
    add_index=False,
)

train_metabooks_amazon = load_csv(
    "input/datasets/books/testsets/metabooks_2_amazon.csv",
    name="ground_truth_metabooks_amazon_train",
    add_index=False,
)

train_metabooks_goodreads = load_csv(
    "input/datasets/books/testsets/metabooks_2_goodreads.csv",
    name="ground_truth_metabooks_goodreads_train",
    add_index=False,
)


# --------------------------------
# Extract features
# --------------------------------

train_goodreads_amazon_features = feature_extractor_goodreads_amazon.create_features(
    goodreads_small,
    amazon_small,
    train_goodreads_amazon[["id1", "id2"]],
    labels=train_goodreads_amazon["label"],
    id_column="id",
)

train_metabooks_amazon_features = feature_extractor_metabooks_amazon.create_features(
    metabooks_small,
    amazon_small,
    train_metabooks_amazon[["id1", "id2"]],
    labels=train_metabooks_amazon["label"],
    id_column="id",
)

train_metabooks_goodreads_features = (
    feature_extractor_metabooks_goodreads.create_features(
        metabooks_small,
        goodreads_small,
        train_metabooks_goodreads[["id1", "id2"]],
        labels=train_metabooks_goodreads["label"],
        id_column="id",
    )
)


# --------------------------------
# Prepare data for ML training
# --------------------------------

feat_cols_goodreads_amazon = [
    col
    for col in train_goodreads_amazon_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_goodreads_amazon = train_goodreads_amazon_features[feat_cols_goodreads_amazon]
y_train_goodreads_amazon = train_goodreads_amazon_features["label"]

feat_cols_metabooks_amazon = [
    col
    for col in train_metabooks_amazon_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_metabooks_amazon = train_metabooks_amazon_features[feat_cols_metabooks_amazon]
y_train_metabooks_amazon = train_metabooks_amazon_features["label"]

feat_cols_metabooks_goodreads = [
    col
    for col in train_metabooks_goodreads_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_metabooks_goodreads = train_metabooks_goodreads_features[
    feat_cols_metabooks_goodreads
]
y_train_metabooks_goodreads = train_metabooks_goodreads_features["label"]

training_datasets = [
    (X_train_goodreads_amazon, y_train_goodreads_amazon),
    (X_train_metabooks_amazon, y_train_metabooks_amazon),
    (X_train_metabooks_goodreads, y_train_metabooks_goodreads),
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

# --------------------------------
# the order of datasets in correspondences is important
# and must correspond to the order of columns within the testsets
# --------------------------------

ml_matcher_goodreads_amazon = MLBasedMatcher(feature_extractor_goodreads_amazon)
ml_matcher_metabooks_amazon = MLBasedMatcher(feature_extractor_metabooks_amazon)
ml_matcher_metabooks_goodreads = MLBasedMatcher(feature_extractor_metabooks_goodreads)

ml_correspondences_goodreads_amazon = ml_matcher_goodreads_amazon.match(
    goodreads_small,
    amazon_small,
    candidates=blocker_goodreads_amazon,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_metabooks_amazon = ml_matcher_metabooks_amazon.match(
    metabooks_small,
    amazon_small,
    candidates=blocker_metabooks_amazon,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_metabooks_goodreads = ml_matcher_metabooks_goodreads.match(
    metabooks_small,
    goodreads_small,
    candidates=blocker_metabooks_goodreads,
    id_column="id",
    trained_classifier=best_models[2],
)


print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_goodreads_amazon = clusterer.cluster(
    ml_correspondences_goodreads_amazon
)
ml_correspondences_metabooks_amazon = clusterer.cluster(
    ml_correspondences_metabooks_amazon
)
ml_correspondences_metabooks_goodreads = clusterer.cluster(
    ml_correspondences_metabooks_goodreads
)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_goodreads_amazon.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_amazon_small.csv",
    ),
    index=False,
)
ml_correspondences_metabooks_amazon.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_amazon_small.csv",
    ),
    index=False,
)
ml_correspondences_metabooks_goodreads.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_goodreads_small.csv",
    ),
    index=False,
)


print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_goodreads_amazon,
        ml_correspondences_metabooks_amazon,
        ml_correspondences_metabooks_goodreads,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publish_year", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("rating", longest_string)
strategy.add_attribute_fuser("numratings", longest_string)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", longest_string)
strategy.add_attribute_fuser("price", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused_standard_blocker = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
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
accuracy_score=55.71%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
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
# --------------------------------

amazon_small = load_parquet(
    "output/schema-matching/amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    "output/schema-matching/goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    "output/schema-matching/metabooks_small.parquet",
    name="metabooks_small",
)

# create id columns
amazon_small["id"] = amazon_small["id"]
goodreads_small["id"] = goodreads_small["id"]
metabooks_small["id"] = metabooks_small["id"]

datasets = [amazon_small, goodreads_small, metabooks_small]


# --------------------------------
# Perform Blocking
# MUST use provided precomputed blocking configuration
# --------------------------------

print("Performing Blocking")

blocker_goodreads_amazon = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_amazon = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_goodreads = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()


comparators_goodreads_amazon = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=365,
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_metabooks_amazon = [
    StringComparator(
        column="title",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_metabooks_goodreads = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
    ),
    NumericComparator(
        column="page_count",
        max_difference=10.0,
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=366,
    ),
]

feature_extractor_goodreads_amazon = FeatureExtractor(comparators_goodreads_amazon)
feature_extractor_metabooks_amazon = FeatureExtractor(comparators_metabooks_amazon)
feature_extractor_metabooks_goodreads = FeatureExtractor(comparators_metabooks_goodreads)


# --------------------------------
# Load ground truth correspondences (ML training/test)
# --------------------------------

train_goodreads_amazon = load_csv(
    "input/datasets/books/testsets/goodreads_2_amazon.csv",
    name="ground_truth_goodreads_amazon_train",
    add_index=False,
)

train_metabooks_amazon = load_csv(
    "input/datasets/books/testsets/metabooks_2_amazon.csv",
    name="ground_truth_metabooks_amazon_train",
    add_index=False,
)

train_metabooks_goodreads = load_csv(
    "input/datasets/books/testsets/metabooks_2_goodreads.csv",
    name="ground_truth_metabooks_goodreads_train",
    add_index=False,
)


# --------------------------------
# Extract features
# --------------------------------

train_goodreads_amazon_features = feature_extractor_goodreads_amazon.create_features(
    goodreads_small,
    amazon_small,
    train_goodreads_amazon[["id1", "id2"]],
    labels=train_goodreads_amazon["label"],
    id_column="id",
)

train_metabooks_amazon_features = feature_extractor_metabooks_amazon.create_features(
    metabooks_small,
    amazon_small,
    train_metabooks_amazon[["id1", "id2"]],
    labels=train_metabooks_amazon["label"],
    id_column="id",
)

train_metabooks_goodreads_features = feature_extractor_metabooks_goodreads.create_features(
    metabooks_small,
    goodreads_small,
    train_metabooks_goodreads[["id1", "id2"]],
    labels=train_metabooks_goodreads["label"],
    id_column="id",
)


# --------------------------------
# Prepare data for ML training
# --------------------------------

feat_cols_goodreads_amazon = [
    col for col in train_goodreads_amazon_features.columns if col not in ["id1", "id2", "label"]
]
X_train_goodreads_amazon = train_goodreads_amazon_features[feat_cols_goodreads_amazon]
y_train_goodreads_amazon = train_goodreads_amazon_features["label"]

feat_cols_metabooks_amazon = [
    col for col in train_metabooks_amazon_features.columns if col not in ["id1", "id2", "label"]
]
X_train_metabooks_amazon = train_metabooks_amazon_features[feat_cols_metabooks_amazon]
y_train_metabooks_amazon = train_metabooks_amazon_features["label"]

feat_cols_metabooks_goodreads = [
    col for col in train_metabooks_goodreads_features.columns if col not in ["id1", "id2", "label"]
]
X_train_metabooks_goodreads = train_metabooks_goodreads_features[feat_cols_metabooks_goodreads]
y_train_metabooks_goodreads = train_metabooks_goodreads_features["label"]

training_datasets = [
    (X_train_goodreads_amazon, y_train_goodreads_amazon),
    (X_train_metabooks_amazon, y_train_metabooks_amazon),
    (X_train_metabooks_goodreads, y_train_metabooks_goodreads),
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

# --------------------------------
# Order must correspond to testset filename order
# --------------------------------

ml_matcher_goodreads_amazon = MLBasedMatcher(feature_extractor_goodreads_amazon)
ml_matcher_metabooks_amazon = MLBasedMatcher(feature_extractor_metabooks_amazon)
ml_matcher_metabooks_goodreads = MLBasedMatcher(feature_extractor_metabooks_goodreads)

ml_correspondences_goodreads_amazon = ml_matcher_goodreads_amazon.match(
    goodreads_small,
    amazon_small,
    candidates=blocker_goodreads_amazon,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_metabooks_amazon = ml_matcher_metabooks_amazon.match(
    metabooks_small,
    amazon_small,
    candidates=blocker_metabooks_amazon,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_metabooks_goodreads = ml_matcher_metabooks_goodreads.match(
    metabooks_small,
    goodreads_small,
    candidates=blocker_metabooks_goodreads,
    id_column="id",
    trained_classifier=best_models[2],
)


print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_goodreads_amazon = clusterer.cluster(ml_correspondences_goodreads_amazon)
ml_correspondences_metabooks_amazon = clusterer.cluster(ml_correspondences_metabooks_amazon)
ml_correspondences_metabooks_goodreads = clusterer.cluster(ml_correspondences_metabooks_goodreads)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_goodreads_amazon.to_csv(
    os.path.join(CORR_DIR, "correspondences_goodreads_small_amazon_small.csv"),
    index=False,
)
ml_correspondences_metabooks_amazon.to_csv(
    os.path.join(CORR_DIR, "correspondences_metabooks_small_amazon_small.csv"),
    index=False,
)
ml_correspondences_metabooks_goodreads.to_csv(
    os.path.join(CORR_DIR, "correspondences_metabooks_small_goodreads_small.csv"),
    index=False,
)


print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_goodreads_amazon,
        ml_correspondences_metabooks_amazon,
        ml_correspondences_metabooks_goodreads,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publish_year", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("rating", longest_string)
strategy.add_attribute_fuser("numratings", longest_string)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", longest_string)
strategy.add_attribute_fuser("price", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused_standard_blocker = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
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
accuracy_score=53.57%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
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
# --------------------------------

amazon_small = load_parquet(
    "output/schema-matching/amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    "output/schema-matching/goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    "output/schema-matching/metabooks_small.parquet",
    name="metabooks_small",
)

# create id columns
amazon_small["id"] = amazon_small["id"]
goodreads_small["id"] = goodreads_small["id"]
metabooks_small["id"] = metabooks_small["id"]

datasets = [amazon_small, goodreads_small, metabooks_small]


# --------------------------------
# Perform Blocking
# MUST use provided precomputed blocking configuration
# --------------------------------

print("Performing Blocking")

goodreads_small["publish_year_block"] = goodreads_small["publish_year"].astype("string")
amazon_small["publish_year_block"] = amazon_small["publish_year"].astype("string")

blocker_goodreads_amazon = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year_block"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_amazon = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_goodreads = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)


# --------------------------------
# Matching configuration
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip()


comparators_goodreads_amazon = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=365,
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_metabooks_amazon = [
    StringComparator(
        column="title",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_metabooks_goodreads = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="page_count",
        max_difference=10.0,
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=366,
    ),
]

feature_extractor_goodreads_amazon = FeatureExtractor(comparators_goodreads_amazon)
feature_extractor_metabooks_amazon = FeatureExtractor(comparators_metabooks_amazon)
feature_extractor_metabooks_goodreads = FeatureExtractor(comparators_metabooks_goodreads)


# --------------------------------
# Load ground truth correspondences (ML training/test)
# --------------------------------

train_goodreads_amazon = load_csv(
    "input/datasets/books/testsets/goodreads_2_amazon.csv",
    name="ground_truth_goodreads_amazon_train",
    add_index=False,
)

train_metabooks_amazon = load_csv(
    "input/datasets/books/testsets/metabooks_2_amazon.csv",
    name="ground_truth_metabooks_amazon_train",
    add_index=False,
)

train_metabooks_goodreads = load_csv(
    "input/datasets/books/testsets/metabooks_2_goodreads.csv",
    name="ground_truth_metabooks_goodreads_train",
    add_index=False,
)


# --------------------------------
# Extract features
# --------------------------------

train_goodreads_amazon_features = feature_extractor_goodreads_amazon.create_features(
    goodreads_small,
    amazon_small,
    train_goodreads_amazon[["id1", "id2"]],
    labels=train_goodreads_amazon["label"],
    id_column="id",
)

train_metabooks_amazon_features = feature_extractor_metabooks_amazon.create_features(
    metabooks_small,
    amazon_small,
    train_metabooks_amazon[["id1", "id2"]],
    labels=train_metabooks_amazon["label"],
    id_column="id",
)

train_metabooks_goodreads_features = feature_extractor_metabooks_goodreads.create_features(
    metabooks_small,
    goodreads_small,
    train_metabooks_goodreads[["id1", "id2"]],
    labels=train_metabooks_goodreads["label"],
    id_column="id",
)


# --------------------------------
# Prepare data for ML training
# --------------------------------

feat_cols_goodreads_amazon = [
    col for col in train_goodreads_amazon_features.columns if col not in ["id1", "id2", "label"]
]
X_train_goodreads_amazon = train_goodreads_amazon_features[feat_cols_goodreads_amazon]
y_train_goodreads_amazon = train_goodreads_amazon_features["label"]

feat_cols_metabooks_amazon = [
    col for col in train_metabooks_amazon_features.columns if col not in ["id1", "id2", "label"]
]
X_train_metabooks_amazon = train_metabooks_amazon_features[feat_cols_metabooks_amazon]
y_train_metabooks_amazon = train_metabooks_amazon_features["label"]

feat_cols_metabooks_goodreads = [
    col for col in train_metabooks_goodreads_features.columns if col not in ["id1", "id2", "label"]
]
X_train_metabooks_goodreads = train_metabooks_goodreads_features[feat_cols_metabooks_goodreads]
y_train_metabooks_goodreads = train_metabooks_goodreads_features["label"]

training_datasets = [
    (X_train_goodreads_amazon, y_train_goodreads_amazon),
    (X_train_metabooks_amazon, y_train_metabooks_amazon),
    (X_train_metabooks_goodreads, y_train_metabooks_goodreads),
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

# --------------------------------
# Order must correspond to testset filename order
# --------------------------------

ml_matcher_goodreads_amazon = MLBasedMatcher(feature_extractor_goodreads_amazon)
ml_matcher_metabooks_amazon = MLBasedMatcher(feature_extractor_metabooks_amazon)
ml_matcher_metabooks_goodreads = MLBasedMatcher(feature_extractor_metabooks_goodreads)

ml_correspondences_goodreads_amazon = ml_matcher_goodreads_amazon.match(
    goodreads_small,
    amazon_small,
    candidates=blocker_goodreads_amazon,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_metabooks_amazon = ml_matcher_metabooks_amazon.match(
    metabooks_small,
    amazon_small,
    candidates=blocker_metabooks_amazon,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_metabooks_goodreads = ml_matcher_metabooks_goodreads.match(
    metabooks_small,
    goodreads_small,
    candidates=blocker_metabooks_goodreads,
    id_column="id",
    trained_classifier=best_models[2],
)


print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_goodreads_amazon = clusterer.cluster(ml_correspondences_goodreads_amazon)
ml_correspondences_metabooks_amazon = clusterer.cluster(ml_correspondences_metabooks_amazon)
ml_correspondences_metabooks_goodreads = clusterer.cluster(ml_correspondences_metabooks_goodreads)

CORR_DIR = "output/correspondences"
if os.path.exists(CORR_DIR):
    shutil.rmtree(CORR_DIR)
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_goodreads_amazon.to_csv(
    os.path.join(CORR_DIR, "correspondences_goodreads_small_amazon_small.csv"),
    index=False,
)
ml_correspondences_metabooks_amazon.to_csv(
    os.path.join(CORR_DIR, "correspondences_metabooks_small_amazon_small.csv"),
    index=False,
)
ml_correspondences_metabooks_goodreads.to_csv(
    os.path.join(CORR_DIR, "correspondences_metabooks_small_goodreads_small.csv"),
    index=False,
)


print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_goodreads_amazon,
        ml_correspondences_metabooks_amazon,
        ml_correspondences_metabooks_goodreads,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publish_year", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("rating", longest_string)
strategy.add_attribute_fuser("numratings", longest_string)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", longest_string)
strategy.add_attribute_fuser("price", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused_standard_blocker = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=False,
)

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

