# Pipeline Snapshots

notebook_name=ClusterDoc
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=58.57%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv
from PyDI.schemamatching import LLMBasedSchemaMatcher
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    MLBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.entitymatching import EmbeddingBlocker
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
)
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
import shutil


# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = ""

# Define API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
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

# Create id columns (already present as "id", but align with pipeline)
amazon_small["id"] = amazon_small["id"]
goodreads_small["id"] = goodreads_small["id"]
metabooks_small["id"] = metabooks_small["id"]

datasets = [amazon_small, goodreads_small, metabooks_small]

# --------------------------------
# Perform Schema Matching
# Schema of goodreads_small and metabooks_small will be matched to amazon_small
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# match schema of amazon_small with goodreads_small and rename goodreads_small
schema_correspondences = matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

# match schema of amazon_small with metabooks_small and rename metabooks_small
schema_correspondences = matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Blocking (use precomputed optimal blocking strategies)
# --------------------------------

print("Performing Blocking")

# All datasets use "id" as id_column
id_column_map = {
    "amazon_small": "id",
    "goodreads_small": "id",
    "metabooks_small": "id",
}

# goodreads_small <-> amazon_small
blocker_goodreads_amazon = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_map["goodreads_small"],
)

# metabooks_small <-> amazon_small
blocker_metabooks_amazon = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_map["metabooks_small"],
)

# metabooks_small <-> goodreads_small
blocker_metabooks_goodreads = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_map["metabooks_small"],
)

# --------------------------------
# Matching configuration: build comparators from precomputed strategies
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip() if pd.notnull(x) else ""


# goodreads_small <-> amazon_small comparators
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
    NumericComparator(
        column="publish_year",
        max_difference=2.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

# metabooks_small <-> amazon_small comparators
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
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=366,
    ),
]

# metabooks_small <-> goodreads_small comparators
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
    DateComparator(
        column="publish_year",
        max_days_difference=365,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="page_count",
        max_difference=30.0,
    ),
    NumericComparator(
        column="rating",
        max_difference=0.3,
    ),
    NumericComparator(
        column="numratings",
        max_difference=5000.0,
    ),
    StringComparator(
        column="genres",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

feature_extractor_goodreads_amazon = FeatureExtractor(comparators_goodreads_amazon)
feature_extractor_metabooks_amazon = FeatureExtractor(comparators_metabooks_amazon)
feature_extractor_metabooks_goodreads = FeatureExtractor(comparators_metabooks_goodreads)

# --------------------------------
# Load ground truth correspondences (training/test sets)
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
# Extract features for ML training
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

feat_cols_goodreads_amazon = [
    c
    for c in train_goodreads_amazon_features.columns
    if c not in ["id1", "id2", "label"]
]
X_train_goodreads_amazon = train_goodreads_amazon_features[feat_cols_goodreads_amazon]
y_train_goodreads_amazon = train_goodreads_amazon_features["label"]

feat_cols_metabooks_amazon = [
    c
    for c in train_metabooks_amazon_features.columns
    if c not in ["id1", "id2", "label"]
]
X_train_metabooks_amazon = train_metabooks_amazon_features[feat_cols_metabooks_amazon]
y_train_metabooks_amazon = train_metabooks_amazon_features["label"]

feat_cols_metabooks_goodreads = [
    c
    for c in train_metabooks_goodreads_features.columns
    if c not in ["id1", "id2", "label"]
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
# Select Best Model per pair using GridSearchCV
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

ml_matcher_goodreads_amazon = MLBasedMatcher(feature_extractor_goodreads_amazon)
ml_matcher_metabooks_amazon = MLBasedMatcher(feature_extractor_metabooks_amazon)
ml_matcher_metabooks_goodreads = MLBasedMatcher(feature_extractor_metabooks_goodreads)

# The order of datasets in matcher.match must correspond to order in testsets
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

print("Clustering Matched Entities")

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

# --------------------------------
# Save correspondences
# --------------------------------

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

# --------------------------------
# Data Fusion
# --------------------------------

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_goodreads_amazon,
        ml_correspondences_metabooks_amazon,
        ml_correspondences_metabooks_goodreads,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy_books")

# Use longest_string for textual attributes and union for list-like (genres)
strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("publish_year", longest_string)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("genres", union)

# Numeric-like attributes as strings in this simple example (longest_string)
strategy.add_attribute_fuser("rating", longest_string)
strategy.add_attribute_fuser("numratings", longest_string)
strategy.add_attribute_fuser("page_count", longest_string)
strategy.add_attribute_fuser("price", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
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
accuracy_score=60.00%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv
from PyDI.schemamatching import LLMBasedSchemaMatcher
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    MLBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.entitymatching import EmbeddingBlocker
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
)
from langchain_openai import ChatOpenAI

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import os
import shutil
from dotenv import load_dotenv


# --------------------------------
# Prepare Data
# --------------------------------

# Define API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
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

# Ensure id column is explicitly set (ids already exist, we just align with pipeline)
amazon_small["id"] = amazon_small["id"]
goodreads_small["id"] = goodreads_small["id"]
metabooks_small["id"] = metabooks_small["id"]

datasets = [amazon_small, goodreads_small, metabooks_small]

# --------------------------------
# Perform Schema Matching
# Schema of goodreads_small and metabooks_small will be matched to amazon_small
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# Match schema of amazon_small with goodreads_small and rename goodreads_small
schema_correspondences = matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

# Match schema of amazon_small with metabooks_small and rename metabooks_small
schema_correspondences = matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Blocking (use precomputed optimal blocking strategies)
# --------------------------------

print("Performing Blocking")

id_column_map = {
    "amazon_small": "id",
    "goodreads_small": "id",
    "metabooks_small": "id",
}

# semantic_similarity -> EmbeddingBlocker with top_k as specified

# goodreads_small <-> amazon_small
blocker_goodreads_amazon = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_map["goodreads_small"],
)

# metabooks_small <-> amazon_small
blocker_metabooks_amazon = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_map["metabooks_small"],
)

# metabooks_small <-> goodreads_small
blocker_metabooks_goodreads = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_map["metabooks_small"],
)

# --------------------------------
# Matching configuration: build comparators from precomputed strategies
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip() if pd.notnull(x) else ""


# goodreads_small <-> amazon_small comparators
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
    NumericComparator(
        column="publish_year",
        max_difference=2.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

# metabooks_small <-> amazon_small comparators
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
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=366,
    ),
]

# metabooks_small <-> goodreads_small comparators
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
    DateComparator(
        column="publish_year",
        max_days_difference=365,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="page_count",
        max_difference=30.0,
    ),
    NumericComparator(
        column="rating",
        max_difference=0.3,
    ),
    NumericComparator(
        column="numratings",
        max_difference=5000.0,
    ),
    StringComparator(
        column="genres",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

feature_extractor_goodreads_amazon = FeatureExtractor(comparators_goodreads_amazon)
feature_extractor_metabooks_amazon = FeatureExtractor(comparators_metabooks_amazon)
feature_extractor_metabooks_goodreads = FeatureExtractor(comparators_metabooks_goodreads)

# --------------------------------
# Load ground truth correspondences (training/test sets)
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
# Extract features for ML training
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

feat_cols_goodreads_amazon = [
    c
    for c in train_goodreads_amazon_features.columns
    if c not in ["id1", "id2", "label"]
]
X_train_goodreads_amazon = train_goodreads_amazon_features[feat_cols_goodreads_amazon]
y_train_goodreads_amazon = train_goodreads_amazon_features["label"]

feat_cols_metabooks_amazon = [
    c
    for c in train_metabooks_amazon_features.columns
    if c not in ["id1", "id2", "label"]
]
X_train_metabooks_amazon = train_metabooks_amazon_features[feat_cols_metabooks_amazon]
y_train_metabooks_amazon = train_metabooks_amazon_features["label"]

feat_cols_metabooks_goodreads = [
    c
    for c in train_metabooks_goodreads_features.columns
    if c not in ["id1", "id2", "label"]
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
# Select Best Model per pair using GridSearchCV
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

ml_matcher_goodreads_amazon = MLBasedMatcher(feature_extractor_goodreads_amazon)
ml_matcher_metabooks_amazon = MLBasedMatcher(feature_extractor_metabooks_amazon)
ml_matcher_metabooks_goodreads = MLBasedMatcher(feature_extractor_metabooks_goodreads)

# The order of datasets in matcher.match must correspond to order in testsets

# goodreads_small <-> amazon_small
ml_correspondences_goodreads_amazon = ml_matcher_goodreads_amazon.match(
    goodreads_small,
    amazon_small,
    candidates=blocker_goodreads_amazon,
    id_column="id",
    trained_classifier=best_models[0],
)

# metabooks_small <-> amazon_small
ml_correspondences_metabooks_amazon = ml_matcher_metabooks_amazon.match(
    metabooks_small,
    amazon_small,
    candidates=blocker_metabooks_amazon,
    id_column="id",
    trained_classifier=best_models[1],
)

# metabooks_small <-> goodreads_small
ml_correspondences_metabooks_goodreads = ml_matcher_metabooks_goodreads.match(
    metabooks_small,
    goodreads_small,
    candidates=blocker_metabooks_goodreads,
    id_column="id",
    trained_classifier=best_models[2],
)

print("Clustering Matched Entities")

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

# --------------------------------
# Save correspondences
# --------------------------------

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

# --------------------------------
# Data Fusion
# --------------------------------

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_goodreads_amazon,
        ml_correspondences_metabooks_amazon,
        ml_correspondences_metabooks_goodreads,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy_books")

# String attributes
strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("genres", union)

# Numeric attributes (fused as strings for simplicity here)
strategy.add_attribute_fuser("publish_year", longest_string)
strategy.add_attribute_fuser("rating", longest_string)
strategy.add_attribute_fuser("numratings", longest_string)
strategy.add_attribute_fuser("page_count", longest_string)
strategy.add_attribute_fuser("price", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
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
accuracy_score=55.71%
------------------------------------------------------------

```python
from PyDI.io import load_parquet, load_csv
from PyDI.schemamatching import LLMBasedSchemaMatcher
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    MLBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.entitymatching import EmbeddingBlocker
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
)
from langchain_openai import ChatOpenAI

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import os
import shutil
from dotenv import load_dotenv


# --------------------------------
# Prepare Data
# --------------------------------

# Define API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
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

# Ensure id column is explicitly set (ids already exist, we just align with pipeline)
amazon_small["id"] = amazon_small["id"]
goodreads_small["id"] = goodreads_small["id"]
metabooks_small["id"] = metabooks_small["id"]

datasets = [amazon_small, goodreads_small, metabooks_small]

# --------------------------------
# Perform Schema Matching
# Schema of goodreads_small and metabooks_small will be matched to amazon_small
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

# Match schema of amazon_small with goodreads_small and rename goodreads_small
schema_correspondences = matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

# Match schema of amazon_small with metabooks_small and rename metabooks_small
schema_correspondences = matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Blocking (use precomputed optimal blocking strategies)
# --------------------------------

print("Performing Blocking")

id_column_map = {
    "amazon_small": "id",
    "goodreads_small": "id",
    "metabooks_small": "id",
}

# semantic_similarity -> EmbeddingBlocker with top_k as specified

# goodreads_small <-> amazon_small
blocker_goodreads_amazon = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_map["goodreads_small"],
)

# metabooks_small <-> amazon_small
blocker_metabooks_amazon = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_map["metabooks_small"],
)

# metabooks_small <-> goodreads_small
blocker_metabooks_goodreads = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column=id_column_map["metabooks_small"],
)

# --------------------------------
# Matching configuration: build comparators from precomputed strategies
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip() if pd.notnull(x) else ""


# goodreads_small <-> amazon_small comparators
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
    NumericComparator(
        column="publish_year",
        max_difference=2.0,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

# metabooks_small <-> amazon_small comparators
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
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=366,
    ),
]

# metabooks_small <-> goodreads_small comparators
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
    DateComparator(
        column="publish_year",
        max_days_difference=365,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="page_count",
        max_difference=30.0,
    ),
    NumericComparator(
        column="rating",
        max_difference=0.3,
    ),
    NumericComparator(
        column="numratings",
        max_difference=5000.0,
    ),
    StringComparator(
        column="genres",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
]

feature_extractor_goodreads_amazon = FeatureExtractor(comparators_goodreads_amazon)
feature_extractor_metabooks_amazon = FeatureExtractor(comparators_metabooks_amazon)
feature_extractor_metabooks_goodreads = FeatureExtractor(comparators_metabooks_goodreads)

# --------------------------------
# Load ground truth correspondences (training/test sets)
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
# Extract features for ML training
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

feat_cols_goodreads_amazon = [
    c
    for c in train_goodreads_amazon_features.columns
    if c not in ["id1", "id2", "label"]
]
X_train_goodreads_amazon = train_goodreads_amazon_features[feat_cols_goodreads_amazon]
y_train_goodreads_amazon = train_goodreads_amazon_features["label"]

feat_cols_metabooks_amazon = [
    c
    for c in train_metabooks_amazon_features.columns
    if c not in ["id1", "id2", "label"]
]
X_train_metabooks_amazon = train_metabooks_amazon_features[feat_cols_metabooks_amazon]
y_train_metabooks_amazon = train_metabooks_amazon_features["label"]

feat_cols_metabooks_goodreads = [
    c
    for c in train_metabooks_goodreads_features.columns
    if c not in ["id1", "id2", "label"]
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
# Select Best Model per pair using GridSearchCV
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

ml_matcher_goodreads_amazon = MLBasedMatcher(feature_extractor_goodreads_amazon)
ml_matcher_metabooks_amazon = MLBasedMatcher(feature_extractor_metabooks_amazon)
ml_matcher_metabooks_goodreads = MLBasedMatcher(feature_extractor_metabooks_goodreads)

# The order of datasets in matcher.match must correspond to order in testsets

# goodreads_small <-> amazon_small
ml_correspondences_goodreads_amazon = ml_matcher_goodreads_amazon.match(
    goodreads_small,
    amazon_small,
    candidates=blocker_goodreads_amazon,
    id_column="id",
    trained_classifier=best_models[0],
)

# metabooks_small <-> amazon_small
ml_correspondences_metabooks_amazon = ml_matcher_metabooks_amazon.match(
    metabooks_small,
    amazon_small,
    candidates=blocker_metabooks_amazon,
    id_column="id",
    trained_classifier=best_models[1],
)

# metabooks_small <-> goodreads_small
ml_correspondences_metabooks_goodreads = ml_matcher_metabooks_goodreads.match(
    metabooks_small,
    goodreads_small,
    candidates=blocker_metabooks_goodreads,
    id_column="id",
    trained_classifier=best_models[2],
)

print("Clustering Matched Entities")

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

# --------------------------------
# Save correspondences
# --------------------------------

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

# --------------------------------
# Data Fusion
# --------------------------------

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_goodreads_amazon,
        ml_correspondences_metabooks_amazon,
        ml_correspondences_metabooks_goodreads,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy_books")

# String attributes
strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("genres", union)

# Numeric attributes (fused as strings for simplicity here)
strategy.add_attribute_fuser("publish_year", longest_string)
strategy.add_attribute_fuser("rating", longest_string)
strategy.add_attribute_fuser("numratings", longest_string)
strategy.add_attribute_fuser("page_count", longest_string)
strategy.add_attribute_fuser("price", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
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

