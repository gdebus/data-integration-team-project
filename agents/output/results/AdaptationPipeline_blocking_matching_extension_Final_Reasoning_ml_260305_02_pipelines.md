# Pipeline Snapshots

notebook_name=AdaptationPipeline_blocking_matching_extension_Final_Reasoning
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=60.00%
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

from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"
TESTSET_DIR = "input/datasets/books/testsets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets (already schema-matched versions!)
amazon_small = load_parquet(
    DATA_DIR + "amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    DATA_DIR + "goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    DATA_DIR + "metabooks_small.parquet",
    name="metabooks_small",
)

# Set id columns according to config
amazon_small["id"] = amazon_small["id"]
goodreads_small["id"] = goodreads_small["id"]
metabooks_small["id"] = metabooks_small["id"]

datasets = [amazon_small, goodreads_small, metabooks_small]

# --------------------------------
# Blocking (use precomputed blocking configuration)
# --------------------------------

print("Performing Blocking")

# Blocking configuration (from pre-computed JSON)
blocking_config = {
    "blocking_strategies": {
        "goodreads_small_amazon_small": {
            "strategy": "semantic_similarity",
            "columns": ["title", "author", "publish_year"],
            "params": {"top_k": 20},
        },
        "metabooks_small_amazon_small": {
            "strategy": "semantic_similarity",
            "columns": ["title", "author", "publish_year"],
            "params": {"top_k": 15},
        },
        "metabooks_small_goodreads_small": {
            "strategy": "semantic_similarity",
            "columns": ["title", "author", "publisher"],
            "params": {"top_k": 15},
        },
    },
    "id_columns": {
        "amazon_small": "id",
        "goodreads_small": "id",
        "metabooks_small": "id",
    },
}

id_columns = blocking_config["id_columns"]

# Helper to create EmbeddingBlocker for semantic_similarity
def create_embedding_blocker(left_df, right_df, left_name, right_name, cfg_key):
    strat = blocking_config["blocking_strategies"][cfg_key]
    cols = strat["columns"]
    top_k = strat["params"]["top_k"]

    # Create a concatenated text column to reflect multiple blocking columns
    concat_col = "__block_text__"
    left_df[concat_col] = left_df[cols].astype(str).agg(" ".join, axis=1)
    right_df[concat_col] = right_df[cols].astype(str).agg(" ".join, axis=1)

    blocker = EmbeddingBlocker(
        left_df,
        right_df,
        text_cols=[concat_col],
        model="sentence-transformers/all-MiniLM-L6-v2",
        index_backend="sklearn",
        top_k=top_k,
        batch_size=1000,
        output_dir="output/blocking-evaluation",
        id_column="id",
    )
    return blocker

# goodreads_small <-> amazon_small
blocker_goodreads_amazon = create_embedding_blocker(
    goodreads_small,
    amazon_small,
    "goodreads_small",
    "amazon_small",
    "goodreads_small_amazon_small",
)

# metabooks_small <-> amazon_small
blocker_metabooks_amazon = create_embedding_blocker(
    metabooks_small,
    amazon_small,
    "metabooks_small",
    "amazon_small",
    "metabooks_small_amazon_small",
)

# metabooks_small <-> goodreads_small
blocker_metabooks_goodreads = create_embedding_blocker(
    metabooks_small,
    goodreads_small,
    "metabooks_small",
    "goodreads_small",
    "metabooks_small_goodreads_small",
)

# --------------------------------
# Matching configuration -> comparators and feature extractors
# --------------------------------

print("Configuring Comparators and Feature Extractors")

matching_config = {
    "id_columns": {
        "amazon_small": "id",
        "goodreads_small": "id",
        "metabooks_small": "id",
    },
    "matching_strategies": {
        "goodreads_small_amazon_small": {
            "comparators": [
                {
                    "type": "string",
                    "column": "title",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "author",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "date",
                    "column": "publish_year",
                    "max_days_difference": 365,
                },
                {
                    "type": "string",
                    "column": "publisher",
                    "similarity_function": "cosine",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
            ]
        },
        "metabooks_small_amazon_small": {
            "comparators": [
                {
                    "type": "string",
                    "column": "title",
                    "similarity_function": "cosine",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "author",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "publisher",
                    "similarity_function": "cosine",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "date",
                    "column": "publish_year",
                    "max_days_difference": 366,
                },
            ]
        },
        "metabooks_small_goodreads_small": {
            "comparators": [
                {
                    "type": "string",
                    "column": "title",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "author",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "publisher",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "genres",
                    "similarity_function": "cosine",
                    "preprocess": "lower_strip",
                    "list_strategy": "set_jaccard",
                },
                {
                    "type": "numeric",
                    "column": "page_count",
                    "max_difference": 50.0,
                },
                {
                    "type": "numeric",
                    "column": "price",
                    "max_difference": 5.0,
                },
                {
                    "type": "numeric",
                    "column": "rating",
                    "max_difference": 0.3,
                },
                {
                    "type": "numeric",
                    "column": "numratings",
                    "max_difference": 5000.0,
                },
                {
                    "type": "date",
                    "column": "publish_year",
                    "max_days_difference": 365,
                },
            ]
        },
    },
}

def get_preprocess_fn(name):
    if name is None:
        return None
    if name == "lower":
        return str.lower
    if name == "strip":
        return str.strip
    if name == "lower_strip":
        return lambda x: str(x).lower().strip()
    return None

def build_comparators(strategy):
    comps = []
    for spec in strategy["comparators"]:
        ctype = spec["type"]
        if ctype == "string":
            preprocess_fn = get_preprocess_fn(spec.get("preprocess"))
            comps.append(
                StringComparator(
                    column=spec["column"],
                    similarity_function=spec["similarity_function"],
                    preprocess=preprocess_fn,
                    list_strategy=spec.get("list_strategy"),
                )
            )
        elif ctype == "numeric":
            comps.append(
                NumericComparator(
                    column=spec["column"],
                    max_difference=spec["max_difference"],
                )
            )
        elif ctype == "date":
            comps.append(
                DateComparator(
                    column=spec["column"],
                    max_days_difference=spec["max_days_difference"],
                )
            )
    return comps

# goodreads_small <-> amazon_small
comparators_goodreads_amazon = build_comparators(
    matching_config["matching_strategies"]["goodreads_small_amazon_small"]
)
feature_extractor_goodreads_amazon = FeatureExtractor(comparators_goodreads_amazon)

# metabooks_small <-> amazon_small
comparators_metabooks_amazon = build_comparators(
    matching_config["matching_strategies"]["metabooks_small_amazon_small"]
)
feature_extractor_metabooks_amazon = FeatureExtractor(comparators_metabooks_amazon)

# metabooks_small <-> goodreads_small
comparators_metabooks_goodreads = build_comparators(
    matching_config["matching_strategies"]["metabooks_small_goodreads_small"]
)
feature_extractor_metabooks_goodreads = FeatureExtractor(comparators_metabooks_goodreads)

# --------------------------------
# Load ground truth pairs (entity matching testsets)
# --------------------------------

print("Loading Ground Truth Pairs")

train_goodreads_amazon = load_csv(
    TESTSET_DIR + "goodreads_2_amazon.csv",
    name="ground_truth_goodreads_amazon",
    add_index=False,
)

train_metabooks_amazon = load_csv(
    TESTSET_DIR + "metabooks_2_amazon.csv",
    name="ground_truth_metabooks_amazon",
    add_index=False,
)

train_metabooks_goodreads = load_csv(
    TESTSET_DIR + "metabooks_2_goodreads.csv",
    name="ground_truth_metabooks_goodreads",
    add_index=False,
)

# --------------------------------
# Feature extraction for ML training
# --------------------------------

print("Extracting Features")

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
    c for c in train_goodreads_amazon_features.columns if c not in ["id1", "id2", "label"]
]
X_train_goodreads_amazon = train_goodreads_amazon_features[feat_cols_goodreads_amazon]
y_train_goodreads_amazon = train_goodreads_amazon_features["label"]

feat_cols_metabooks_amazon = [
    c for c in train_metabooks_amazon_features.columns if c not in ["id1", "id2", "label"]
]
X_train_metabooks_amazon = train_metabooks_amazon_features[feat_cols_metabooks_amazon]
y_train_metabooks_amazon = train_metabooks_amazon_features["label"]

feat_cols_metabooks_goodreads = [
    c for c in train_metabooks_goodreads_features.columns if c not in ["id1", "id2", "label"]
]
X_train_metabooks_goodreads = train_metabooks_goodreads_features[feat_cols_metabooks_goodreads]
y_train_metabooks_goodreads = train_metabooks_goodreads_features["label"]

training_datasets = [
    (X_train_goodreads_amazon, y_train_goodreads_amazon),
    (X_train_metabooks_amazon, y_train_metabooks_amazon),
    (X_train_metabooks_goodreads, y_train_metabooks_goodreads),
]

# --------------------------------
# Select Best Model per pair (Grid Search)
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

# --------------------------------
# Matching Entities
# --------------------------------

print("Matching Entities")

ml_matcher_goodreads_amazon = MLBasedMatcher(feature_extractor_goodreads_amazon)
ml_matcher_metabooks_amazon = MLBasedMatcher(feature_extractor_metabooks_amazon)
ml_matcher_metabooks_goodreads = MLBasedMatcher(feature_extractor_metabooks_goodreads)

# Order of datasets must correspond to id1/id2 order in testsets:
# goodreads_2_amazon.csv -> (goodreads_small, amazon_small)
ml_correspondences_goodreads_amazon = ml_matcher_goodreads_amazon.match(
    goodreads_small,
    amazon_small,
    candidates=blocker_goodreads_amazon,
    id_column="id",
    trained_classifier=best_models[0],
)

# metabooks_2_amazon.csv -> (metabooks_small, amazon_small)
ml_correspondences_metabooks_amazon = ml_matcher_metabooks_amazon.match(
    metabooks_small,
    amazon_small,
    candidates=blocker_metabooks_amazon,
    id_column="id",
    trained_classifier=best_models[1],
)

# metabooks_2_goodreads.csv -> (metabooks_small, goodreads_small)
ml_correspondences_metabooks_goodreads = ml_matcher_metabooks_goodreads.match(
    metabooks_small,
    goodreads_small,
    candidates=blocker_metabooks_goodreads,
    id_column="id",
    trained_classifier=best_models[2],
)

# --------------------------------
# Clustering with Maximum Bipartite Matching
# --------------------------------

print("Clustering Correspondences")

clusterer = MaximumBipartiteMatching()
ml_correspondences_goodreads_amazon = clusterer.cluster(ml_correspondences_goodreads_amazon)
ml_correspondences_metabooks_amazon = clusterer.cluster(ml_correspondences_metabooks_amazon)
ml_correspondences_metabooks_goodreads = clusterer.cluster(ml_correspondences_metabooks_goodreads)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_goodreads_amazon,
        ml_correspondences_metabooks_amazon,
        ml_correspondences_metabooks_goodreads,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("books_ml_fusion_strategy")

# Common attributes across the three datasets after schema matching
strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("publish_year", longest_string)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", longest_string)
strategy.add_attribute_fuser("rating", longest_string)
strategy.add_attribute_fuser("numratings", longest_string)
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

# --------------------------------
# Write Final Output
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
accuracy_score=62.86%
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

from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
import re
from collections import Counter

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"
TESTSET_DIR = "input/datasets/books/testsets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets (already schema-matched versions!)
amazon_small = load_parquet(
    DATA_DIR + "amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    DATA_DIR + "goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    DATA_DIR + "metabooks_small.parquet",
    name="metabooks_small",
)

# Set id columns according to config
amazon_small["id"] = amazon_small["id"]
goodreads_small["id"] = goodreads_small["id"]
metabooks_small["id"] = metabooks_small["id"]

datasets = [amazon_small, goodreads_small, metabooks_small]

# --------------------------------
# Blocking (use precomputed blocking configuration)
# --------------------------------

print("Performing Blocking")

blocking_config = {
    "blocking_strategies": {
        "goodreads_small_amazon_small": {
            "strategy": "semantic_similarity",
            "columns": ["title", "author", "publish_year"],
            "params": {"top_k": 20},
        },
        "metabooks_small_amazon_small": {
            "strategy": "semantic_similarity",
            "columns": ["title", "author", "publish_year"],
            "params": {"top_k": 15},
        },
        "metabooks_small_goodreads_small": {
            "strategy": "semantic_similarity",
            "columns": ["title", "author", "publisher"],
            "params": {"top_k": 15},
        },
    },
    "id_columns": {
        "amazon_small": "id",
        "goodreads_small": "id",
        "metabooks_small": "id",
    },
}

id_columns = blocking_config["id_columns"]

def create_embedding_blocker(left_df, right_df, left_name, right_name, cfg_key):
    strat = blocking_config["blocking_strategies"][cfg_key]
    cols = strat["columns"]
    top_k = strat["params"]["top_k"]

    concat_col = "__block_text__"
    left_df[concat_col] = left_df[cols].astype(str).agg(" ".join, axis=1)
    right_df[concat_col] = right_df[cols].astype(str).agg(" ".join, axis=1)

    blocker = EmbeddingBlocker(
        left_df,
        right_df,
        text_cols=[concat_col],
        model="sentence-transformers/all-MiniLM-L6-v2",
        index_backend="sklearn",
        top_k=top_k,
        batch_size=1000,
        output_dir="output/blocking-evaluation",
        id_column="id",
    )
    return blocker

# goodreads_small <-> amazon_small
blocker_goodreads_amazon = create_embedding_blocker(
    goodreads_small,
    amazon_small,
    "goodreads_small",
    "amazon_small",
    "goodreads_small_amazon_small",
)

# metabooks_small <-> amazon_small
blocker_metabooks_amazon = create_embedding_blocker(
    metabooks_small,
    amazon_small,
    "metabooks_small",
    "amazon_small",
    "metabooks_small_amazon_small",
)

# metabooks_small <-> goodreads_small
blocker_metabooks_goodreads = create_embedding_blocker(
    metabooks_small,
    goodreads_small,
    "metabooks_small",
    "goodreads_small",
    "metabooks_small_goodreads_small",
)

# --------------------------------
# Matching configuration -> comparators and feature extractors
# --------------------------------

print("Configuring Comparators and Feature Extractors")

matching_config = {
    "id_columns": {
        "amazon_small": "id",
        "goodreads_small": "id",
        "metabooks_small": "id",
    },
    "matching_strategies": {
        "goodreads_small_amazon_small": {
            "comparators": [
                {
                    "type": "string",
                    "column": "title",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "author",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "date",
                    "column": "publish_year",
                    "max_days_difference": 365,
                },
                {
                    "type": "string",
                    "column": "publisher",
                    "similarity_function": "cosine",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
            ]
        },
        "metabooks_small_amazon_small": {
            "comparators": [
                {
                    "type": "string",
                    "column": "title",
                    "similarity_function": "cosine",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "author",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "publisher",
                    "similarity_function": "cosine",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "date",
                    "column": "publish_year",
                    "max_days_difference": 366,
                },
            ]
        },
        "metabooks_small_goodreads_small": {
            "comparators": [
                {
                    "type": "string",
                    "column": "title",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "author",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "publisher",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "genres",
                    "similarity_function": "cosine",
                    "preprocess": "lower_strip",
                    "list_strategy": "set_jaccard",
                },
                {
                    "type": "numeric",
                    "column": "page_count",
                    "max_difference": 50.0,
                },
                {
                    "type": "numeric",
                    "column": "price",
                    "max_difference": 5.0,
                },
                {
                    "type": "numeric",
                    "column": "rating",
                    "max_difference": 0.3,
                },
                {
                    "type": "numeric",
                    "column": "numratings",
                    "max_difference": 5000.0,
                },
                {
                    "type": "date",
                    "column": "publish_year",
                    "max_days_difference": 365,
                },
            ]
        },
    },
}

def get_preprocess_fn(name):
    if name is None:
        return None
    if name == "lower":
        return str.lower
    if name == "strip":
        return str.strip
    if name == "lower_strip":
        return lambda x: str(x).lower().strip()
    return None

def build_comparators(strategy):
    comps = []
    for spec in strategy["comparators"]:
        ctype = spec["type"]
        if ctype == "string":
            preprocess_fn = get_preprocess_fn(spec.get("preprocess"))
            comps.append(
                StringComparator(
                    column=spec["column"],
                    similarity_function=spec["similarity_function"],
                    preprocess=preprocess_fn,
                    list_strategy=spec.get("list_strategy"),
                )
            )
        elif ctype == "numeric":
            comps.append(
                NumericComparator(
                    column=spec["column"],
                    max_difference=spec["max_difference"],
                )
            )
        elif ctype == "date":
            comps.append(
                DateComparator(
                    column=spec["column"],
                    max_days_difference=spec["max_days_difference"],
                )
            )
    return comps

# goodreads_small <-> amazon_small
comparators_goodreads_amazon = build_comparators(
    matching_config["matching_strategies"]["goodreads_small_amazon_small"]
)
feature_extractor_goodreads_amazon = FeatureExtractor(comparators_goodreads_amazon)

# metabooks_small <-> amazon_small
comparators_metabooks_amazon = build_comparators(
    matching_config["matching_strategies"]["metabooks_small_amazon_small"]
)
feature_extractor_metabooks_amazon = FeatureExtractor(comparators_metabooks_amazon)

# metabooks_small <-> goodreads_small
comparators_metabooks_goodreads = build_comparators(
    matching_config["matching_strategies"]["metabooks_small_goodreads_small"]
)
feature_extractor_metabooks_goodreads = FeatureExtractor(comparators_metabooks_goodreads)

# --------------------------------
# Load ground truth pairs (entity matching testsets)
# --------------------------------

print("Loading Ground Truth Pairs")

train_goodreads_amazon = load_csv(
    TESTSET_DIR + "goodreads_2_amazon.csv",
    name="ground_truth_goodreads_amazon",
    add_index=False,
)

train_metabooks_amazon = load_csv(
    TESTSET_DIR + "metabooks_2_amazon.csv",
    name="ground_truth_metabooks_amazon",
    add_index=False,
)

train_metabooks_goodreads = load_csv(
    TESTSET_DIR + "metabooks_2_goodreads.csv",
    name="ground_truth_metabooks_goodreads",
    add_index=False,
)

# --------------------------------
# Feature extraction for ML training
# --------------------------------

print("Extracting Features")

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
    c for c in train_goodreads_amazon_features.columns if c not in ["id1", "id2", "label"]
]
X_train_goodreads_amazon = train_goodreads_amazon_features[feat_cols_goodreads_amazon]
y_train_goodreads_amazon = train_goodreads_amazon_features["label"]

feat_cols_metabooks_amazon = [
    c for c in train_metabooks_amazon_features.columns if c not in ["id1", "id2", "label"]
]
X_train_metabooks_amazon = train_metabooks_amazon_features[feat_cols_metabooks_amazon]
y_train_metabooks_amazon = train_metabooks_amazon_features["label"]

feat_cols_metabooks_goodreads = [
    c for c in train_metabooks_goodreads_features.columns if c not in ["id1", "id2", "label"]
]
X_train_metabooks_goodreads = train_metabooks_goodreads_features[feat_cols_metabooks_goodreads]
y_train_metabooks_goodreads = train_metabooks_goodreads_features["label"]

training_datasets = [
    (X_train_goodreads_amazon, y_train_goodreads_amazon),
    (X_train_metabooks_amazon, y_train_metabooks_amazon),
    (X_train_metabooks_goodreads, y_train_metabooks_goodreads),
]

# --------------------------------
# Select Best Model per pair (Grid Search)
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

# --------------------------------
# Matching Entities
# --------------------------------

print("Matching Entities")

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

# --------------------------------
# Clustering with Maximum Bipartite Matching
# --------------------------------

print("Clustering Correspondences")

clusterer = MaximumBipartiteMatching()
ml_correspondences_goodreads_amazon = clusterer.cluster(ml_correspondences_goodreads_amazon)
ml_correspondences_metabooks_amazon = clusterer.cluster(ml_correspondences_metabooks_amazon)
ml_correspondences_metabooks_goodreads = clusterer.cluster(ml_correspondences_metabooks_goodreads)

# --------------------------------
# Custom Fusers for Improved Fusion Quality
# --------------------------------

print("Defining Custom Fusers")

ROLE_PATTERN = re.compile(
    r"\s*\((goodreads author|author|editor|illustrator|designed by|foreword|translator|adaptation)\)$",
    flags=re.IGNORECASE,
)

SERIES_PATTERN = re.compile(
    r"\s*\([^)]*(series|mysteries|contemporaries|book club|trophy newbery)[^)]*\)\s*$",
    flags=re.IGNORECASE,
)

HTML_AMP_PATTERN = re.compile(r"&amp;")

def clean_author(text):
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    t = t.split(",")[0]
    t = ROLE_PATTERN.sub("", t).strip()
    return t or None

def clean_title(text):
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    t = SERIES_PATTERN.sub("", t).strip()
    return t or None

def clean_publisher(text):
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    t = HTML_AMP_PATTERN.sub("&", t)
    return t or None

def majority_clean_fuser(clean_fn):
    def fuser(values, sources=None):
        cleaned = [clean_fn(v) for v in values if v is not None]
        cleaned = [v for v in cleaned if v]
        if not cleaned:
            return None
        counts = Counter(cleaned)
        max_count = max(counts.values())
        candidates = [v for v, c in counts.items() if c == max_count]
        return min(candidates, key=len)
    return fuser

def normalize_genre_tokens(values):
    tokens = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, (list, tuple, set)):
            iterable = v
        else:
            iterable = [v]
        for elem in iterable:
            if elem is None:
                continue
            for t in str(elem).split(","):
                tt = t.strip()
                if tt:
                    tokens.add(tt)
    return list(sorted(tokens))

def genres_union(values, sources=None):
    return normalize_genre_tokens(values)

def numeric_median(values, sources=None):
    nums = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s == "":
            continue
        try:
            nums.append(float(s))
        except Exception:
            continue
    if not nums:
        return None
    return float(np.median(nums))

YEAR_PATTERN = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

def year_mode(values, sources=None):
    years = []
    for v in values:
        if v is None:
            continue
        s = str(v)
        m = YEAR_PATTERN.search(s)
        if m:
            try:
                years.append(int(m.group(1)))
            except Exception:
                continue
    if not years:
        return None
    counts = Counter(years)
    return counts.most_common(1)[0][0]

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_goodreads_amazon,
        ml_correspondences_metabooks_amazon,
        ml_correspondences_metabooks_goodreads,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("books_ml_fusion_strategy")

# Use cleaned / robust fusers instead of raw longest_string where it hurt accuracy
strategy.add_attribute_fuser("title", majority_clean_fuser(clean_title))
strategy.add_attribute_fuser("author", majority_clean_fuser(clean_author))
strategy.add_attribute_fuser("publisher", majority_clean_fuser(clean_publisher))
strategy.add_attribute_fuser("publish_year", year_mode)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", genres_union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", numeric_median)
strategy.add_attribute_fuser("rating", numeric_median)
strategy.add_attribute_fuser("numratings", numeric_median)
strategy.add_attribute_fuser("price", numeric_median)

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
    include_singletons=True,
)

# --------------------------------
# Write Final Output
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

from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
import re
from collections import Counter

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"
TESTSET_DIR = "input/datasets/books/testsets/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets (already schema-matched versions!)
amazon_small = load_parquet(
    DATA_DIR + "amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    DATA_DIR + "goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    DATA_DIR + "metabooks_small.parquet",
    name="metabooks_small",
)

# Set id columns according to config
amazon_small["id"] = amazon_small["id"]
goodreads_small["id"] = goodreads_small["id"]
metabooks_small["id"] = metabooks_small["id"]

datasets = [amazon_small, goodreads_small, metabooks_small]

# --------------------------------
# Blocking (use precomputed blocking configuration)
# --------------------------------

print("Performing Blocking")

blocking_config = {
    "blocking_strategies": {
        "goodreads_small_amazon_small": {
            "strategy": "semantic_similarity",
            "columns": ["title", "author", "publish_year"],
            "params": {"top_k": 20},
        },
        "metabooks_small_amazon_small": {
            "strategy": "semantic_similarity",
            "columns": ["title", "author", "publish_year"],
            "params": {"top_k": 15},
        },
        "metabooks_small_goodreads_small": {
            "strategy": "semantic_similarity",
            "columns": ["title", "author", "publisher"],
            "params": {"top_k": 15},
        },
    },
    "id_columns": {
        "amazon_small": "id",
        "goodreads_small": "id",
        "metabooks_small": "id",
    },
}

id_columns = blocking_config["id_columns"]

def create_embedding_blocker(left_df, right_df, left_name, right_name, cfg_key):
    strat = blocking_config["blocking_strategies"][cfg_key]
    cols = strat["columns"]
    top_k = strat["params"]["top_k"]

    concat_col = "__block_text__"
    left_df[concat_col] = left_df[cols].astype(str).agg(" ".join, axis=1)
    right_df[concat_col] = right_df[cols].astype(str).agg(" ".join, axis=1)

    blocker = EmbeddingBlocker(
        left_df,
        right_df,
        text_cols=[concat_col],
        model="sentence-transformers/all-MiniLM-L6-v2",
        index_backend="sklearn",
        top_k=top_k,
        batch_size=1000,
        output_dir="output/blocking-evaluation",
        id_column="id",
    )
    return blocker

# goodreads_small <-> amazon_small
blocker_goodreads_amazon = create_embedding_blocker(
    goodreads_small,
    amazon_small,
    "goodreads_small",
    "amazon_small",
    "goodreads_small_amazon_small",
)

# metabooks_small <-> amazon_small
blocker_metabooks_amazon = create_embedding_blocker(
    metabooks_small,
    amazon_small,
    "metabooks_small",
    "amazon_small",
    "metabooks_small_amazon_small",
)

# metabooks_small <-> goodreads_small
blocker_metabooks_goodreads = create_embedding_blocker(
    metabooks_small,
    goodreads_small,
    "metabooks_small",
    "goodreads_small",
    "metabooks_small_goodreads_small",
)

# --------------------------------
# Matching configuration -> comparators and feature extractors
# --------------------------------

print("Configuring Comparators and Feature Extractors")

matching_config = {
    "id_columns": {
        "amazon_small": "id",
        "goodreads_small": "id",
        "metabooks_small": "id",
    },
    "matching_strategies": {
        "goodreads_small_amazon_small": {
            "comparators": [
                {
                    "type": "string",
                    "column": "title",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "author",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "date",
                    "column": "publish_year",
                    "max_days_difference": 365,
                },
                {
                    "type": "string",
                    "column": "publisher",
                    "similarity_function": "cosine",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
            ]
        },
        "metabooks_small_amazon_small": {
            "comparators": [
                {
                    "type": "string",
                    "column": "title",
                    "similarity_function": "cosine",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "author",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "publisher",
                    "similarity_function": "cosine",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "date",
                    "column": "publish_year",
                    "max_days_difference": 366,
                },
            ]
        },
        "metabooks_small_goodreads_small": {
            "comparators": [
                {
                    "type": "string",
                    "column": "title",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "author",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "publisher",
                    "similarity_function": "jaro_winkler",
                    "preprocess": "lower_strip",
                    "list_strategy": "concatenate",
                },
                {
                    "type": "string",
                    "column": "genres",
                    "similarity_function": "cosine",
                    "preprocess": "lower_strip",
                    "list_strategy": "set_jaccard",
                },
                {
                    "type": "numeric",
                    "column": "page_count",
                    "max_difference": 50.0,
                },
                {
                    "type": "numeric",
                    "column": "price",
                    "max_difference": 5.0,
                },
                {
                    "type": "numeric",
                    "column": "rating",
                    "max_difference": 0.3,
                },
                {
                    "type": "numeric",
                    "column": "numratings",
                    "max_difference": 5000.0,
                },
                {
                    "type": "date",
                    "column": "publish_year",
                    "max_days_difference": 365,
                },
            ]
        },
    },
}

def get_preprocess_fn(name):
    if name is None:
        return None
    if name == "lower":
        return str.lower
    if name == "strip":
        return str.strip
    if name == "lower_strip":
        return lambda x: str(x).lower().strip()
    return None

def build_comparators(strategy):
    comps = []
    for spec in strategy["comparators"]:
        ctype = spec["type"]
        if ctype == "string":
            preprocess_fn = get_preprocess_fn(spec.get("preprocess"))
            comps.append(
                StringComparator(
                    column=spec["column"],
                    similarity_function=spec["similarity_function"],
                    preprocess=preprocess_fn,
                    list_strategy=spec.get("list_strategy"),
                )
            )
        elif ctype == "numeric":
            comps.append(
                NumericComparator(
                    column=spec["column"],
                    max_difference=spec["max_difference"],
                )
            )
        elif ctype == "date":
            comps.append(
                DateComparator(
                    column=spec["column"],
                    max_days_difference=spec["max_days_difference"],
                )
            )
    return comps

# goodreads_small <-> amazon_small
comparators_goodreads_amazon = build_comparators(
    matching_config["matching_strategies"]["goodreads_small_amazon_small"]
)
feature_extractor_goodreads_amazon = FeatureExtractor(comparators_goodreads_amazon)

# metabooks_small <-> amazon_small
comparators_metabooks_amazon = build_comparators(
    matching_config["matching_strategies"]["metabooks_small_amazon_small"]
)
feature_extractor_metabooks_amazon = FeatureExtractor(comparators_metabooks_amazon)

# metabooks_small <-> goodreads_small
comparators_metabooks_goodreads = build_comparators(
    matching_config["matching_strategies"]["metabooks_small_goodreads_small"]
)
feature_extractor_metabooks_goodreads = FeatureExtractor(comparators_metabooks_goodreads)

# --------------------------------
# Load ground truth pairs (entity matching testsets)
# --------------------------------

print("Loading Ground Truth Pairs")

train_goodreads_amazon = load_csv(
    TESTSET_DIR + "goodreads_2_amazon.csv",
    name="ground_truth_goodreads_amazon",
    add_index=False,
)

train_metabooks_amazon = load_csv(
    TESTSET_DIR + "metabooks_2_amazon.csv",
    name="ground_truth_metabooks_amazon",
    add_index=False,
)

train_metabooks_goodreads = load_csv(
    TESTSET_DIR + "metabooks_2_goodreads.csv",
    name="ground_truth_metabooks_goodreads",
    add_index=False,
)

# --------------------------------
# Feature extraction for ML training
# --------------------------------

print("Extracting Features")

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
    c for c in train_goodreads_amazon_features.columns if c not in ["id1", "id2", "label"]
]
X_train_goodreads_amazon = train_goodreads_amazon_features[feat_cols_goodreads_amazon]
y_train_goodreads_amazon = train_goodreads_amazon_features["label"]

feat_cols_metabooks_amazon = [
    c for c in train_metabooks_amazon_features.columns if c not in ["id1", "id2", "label"]
]
X_train_metabooks_amazon = train_metabooks_amazon_features[feat_cols_metabooks_amazon]
y_train_metabooks_amazon = train_metabooks_amazon_features["label"]

feat_cols_metabooks_goodreads = [
    c for c in train_metabooks_goodreads_features.columns if c not in ["id1", "id2", "label"]
]
X_train_metabooks_goodreads = train_metabooks_goodreads_features[feat_cols_metabooks_goodreads]
y_train_metabooks_goodreads = train_metabooks_goodreads_features["label"]

training_datasets = [
    (X_train_goodreads_amazon, y_train_goodreads_amazon),
    (X_train_metabooks_amazon, y_train_metabooks_amazon),
    (X_train_metabooks_goodreads, y_train_metabooks_goodreads),
]

# --------------------------------
# Select Best Model per pair (Grid Search)
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

# --------------------------------
# Matching Entities
# --------------------------------

print("Matching Entities")

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

# --------------------------------
# Clustering with Maximum Bipartite Matching
# --------------------------------

print("Clustering Correspondences")

clusterer = MaximumBipartiteMatching()
ml_correspondences_goodreads_amazon = clusterer.cluster(ml_correspondences_goodreads_amazon)
ml_correspondences_metabooks_amazon = clusterer.cluster(ml_correspondences_metabooks_amazon)
ml_correspondences_metabooks_goodreads = clusterer.cluster(ml_correspondences_metabooks_goodreads)

# --------------------------------
# Custom Fusers for Improved Fusion Quality
# --------------------------------

print("Defining Custom Fusers")

ROLE_PATTERN = re.compile(
    r"\s*\((goodreads author|author|editor|illustrator|designed by|foreword|translator|adaptation)\)$",
    flags=re.IGNORECASE,
)

SERIES_PATTERN = re.compile(
    r"\s*\([^)]*(series|mysteries|contemporaries|book club|trophy newbery)[^)]*\)\s*$",
    flags=re.IGNORECASE,
)

HTML_AMP_PATTERN = re.compile(r"&amp;")

def clean_author(text):
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    # keep all authors but strip role suffixes
    parts = [p.strip() for p in t.split(",") if p.strip()]
    clean_parts = []
    for p in parts:
        p_clean = ROLE_PATTERN.sub("", p).strip()
        if p_clean:
            clean_parts.append(p_clean)
    if not clean_parts:
        return None
    # join back authors; keeps multi-author info for better matching with gold
    return ", ".join(clean_parts)

def clean_title(text):
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    # keep more informative titles (do not drop series info completely)
    t = SERIES_PATTERN.sub("", t).strip()
    return t or None

def clean_publisher(text):
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    t = HTML_AMP_PATTERN.sub("&", t)
    return t or None

def majority_clean_fuser(clean_fn):
    def fuser(values, sources=None):
        cleaned = [clean_fn(v) for v in values if v is not None]
        cleaned = [v for v in cleaned if v]
        if not cleaned:
            return None
        counts = Counter(cleaned)
        max_count = max(counts.values())
        candidates = [v for v, c in counts.items() if c == max_count]
        # prefer longest representation to keep informative subtitles / extra authors
        return max(candidates, key=len)
    return fuser

def normalize_genre_tokens(values):
    tokens = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, (list, tuple, set)):
            iterable = v
        else:
            iterable = [v]
        for elem in iterable:
            if elem is None:
                continue
            for t in str(elem).split(","):
                tt = t.strip().lower()
                if tt:
                    tokens.add(tt)
    return list(sorted(tokens))

def genres_union(values, sources=None):
    # robust union that keeps all normalized tokens across sources
    return ", ".join(normalize_genre_tokens(values))

def numeric_median(values, sources=None):
    nums = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s == "":
            continue
        try:
            nums.append(float(s))
        except Exception:
            continue
    if not nums:
        return None
    return float(np.median(nums))

YEAR_PATTERN = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

def year_mode(values, sources=None):
    years = []
    for v in values:
        if v is None:
            continue
        s = str(v)
        m = YEAR_PATTERN.search(s)
        if m:
            try:
                years.append(int(m.group(1)))
            except Exception:
                continue
    if not years:
        return None
    counts = Counter(years)
    return counts.most_common(1)[0][0]

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

# Build a graph-based clustering over all correspondences to improve coverage
all_ml_correspondences = pd.concat(
    [
        ml_correspondences_goodreads_amazon,
        ml_correspondences_metabooks_amazon,
        ml_correspondences_metabooks_goodreads,
    ],
    ignore_index=True,
)

# Ensure we treat correspondences as an undirected graph over ids and cluster by connectivity
# This increases the chance that all matching amazon/goodreads/metabooks records participate in fusion
def build_global_clusters(corr_df):
    # corr_df is expected to have columns: id1, id2
    parent = {}

    def find(x):
        if parent.setdefault(x, x) != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for _, row in corr_df[["id1", "id2"]].dropna().iterrows():
        union(row["id1"], row["id2"])

    # map each id to its cluster representative
    clusters = {}
    for node in list(parent.keys()):
        root = find(node)
        clusters.setdefault(root, []).append(node)
    return clusters

global_clusters = build_global_clusters(all_ml_correspondences)

# Add a cluster_id column to correspondences so that DataFusionEngine can better aggregate
# We assign the same cluster_id (root id) to all edges involving nodes in that cluster
id_to_cluster = {}
for cluster_id, members in global_clusters.items():
    for m in members:
        id_to_cluster[m] = cluster_id

all_ml_correspondences["cluster_id"] = all_ml_correspondences["id1"].map(id_to_cluster)

strategy = DataFusionStrategy("books_ml_fusion_strategy")

# Use cleaned / robust fusers instead of raw longest_string where it hurt accuracy
strategy.add_attribute_fuser("title", majority_clean_fuser(clean_title))
strategy.add_attribute_fuser("author", majority_clean_fuser(clean_author))
strategy.add_attribute_fuser("publisher", majority_clean_fuser(clean_publisher))
strategy.add_attribute_fuser("publish_year", year_mode)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", genres_union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", numeric_median)
strategy.add_attribute_fuser("rating", numeric_median)
strategy.add_attribute_fuser("numratings", numeric_median)
strategy.add_attribute_fuser("price", numeric_median)

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
    include_singletons=True,
)

# --------------------------------
# Write Final Output
# --------------------------------

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

