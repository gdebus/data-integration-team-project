# Pipeline Snapshots

notebook_name=AdaptationPipeline_with_new_Cluster
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=78.99%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    EmbeddingBlocker,
    MLBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import os
import shutil


def lower_strip(x):
    return str(x).lower().strip()


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

dbpedia = load_csv(
    "output/schema-matching/dbpedia.csv",
    name="dbpedia",
)

metacritic = load_csv(
    "output/schema-matching/metacritic.csv",
    name="metacritic",
)

sales = load_csv(
    "output/schema-matching/sales.csv",
    name="sales",
)

# create id columns
dbpedia["id"] = dbpedia["id"]
metacritic["id"] = metacritic["id"]
sales["id"] = sales["id"]

datasets = [dbpedia, metacritic, sales]

print("Performing Blocking")

# dbpedia <-> sales
blocker_1_2 = EmbeddingBlocker(
    dbpedia,
    sales,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# metacritic <-> dbpedia
blocker_1_3 = EmbeddingBlocker(
    metacritic,
    dbpedia,
    text_cols=["name", "developer", "platform"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# metacritic <-> sales
blocker_2_3 = EmbeddingBlocker(
    metacritic,
    sales,
    text_cols=["name", "platform", "releaseYear"],
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

comparators_1_2 = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=730,
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
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
]

comparators_2_3 = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
    NumericComparator(
        column="criticScore",
        max_difference=5.0,
    ),
    NumericComparator(
        column="userScore",
        max_difference=1.0,
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="ESRB",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

# Load ground truth correspondences (ML training/test)
train_1_2 = load_csv(
    "input/datasets/games/testsets/dbpedia_2_sales_test.csv",
    name="ground_truth_dbpedia_sales_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/datasets/games/testsets/metacritic_2_dbpedia_test.csv",
    name="ground_truth_metacritic_dbpedia_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/games/testsets/metacritic_2_sales_test.csv",
    name="ground_truth_metacritic_sales_train",
    add_index=False,
)

# Extract features
train_1_2_features = feature_extractor_1_2.create_features(
    dbpedia,
    sales,
    train_1_2[["id1", "id2"]],
    labels=train_1_2["label"],
    id_column="id",
)

train_1_3_features = feature_extractor_1_3.create_features(
    metacritic,
    dbpedia,
    train_1_3[["id1", "id2"]],
    labels=train_1_3["label"],
    id_column="id",
)

train_2_3_features = feature_extractor_2_3.create_features(
    metacritic,
    sales,
    train_2_3[["id1", "id2"]],
    labels=train_2_3["label"],
    id_column="id",
)

# Prepare data for ML training
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

ml_matcher_1_2 = MLBasedMatcher(feature_extractor_1_2)
ml_matcher_1_3 = MLBasedMatcher(feature_extractor_1_3)
ml_matcher_2_3 = MLBasedMatcher(feature_extractor_2_3)

ml_correspondences_1_2 = ml_matcher_1_2.match(
    dbpedia,
    sales,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    metacritic,
    dbpedia,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    metacritic,
    sales,
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
        "correspondences_dbpedia_sales.csv",
    ),
    index=False,
)
ml_correspondences_1_3.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metacritic_dbpedia.csv",
    ),
    index=False,
)
ml_correspondences_2_3.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metacritic_sales.csv",
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
strategy.add_attribute_fuser("releaseYear", longest_string)
strategy.add_attribute_fuser("developer", longest_string)
strategy.add_attribute_fuser("platform", longest_string)
strategy.add_attribute_fuser("series", longest_string)
strategy.add_attribute_fuser("criticScore", longest_string)
strategy.add_attribute_fuser("userScore", longest_string)
strategy.add_attribute_fuser("ESRB", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("globalSales", longest_string)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused_standard_blocker = engine.run(
    datasets=[dbpedia, metacritic, sales],
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
accuracy_score=74.79%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    EmbeddingBlocker,
    MLBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import os
import shutil


def lower_strip(x):
    return str(x).lower().strip()


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

dbpedia = load_csv(
    DATA_DIR + "output/schema-matching/dbpedia.csv",
    name="dbpedia",
)

metacritic = load_csv(
    DATA_DIR + "output/schema-matching/metacritic.csv",
    name="metacritic",
)

sales = load_csv(
    DATA_DIR + "output/schema-matching/sales.csv",
    name="sales",
)

# create id columns
dbpedia["id"] = dbpedia["id"]
metacritic["id"] = metacritic["id"]
sales["id"] = sales["id"]

datasets = [dbpedia, metacritic, sales]

print("Performing Blocking")

# dbpedia <-> sales
blocker_1_2 = EmbeddingBlocker(
    dbpedia,
    sales,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# metacritic <-> dbpedia
blocker_1_3 = EmbeddingBlocker(
    metacritic,
    dbpedia,
    text_cols=["name", "developer", "platform"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# metacritic <-> sales
blocker_2_3 = EmbeddingBlocker(
    metacritic,
    sales,
    text_cols=["name", "platform", "releaseYear"],
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

comparators_1_2 = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=730,
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
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
]

comparators_2_3 = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
    NumericComparator(
        column="criticScore",
        max_difference=5.0,
    ),
    NumericComparator(
        column="userScore",
        max_difference=1.0,
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="ESRB",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

# Load ground truth correspondences (ML training/test)
train_1_2 = load_csv(
    "input/datasets/games/testsets/dbpedia_2_sales_test.csv",
    name="ground_truth_dbpedia_sales_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/datasets/games/testsets/metacritic_2_dbpedia_test.csv",
    name="ground_truth_metacritic_dbpedia_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/games/testsets/metacritic_2_sales_test.csv",
    name="ground_truth_metacritic_sales_train",
    add_index=False,
)

# Extract features
train_1_2_features = feature_extractor_1_2.create_features(
    dbpedia,
    sales,
    train_1_2[["id1", "id2"]],
    labels=train_1_2["label"],
    id_column="id",
)

train_1_3_features = feature_extractor_1_3.create_features(
    metacritic,
    dbpedia,
    train_1_3[["id1", "id2"]],
    labels=train_1_3["label"],
    id_column="id",
)

train_2_3_features = feature_extractor_2_3.create_features(
    metacritic,
    sales,
    train_2_3[["id1", "id2"]],
    labels=train_2_3["label"],
    id_column="id",
)

# Prepare data for ML training
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
        "model": LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs"),
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
    dbpedia,
    sales,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    metacritic,
    dbpedia,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    metacritic,
    sales,
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
    os.path.join(CORR_DIR, "correspondences_dbpedia_sales.csv"),
    index=False,
)
ml_correspondences_1_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_dbpedia.csv"),
    index=False,
)
ml_correspondences_2_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_sales.csv"),
    index=False,
)

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("releaseYear", longest_string)
strategy.add_attribute_fuser("developer", longest_string)
strategy.add_attribute_fuser("platform", longest_string)
strategy.add_attribute_fuser("series", longest_string)
strategy.add_attribute_fuser("criticScore", longest_string)
strategy.add_attribute_fuser("userScore", longest_string)
strategy.add_attribute_fuser("ESRB", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("globalSales", longest_string)

os.makedirs("output/data_fusion", exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused_standard_blocker = engine.run(
    datasets=[dbpedia, metacritic, sales],
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
accuracy_score=74.79%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    EmbeddingBlocker,
    MLBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import os
import shutil


def lower_strip(x):
    return str(x).lower().strip()


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

dbpedia = load_csv(
    "output/schema-matching/dbpedia.csv",
    name="dbpedia",
)

metacritic = load_csv(
    "output/schema-matching/metacritic.csv",
    name="metacritic",
)

sales = load_csv(
    "output/schema-matching/sales.csv",
    name="sales",
)

# create id columns
dbpedia["id"] = dbpedia["id"]
metacritic["id"] = metacritic["id"]
sales["id"] = sales["id"]

datasets = [dbpedia, metacritic, sales]

print("Performing Blocking")

# dbpedia <-> sales
blocker_1_2 = EmbeddingBlocker(
    dbpedia,
    sales,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# metacritic <-> dbpedia
blocker_1_3 = EmbeddingBlocker(
    metacritic,
    dbpedia,
    text_cols=["name", "developer", "platform"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# metacritic <-> sales
blocker_2_3 = EmbeddingBlocker(
    metacritic,
    sales,
    text_cols=["name", "platform", "releaseYear"],
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

comparators_1_2 = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=730,
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
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
]

comparators_2_3 = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=365,
    ),
    NumericComparator(
        column="criticScore",
        max_difference=5.0,
    ),
    NumericComparator(
        column="userScore",
        max_difference=1.0,
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="ESRB",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

# Load ground truth correspondences (ML training/test)
train_1_2 = load_csv(
    "input/datasets/games/testsets/dbpedia_2_sales_test.csv",
    name="ground_truth_dbpedia_sales_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/datasets/games/testsets/metacritic_2_dbpedia_test.csv",
    name="ground_truth_metacritic_dbpedia_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/games/testsets/metacritic_2_sales_test.csv",
    name="ground_truth_metacritic_sales_train",
    add_index=False,
)

# Extract features
train_1_2_features = feature_extractor_1_2.create_features(
    dbpedia,
    sales,
    train_1_2[["id1", "id2"]],
    labels=train_1_2["label"],
    id_column="id",
)

train_1_3_features = feature_extractor_1_3.create_features(
    metacritic,
    dbpedia,
    train_1_3[["id1", "id2"]],
    labels=train_1_3["label"],
    id_column="id",
)

train_2_3_features = feature_extractor_2_3.create_features(
    metacritic,
    sales,
    train_2_3[["id1", "id2"]],
    labels=train_2_3["label"],
    id_column="id",
)

# Prepare data for ML training
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
        "model": LogisticRegression(random_state=42, max_iter=1000, solver="liblinear"),
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
    dbpedia,
    sales,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    metacritic,
    dbpedia,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    metacritic,
    sales,
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
    os.path.join(CORR_DIR, "correspondences_dbpedia_sales.csv"),
    index=False,
)
ml_correspondences_1_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_dbpedia.csv"),
    index=False,
)
ml_correspondences_2_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_metacritic_sales.csv"),
    index=False,
)

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("releaseYear", longest_string)
strategy.add_attribute_fuser("developer", longest_string)
strategy.add_attribute_fuser("platform", longest_string)
strategy.add_attribute_fuser("series", longest_string)
strategy.add_attribute_fuser("criticScore", longest_string)
strategy.add_attribute_fuser("userScore", longest_string)
strategy.add_attribute_fuser("ESRB", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("globalSales", longest_string)

os.makedirs("output/data_fusion", exist_ok=True)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused_standard_blocker = engine.run(
    datasets=[dbpedia, metacritic, sales],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=False,
)

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

