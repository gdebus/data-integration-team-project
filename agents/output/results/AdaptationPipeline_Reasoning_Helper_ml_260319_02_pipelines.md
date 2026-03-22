# Pipeline Snapshots

notebook_name=AdaptationPipeline_nblazek_setcount_guardline
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=7
node_name=execute_pipeline
accuracy_score=72.50%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import MLBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    most_recent,
    median,
)
from PyDI.entitymatching import MaximumBipartiteMatching

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import os
import pandas as pd
from pathlib import Path
import sys

try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _candidates = [
        Path.cwd(),
        Path.cwd() / "agents",
        Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd(),
        (Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()),
        (Path(__file__).resolve().parent.parent.parent if "__file__" in globals() else Path.cwd()),
        (Path(__file__).resolve().parent.parent.parent.parent if "__file__" in globals() else Path.cwd()),
    ]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            _path_str = str(_path.resolve())
            if _path_str not in sys.path:
                sys.path.append(_path_str)
    from list_normalization import detect_list_like_columns, normalize_list_like_columns

# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = ""

os.makedirs("output/correspondences", exist_ok=True)
os.makedirs("output/data_fusion", exist_ok=True)
os.makedirs("output/blocking-evaluation", exist_ok=True)

good_dataset_name_1 = load_csv(
    DATA_DIR + "output/normalization/attempt_1/dbpedia.csv",
    name="dbpedia",
)

good_dataset_name_2 = load_csv(
    DATA_DIR + "output/normalization/attempt_1/metacritic.csv",
    name="metacritic",
)

good_dataset_name_3 = load_csv(
    DATA_DIR + "output/normalization/attempt_1/sales.csv",
    name="sales",
)

good_dataset_name_1["id"] = good_dataset_name_1["id"]
good_dataset_name_2["id"] = good_dataset_name_2["id"]
good_dataset_name_3["id"] = good_dataset_name_3["id"]

list_like_columns = detect_list_like_columns(
    [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
    exclude_columns={"id", "_id"},
)
if list_like_columns:
    (
        good_dataset_name_1,
        good_dataset_name_2,
        good_dataset_name_3,
    ) = normalize_list_like_columns(
        [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
        list_like_columns,
    )
    print(f"Normalized list-like columns: {', '.join(list_like_columns)}")

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

def preprocess_lower(x):
    if pd.isna(x):
        return ""
    return str(x).lower()

def preprocess_lower_strip(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()

print("Performing Blocking")

blocker_1_2 = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    on=["name"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_1_3 = EmbeddingBlocker(
    good_dataset_name_2,
    good_dataset_name_1,
    text_cols=["name", "developer", "platform"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_2_3 = EmbeddingBlocker(
    good_dataset_name_2,
    good_dataset_name_3,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

comparators_1_2 = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=366,
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_1_3 = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=366,
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
]

comparators_2_3 = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=366,
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="criticScore",
        max_difference=5.0,
        list_strategy="average",
    ),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

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

def to_pair_ids(df):
    known_pairs = [
        ("id1", "id2"),
        ("id_a", "id_b"),
        ("left_id", "right_id"),
        ("source_id", "target_id"),
    ]
    for left_col, right_col in known_pairs:
        if left_col in df.columns and right_col in df.columns:
            out = df[[left_col, right_col]].copy()
            out.columns = ["id1", "id2"]
            return out

    id_like = [
        c for c in df.columns
        if c != "label" and ("id" in str(c).lower() or str(c).lower().endswith("_id"))
    ]
    if len(id_like) >= 2:
        out = df[[id_like[0], id_like[1]]].copy()
        out.columns = ["id1", "id2"]
        return out

    raise ValueError(f"Could not infer pair ID columns from: {list(df.columns)}")

def ensure_binary_label(df):
    if "label" in df.columns:
        return df["label"].astype(int)

    lowered = {str(c).lower(): c for c in df.columns}
    for candidate in ["match", "is_match", "gold", "duplicate"]:
        if candidate in lowered:
            return df[lowered[candidate]].astype(int)

    pair_cols = set(to_pair_ids(df).columns)
    non_id_cols = [c for c in df.columns if c not in pair_cols]
    if len(non_id_cols) == 1:
        return df[non_id_cols[0]].astype(int)

    raise ValueError(f"Could not infer label column from: {list(df.columns)}")

label_1_2 = ensure_binary_label(train_1_2)
label_1_3 = ensure_binary_label(train_1_3)
label_2_3 = ensure_binary_label(train_2_3)

train_1_2_features = feature_extractor_1_2.create_features(
    good_dataset_name_1,
    good_dataset_name_3,
    to_pair_ids(train_1_2),
    labels=label_1_2,
    id_column="id",
)

train_1_3_features = feature_extractor_1_3.create_features(
    good_dataset_name_2,
    good_dataset_name_1,
    to_pair_ids(train_1_3),
    labels=label_1_3,
    id_column="id",
)

train_2_3_features = feature_extractor_2_3.create_features(
    good_dataset_name_2,
    good_dataset_name_3,
    to_pair_ids(train_2_3),
    labels=label_2_3,
    id_column="id",
)

feat_cols_1_2 = [col for col in train_1_2_features.columns if col not in ["id1", "id2", "label"]]
X_train_1_2 = train_1_2_features[feat_cols_1_2]
y_train_1_2 = train_1_2_features["label"]

feat_cols_1_3 = [col for col in train_1_3_features.columns if col not in ["id1", "id2", "label"]]
X_train_1_3 = train_1_3_features[feat_cols_1_3]
y_train_1_3 = train_1_3_features["label"]

feat_cols_2_3 = [col for col in train_2_3_features.columns if col not in ["id1", "id2", "label"]]
X_train_2_3 = train_2_3_features[feat_cols_2_3]
y_train_2_3 = train_2_3_features["label"]

training_datasets = [
    (X_train_1_2, y_train_1_2),
    (X_train_1_3, y_train_1_3),
    (X_train_2_3, y_train_2_3),
]

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
    good_dataset_name_1,
    good_dataset_name_3,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
    threshold=0.78,
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    good_dataset_name_2,
    good_dataset_name_1,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[1],
    threshold=0.72,
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    good_dataset_name_2,
    good_dataset_name_3,
    candidates=blocker_2_3,
    id_column="id",
    trained_classifier=best_models[2],
    threshold=0.78,
)

ml_correspondences_1_2.to_csv(
    "output/correspondences/correspondences_dbpedia_sales.csv",
    index=False,
)
ml_correspondences_1_3.to_csv(
    "output/correspondences/correspondences_metacritic_dbpedia.csv",
    index=False,
)
ml_correspondences_2_3.to_csv(
    "output/correspondences/correspondences_metacritic_sales.csv",
    index=False,
)

clusterer = MaximumBipartiteMatching()
ml_correspondences_1_2 = clusterer.cluster(ml_correspondences_1_2)
ml_correspondences_1_3 = clusterer.cluster(ml_correspondences_1_3)
ml_correspondences_2_3 = clusterer.cluster(ml_correspondences_2_3)

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("releaseYear", most_recent)
strategy.add_attribute_fuser("developer", longest_string)
strategy.add_attribute_fuser("platform", longest_string)
strategy.add_attribute_fuser("series", longest_string)
strategy.add_attribute_fuser("criticScore", median)
strategy.add_attribute_fuser("userScore", median)
strategy.add_attribute_fuser("ESRB", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("globalSales", median)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused_standard_blocker = engine.run(
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
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
node_index=14
node_name=execute_pipeline
accuracy_score=86.25%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import MLBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    median,
    prefer_higher_trust,
)
from PyDI.entitymatching import MaximumBipartiteMatching

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import os
import pandas as pd
from pathlib import Path
import sys

try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _candidates = [
        Path.cwd(),
        Path.cwd() / "agents",
        Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd(),
        (Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()),
        (Path(__file__).resolve().parent.parent.parent if "__file__" in globals() else Path.cwd()),
        (Path(__file__).resolve().parent.parent.parent.parent if "__file__" in globals() else Path.cwd()),
    ]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            _path_str = str(_path.resolve())
            if _path_str not in sys.path:
                sys.path.append(_path_str)
    from list_normalization import detect_list_like_columns, normalize_list_like_columns

# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = ""

os.makedirs("output/correspondences", exist_ok=True)
os.makedirs("output/data_fusion", exist_ok=True)
os.makedirs("output/blocking-evaluation", exist_ok=True)

good_dataset_name_1 = load_csv(
    DATA_DIR + "output/normalization/attempt_2/dbpedia.csv",
    name="dbpedia",
)

good_dataset_name_2 = load_csv(
    DATA_DIR + "output/normalization/attempt_2/metacritic.csv",
    name="metacritic",
)

good_dataset_name_3 = load_csv(
    DATA_DIR + "output/normalization/attempt_2/sales.csv",
    name="sales",
)

good_dataset_name_1["id"] = good_dataset_name_1["id"]
good_dataset_name_2["id"] = good_dataset_name_2["id"]
good_dataset_name_3["id"] = good_dataset_name_3["id"]

def _normalize_platform_value(x):
    if pd.isna(x):
        return x
    v = str(x).strip().lower()
    mapping = {
        "pc": "pc",
        "windows": "pc",
        "microsoft windows": "pc",
        "win": "pc",
        "linux": "pc",
        "mac": "pc",
        "mac os": "pc",
        "macos": "pc",
        "os x": "pc",
        "computer": "pc",
        "xbox360": "xbox 360",
        "xbox 360": "xbox 360",
        "xb360": "xbox 360",
        "ps3": "playstation 3",
        "playstation 3": "playstation 3",
        "ps2": "playstation 2",
        "playstation 2": "playstation 2",
        "ps4": "playstation 4",
        "playstation 4": "playstation 4",
        "ps5": "playstation 5",
        "playstation 5": "playstation 5",
    }
    return mapping.get(v, v)

def _normalize_esrb_value(x):
    if pd.isna(x):
        return x
    v = str(x).strip().lower()
    mapping = {
        "e": "e",
        "everyone": "e",
        "e10+": "e10+",
        "everyone 10+": "e10+",
        "everyone 10 plus": "e10+",
        "t": "t",
        "teen": "t",
        "m": "m",
        "mature": "m",
        "mature 17+": "m",
        "ao": "ao",
        "adults only": "ao",
        "rp": "rp",
        "rating pending": "rp",
    }
    return mapping.get(v, v)

def _normalize_user_score_value(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().lower()
    if s in {"", "nan", "none", "null", "tbd", "n/a", "na"}:
        return pd.NA
    try:
        return float(s)
    except Exception:
        return pd.NA

def _normalize_critic_score_value(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().lower()
    if s in {"", "nan", "none", "null", "tbd", "n/a", "na"}:
        return pd.NA
    try:
        return float(s)
    except Exception:
        return pd.NA

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    if "platform" in df.columns:
        df["platform"] = df["platform"].apply(_normalize_platform_value)
    if "ESRB" in df.columns:
        df["ESRB"] = df["ESRB"].apply(_normalize_esrb_value)
    if "userScore" in df.columns:
        df["userScore"] = df["userScore"].apply(_normalize_user_score_value)
    if "criticScore" in df.columns:
        df["criticScore"] = df["criticScore"].apply(_normalize_critic_score_value)

list_like_columns = detect_list_like_columns(
    [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
    exclude_columns={"id", "_id"},
)
if list_like_columns:
    (
        good_dataset_name_1,
        good_dataset_name_2,
        good_dataset_name_3,
    ) = normalize_list_like_columns(
        [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
        list_like_columns,
    )
    print(f"Normalized list-like columns: {', '.join(list_like_columns)}")

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

def preprocess_lower(x):
    if pd.isna(x):
        return ""
    return str(x).lower()

def preprocess_lower_strip(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()

print("Performing Blocking")

blocker_1_2 = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    on=["name"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_1_3 = EmbeddingBlocker(
    good_dataset_name_2,
    good_dataset_name_1,
    text_cols=["name", "developer", "platform"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_2_3 = EmbeddingBlocker(
    good_dataset_name_2,
    good_dataset_name_3,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

comparators_1_2 = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=366,
    ),
    StringComparator(
        column="developer",
        similarity_function="cosine",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_1_3 = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=366,
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
]

comparators_2_3 = [
    StringComparator(
        column="name",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="platform",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="releaseYear",
        max_days_difference=366,
    ),
    StringComparator(
        column="developer",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="criticScore",
        max_difference=5.0,
        list_strategy="average",
    ),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

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

def to_pair_ids(df):
    known_pairs = [
        ("id1", "id2"),
        ("id_a", "id_b"),
        ("left_id", "right_id"),
        ("source_id", "target_id"),
    ]
    for left_col, right_col in known_pairs:
        if left_col in df.columns and right_col in df.columns:
            out = df[[left_col, right_col]].copy()
            out.columns = ["id1", "id2"]
            return out

    id_like = [
        c for c in df.columns
        if c != "label" and ("id" in str(c).lower() or str(c).lower().endswith("_id"))
    ]
    if len(id_like) >= 2:
        out = df[[id_like[0], id_like[1]]].copy()
        out.columns = ["id1", "id2"]
        return out

    raise ValueError(f"Could not infer pair ID columns from: {list(df.columns)}")

def ensure_binary_label(df):
    if "label" in df.columns:
        return df["label"].astype(int)

    lowered = {str(c).lower(): c for c in df.columns}
    for candidate in ["match", "is_match", "gold", "duplicate"]:
        if candidate in lowered:
            return df[lowered[candidate]].astype(int)

    pair_cols = set(to_pair_ids(df).columns)
    non_id_cols = [c for c in df.columns if c not in pair_cols]
    if len(non_id_cols) == 1:
        return df[non_id_cols[0]].astype(int)

    raise ValueError(f"Could not infer label column from: {list(df.columns)}")

label_1_2 = ensure_binary_label(train_1_2)
label_1_3 = ensure_binary_label(train_1_3)
label_2_3 = ensure_binary_label(train_2_3)

train_1_2_features = feature_extractor_1_2.create_features(
    good_dataset_name_1,
    good_dataset_name_3,
    to_pair_ids(train_1_2),
    labels=label_1_2,
    id_column="id",
)

train_1_3_features = feature_extractor_1_3.create_features(
    good_dataset_name_2,
    good_dataset_name_1,
    to_pair_ids(train_1_3),
    labels=label_1_3,
    id_column="id",
)

train_2_3_features = feature_extractor_2_3.create_features(
    good_dataset_name_2,
    good_dataset_name_3,
    to_pair_ids(train_2_3),
    labels=label_2_3,
    id_column="id",
)

feat_cols_1_2 = [col for col in train_1_2_features.columns if col not in ["id1", "id2", "label"]]
X_train_1_2 = train_1_2_features[feat_cols_1_2]
y_train_1_2 = train_1_2_features["label"]

feat_cols_1_3 = [col for col in train_1_3_features.columns if col not in ["id1", "id2", "label"]]
X_train_1_3 = train_1_3_features[feat_cols_1_3]
y_train_1_3 = train_1_3_features["label"]

feat_cols_2_3 = [col for col in train_2_3_features.columns if col not in ["id1", "id2", "label"]]
X_train_2_3 = train_2_3_features[feat_cols_2_3]
y_train_2_3 = train_2_3_features["label"]

training_datasets = [
    (X_train_1_2, y_train_1_2),
    (X_train_1_3, y_train_1_3),
    (X_train_2_3, y_train_2_3),
]

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
    good_dataset_name_1,
    good_dataset_name_3,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
    threshold=0.78,
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    good_dataset_name_2,
    good_dataset_name_1,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[1],
    threshold=0.72,
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    good_dataset_name_2,
    good_dataset_name_3,
    candidates=blocker_2_3,
    id_column="id",
    trained_classifier=best_models[2],
    threshold=0.78,
)

ml_correspondences_1_2.to_csv(
    "output/correspondences/correspondences_dbpedia_sales.csv",
    index=False,
)
ml_correspondences_1_3.to_csv(
    "output/correspondences/correspondences_metacritic_dbpedia.csv",
    index=False,
)
ml_correspondences_2_3.to_csv(
    "output/correspondences/correspondences_metacritic_sales.csv",
    index=False,
)

clusterer = MaximumBipartiteMatching()
ml_correspondences_1_2 = clusterer.cluster(ml_correspondences_1_2)
ml_correspondences_1_3 = clusterer.cluster(ml_correspondences_1_3)
ml_correspondences_2_3 = clusterer.cluster(ml_correspondences_2_3)

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
)

trust_map = {
    "dbpedia": 0.70,
    "metacritic": 1.00,
    "sales": 0.90,
}

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("releaseYear", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("developer", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("platform", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("series", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("criticScore", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("userScore", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("ESRB", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("publisher", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("globalSales", median)

engine = DataFusionEngine(
    strategy,
    debug=True,
    debug_format="json",
    debug_file="output/data_fusion/debug_fusion_data.jsonl",
)

ml_fused_standard_blocker = engine.run(
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=True,
)

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

