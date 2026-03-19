# Pipeline Snapshots

notebook_name=AdaptationPipeline_nblazek_setcount_guardline
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=7
node_name=execute_pipeline
accuracy_score=74.22%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import MLBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    prefer_higher_trust,
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
import numpy as np
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

DATA_DIR = ""

os.makedirs("output/correspondences", exist_ok=True)
os.makedirs("output/data_fusion", exist_ok=True)
os.makedirs("output/blocking-evaluation", exist_ok=True)

good_dataset_name_1 = load_csv(
    DATA_DIR + "output/normalization/attempt_1/kaggle_small.csv",
    name="kaggle_small",
)

good_dataset_name_2 = load_csv(
    DATA_DIR + "output/normalization/attempt_1/uber_eats_small.csv",
    name="uber_eats_small",
)

good_dataset_name_3 = load_csv(
    DATA_DIR + "output/normalization/attempt_1/yelp_small.csv",
    name="yelp_small",
)

good_dataset_name_1["id"] = good_dataset_name_1["id"]
good_dataset_name_2["id"] = good_dataset_name_2["id"]
good_dataset_name_3["id"] = good_dataset_name_3["id"]

def lower_strip(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).lower().strip()

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

print("Performing Blocking")

blocker_1_2 = EmbeddingBlocker(
    good_dataset_name_1,
    good_dataset_name_2,
    text_cols=["name_norm", "address_line1", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_1_3 = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    on=["phone_raw"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_2_3 = StandardBlocker(
    good_dataset_name_2,
    good_dataset_name_3,
    on=["postal_code", "house_number"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

comparators_1_2 = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    StringComparator(
        column="address_line1",
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
        column="postal_code",
        max_difference=1.0,
        list_strategy="average",
    ),
    NumericComparator(
        column="latitude",
        max_difference=0.01,
        list_strategy="average",
    ),
    NumericComparator(
        column="longitude",
        max_difference=0.01,
        list_strategy="average",
    ),
]

comparators_1_3 = [
    NumericComparator(
        column="phone_raw",
        max_difference=0.0,
        list_strategy="average",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="address_line1",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="house_number",
        max_difference=0.0,
        list_strategy="average",
    ),
]

comparators_2_3 = [
    StringComparator(
        column="name_norm",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    StringComparator(
        column="address_line1",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="set_jaccard",
    ),
    NumericComparator(
        column="postal_code",
        max_difference=0.0,
        list_strategy="average",
    ),
    NumericComparator(
        column="house_number",
        max_difference=0.0,
        list_strategy="average",
    ),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

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

    pair_ids = to_pair_ids(df)
    pair_cols = set(pair_ids.columns)
    non_id_cols = [c for c in df.columns if c not in pair_cols]
    if len(non_id_cols) == 1:
        return df[non_id_cols[0]].astype(int)

    raise ValueError(f"Could not infer label column from: {list(df.columns)}")

label_1_2 = ensure_binary_label(train_1_2)
label_1_3 = ensure_binary_label(train_1_3)
label_2_3 = ensure_binary_label(train_2_3)

train_1_2_features = feature_extractor_1_2.create_features(
    good_dataset_name_1,
    good_dataset_name_2,
    to_pair_ids(train_1_2),
    labels=label_1_2,
    id_column="id",
)

train_1_3_features = feature_extractor_1_3.create_features(
    good_dataset_name_1,
    good_dataset_name_3,
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
    good_dataset_name_2,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
    threshold=0.72,
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    good_dataset_name_1,
    good_dataset_name_3,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[1],
    threshold=0.78,
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
    "output/correspondences/correspondences_kaggle_small__uber_eats_small.csv",
    index=False,
)
ml_correspondences_1_3.to_csv(
    "output/correspondences/correspondences_kaggle_small__yelp_small.csv",
    index=False,
)
ml_correspondences_2_3.to_csv(
    "output/correspondences/correspondences_uber_eats_small__yelp_small.csv",
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
    "kaggle_small": 0.8,
    "uber_eats_small": 0.7,
    "yelp_small": 0.9,
}

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("source", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", longest_string)
strategy.add_attribute_fuser("website", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("map_url", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("phone_raw", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("phone_e164", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("address_line1", longest_string)
strategy.add_attribute_fuser("address_line2", longest_string)
strategy.add_attribute_fuser("street", longest_string)
strategy.add_attribute_fuser("house_number", longest_string)
strategy.add_attribute_fuser("city", longest_string)
strategy.add_attribute_fuser("state", longest_string)
strategy.add_attribute_fuser("postal_code", longest_string)
strategy.add_attribute_fuser("country", longest_string)
strategy.add_attribute_fuser("latitude", median)
strategy.add_attribute_fuser("longitude", median)
strategy.add_attribute_fuser("categories", union)
strategy.add_attribute_fuser("rating", median)
strategy.add_attribute_fuser("rating_count", median)

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
PIPELINE SNAPSHOT 01 END
============================================================

