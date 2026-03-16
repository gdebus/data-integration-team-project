# Pipeline Snapshots

notebook_name=AdaptationPipeline_blocking_matching_extension_Final_Reasoning
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=50.19%
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


def lower_strip(x):
    return str(x).lower().strip()


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

# Define dataset paths
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

# create id columns using provided id column configuration
kaggle_small["id"] = kaggle_small["kaggle380k_id"] if "kaggle380k_id" in kaggle_small.columns else kaggle_small["id"]
uber_eats_small["id"] = uber_eats_small["kaggle380k_id"] if "kaggle380k_id" in uber_eats_small.columns else uber_eats_small["id"]
yelp_small["id"] = yelp_small["kaggle380k_id"] if "kaggle380k_id" in yelp_small.columns else yelp_small["id"]

datasets = [kaggle_small, uber_eats_small, yelp_small]

# --------------------------------
# Perform Blocking
# Must use the precomputed blocker types and parameter settings
# --------------------------------

print("Performing Blocking")

blocker_1_2 = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_1_3 = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_2_3 = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# --------------------------------
# Matching configuration
# Must use the supplied comparator settings
# --------------------------------

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
        preprocess=str.strip,
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

# Load ground truth correspondences (ML training/test)
train_1_2 = load_csv(
    "input/datasets/restaurant/testsets/kaggle_uber_eats_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_small_uber_eats_small_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/datasets/restaurant/testsets/kaggle_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_small_yelp_small_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/restaurant/testsets/uber_eats_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_uber_eats_small_yelp_small_train",
    add_index=False,
)

# Extract features
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
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_1_2.to_csv(
    os.path.join(CORR_DIR, "correspondences_kaggle_small_uber_eats_small.csv"),
    index=False,
)
ml_correspondences_1_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_kaggle_small_yelp_small.csv"),
    index=False,
)
ml_correspondences_2_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_uber_eats_small_yelp_small.csv"),
    index=False,
)

# -------------- Data Fusion ------------------

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
node_index=16
node_name=execute_pipeline
accuracy_score=77.60%
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

from PyDI.fusion import DataFusionStrategy, DataFusionEngine

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import numpy as np
import os
import ast
import json


def lower_strip(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


def strip_only(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_phone(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    if s.startswith("+"):
        digits = "".join(ch for ch in s[1:] if ch.isdigit())
        return f"+{digits}" if digits else None
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits if digits else None


def normalize_text(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    return s


def normalize_url(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    return s


def normalize_postal_code(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s if s else None


def normalize_house_number(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s if s else None


def normalize_categories(x):
    if pd.isna(x):
        return None
    if isinstance(x, list):
        vals = [str(v).strip().lower() for v in x if str(v).strip()]
        return json.dumps(sorted(set(vals))) if vals else None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            vals = [str(v).strip().lower() for v in parsed if str(v).strip()]
            return json.dumps(sorted(set(vals))) if vals else None
    except Exception:
        pass
    return json.dumps(sorted(set([p.strip() for p in s.lower().split(",") if p.strip()])))


def normalize_float(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def _extract_raw_values(values):
    cleaned = []
    for item in values:
        if isinstance(item, tuple):
            if len(item) >= 1:
                value = item[0]
            else:
                continue
        else:
            value = item
        if pd.isna(value):
            continue
        if isinstance(value, str):
            s = value.strip()
            if not s or s.lower() in {"nan", "none", ""}:
                continue
            cleaned.append(s)
        else:
            cleaned.append(value)
    return cleaned


def most_frequent_non_null(values):
    cleaned = _extract_raw_values(values)
    if not cleaned:
        return None

    counts = {}
    first_seen = {}
    for i, v in enumerate(cleaned):
        key = json.dumps(v, sort_keys=True, default=str)
        counts[key] = counts.get(key, 0) + 1
        if key not in first_seen:
            first_seen[key] = (i, v)

    best_key = sorted(counts.keys(), key=lambda k: (-counts[k], first_seen[k][0]))[0]
    return first_seen[best_key][1]


def shortest_non_null_string(values):
    cleaned = []
    for v in _extract_raw_values(values):
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", ""}:
            cleaned.append(s)
    if not cleaned:
        return None
    return min(cleaned, key=len)


def median_numeric(values):
    cleaned = []
    for v in _extract_raw_values(values):
        try:
            cleaned.append(float(v))
        except Exception:
            continue
    if not cleaned:
        return None
    return float(pd.Series(cleaned).median())


def union_resolver(values):
    merged = []
    seen = set()

    for item in values:
        if isinstance(item, tuple):
            if len(item) >= 1:
                value = item[0]
            else:
                continue
        else:
            value = item

        if pd.isna(value):
            continue

        if isinstance(value, list):
            vals = value
        else:
            vals = None
            if isinstance(value, str):
                s = value.strip()
                if not s or s.lower() in {"nan", "none", ""}:
                    continue
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, list):
                        vals = parsed
                except Exception:
                    vals = None
            if vals is None:
                vals = [value]

        for v in vals:
            sv = str(v).strip()
            if not sv or sv.lower() in {"nan", "none", ""}:
                continue
            if sv not in seen:
                seen.add(sv)
                merged.append(sv)

    return json.dumps(merged) if merged else None


# --------------------------------
# Prepare Data
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

kaggle_small["id"] = (
    kaggle_small["kaggle380k_id"]
    if "kaggle380k_id" in kaggle_small.columns
    else kaggle_small["id"]
)
uber_eats_small["id"] = (
    uber_eats_small["kaggle380k_id"]
    if "kaggle380k_id" in uber_eats_small.columns
    else uber_eats_small["id"]
)
yelp_small["id"] = (
    yelp_small["kaggle380k_id"]
    if "kaggle380k_id" in yelp_small.columns
    else yelp_small["id"]
)

datasets = [kaggle_small, uber_eats_small, yelp_small]

for df in datasets:
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    if "phone_e164" in df.columns:
        df["phone_e164"] = df["phone_e164"].apply(normalize_phone)
    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].apply(normalize_phone)
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].apply(normalize_postal_code)
    if "house_number" in df.columns:
        df["house_number"] = df["house_number"].apply(normalize_house_number)
    if "website" in df.columns:
        df["website"] = df["website"].apply(normalize_url)
    if "map_url" in df.columns:
        df["map_url"] = df["map_url"].apply(normalize_url)
    if "name" in df.columns:
        df["name"] = df["name"].apply(normalize_text)
    if "name_norm" in df.columns:
        df["name_norm"] = df["name_norm"].apply(lower_strip)
    if "address_line1" in df.columns:
        df["address_line1"] = df["address_line1"].apply(normalize_text)
    if "address_line2" in df.columns:
        df["address_line2"] = df["address_line2"].apply(normalize_text)
    if "street" in df.columns:
        df["street"] = df["street"].apply(normalize_text)
    if "city" in df.columns:
        df["city"] = df["city"].apply(normalize_text)
    if "state" in df.columns:
        df["state"] = df["state"].apply(normalize_text)
    if "country" in df.columns:
        df["country"] = df["country"].apply(normalize_text)
    if "categories" in df.columns:
        df["categories"] = df["categories"].apply(normalize_categories)
    if "latitude" in df.columns:
        df["latitude"] = df["latitude"].apply(normalize_float)
    if "longitude" in df.columns:
        df["longitude"] = df["longitude"].apply(normalize_float)
    if "rating" in df.columns:
        df["rating"] = df["rating"].apply(normalize_float)
    if "rating_count" in df.columns:
        df["rating_count"] = df["rating_count"].apply(normalize_float)

# --------------------------------
# Perform Blocking
# --------------------------------

print("Performing Blocking")

blocker_1_2 = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_1_3 = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_2_3 = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
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
    NumericComparator(column="latitude", max_difference=0.01),
    NumericComparator(column="longitude", max_difference=0.01),
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
    NumericComparator(column="latitude", max_difference=0.001),
    NumericComparator(column="longitude", max_difference=0.001),
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
    NumericComparator(column="latitude", max_difference=0.01),
    NumericComparator(column="longitude", max_difference=0.01),
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

train_1_2 = load_csv(
    "input/datasets/restaurant/testsets/kaggle_uber_eats_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_small_uber_eats_small_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/datasets/restaurant/testsets/kaggle_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_small_yelp_small_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/restaurant/testsets/uber_eats_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_uber_eats_small_yelp_small_train",
    add_index=False,
)

train_1_2["id1"] = train_1_2["id1"].astype(str)
train_1_2["id2"] = train_1_2["id2"].astype(str)
train_1_3["id1"] = train_1_3["id1"].astype(str)
train_1_3["id2"] = train_1_3["id2"].astype(str)
train_2_3["id1"] = train_2_3["id1"].astype(str)
train_2_3["id2"] = train_2_3["id2"].astype(str)

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
            "n_estimators": [100, 200],
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
            "learning_rate": [0.05, 0.1, 0.2],
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

clusterer = MaximumBipartiteMatching()
ml_correspondences_1_2 = clusterer.cluster(ml_correspondences_1_2)
ml_correspondences_1_3 = clusterer.cluster(ml_correspondences_1_3)
ml_correspondences_2_3 = clusterer.cluster(ml_correspondences_2_3)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_1_2.to_csv(
    os.path.join(CORR_DIR, "correspondences_kaggle_small_uber_eats_small.csv"),
    index=False,
)
ml_correspondences_1_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_kaggle_small_yelp_small.csv"),
    index=False,
)
ml_correspondences_2_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_uber_eats_small_yelp_small.csv"),
    index=False,
)

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", most_frequent_non_null)
strategy.add_attribute_fuser("name_norm", most_frequent_non_null)
strategy.add_attribute_fuser("website", shortest_non_null_string)
strategy.add_attribute_fuser("map_url", shortest_non_null_string)
strategy.add_attribute_fuser("phone_raw", most_frequent_non_null)
strategy.add_attribute_fuser("phone_e164", most_frequent_non_null)
strategy.add_attribute_fuser("address_line1", most_frequent_non_null)
strategy.add_attribute_fuser("address_line2", most_frequent_non_null)
strategy.add_attribute_fuser("street", most_frequent_non_null)
strategy.add_attribute_fuser("house_number", most_frequent_non_null)
strategy.add_attribute_fuser("city", most_frequent_non_null)
strategy.add_attribute_fuser("state", most_frequent_non_null)
strategy.add_attribute_fuser("postal_code", most_frequent_non_null)
strategy.add_attribute_fuser("country", most_frequent_non_null)
strategy.add_attribute_fuser("latitude", median_numeric)
strategy.add_attribute_fuser("longitude", median_numeric)
strategy.add_attribute_fuser("categories", union_resolver)
strategy.add_attribute_fuser("rating", median_numeric)
strategy.add_attribute_fuser("rating_count", median_numeric)
strategy.add_attribute_fuser("source", union_resolver)

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
    include_singletons=True,
)

if "phone_e164" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["phone_e164"] = ml_fused_standard_blocker["phone_e164"].apply(normalize_phone)

if "phone_raw" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["phone_raw"] = ml_fused_standard_blocker["phone_raw"].apply(normalize_phone)

if "website" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["website"] = ml_fused_standard_blocker["website"].apply(normalize_url)

if "map_url" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["map_url"] = ml_fused_standard_blocker["map_url"].apply(normalize_url)

for col in ["postal_code"]:
    if col in ml_fused_standard_blocker.columns:
        ml_fused_standard_blocker[col] = ml_fused_standard_blocker[col].apply(normalize_postal_code)

for col in ["house_number"]:
    if col in ml_fused_standard_blocker.columns:
        ml_fused_standard_blocker[col] = ml_fused_standard_blocker[col].apply(normalize_house_number)

for col in ["latitude", "longitude", "rating", "rating_count"]:
    if col in ml_fused_standard_blocker.columns:
        ml_fused_standard_blocker[col] = ml_fused_standard_blocker[col].apply(normalize_float)

if "categories" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["categories"] = ml_fused_standard_blocker["categories"].apply(
        lambda x: normalize_categories(x) if not pd.isna(x) else None
    )

if "source" in ml_fused_standard_blocker.columns:
    def normalize_source_value(x):
        if pd.isna(x):
            return "[]"
        if isinstance(x, list):
            vals = [str(v).strip() for v in x if str(v).strip()]
            return json.dumps(sorted(set(vals)))
        s = str(x).strip()
        if not s:
            return "[]"
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                vals = [str(v).strip() for v in parsed if str(v).strip()]
                return json.dumps(sorted(set(vals)))
        except Exception:
            pass
        return json.dumps([s])

    ml_fused_standard_blocker["source"] = ml_fused_standard_blocker["source"].apply(normalize_source_value)

if "_id" not in ml_fused_standard_blocker.columns:
    if "id" in ml_fused_standard_blocker.columns:
        ml_fused_standard_blocker["_id"] = ml_fused_standard_blocker["id"].astype(str)
    else:
        ml_fused_standard_blocker["_id"] = [
            f"fused_{i}" for i in range(len(ml_fused_standard_blocker))
        ]

ml_fused_standard_blocker["_id"] = ml_fused_standard_blocker["_id"].astype(str)

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=22
node_name=execute_pipeline
accuracy_score=45.18%
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

from PyDI.fusion import DataFusionStrategy, DataFusionEngine

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import numpy as np
import os
import ast
import json
import hashlib
from urllib.parse import urlparse, urlunparse


def lower_strip(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


def strip_only(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_phone(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    if s.startswith("+"):
        digits = "".join(ch for ch in s[1:] if ch.isdigit())
        return f"+{digits}" if digits else None
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits if digits else None


def normalize_text(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    return s


def normalize_postal_code(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s if s else None


def normalize_house_number(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s if s else None


def normalize_float(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def normalize_url(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    try:
        parsed = urlparse(s)
        scheme = parsed.scheme.lower() if parsed.scheme else "https"
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/")
        normalized = urlunparse((scheme, netloc, path, "", parsed.query, ""))
        return normalized if normalized else s
    except Exception:
        return s


def parse_list_like(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", ""}:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def _title_case_category(text):
    words = str(text).strip().split()
    return " ".join(w.capitalize() if not w.isupper() else w for w in words)


def canonicalize_category(cat):
    s = str(cat).strip()
    if not s or s.lower() in {"nan", "none", ""}:
        return None
    s_clean = " ".join(s.replace("&", " & ").split())
    lookup = {
        "bbq": "BBQ",
        "barbeque": "BBQ",
        "barbecue": "BBQ",
        "burgers": "Burgers",
        "burger": "Burgers",
        "american": "American",
        "sandwiches": "Sandwiches",
        "sandwich": "Sandwiches",
        "breakfast": "Breakfast",
        "seafood": "Seafood",
        "fast food restaurant": "Fast Food Restaurant",
        "ice cream shop": "Ice Cream Shop",
        "wine bar": "Wine Bar",
        "pizza": "Pizza",
        "coffee": "Coffee",
        "cafe": "Cafe",
        "cafes": "Cafe",
        "restaurant": "Restaurant",
    }
    key = s_clean.lower()
    if key in lookup:
        return lookup[key]
    return _title_case_category(s_clean)


def normalize_categories(x):
    vals = parse_list_like(x)
    cleaned = []
    seen = set()
    for v in vals:
        c = canonicalize_category(v)
        if c is None:
            continue
        key = c.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(c)
    if not cleaned:
        return None
    cleaned = sorted(cleaned, key=lambda z: z.lower())
    return json.dumps(cleaned, ensure_ascii=False)


def _extract_raw_values(values):
    cleaned = []
    for item in values:
        if isinstance(item, tuple):
            value = item[0] if len(item) >= 1 else None
        else:
            value = item
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        if isinstance(value, str):
            s = value.strip()
            if not s or s.lower() in {"nan", "none", ""}:
                continue
            cleaned.append(s)
        else:
            cleaned.append(value)
    return cleaned


def first_non_null(values):
    cleaned = _extract_raw_values(values)
    return cleaned[0] if cleaned else None


def most_frequent_non_null(values):
    cleaned = _extract_raw_values(values)
    if not cleaned:
        return None
    counts = {}
    first_seen = {}
    for i, v in enumerate(cleaned):
        key = json.dumps(v, sort_keys=True, default=str)
        counts[key] = counts.get(key, 0) + 1
        if key not in first_seen:
            first_seen[key] = (i, v)
    best_key = sorted(counts.keys(), key=lambda k: (-counts[k], first_seen[k][0]))[0]
    return first_seen[best_key][1]


def longest_non_null_string(values):
    cleaned = []
    for v in _extract_raw_values(values):
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", ""}:
            cleaned.append(s)
    if not cleaned:
        return None
    return max(cleaned, key=lambda x: (len(x), x))


def choose_best_address_line1(values):
    cleaned = []
    for v in _extract_raw_values(values):
        s = str(v).strip()
        if s:
            cleaned.append(s)
    if not cleaned:
        return None
    scored = []
    for s in cleaned:
        has_digit = any(ch.isdigit() for ch in s)
        score = (
            1 if has_digit else 0,
            len(s.split()),
            len(s),
        )
        scored.append((score, s))
    scored.sort(reverse=True)
    return scored[0][1]


def prefer_non_null_max_numeric(values):
    cleaned = []
    for v in _extract_raw_values(values):
        try:
            cleaned.append(float(v))
        except Exception:
            continue
    if not cleaned:
        return None
    return float(max(cleaned))


def prefer_yelp_rating(values):
    cleaned = []
    for item in values:
        if isinstance(item, tuple):
            value = item[0] if len(item) >= 1 else None
            source_name = str(item[1]).lower() if len(item) >= 2 and item[1] is not None else ""
        else:
            value = item
            source_name = ""
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        try:
            fv = float(value)
        except Exception:
            continue
        cleaned.append((source_name, fv))
    if not cleaned:
        return None
    for source_name, fv in cleaned:
        if "yelp" in source_name:
            return float(round(fv, 1))
    for source_name, fv in cleaned:
        if "kaggle" in source_name:
            return float(round(fv, 1))
    for source_name, fv in cleaned:
        if "uber" in source_name:
            return float(round(fv, 1))
    return float(round(cleaned[0][1], 1))


def choose_best_map_url(values):
    cleaned = []
    for v in _extract_raw_values(values):
        s = normalize_url(v)
        if s:
            cleaned.append(s)
    if not cleaned:
        return None

    def score(url):
        u = url.lower()
        return (
            1 if "yelp.com" in u else 0,
            1 if "google.com/maps" in u or "maps.google.com" in u else 0,
            1 if "google.com" in u else 0,
            len(url),
        )

    return sorted(cleaned, key=score, reverse=True)[0]


def choose_best_website(values):
    cleaned = []
    for v in _extract_raw_values(values):
        s = normalize_url(v)
        if s:
            cleaned.append(s)
    if not cleaned:
        return None

    def score(url):
        u = url.lower()
        return (
            0 if "google.com" in u or "yelp.com" in u or "ubereats.com" in u else 1,
            0 if "maps" in u else 1,
            len(url),
        )

    return sorted(cleaned, key=score, reverse=True)[0]


def source_priority_text(values):
    cleaned = []
    for item in values:
        if isinstance(item, tuple):
            value = item[0] if len(item) >= 1 else None
            source_name = str(item[1]).lower() if len(item) >= 2 and item[1] is not None else ""
        else:
            value = item
            source_name = ""
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        s = str(value).strip()
        if not s or s.lower() in {"nan", "none", ""}:
            continue
        priority = 0
        if "yelp" in source_name:
            priority = 3
        elif "kaggle" in source_name:
            priority = 2
        elif "uber" in source_name:
            priority = 1
        cleaned.append((priority, len(s), s))
    if not cleaned:
        return None
    cleaned.sort(reverse=True)
    return cleaned[0][2]


def categories_resolver(values):
    merged = []
    seen = set()
    for item in values:
        value = item[0] if isinstance(item, tuple) and len(item) >= 1 else item
        for v in parse_list_like(value):
            c = canonicalize_category(v)
            if c is None:
                continue
            key = c.lower()
            if key not in seen:
                seen.add(key)
                merged.append(c)
    if not merged:
        return None
    merged = sorted(merged, key=lambda z: z.lower())
    return json.dumps(merged, ensure_ascii=False)


def source_union_resolver(values):
    merged = []
    seen = set()
    for item in values:
        value = item[0] if isinstance(item, tuple) and len(item) >= 1 else item
        if value is None:
            continue
        if isinstance(value, str):
            parsed = parse_list_like(value)
        elif isinstance(value, list):
            parsed = value
        else:
            parsed = [value]
        for v in parsed:
            sv = str(v).strip()
            if not sv or sv.lower() in {"nan", "none", ""}:
                continue
            if sv not in seen:
                seen.add(sv)
                merged.append(sv)
    if not merged:
        return None
    return json.dumps(sorted(merged), ensure_ascii=False)


def deterministic_cluster_id(row):
    source_vals = row.get("source", None)
    if source_vals is None or (isinstance(source_vals, float) and pd.isna(source_vals)):
        base = row.get("id", "")
        return str(base)
    vals = parse_list_like(source_vals)
    vals = sorted(str(v).strip() for v in vals if str(v).strip())
    if not vals:
        base = row.get("id", "")
        return str(base)
    joined = "|".join(vals)
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()[:16]
    return f"cluster_{digest}"


# --------------------------------
# Prepare Data
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

kaggle_small["id"] = (
    kaggle_small["kaggle380k_id"]
    if "kaggle380k_id" in kaggle_small.columns
    else kaggle_small["id"]
)
uber_eats_small["id"] = (
    uber_eats_small["kaggle380k_id"]
    if "kaggle380k_id" in uber_eats_small.columns
    else uber_eats_small["id"]
)
yelp_small["id"] = (
    yelp_small["kaggle380k_id"]
    if "kaggle380k_id" in yelp_small.columns
    else yelp_small["id"]
)

datasets = [kaggle_small, uber_eats_small, yelp_small]

for df in datasets:
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    if "phone_e164" in df.columns:
        df["phone_e164"] = df["phone_e164"].apply(normalize_phone)
    if "phone_raw" in df.columns:
        df["phone_raw"] = df["phone_raw"].apply(normalize_phone)
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].apply(normalize_postal_code)
    if "house_number" in df.columns:
        df["house_number"] = df["house_number"].apply(normalize_house_number)
    if "website" in df.columns:
        df["website"] = df["website"].apply(normalize_url)
    if "map_url" in df.columns:
        df["map_url"] = df["map_url"].apply(normalize_url)
    if "name" in df.columns:
        df["name"] = df["name"].apply(normalize_text)
    if "name_norm" in df.columns:
        df["name_norm"] = df["name_norm"].apply(lower_strip)
    if "address_line1" in df.columns:
        df["address_line1"] = df["address_line1"].apply(normalize_text)
    if "address_line2" in df.columns:
        df["address_line2"] = df["address_line2"].apply(normalize_text)
    if "street" in df.columns:
        df["street"] = df["street"].apply(normalize_text)
    if "city" in df.columns:
        df["city"] = df["city"].apply(normalize_text)
    if "state" in df.columns:
        df["state"] = df["state"].apply(normalize_text)
    if "country" in df.columns:
        df["country"] = df["country"].apply(normalize_text)
    if "categories" in df.columns:
        df["categories"] = df["categories"].apply(normalize_categories)
    if "latitude" in df.columns:
        df["latitude"] = df["latitude"].apply(normalize_float)
    if "longitude" in df.columns:
        df["longitude"] = df["longitude"].apply(normalize_float)
    if "rating" in df.columns:
        df["rating"] = df["rating"].apply(normalize_float)
    if "rating_count" in df.columns:
        df["rating_count"] = df["rating_count"].apply(normalize_float)

# --------------------------------
# Perform Blocking
# --------------------------------

print("Performing Blocking")

blocker_1_2 = EmbeddingBlocker(
    kaggle_small,
    uber_eats_small,
    text_cols=["name_norm", "city", "state"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_1_3 = StandardBlocker(
    kaggle_small,
    yelp_small,
    on=["phone_e164", "postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_2_3 = EmbeddingBlocker(
    uber_eats_small,
    yelp_small,
    text_cols=["name_norm", "street", "city"],
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
    NumericComparator(column="latitude", max_difference=0.01),
    NumericComparator(column="longitude", max_difference=0.01),
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
    NumericComparator(column="latitude", max_difference=0.001),
    NumericComparator(column="longitude", max_difference=0.001),
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
    NumericComparator(column="latitude", max_difference=0.01),
    NumericComparator(column="longitude", max_difference=0.01),
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

train_1_2 = load_csv(
    "input/datasets/restaurant/testsets/kaggle_uber_eats_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_small_uber_eats_small_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/datasets/restaurant/testsets/kaggle_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_kaggle_small_yelp_small_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/restaurant/testsets/uber_eats_yelp_goldstandard_blocking_small.csv",
    name="ground_truth_uber_eats_small_yelp_small_train",
    add_index=False,
)

train_1_2["id1"] = train_1_2["id1"].astype(str)
train_1_2["id2"] = train_1_2["id2"].astype(str)
train_1_3["id1"] = train_1_3["id1"].astype(str)
train_1_3["id2"] = train_1_3["id2"].astype(str)
train_2_3["id1"] = train_2_3["id1"].astype(str)
train_2_3["id2"] = train_2_3["id2"].astype(str)

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
            "n_estimators": [100, 200],
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
            "learning_rate": [0.05, 0.1, 0.2],
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

clusterer = MaximumBipartiteMatching()
ml_correspondences_1_2 = clusterer.cluster(ml_correspondences_1_2)
ml_correspondences_1_3 = clusterer.cluster(ml_correspondences_1_3)
ml_correspondences_2_3 = clusterer.cluster(ml_correspondences_2_3)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_1_2.to_csv(
    os.path.join(CORR_DIR, "correspondences_kaggle_small_uber_eats_small.csv"),
    index=False,
)
ml_correspondences_1_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_kaggle_small_yelp_small.csv"),
    index=False,
)
ml_correspondences_2_3.to_csv(
    os.path.join(CORR_DIR, "correspondences_uber_eats_small_yelp_small.csv"),
    index=False,
)

print("Fusing Data")

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", source_priority_text)
strategy.add_attribute_fuser("name_norm", source_priority_text)
strategy.add_attribute_fuser("website", choose_best_website)
strategy.add_attribute_fuser("map_url", choose_best_map_url)
strategy.add_attribute_fuser("phone_raw", source_priority_text)
strategy.add_attribute_fuser("phone_e164", source_priority_text)
strategy.add_attribute_fuser("address_line1", choose_best_address_line1)
strategy.add_attribute_fuser("address_line2", most_frequent_non_null)
strategy.add_attribute_fuser("street", source_priority_text)
strategy.add_attribute_fuser("house_number", source_priority_text)
strategy.add_attribute_fuser("city", source_priority_text)
strategy.add_attribute_fuser("state", source_priority_text)
strategy.add_attribute_fuser("postal_code", source_priority_text)
strategy.add_attribute_fuser("country", source_priority_text)
strategy.add_attribute_fuser("latitude", first_non_null)
strategy.add_attribute_fuser("longitude", first_non_null)
strategy.add_attribute_fuser("categories", categories_resolver)
strategy.add_attribute_fuser("rating", prefer_yelp_rating)
strategy.add_attribute_fuser("rating_count", prefer_non_null_max_numeric)
strategy.add_attribute_fuser("source", source_union_resolver)

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
    include_singletons=True,
)

if "phone_e164" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["phone_e164"] = ml_fused_standard_blocker["phone_e164"].apply(normalize_phone)

if "phone_raw" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["phone_raw"] = ml_fused_standard_blocker["phone_raw"].apply(normalize_phone)

if "website" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["website"] = ml_fused_standard_blocker["website"].apply(normalize_url)

if "map_url" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["map_url"] = ml_fused_standard_blocker["map_url"].apply(normalize_url)

for col in ["postal_code"]:
    if col in ml_fused_standard_blocker.columns:
        ml_fused_standard_blocker[col] = ml_fused_standard_blocker[col].apply(normalize_postal_code)

for col in ["house_number"]:
    if col in ml_fused_standard_blocker.columns:
        ml_fused_standard_blocker[col] = ml_fused_standard_blocker[col].apply(normalize_house_number)

for col in ["latitude", "longitude", "rating", "rating_count"]:
    if col in ml_fused_standard_blocker.columns:
        ml_fused_standard_blocker[col] = ml_fused_standard_blocker[col].apply(normalize_float)

for col in [
    "name",
    "address_line1",
    "address_line2",
    "street",
    "city",
    "state",
    "country",
]:
    if col in ml_fused_standard_blocker.columns:
        ml_fused_standard_blocker[col] = ml_fused_standard_blocker[col].apply(normalize_text)

if "name_norm" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["name_norm"] = ml_fused_standard_blocker["name_norm"].apply(lower_strip)

if "categories" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["categories"] = ml_fused_standard_blocker["categories"].apply(
        lambda x: normalize_categories(x) if not pd.isna(x) else None
    )

if "source" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["source"] = ml_fused_standard_blocker["source"].apply(
        lambda x: source_union_resolver([x]) if not pd.isna(x) else json.dumps([])
    )

ml_fused_standard_blocker["_id"] = ml_fused_standard_blocker.apply(deterministic_cluster_id, axis=1)
ml_fused_standard_blocker["_id"] = ml_fused_standard_blocker["_id"].astype(str)

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

