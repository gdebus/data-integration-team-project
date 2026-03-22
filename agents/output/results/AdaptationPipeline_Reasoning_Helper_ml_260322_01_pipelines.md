# Pipeline Snapshots

notebook_name=AdaptationPipeline_Reasoning_Helper
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=7
node_name=execute_pipeline
accuracy_score=78.22%
------------------------------------------------------------

```python
from PyDI.io import load_csv
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
    median,
    union,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

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
        Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd(),
        Path(__file__).resolve().parent.parent.parent if "__file__" in globals() else Path.cwd(),
        Path(__file__).resolve().parent.parent.parent.parent if "__file__" in globals() else Path.cwd(),
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

Path("output/correspondences").mkdir(parents=True, exist_ok=True)
Path("output/data_fusion").mkdir(parents=True, exist_ok=True)
Path("output/blocking-evaluation").mkdir(parents=True, exist_ok=True)


def is_missing(x):
    if x is None:
        return True
    try:
        result = pd.isna(x)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        return False
    except Exception:
        return False


def lower_strip(x):
    if is_missing(x):
        return ""
    return str(x).lower().strip()


def stringify_for_embedding(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        cleaned = []
        for item in value:
            if item is None:
                continue
            try:
                if pd.isna(item):
                    continue
            except Exception:
                pass
            s = str(item).strip()
            if s:
                cleaned.append(s)
        return " ".join(cleaned)
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def standardize_correspondence_columns(df):
    out = df.copy()
    rename_candidates = [
        ("left_id", "id1"),
        ("right_id", "id2"),
        ("source_id", "id1"),
        ("target_id", "id2"),
        ("entity_id_1", "id1"),
        ("entity_id_2", "id2"),
    ]
    for old, new in rename_candidates:
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})

    if "id1" not in out.columns or "id2" not in out.columns:
        id_like = [c for c in out.columns if "id" in str(c).lower()]
        if len(id_like) >= 2:
            out = out.rename(columns={id_like[0]: "id1", id_like[1]: "id2"})

    if "id1" not in out.columns or "id2" not in out.columns:
        raise ValueError(f"Correspondence columns could not be standardized: {list(df.columns)}")

    keep = [c for c in out.columns if c in {"id1", "id2", "score", "similarity", "label", "confidence", "probability"}]
    if keep:
        out = out[keep].copy()

    out["id1"] = out["id1"].astype(str)
    out["id2"] = out["id2"].astype(str)
    return out.drop_duplicates(subset=["id1", "id2"])


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
    remaining_cols = [c for c in df.columns if c not in pair_ids.columns]
    if len(remaining_cols) == 1:
        return df[remaining_cols[0]].astype(int)

    raise ValueError(f"Could not infer label column from: {list(df.columns)}")


def prepare_embedding_views(df, text_cols):
    emb_df = df.copy()
    for col in text_cols:
        if col in emb_df.columns:
            emb_df[col] = emb_df[col].apply(stringify_for_embedding)
    return emb_df


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

embedding_text_cols_1_2 = ["address_line1", "name_norm", "city"]
blocking_dataset_1_2_left = prepare_embedding_views(good_dataset_name_1, embedding_text_cols_1_2)
blocking_dataset_1_2_right = prepare_embedding_views(good_dataset_name_2, embedding_text_cols_1_2)

blocker_1_2 = EmbeddingBlocker(
    blocking_dataset_1_2_left,
    blocking_dataset_1_2_right,
    text_cols=embedding_text_cols_1_2,
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
    on=["postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

comparators_1_2 = [
    StringComparator(
        column="address_line1",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
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
    StringComparator(
        column="state",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
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
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="house_number",
        similarity_function="levenshtein",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_2_3 = [
    NumericComparator(
        column="postal_code",
        max_difference=0.0,
        list_strategy="average",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="address_line1",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
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

ml_matcher_1_2 = MLBasedMatcher(feature_extractor_1_2)
ml_matcher_1_3 = MLBasedMatcher(feature_extractor_1_3)
ml_matcher_2_3 = MLBasedMatcher(feature_extractor_2_3)

ml_correspondences_1_2 = ml_matcher_1_2.match(
    good_dataset_name_1,
    good_dataset_name_2,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
    threshold=0.78,
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

ml_correspondences_1_2 = standardize_correspondence_columns(ml_correspondences_1_2)
ml_correspondences_1_3 = standardize_correspondence_columns(ml_correspondences_1_3)
ml_correspondences_2_3 = standardize_correspondence_columns(ml_correspondences_2_3)

print(f"kaggle-uber_eats correspondences: {len(ml_correspondences_1_2)}")
print(f"kaggle-yelp correspondences: {len(ml_correspondences_1_3)}")
print(f"uber_eats-yelp correspondences: {len(ml_correspondences_2_3)}")

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

ml_correspondences_1_2.to_csv(
    "output/correspondences/correspondences_kaggle_small_uber_eats_small.csv",
    index=False,
)
ml_correspondences_1_3.to_csv(
    "output/correspondences/correspondences_kaggle_small_yelp_small.csv",
    index=False,
)
ml_correspondences_2_3.to_csv(
    "output/correspondences/correspondences_uber_eats_small_yelp_small.csv",
    index=False,
)

print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_1_2 = standardize_correspondence_columns(clusterer.cluster(ml_correspondences_1_2))
ml_correspondences_1_3 = standardize_correspondence_columns(clusterer.cluster(ml_correspondences_1_3))
ml_correspondences_2_3 = standardize_correspondence_columns(clusterer.cluster(ml_correspondences_2_3))

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
).drop_duplicates()

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("id", longest_string)
strategy.add_attribute_fuser("source", union)
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("name_norm", longest_string)
strategy.add_attribute_fuser("website", longest_string)
strategy.add_attribute_fuser("map_url", longest_string)
strategy.add_attribute_fuser("phone_raw", median)
strategy.add_attribute_fuser("phone_e164", median)
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

============================================================
PIPELINE SNAPSHOT 02 START
============================================================
node_index=14
node_name=execute_pipeline
accuracy_score=85.11%
------------------------------------------------------------

```python
from PyDI.io import load_csv
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
    median,
    union,
    prefer_higher_trust,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re
import ast

try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _candidates = [
        Path.cwd(),
        Path.cwd() / "agents",
        Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd(),
        Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd(),
        Path(__file__).resolve().parent.parent.parent if "__file__" in globals() else Path.cwd(),
        Path(__file__).resolve().parent.parent.parent.parent if "__file__" in globals() else Path.cwd(),
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

Path("output/correspondences").mkdir(parents=True, exist_ok=True)
Path("output/data_fusion").mkdir(parents=True, exist_ok=True)
Path("output/blocking-evaluation").mkdir(parents=True, exist_ok=True)


def is_missing(x):
    if x is None:
        return True
    try:
        result = pd.isna(x)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        return False
    except Exception:
        return False


def lower_strip(x):
    if is_missing(x):
        return ""
    return str(x).lower().strip()


def stringify_for_embedding(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        cleaned = []
        for item in value:
            if item is None:
                continue
            try:
                if pd.isna(item):
                    continue
            except Exception:
                pass
            s = str(item).strip()
            if s:
                cleaned.append(s)
        return " ".join(cleaned)
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def parse_list_like(value):
    if is_missing(value):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if not is_missing(v) and str(v).strip()]
    if isinstance(value, (tuple, set)):
        return [str(v).strip() for v in value if not is_missing(v) and str(v).strip()]
    s = str(value).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple, set)):
            return [str(v).strip() for v in parsed if not is_missing(v) and str(v).strip()]
    except Exception:
        pass
    return [s]


def canonicalize_postal_code(value):
    if is_missing(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 5:
        return digits[:5]
    return digits if digits else np.nan


def canonicalize_phone(value):
    if is_missing(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"\D", "", s)
    return digits if digits else np.nan


def clean_name_norm(value):
    if is_missing(value):
        return np.nan
    s = lower_strip(value)
    s = re.sub(r"\brestaurant\b", "", s)
    s = re.sub(r"\bgrill\b(?=\s*&\s*chill)", "grill", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else np.nan


def clean_text_scalar(value):
    if is_missing(value):
        return np.nan
    s = str(value).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else np.nan


def canonicalize_category_token(token):
    t = lower_strip(token)
    t = re.sub(r"\brestaurants?\b", "", t)
    t = re.sub(r"\brestaurant\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    replacements = {
        "latin american restaurant": "latin american",
        "american restaurant": "american",
        "breakfast brunch": "breakfast",
        "steak house": "steakhouse",
        "ice cream shop": "ice cream",
        "fast food restaurant": "fast food",
    }
    return replacements.get(t, t)


def canonicalize_categories(value):
    items = parse_list_like(value)
    cleaned = []
    seen = set()
    for item in items:
        token = canonicalize_category_token(item)
        if token and token not in seen:
            seen.add(token)
            cleaned.append(token)
    return cleaned if cleaned else np.nan


def standardize_correspondence_columns(df):
    out = df.copy()
    rename_candidates = [
        ("left_id", "id1"),
        ("right_id", "id2"),
        ("source_id", "id1"),
        ("target_id", "id2"),
        ("entity_id_1", "id1"),
        ("entity_id_2", "id2"),
    ]
    for old, new in rename_candidates:
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})

    if "id1" not in out.columns or "id2" not in out.columns:
        id_like = [c for c in out.columns if "id" in str(c).lower()]
        if len(id_like) >= 2:
            out = out.rename(columns={id_like[0]: "id1", id_like[1]: "id2"})

    if "id1" not in out.columns or "id2" not in out.columns:
        raise ValueError(f"Correspondence columns could not be standardized: {list(df.columns)}")

    keep = [c for c in out.columns if c in {"id1", "id2", "score", "similarity", "label", "confidence", "probability"}]
    if keep:
        out = out[keep].copy()

    out["id1"] = out["id1"].astype(str)
    out["id2"] = out["id2"].astype(str)
    return out.drop_duplicates(subset=["id1", "id2"])


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
    remaining_cols = [c for c in df.columns if c not in pair_ids.columns]
    if len(remaining_cols) == 1:
        return df[remaining_cols[0]].astype(int)

    raise ValueError(f"Could not infer label column from: {list(df.columns)}")


def prepare_embedding_views(df, text_cols):
    emb_df = df.copy()
    for col in text_cols:
        if col in emb_df.columns:
            emb_df[col] = emb_df[col].apply(stringify_for_embedding)
    return emb_df


def preprocess_dataset(df):
    out = df.copy()
    if "postal_code" in out.columns:
        out["postal_code"] = out["postal_code"].apply(canonicalize_postal_code)
    if "phone_raw" in out.columns:
        out["phone_raw"] = out["phone_raw"].apply(canonicalize_phone)
    if "phone_e164" in out.columns:
        out["phone_e164"] = out["phone_e164"].apply(canonicalize_phone)
    if "name_norm" in out.columns:
        out["name_norm"] = out["name_norm"].apply(clean_name_norm)
    text_cols = [
        "name",
        "website",
        "map_url",
        "address_line1",
        "address_line2",
        "street",
        "house_number",
        "city",
        "state",
        "country",
        "source",
    ]
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].apply(clean_text_scalar)
    if "categories" in out.columns:
        out["categories"] = out["categories"].apply(canonicalize_categories)
    return out


good_dataset_name_1 = load_csv(
    DATA_DIR + "output/normalization/attempt_2/kaggle_small.csv",
    name="kaggle_small",
)

good_dataset_name_2 = load_csv(
    DATA_DIR + "output/normalization/attempt_2/uber_eats_small.csv",
    name="uber_eats_small",
)

good_dataset_name_3 = load_csv(
    DATA_DIR + "output/normalization/attempt_2/yelp_small.csv",
    name="yelp_small",
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

good_dataset_name_1 = preprocess_dataset(good_dataset_name_1)
good_dataset_name_2 = preprocess_dataset(good_dataset_name_2)
good_dataset_name_3 = preprocess_dataset(good_dataset_name_3)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

print("Performing Blocking")

embedding_text_cols_1_2 = ["address_line1", "name_norm", "city"]
blocking_dataset_1_2_left = prepare_embedding_views(good_dataset_name_1, embedding_text_cols_1_2)
blocking_dataset_1_2_right = prepare_embedding_views(good_dataset_name_2, embedding_text_cols_1_2)

blocker_1_2 = EmbeddingBlocker(
    blocking_dataset_1_2_left,
    blocking_dataset_1_2_right,
    text_cols=embedding_text_cols_1_2,
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
    on=["postal_code"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

comparators_1_2 = [
    StringComparator(
        column="address_line1",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
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
    StringComparator(
        column="state",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
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
    StringComparator(
        column="street",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="house_number",
        similarity_function="levenshtein",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_2_3 = [
    NumericComparator(
        column="postal_code",
        max_difference=0.0,
        list_strategy="average",
    ),
    StringComparator(
        column="name_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="address_line1",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
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
    good_dataset_name_1,
    good_dataset_name_2,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
    threshold=0.78,
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

ml_correspondences_1_2 = standardize_correspondence_columns(ml_correspondences_1_2)
ml_correspondences_1_3 = standardize_correspondence_columns(ml_correspondences_1_3)
ml_correspondences_2_3 = standardize_correspondence_columns(ml_correspondences_2_3)

print(f"kaggle-uber_eats correspondences: {len(ml_correspondences_1_2)}")
print(f"kaggle-yelp correspondences: {len(ml_correspondences_1_3)}")
print(f"uber_eats-yelp correspondences: {len(ml_correspondences_2_3)}")

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

ml_correspondences_1_2.to_csv(
    "output/correspondences/correspondences_kaggle_small_uber_eats_small.csv",
    index=False,
)
ml_correspondences_1_3.to_csv(
    "output/correspondences/correspondences_kaggle_small_yelp_small.csv",
    index=False,
)
ml_correspondences_2_3.to_csv(
    "output/correspondences/correspondences_uber_eats_small_yelp_small.csv",
    index=False,
)

print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_1_2 = standardize_correspondence_columns(clusterer.cluster(ml_correspondences_1_2))
ml_correspondences_1_3 = standardize_correspondence_columns(clusterer.cluster(ml_correspondences_1_3))
ml_correspondences_2_3 = standardize_correspondence_columns(clusterer.cluster(ml_correspondences_2_3))

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
).drop_duplicates()

trust_map = {
    "yelp_small": 1.0,
    "uber_eats_small": 0.9,
    "kaggle_small": 0.8,
    "yelp": 1.0,
    "uber_eats": 0.9,
    "kaggle_380k": 0.8,
}

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("id", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("source", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("name", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("name_norm", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("website", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("map_url", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("phone_raw", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("phone_e164", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("address_line1", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("address_line2", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("street", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("house_number", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("city", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("state", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("postal_code", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("country", prefer_higher_trust, trust_map=trust_map)
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

if "postal_code" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["postal_code"] = ml_fused_standard_blocker["postal_code"].apply(canonicalize_postal_code)
if "phone_raw" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["phone_raw"] = ml_fused_standard_blocker["phone_raw"].apply(canonicalize_phone)
if "phone_e164" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["phone_e164"] = ml_fused_standard_blocker["phone_e164"].apply(canonicalize_phone)
if "name_norm" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["name_norm"] = ml_fused_standard_blocker["name_norm"].apply(clean_name_norm)
if "categories" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["categories"] = ml_fused_standard_blocker["categories"].apply(canonicalize_categories)

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

