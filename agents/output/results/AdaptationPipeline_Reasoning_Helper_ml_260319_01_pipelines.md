# Pipeline Snapshots

notebook_name=AdaptationPipeline_nblazek_setcount_guardline
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=9
node_name=execute_pipeline
accuracy_score=7.20%
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
    union,
    median,
    most_recent,
)
from PyDI.entitymatching import MaximumBipartiteMatching

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

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
# Prepare Data
# --------------------------------

DATA_DIR = ""

good_dataset_name_1 = load_csv(
    DATA_DIR + "output/normalization/attempt_1/discogs.csv",
    name="discogs",
)

good_dataset_name_2 = load_csv(
    DATA_DIR + "output/normalization/attempt_1/lastfm.csv",
    name="lastfm",
)

good_dataset_name_3 = load_csv(
    DATA_DIR + "output/normalization/attempt_1/musicbrainz.csv",
    name="musicbrainz",
)

good_dataset_name_1["id"] = good_dataset_name_1["id"]
good_dataset_name_2["id"] = good_dataset_name_2["id"]
good_dataset_name_3["id"] = good_dataset_name_3["id"]

# Normalize list-like attributes so list comparators/fusers work on true lists.
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

# EmbeddingBlocker fails on list-valued columns because it calls pd.isna(value)
# on each text column value. Therefore, create dedicated scalar text columns for
# blocking only, while keeping the original normalized list columns for matching/fusion.
def scalarize_for_blocking(value):
    if value is None:
        return ""
    if isinstance(value, list):
        cleaned = [str(v).strip() for v in value if pd.notna(v) and str(v).strip() != ""]
        return " ".join(cleaned)
    if pd.isna(value):
        return ""
    return str(value).strip()

for df in datasets:
    df["tracks_track_duration_block"] = df["tracks_track_duration"].apply(scalarize_for_blocking) if "tracks_track_duration" in df.columns else ""
    df["duration_block"] = df["duration"].apply(scalarize_for_blocking) if "duration" in df.columns else ""
    df["name_block"] = df["name"].apply(scalarize_for_blocking) if "name" in df.columns else ""
    df["artist_block"] = df["artist"].apply(scalarize_for_blocking) if "artist" in df.columns else ""

print("Performing Blocking")

blocker_1_2 = EmbeddingBlocker(
    good_dataset_name_1,
    good_dataset_name_2,
    text_cols=["name_block", "artist_block", "tracks_track_duration_block"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_1_3 = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    on=["name"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_2_3 = EmbeddingBlocker(
    good_dataset_name_3,
    good_dataset_name_2,
    text_cols=["name_block", "artist_block", "duration_block"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

def lower_only(x):
    if x is None:
        return ""
    if isinstance(x, list):
        x = " ".join([str(v) for v in x if pd.notna(v)])
    elif pd.isna(x):
        return ""
    return str(x).lower()

def lower_strip(x):
    if x is None:
        return ""
    if isinstance(x, list):
        x = " ".join([str(v) for v in x if pd.notna(v)])
    elif pd.isna(x):
        return ""
    return str(x).lower().strip()

comparators_1_2 = [
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lower_only,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_only,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=10.0,
        list_strategy="average",
    ),
]

comparators_1_3 = [
    StringComparator(
        column="name",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="release-date",
        max_days_difference=365,
    ),
    StringComparator(
        column="release-country",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaccard",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_2_3 = [
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

train_1_2 = load_csv(
    "input/gen_TrainSets/discogs_lastfm_goldstandard_blocking.csv",
    name="ground_truth_discogs_lastfm_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/gen_TrainSets/discogs_musicbrainz_goldstandard_blocking.csv",
    name="ground_truth_discogs_musicbrainz_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/gen_TrainSets/musicbrainz_lastfm_goldstandard_blocking.csv",
    name="ground_truth_musicbrainz_lastfm_train",
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
    good_dataset_name_3,
    good_dataset_name_2,
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
    threshold=0.72,
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    good_dataset_name_3,
    good_dataset_name_2,
    candidates=blocker_2_3,
    id_column="id",
    trained_classifier=best_models[2],
    threshold=0.72,
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

for attr in ["name", "artist", "release-country", "label", "genre"]:
    strategy.add_attribute_fuser(attr, longest_string)

strategy.add_attribute_fuser("release-date", most_recent)
strategy.add_attribute_fuser("duration", median)
strategy.add_attribute_fuser("tracks_track_name", union)
strategy.add_attribute_fuser("tracks_track_position", union)
strategy.add_attribute_fuser("tracks_track_duration", union)

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
node_index=16
node_name=execute_pipeline
accuracy_score=56.80%
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
    most_recent,
)
from PyDI.entitymatching import MaximumBipartiteMatching

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import os
import ast
import json
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
    DATA_DIR + "output/normalization/attempt_2/discogs.csv",
    name="discogs",
)

good_dataset_name_2 = load_csv(
    DATA_DIR + "output/normalization/attempt_2/lastfm.csv",
    name="lastfm",
)

good_dataset_name_3 = load_csv(
    DATA_DIR + "output/normalization/attempt_2/musicbrainz.csv",
    name="musicbrainz",
)

good_dataset_name_1["id"] = good_dataset_name_1["id"]
good_dataset_name_2["id"] = good_dataset_name_2["id"]
good_dataset_name_3["id"] = good_dataset_name_3["id"]

# Normalize list-like attributes so list comparators/fusers work on true lists.
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

def is_nullish(value):
    if value is None:
        return True
    try:
        return pd.isna(value)
    except Exception:
        return False

def ensure_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text == "[]":
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        if "|" in text:
            return [v.strip() for v in text.split("|") if str(v).strip() != ""]
        return [text]
    if is_nullish(value):
        return []
    return [value]

def clean_scalar_text(value):
    if value is None or is_nullish(value):
        return ""
    return str(value).strip()

def scalarize_for_blocking(value):
    values = ensure_list(value)
    if values:
        cleaned = [clean_scalar_text(v) for v in values if clean_scalar_text(v) != ""]
        return " ".join(cleaned).strip()
    return clean_scalar_text(value)

def preprocess_lower(x):
    if isinstance(x, list):
        x = " ".join([str(v) for v in x if not is_nullish(v)])
    if x is None or is_nullish(x):
        return ""
    return str(x).lower()

def preprocess_lower_strip(x):
    if isinstance(x, list):
        x = " ".join([str(v) for v in x if not is_nullish(v)])
    if x is None or is_nullish(x):
        return ""
    return str(x).lower().strip()

def normalize_numeric_scalar(value):
    if isinstance(value, list):
        numeric_vals = []
        for v in value:
            try:
                if str(v).strip() != "":
                    numeric_vals.append(float(v))
            except Exception:
                continue
        if not numeric_vals:
            return None
        return float(sum(numeric_vals) / len(numeric_vals))
    if value is None or is_nullish(value):
        return None
    try:
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None

for df in datasets:
    for col in ["name", "artist", "duration", "tracks_track_name", "tracks_track_position", "tracks_track_duration"]:
        if col not in df.columns:
            df[col] = None

    df["tracks_track_duration_block"] = df["tracks_track_duration"].apply(scalarize_for_blocking)
    df["duration_block"] = df["duration"].apply(scalarize_for_blocking)
    df["name_block"] = df["name"].apply(scalarize_for_blocking)
    df["artist_block"] = df["artist"].apply(scalarize_for_blocking)
    df["duration"] = df["duration"].apply(normalize_numeric_scalar)

print("Performing Blocking")

blocker_1_2 = EmbeddingBlocker(
    good_dataset_name_1,
    good_dataset_name_2,
    text_cols=["name_block", "artist_block", "tracks_track_duration_block"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_1_3 = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    on=["name"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_2_3 = EmbeddingBlocker(
    good_dataset_name_3,
    good_dataset_name_2,
    text_cols=["name_block", "artist_block", "duration_block"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

comparators_1_2 = [
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=10.0,
        list_strategy="average",
    ),
]

comparators_1_3 = [
    StringComparator(
        column="name",
        similarity_function="jaccard",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="release-date",
        max_days_difference=365,
    ),
    StringComparator(
        column="release-country",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaccard",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_2_3 = [
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

train_1_2 = load_csv(
    "input/gen_TrainSets/discogs_lastfm_goldstandard_blocking.csv",
    name="ground_truth_discogs_lastfm_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/gen_TrainSets/discogs_musicbrainz_goldstandard_blocking.csv",
    name="ground_truth_discogs_musicbrainz_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/gen_TrainSets/musicbrainz_lastfm_goldstandard_blocking.csv",
    name="ground_truth_musicbrainz_lastfm_train",
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
    good_dataset_name_3,
    good_dataset_name_2,
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
    threshold=0.72,
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    good_dataset_name_3,
    good_dataset_name_2,
    candidates=blocker_2_3,
    id_column="id",
    trained_classifier=best_models[2],
    threshold=0.72,
)

ml_correspondences_1_2.to_csv(
    "output/correspondences/correspondences_discogs_lastfm.csv",
    index=False,
)
ml_correspondences_1_3.to_csv(
    "output/correspondences/correspondences_discogs_musicbrainz.csv",
    index=False,
)
ml_correspondences_2_3.to_csv(
    "output/correspondences/correspondences_musicbrainz_lastfm.csv",
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

def choose_best_non_list_string(inputs, preferred_sources=None, **kwargs):
    preferred_sources = preferred_sources or []
    cleaned = []
    for item in inputs:
        if isinstance(item, dict):
            value = item.get("value")
            source = item.get("source") or item.get("dataset") or item.get("dataset_name")
        else:
            value = item
            source = None
        if isinstance(value, list):
            continue
        if value is None or is_nullish(value):
            continue
        text = str(value).strip()
        if text == "":
            continue
        cleaned.append((text, source))

    if not cleaned:
        return None, 0.0, {}

    for preferred_source in preferred_sources:
        preferred_vals = [text for text, source in cleaned if source == preferred_source]
        if preferred_vals:
            best = max(preferred_vals, key=len)
            return best, 1.0, {"selected_source": preferred_source}

    best = max([text for text, _ in cleaned], key=len)
    return best, 1.0, {}

def fuse_track_field(inputs, field_type="generic", **kwargs):
    records = []
    for item in inputs:
        if isinstance(item, dict):
            value = item.get("value")
            source = item.get("source") or item.get("dataset") or item.get("dataset_name")
        else:
            value = item
            source = None

        values = ensure_list(value)
        cleaned = []
        for v in values:
            if v is None or is_nullish(v):
                continue
            text = str(v).strip()
            if text != "":
                cleaned.append(text)

        if cleaned:
            records.append((source, cleaned))

    if not records:
        return [], 0.0, {}

    preferred_order = ["musicbrainz", "discogs", "lastfm"]

    if field_type == "position":
        def position_key(values):
            numeric = []
            for v in values:
                try:
                    numeric.append(int(float(str(v).strip())))
                except Exception:
                    numeric.append(10**9)
            return (len(values), -sum(1 for x in numeric if x != 10**9))

        selected_source, selected_values = max(records, key=lambda x: position_key(x[1]))
        normalized = []
        seen = set()
        sortable = []
        unsortable = []
        for v in selected_values:
            try:
                iv = int(float(str(v).strip()))
                sortable.append(iv)
            except Exception:
                unsortable.append(str(v).strip())
        for iv in sorted(set(sortable)):
            sv = str(iv)
            if sv not in seen:
                normalized.append(sv)
                seen.add(sv)
        for v in unsortable:
            if v not in seen:
                normalized.append(v)
                seen.add(v)
        return normalized, 1.0, {"selected_source": selected_source}

    if field_type == "duration":
        for preferred_source in preferred_order:
            for source, values in records:
                if source == preferred_source:
                    numeric_count = 0
                    for v in values:
                        try:
                            float(str(v).strip())
                            numeric_count += 1
                        except Exception:
                            pass
                    if numeric_count > 0:
                        return values, 1.0, {"selected_source": preferred_source}
        selected_source, selected_values = max(records, key=lambda x: len(x[1]))
        return selected_values, 1.0, {"selected_source": selected_source}

    if field_type == "name":
        for preferred_source in preferred_order:
            for source, values in records:
                if source == preferred_source and len(values) > 0:
                    return values, 1.0, {"selected_source": preferred_source}
        selected_source, selected_values = max(records, key=lambda x: len(x[1]))
        return selected_values, 1.0, {"selected_source": selected_source}

    selected_source, selected_values = max(records, key=lambda x: len(x[1]))
    return selected_values, 1.0, {"selected_source": selected_source}

def serialize_provenance(value):
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if value is None or is_nullish(value):
        return json.dumps([])
    text = str(value).strip()
    if text == "":
        return json.dumps([])
    return text

strategy = DataFusionStrategy("ml_fusion_strategy")

def _pydi_safe_fuser(fn):
    """Adapt scalar custom fusers to PyDI resolver contract."""
    def _wrapped(values, **kwargs):
        try:
            try:
                result = fn(values, **kwargs)
            except TypeError:
                try:
                    result = fn(values, None, **kwargs)
                except TypeError:
                    result = fn(values)
        except Exception as e:
            fallback = values[0] if values else None
            return fallback, 0.1, {"error": str(e), "fallback": "first_value"}

        if isinstance(result, tuple) and len(result) == 3:
            return result
        return result, 1.0, {}

    return _wrapped

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("release-date", most_recent)
strategy.add_attribute_fuser("duration", median)
strategy.add_attribute_fuser("label", _pydi_safe_fuser(choose_best_non_list_string), preferred_sources=["discogs", "musicbrainz", "lastfm"])
strategy.add_attribute_fuser("genre", _pydi_safe_fuser(choose_best_non_list_string), preferred_sources=["discogs", "musicbrainz", "lastfm"])
strategy.add_attribute_fuser("tracks_track_name", _pydi_safe_fuser(fuse_track_field), field_type="name")
strategy.add_attribute_fuser("tracks_track_position", _pydi_safe_fuser(fuse_track_field), field_type="position")
strategy.add_attribute_fuser("tracks_track_duration", _pydi_safe_fuser(fuse_track_field), field_type="duration")

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

for provenance_col in ["_fusion_sources", "_fusion_source_datasets", "_fusion_metadata"]:
    if provenance_col in ml_fused_standard_blocker.columns:
        ml_fused_standard_blocker[provenance_col] = ml_fused_standard_blocker[provenance_col].apply(serialize_provenance)

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=23
node_name=execute_pipeline
accuracy_score=52.80%
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
    prefer_higher_trust,
)
from PyDI.entitymatching import MaximumBipartiteMatching

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import os
import ast
import json
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
    DATA_DIR + "output/normalization/attempt_3/discogs.csv",
    name="discogs",
)

good_dataset_name_2 = load_csv(
    DATA_DIR + "output/normalization/attempt_3/lastfm.csv",
    name="lastfm",
)

good_dataset_name_3 = load_csv(
    DATA_DIR + "output/normalization/attempt_3/musicbrainz.csv",
    name="musicbrainz",
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

def is_nullish(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False

def ensure_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text == "[]":
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        if "|" in text:
            return [v.strip() for v in text.split("|") if str(v).strip() != ""]
        return [text]
    if is_nullish(value):
        return []
    return [value]

def clean_scalar_text(value):
    if value is None or is_nullish(value):
        return ""
    return str(value).strip()

def scalarize_for_blocking(value):
    values = ensure_list(value)
    cleaned = [clean_scalar_text(v) for v in values if clean_scalar_text(v) != ""]
    return " ".join(cleaned).strip()

def preprocess_lower(x):
    if isinstance(x, list):
        x = " ".join([str(v) for v in x if not is_nullish(v)])
    if x is None or is_nullish(x):
        return ""
    return str(x).lower()

def preprocess_lower_strip(x):
    if isinstance(x, list):
        x = " ".join([str(v) for v in x if not is_nullish(v)])
    if x is None or is_nullish(x):
        return ""
    return str(x).lower().strip()

def normalize_numeric_scalar(value):
    if isinstance(value, list):
        numeric_vals = []
        for v in value:
            try:
                txt = str(v).strip()
                if txt != "":
                    numeric_vals.append(float(txt))
            except Exception:
                continue
        if not numeric_vals:
            return None
        return float(sum(numeric_vals) / len(numeric_vals))
    if value is None or is_nullish(value):
        return None
    try:
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None

def normalize_date_text(value):
    if value is None or is_nullish(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text

for df in datasets:
    for col in [
        "name",
        "artist",
        "duration",
        "release-date",
        "release-country",
        "label",
        "genre",
        "tracks_track_name",
        "tracks_track_position",
        "tracks_track_duration",
    ]:
        if col not in df.columns:
            df[col] = None

    df["tracks_track_duration_block"] = df["tracks_track_duration"].apply(scalarize_for_blocking)
    df["duration_block"] = df["duration"].apply(scalarize_for_blocking)
    df["name_block"] = df["name"].apply(scalarize_for_blocking)
    df["artist_block"] = df["artist"].apply(scalarize_for_blocking)
    df["duration"] = df["duration"].apply(normalize_numeric_scalar)
    df["release-date"] = df["release-date"].apply(normalize_date_text)

print("Performing Blocking")

blocker_1_2 = EmbeddingBlocker(
    good_dataset_name_1,
    good_dataset_name_2,
    text_cols=["name", "artist", "tracks_track_duration_block"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_1_3 = StandardBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    on=["name"],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_2_3 = EmbeddingBlocker(
    good_dataset_name_3,
    good_dataset_name_2,
    text_cols=["name", "artist", "duration_block"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=10,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

comparators_1_2 = [
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="duration",
        max_difference=10.0,
        list_strategy="average",
    ),
]

comparators_1_3 = [
    StringComparator(
        column="name",
        similarity_function="jaccard",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="release-date",
        max_days_difference=365,
    ),
    StringComparator(
        column="release-country",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaccard",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_2_3 = [
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

train_1_2 = load_csv(
    "input/gen_TrainSets/discogs_lastfm_goldstandard_blocking.csv",
    name="ground_truth_discogs_lastfm_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/gen_TrainSets/discogs_musicbrainz_goldstandard_blocking.csv",
    name="ground_truth_discogs_musicbrainz_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/gen_TrainSets/musicbrainz_lastfm_goldstandard_blocking.csv",
    name="ground_truth_musicbrainz_lastfm_train",
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
    good_dataset_name_3,
    good_dataset_name_2,
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
    threshold=0.72,
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    good_dataset_name_3,
    good_dataset_name_2,
    candidates=blocker_2_3,
    id_column="id",
    trained_classifier=best_models[2],
    threshold=0.72,
)

ml_correspondences_1_2.to_csv(
    "output/correspondences/correspondences_discogs_lastfm.csv",
    index=False,
)
ml_correspondences_1_3.to_csv(
    "output/correspondences/correspondences_discogs_musicbrainz.csv",
    index=False,
)
ml_correspondences_2_3.to_csv(
    "output/correspondences/correspondences_musicbrainz_lastfm.csv",
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

def canonical_track_name_list(value):
    values = ensure_list(value)
    out = []
    for v in values:
        if is_nullish(v):
            continue
        text = str(v).strip()
        if text != "":
            out.append(text)
    return out

def canonical_track_position_list(value):
    values = ensure_list(value)
    pairs = []
    residual = []
    for idx, v in enumerate(values):
        if is_nullish(v):
            continue
        text = str(v).strip()
        if text == "":
            continue
        try:
            num = int(float(text))
            pairs.append((idx, num))
        except Exception:
            residual.append(text)
    ordered = [str(num) for _, num in sorted(pairs, key=lambda x: (x[1], x[0]))]
    ordered.extend(residual)
    return ordered

def canonical_track_duration_list(value):
    values = ensure_list(value)
    out = []
    for v in values:
        if is_nullish(v):
            continue
        text = str(v).strip()
        if text == "":
            continue
        try:
            num = int(round(float(text)))
            out.append(str(num))
        except Exception:
            out.append(text)
    return out

def choose_preferred_track_list(inputs, preferred_sources=None, field_type="name", **kwargs):
    preferred_sources = preferred_sources or []
    cleaned = []

    for item in inputs:
        if isinstance(item, dict):
            value = item.get("value")
            source = item.get("source") or item.get("dataset") or item.get("dataset_name")
        else:
            value = item
            source = None

        if field_type == "name":
            canon = canonical_track_name_list(value)
        elif field_type == "position":
            canon = canonical_track_position_list(value)
        else:
            canon = canonical_track_duration_list(value)

        if canon:
            cleaned.append((source, canon))

    if not cleaned:
        return [], 0.0, {}

    for preferred_source in preferred_sources:
        preferred = [vals for src, vals in cleaned if src == preferred_source and vals]
        if preferred:
            return preferred[0], 1.0, {"selected_source": preferred_source}

    best_source, best_vals = max(cleaned, key=lambda x: len(x[1]))
    return best_vals, 1.0, {"selected_source": best_source}

def serialize_provenance(value):
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if value is None or is_nullish(value):
        return json.dumps([])
    text = str(value).strip()
    if text == "":
        return json.dumps([])
    return text

strategy = DataFusionStrategy("ml_fusion_strategy")

trust_map = {
    "discogs": 0.95,
    "lastfm": 0.90,
    "musicbrainz": 0.92,
}

track_name_trust_map = {
    "musicbrainz": 0.97,
    "discogs": 0.93,
    "lastfm": 0.88,
}

track_position_trust_map = {
    "musicbrainz": 0.97,
    "discogs": 0.93,
    "lastfm": 0.88,
}

track_duration_trust_map = {
    "musicbrainz": 0.97,
    "discogs": 0.91,
    "lastfm": 0.86,
}

duration_trust_map = {
    "lastfm": 0.98,
    "musicbrainz": 0.88,
    "discogs": 0.80,
}

def _pydi_safe_fuser(fn):
    """Adapt scalar custom fusers to PyDI resolver contract."""
    def _wrapped(values, **kwargs):
        try:
            try:
                result = fn(values, **kwargs)
            except TypeError:
                try:
                    result = fn(values, None, **kwargs)
                except TypeError:
                    result = fn(values)
        except Exception as e:
            fallback = values[0] if values else None
            return fallback, 0.1, {"error": str(e), "fallback": "first_value"}

        if isinstance(result, tuple) and len(result) == 3:
            return result
        return result, 1.0, {}

    return _wrapped

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("release-date", most_recent)
strategy.add_attribute_fuser("duration", prefer_higher_trust, trust_map=duration_trust_map)
strategy.add_attribute_fuser("label", prefer_higher_trust, trust_map={"discogs": 0.99, "musicbrainz": 0.70, "lastfm": 0.60})
strategy.add_attribute_fuser("genre", prefer_higher_trust, trust_map={"discogs": 0.99, "musicbrainz": 0.70, "lastfm": 0.60})
strategy.add_attribute_fuser(
    "tracks_track_name",
    _pydi_safe_fuser(choose_preferred_track_list),
    preferred_sources=["musicbrainz", "discogs", "lastfm"],
    field_type="name",
    trust_map=track_name_trust_map,
)
strategy.add_attribute_fuser(
    "tracks_track_position",
    _pydi_safe_fuser(choose_preferred_track_list),
    preferred_sources=["musicbrainz", "discogs", "lastfm"],
    field_type="position",
    trust_map=track_position_trust_map,
)
strategy.add_attribute_fuser(
    "tracks_track_duration",
    _pydi_safe_fuser(choose_preferred_track_list),
    preferred_sources=["musicbrainz", "discogs", "lastfm"],
    field_type="duration",
    trust_map=track_duration_trust_map,
)

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

for track_col in ["tracks_track_name", "tracks_track_position", "tracks_track_duration"]:
    if track_col in ml_fused_standard_blocker.columns:
        ml_fused_standard_blocker[track_col] = ml_fused_standard_blocker[track_col].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else serialize_provenance(x)
        )

for provenance_col in ["_fusion_sources", "_fusion_source_datasets", "_fusion_metadata"]:
    if provenance_col in ml_fused_standard_blocker.columns:
        ml_fused_standard_blocker[provenance_col] = ml_fused_standard_blocker[provenance_col].apply(serialize_provenance)

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

