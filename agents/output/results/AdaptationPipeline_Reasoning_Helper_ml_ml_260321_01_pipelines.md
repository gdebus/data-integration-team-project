# Pipeline Snapshots

notebook_name=AdaptationPipeline_Reasoning_Helper_ml
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=9
node_name=execute_pipeline
accuracy_score=57.60%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    StandardBlocker,
    TokenBlocker,
    EmbeddingBlocker,
    MLBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    median,
    most_recent,
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


def lower(x):
    if is_missing(x):
        return ""
    return str(x).lower()


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


def coerce_duration_scalar(value):
    if is_missing(value):
        return np.nan
    try:
        return float(value)
    except Exception:
        s = str(value).strip()
        if s == "":
            return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan


def prepare_dataset(df, dataset_name):
    df = df.copy()

    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(coerce_duration_scalar)

    df["source"] = dataset_name
    df["discogs_id"] = df["id"] if dataset_name == "discogs" else np.nan
    df["lastfm_id"] = df["id"] if dataset_name == "lastfm" else np.nan
    df["musicbrainz_id"] = df["id"] if dataset_name == "musicbrainz" else np.nan

    return df


def prepare_embedding_views(df, text_cols):
    emb_df = df.copy()
    for col in text_cols:
        if col in emb_df.columns:
            emb_df[col] = emb_df[col].apply(stringify_for_embedding)
    return emb_df


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

good_dataset_name_1 = prepare_dataset(good_dataset_name_1, "discogs")
good_dataset_name_2 = prepare_dataset(good_dataset_name_2, "lastfm")
good_dataset_name_3 = prepare_dataset(good_dataset_name_3, "musicbrainz")

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

print("Performing Blocking")

embedding_text_cols_1_2 = ["name", "artist", "tracks_track_duration"]
blocking_dataset_1_2_left = prepare_embedding_views(good_dataset_name_1, embedding_text_cols_1_2)
blocking_dataset_1_2_right = prepare_embedding_views(good_dataset_name_2, embedding_text_cols_1_2)

blocker_1_2 = EmbeddingBlocker(
    blocking_dataset_1_2_left,
    blocking_dataset_1_2_right,
    text_cols=embedding_text_cols_1_2,
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

blocker_2_3 = TokenBlocker(
    good_dataset_name_3,
    good_dataset_name_2,
    column="name",
    min_token_len=5,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

comparators_1_2 = [
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lower,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower,
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
        similarity_function="cosine",
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

ml_correspondences_1_2 = standardize_correspondence_columns(ml_correspondences_1_2)
ml_correspondences_1_3 = standardize_correspondence_columns(ml_correspondences_1_3)
ml_correspondences_2_3 = standardize_correspondence_columns(ml_correspondences_2_3)

print(f"discogs-lastfm correspondences: {len(ml_correspondences_1_2)}")
print(f"discogs-musicbrainz correspondences: {len(ml_correspondences_1_3)}")
print(f"musicbrainz-lastfm correspondences: {len(ml_correspondences_2_3)}")

ml_correspondences_1_2.to_csv(
    "output/correspondences/correspondences_discogs__lastfm.csv",
    index=False,
)
ml_correspondences_1_3.to_csv(
    "output/correspondences/correspondences_discogs__musicbrainz.csv",
    index=False,
)
ml_correspondences_2_3.to_csv(
    "output/correspondences/correspondences_musicbrainz__lastfm.csv",
    index=False,
)

print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_1_2 = clusterer.cluster(ml_correspondences_1_2)
ml_correspondences_1_3 = clusterer.cluster(ml_correspondences_1_3)
ml_correspondences_2_3 = clusterer.cluster(ml_correspondences_2_3)

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
).drop_duplicates()

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("source", union)
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", most_recent)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("duration", median)
strategy.add_attribute_fuser("label", longest_string)
strategy.add_attribute_fuser("genre", longest_string)
strategy.add_attribute_fuser("tracks_track_name", union)
strategy.add_attribute_fuser("tracks_track_position", union)
strategy.add_attribute_fuser("tracks_track_duration", union)
strategy.add_attribute_fuser("discogs_id", longest_string)
strategy.add_attribute_fuser("lastfm_id", longest_string)
strategy.add_attribute_fuser("musicbrainz_id", longest_string)

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

if "_id" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker = ml_fused_standard_blocker.drop(columns=["_id"])

if "discogs_id" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["_id"] = ml_fused_standard_blocker["discogs_id"]
else:
    ml_fused_standard_blocker["_id"] = ml_fused_standard_blocker["id"]

ml_fused_standard_blocker["_id"] = ml_fused_standard_blocker["_id"].astype("string")

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
accuracy_score=59.60%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    StandardBlocker,
    TokenBlocker,
    EmbeddingBlocker,
    MLBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    union,
    median,
    most_recent,
    prefer_higher_trust,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import numpy as np
import ast
import re
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


def repair_mojibake_text(value):
    if is_missing(value):
        return value
    s = str(value)
    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€�": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "â€": '"',
        "Â": "",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_country_name(value):
    if is_missing(value):
        return value
    s = repair_mojibake_text(value).strip()
    if s == "":
        return s
    mapping = {
        "usa": "United States",
        "u.s.a.": "United States",
        "us": "United States",
        "u.s.": "United States",
        "united states of america": "United States",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
    }
    key = s.lower()
    return mapping.get(key, s)


def lower(x):
    if is_missing(x):
        return ""
    return repair_mojibake_text(x).lower()


def lower_strip(x):
    if is_missing(x):
        return ""
    return repair_mojibake_text(x).lower().strip()


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
            s = repair_mojibake_text(item)
            s = str(s).strip()
            if s:
                cleaned.append(s)
        return " ".join(cleaned)
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(repair_mojibake_text(value)).strip()


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


def coerce_duration_scalar(value):
    if is_missing(value):
        return np.nan
    try:
        return float(value)
    except Exception:
        s = str(value).strip()
        if s == "":
            return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan


def ensure_list(value):
    if is_missing(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        except Exception:
            pass
        return [s]
    return [value]


def clean_list_values(values, numeric=False):
    out = []
    seen = set()
    for item in ensure_list(values):
        if is_missing(item):
            continue
        val = repair_mojibake_text(item)
        val = str(val).strip()
        if val == "" or val == "[]":
            continue
        if numeric:
            try:
                val_num = str(int(float(val)))
            except Exception:
                val_num = val
            key = val_num
            val = val_num
        else:
            key = val.lower()
        if key not in seen:
            seen.add(key)
            out.append(val)
    return out


def prepare_dataset(df, dataset_name):
    df = df.copy()

    text_columns = ["name", "artist", "label", "genre"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(repair_mojibake_text)

    if "release-country" in df.columns:
        df["release-country"] = df["release-country"].apply(normalize_country_name)

    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(coerce_duration_scalar)

    for col in ["tracks_track_name", "tracks_track_position", "tracks_track_duration"]:
        if col in df.columns:
            numeric = col != "tracks_track_name"
            df[col] = df[col].apply(lambda x: clean_list_values(x, numeric=numeric))

    if "tracks_track_name" in df.columns:
        df["tracks_track_name_text"] = df["tracks_track_name"].apply(
            lambda x: " | ".join(clean_list_values(x, numeric=False))
        )
    if "tracks_track_position" in df.columns:
        df["tracks_track_position_text"] = df["tracks_track_position"].apply(
            lambda x: " | ".join(clean_list_values(x, numeric=True))
        )
    if "tracks_track_duration" in df.columns:
        df["tracks_track_duration_text"] = df["tracks_track_duration"].apply(
            lambda x: " | ".join(clean_list_values(x, numeric=True))
        )

    df["source"] = dataset_name
    df["discogs_id"] = df["id"] if dataset_name == "discogs" else np.nan
    df["lastfm_id"] = df["id"] if dataset_name == "lastfm" else np.nan
    df["musicbrainz_id"] = df["id"] if dataset_name == "musicbrainz" else np.nan

    return df


def prepare_embedding_views(df, text_cols):
    emb_df = df.copy()
    for col in text_cols:
        if col in emb_df.columns:
            emb_df[col] = emb_df[col].apply(stringify_for_embedding)
    return emb_df


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

good_dataset_name_1 = prepare_dataset(good_dataset_name_1, "discogs")
good_dataset_name_2 = prepare_dataset(good_dataset_name_2, "lastfm")
good_dataset_name_3 = prepare_dataset(good_dataset_name_3, "musicbrainz")

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

print("Performing Blocking")

embedding_text_cols_1_2 = ["name", "artist", "tracks_track_duration_text"]
blocking_dataset_1_2_left = prepare_embedding_views(good_dataset_name_1, embedding_text_cols_1_2)
blocking_dataset_1_2_right = prepare_embedding_views(good_dataset_name_2, embedding_text_cols_1_2)

blocker_1_2 = EmbeddingBlocker(
    blocking_dataset_1_2_left,
    blocking_dataset_1_2_right,
    text_cols=embedding_text_cols_1_2,
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

blocker_2_3 = TokenBlocker(
    good_dataset_name_3,
    good_dataset_name_2,
    column="name",
    min_token_len=5,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

comparators_1_2 = [
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lower,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower,
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
        similarity_function="cosine",
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

ml_correspondences_1_2 = standardize_correspondence_columns(ml_correspondences_1_2)
ml_correspondences_1_3 = standardize_correspondence_columns(ml_correspondences_1_3)
ml_correspondences_2_3 = standardize_correspondence_columns(ml_correspondences_2_3)

print(f"discogs-lastfm correspondences: {len(ml_correspondences_1_2)}")
print(f"discogs-musicbrainz correspondences: {len(ml_correspondences_1_3)}")
print(f"musicbrainz-lastfm correspondences: {len(ml_correspondences_2_3)}")

ml_correspondences_1_2.to_csv(
    "output/correspondences/correspondences_discogs__lastfm.csv",
    index=False,
)
ml_correspondences_1_3.to_csv(
    "output/correspondences/correspondences_discogs__musicbrainz.csv",
    index=False,
)
ml_correspondences_2_3.to_csv(
    "output/correspondences/correspondences_musicbrainz__lastfm.csv",
    index=False,
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

print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_1_2 = clusterer.cluster(ml_correspondences_1_2)
ml_correspondences_1_3 = clusterer.cluster(ml_correspondences_1_3)
ml_correspondences_2_3 = clusterer.cluster(ml_correspondences_2_3)

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True,
).drop_duplicates()

strategy = DataFusionStrategy("ml_fusion_strategy")

trust_map_general = {
    "discogs": 0.95,
    "musicbrainz": 0.90,
    "lastfm": 0.70,
}
trust_map_duration = {
    "lastfm": 1.00,
    "musicbrainz": 0.85,
    "discogs": 0.60,
}
trust_map_track_lists = {
    "musicbrainz": 1.00,
    "discogs": 0.90,
    "lastfm": 0.40,
}
trust_map_discogs_only = {
    "discogs": 1.00,
    "musicbrainz": 0.10,
    "lastfm": 0.10,
}
trust_map_country = {
    "musicbrainz": 1.00,
    "discogs": 0.90,
    "lastfm": 0.10,
}

strategy.add_attribute_fuser("source", union)
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", most_recent)
strategy.add_attribute_fuser("release-country", prefer_higher_trust, trust_map=trust_map_country)
strategy.add_attribute_fuser("duration", prefer_higher_trust, trust_map=trust_map_duration)
strategy.add_attribute_fuser("label", prefer_higher_trust, trust_map=trust_map_discogs_only)
strategy.add_attribute_fuser("genre", prefer_higher_trust, trust_map=trust_map_discogs_only)
strategy.add_attribute_fuser("tracks_track_name", prefer_higher_trust, trust_map=trust_map_track_lists)
strategy.add_attribute_fuser("tracks_track_position", prefer_higher_trust, trust_map=trust_map_track_lists)
strategy.add_attribute_fuser("tracks_track_duration", prefer_higher_trust, trust_map=trust_map_track_lists)
strategy.add_attribute_fuser("discogs_id", prefer_higher_trust, trust_map=trust_map_discogs_only)
strategy.add_attribute_fuser("lastfm_id", prefer_higher_trust, trust_map={"lastfm": 1.0, "discogs": 0.1, "musicbrainz": 0.1})
strategy.add_attribute_fuser("musicbrainz_id", prefer_higher_trust, trust_map={"musicbrainz": 1.0, "discogs": 0.1, "lastfm": 0.1})

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

if "_id" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker = ml_fused_standard_blocker.drop(columns=["_id"])

for col in ["name", "artist", "label", "genre"]:
    if col in ml_fused_standard_blocker.columns:
        ml_fused_standard_blocker[col] = ml_fused_standard_blocker[col].apply(repair_mojibake_text)

if "release-country" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["release-country"] = ml_fused_standard_blocker["release-country"].apply(normalize_country_name)

for col in ["tracks_track_name", "tracks_track_position", "tracks_track_duration"]:
    if col in ml_fused_standard_blocker.columns:
        numeric = col != "tracks_track_name"
        ml_fused_standard_blocker[col] = ml_fused_standard_blocker[col].apply(lambda x: clean_list_values(x, numeric=numeric))

if "discogs_id" in ml_fused_standard_blocker.columns and ml_fused_standard_blocker["discogs_id"].notna().any():
    ml_fused_standard_blocker["_id"] = ml_fused_standard_blocker["discogs_id"]
elif "musicbrainz_id" in ml_fused_standard_blocker.columns and ml_fused_standard_blocker["musicbrainz_id"].notna().any():
    ml_fused_standard_blocker["_id"] = ml_fused_standard_blocker["musicbrainz_id"]
elif "lastfm_id" in ml_fused_standard_blocker.columns and ml_fused_standard_blocker["lastfm_id"].notna().any():
    ml_fused_standard_blocker["_id"] = ml_fused_standard_blocker["lastfm_id"]
else:
    ml_fused_standard_blocker["_id"] = ml_fused_standard_blocker["id"]

ml_fused_standard_blocker["_id"] = ml_fused_standard_blocker["_id"].astype("string")

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
accuracy_score=58.40%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import (
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
    StandardBlocker,
    TokenBlocker,
    EmbeddingBlocker,
    MLBasedMatcher,
    MaximumBipartiteMatching,
)
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    median,
    most_recent,
    prefer_higher_trust,
    union,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import numpy as np
import ast
import re
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


def repair_mojibake_text(value):
    if is_missing(value):
        return value
    s = str(value)
    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€�": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "â€": '"',
        "Â": "",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = s.replace("´", "'").replace("`", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_music_text(value):
    if is_missing(value):
        return value
    s = repair_mojibake_text(value)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_artist_text(value):
    if is_missing(value):
        return value
    s = clean_music_text(value)
    s = re.sub(r"\b(feat\.?|featuring|ft\.?)\b.*$", "", s, flags=re.IGNORECASE).strip(" -;/,")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_country_name(value):
    if is_missing(value):
        return value
    s = clean_music_text(value)
    if s == "":
        return s

    region_like = {
        "europe": "",
        "worldwide": "",
        "international": "",
        "xw": "",
        "[worldwide]": "",
    }
    key = s.lower()
    if key in region_like:
        return region_like[key]

    mapping = {
        "usa": "United States",
        "u.s.a.": "United States",
        "us": "United States",
        "u.s.": "United States",
        "united states of america": "United States",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
        "great britain": "United Kingdom",
        "england": "United Kingdom",
    }
    return mapping.get(key, s)


def lower(x):
    if is_missing(x):
        return ""
    return clean_music_text(x).lower()


def lower_strip(x):
    if is_missing(x):
        return ""
    return clean_music_text(x).lower().strip()


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
            s = clean_music_text(item)
            s = str(s).strip()
            if s:
                cleaned.append(s)
        return " ".join(cleaned)
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(clean_music_text(value)).strip()


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


def coerce_duration_scalar(value):
    if is_missing(value):
        return np.nan
    try:
        return float(value)
    except Exception:
        s = str(value).strip()
        if s == "":
            return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan


def ensure_list(value):
    if is_missing(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        except Exception:
            pass
        return [s]
    return [value]


def clean_list_values(values, numeric=False):
    out = []
    seen = set()
    for item in ensure_list(values):
        if is_missing(item):
            continue
        val = clean_music_text(item)
        val = str(val).strip()
        if val == "" or val == "[]":
            continue
        if numeric:
            try:
                val_num = str(int(float(val)))
            except Exception:
                val_num = val
            key = val_num
            val = val_num
        else:
            key = val.lower()
        if key not in seen:
            seen.add(key)
            out.append(val)
    return out


def track_positions_are_contiguous(values):
    vals = clean_list_values(values, numeric=True)
    if not vals:
        return False
    nums = []
    for v in vals:
        if not str(v).isdigit():
            return False
        nums.append(int(v))
    if not nums:
        return False
    return nums == list(range(1, len(nums) + 1))


def choose_best_track_list(inputs, preferred_sources=None, require_contiguous=False, **kwargs):
    preferred_sources = preferred_sources or []
    candidates = []

    for item in inputs:
        if isinstance(item, dict):
            value = item.get("value")
            source = item.get("source") or item.get("dataset") or item.get("dataset_name")
        else:
            value = item
            source = None

        cleaned = clean_list_values(value, numeric=require_contiguous or False)
        if not cleaned:
            continue

        score = 0
        if source in preferred_sources:
            score += (len(preferred_sources) - preferred_sources.index(source)) * 1000

        if require_contiguous and track_positions_are_contiguous(cleaned):
            score += 200

        score += len(cleaned)
        candidates.append((score, source, cleaned))

    if not candidates:
        return [], 0.0, {}

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_source, best_value = candidates[0]
    return best_value, 1.0, {"selected_source": best_source, "score": best_score}


def choose_best_track_names(inputs, **kwargs):
    return choose_best_track_list(
        inputs,
        preferred_sources=["musicbrainz", "discogs", "lastfm"],
        require_contiguous=False,
        **kwargs,
    )


def choose_best_track_positions(inputs, **kwargs):
    return choose_best_track_list(
        inputs,
        preferred_sources=["musicbrainz", "discogs", "lastfm"],
        require_contiguous=True,
        **kwargs,
    )


def choose_best_track_durations(inputs, **kwargs):
    return choose_best_track_list(
        inputs,
        preferred_sources=["musicbrainz", "discogs", "lastfm"],
        require_contiguous=False,
        **kwargs,
    )


def prepare_dataset(df, dataset_name):
    df = df.copy()

    if "name" in df.columns:
        df["name"] = df["name"].apply(clean_music_text)
    if "artist" in df.columns:
        df["artist"] = df["artist"].apply(clean_artist_text)
    if "label" in df.columns:
        df["label"] = df["label"].apply(clean_music_text)
    if "genre" in df.columns:
        df["genre"] = df["genre"].apply(clean_music_text)

    if "release-country" in df.columns:
        df["release-country"] = df["release-country"].apply(normalize_country_name)

    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(coerce_duration_scalar)

    for col in ["tracks_track_name", "tracks_track_position", "tracks_track_duration"]:
        if col in df.columns:
            numeric = col != "tracks_track_name"
            df[col] = df[col].apply(lambda x: clean_list_values(x, numeric=numeric))

    if "tracks_track_name" in df.columns:
        df["tracks_track_name"] = df["tracks_track_name"].apply(
            lambda vals: [clean_music_text(v) for v in clean_list_values(vals, numeric=False)]
        )
        df["tracks_track_name_text"] = df["tracks_track_name"].apply(lambda x: " | ".join(x))

    if "tracks_track_position" in df.columns:
        df["tracks_track_position"] = df["tracks_track_position"].apply(lambda x: clean_list_values(x, numeric=True))
        df["tracks_track_position_text"] = df["tracks_track_position"].apply(lambda x: " | ".join(x))

    if "tracks_track_duration" in df.columns:
        df["tracks_track_duration"] = df["tracks_track_duration"].apply(lambda x: clean_list_values(x, numeric=True))
        df["tracks_track_duration_text"] = df["tracks_track_duration"].apply(lambda x: " | ".join(x))

    df["source"] = dataset_name
    df["discogs_id"] = df["id"] if dataset_name == "discogs" else np.nan
    df["lastfm_id"] = df["id"] if dataset_name == "lastfm" else np.nan
    df["musicbrainz_id"] = df["id"] if dataset_name == "musicbrainz" else np.nan

    return df


def prepare_embedding_views(df, text_cols):
    emb_df = df.copy()
    for col in text_cols:
        if col in emb_df.columns:
            emb_df[col] = emb_df[col].apply(stringify_for_embedding)
    return emb_df


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

good_dataset_name_1 = prepare_dataset(good_dataset_name_1, "discogs")
good_dataset_name_2 = prepare_dataset(good_dataset_name_2, "lastfm")
good_dataset_name_3 = prepare_dataset(good_dataset_name_3, "musicbrainz")

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

print("Performing Blocking")

embedding_text_cols_1_2 = ["name", "artist", "tracks_track_duration_text"]
blocking_dataset_1_2_left = prepare_embedding_views(good_dataset_name_1, embedding_text_cols_1_2)
blocking_dataset_1_2_right = prepare_embedding_views(good_dataset_name_2, embedding_text_cols_1_2)

blocker_1_2 = EmbeddingBlocker(
    blocking_dataset_1_2_left,
    blocking_dataset_1_2_right,
    text_cols=embedding_text_cols_1_2,
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

blocker_2_3 = TokenBlocker(
    good_dataset_name_3,
    good_dataset_name_2,
    column="name",
    min_token_len=5,
    id_column="id",
    output_dir="output/blocking-evaluation",
)

comparators_1_2 = [
    StringComparator(
        column="name",
        similarity_function="cosine",
        preprocess=lower,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="artist",
        similarity_function="jaro_winkler",
        preprocess=lower,
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
        similarity_function="cosine",
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

ml_correspondences_1_2 = standardize_correspondence_columns(ml_correspondences_1_2)
ml_correspondences_1_3 = standardize_correspondence_columns(ml_correspondences_1_3)
ml_correspondences_2_3 = standardize_correspondence_columns(ml_correspondences_2_3)

print(f"discogs-lastfm correspondences: {len(ml_correspondences_1_2)}")
print(f"discogs-musicbrainz correspondences: {len(ml_correspondences_1_3)}")
print(f"musicbrainz-lastfm correspondences: {len(ml_correspondences_2_3)}")

ml_correspondences_1_2.to_csv(
    "output/correspondences/correspondences_discogs__lastfm.csv",
    index=False,
)
ml_correspondences_1_3.to_csv(
    "output/correspondences/correspondences_discogs__musicbrainz.csv",
    index=False,
)
ml_correspondences_2_3.to_csv(
    "output/correspondences/correspondences_musicbrainz__lastfm.csv",
    index=False,
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

trust_map_general = {
    "discogs": 0.95,
    "musicbrainz": 0.90,
    "lastfm": 0.70,
}
trust_map_duration = {
    "lastfm": 1.00,
    "musicbrainz": 0.85,
    "discogs": 0.60,
}
trust_map_country = {
    "musicbrainz": 1.00,
    "discogs": 0.90,
    "lastfm": 0.10,
}
trust_map_discogs_label_genre = {
    "discogs": 1.00,
    "musicbrainz": 0.20,
    "lastfm": 0.10,
}
trust_map_discogs_id = {
    "discogs": 1.0,
    "musicbrainz": 0.1,
    "lastfm": 0.1,
}
trust_map_lastfm_id = {
    "lastfm": 1.0,
    "discogs": 0.1,
    "musicbrainz": 0.1,
}
trust_map_musicbrainz_id = {
    "musicbrainz": 1.0,
    "discogs": 0.1,
    "lastfm": 0.1,
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

strategy.add_attribute_fuser("source", union)
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", prefer_higher_trust, trust_map=trust_map_general)
strategy.add_attribute_fuser("release-date", most_recent)
strategy.add_attribute_fuser("release-country", prefer_higher_trust, trust_map=trust_map_country)
strategy.add_attribute_fuser("duration", prefer_higher_trust, trust_map=trust_map_duration)
strategy.add_attribute_fuser("label", prefer_higher_trust, trust_map=trust_map_discogs_label_genre)
strategy.add_attribute_fuser("genre", prefer_higher_trust, trust_map=trust_map_discogs_label_genre)
strategy.add_attribute_fuser("tracks_track_name", _pydi_safe_fuser(choose_best_track_names))
strategy.add_attribute_fuser("tracks_track_position", _pydi_safe_fuser(choose_best_track_positions))
strategy.add_attribute_fuser("tracks_track_duration", _pydi_safe_fuser(choose_best_track_durations))
strategy.add_attribute_fuser("discogs_id", prefer_higher_trust, trust_map=trust_map_discogs_id)
strategy.add_attribute_fuser("lastfm_id", prefer_higher_trust, trust_map=trust_map_lastfm_id)
strategy.add_attribute_fuser("musicbrainz_id", prefer_higher_trust, trust_map=trust_map_musicbrainz_id)

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

if "_id" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker = ml_fused_standard_blocker.drop(columns=["_id"])

for col in ["name", "artist", "label", "genre"]:
    if col in ml_fused_standard_blocker.columns:
        if col == "artist":
            ml_fused_standard_blocker[col] = ml_fused_standard_blocker[col].apply(clean_artist_text)
        else:
            ml_fused_standard_blocker[col] = ml_fused_standard_blocker[col].apply(clean_music_text)

if "release-country" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["release-country"] = ml_fused_standard_blocker["release-country"].apply(normalize_country_name)

if "duration" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker["duration"] = ml_fused_standard_blocker["duration"].apply(coerce_duration_scalar)

for col in ["tracks_track_name", "tracks_track_position", "tracks_track_duration"]:
    if col in ml_fused_standard_blocker.columns:
        numeric = col != "tracks_track_name"
        ml_fused_standard_blocker[col] = ml_fused_standard_blocker[col].apply(lambda x: clean_list_values(x, numeric=numeric))

if "discogs_id" in ml_fused_standard_blocker.columns:
    discogs_mask = ml_fused_standard_blocker["discogs_id"].notna() & (ml_fused_standard_blocker["discogs_id"].astype(str).str.strip() != "")
else:
    discogs_mask = pd.Series(False, index=ml_fused_standard_blocker.index)

if "musicbrainz_id" in ml_fused_standard_blocker.columns:
    musicbrainz_mask = ml_fused_standard_blocker["musicbrainz_id"].notna() & (ml_fused_standard_blocker["musicbrainz_id"].astype(str).str.strip() != "")
else:
    musicbrainz_mask = pd.Series(False, index=ml_fused_standard_blocker.index)

if "lastfm_id" in ml_fused_standard_blocker.columns:
    lastfm_mask = ml_fused_standard_blocker["lastfm_id"].notna() & (ml_fused_standard_blocker["lastfm_id"].astype(str).str.strip() != "")
else:
    lastfm_mask = pd.Series(False, index=ml_fused_standard_blocker.index)

ml_fused_standard_blocker["_id"] = ml_fused_standard_blocker["id"].astype("string")
if "lastfm_id" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker.loc[lastfm_mask, "_id"] = ml_fused_standard_blocker.loc[lastfm_mask, "lastfm_id"].astype("string")
if "musicbrainz_id" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker.loc[musicbrainz_mask, "_id"] = ml_fused_standard_blocker.loc[musicbrainz_mask, "musicbrainz_id"].astype("string")
if "discogs_id" in ml_fused_standard_blocker.columns:
    ml_fused_standard_blocker.loc[discogs_mask, "_id"] = ml_fused_standard_blocker.loc[discogs_mask, "discogs_id"].astype("string")

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

