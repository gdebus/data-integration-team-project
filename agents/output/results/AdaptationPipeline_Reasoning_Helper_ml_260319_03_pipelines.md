# Pipeline Snapshots

notebook_name=AdaptationPipeline_nblazek_setcount_guardline
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=7
node_name=execute_pipeline
accuracy_score=67.86%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import MLBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    longest_string,
    median,
    union,
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
    DATA_DIR + "output/normalization/attempt_1/amazon_small.csv",
    name="amazon_small",
)

good_dataset_name_2 = load_csv(
    DATA_DIR + "output/normalization/attempt_1/goodreads_small.csv",
    name="goodreads_small",
)

good_dataset_name_3 = load_csv(
    DATA_DIR + "output/normalization/attempt_1/metabooks_small.csv",
    name="metabooks_small",
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

blocker_1_2 = EmbeddingBlocker(
    good_dataset_name_2,
    good_dataset_name_1,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_1_3 = EmbeddingBlocker(
    good_dataset_name_3,
    good_dataset_name_1,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_2_3 = EmbeddingBlocker(
    good_dataset_name_3,
    good_dataset_name_2,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

comparators_1_2 = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=365,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=preprocess_lower,
        list_strategy="concatenate",
    ),
]

comparators_1_3 = [
    StringComparator(
        column="title",
        similarity_function="cosine",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_2_3 = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=preprocess_lower_strip,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="page_count",
        max_difference=10.0,
        list_strategy="average",
    ),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

train_1_2 = load_csv(
    "input/datasets/books/testsets/goodreads_2_amazon.csv",
    name="ground_truth_goodreads_amazon_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/datasets/books/testsets/metabooks_2_amazon.csv",
    name="ground_truth_metabooks_amazon_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/books/testsets/metabooks_2_goodreads.csv",
    name="ground_truth_metabooks_goodreads_train",
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
    good_dataset_name_2,
    good_dataset_name_1,
    to_pair_ids(train_1_2),
    labels=label_1_2,
    id_column="id",
)

train_1_3_features = feature_extractor_1_3.create_features(
    good_dataset_name_3,
    good_dataset_name_1,
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
    good_dataset_name_2,
    good_dataset_name_1,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
    threshold=0.78,
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    good_dataset_name_3,
    good_dataset_name_1,
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
    threshold=0.78,
)

ml_correspondences_1_2.to_csv(
    "output/correspondences/correspondences_goodreads_amazon.csv",
    index=False,
)
ml_correspondences_1_3.to_csv(
    "output/correspondences/correspondences_metabooks_amazon.csv",
    index=False,
)
ml_correspondences_2_3.to_csv(
    "output/correspondences/correspondences_metabooks_goodreads.csv",
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

strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publish_year", median)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("rating", median)
strategy.add_attribute_fuser("numratings", median)
strategy.add_attribute_fuser("language", longest_string)
strategy.add_attribute_fuser("genres", union)
strategy.add_attribute_fuser("bookformat", longest_string)
strategy.add_attribute_fuser("edition", longest_string)
strategy.add_attribute_fuser("page_count", median)
strategy.add_attribute_fuser("price", median)

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
accuracy_score=67.86%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import MLBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    union,
    prefer_higher_trust,
)
from PyDI.entitymatching import MaximumBipartiteMatching

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import os
import re
import ast
import math
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
    DATA_DIR + "output/normalization/attempt_2/amazon_small.csv",
    name="amazon_small",
)

good_dataset_name_2 = load_csv(
    DATA_DIR + "output/normalization/attempt_2/goodreads_small.csv",
    name="goodreads_small",
)

good_dataset_name_3 = load_csv(
    DATA_DIR + "output/normalization/attempt_2/metabooks_small.csv",
    name="metabooks_small",
)

good_dataset_name_1["id"] = good_dataset_name_1["id"]
good_dataset_name_2["id"] = good_dataset_name_2["id"]
good_dataset_name_3["id"] = good_dataset_name_3["id"]

def is_nullish(x):
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "[]", "[ ]"}

def safe_listify(value):
    if is_nullish(value):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if not is_nullish(v)]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if not is_nullish(v)]
    s = str(value).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            out = []
            for item in parsed:
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        if not is_nullish(sub):
                            out.append(str(sub).strip())
                elif not is_nullish(item):
                    out.append(str(item).strip())
            return out
    except Exception:
        pass
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    return [s] if s else []

def flatten_text_tokens(value):
    values = safe_listify(value)
    out = []
    for item in values:
        if is_nullish(item):
            continue
        item = str(item).strip()
        if not item:
            continue
        if "," in item:
            parts = [p.strip() for p in item.split(",") if p.strip()]
            out.extend(parts)
        else:
            out.append(item)
    deduped = []
    seen = set()
    for item in out:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped

def normalize_space_text(s):
    return re.sub(r"\s+", " ", str(s)).strip()

def clean_author_piece(s):
    s = normalize_space_text(s)
    s = re.sub(
        r"\s*\((goodreads author|editor|illustrator|foreword|translator|adaptation|adapted by|designed by|introduction|introduction by|afterword|afterword by|preface|preface by|contributor|contributors?)\)\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\s+", " ", s).strip(" ,;/")
    return s

def canonical_author_value(value):
    authors = flatten_text_tokens(value)
    cleaned = []
    for a in authors:
        a = clean_author_piece(a)
        if a:
            cleaned.append(a)
    if not cleaned:
        return None
    return cleaned[0]

def clean_title_value(value):
    if is_nullish(value):
        return None
    s = normalize_space_text(value)
    s = re.sub(r"\s+\((p\.s\.|thoughtless|trophy newbery|penguin classics|novels of ancient egypt|sean dillon|charmed)\)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+\([^)]*(series|classics|edition|vol\.|volume|book\s+\d+)[^)]*\)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*:\s*a novel\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" /:-")
    return s if s else None

def clean_publisher_value(value):
    if is_nullish(value):
        return None
    s = normalize_space_text(value)
    s = re.sub(r"\s*\((trade division|ny|usa|us|uk|ca)\)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bbooks usa\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bgroup\(ca\)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*/\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;/")
    return s if s else None

def clean_language_value(value):
    if is_nullish(value):
        return None
    vals = flatten_text_tokens(value)
    if not vals:
        s = normalize_space_text(value)
        vals = [p.strip() for p in re.split(r"[,;/]", s) if p.strip()]
    return vals[0] if vals else None

def preprocess_lower(x):
    if is_nullish(x):
        return ""
    return str(x).lower()

def preprocess_lower_strip(x):
    if is_nullish(x):
        return ""
    return str(x).lower().strip()

def preprocess_title_match(x):
    v = clean_title_value(x)
    return "" if v is None else v.lower().strip()

def preprocess_author_match(x):
    v = canonical_author_value(x)
    return "" if v is None else v.lower().strip()

def preprocess_publisher_match(x):
    v = clean_publisher_value(x)
    return "" if v is None else v.lower().strip()

def preprocess_language_match(x):
    v = clean_language_value(x)
    return "" if v is None else v.lower().strip()

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

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    if "author" in df.columns:
        df["author_primary"] = df["author"].apply(canonical_author_value)
    if "title" in df.columns:
        df["title_clean"] = df["title"].apply(clean_title_value)
    if "publisher" in df.columns:
        df["publisher_clean"] = df["publisher"].apply(clean_publisher_value)
    if "language" in df.columns:
        df["language_primary"] = df["language"].apply(clean_language_value)
    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(flatten_text_tokens)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

print("Performing Blocking")

blocker_1_2 = EmbeddingBlocker(
    good_dataset_name_2,
    good_dataset_name_1,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_1_3 = EmbeddingBlocker(
    good_dataset_name_3,
    good_dataset_name_1,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_2_3 = EmbeddingBlocker(
    good_dataset_name_3,
    good_dataset_name_2,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

comparators_1_2 = [
    StringComparator(
        column="title_clean",
        similarity_function="jaro_winkler",
        preprocess=preprocess_title_match,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author_primary",
        similarity_function="jaro_winkler",
        preprocess=preprocess_author_match,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=365,
    ),
    StringComparator(
        column="publisher_clean",
        similarity_function="cosine",
        preprocess=preprocess_publisher_match,
        list_strategy="concatenate",
    ),
]

comparators_1_3 = [
    StringComparator(
        column="title_clean",
        similarity_function="cosine",
        preprocess=preprocess_title_match,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author_primary",
        similarity_function="jaro_winkler",
        preprocess=preprocess_author_match,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher_clean",
        similarity_function="jaro_winkler",
        preprocess=preprocess_publisher_match,
        list_strategy="concatenate",
    ),
]

comparators_2_3 = [
    StringComparator(
        column="title_clean",
        similarity_function="jaro_winkler",
        preprocess=preprocess_title_match,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author_primary",
        similarity_function="jaro_winkler",
        preprocess=preprocess_author_match,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher_clean",
        similarity_function="jaro_winkler",
        preprocess=preprocess_publisher_match,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="page_count",
        max_difference=10.0,
        list_strategy="average",
    ),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

train_1_2 = load_csv(
    "input/datasets/books/testsets/goodreads_2_amazon.csv",
    name="ground_truth_goodreads_amazon_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/datasets/books/testsets/metabooks_2_amazon.csv",
    name="ground_truth_metabooks_amazon_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/books/testsets/metabooks_2_goodreads.csv",
    name="ground_truth_metabooks_goodreads_train",
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
    good_dataset_name_2,
    good_dataset_name_1,
    to_pair_ids(train_1_2),
    labels=label_1_2,
    id_column="id",
)

train_1_3_features = feature_extractor_1_3.create_features(
    good_dataset_name_3,
    good_dataset_name_1,
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
    good_dataset_name_2,
    good_dataset_name_1,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
    threshold=0.78,
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    good_dataset_name_3,
    good_dataset_name_1,
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
    threshold=0.78,
)

ml_correspondences_1_2.to_csv(
    "output/correspondences/correspondences_goodreads_amazon.csv",
    index=False,
)
ml_correspondences_1_3.to_csv(
    "output/correspondences/correspondences_metabooks_amazon.csv",
    index=False,
)
ml_correspondences_2_3.to_csv(
    "output/correspondences/correspondences_metabooks_goodreads.csv",
    index=False,
)

# also write expected artifact aliases for diagnostics
ml_correspondences_1_2.to_csv(
    "output/correspondences/correspondences_amazon_small__goodreads_small.csv",
    index=False,
)
ml_correspondences_1_3.to_csv(
    "output/correspondences/correspondences_amazon_small__metabooks_small.csv",
    index=False,
)
ml_correspondences_2_3.to_csv(
    "output/correspondences/correspondences_goodreads_small__metabooks_small.csv",
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

def clean_string_for_compare(value):
    if is_nullish(value):
        return None
    return normalize_space_text(value).casefold()

def choose_best_string(inputs, cleaners, prefer_shorter=True, **kwargs):
    candidates = []
    for item in inputs:
        if isinstance(item, dict):
            value = item.get("value")
            source = item.get("source") or item.get("dataset") or item.get("source_dataset")
        else:
            value = item
            source = None
        if is_nullish(value):
            continue
        cleaned_value = value
        for cleaner in cleaners:
            cleaned_value = cleaner(cleaned_value)
            if cleaned_value is None:
                break
        if cleaned_value is None or is_nullish(cleaned_value):
            continue
        candidates.append(
            {
                "value": cleaned_value,
                "source": source,
                "norm": clean_string_for_compare(cleaned_value),
                "len": len(str(cleaned_value)),
            }
        )

    if not candidates:
        return None, 0.0, {}

    unique = []
    seen = set()
    for c in candidates:
        key = c["norm"]
        if key not in seen:
            seen.add(key)
            unique.append(c)

    trust_order = {
        "metabooks_small": 3.0,
        "goodreads_small": 2.0,
        "amazon_small": 1.0,
        None: 0.0,
    }

    unique = sorted(
        unique,
        key=lambda x: (
            -trust_order.get(x["source"], 0.0),
            x["len"] if prefer_shorter else -x["len"],
        ),
    )
    best = unique[0]
    return best["value"], 1.0, {"selected_source": best["source"], "candidates": len(unique)}

def title_fuser(inputs, **kwargs):
    return choose_best_string(inputs, [clean_title_value], prefer_shorter=True, **kwargs)

def author_fuser(inputs, **kwargs):
    return choose_best_string(inputs, [canonical_author_value], prefer_shorter=True, **kwargs)

def publisher_fuser(inputs, **kwargs):
    return choose_best_string(inputs, [clean_publisher_value], prefer_shorter=True, **kwargs)

def language_fuser(inputs, **kwargs):
    return choose_best_string(inputs, [clean_language_value], prefer_shorter=True, **kwargs)

def observed_numeric_prefer_trust(inputs, round_int=False, **kwargs):
    trust_order = {
        "metabooks_small": 3.0,
        "goodreads_small": 2.0,
        "amazon_small": 1.0,
        None: 0.0,
    }
    candidates = []
    for item in inputs:
        if isinstance(item, dict):
            value = item.get("value")
            source = item.get("source") or item.get("dataset") or item.get("source_dataset")
        else:
            value = item
            source = None
        if is_nullish(value):
            continue
        try:
            num = float(value)
            if math.isnan(num):
                continue
            candidates.append((source, num))
        except Exception:
            continue

    if not candidates:
        return None, 0.0, {}

    candidates = sorted(candidates, key=lambda x: (-trust_order.get(x[0], 0.0), x[1]))
    source, num = candidates[0]
    if round_int:
        num = int(round(num))
    return num, 1.0, {"selected_source": source}

def page_count_fuser(inputs, **kwargs):
    return observed_numeric_prefer_trust(inputs, round_int=True, **kwargs)

def publish_year_fuser(inputs, **kwargs):
    return observed_numeric_prefer_trust(inputs, round_int=True, **kwargs)

trust_map = {
    "amazon_small": 0.70,
    "goodreads_small": 0.85,
    "metabooks_small": 1.00,
}

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

strategy.add_attribute_fuser("title", _pydi_safe_fuser(title_fuser))
strategy.add_attribute_fuser("author", _pydi_safe_fuser(author_fuser))
strategy.add_attribute_fuser("publish_year", _pydi_safe_fuser(publish_year_fuser))
strategy.add_attribute_fuser("publisher", _pydi_safe_fuser(publisher_fuser))
strategy.add_attribute_fuser("rating", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("numratings", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("language", _pydi_safe_fuser(language_fuser))
strategy.add_attribute_fuser("genres", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("bookformat", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("edition", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("page_count", _pydi_safe_fuser(page_count_fuser))
strategy.add_attribute_fuser("price", prefer_higher_trust, trust_map=trust_map)

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

if "_fusion_source_datasets" in ml_fused_standard_blocker.columns and "id" in ml_fused_standard_blocker.columns:
    def preserve_source_ids(row):
        fused_id = row["id"]
        srcs = row.get("_fusion_source_datasets")
        if is_nullish(srcs):
            return fused_id
        return fused_id
    ml_fused_standard_blocker["id"] = ml_fused_standard_blocker.apply(preserve_source_ids, axis=1)

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
accuracy_score=72.14%
------------------------------------------------------------

```python
from PyDI.io import load_csv
from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import EmbeddingBlocker
from PyDI.entitymatching import MLBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEngine,
    prefer_higher_trust,
)
from PyDI.entitymatching import MaximumBipartiteMatching

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import os
import re
import ast
import math
import numpy as np
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

DATA_DIR = ""

os.makedirs("output/correspondences", exist_ok=True)
os.makedirs("output/data_fusion", exist_ok=True)
os.makedirs("output/blocking-evaluation", exist_ok=True)

good_dataset_name_1 = load_csv(
    DATA_DIR + "output/normalization/attempt_3/amazon_small.csv",
    name="amazon_small",
)

good_dataset_name_2 = load_csv(
    DATA_DIR + "output/normalization/attempt_3/goodreads_small.csv",
    name="goodreads_small",
)

good_dataset_name_3 = load_csv(
    DATA_DIR + "output/normalization/attempt_3/metabooks_small.csv",
    name="metabooks_small",
)

good_dataset_name_1["id"] = good_dataset_name_1["id"]
good_dataset_name_2["id"] = good_dataset_name_2["id"]
good_dataset_name_3["id"] = good_dataset_name_3["id"]


def is_nullish(x):
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
        return len(x) == 0
    try:
        result = pd.isna(x)
        if isinstance(result, (bool, np.bool_)):
            if result:
                return True
        elif isinstance(result, (np.ndarray, pd.Series, list, tuple)):
            return bool(np.all(result))
    except Exception:
        pass
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "[]", "[ ]"}


def safe_listify(value):
    if is_nullish(value):
        return []
    if isinstance(value, np.ndarray):
        return [str(v).strip() for v in value.tolist() if not is_nullish(v)]
    if isinstance(value, pd.Series):
        return [str(v).strip() for v in value.tolist() if not is_nullish(v)]
    if isinstance(value, list):
        return [str(v).strip() for v in value if not is_nullish(v)]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if not is_nullish(v)]
    s = str(value).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, np.ndarray):
            parsed = parsed.tolist()
        if isinstance(parsed, (list, tuple)):
            out = []
            for item in parsed:
                if isinstance(item, np.ndarray):
                    item = item.tolist()
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        if not is_nullish(sub):
                            out.append(str(sub).strip())
                elif not is_nullish(item):
                    out.append(str(item).strip())
            return out
    except Exception:
        pass
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    return [s] if s else []


def normalize_space_text(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def flatten_text_tokens(value):
    values = safe_listify(value)
    out = []
    for item in values:
        if is_nullish(item):
            continue
        item = str(item).strip()
        if not item:
            continue
        if "," in item:
            parts = [p.strip() for p in item.split(",") if p.strip()]
            out.extend(parts)
        else:
            out.append(item)
    deduped = []
    seen = set()
    for item in out:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def clean_author_piece(s):
    s = normalize_space_text(s)
    s = re.sub(
        r"\s*\((goodreads author|editor|illustrator|foreword|translator|adaptation|adapted by|designed by|introduction|introduction by|afterword|afterword by|preface|preface by|contributor|contributors?)\)\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\s+", " ", s).strip(" ,;/")
    return s


def canonical_author_value(value):
    authors = flatten_text_tokens(value)
    cleaned = []
    for a in authors:
        a = clean_author_piece(a)
        if a:
            cleaned.append(a)
    if not cleaned:
        return None
    return cleaned[0]


def canonical_author_list(value):
    authors = flatten_text_tokens(value)
    cleaned = []
    seen = set()
    for a in authors:
        a = clean_author_piece(a)
        if not a:
            continue
        key = a.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(a)
    return cleaned


def clean_title_value(value):
    if is_nullish(value):
        return None
    s = normalize_space_text(value)
    s = s.replace("&", " and ")
    s = re.sub(
        r"\s+\((p\.s\.|thoughtless|trophy newbery|penguin classics|novels of ancient egypt|sean dillon|charmed)\)\s*$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\s+\([^)]*(series|classics|edition|vol\.|volume|book\s+\d+)[^)]*\)\s*$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\s*:\s*a novel\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" /:-")
    return s if s else None


_PUBLISHER_CANONICAL_MAP = {
    "ace": "Ace Books",
    "ace books": "Ace Books",
    "jove": "Jove Books",
    "jove books": "Jove Books",
    "eos": "Eos",
    "perennial": "Harper Perennial",
    "harper perennial": "Harper Perennial",
    "harper perennial modern classics": "Harper Perennial",
    "western pub. co": "Western Publishing Company",
    "western publishing company": "Western Publishing Company",
    "western publishing company, inc.": "Western Publishing Company",
    "barbour publishing": "Barbour Publishing",
    "barbour publishing, incorporated": "Barbour Publishing",
    "penguin group daw": "DAW",
    "daw": "DAW",
}


def clean_publisher_value(value):
    if is_nullish(value):
        return None
    s = normalize_space_text(value)
    s = re.sub(r"\s*\((trade division|ny|usa|us|uk|ca)\)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bbooks usa\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bgroup\(ca\)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*/\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;/")
    if not s:
        return None
    mapped = _PUBLISHER_CANONICAL_MAP.get(s.casefold())
    return mapped if mapped else s


def clean_language_value(value):
    if is_nullish(value):
        return None
    vals = flatten_text_tokens(value)
    if not vals:
        s = normalize_space_text(value)
        vals = [p.strip() for p in re.split(r"[,;/]", s) if p.strip()]
    if not vals:
        return None
    lang = vals[0]
    lang_map = {
        "english": "english",
        "eng": "english",
    }
    return lang_map.get(lang.casefold(), lang)


def canonical_genres_list(value):
    tokens = flatten_text_tokens(value)
    out = []
    seen = set()
    synonym_map = {
        "children's books": "childrens",
        "childrens": "childrens",
        "kids": "childrens",
    }
    for token in tokens:
        tok = normalize_space_text(token).strip(" ,;/")
        if not tok:
            continue
        canonical = synonym_map.get(tok.casefold(), tok)
        key = canonical.casefold()
        if key not in seen:
            seen.add(key)
            out.append(canonical)
    return out


def make_blocking_text(value):
    if is_nullish(value):
        return ""
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        vals = safe_listify(value)
        return " ".join(str(v) for v in vals if not is_nullish(v))
    return normalize_space_text(value)


def preprocess_title_match(x):
    v = clean_title_value(x)
    return "" if v is None else v.lower().strip()


def preprocess_author_match(x):
    v = canonical_author_value(x)
    return "" if v is None else v.lower().strip()


def preprocess_publisher_match(x):
    v = clean_publisher_value(x)
    return "" if v is None else v.lower().strip()


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

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    if "author" in df.columns:
        df["author_primary"] = df["author"].apply(canonical_author_value)
        df["author"] = df["author"].apply(canonical_author_list)
    if "title" in df.columns:
        df["title_clean"] = df["title"].apply(clean_title_value)
    if "publisher" in df.columns:
        df["publisher_clean"] = df["publisher"].apply(clean_publisher_value)
    if "language" in df.columns:
        df["language_primary"] = df["language"].apply(clean_language_value)
    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(canonical_genres_list)

    if "title" in df.columns:
        df["blocking_title"] = df["title_clean"].fillna(df["title"]).apply(make_blocking_text)
    if "author" in df.columns:
        df["blocking_author"] = df["author_primary"].fillna("").apply(make_blocking_text)
    if "publish_year" in df.columns:
        df["blocking_publish_year"] = df["publish_year"].fillna("").astype(str)
    if "publisher" in df.columns:
        df["blocking_publisher"] = df["publisher_clean"].fillna("").apply(make_blocking_text)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

print("Performing Blocking")

blocker_1_2 = EmbeddingBlocker(
    good_dataset_name_2,
    good_dataset_name_1,
    text_cols=["blocking_title", "blocking_author", "blocking_publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_1_3 = EmbeddingBlocker(
    good_dataset_name_3,
    good_dataset_name_1,
    text_cols=["blocking_title", "blocking_author", "blocking_publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_2_3 = EmbeddingBlocker(
    good_dataset_name_3,
    good_dataset_name_2,
    text_cols=["blocking_title", "blocking_author", "blocking_publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

comparators_1_2 = [
    StringComparator(
        column="title_clean",
        similarity_function="jaro_winkler",
        preprocess=preprocess_title_match,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author_primary",
        similarity_function="jaro_winkler",
        preprocess=preprocess_author_match,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=365,
    ),
    StringComparator(
        column="publisher_clean",
        similarity_function="cosine",
        preprocess=preprocess_publisher_match,
        list_strategy="concatenate",
    ),
]

comparators_1_3 = [
    StringComparator(
        column="title_clean",
        similarity_function="cosine",
        preprocess=preprocess_title_match,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author_primary",
        similarity_function="jaro_winkler",
        preprocess=preprocess_author_match,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher_clean",
        similarity_function="jaro_winkler",
        preprocess=preprocess_publisher_match,
        list_strategy="concatenate",
    ),
]

comparators_2_3 = [
    StringComparator(
        column="title_clean",
        similarity_function="jaro_winkler",
        preprocess=preprocess_title_match,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author_primary",
        similarity_function="jaro_winkler",
        preprocess=preprocess_author_match,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher_clean",
        similarity_function="jaro_winkler",
        preprocess=preprocess_publisher_match,
        list_strategy="concatenate",
    ),
    NumericComparator(
        column="page_count",
        max_difference=10.0,
        list_strategy="average",
    ),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

train_1_2 = load_csv(
    "input/datasets/books/testsets/goodreads_2_amazon.csv",
    name="ground_truth_goodreads_amazon_train",
    add_index=False,
)

train_1_3 = load_csv(
    "input/datasets/books/testsets/metabooks_2_amazon.csv",
    name="ground_truth_metabooks_amazon_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/books/testsets/metabooks_2_goodreads.csv",
    name="ground_truth_metabooks_goodreads_train",
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
    good_dataset_name_2,
    good_dataset_name_1,
    to_pair_ids(train_1_2),
    labels=label_1_2,
    id_column="id",
)

train_1_3_features = feature_extractor_1_3.create_features(
    good_dataset_name_3,
    good_dataset_name_1,
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
    good_dataset_name_2,
    good_dataset_name_1,
    candidates=blocker_1_2,
    id_column="id",
    trained_classifier=best_models[0],
    threshold=0.75,
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    good_dataset_name_3,
    good_dataset_name_1,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[1],
    threshold=0.70,
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    good_dataset_name_3,
    good_dataset_name_2,
    candidates=blocker_2_3,
    id_column="id",
    trained_classifier=best_models[2],
    threshold=0.75,
)

ml_correspondences_1_2.to_csv(
    "output/correspondences/correspondences_goodreads_amazon.csv",
    index=False,
)
ml_correspondences_1_3.to_csv(
    "output/correspondences/correspondences_metabooks_amazon.csv",
    index=False,
)
ml_correspondences_2_3.to_csv(
    "output/correspondences/correspondences_metabooks_goodreads.csv",
    index=False,
)

ml_correspondences_1_2.to_csv(
    "output/correspondences/correspondences_amazon_small__goodreads_small.csv",
    index=False,
)
ml_correspondences_1_3.to_csv(
    "output/correspondences/correspondences_amazon_small__metabooks_small.csv",
    index=False,
)
ml_correspondences_2_3.to_csv(
    "output/correspondences/correspondences_goodreads_small__metabooks_small.csv",
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
    "amazon_small": 0.70,
    "goodreads_small": 0.85,
    "metabooks_small": 1.00,
}


def _extract_items(inputs):
    items = []
    for item in inputs:
        if isinstance(item, dict):
            value = item.get("value")
            source = item.get("source") or item.get("dataset") or item.get("source_dataset")
        else:
            value = item
            source = None
        if is_nullish(value):
            continue
        items.append({"value": value, "source": source})
    return items


def _pydi_safe_fuser(fn):
    def _wrapped(values, **kwargs):
        try:
            result = fn(values, **kwargs)
            if isinstance(result, tuple) and len(result) == 3:
                return result
            return result, 1.0, {}
        except Exception as e:
            fallback = None
            if values:
                try:
                    fallback = values[0]["value"] if isinstance(values[0], dict) else values[0]
                except Exception:
                    fallback = None
            return fallback, 0.1, {"error": str(e), "fallback": "first_value"}
    return _wrapped


def title_fuser(inputs, **kwargs):
    items = _extract_items(inputs)
    if not items:
        return None, 0.0, {}
    candidates = []
    for item in items:
        cleaned = clean_title_value(item["value"])
        if is_nullish(cleaned):
            continue
        norm = cleaned.casefold()
        token_count = len([t for t in re.split(r"\W+", norm) if t])
        candidates.append(
            {
                "value": cleaned,
                "source": item["source"],
                "norm": norm,
                "token_count": token_count,
                "length": len(cleaned),
            }
        )
    if not candidates:
        return None, 0.0, {}
    freq = {}
    for c in candidates:
        freq[c["norm"]] = freq.get(c["norm"], 0) + 1
    candidates = sorted(
        candidates,
        key=lambda x: (
            -freq.get(x["norm"], 0),
            -trust_map.get(x["source"], 0.0),
            -x["token_count"],
            -x["length"],
        ),
    )
    best = candidates[0]
    return best["value"], 1.0, {"selected_source": best["source"]}


def author_fuser(inputs, **kwargs):
    items = _extract_items(inputs)
    if not items:
        return None, 0.0, {}
    candidates = []
    for item in items:
        authors = canonical_author_list(item["value"])
        if not authors:
            continue
        primary = authors[0]
        candidates.append(
            {
                "value": primary,
                "source": item["source"],
                "norm": primary.casefold(),
            }
        )
    if not candidates:
        return None, 0.0, {}
    source_pref = {"metabooks_small": 3, "goodreads_small": 2, "amazon_small": 1, None: 0}
    freq = {}
    for c in candidates:
        freq[c["norm"]] = freq.get(c["norm"], 0) + 1
    candidates = sorted(
        candidates,
        key=lambda x: (
            -source_pref.get(x["source"], 0),
            -freq.get(x["norm"], 0),
            -len(x["value"]),
        ),
    )
    best = candidates[0]
    return best["value"], 1.0, {"selected_source": best["source"]}


def publisher_fuser(inputs, **kwargs):
    items = _extract_items(inputs)
    if not items:
        return None, 0.0, {}
    candidates = []
    for item in items:
        cleaned = clean_publisher_value(item["value"])
        if is_nullish(cleaned):
            continue
        norm = cleaned.casefold()
        token_count = len([t for t in re.split(r"\W+", norm) if t])
        candidates.append(
            {
                "value": cleaned,
                "source": item["source"],
                "norm": norm,
                "token_count": token_count,
                "length": len(cleaned),
            }
        )
    if not candidates:
        return None, 0.0, {}
    freq = {}
    for c in candidates:
        freq[c["norm"]] = freq.get(c["norm"], 0) + 1
    candidates = sorted(
        candidates,
        key=lambda x: (
            -trust_map.get(x["source"], 0.0),
            -freq.get(x["norm"], 0),
            -x["token_count"],
            -x["length"],
        ),
    )
    best = candidates[0]
    return best["value"], 1.0, {"selected_source": best["source"]}


def language_fuser(inputs, **kwargs):
    items = _extract_items(inputs)
    if not items:
        return None, 0.0, {}
    candidates = []
    for item in items:
        cleaned = clean_language_value(item["value"])
        if is_nullish(cleaned):
            continue
        candidates.append(
            {
                "value": cleaned,
                "source": item["source"],
                "norm": cleaned.casefold(),
            }
        )
    if not candidates:
        return None, 0.0, {}
    freq = {}
    for c in candidates:
        freq[c["norm"]] = freq.get(c["norm"], 0) + 1
    candidates = sorted(
        candidates,
        key=lambda x: (
            -freq.get(x["norm"], 0),
            -trust_map.get(x["source"], 0.0),
        ),
    )
    best = candidates[0]
    return best["value"], 1.0, {"selected_source": best["source"]}


def genres_fuser(inputs, **kwargs):
    items = _extract_items(inputs)
    if not items:
        return [], 0.0, {}
    preferred_order = ["metabooks_small", "goodreads_small", "amazon_small", None]
    best_by_source = {}
    for source_name in preferred_order:
        for item in items:
            if item["source"] == source_name:
                genres = canonical_genres_list(item["value"])
                if genres:
                    best_by_source[source_name] = genres
                    break
    for source_name in preferred_order:
        if source_name in best_by_source:
            result = best_by_source[source_name][:12]
            return result, 1.0, {"selected_source": source_name, "num_tokens": len(result)}
    return [], 0.0, {}


def robust_numeric_values(inputs):
    vals = []
    for item in _extract_items(inputs):
        try:
            num = float(item["value"])
            if not math.isnan(num):
                vals.append((item["source"], num))
        except Exception:
            continue
    return vals


def publish_year_fuser(inputs, **kwargs):
    vals = robust_numeric_values(inputs)
    if not vals:
        return None, 0.0, {}
    source_pref = {"metabooks_small": 3, "goodreads_small": 2, "amazon_small": 1, None: 0}
    vals = sorted(vals, key=lambda x: (-source_pref.get(x[0], 0),))
    chosen = vals[0]
    return int(round(chosen[1])), 1.0, {"selected_source": chosen[0]}


def page_count_fuser(inputs, **kwargs):
    vals = robust_numeric_values(inputs)
    if not vals:
        return None, 0.0, {}
    source_pref = {"metabooks_small": 3, "goodreads_small": 2, "amazon_small": 1, None: 0}
    vals = sorted(vals, key=lambda x: (-source_pref.get(x[0], 0),))
    chosen = vals[0]
    return int(round(chosen[1])), 1.0, {"selected_source": chosen[0]}


strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("title", _pydi_safe_fuser(title_fuser))
strategy.add_attribute_fuser("author", _pydi_safe_fuser(author_fuser))
strategy.add_attribute_fuser("publish_year", _pydi_safe_fuser(publish_year_fuser))
strategy.add_attribute_fuser("publisher", _pydi_safe_fuser(publisher_fuser))
strategy.add_attribute_fuser("rating", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("numratings", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("language", _pydi_safe_fuser(language_fuser))
strategy.add_attribute_fuser("genres", _pydi_safe_fuser(genres_fuser))
strategy.add_attribute_fuser("bookformat", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("edition", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("page_count", _pydi_safe_fuser(page_count_fuser))
strategy.add_attribute_fuser("price", prefer_higher_trust, trust_map=trust_map)

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
PIPELINE SNAPSHOT 03 END
============================================================

