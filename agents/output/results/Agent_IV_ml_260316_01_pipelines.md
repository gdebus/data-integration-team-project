# Pipeline Snapshots

notebook_name=Agent III & IV
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=78.99%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

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
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import os


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = ""

good_dataset_name_1 = load_csv(
    "output/schema-matching/dbpedia.csv",
    name="dbpedia",
)

good_dataset_name_2 = load_csv(
    "output/schema-matching/metacritic.csv",
    name="metacritic",
)

good_dataset_name_3 = load_csv(
    "output/schema-matching/sales.csv",
    name="sales",
)

# create id columns
good_dataset_name_1["id"] = good_dataset_name_1["id"]
good_dataset_name_2["id"] = good_dataset_name_2["id"]
good_dataset_name_3["id"] = good_dataset_name_3["id"]

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# --------------------------------
# Perform Schema Matching
# Schema is already aligned in the provided schema-matching outputs.
# --------------------------------

print("Matching Schema")

# --------------------------------
# Perform Blocking
# MUST use the precomputed blocker types and parameter settings
# --------------------------------

print("Performing Blocking")

# dbpedia_sales -> semantic_similarity on ["name", "platform", "releaseYear"], top_k=20
blocker_1_3 = EmbeddingBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

# metacritic_dbpedia -> semantic_similarity on ["name", "developer", "platform"], top_k=20
blocker_2_1 = EmbeddingBlocker(
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

# metacritic_sales -> semantic_similarity on ["name", "platform", "releaseYear"], top_k=20
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


# --------------------------------
# Matching Configuration
# MUST use the supplied comparator settings
# --------------------------------

lower_strip = lambda x: str(x).lower().strip() if pd.notna(x) else ""

comparators_1_3 = [
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

comparators_2_1 = [
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

feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_1 = FeatureExtractor(comparators_2_1)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

# Load ground truth correspondences (ML training/test)
train_1_3 = load_csv(
    "input/datasets/games/testsets/dbpedia_2_sales_test.csv",
    name="ground_truth_dbpedia_sales_train",
    add_index=False,
)

train_2_1 = load_csv(
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
train_1_3_features = feature_extractor_1_3.create_features(
    good_dataset_name_1,
    good_dataset_name_3,
    train_1_3[["id1", "id2"]],
    labels=train_1_3["label"],
    id_column="id",
)

train_2_1_features = feature_extractor_2_1.create_features(
    good_dataset_name_2,
    good_dataset_name_1,
    train_2_1[["id1", "id2"]],
    labels=train_2_1["label"],
    id_column="id",
)

train_2_3_features = feature_extractor_2_3.create_features(
    good_dataset_name_2,
    good_dataset_name_3,
    train_2_3[["id1", "id2"]],
    labels=train_2_3["label"],
    id_column="id",
)

# Prepare data for ML training
feat_cols_1_3 = [
    col for col in train_1_3_features.columns if col not in ["id1", "id2", "label"]
]
X_train_1_3 = train_1_3_features[feat_cols_1_3]
y_train_1_3 = train_1_3_features["label"]

feat_cols_2_1 = [
    col for col in train_2_1_features.columns if col not in ["id1", "id2", "label"]
]
X_train_2_1 = train_2_1_features[feat_cols_2_1]
y_train_2_1 = train_2_1_features["label"]

feat_cols_2_3 = [
    col for col in train_2_3_features.columns if col not in ["id1", "id2", "label"]
]
X_train_2_3 = train_2_3_features[feat_cols_2_3]
y_train_2_3 = train_2_3_features["label"]

training_datasets = [
    (X_train_1_3, y_train_1_3),
    (X_train_2_1, y_train_2_1),
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

# --------------------------------
# the order of datasets in correspondences must correspond
# to the order of columns within the testsets
# --------------------------------

ml_matcher_1_3 = MLBasedMatcher(feature_extractor_1_3)
ml_matcher_2_1 = MLBasedMatcher(feature_extractor_2_1)
ml_matcher_2_3 = MLBasedMatcher(feature_extractor_2_3)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    good_dataset_name_1,
    good_dataset_name_3,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_2_1 = ml_matcher_2_1.match(
    good_dataset_name_2,
    good_dataset_name_1,
    candidates=blocker_2_1,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    good_dataset_name_2,
    good_dataset_name_3,
    candidates=blocker_2_3,
    id_column="id",
    trained_classifier=best_models[2],
)

print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_1_3 = clusterer.cluster(ml_correspondences_1_3)
ml_correspondences_2_1 = clusterer.cluster(ml_correspondences_2_1)
ml_correspondences_2_3 = clusterer.cluster(ml_correspondences_2_3)

"""
Make sure to save correspondences for each pair afer applying Matcher.
**Use proper filename with correct dataset name to save the correspondences**
"""

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_1_3.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_dbpedia_sales.csv",
    ),
    index=False,
)
ml_correspondences_2_1.to_csv(
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

# -------------- Data Fusion ------------------

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_3, ml_correspondences_2_1, ml_correspondences_2_3],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("releaseYear", longest_string)
strategy.add_attribute_fuser("developer", longest_string)
strategy.add_attribute_fuser("platform", longest_string)
strategy.add_attribute_fuser("series", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)
strategy.add_attribute_fuser("criticScore", longest_string)
strategy.add_attribute_fuser("userScore", longest_string)
strategy.add_attribute_fuser("ESRB", longest_string)
strategy.add_attribute_fuser("globalSales", longest_string)

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
accuracy_score=82.35%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

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
from PyDI.fusion import DataFusionStrategy, DataFusionEngine
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import numpy as np
import os
import re


# --------------------------------
# Helper Functions
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip() if pd.notna(x) else ""


def is_missing(x):
    if pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s in {"", "nan", "none", "null"}


def clean_title(x):
    if is_missing(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\s*\([^)]*video game[^)]*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None


def canonical_company(x):
    if is_missing(x):
        return None
    s = str(x).strip()
    s_norm = lower_strip(s)
    mapping = {
        "tt games": "Traveller's Tales",
        "traveller's tales": "Traveller's Tales",
        "travelers tales": "Traveller's Tales",
        "scee london studio": "London Studio",
        "london studio": "London Studio",
        "sony computer entertainment": "Sony Computer Entertainment",
        "scea": "Sony Computer Entertainment",
        "scee": "Sony Computer Entertainment",
        "scei": "Sony Computer Entertainment",
        "rockstar studios": "Rockstar Games",
        "rockstar games": "Rockstar Games",
        "koei": "Koei",
        "koei tecmo": "Koei Tecmo",
        "tecmo koei": "Koei Tecmo",
    }
    return mapping.get(s_norm, s)


def canonical_esrb(x):
    if is_missing(x):
        return None
    s = lower_strip(x).replace(" ", "")
    mapping = {
        "e": "E",
        "e10+": "E10+",
        "t": "T",
        "m": "M",
        "m17+": "M",
        "mature17+": "M",
        "ao": "AO",
        "adultsonly": "AO",
        "rp": "RP",
        "ec": "EC",
    }
    return mapping.get(s, str(x).strip())


def parse_date_safe(x):
    if is_missing(x):
        return pd.NaT
    return pd.to_datetime(x, errors="coerce")


def parse_float_safe(x):
    if is_missing(x):
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def extract_source_and_value(value):
    if isinstance(value, tuple) and len(value) >= 2:
        return str(value[0]), value[1]
    return None, value


def normalize_numeric_output(num):
    if pd.isna(num):
        return None
    num = float(num)
    return int(num) if num.is_integer() else num


def choose_by_priority(cleaned_values, preferred_prefixes):
    if not cleaned_values:
        return None

    for prefix in preferred_prefixes:
        for rec_id, val in cleaned_values:
            if rec_id is not None and str(rec_id).startswith(prefix):
                return val

    counts = {}
    first_seen = {}
    for idx, (_, val) in enumerate(cleaned_values):
        key = str(val)
        counts[key] = counts.get(key, 0) + 1
        if key not in first_seen:
            first_seen[key] = (idx, val)

    best_key = sorted(counts.items(), key=lambda x: (-x[1], first_seen[x[0]][0]))[0][0]
    return first_seen[best_key][1]


def fuser_name(values):
    cleaned = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        val = clean_title(val)
        if not is_missing(val):
            cleaned.append((rec_id, val))
    return choose_by_priority(cleaned, ["metacritic_", "sales_", "dbpedia_"])


def fuser_release_year(values):
    cleaned = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        dt = parse_date_safe(val)
        if pd.notna(dt):
            cleaned.append((rec_id, dt))

    if not cleaned:
        return None

    for prefix in ["metacritic_", "sales_", "dbpedia_"]:
        for rec_id, dt in cleaned:
            if rec_id is not None and str(rec_id).startswith(prefix):
                return dt.strftime("%Y-%m-%d")

    dts = [dt for _, dt in cleaned]
    return min(dts).strftime("%Y-%m-%d")


def fuser_developer(values):
    cleaned = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        val = canonical_company(val)
        if not is_missing(val):
            cleaned.append((rec_id, val))
    return choose_by_priority(cleaned, ["metacritic_", "sales_", "dbpedia_"])


def fuser_publisher(values):
    cleaned = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        val = canonical_company(val)
        if not is_missing(val):
            cleaned.append((rec_id, val))
    return choose_by_priority(cleaned, ["sales_", "metacritic_", "dbpedia_"])


def fuser_platform(values):
    cleaned = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        if not is_missing(val):
            cleaned.append((rec_id, str(val).strip()))
    return choose_by_priority(cleaned, ["metacritic_", "sales_", "dbpedia_"])


def fuser_esrb(values):
    cleaned = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        val = canonical_esrb(val)
        if not is_missing(val):
            cleaned.append((rec_id, val))
    return choose_by_priority(cleaned, ["metacritic_", "sales_", "dbpedia_"])


def fuser_numeric_prefer_metacritic(values):
    parsed = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        num = parse_float_safe(val)
        if not np.isnan(num):
            parsed.append((rec_id, num))

    if not parsed:
        return None

    for prefix in ["metacritic_", "sales_", "dbpedia_"]:
        for rec_id, num in parsed:
            if rec_id is not None and str(rec_id).startswith(prefix):
                return normalize_numeric_output(num)

    nums = [num for _, num in parsed]
    return normalize_numeric_output(np.median(nums))


def fuser_numeric_prefer_sales(values):
    parsed = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        num = parse_float_safe(val)
        if not np.isnan(num):
            parsed.append((rec_id, num))

    if not parsed:
        return None

    for prefix in ["sales_", "metacritic_", "dbpedia_"]:
        for rec_id, num in parsed:
            if rec_id is not None and str(rec_id).startswith(prefix):
                return normalize_numeric_output(num)

    nums = [num for _, num in parsed]
    return normalize_numeric_output(np.median(nums))


def fuser_series(values):
    cleaned = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        if not is_missing(val):
            cleaned.append((rec_id, str(val).strip()))
    return choose_by_priority(cleaned, ["dbpedia_", "metacritic_", "sales_"])


def drop_suspicious_correspondences(corr_df, left_df, right_df):
    if corr_df is None or len(corr_df) == 0:
        return corr_df

    left_lookup = left_df.set_index("id")
    right_lookup = right_df.set_index("id")

    keep_rows = []
    for _, row in corr_df.iterrows():
        id1 = row["id1"]
        id2 = row["id2"]

        if id1 not in left_lookup.index or id2 not in right_lookup.index:
            continue

        l = left_lookup.loc[id1]
        r = right_lookup.loc[id2]

        name_l = clean_title(l.get("name"))
        name_r = clean_title(r.get("name"))
        platform_l = lower_strip(l.get("platform"))
        platform_r = lower_strip(r.get("platform"))
        date_l = parse_date_safe(l.get("releaseYear"))
        date_r = parse_date_safe(r.get("releaseYear"))

        title_ok = False
        if not is_missing(name_l) and not is_missing(name_r):
            nl = lower_strip(name_l)
            nr = lower_strip(name_r)
            title_ok = nl == nr or nl in nr or nr in nl

        platform_ok = False
        if not is_missing(platform_l) and not is_missing(platform_r):
            platform_ok = platform_l == platform_r

        date_ok = True
        if pd.notna(date_l) and pd.notna(date_r):
            date_ok = abs((date_l - date_r).days) <= 730

        if title_ok and date_ok:
            keep_rows.append(row)
        elif title_ok and (platform_ok or pd.isna(date_l) or pd.isna(date_r)):
            keep_rows.append(row)

    if len(keep_rows) == 0:
        return corr_df.iloc[0:0].copy()

    return pd.DataFrame(keep_rows)


# --------------------------------
# Prepare Data
# --------------------------------

good_dataset_name_1 = load_csv(
    "output/schema-matching/dbpedia.csv",
    name="dbpedia",
)

good_dataset_name_2 = load_csv(
    "output/schema-matching/metacritic.csv",
    name="metacritic",
)

good_dataset_name_3 = load_csv(
    "output/schema-matching/sales.csv",
    name="sales",
)

good_dataset_name_1["id"] = good_dataset_name_1["id"]
good_dataset_name_2["id"] = good_dataset_name_2["id"]
good_dataset_name_3["id"] = good_dataset_name_3["id"]

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    if "name" in df.columns:
        df["name"] = df["name"].apply(clean_title)
    if "developer" in df.columns:
        df["developer"] = df["developer"].apply(canonical_company)
    if "publisher" in df.columns:
        df["publisher"] = df["publisher"].apply(canonical_company)
    if "ESRB" in df.columns:
        df["ESRB"] = df["ESRB"].apply(canonical_esrb)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# --------------------------------
# Perform Schema Matching
# Schema is already aligned in the provided schema-matching outputs.
# --------------------------------

print("Matching Schema")

# --------------------------------
# Perform Blocking
# MUST use the precomputed blocker types and parameter settings
# --------------------------------

print("Performing Blocking")

blocker_1_3 = EmbeddingBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_2_1 = EmbeddingBlocker(
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

# --------------------------------
# Matching Configuration
# MUST use the supplied comparator settings
# --------------------------------

comparators_1_3 = [
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

comparators_2_1 = [
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

feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_1 = FeatureExtractor(comparators_2_1)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

train_1_3 = load_csv(
    "input/datasets/games/testsets/dbpedia_2_sales_test.csv",
    name="ground_truth_dbpedia_sales_train",
    add_index=False,
)

train_2_1 = load_csv(
    "input/datasets/games/testsets/metacritic_2_dbpedia_test.csv",
    name="ground_truth_metacritic_dbpedia_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/games/testsets/metacritic_2_sales_test.csv",
    name="ground_truth_metacritic_sales_train",
    add_index=False,
)

train_1_3_features = feature_extractor_1_3.create_features(
    good_dataset_name_1,
    good_dataset_name_3,
    train_1_3[["id1", "id2"]],
    labels=train_1_3["label"],
    id_column="id",
)

train_2_1_features = feature_extractor_2_1.create_features(
    good_dataset_name_2,
    good_dataset_name_1,
    train_2_1[["id1", "id2"]],
    labels=train_2_1["label"],
    id_column="id",
)

train_2_3_features = feature_extractor_2_3.create_features(
    good_dataset_name_2,
    good_dataset_name_3,
    train_2_3[["id1", "id2"]],
    labels=train_2_3["label"],
    id_column="id",
)

feat_cols_1_3 = [
    col for col in train_1_3_features.columns if col not in ["id1", "id2", "label"]
]
X_train_1_3 = train_1_3_features[feat_cols_1_3]
y_train_1_3 = train_1_3_features["label"]

feat_cols_2_1 = [
    col for col in train_2_1_features.columns if col not in ["id1", "id2", "label"]
]
X_train_2_1 = train_2_1_features[feat_cols_2_1]
y_train_2_1 = train_2_1_features["label"]

feat_cols_2_3 = [
    col for col in train_2_3_features.columns if col not in ["id1", "id2", "label"]
]
X_train_2_3 = train_2_3_features[feat_cols_2_3]
y_train_2_3 = train_2_3_features["label"]

training_datasets = [
    (X_train_1_3, y_train_1_3),
    (X_train_2_1, y_train_2_1),
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
            "max_depth": [10, None],
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
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
        },
    },
    "SVM": {
        "model": SVC(random_state=42, probability=True),
        "params": {
            "C": [1.0, 10.0],
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

ml_matcher_1_3 = MLBasedMatcher(feature_extractor_1_3)
ml_matcher_2_1 = MLBasedMatcher(feature_extractor_2_1)
ml_matcher_2_3 = MLBasedMatcher(feature_extractor_2_3)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    good_dataset_name_1,
    good_dataset_name_3,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_2_1 = ml_matcher_2_1.match(
    good_dataset_name_2,
    good_dataset_name_1,
    candidates=blocker_2_1,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    good_dataset_name_2,
    good_dataset_name_3,
    candidates=blocker_2_3,
    id_column="id",
    trained_classifier=best_models[2],
)

clusterer = MaximumBipartiteMatching()
ml_correspondences_1_3 = clusterer.cluster(ml_correspondences_1_3)
ml_correspondences_2_1 = clusterer.cluster(ml_correspondences_2_1)
ml_correspondences_2_3 = clusterer.cluster(ml_correspondences_2_3)

ml_correspondences_1_3 = drop_suspicious_correspondences(
    ml_correspondences_1_3,
    good_dataset_name_1,
    good_dataset_name_3,
)

ml_correspondences_2_1 = drop_suspicious_correspondences(
    ml_correspondences_2_1,
    good_dataset_name_2,
    good_dataset_name_1,
)

ml_correspondences_2_3 = drop_suspicious_correspondences(
    ml_correspondences_2_3,
    good_dataset_name_2,
    good_dataset_name_3,
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_1_3.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_dbpedia_sales.csv",
    ),
    index=False,
)
ml_correspondences_2_1.to_csv(
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
    [ml_correspondences_1_3, ml_correspondences_2_1, ml_correspondences_2_3],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", fuser_name)
strategy.add_attribute_fuser("releaseYear", fuser_release_year)
strategy.add_attribute_fuser("developer", fuser_developer)
strategy.add_attribute_fuser("platform", fuser_platform)
strategy.add_attribute_fuser("series", fuser_series)
strategy.add_attribute_fuser("publisher", fuser_publisher)
strategy.add_attribute_fuser("criticScore", fuser_numeric_prefer_metacritic)
strategy.add_attribute_fuser("userScore", fuser_numeric_prefer_metacritic)
strategy.add_attribute_fuser("ESRB", fuser_esrb)
strategy.add_attribute_fuser("globalSales", fuser_numeric_prefer_sales)

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
PIPELINE SNAPSHOT 02 END
============================================================

============================================================
PIPELINE SNAPSHOT 03 START
============================================================
node_index=20
node_name=execute_pipeline
accuracy_score=84.87%
------------------------------------------------------------

```python
# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

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
from PyDI.fusion import DataFusionStrategy, DataFusionEngine
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import numpy as np
import os
import re


# --------------------------------
# Helper Functions
# --------------------------------

def lower_strip(x):
    return str(x).lower().strip() if pd.notna(x) else ""


def is_missing(x):
    if pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s in {"", "nan", "none", "null"}


def clean_title(x):
    if is_missing(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\s*\([^)]*video game[^)]*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None


def normalize_company_for_matching(x):
    if is_missing(x):
        return None
    s = str(x).strip()
    s_norm = lower_strip(s)
    mapping = {
        "tt games": "travellers tales",
        "traveller's tales": "travellers tales",
        "travelers tales": "travellers tales",
        "scee london studio": "london studio",
        "london studio": "london studio",
        "sony computer entertainment": "sony computer entertainment",
        "scea": "sony computer entertainment",
        "scee": "sony computer entertainment",
        "scei": "sony computer entertainment",
        "rockstar studios": "rockstar games",
        "rockstar games": "rockstar games",
        "koei": "koei",
        "koei tecmo": "koei tecmo",
        "tecmo koei": "koei tecmo",
    }
    return mapping.get(s_norm, s)


def canonical_esrb_for_matching(x):
    if is_missing(x):
        return None
    s = lower_strip(x).replace(" ", "")
    mapping = {
        "e": "E",
        "e10+": "E10+",
        "t": "T",
        "m": "M",
        "m17+": "M17+",
        "mature17+": "M17+",
        "ao": "AO",
        "adultsonly": "AO",
        "rp": "RP",
        "ec": "EC",
    }
    return mapping.get(s, str(x).strip())


def canonical_esrb_for_output(x):
    if is_missing(x):
        return None
    s = lower_strip(x).replace(" ", "")
    mapping = {
        "e": "E",
        "everyone": "E",
        "e10+": "E10+",
        "everyone10+": "E10+",
        "t": "T",
        "teen": "T",
        "m": "M",
        "m17+": "M17+",
        "mature17+": "M17+",
        "ao": "AO",
        "adultsonly": "AO",
        "rp": "RP",
        "ratingpending": "RP",
        "ec": "EC",
        "earlychildhood": "EC",
    }
    return mapping.get(s, str(x).strip())


def parse_date_safe(x):
    if is_missing(x):
        return pd.NaT
    return pd.to_datetime(x, errors="coerce")


def parse_float_safe(x):
    if is_missing(x):
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def extract_source_and_value(value):
    if isinstance(value, tuple) and len(value) >= 2:
        return str(value[0]), value[1]
    return None, value


def normalize_numeric_output(num):
    if pd.isna(num):
        return None
    num = float(num)
    return int(num) if num.is_integer() else round(num, 1)


def year_from_any(x):
    dt = parse_date_safe(x)
    if pd.notna(dt):
        return int(dt.year)
    if is_missing(x):
        return None
    m = re.search(r"(19|20)\d{2}", str(x))
    return int(m.group(0)) if m else None


def jaccard_tokens(a, b):
    if is_missing(a) or is_missing(b):
        return 0.0
    ta = set(re.findall(r"[a-z0-9]+", lower_strip(a)))
    tb = set(re.findall(r"[a-z0-9]+", lower_strip(b)))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def choose_surface_form(cleaned_values, preferred_prefixes):
    if not cleaned_values:
        return None

    grouped = {}
    order = []
    for idx, (rec_id, original, canonical) in enumerate(cleaned_values):
        key = lower_strip(canonical)
        if key not in grouped:
            grouped[key] = []
            order.append((idx, key))
        grouped[key].append((rec_id, original, canonical))

    ranked = sorted(
        grouped.items(),
        key=lambda kv: (-len(kv[1]), min(i for i, k in order if k == kv[0])),
    )
    winning_key, winning_group = ranked[0]

    for prefix in preferred_prefixes:
        for rec_id, original, _ in winning_group:
            if rec_id is not None and str(rec_id).startswith(prefix):
                return original

    return winning_group[0][1]


def choose_text_majority(cleaned_values, preferred_prefixes):
    if not cleaned_values:
        return None

    counts = {}
    first_seen = {}
    originals = {}
    for idx, (rec_id, val) in enumerate(cleaned_values):
        key = lower_strip(val)
        counts[key] = counts.get(key, 0) + 1
        if key not in first_seen:
            first_seen[key] = idx
            originals[key] = []
        originals[key].append((rec_id, val))

    best_key = sorted(counts.keys(), key=lambda k: (-counts[k], first_seen[k]))[0]
    candidates = originals[best_key]

    for prefix in preferred_prefixes:
        for rec_id, val in candidates:
            if rec_id is not None and str(rec_id).startswith(prefix):
                return val

    return candidates[0][1]


def robust_numeric_fuser(values, preferred_prefixes, tolerance, decimals=1):
    parsed = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        num = parse_float_safe(val)
        if not np.isnan(num):
            parsed.append((rec_id, float(num)))

    if not parsed:
        return None
    if len(parsed) == 1:
        return normalize_numeric_output(parsed[0][1])

    nums = [num for _, num in parsed]

    clusters = []
    for rec_id, num in parsed:
        placed = False
        for cluster in clusters:
            if abs(cluster["center"] - num) <= tolerance:
                cluster["values"].append((rec_id, num))
                cluster["center"] = np.median([v for _, v in cluster["values"]])
                placed = True
                break
        if not placed:
            clusters.append({"center": num, "values": [(rec_id, num)]})

    best_cluster = sorted(
        clusters,
        key=lambda c: (-len(c["values"]), np.std([v for _, v in c["values"]]) if len(c["values"]) > 1 else 0.0),
    )[0]

    if len(best_cluster["values"]) >= 2:
        consensus = np.median([v for _, v in best_cluster["values"]])
        return normalize_numeric_output(round(consensus, decimals))

    sorted_vals = sorted(nums)
    if len(sorted_vals) >= 3:
        median_val = float(np.median(sorted_vals))
        close_vals = [v for v in sorted_vals if abs(v - median_val) <= tolerance]
        if close_vals:
            return normalize_numeric_output(round(float(np.median(close_vals)), decimals))

    for prefix in preferred_prefixes:
        for rec_id, num in parsed:
            if rec_id is not None and str(rec_id).startswith(prefix):
                return normalize_numeric_output(round(num, decimals))

    return normalize_numeric_output(round(float(np.median(nums)), decimals))


def fuser_name(values):
    cleaned = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        val = clean_title(val)
        if not is_missing(val):
            cleaned.append((rec_id, val))
    return choose_text_majority(cleaned, ["metacritic_", "sales_", "dbpedia_"])


def fuser_release_year(values):
    parsed = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        year = year_from_any(val)
        if year is not None:
            parsed.append((rec_id, year))

    if not parsed:
        return None

    counts = {}
    first_pos = {}
    for idx, (_, year) in enumerate(parsed):
        counts[year] = counts.get(year, 0) + 1
        if year not in first_pos:
            first_pos[year] = idx

    max_count = max(counts.values())
    candidates = [y for y, c in counts.items() if c == max_count]

    if len(candidates) == 1:
        best_year = candidates[0]
    else:
        candidates_sorted = sorted(candidates)
        if len(candidates_sorted) >= 2 and (candidates_sorted[-1] - candidates_sorted[0]) <= 1:
            best_year = min(candidates_sorted)
        else:
            for prefix in ["sales_", "metacritic_", "dbpedia_"]:
                for rec_id, year in parsed:
                    if year in candidates and rec_id is not None and str(rec_id).startswith(prefix):
                        best_year = year
                        break
                else:
                    continue
                break
            else:
                best_year = min(candidates_sorted)

    return f"{int(best_year)}-01-01"


def fuser_developer(values):
    cleaned = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        if not is_missing(val):
            original = str(val).strip()
            canonical = normalize_company_for_matching(original)
            cleaned.append((rec_id, original, canonical))
    return choose_surface_form(cleaned, ["metacritic_", "sales_", "dbpedia_"])


def fuser_publisher(values):
    cleaned = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        if not is_missing(val):
            original = str(val).strip()
            canonical = normalize_company_for_matching(original)
            cleaned.append((rec_id, original, canonical))
    return choose_surface_form(cleaned, ["sales_", "metacritic_", "dbpedia_"])


def fuser_platform(values):
    cleaned = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        if not is_missing(val):
            cleaned.append((rec_id, str(val).strip()))
    return choose_text_majority(cleaned, ["metacritic_", "sales_", "dbpedia_"])


def fuser_esrb(values):
    cleaned = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        val = canonical_esrb_for_output(val)
        if not is_missing(val):
            cleaned.append((rec_id, val))
    return choose_text_majority(cleaned, ["metacritic_", "sales_", "dbpedia_"])


def fuser_critic_score(values):
    return robust_numeric_fuser(
        values,
        preferred_prefixes=["metacritic_", "sales_", "dbpedia_"],
        tolerance=5.0,
        decimals=1,
    )


def fuser_user_score(values):
    return robust_numeric_fuser(
        values,
        preferred_prefixes=["sales_", "metacritic_", "dbpedia_"],
        tolerance=0.5,
        decimals=1,
    )


def fuser_numeric_prefer_sales(values):
    parsed = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        num = parse_float_safe(val)
        if not np.isnan(num):
            parsed.append((rec_id, num))

    if not parsed:
        return None

    for prefix in ["sales_", "metacritic_", "dbpedia_"]:
        for rec_id, num in parsed:
            if rec_id is not None and str(rec_id).startswith(prefix):
                return normalize_numeric_output(num)

    nums = [num for _, num in parsed]
    return normalize_numeric_output(np.median(nums))


def fuser_series(values):
    cleaned = []
    for value in values:
        rec_id, val = extract_source_and_value(value)
        if not is_missing(val):
            cleaned.append((rec_id, str(val).strip()))
    return choose_text_majority(cleaned, ["dbpedia_", "metacritic_", "sales_"])


def drop_suspicious_correspondences(corr_df, left_df, right_df):
    if corr_df is None or len(corr_df) == 0:
        return corr_df

    left_lookup = left_df.set_index("id")
    right_lookup = right_df.set_index("id")

    keep_rows = []
    for _, row in corr_df.iterrows():
        id1 = row["id1"]
        id2 = row["id2"]

        if id1 not in left_lookup.index or id2 not in right_lookup.index:
            continue

        l = left_lookup.loc[id1]
        r = right_lookup.loc[id2]

        name_l = clean_title(l.get("name"))
        name_r = clean_title(r.get("name"))
        platform_l = lower_strip(l.get("platform"))
        platform_r = lower_strip(r.get("platform"))
        date_l = parse_date_safe(l.get("releaseYear"))
        date_r = parse_date_safe(r.get("releaseYear"))
        dev_l = normalize_company_for_matching(l.get("developer"))
        dev_r = normalize_company_for_matching(r.get("developer"))

        title_sim = jaccard_tokens(name_l, name_r)

        title_ok = False
        if not is_missing(name_l) and not is_missing(name_r):
            nl = lower_strip(name_l)
            nr = lower_strip(name_r)
            title_ok = nl == nr or title_sim >= 0.5

        platform_ok = False
        if not is_missing(platform_l) and not is_missing(platform_r):
            platform_ok = platform_l == platform_r

        developer_ok = False
        if not is_missing(dev_l) and not is_missing(dev_r):
            developer_ok = jaccard_tokens(dev_l, dev_r) >= 0.5 or lower_strip(dev_l) == lower_strip(dev_r)

        date_ok = True
        year_gap = 0
        if pd.notna(date_l) and pd.notna(date_r):
            year_gap = abs((date_l - date_r).days)
            date_ok = year_gap <= 365

        if title_ok and platform_ok and date_ok:
            keep_rows.append(row)
        elif title_ok and platform_ok and (pd.isna(date_l) or pd.isna(date_r)):
            keep_rows.append(row)
        elif title_sim >= 0.8 and date_ok and (platform_ok or developer_ok):
            keep_rows.append(row)
        elif lower_strip(name_l) == lower_strip(name_r) and platform_ok and year_gap <= 730:
            keep_rows.append(row)

    if len(keep_rows) == 0:
        return corr_df.iloc[0:0].copy()

    return pd.DataFrame(keep_rows)


# --------------------------------
# Prepare Data
# --------------------------------

good_dataset_name_1 = load_csv(
    "output/schema-matching/dbpedia.csv",
    name="dbpedia",
)

good_dataset_name_2 = load_csv(
    "output/schema-matching/metacritic.csv",
    name="metacritic",
)

good_dataset_name_3 = load_csv(
    "output/schema-matching/sales.csv",
    name="sales",
)

good_dataset_name_1["id"] = good_dataset_name_1["id"]
good_dataset_name_2["id"] = good_dataset_name_2["id"]
good_dataset_name_3["id"] = good_dataset_name_3["id"]

for df in [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]:
    if "name" in df.columns:
        df["name"] = df["name"].apply(clean_title)
    if "developer" in df.columns:
        df["developer"] = df["developer"].apply(normalize_company_for_matching)
    if "publisher" in df.columns:
        df["publisher"] = df["publisher"].apply(lambda x: str(x).strip() if not is_missing(x) else x)
    if "ESRB" in df.columns:
        df["ESRB"] = df["ESRB"].apply(canonical_esrb_for_matching)

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# --------------------------------
# Perform Schema Matching
# Schema is already aligned in the provided schema-matching outputs.
# --------------------------------

print("Matching Schema")

# --------------------------------
# Perform Blocking
# MUST use the precomputed blocker types and parameter settings
# --------------------------------

print("Performing Blocking")

blocker_1_3 = EmbeddingBlocker(
    good_dataset_name_1,
    good_dataset_name_3,
    text_cols=["name", "platform", "releaseYear"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_2_1 = EmbeddingBlocker(
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

# --------------------------------
# Matching Configuration
# MUST use the supplied comparator settings
# --------------------------------

comparators_1_3 = [
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

comparators_2_1 = [
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

feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_1 = FeatureExtractor(comparators_2_1)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

train_1_3 = load_csv(
    "input/datasets/games/testsets/dbpedia_2_sales_test.csv",
    name="ground_truth_dbpedia_sales_train",
    add_index=False,
)

train_2_1 = load_csv(
    "input/datasets/games/testsets/metacritic_2_dbpedia_test.csv",
    name="ground_truth_metacritic_dbpedia_train",
    add_index=False,
)

train_2_3 = load_csv(
    "input/datasets/games/testsets/metacritic_2_sales_test.csv",
    name="ground_truth_metacritic_sales_train",
    add_index=False,
)

train_1_3_features = feature_extractor_1_3.create_features(
    good_dataset_name_1,
    good_dataset_name_3,
    train_1_3[["id1", "id2"]],
    labels=train_1_3["label"],
    id_column="id",
)

train_2_1_features = feature_extractor_2_1.create_features(
    good_dataset_name_2,
    good_dataset_name_1,
    train_2_1[["id1", "id2"]],
    labels=train_2_1["label"],
    id_column="id",
)

train_2_3_features = feature_extractor_2_3.create_features(
    good_dataset_name_2,
    good_dataset_name_3,
    train_2_3[["id1", "id2"]],
    labels=train_2_3["label"],
    id_column="id",
)

feat_cols_1_3 = [
    col for col in train_1_3_features.columns if col not in ["id1", "id2", "label"]
]
X_train_1_3 = train_1_3_features[feat_cols_1_3]
y_train_1_3 = train_1_3_features["label"]

feat_cols_2_1 = [
    col for col in train_2_1_features.columns if col not in ["id1", "id2", "label"]
]
X_train_2_1 = train_2_1_features[feat_cols_2_1]
y_train_2_1 = train_2_1_features["label"]

feat_cols_2_3 = [
    col for col in train_2_3_features.columns if col not in ["id1", "id2", "label"]
]
X_train_2_3 = train_2_3_features[feat_cols_2_3]
y_train_2_3 = train_2_3_features["label"]

training_datasets = [
    (X_train_1_3, y_train_1_3),
    (X_train_2_1, y_train_2_1),
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
            "max_depth": [10, None],
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
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
        },
    },
    "SVM": {
        "model": SVC(random_state=42, probability=True),
        "params": {
            "C": [1.0, 10.0],
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

ml_matcher_1_3 = MLBasedMatcher(feature_extractor_1_3)
ml_matcher_2_1 = MLBasedMatcher(feature_extractor_2_1)
ml_matcher_2_3 = MLBasedMatcher(feature_extractor_2_3)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    good_dataset_name_1,
    good_dataset_name_3,
    candidates=blocker_1_3,
    id_column="id",
    trained_classifier=best_models[0],
)

ml_correspondences_2_1 = ml_matcher_2_1.match(
    good_dataset_name_2,
    good_dataset_name_1,
    candidates=blocker_2_1,
    id_column="id",
    trained_classifier=best_models[1],
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    good_dataset_name_2,
    good_dataset_name_3,
    candidates=blocker_2_3,
    id_column="id",
    trained_classifier=best_models[2],
)

clusterer = MaximumBipartiteMatching()
ml_correspondences_1_3 = clusterer.cluster(ml_correspondences_1_3)
ml_correspondences_2_1 = clusterer.cluster(ml_correspondences_2_1)
ml_correspondences_2_3 = clusterer.cluster(ml_correspondences_2_3)

ml_correspondences_1_3 = drop_suspicious_correspondences(
    ml_correspondences_1_3,
    good_dataset_name_1,
    good_dataset_name_3,
)

ml_correspondences_2_1 = drop_suspicious_correspondences(
    ml_correspondences_2_1,
    good_dataset_name_2,
    good_dataset_name_1,
)

ml_correspondences_2_3 = drop_suspicious_correspondences(
    ml_correspondences_2_3,
    good_dataset_name_2,
    good_dataset_name_3,
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_1_3.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_dbpedia_sales.csv",
    ),
    index=False,
)
ml_correspondences_2_1.to_csv(
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
    [ml_correspondences_1_3, ml_correspondences_2_1, ml_correspondences_2_3],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", fuser_name)
strategy.add_attribute_fuser("releaseYear", fuser_release_year)
strategy.add_attribute_fuser("developer", fuser_developer)
strategy.add_attribute_fuser("platform", fuser_platform)
strategy.add_attribute_fuser("series", fuser_series)
strategy.add_attribute_fuser("publisher", fuser_publisher)
strategy.add_attribute_fuser("criticScore", fuser_critic_score)
strategy.add_attribute_fuser("userScore", fuser_user_score)
strategy.add_attribute_fuser("ESRB", fuser_esrb)
strategy.add_attribute_fuser("globalSales", fuser_numeric_prefer_sales)

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
PIPELINE SNAPSHOT 03 END
============================================================

