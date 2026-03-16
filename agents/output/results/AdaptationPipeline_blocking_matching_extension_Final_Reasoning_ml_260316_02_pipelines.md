# Pipeline Snapshots

notebook_name=AdaptationPipeline_blocking_matching_extension_Final_Reasoning
matcher_mode=ml

============================================================
PIPELINE SNAPSHOT 01 START
============================================================
node_index=6
node_name=execute_pipeline
accuracy_score=57.14%
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
)
from langchain_openai import ChatOpenAI
from PyDI.schemamatching import LLMBasedSchemaMatcher

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
from dotenv import load_dotenv
import os


# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

amazon_small = load_parquet(
    "output/schema-matching/amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    "output/schema-matching/goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    "output/schema-matching/metabooks_small.parquet",
    name="metabooks_small",
)

# create id columns
amazon_small["id"] = amazon_small["id"]
goodreads_small["id"] = goodreads_small["id"]
metabooks_small["id"] = metabooks_small["id"]

datasets = [amazon_small, goodreads_small, metabooks_small]


# --------------------------------
# Perform Schema Matching (LLM-based matching)
# The resulting columns for all datasets will have the schema of amazon_small.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

schema_correspondences = matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)


# --------------------------------
# Perform Blocking using precomputed configuration
# --------------------------------

print("Performing Blocking")

blocker_goodreads_amazon = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_amazon = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_goodreads = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publisher"],
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

lower_strip = lambda x: str(x).lower().strip()

comparators_goodreads_amazon = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=365,
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_metabooks_amazon = [
    StringComparator(
        column="title",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_metabooks_goodreads = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
    ),
    NumericComparator(
        column="page_count",
        max_difference=10.0,
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=366,
    ),
]

feature_extractor_goodreads_amazon = FeatureExtractor(comparators_goodreads_amazon)
feature_extractor_metabooks_amazon = FeatureExtractor(comparators_metabooks_amazon)
feature_extractor_metabooks_goodreads = FeatureExtractor(comparators_metabooks_goodreads)


# --------------------------------
# Load ground truth correspondences (ML training/test)
# --------------------------------

train_goodreads_amazon = load_csv(
    "input/datasets/books/testsets/goodreads_2_amazon.csv",
    name="ground_truth_goodreads_amazon_train",
    add_index=False,
)

train_metabooks_amazon = load_csv(
    "input/datasets/books/testsets/metabooks_2_amazon.csv",
    name="ground_truth_metabooks_amazon_train",
    add_index=False,
)

train_metabooks_goodreads = load_csv(
    "input/datasets/books/testsets/metabooks_2_goodreads.csv",
    name="ground_truth_metabooks_goodreads_train",
    add_index=False,
)


# --------------------------------
# Extract features
# --------------------------------

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
    col
    for col in train_goodreads_amazon_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_goodreads_amazon = train_goodreads_amazon_features[feat_cols_goodreads_amazon]
y_train_goodreads_amazon = train_goodreads_amazon_features["label"]

feat_cols_metabooks_amazon = [
    col
    for col in train_metabooks_amazon_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_metabooks_amazon = train_metabooks_amazon_features[feat_cols_metabooks_amazon]
y_train_metabooks_amazon = train_metabooks_amazon_features["label"]

feat_cols_metabooks_goodreads = [
    col
    for col in train_metabooks_goodreads_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_metabooks_goodreads = train_metabooks_goodreads_features[feat_cols_metabooks_goodreads]
y_train_metabooks_goodreads = train_metabooks_goodreads_features["label"]

training_datasets = [
    (X_train_goodreads_amazon, y_train_goodreads_amazon),
    (X_train_metabooks_amazon, y_train_metabooks_amazon),
    (X_train_metabooks_goodreads, y_train_metabooks_goodreads),
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


print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_goodreads_amazon = clusterer.cluster(ml_correspondences_goodreads_amazon)
ml_correspondences_metabooks_amazon = clusterer.cluster(ml_correspondences_metabooks_amazon)
ml_correspondences_metabooks_goodreads = clusterer.cluster(ml_correspondences_metabooks_goodreads)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_goodreads_amazon.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_amazon_small.csv",
    ),
    index=False,
)

ml_correspondences_metabooks_amazon.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_amazon_small.csv",
    ),
    index=False,
)

ml_correspondences_metabooks_goodreads.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_goodreads_small.csv",
    ),
    index=False,
)


print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_goodreads_amazon,
        ml_correspondences_metabooks_amazon,
        ml_correspondences_metabooks_goodreads,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")
strategy.add_attribute_fuser("title", longest_string)
strategy.add_attribute_fuser("author", longest_string)
strategy.add_attribute_fuser("publish_year", longest_string)
strategy.add_attribute_fuser("publisher", longest_string)

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
accuracy_score=56.43%
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
    union,
)
from langchain_openai import ChatOpenAI
from PyDI.schemamatching import LLMBasedSchemaMatcher

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import re
import html


# --------------------------------
# Helpers
# --------------------------------

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", safe_str(text)).strip()


def clean_html(text):
    return html.unescape(normalize_whitespace(text))


def lower_strip(x):
    return clean_html(x).lower().strip()


def normalize_title(text):
    text = clean_html(text).lower().strip()
    text = re.sub(r"\([^)]*(goodreads author|editor|illustrator|foreword|introduction)[^)]*\)", "", text)
    text = re.sub(r"\([^)]{1,40}\)$", "", text)
    text = re.sub(r"\s*:\s*", " : ", text)
    text = re.sub(r"\s+", " ", text).strip(" -:;,")
    return text


def normalize_author(text):
    text = clean_html(text)
    text = re.sub(r"\([^)]*(goodreads author|editor|illustrator|foreword|introduction)[^)]*\)", "", text, flags=re.I)
    text = re.sub(r"\bet al\.?\b", "", text, flags=re.I)
    text = normalize_whitespace(text).strip(" ,;")
    return text.lower()


def normalize_publisher(text):
    text = clean_html(text).lower().strip()
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\b(books usa|usa|trade division|publishing company|publishing|publishers|publisher)\b", "", text)
    text = text.replace("&", " and ")
    text = re.sub(r"\bco\.\b", "company", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;.-")
    return text


def normalize_language(text):
    text = clean_html(text).lower().strip()
    mapping = {
        "english": "English",
        "en": "English",
        "eng": "English",
    }
    return mapping.get(text, clean_html(text).strip())


def normalize_genres_value(value):
    if pd.isna(value):
        return []
    text = clean_html(value)
    if text.lower() == "nan":
        return []
    parts = re.split(r"[,;/|]", text)
    cleaned = []
    seen = set()
    for part in parts:
        g = normalize_whitespace(part).lower()
        if not g:
            continue
        if g in {"fiction & literature", "fiction and literature"}:
            g = "fiction"
        if g not in seen:
            seen.add(g)
            cleaned.append(g.title())
    return cleaned


def normalize_page_count(value):
    if pd.isna(value):
        return pd.NA
    try:
        return int(float(value))
    except Exception:
        return pd.NA


def normalize_publish_year(value):
    if pd.isna(value):
        return pd.NA
    try:
        year = int(float(value))
        if 0 < year < 3000:
            return year
    except Exception:
        pass
    return pd.NA


def choose_best_string(series, normalizer=None, prefer_shortest=False):
    vals = [v for v in series if not pd.isna(v) and safe_str(v).strip() and safe_str(v).strip().lower() != "nan"]
    if not vals:
        return pd.NA
    if normalizer is None:
        normalizer = lambda x: clean_html(x).strip()
    grouped = {}
    for v in vals:
        key = normalizer(v)
        if not key:
            continue
        grouped.setdefault(key, []).append(v)
    if not grouped:
        return clean_html(vals[0]).strip()
    ranked = sorted(
        grouped.items(),
        key=lambda kv: (
            -len(kv[1]),
            min(len(clean_html(x).strip()) for x in kv[1]) if prefer_shortest else -max(len(clean_html(x).strip()) for x in kv[1]),
        ),
    )
    best_key, originals = ranked[0]
    if prefer_shortest:
        return min((clean_html(x).strip() for x in originals), key=len)
    return max((clean_html(x).strip() for x in originals), key=len)


def fuse_title(series):
    return choose_best_string(series, normalizer=normalize_title, prefer_shortest=True)


def fuse_author(series):
    return choose_best_string(series, normalizer=normalize_author, prefer_shortest=True)


def fuse_publisher(series):
    return choose_best_string(series, normalizer=normalize_publisher, prefer_shortest=True)


def fuse_language(series):
    vals = [normalize_language(v) for v in series if not pd.isna(v) and safe_str(v).strip() and safe_str(v).strip().lower() != "nan"]
    if not vals:
        return pd.NA
    counts = pd.Series(vals).value_counts()
    return counts.index[0]


def fuse_numeric_majority(series):
    vals = []
    for v in series:
        if pd.isna(v):
            continue
        try:
            vals.append(int(float(v)))
        except Exception:
            continue
    if not vals:
        return pd.NA
    counts = pd.Series(vals).value_counts()
    return int(counts.index[0])


def fuse_publish_year(series):
    vals = [normalize_publish_year(v) for v in series]
    vals = [v for v in vals if not pd.isna(v)]
    if not vals:
        return pd.NA
    counts = pd.Series(vals).value_counts()
    return int(counts.index[0])


def fuse_page_count(series):
    vals = [normalize_page_count(v) for v in series]
    vals = [v for v in vals if not pd.isna(v)]
    if not vals:
        return pd.NA
    counts = pd.Series(vals).value_counts()
    return int(counts.index[0])


def genres_union(series):
    all_genres = []
    seen = set()
    for value in series:
        for genre in normalize_genres_value(value):
            if genre not in seen:
                seen.add(genre)
                all_genres.append(genre)
    if not all_genres:
        return pd.NA
    return ", ".join(all_genres)


def prepare_dataset(df):
    df = df.copy()

    if "title" in df.columns:
        df["title"] = df["title"].apply(lambda x: clean_html(x) if not pd.isna(x) else x)
        df["title_norm"] = df["title"].apply(normalize_title)

    if "author" in df.columns:
        df["author"] = df["author"].apply(lambda x: clean_html(x) if not pd.isna(x) else x)
        df["author_norm"] = df["author"].apply(normalize_author)

    if "publisher" in df.columns:
        df["publisher"] = df["publisher"].apply(lambda x: clean_html(x) if not pd.isna(x) else x)
        df["publisher_norm"] = df["publisher"].apply(normalize_publisher)

    if "language" in df.columns:
        df["language"] = df["language"].apply(lambda x: normalize_language(x) if not pd.isna(x) else x)

    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(lambda x: ", ".join(normalize_genres_value(x)) if not pd.isna(x) else x)

    if "page_count" in df.columns:
        df["page_count"] = df["page_count"].apply(normalize_page_count)

    if "publish_year" in df.columns:
        df["publish_year"] = df["publish_year"].apply(normalize_publish_year)

    return df


# --------------------------------
# Prepare Data
# --------------------------------

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

amazon_small = load_parquet(
    "output/schema-matching/amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    "output/schema-matching/goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    "output/schema-matching/metabooks_small.parquet",
    name="metabooks_small",
)

amazon_small["id"] = amazon_small["id"]
goodreads_small["id"] = goodreads_small["id"]
metabooks_small["id"] = metabooks_small["id"]

datasets = [amazon_small, goodreads_small, metabooks_small]


# --------------------------------
# Perform Schema Matching
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

schema_correspondences = matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

amazon_small = prepare_dataset(amazon_small)
goodreads_small = prepare_dataset(goodreads_small)
metabooks_small = prepare_dataset(metabooks_small)


# --------------------------------
# Perform Blocking using precomputed configuration
# --------------------------------

print("Performing Blocking")

blocker_goodreads_amazon = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_amazon = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_goodreads = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publisher"],
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

comparators_goodreads_amazon = [
    StringComparator(
        column="title_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=365,
    ),
    StringComparator(
        column="publisher_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_metabooks_amazon = [
    StringComparator(
        column="title_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_metabooks_goodreads = [
    StringComparator(
        column="title_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="author_norm",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="publisher_norm",
        similarity_function="cosine",
        preprocess=lower_strip,
    ),
    NumericComparator(
        column="page_count",
        max_difference=10.0,
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=366,
    ),
]

feature_extractor_goodreads_amazon = FeatureExtractor(comparators_goodreads_amazon)
feature_extractor_metabooks_amazon = FeatureExtractor(comparators_metabooks_amazon)
feature_extractor_metabooks_goodreads = FeatureExtractor(comparators_metabooks_goodreads)


# --------------------------------
# Load ground truth correspondences
# --------------------------------

train_goodreads_amazon = load_csv(
    "input/datasets/books/testsets/goodreads_2_amazon.csv",
    name="ground_truth_goodreads_amazon_train",
    add_index=False,
)

train_metabooks_amazon = load_csv(
    "input/datasets/books/testsets/metabooks_2_amazon.csv",
    name="ground_truth_metabooks_amazon_train",
    add_index=False,
)

train_metabooks_goodreads = load_csv(
    "input/datasets/books/testsets/metabooks_2_goodreads.csv",
    name="ground_truth_metabooks_goodreads_train",
    add_index=False,
)


# --------------------------------
# Extract features
# --------------------------------

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
    col
    for col in train_goodreads_amazon_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_goodreads_amazon = train_goodreads_amazon_features[feat_cols_goodreads_amazon]
y_train_goodreads_amazon = train_goodreads_amazon_features["label"]

feat_cols_metabooks_amazon = [
    col
    for col in train_metabooks_amazon_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_metabooks_amazon = train_metabooks_amazon_features[feat_cols_metabooks_amazon]
y_train_metabooks_amazon = train_metabooks_amazon_features["label"]

feat_cols_metabooks_goodreads = [
    col
    for col in train_metabooks_goodreads_features.columns
    if col not in ["id1", "id2", "label"]
]
X_train_metabooks_goodreads = train_metabooks_goodreads_features[feat_cols_metabooks_goodreads]
y_train_metabooks_goodreads = train_metabooks_goodreads_features["label"]

training_datasets = [
    (X_train_goodreads_amazon, y_train_goodreads_amazon),
    (X_train_metabooks_amazon, y_train_metabooks_amazon),
    (X_train_metabooks_goodreads, y_train_metabooks_goodreads),
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


print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_goodreads_amazon = clusterer.cluster(ml_correspondences_goodreads_amazon)
ml_correspondences_metabooks_amazon = clusterer.cluster(ml_correspondences_metabooks_amazon)
ml_correspondences_metabooks_goodreads = clusterer.cluster(ml_correspondences_metabooks_goodreads)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_goodreads_amazon.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_amazon_small.csv",
    ),
    index=False,
)

ml_correspondences_metabooks_amazon.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_amazon_small.csv",
    ),
    index=False,
)

ml_correspondences_metabooks_goodreads.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_goodreads_small.csv",
    ),
    index=False,
)


print("Fusing Data")

all_ml_correspondences = pd.concat(
    [
        ml_correspondences_goodreads_amazon,
        ml_correspondences_metabooks_amazon,
        ml_correspondences_metabooks_goodreads,
    ],
    ignore_index=True,
)

strategy = DataFusionStrategy("ml_fusion_strategy")
strategy.add_attribute_fuser("title", fuse_title)
strategy.add_attribute_fuser("author", fuse_author)
strategy.add_attribute_fuser("publish_year", fuse_publish_year)
strategy.add_attribute_fuser("publisher", fuse_publisher)
strategy.add_attribute_fuser("page_count", fuse_page_count)
strategy.add_attribute_fuser("language", fuse_language)
strategy.add_attribute_fuser("genres", genres_union)

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
accuracy_score=65.00%
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

from PyDI.fusion import DataFusionStrategy, DataFusionEngine
from langchain_openai import ChatOpenAI
from PyDI.schemamatching import LLMBasedSchemaMatcher

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import re
import html


# --------------------------------
# Helpers
# --------------------------------

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", safe_str(text)).strip()


def clean_html(text):
    return html.unescape(normalize_whitespace(text))


def lower_strip(x):
    return clean_html(x).lower().strip()


def normalize_title(text):
    text = clean_html(text).lower().strip()
    text = re.sub(
        r"\([^)]*(goodreads author|editor|illustrator|foreword|introduction)[^)]*\)",
        "",
        text,
    )
    text = re.sub(
        r"\((paperback|hardcover|mass market paperback|kindle edition|ebook|audio cd|library binding|reprint|abridged|unabridged|trophy newbery|book club edition)[^)]*\)",
        "",
        text,
    )
    text = re.sub(r"\s*:\s*(a novel|a memoir|stories|an autobiography|a biography)\s*$", "", text)
    text = re.sub(r"\s*-\s*(a novel|a memoir|stories)\s*$", "", text)
    text = re.sub(r"\([^)]{1,40}\)$", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -:;,")
    return text


def normalize_author(text):
    text = clean_html(text)
    text = re.sub(
        r"\([^)]*(goodreads author|editor|illustrator|foreword|introduction)[^)]*\)",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\bet al\.?\b", "", text, flags=re.I)
    text = text.replace("&", " and ")
    text = normalize_whitespace(text).strip(" ,;")
    return text.lower()


def normalize_publisher(text):
    text = clean_html(text).lower().strip()
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace("&", " and ")
    text = re.sub(
        r"\b(books usa|usa|trade division|publishing company|publishing|publishers|publisher)\b",
        "",
        text,
    )
    text = re.sub(r"\bco\.\b", "company", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;.-")

    aliases = {
        "harpertrophy": "harpercollins",
        "harper trophy": "harpercollins",
        "perennial": "harper perennial",
        "harper perennial library": "harper perennial",
        "perennial classics": "harper perennial",
        "vintage books": "vintage",
        "vintage books a division of random house": "vintage",
        "vintage a division of random house": "vintage",
        "vintage random house": "vintage",
        "simon and schuster trade division": "simon and schuster",
        "simon and schuster inc": "simon and schuster",
        "farrar straus giroux": "farrar, straus and giroux",
        "henry holt and company": "henry holt and co.",
        "algonquin": "algonquin books",
        "ballantine books": "ballantine",
        "delacorte press": "delacorte",
        "harcourt brace": "harcourt",
    }
    return aliases.get(text, text)


def normalize_language(text):
    text = clean_html(text).lower().strip()
    mapping = {
        "english": "English",
        "en": "English",
        "eng": "English",
    }
    return mapping.get(text, clean_html(text).strip())


def canonical_genre(g):
    g = clean_html(g).lower().strip()
    g = g.replace("&", "and")
    g = re.sub(r"\s+", " ", g)

    mapping = {
        "fiction and literature": "Fiction",
        "fiction literature": "Fiction",
        "fiction": "Fiction",
        "literature and fiction": "Fiction",
        "literature": "Literature",
        "childrens": "Children's Books",
        "children": "Children's Books",
        "children's": "Children's Books",
        "kids": "Children's Books",
        "juvenile": "Children's Books",
        "young adult": "Young Adult",
        "ya": "Young Adult",
        "biographies": "Biography",
        "biography": "Biography",
        "memoirs": "Memoir",
        "memoir": "Memoir",
        "historical fiction": "Historical Fiction",
        "historical": "Historical",
        "novels": "Fiction",
        "novel": "Fiction",
        "adult fiction": "Fiction",
        "contemporary fiction": "Contemporary",
        "contemporary": "Contemporary",
        "world war ii": "World War II",
        "wwii": "World War II",
        "science fiction": "Science Fiction",
        "sci fi": "Science Fiction",
        "fantasy fiction": "Fantasy",
        "fantasy": "Fantasy",
        "romantic": "Romance",
        "romance": "Romance",
        "classics": "Classics",
        "historical romance": "Romance",
        "american": "Fiction",
    }
    if g in mapping:
        return mapping[g]

    return " ".join(word.capitalize() for word in g.split())


def normalize_genres_value(value):
    if pd.isna(value):
        return []
    text = clean_html(value)
    if text.lower() == "nan":
        return []

    parts = re.split(r"[,;/|]", text)
    cleaned = []
    seen = set()

    for part in parts:
        genre = canonical_genre(part)
        if not genre:
            continue
        if genre not in seen:
            seen.add(genre)
            cleaned.append(genre)

    expanded = list(cleaned)

    if "Historical Fiction" in seen:
        expanded.extend(["Historical", "Fiction", "Literature"])
    if "Romance" in seen:
        expanded.extend(["Fiction"])
    if "Fantasy" in seen:
        expanded.extend(["Fiction"])
    if "Science Fiction" in seen:
        expanded.extend(["Fiction"])
    if "Memoir" in seen:
        expanded.extend(["Biography", "Nonfiction"])
    if "Biography" in seen:
        expanded.extend(["Nonfiction"])
    if "Children's Books" in seen:
        expanded.extend(["Fiction"])
    if "Young Adult" in seen:
        expanded.extend(["Fiction"])
    if "Classics" in seen:
        expanded.extend(["Literature", "Fiction"])

    final = []
    seen2 = set()
    for genre in expanded:
        if genre not in seen2:
            seen2.add(genre)
            final.append(genre)
    return final


def normalize_page_count(value):
    if pd.isna(value):
        return pd.NA
    try:
        v = int(float(value))
        if v > 0:
            return v
    except Exception:
        pass
    return pd.NA


def normalize_publish_year(value):
    if pd.isna(value):
        return pd.NA
    try:
        year = int(float(value))
        if 0 < year < 3000:
            return year
    except Exception:
        pass
    return pd.NA


def choose_best_string(series, normalizer=None):
    vals = [
        v
        for v in series
        if not pd.isna(v) and safe_str(v).strip() and safe_str(v).strip().lower() != "nan"
    ]
    if not vals:
        return pd.NA

    if normalizer is None:
        normalizer = lambda x: clean_html(x).strip()

    grouped = {}
    for v in vals:
        key = normalizer(v)
        if key:
            grouped.setdefault(key, []).append(clean_html(v).strip())

    if not grouped:
        return clean_html(vals[0]).strip()

    def noise_score(text):
        t = clean_html(text).strip()
        penalty = 0
        if "(" in t and ")" in t:
            penalty += 2
        if re.search(r":\s*(a novel|a memoir|stories|an autobiography|a biography)\s*$", t, flags=re.I):
            penalty += 1
        if len(t) < 3:
            penalty += 5
        return penalty

    ranked = sorted(
        grouped.items(),
        key=lambda kv: (
            -len(kv[1]),
            min(noise_score(x) for x in kv[1]),
            min(abs(len(clean_html(x).strip()) - len(kv[0])) for x in kv[1]),
            -max(len(clean_html(x).strip()) for x in kv[1]),
        ),
    )

    _, originals = ranked[0]
    cleaned_originals = sorted(
        set(clean_html(x).strip() for x in originals),
        key=lambda x: (
            noise_score(x),
            abs(len(x) - np.median([len(y) for y in originals])),
            -len(x),
            x,
        ),
    )
    return cleaned_originals[0]


def fuse_title(series):
    return choose_best_string(series, normalizer=normalize_title)


def split_authors(text):
    text = clean_html(text)
    text = re.sub(r"\([^)]*\)", "", text)
    parts = re.split(r"\s+(?:and|&)\s+|,|;|/", text)
    authors = []
    for part in parts:
        a = normalize_whitespace(part).strip(" ,;")
        if a:
            authors.append(a)
    return authors


def fuse_author(series):
    vals = [
        v
        for v in series
        if not pd.isna(v) and safe_str(v).strip() and safe_str(v).strip().lower() != "nan"
    ]
    if not vals:
        return pd.NA

    grouped = {}
    token_counts = {}

    for v in vals:
        key = normalize_author(v)
        if key:
            grouped.setdefault(key, []).append(clean_html(v).strip())
        for a in split_authors(v):
            norm_a = normalize_author(a)
            if norm_a:
                token_counts[norm_a] = token_counts.get(norm_a, 0) + 1

    if grouped:
        best_key, originals = sorted(
            grouped.items(),
            key=lambda kv: (-len(kv[1]), -max(len(x) for x in kv[1]), kv[0]),
        )[0]
        return sorted(set(originals), key=lambda x: (-len(x), x))[0]

    if token_counts:
        best = sorted(token_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return " ".join(w.capitalize() for w in best.split())

    return clean_html(vals[0]).strip()


def fuse_publisher(series):
    vals = [
        v
        for v in series
        if not pd.isna(v) and safe_str(v).strip() and safe_str(v).strip().lower() != "nan"
    ]
    if not vals:
        return pd.NA

    grouped = {}
    for v in vals:
        key = normalize_publisher(v)
        if key:
            grouped.setdefault(key, []).append(clean_html(v).strip())

    if not grouped:
        return clean_html(vals[0]).strip()

    canonical_surface = {
        "harpercollins": "HarperCollins",
        "harper perennial": "Harper Perennial",
        "vintage": "Vintage",
        "simon and schuster": "Simon & Schuster",
        "farrar, straus and giroux": "Farrar, Straus and Giroux",
        "henry holt and co.": "Henry Holt and Co.",
        "algonquin books": "Algonquin Books",
        "ballantine": "Ballantine",
        "delacorte": "Delacorte Press",
        "harcourt": "Harcourt",
    }

    best_key, originals = sorted(
        grouped.items(),
        key=lambda kv: (
            -len(kv[1]),
            -(1 if kv[0] in canonical_surface else 0),
            -max(len(x) for x in kv[1]),
            kv[0],
        ),
    )[0]

    if best_key in canonical_surface:
        return canonical_surface[best_key]

    return sorted(set(originals), key=lambda x: (-len(x), x))[0]


def fuse_language(series):
    vals = [
        normalize_language(v)
        for v in series
        if not pd.isna(v) and safe_str(v).strip() and safe_str(v).strip().lower() != "nan"
    ]
    if not vals:
        return pd.NA
    counts = pd.Series(vals).value_counts()
    return counts.index[0]


def fuse_publish_year(series):
    vals = [normalize_publish_year(v) for v in series]
    vals = [v for v in vals if not pd.isna(v)]
    if not vals:
        return pd.NA

    counts = pd.Series(vals).value_counts()
    top_count = counts.iloc[0]
    top_vals = counts[counts == top_count].index.tolist()

    if len(top_vals) == 1:
        return int(top_vals[0])

    return int(round(float(np.median(vals))))


def fuse_page_count(series):
    vals = [normalize_page_count(v) for v in series]
    vals = [int(v) for v in vals if not pd.isna(v)]
    if not vals:
        return pd.NA

    vals = sorted(vals)
    clusters = []

    for v in vals:
        placed = False
        for cluster in clusters:
            if abs(v - int(round(np.median(cluster)))) <= 16:
                cluster.append(v)
                placed = True
                break
        if not placed:
            clusters.append([v])

    best_cluster = sorted(
        clusters,
        key=lambda c: (
            -len(c),
            -sum(c),
        ),
    )[0]

    return int(round(float(np.median(best_cluster))))


def genres_union(series):
    genre_counts = {}
    for value in series:
        genres = normalize_genres_value(value)
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    if not genre_counts:
        return pd.NA

    kept = []
    for genre, count in sorted(genre_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if count >= 2:
            kept.append(genre)

    if not kept:
        kept = [g for g, _ in sorted(genre_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:6]]

    kept_set = set(kept)

    if "Historical Fiction" in kept_set:
        kept_set.update(["Historical", "Fiction", "Literature"])
    if "Young Adult" in kept_set:
        kept_set.update(["Fiction"])
    if "Children's Books" in kept_set:
        kept_set.update(["Fiction"])
    if "Memoir" in kept_set:
        kept_set.update(["Biography", "Nonfiction"])
    if "Biography" in kept_set:
        kept_set.update(["Nonfiction"])
    if "Romance" in kept_set:
        kept_set.update(["Fiction"])
    if "Fantasy" in kept_set:
        kept_set.update(["Fiction"])
    if "Science Fiction" in kept_set:
        kept_set.update(["Fiction"])
    if "Classics" in kept_set:
        kept_set.update(["Literature", "Fiction"])

    priority = {
        "Fiction": 1,
        "Literature": 2,
        "Historical Fiction": 3,
        "Historical": 4,
        "Romance": 5,
        "Fantasy": 6,
        "Science Fiction": 7,
        "Young Adult": 8,
        "Children's Books": 9,
        "Biography": 10,
        "Memoir": 11,
        "Nonfiction": 12,
        "Contemporary": 13,
        "World War II": 14,
        "Classics": 15,
    }

    kept = sorted(kept_set, key=lambda x: (priority.get(x, 999), x))
    return ", ".join(kept)


def ensure_columns(df, columns):
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def prepare_dataset(df):
    df = df.copy()

    if "title" in df.columns:
        df["title"] = df["title"].apply(lambda x: clean_html(x) if not pd.isna(x) else x)

    if "author" in df.columns:
        df["author"] = df["author"].apply(lambda x: clean_html(x) if not pd.isna(x) else x)

    if "publisher" in df.columns:
        df["publisher"] = df["publisher"].apply(lambda x: clean_html(x) if not pd.isna(x) else x)

    if "language" in df.columns:
        df["language"] = df["language"].apply(lambda x: normalize_language(x) if not pd.isna(x) else x)

    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(
            lambda x: ", ".join(normalize_genres_value(x)) if not pd.isna(x) else x
        )

    if "page_count" in df.columns:
        df["page_count"] = df["page_count"].apply(normalize_page_count)

    if "publish_year" in df.columns:
        df["publish_year"] = df["publish_year"].apply(normalize_publish_year)

    return df


def ensure_score_column(corr):
    corr = corr.copy()
    if "score" in corr.columns:
        corr["score"] = pd.to_numeric(corr["score"], errors="coerce").fillna(1.0)
    elif "similarity" in corr.columns:
        corr["score"] = pd.to_numeric(corr["similarity"], errors="coerce").fillna(1.0)
    elif "probability" in corr.columns:
        corr["score"] = pd.to_numeric(corr["probability"], errors="coerce").fillna(1.0)
    elif "prediction_proba" in corr.columns:
        corr["score"] = pd.to_numeric(corr["prediction_proba"], errors="coerce").fillna(1.0)
    else:
        corr["score"] = 1.0
    return corr


def build_global_correspondences(correspondence_tables):
    edges = []
    score_map = {}

    for corr in correspondence_tables:
        if corr is None or len(corr) == 0:
            continue
        local = ensure_score_column(corr)
        if "id1" not in local.columns or "id2" not in local.columns:
            continue

        for _, row in local[["id1", "id2", "score"]].dropna(subset=["id1", "id2"]).iterrows():
            id1 = row["id1"]
            id2 = row["id2"]
            score = float(row["score"])
            key = tuple(sorted([id1, id2]))
            if key not in score_map or score > score_map[key]:
                score_map[key] = score
            edges.append((id1, id2))

    parent = {}
    rank = {}

    def find(x):
        if x not in parent:
            parent[x] = x
            rank[x] = 0
            return x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1

    for id1, id2 in edges:
        union(id1, id2)

    components = {}
    for node in list(parent.keys()):
        root = find(node)
        components.setdefault(root, []).append(node)

    rows = []
    for _, members in components.items():
        members = sorted(set(members))
        if len(members) < 2:
            continue
        component_scores = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                key = tuple(sorted([members[i], members[j]]))
                if key in score_map:
                    component_scores.append(score_map[key])

        default_component_score = float(np.mean(component_scores)) if component_scores else 1.0

        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                key = tuple(sorted([members[i], members[j]]))
                rows.append(
                    {
                        "id1": members[i],
                        "id2": members[j],
                        "score": float(score_map.get(key, default_component_score)),
                    }
                )

    return pd.DataFrame(rows, columns=["id1", "id2", "score"])


# --------------------------------
# Prepare Data
# --------------------------------

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

amazon_small = load_parquet(
    "output/schema-matching/amazon_small.parquet",
    name="amazon_small",
)

goodreads_small = load_parquet(
    "output/schema-matching/goodreads_small.parquet",
    name="goodreads_small",
)

metabooks_small = load_parquet(
    "output/schema-matching/metabooks_small.parquet",
    name="metabooks_small",
)

amazon_small["id"] = amazon_small["id"]
goodreads_small["id"] = goodreads_small["id"]
metabooks_small["id"] = metabooks_small["id"]


# --------------------------------
# Perform Schema Matching
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(amazon_small, goodreads_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
goodreads_small = goodreads_small.rename(columns=rename_map)

schema_correspondences = matcher.match(amazon_small, metabooks_small)
rename_map = schema_correspondences.set_index("target_column")["source_column"].to_dict()
metabooks_small = metabooks_small.rename(columns=rename_map)

required_columns = [
    "id",
    "title",
    "author",
    "publisher",
    "publish_year",
    "page_count",
    "language",
    "genres",
]

amazon_small = ensure_columns(amazon_small, required_columns)
goodreads_small = ensure_columns(goodreads_small, required_columns)
metabooks_small = ensure_columns(metabooks_small, required_columns)

amazon_small = prepare_dataset(amazon_small)
goodreads_small = prepare_dataset(goodreads_small)
metabooks_small = prepare_dataset(metabooks_small)


# --------------------------------
# Perform Blocking using precomputed configuration
# --------------------------------

print("Performing Blocking")

blocker_goodreads_amazon = EmbeddingBlocker(
    goodreads_small,
    amazon_small,
    text_cols=["title", "author", "publish_year"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_amazon = EmbeddingBlocker(
    metabooks_small,
    amazon_small,
    text_cols=["title", "author", "publisher"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=15,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column="id",
)

blocker_metabooks_goodreads = EmbeddingBlocker(
    metabooks_small,
    goodreads_small,
    text_cols=["title", "author", "publisher"],
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

comparators_goodreads_amazon = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=365,
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_metabooks_amazon = [
    StringComparator(
        column="title",
        similarity_function="cosine",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
    StringComparator(
        column="publisher",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
        list_strategy="concatenate",
    ),
]

comparators_metabooks_goodreads = [
    StringComparator(
        column="title",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="author",
        similarity_function="jaro_winkler",
        preprocess=lower_strip,
    ),
    StringComparator(
        column="publisher",
        similarity_function="cosine",
        preprocess=lower_strip,
    ),
    NumericComparator(
        column="page_count",
        max_difference=10.0,
    ),
    DateComparator(
        column="publish_year",
        max_days_difference=366,
    ),
]

feature_extractor_goodreads_amazon = FeatureExtractor(comparators_goodreads_amazon)
feature_extractor_metabooks_amazon = FeatureExtractor(comparators_metabooks_amazon)
feature_extractor_metabooks_goodreads = FeatureExtractor(comparators_metabooks_goodreads)


# --------------------------------
# Load ground truth correspondences
# --------------------------------

train_goodreads_amazon = load_csv(
    "input/datasets/books/testsets/goodreads_2_amazon.csv",
    name="ground_truth_goodreads_amazon_train",
    add_index=False,
)

train_metabooks_amazon = load_csv(
    "input/datasets/books/testsets/metabooks_2_amazon.csv",
    name="ground_truth_metabooks_amazon_train",
    add_index=False,
)

train_metabooks_goodreads = load_csv(
    "input/datasets/books/testsets/metabooks_2_goodreads.csv",
    name="ground_truth_metabooks_goodreads_train",
    add_index=False,
)


# --------------------------------
# Extract features
# --------------------------------

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
    col for col in train_goodreads_amazon_features.columns if col not in ["id1", "id2", "label"]
]
X_train_goodreads_amazon = train_goodreads_amazon_features[feat_cols_goodreads_amazon]
y_train_goodreads_amazon = train_goodreads_amazon_features["label"]

feat_cols_metabooks_amazon = [
    col for col in train_metabooks_amazon_features.columns if col not in ["id1", "id2", "label"]
]
X_train_metabooks_amazon = train_metabooks_amazon_features[feat_cols_metabooks_amazon]
y_train_metabooks_amazon = train_metabooks_amazon_features["label"]

feat_cols_metabooks_goodreads = [
    col for col in train_metabooks_goodreads_features.columns if col not in ["id1", "id2", "label"]
]
X_train_metabooks_goodreads = train_metabooks_goodreads_features[feat_cols_metabooks_goodreads]
y_train_metabooks_goodreads = train_metabooks_goodreads_features["label"]

training_datasets = [
    (X_train_goodreads_amazon, y_train_goodreads_amazon),
    (X_train_metabooks_amazon, y_train_metabooks_amazon),
    (X_train_metabooks_goodreads, y_train_metabooks_goodreads),
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

ml_correspondences_goodreads_amazon = ensure_score_column(ml_correspondences_goodreads_amazon)
ml_correspondences_metabooks_amazon = ensure_score_column(ml_correspondences_metabooks_amazon)
ml_correspondences_metabooks_goodreads = ensure_score_column(ml_correspondences_metabooks_goodreads)


print("Fusing Data")

clusterer = MaximumBipartiteMatching()
ml_correspondences_goodreads_amazon = ensure_score_column(
    clusterer.cluster(ml_correspondences_goodreads_amazon)
)
ml_correspondences_metabooks_amazon = ensure_score_column(
    clusterer.cluster(ml_correspondences_metabooks_amazon)
)
ml_correspondences_metabooks_goodreads = ensure_score_column(
    clusterer.cluster(ml_correspondences_metabooks_goodreads)
)

CORR_DIR = "output/correspondences"
os.makedirs(CORR_DIR, exist_ok=True)

ml_correspondences_goodreads_amazon.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_goodreads_small_amazon_small.csv",
    ),
    index=False,
)

ml_correspondences_metabooks_amazon.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_amazon_small.csv",
    ),
    index=False,
)

ml_correspondences_metabooks_goodreads.to_csv(
    os.path.join(
        CORR_DIR,
        "correspondences_metabooks_small_goodreads_small.csv",
    ),
    index=False,
)


print("Fusing Data")

all_ml_correspondences = build_global_correspondences(
    [
        ml_correspondences_goodreads_amazon,
        ml_correspondences_metabooks_amazon,
        ml_correspondences_metabooks_goodreads,
    ]
)

strategy = DataFusionStrategy("ml_fusion_strategy")
strategy.add_attribute_fuser("title", fuse_title)
strategy.add_attribute_fuser("author", fuse_author)
strategy.add_attribute_fuser("publish_year", fuse_publish_year)
strategy.add_attribute_fuser("publisher", fuse_publisher)
strategy.add_attribute_fuser("page_count", fuse_page_count)
strategy.add_attribute_fuser("language", fuse_language)
strategy.add_attribute_fuser("genres", genres_union)

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

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
```

============================================================
PIPELINE SNAPSHOT 03 END
============================================================

