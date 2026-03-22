# CRITICAL: Do not adjust output file names.
# CRITICAL: Always use OUTPUT_DIR for all output paths (correspondences, fusion, debug).

from PyDI.io import load_parquet, load_csv, load_xml
from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import (
    StandardBlocker, EmbeddingBlocker, SortedNeighbourhoodBlocker, TokenBlocker,
)
from PyDI.entitymatching import MLBasedMatcher
from PyDI.entitymatching import MaximumBipartiteMatching
from PyDI.fusion import (
    DataFusionStrategy, DataFusionEngine,
    # String: voting, longest_string, shortest_string, most_complete
    # Numeric: median, average, maximum, minimum, sum_values
    # Date: most_recent, earliest
    # List: union, intersection, intersection_k_sources  (pass separator= for delimited strings)
    # Trust: prefer_higher_trust, favour_sources
    voting, longest_string, most_complete,
    median, average,
    most_recent, earliest,
    union, intersection, intersection_k_sources,
    prefer_higher_trust, favour_sources,
)
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _candidates = [Path.cwd(), Path.cwd() / "agents",
                   Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent,
                   Path(__file__).resolve().parent.parent.parent]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            if str(_path.resolve()) not in sys.path:
                sys.path.append(str(_path.resolve()))
    from list_normalization import detect_list_like_columns, normalize_list_like_columns

# === 0. OUTPUT DIRECTORY ===
# CRITICAL: Use OUTPUT_DIR for ALL output paths. It will be provided in the prompt.
# Do NOT hardcode "output/" — always use os.path.join(OUTPUT_DIR, ...).
OUTPUT_DIR = "output"  # Will be replaced by prompt with the actual run-scoped directory

# === 1. LOAD DATA ===
DATA_DIR = "input/datasets/"
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_parquet(DATA_DIR + "<dataset-1>.parquet", name="dataset_name_1")
good_dataset_name_2 = load_parquet(DATA_DIR + "<dataset-2>.parquet", name="dataset_name_2")
good_dataset_name_3 = load_parquet(DATA_DIR + "<dataset-3>.parquet", name="dataset_name_3")

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# === 2. SCHEMA MATCHING ===
print("Matching Schema")
llm = ChatOpenAI(model="gpt-5.1")
matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_2)
good_dataset_name_2 = good_dataset_name_2.rename(
    columns=schema_correspondences.set_index("target_column")["source_column"].to_dict())

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_3)
good_dataset_name_3 = good_dataset_name_3.rename(
    columns=schema_correspondences.set_index("target_column")["source_column"].to_dict())

# === 3. LIST NORMALIZATION ===
list_like_columns = detect_list_like_columns(
    [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
    exclude_columns={"id", "_id"},
)
if list_like_columns:
    (good_dataset_name_1, good_dataset_name_2, good_dataset_name_3,
     ) = normalize_list_like_columns(
        [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3], list_like_columns)
    print(f"Normalized list-like columns: {', '.join(list_like_columns)}")

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# === 4. BLOCKING ===
# CRITICAL: Use the precomputed blocking config from "5. **BLOCKING CONFIGURATION**".
# See example_pipeline.py for blocker type signatures.

print("Performing Blocking")

blocker_1_2 = StandardBlocker(
    good_dataset_name_1, good_dataset_name_2,
    on=["city"], id_column="id", batch_size=100000, )
blocker_1_3 = StandardBlocker(
    good_dataset_name_1, good_dataset_name_3,
    on=["city"], id_column="id", batch_size=100000, )
blocker_2_3 = StandardBlocker(
    good_dataset_name_2, good_dataset_name_3,
    on=["city"], id_column="id", batch_size=100000, )

# === 5. ENTITY MATCHING (ML-BASED) ===
# CRITICAL: Use matching config from "6. **MATCHING CONFIGURATION**" for comparators.
# See example_pipeline.py for comparator signatures and list_strategy values.
#
# ML matching flow:
#   1. Define comparators (same as rule-based — used as feature extractors)
#   2. Load labeled training pairs (from entity matching testsets)
#   3. Extract features using FeatureExtractor.create_features()
#   4. Train classifier via GridSearchCV
#   5. Match using MLBasedMatcher.match(... trained_classifier=best_model)

print("Matching Entities")

comparators_1_2 = [
    StringComparator(column="name", similarity_function="jaro_winkler"),
    StringComparator(column="city", similarity_function="jaro_winkler", preprocess=str.lower),
    NumericComparator(column="house_number", method="absolute_difference", max_difference=2),
    StringComparator(column="categories", similarity_function="jaccard",
                     preprocess=str.lower, list_strategy="set_jaccard"),
]

comparators_1_3 = [
    StringComparator(column="name", similarity_function="jaro_winkler"),
    StringComparator(column="city", similarity_function="jaro_winkler", preprocess=str.lower),
    NumericComparator(column="house_number", max_difference=2),
    StringComparator(column="categories", similarity_function="jaccard",
                     preprocess=str.lower, list_strategy="set_jaccard"),
]

comparators_2_3 = [
    StringComparator(column="name", similarity_function="jaro_winkler"),
    StringComparator(column="city", similarity_function="jaro_winkler", preprocess=str.lower),
    NumericComparator(column="house_number", max_difference=2),
    StringComparator(column="categories", similarity_function="jaccard",
                     preprocess=str.lower, list_strategy="set_jaccard"),
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

# --- Load labeled training pairs ---
# CRITICAL: Use the entity matching testsets provided to you.
train_1_2 = load_csv("testsets/usecase/<ground_truth_df1_df2_train.csv>",
                      name="ground_truth_df1_df2_train", add_index=False)
train_1_3 = load_csv("testsets/usecase/<ground_truth_df1_df3_train.csv>",
                      name="ground_truth_df1_df3_train", add_index=False)
train_2_3 = load_csv("testsets/usecase/<ground_truth_df2_df3_train.csv>",
                      name="ground_truth_df2_df3_train", add_index=False)

# Resolve pair-ID columns across common naming schemes.
def to_pair_ids(df):
    known_pairs = [("id1", "id2"), ("id_a", "id_b"), ("left_id", "right_id"), ("source_id", "target_id")]
    for left_col, right_col in known_pairs:
        if left_col in df.columns and right_col in df.columns:
            out = df[[left_col, right_col]].copy()
            out.columns = ["id1", "id2"]
            return out
    id_like = [c for c in df.columns
               if c != "label" and ("id" in str(c).lower() or str(c).lower().endswith("_id"))]
    if len(id_like) >= 2:
        out = df[[id_like[0], id_like[1]]].copy()
        out.columns = ["id1", "id2"]
        return out
    raise ValueError(f"Could not infer pair ID columns from: {list(df.columns)}")

# --- Extract features ---
train_1_2_features = feature_extractor_1_2.create_features(
    good_dataset_name_1, good_dataset_name_2, to_pair_ids(train_1_2),
    labels=train_1_2["label"], id_column="id",
)
train_1_3_features = feature_extractor_1_3.create_features(
    good_dataset_name_1, good_dataset_name_3, to_pair_ids(train_1_3),
    labels=train_1_3["label"], id_column="id",
)
train_2_3_features = feature_extractor_2_3.create_features(
    good_dataset_name_2, good_dataset_name_3, to_pair_ids(train_2_3),
    labels=train_2_3["label"], id_column="id",
)

# --- Prepare ML training data ---
def split_features_labels(features_df):
    feat_cols = [c for c in features_df.columns if c not in ["id1", "id2", "label"]]
    return features_df[feat_cols], features_df["label"]

X_1_2, y_1_2 = split_features_labels(train_1_2_features)
X_1_3, y_1_3 = split_features_labels(train_1_3_features)
X_2_3, y_2_3 = split_features_labels(train_2_3_features)

# --- Model selection via GridSearchCV ---
param_grids = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5], "class_weight": ["balanced", None]},
    },
    "LogisticRegression": {
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {"C": [0.1, 1.0, 10.0], "class_weight": ["balanced", None]},
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {"n_estimators": [50, 100], "learning_rate": [0.1, 0.2], "max_depth": [3, 5]},
    },
    "SVM": {
        "model": SVC(random_state=42, probability=True),
        "params": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"],
                    "class_weight": ["balanced", None]},
    },
}

scorer = make_scorer(f1_score)
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_models = []
for X_train, y_train in [(X_1_2, y_1_2), (X_1_3, y_1_3), (X_2_3, y_2_3)]:
    best_score, best_model = -1, None
    for name, cfg in param_grids.items():
        gs = GridSearchCV(cfg["model"], cfg["params"], scoring=scorer, cv=cv_folds, n_jobs=-1)
        gs.fit(X_train, y_train)
        if gs.best_score_ > best_score:
            best_model, best_score = gs.best_estimator_, gs.best_score_
    best_models.append(best_model)

# --- Run ML matching ---
# CRITICAL: Dataset order in match() must correspond to testset column order.
ml_matcher_1_2 = MLBasedMatcher(feature_extractor_1_2)
ml_matcher_1_3 = MLBasedMatcher(feature_extractor_1_3)
ml_matcher_2_3 = MLBasedMatcher(feature_extractor_2_3)

ml_correspondences_1_2 = ml_matcher_1_2.match(
    good_dataset_name_1, good_dataset_name_2, candidates=blocker_1_2,
    id_column="id", trained_classifier=best_models[0],
)
ml_correspondences_1_3 = ml_matcher_1_3.match(
    good_dataset_name_1, good_dataset_name_3, candidates=blocker_1_3,
    id_column="id", trained_classifier=best_models[1],
)
ml_correspondences_2_3 = ml_matcher_2_3.match(
    good_dataset_name_2, good_dataset_name_3, candidates=blocker_2_3,
    id_column="id", trained_classifier=best_models[2],
)

# === 6. SAVE CORRESPONDENCES (MANDATORY) ===
CORR_DIR = os.path.join(OUTPUT_DIR, "correspondences")
os.makedirs(CORR_DIR, exist_ok=True)
ml_correspondences_1_2.to_csv(os.path.join(CORR_DIR, "correspondences_dataset_name_1_dataset_name_2.csv"), index=False)
ml_correspondences_1_3.to_csv(os.path.join(CORR_DIR, "correspondences_dataset_name_1_dataset_name_3.csv"), index=False)
ml_correspondences_2_3.to_csv(os.path.join(CORR_DIR, "correspondences_dataset_name_2_dataset_name_3.csv"), index=False)

all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3], ignore_index=True)

# === 7. DATA FUSION ===
# See example_pipeline.py for full resolver reference and choosing guidelines.
# DO NOT write custom fusers. Use only PyDI built-in resolvers listed above.

print("Fusing Data")

trust_map = {"dataset_name_1": 3, "dataset_name_2": 2, "dataset_name_3": 1}

strategy = DataFusionStrategy("ml_fusion_strategy")

strategy.add_attribute_fuser("name", voting)
strategy.add_attribute_fuser("street", most_complete)
strategy.add_attribute_fuser("city", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("state", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("country", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("revenue", median)
strategy.add_attribute_fuser("latitude", average)
strategy.add_attribute_fuser("longitude", average)
strategy.add_attribute_fuser("house_number", voting)
strategy.add_attribute_fuser("founded", earliest)
strategy.add_attribute_fuser("categories", union, separator="; ")

# === 8. RUN FUSION ===
# CRITICAL: Always include_singletons=True for full fused dataset.
# CRITICAL: Use OUTPUT_DIR for all output paths.

FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
os.makedirs(FUSION_DIR, exist_ok=True)
engine = DataFusionEngine(
    strategy, debug=True, debug_format="json",
    debug_file=os.path.join(FUSION_DIR, "debug_fusion_data.jsonl"),
)

fused_result = engine.run(
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=True,
)

fused_result.to_csv(os.path.join(FUSION_DIR, "fusion_data.csv"), index=False)
