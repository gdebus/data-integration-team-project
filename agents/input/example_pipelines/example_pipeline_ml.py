# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_parquet, load_csv, load_xml

from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import MLBasedMatcher

from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union, prefer_higher_trust
from PyDI.fusion import DataFusionEvaluator, tokenized_match

from PyDI.schemamatching import LLMBasedSchemaMatcher
from PyDI.entitymatching import MaximumBipartiteMatching
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
    _candidates = [
        Path.cwd(),
        Path.cwd() / "agents",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
        Path(__file__).resolve().parent.parent.parent.parent,
    ]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            _path_str = str(_path.resolve())
            if _path_str not in sys.path:
                sys.path.append(_path_str)
    from list_normalization import detect_list_like_columns, normalize_list_like_columns

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

# Define dataset paths
DATA_DIR = "input/datasets/"

# Define API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load the first dataset
good_dataset_name_1 = load_parquet(
    DATA_DIR + "<path-and-file-name-of-dataset-1>.parquet",
    name="dataset_name_1",
)

# Load the second dataset
good_dataset_name_2 = load_parquet(
    DATA_DIR + "<path-and-file-name-of-dataset-2>.parquet",
    name="dataset_name_2",
)

# Load the third dataset
good_dataset_name_3 = load_parquet(
    DATA_DIR + "<path-and-file-name-of-dataset-3>.parquet",
    name="dataset_name_3",
)

# create id columns (replace with actual id columns)
good_dataset_name_1["id"] = good_dataset_name_1["<dataset1_id_column>"]
good_dataset_name_2["id"] = good_dataset_name_2["<dataset2_id_column>"]
good_dataset_name_3["id"] = good_dataset_name_3["<dataset3_id_column>"]

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# CRITICAL INSTRUCTION FOR AGENTS:
# The here implemented schema matching will match the schema of dataset2 and dataset3 to the schema of dataset1. Therefore, the resulting columns for all
# datasets will have the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True
)

# match schema of good_dataset_name_1 with good_dataset_name_2 and rename schema of good_dataset_name_2
schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_2)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
good_dataset_name_2 = good_dataset_name_2.rename(columns=rename_map)

# match schema of good_dataset_name_1 with good_dataset_name_3 and rename schema of good_dataset_name_3
schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_3)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
good_dataset_name_3 = good_dataset_name_3.rename(columns=rename_map)

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

# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# YOU MUST USE THE PROCOMPUTED BLOCKER TYPES AND PARAMETER SETTINGS PROVIDED TO YOU LATER IN JSON UNDER "5. **BLOCKING CONFIGURATION**"!
# In the following are only implementation EXAMPLES (NOT THE PROCOMPUTED BLOCKER TYPES). YOU MUST USE THE PROCOMPUTED BLOCKER TYPES AND PARAMETER SETTINGS WHICH ARE PROVIDED TO YOU LATER IN THIS PROMT IN JSON
# --------------------------------

print("Performing Blocking")

# Embedding blocker (semantic_similarity) example:
embedding_blocker_dataset1_2_dataset2 = EmbeddingBlocker(
    good_dataset_name_1, good_dataset_name_2, # name of the datasets
    text_cols=['city'], # column which should be used to perform the blocking on
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,          # Top 20 most similar
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column='id'
)

# Standard blocker example:
blocker_1_2 = StandardBlocker(
    good_dataset_name_1, good_dataset_name_2,
    on=['city'],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column='id'
)

# TokenBlocker example (token_blocking)
blocker_1_3 = TokenBlocker(
    good_dataset_name_1, good_dataset_name_3,
    column="name",
    min_token_len=3,
    ngram_size=2,
    ngram_type="word",
    id_column="id",
    output_dir="output/blocking"
)

# Sorted NeighbourhoodBlocker example
blocker_2_3 = SortedNeighbourhoodBlocker(
    good_dataset_name_2, good_dataset_name_3,
    key="name",
    window=20,
    id_column="id",
    output_dir="output/blocking"
)


# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# You MUST use the matching configuration supplied to you under "6. **MATCHING CONFIGURATION**" to set the correct comparators in the following.
# --------------------------------

comparators_1_2 = [
    # Name similarity
    StringComparator(
        column='name_norm',
        similarity_function='jaccard', 
        # no preprocessing needed
    ),
    
    # street name similarity
    StringComparator(
        column='street',
        similarity_function='jaccard', 
        preprocess=str.lower,
    ),

    # house number similarity
    NumericComparator(
        column='house_number',
        max_difference=2,
    ),

    # category similarity
    StringComparator(
        column='categories',
        similarity_function='jaccard',
        preprocess=str.lower,
        list_strategy='concatenate' # Handle list attribute by concatenation
    )
]

comparators_1_3 = [
    # Name similarity
    StringComparator(
        column='name_norm',
        similarity_function='jaccard', 
        # no preprocessing needed
    ),
    
    # street name similarity
    StringComparator(
        column='street',
        similarity_function='jaccard', 
        preprocess=str.lower,
    ),

    # house number similarity
    NumericComparator(
        column='house_number',
        max_difference=2,
    ),

    # category similarity
    StringComparator(
        column='categories',
        similarity_function='jaccard',
        preprocess=str.lower,
        list_strategy='concatenate' # Handle list attribute by concatenation
    )
]

comparators_2_3 = [
    # Name similarity
    StringComparator(
        column='name_norm',
        similarity_function='jaccard', 
        # no preprocessing needed
    ),
    
    # street name similarity
    StringComparator(
        column='street',
        similarity_function='jaccard', 
        preprocess=str.lower,
    ),

    # house number similarity
    NumericComparator(
        column='house_number',
        max_difference=2,
    ),

    # category similarity
    StringComparator(
        column='categories',
        similarity_function='jaccard',
        preprocess=str.lower,
        list_strategy='concatenate' # Handle list attribute by concatenation
    )
]

feature_extractor_1_2 = FeatureExtractor(comparators_1_2)
feature_extractor_1_3 = FeatureExtractor(comparators_1_3)
feature_extractor_2_3 = FeatureExtractor(comparators_2_3)

# Load ground truth correspondences (ML training/test)
# use the entity matching testsets provided to you
# dataset1 <-> dataset2 training/test pairs
train_1_2 = load_csv(
    "testsets/usecase/<ground_truth_df1_df2_train.csv>",
    name="ground_truth_df1_df2_train",
    add_index=False
)

# dataset1 <-> dataset3 training/test pairs
train_1_3 = load_csv(
    "testsets/usecase/<ground_truth_df1_df3_train.csv>",
    name="ground_truth_df1_df3_train",
    add_index=False
)

# dataset2 <-> dataset3 training/test pairs
train_2_3 = load_csv(
    "testsets/usecase/<ground_truth_df2_df3_train.csv>",
    name="ground_truth_df2_df3_train",
    add_index=False
)

# Resolve pair-ID columns across common naming schemes.
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

# Extract features
train_1_2_features = feature_extractor_1_2.create_features(
    good_dataset_name_1, good_dataset_name_2, to_pair_ids(train_1_2), labels=train_1_2['label'], id_column='id'
)

train_1_3_features = feature_extractor_1_3.create_features(
    good_dataset_name_1, good_dataset_name_3, to_pair_ids(train_1_3), labels=train_1_3['label'], id_column='id'
)

train_2_3_features = feature_extractor_2_3.create_features(
    good_dataset_name_2, good_dataset_name_3, to_pair_ids(train_2_3), labels=train_2_3['label'], id_column='id'
)

# Prepare data for ML training
feat_cols_1_2 = [col for col in train_1_2_features.columns if col not in ['id1', 'id2', 'label']]
X_train_1_2 = train_1_2_features[feat_cols_1_2]
y_train_1_2 = train_1_2_features['label']

feat_cols_1_3 = [col for col in train_1_3_features.columns if col not in ['id1', 'id2', 'label']]
X_train_1_3 = train_1_3_features[feat_cols_1_3]
y_train_1_3 = train_1_3_features['label']

feat_cols_2_3 = [col for col in train_2_3_features.columns if col not in ['id1', 'id2', 'label']]
X_train_2_3 = train_2_3_features[feat_cols_2_3]
y_train_2_3 = train_2_3_features['label']

training_datasets = [(X_train_1_2, y_train_1_2), (X_train_1_3, y_train_1_3), (X_train_2_3, y_train_2_3)]

# --------------------------------
# Select Best Model
# --------------------------------

param_grids = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced', None]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.1, 1.0, 10.0],
            'l1_ratio': [0],
            'class_weight': ['balanced', None]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.2],
            'max_depth': [3, 5],
        }
    },
    'SVM': {
        'model': SVC(random_state=42, probability=True),
        'params': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'class_weight': ['balanced', None]
        }
    }
}

scorer = make_scorer(f1_score)
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_models = []

for dataset in training_datasets:
    best_overall_score = -1
    best_overall_model = None

    for model_name, config in param_grids.items():
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            scoring=scorer,
            cv=cv_folds,
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(dataset[0], dataset[1])
        if grid_search.best_score_ > best_overall_score:
            best_overall_model = grid_search.best_estimator_
            best_overall_score = grid_search.best_score_

    best_models.append(best_overall_model)

print("Matching Entities")

# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# the order of good_dataset_name_<num1> and good_dataset_name_<num2> in ml_correspondences_<num1>_<num2> is very important because it must correspond
# to the order of columns within the testsets. The order of columns within the testset is indicated by the filename of the testset file.
# --------------------------------

ml_matcher_1_2 = MLBasedMatcher(feature_extractor_1_2)
ml_matcher_1_3 = MLBasedMatcher(feature_extractor_1_3)
ml_matcher_2_3 = MLBasedMatcher(feature_extractor_2_3)

ml_correspondences_1_2 = ml_matcher_1_2.match(
    good_dataset_name_1, good_dataset_name_2, candidates=blocker_1_2, id_column='id', trained_classifier=best_models[0]
)

ml_correspondences_1_3 = ml_matcher_1_3.match(
    good_dataset_name_1, good_dataset_name_3, candidates=blocker_1_3, id_column='id', trained_classifier=best_models[1]
)

ml_correspondences_2_3 = ml_matcher_2_3.match(
    good_dataset_name_2, good_dataset_name_3, candidates=blocker_2_3, id_column='id', trained_classifier=best_models[2]
)

print("Fusing Data")

# MaximumBipartiteMatching example:
# This will ensure that each entity is matched to at most one entity in the other dataset
clusterer = MaximumBipartiteMatching()
ml_correspondences_1_2 = clusterer.cluster(ml_correspondences_1_2)
ml_correspondences_1_3 = clusterer.cluster(ml_correspondences_1_3)
ml_correspondences_2_3 = clusterer.cluster(ml_correspondences_2_3)

print("Fusing Data")

# -------------- Data Fusion ------------------
# There are following conflict resolution functions available:
# For strings: longest_string, shortest_string, most_complete
# For numerics: average, median, maximum, minimum, sum_values
# For dates: most_recent, earliest
# For lists/sets: union
# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# Prefer built-in PyDI fusers whenever possible.
# If a custom fuser is unavoidable, it MUST accept runtime kwargs and return
# (value, confidence, metadata):
# def my_fuser(inputs, **kwargs): return value, 1.0, {}
# Avoid lambda inputs, context: ... because PyDI passes extra kwargs (e.g., sources).
# For trust-based fusion, register the built-in directly:
# strategy.add_attribute_fuser("<attr>", prefer_higher_trust, trust_map=trust_map)

# merge ML-based correspondences
all_ml_correspondences = pd.concat(
    [ml_correspondences_1_2, ml_correspondences_1_3, ml_correspondences_2_3],
    ignore_index=True
)

# define data fusion strategy
strategy = DataFusionStrategy('ml_fusion_strategy')

strategy.add_attribute_fuser('name', longest_string)
strategy.add_attribute_fuser('street', longest_string)
strategy.add_attribute_fuser('house_number', longest_string)
strategy.add_attribute_fuser('city', longest_string)
strategy.add_attribute_fuser('state', longest_string)
strategy.add_attribute_fuser('postal_code', longest_string)
strategy.add_attribute_fuser('country', longest_string)
strategy.add_attribute_fuser('latitude', longest_string)
strategy.add_attribute_fuser('longitude', longest_string)
strategy.add_attribute_fuser('categories', union)

# run fusion
engine = DataFusionEngine(strategy, debug=True, debug_format='json', debug_file="output/data_fusion/debug_fusion_data.jsonl")

ml_fused_standard_blocker = engine.run(
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_data.csv", index=False)
