from PyDI.io import load_csv
from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import StandardBlocker, TokenBlocker, SortedNeighbourhoodBlocker, EmbeddingBlocker
from PyDI.entitymatching import MLBasedMatcher
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import os
import sys

# Increase recursion limit
sys.setrecursionlimit(3000)

# --------------------------------
# Prepare Data
# --------------------------------

# Define dataset paths
DATA_DIR = "output/schema-matching/"

# Load the datasets
discogs = load_csv(DATA_DIR + "discogs.csv", name="discogs")
lastfm = load_csv(DATA_DIR + "lastfm.csv", name="lastfm")
musicbrainz = load_csv(DATA_DIR + "musicbrainz.csv", name="musicbrainz")

# create id columns
discogs["id"] = discogs["id"]
lastfm["id"] = lastfm["id"]
musicbrainz["id"] = musicbrainz["id"]

datasets = [discogs, lastfm, musicbrainz]

# --------------------------------
# Perform Schema Matching (LLM-based matching)
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

# match schema of discogs with lastfm and rename schema of lastfm
schema_correspondences = matcher.match(discogs, lastfm)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
lastfm = lastfm.rename(columns=rename_map)

# match schema of discogs with musicbrainz and rename schema of musicbrainz
schema_correspondences = matcher.match(discogs, musicbrainz)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
musicbrainz = musicbrainz.rename(columns=rename_map)

# --------------------------------
# Blocking
# --------------------------------

print("Performing Blocking")

# Blocking configuration
blocking_config = {
    "discogs_lastfm": {
        "strategy": "sorted_neighbourhood",
        "columns": ["name"],
        "params": {"window": 20}
    },
    "discogs_musicbrainz": {
        "strategy": "semantic_similarity",
        "columns": ["name", "artist", "release-date"],
        "params": {"top_k": 20}
    },
    "musicbrainz_lastfm": {
        "strategy": "token_blocking",
        "columns": ["name"],
        "params": {"min_token_len": 4}
    }
}

# Sorted NeighbourhoodBlocker for discogs and lastfm
blocker_discogs_lastfm = SortedNeighbourhoodBlocker(
    discogs, lastfm,
    key="name",
    window=20,
    id_column="id",
    output_dir="output/blocking"
)

# EmbeddingBlocker for discogs and musicbrainz
blocker_discogs_musicbrainz = EmbeddingBlocker(
    discogs, musicbrainz,
    text_cols=["name", "artist", "release-date"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking",
    id_column="id"
)

# TokenBlocker for musicbrainz and lastfm
blocker_musicbrainz_lastfm = TokenBlocker(
    musicbrainz, lastfm,
    column="name",
    min_token_len=4,
    ngram_size=2,
    ngram_type="word",
    id_column="id",
    output_dir="output/blocking"
)

# --------------------------------
# Matching
# --------------------------------

print("Matching Entities")

# Matching configuration
matching_config = {
    "discogs_lastfm": [
        StringComparator(column="name", similarity_function="cosine", preprocess=str.lower),
        StringComparator(column="tracks_track_position", similarity_function="cosine", preprocess=str.lower),
        NumericComparator(column="duration", max_difference=5),
        StringComparator(column="tracks_track_name", similarity_function="cosine", preprocess=str.lower)
    ],
    "discogs_musicbrainz": [
        StringComparator(column="name", similarity_function="cosine", preprocess=str.lower),
        StringComparator(column="artist", similarity_function="cosine", preprocess=str.lower),
        DateComparator(column="release-date", max_days_difference=365),
        StringComparator(column="tracks_track_position", similarity_function="cosine", preprocess=str.lower)
    ],
    "musicbrainz_lastfm": [
        StringComparator(column="name", similarity_function="cosine", preprocess=str.lower),
        StringComparator(column="tracks_track_position", similarity_function="cosine", preprocess=str.lower),
        StringComparator(column="duration", similarity_function="cosine", preprocess=str.lower),
        StringComparator(column="tracks_track_name", similarity_function="cosine", preprocess=str.lower)
    ]
}

# Feature extractors
feature_extractor_discogs_lastfm = FeatureExtractor(matching_config["discogs_lastfm"])
feature_extractor_discogs_musicbrainz = FeatureExtractor(matching_config["discogs_musicbrainz"])
feature_extractor_musicbrainz_lastfm = FeatureExtractor(matching_config["musicbrainz_lastfm"])

# Load ground truth correspondences (ML training/test)
train_discogs_lastfm = load_csv(
    "input/datasets/music/testsets/discogs_lastfm_goldstandard_blocking.csv",
    name="ground_truth_discogs_lastfm",
    add_index=False
)

train_discogs_musicbrainz = load_csv(
    "input/datasets/music/testsets/discogs_musicbrainz_goldstandard_blocking.csv",
    name="ground_truth_discogs_musicbrainz",
    add_index=False
)

train_musicbrainz_lastfm = load_csv(
    "input/datasets/music/testsets/musicbrainz_lastfm_goldstandard_blocking.csv",
    name="ground_truth_musicbrainz_lastfm",
    add_index=False
)

# Extract features
train_discogs_lastfm_features = feature_extractor_discogs_lastfm.create_features(
    discogs, lastfm, train_discogs_lastfm[['id1', 'id2']], labels=train_discogs_lastfm['label'], id_column='id'
)

train_discogs_musicbrainz_features = feature_extractor_discogs_musicbrainz.create_features(
    discogs, musicbrainz, train_discogs_musicbrainz[['id1', 'id2']], labels=train_discogs_musicbrainz['label'], id_column='id'
)

train_musicbrainz_lastfm_features = feature_extractor_musicbrainz_lastfm.create_features(
    musicbrainz, lastfm, train_musicbrainz_lastfm[['id1', 'id2']], labels=train_musicbrainz_lastfm['label'], id_column='id'
)

# Prepare data for ML training
feat_cols_discogs_lastfm = [col for col in train_discogs_lastfm_features.columns if col not in ['id1', 'id2', 'label']]
X_train_discogs_lastfm = train_discogs_lastfm_features[feat_cols_discogs_lastfm]
y_train_discogs_lastfm = train_discogs_lastfm_features['label']

feat_cols_discogs_musicbrainz = [col for col in train_discogs_musicbrainz_features.columns if col not in ['id1', 'id2', 'label']]
X_train_discogs_musicbrainz = train_discogs_musicbrainz_features[feat_cols_discogs_musicbrainz]
y_train_discogs_musicbrainz = train_discogs_musicbrainz_features['label']

feat_cols_musicbrainz_lastfm = [col for col in train_musicbrainz_lastfm_features.columns if col not in ['id1', 'id2', 'label']]
X_train_musicbrainz_lastfm = train_musicbrainz_lastfm_features[feat_cols_musicbrainz_lastfm]
y_train_musicbrainz_lastfm = train_musicbrainz_lastfm_features['label']

training_datasets = [
    (X_train_discogs_lastfm, y_train_discogs_lastfm),
    (X_train_discogs_musicbrainz, y_train_discogs_musicbrainz),
    (X_train_musicbrainz_lastfm, y_train_musicbrainz_lastfm)
]

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
            'penalty': ['l2'],
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

# --------------------------------
# Match Entities
# --------------------------------

ml_matcher_discogs_lastfm = MLBasedMatcher(feature_extractor_discogs_lastfm)
ml_matcher_discogs_musicbrainz = MLBasedMatcher(feature_extractor_discogs_musicbrainz)
ml_matcher_musicbrainz_lastfm = MLBasedMatcher(feature_extractor_musicbrainz_lastfm)

ml_correspondences_discogs_lastfm = ml_matcher_discogs_lastfm.match(
    discogs, lastfm, candidates=blocker_discogs_lastfm, id_column='id', trained_classifier=best_models[0]
)

ml_correspondences_discogs_musicbrainz = ml_matcher_discogs_musicbrainz.match(
    discogs, musicbrainz, candidates=blocker_discogs_musicbrainz, id_column='id', trained_classifier=best_models[1]
)

ml_correspondences_musicbrainz_lastfm = ml_matcher_musicbrainz_lastfm.match(
    musicbrainz, lastfm, candidates=blocker_musicbrainz_lastfm, id_column='id', trained_classifier=best_models[2]
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

# merge ML-based correspondences
all_ml_correspondences = pd.concat(
    [ml_correspondences_discogs_lastfm, ml_correspondences_discogs_musicbrainz, ml_correspondences_musicbrainz_lastfm],
    ignore_index=True
)

# define data fusion strategy
strategy = DataFusionStrategy('ml_fusion_strategy')

strategy.add_attribute_fuser('name', longest_string)
strategy.add_attribute_fuser('artist', longest_string)
strategy.add_attribute_fuser('release-date', longest_string)
strategy.add_attribute_fuser('release-country', longest_string)
strategy.add_attribute_fuser('duration', longest_string)
strategy.add_attribute_fuser('label', longest_string)
strategy.add_attribute_fuser('genre', longest_string)
strategy.add_attribute_fuser('tracks_track_name', union)
strategy.add_attribute_fuser('tracks_track_position', union)
strategy.add_attribute_fuser('tracks_track_duration', union)

# run fusion
engine = DataFusionEngine(strategy, debug=True, debug_format='json', debug_file="output/data_fusion/debug_fusion_music.jsonl")

ml_fused_music = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
ml_fused_music.to_csv("output/data_fusion/fusion_music.csv", index=False)