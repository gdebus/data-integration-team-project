# --------------------------------
# Integration pipeline for discogs, lastfm, musicbrainz datasets
# --------------------------------

from PyDI.io import load_csv
from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
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
import numpy as np
from dotenv import load_dotenv
import os

# --------------------------------
# Prepare Data
# --------------------------------

DATA_DIR = "output/schema-matching/"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load datasets
discogs = load_csv(DATA_DIR + "discogs.csv", name="discogs")
lastfm = load_csv(DATA_DIR + "lastfm.csv", name="lastfm")
musicbrainz = load_csv(DATA_DIR + "musicbrainz.csv", name="musicbrainz")

# Set id columns
discogs["id"] = discogs["id"]
lastfm["id"] = lastfm["id"]
musicbrainz["id"] = musicbrainz["id"]

datasets = [discogs, lastfm, musicbrainz]

# --------------------------------
# Schema Matching (LLM-based)
# Match lastfm and musicbrainz schema to discogs schema
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

# Match lastfm to discogs schema
schema_corr_lastfm = matcher.match(discogs, lastfm)
rename_map_lastfm = schema_corr_lastfm.set_index("target_column")["source_column"].to_dict()
lastfm = lastfm.rename(columns=rename_map_lastfm)

# Match musicbrainz to discogs schema
schema_corr_musicbrainz = matcher.match(discogs, musicbrainz)
rename_map_musicbrainz = schema_corr_musicbrainz.set_index("target_column")["source_column"].to_dict()
musicbrainz = musicbrainz.rename(columns=rename_map_musicbrainz)

# --------------------------------
# Blocking
# Use blocking strategies from blocking configuration
# --------------------------------

print("Performing Blocking")

# Blocking config:
# discogs_lastfm: token_blocking on ['artist'] with min_token_len=5 (not acceptable, but use as instructed)
# discogs_musicbrainz: semantic_similarity on ['name', 'artist', 'release-date'] top_k=20 (acceptable)
# musicbrainz_lastfm: token_blocking on ['name'] with min_token_len=5 (not acceptable, but use as instructed)

# Define id columns
id_discogs = "id"
id_lastfm = "id"
id_musicbrainz = "id"

# Blocking discogs <-> lastfm (token blocking on artist)
blocker_discogs_lastfm = StandardBlocker(
    discogs, lastfm,
    on=['artist'],
    min_token_len=5,
    id_column=id_discogs
)

# Blocking discogs <-> musicbrainz (semantic similarity on name, artist, release-date)
blocker_discogs_musicbrainz = EmbeddingBlocker(
    discogs, musicbrainz,
    text_cols=['name', 'artist', 'release-date'],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=500,
    id_column=id_discogs
)

# Blocking musicbrainz <-> lastfm (token blocking on name)
blocker_musicbrainz_lastfm = StandardBlocker(
    musicbrainz, lastfm,
    on=['name'],
    min_token_len=5,
    id_column=id_musicbrainz
)

# --------------------------------
# Feature Extraction
# Use matching configuration comparators for each pair
# --------------------------------

print("Extracting Features")

# Preprocess functions mapping
def lower(x): return str(x).lower()
def lower_strip(x): return str(x).lower().strip()

# discogs_lastfm comparators
comparators_discogs_lastfm = [
    StringComparator(
        column='artist',
        similarity_function='jaro_winkler',
        preprocess=lower
    ),
    StringComparator(
        column='name',
        similarity_function='cosine',
        preprocess=lower
    ),
    NumericComparator(
        column='duration',
        max_difference=10.0
    )
]

# discogs_musicbrainz comparators
comparators_discogs_musicbrainz = [
    StringComparator(
        column='name',
        similarity_function='cosine',
        preprocess=lower_strip,
        list_strategy='concatenate'
    ),
    StringComparator(
        column='artist',
        similarity_function='jaro_winkler',
        preprocess=lower_strip
    ),
    DateComparator(
        column='release-date',
        max_days_difference=365
    )
]

# musicbrainz_lastfm comparators
comparators_musicbrainz_lastfm = [
    StringComparator(
        column='name',
        similarity_function='jaro_winkler',
        preprocess=lower_strip
    ),
    NumericComparator(
        column='duration',
        max_difference=10.0
    ),
    StringComparator(
        column='tracks_track_name',
        similarity_function='cosine',
        preprocess=lower,
        list_strategy='concatenate'
    )
]

# Create feature extractors
fe_discogs_lastfm = FeatureExtractor(comparators_discogs_lastfm)
fe_discogs_musicbrainz = FeatureExtractor(comparators_discogs_musicbrainz)
fe_musicbrainz_lastfm = FeatureExtractor(comparators_musicbrainz_lastfm)

# --------------------------------
# Load ground truth for training
# --------------------------------

train_discogs_lastfm = load_csv(
    "ml-datasets/ground_truth_discogs_lastfm_train.csv",
    name="ground_truth_discogs_lastfm_train",
    header=None,
    names=['id1', 'id2', 'label'],
    add_index=False
)

train_discogs_musicbrainz = load_csv(
    "ml-datasets/ground_truth_discogs_musicbrainz_train.csv",
    name="ground_truth_discogs_musicbrainz_train",
    header=None,
    names=['id1', 'id2', 'label'],
    add_index=False
)

train_musicbrainz_lastfm = load_csv(
    "ml-datasets/ground_truth_musicbrainz_lastfm_train.csv",
    name="ground_truth_musicbrainz_lastfm_train",
    header=None,
    names=['id1', 'id2', 'label'],
    add_index=False
)

# --------------------------------
# Extract features for training
# --------------------------------

train_features_discogs_lastfm = fe_discogs_lastfm.create_features(
    discogs, lastfm, train_discogs_lastfm[['id1', 'id2']], labels=train_discogs_lastfm['label'], id_column=id_discogs
)

train_features_discogs_musicbrainz = fe_discogs_musicbrainz.create_features(
    discogs, musicbrainz, train_discogs_musicbrainz[['id1', 'id2']], labels=train_discogs_musicbrainz['label'], id_column=id_discogs
)

train_features_musicbrainz_lastfm = fe_musicbrainz_lastfm.create_features(
    musicbrainz, lastfm, train_musicbrainz_lastfm[['id1', 'id2']], labels=train_musicbrainz_lastfm['label'], id_column=id_musicbrainz
)

# Prepare X and y for training
def prepare_xy(df):
    feat_cols = [c for c in df.columns if c not in ['id1', 'id2', 'label']]
    X = df[feat_cols]
    y = df['label']
    return X, y

X_discogs_lastfm, y_discogs_lastfm = prepare_xy(train_features_discogs_lastfm)
X_discogs_musicbrainz, y_discogs_musicbrainz = prepare_xy(train_features_discogs_musicbrainz)
X_musicbrainz_lastfm, y_musicbrainz_lastfm = prepare_xy(train_features_musicbrainz_lastfm)

training_datasets = [
    (X_discogs_lastfm, y_discogs_lastfm),
    (X_discogs_musicbrainz, y_discogs_musicbrainz),
    (X_musicbrainz_lastfm, y_musicbrainz_lastfm)
]

# --------------------------------
# Model Selection and Training
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

for X_train, y_train in training_datasets:
    best_score = -1
    best_model = None
    for model_name, config in param_grids.items():
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            scoring=scorer,
            cv=cv_folds,
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
    best_models.append(best_model)

# --------------------------------
# Entity Matching
# --------------------------------

print("Matching Entities")

ml_matcher_discogs_lastfm = MLBasedMatcher(fe_discogs_lastfm)
ml_matcher_discogs_musicbrainz = MLBasedMatcher(fe_discogs_musicbrainz)
ml_matcher_musicbrainz_lastfm = MLBasedMatcher(fe_musicbrainz_lastfm)

ml_corres_discogs_lastfm = ml_matcher_discogs_lastfm.match(
    discogs, lastfm, candidates=blocker_discogs_lastfm, id_column=id_discogs, trained_classifier=best_models[0]
)

ml_corres_discogs_musicbrainz = ml_matcher_discogs_musicbrainz.match(
    discogs, musicbrainz, candidates=blocker_discogs_musicbrainz, id_column=id_discogs, trained_classifier=best_models[1]
)

ml_corres_musicbrainz_lastfm = ml_matcher_musicbrainz_lastfm.match(
    musicbrainz, lastfm, candidates=blocker_musicbrainz_lastfm, id_column=id_musicbrainz, trained_classifier=best_models[2]
)

# --------------------------------
# Data Fusion
# --------------------------------

print("Fusing Data")

all_correspondences = pd.concat(
    [ml_corres_discogs_lastfm, ml_corres_discogs_musicbrainz, ml_corres_musicbrainz_lastfm],
    ignore_index=True
)

strategy = DataFusionStrategy('ml_fusion_strategy')

# Use longest_string for string attributes, union for lists
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

engine = DataFusionEngine(strategy, debug=True, debug_format='json', debug_file="output/data_fusion/debug_fusion_ml.jsonl")

ml_fused = engine.run(
    datasets=[discogs, lastfm, musicbrainz],
    correspondences=all_correspondences,
    id_column="id",
    include_singletons=False,
)

# Write output
ml_fused.to_csv("output/data_fusion/fusion_ml.csv", index=False)