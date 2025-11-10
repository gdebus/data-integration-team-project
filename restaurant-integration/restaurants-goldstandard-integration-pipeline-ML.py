from PyDI.io import load_parquet, load_csv
from PyDI.entitymatching import FeatureExtractor
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import MLBasedMatcher

from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union, prefer_higher_trust
from PyDI.fusion import DataFusionEvaluator, tokenized_match

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

import numpy as np
import pandas as pd

# --------------------------------
# Prepare Data
# --------------------------------

# Define dataset paths
DATA_DIR = "datasets/"

# Load Kaggle 380k dataset
kaggle380k = load_parquet(
    DATA_DIR + "kaggle380k.parquet",
    name="kaggle380k",
)

# Load Uber Eats dataset  
uber_eats = load_parquet(
    DATA_DIR + "uber_eats.parquet",
    name="uber_eats",
)

# Load Yelp dataset
yelp = load_parquet(
    DATA_DIR + "yelp.parquet",
    name="yelp",
)

# create id columns
kaggle380k['id'] = kaggle380k['kaggle380k_id']
uber_eats['id'] = uber_eats['uber_eats_id']
yelp['id'] = yelp['yelp_id']


datasets = [kaggle380k, uber_eats, yelp]

for i, df in enumerate(datasets):
    # Convert numpy.ndarray entries in 'categories' column to Python lists
    df["categories"] = df["categories"].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )
    datasets[i] = df

kaggle380k = datasets[0]
uber_eats = datasets[1]
yelp = datasets[2]

# Load ground truth correspondences
k2y_train = load_csv(
    "ml-datasets/ground_truth_df1_df3_train.csv",
    name="ground_truth_df1_df3_train",
    header=None,
    names=['id1', 'id2', 'label'],
    add_index=False
)

k2y_test = load_csv(
    "ml-datasets/ground_truth_df1_df3_test.csv",
    name="ground_truth_df1_df3_test",
    header=None,
    names=['id1', 'id2', 'label'],
    add_index=False
)

k2u_train = load_csv(
    "ml-datasets/ground_truth_df1_df2_train.csv",
    name="ground_truth_df1_df2_train",
    header=None,
    names=['id1', 'id2', 'label'],
    add_index=False
)

k2u_test = load_csv(
    "ml-datasets/ground_truth_df1_df2_test.csv",
    name="ground_truth_df1_df2_test",
    header=None,
    names=['id1', 'id2', 'label'],
    add_index=False
)

# --------------------------------
# Feature Extraction
# --------------------------------

similarity_comparators = [
    # Name similarity features
    StringComparator("name_norm", similarity_function="jaro_winkler", preprocess=str.lower),
    StringComparator("name_norm", similarity_function="levenshtein", preprocess=str.lower),
    StringComparator("name_norm", similarity_function="cosine", preprocess=str.lower),
    StringComparator("name_norm", similarity_function="jaccard", preprocess=str.lower),

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

    # category similarity - supporting evidence
    StringComparator(
        column='categories',
        similarity_function='jaccard',
        preprocess=str.lower,
        list_strategy='concatenate' # Handle list attribute by concatenation
    ),
    StringComparator(
        column='categories',
        similarity_function='jaccard',
        preprocess=str.lower,
        list_strategy='best_match' # Handle list attribute by concatenation
    )
]

feature_extractor = FeatureExtractor(similarity_comparators)

# Extract features using FeatureExtractor
k2u_train_features = feature_extractor.create_features(
    kaggle380k, uber_eats, k2u_train[['id1', 'id2']], labels=k2u_train['label'], id_column='id'
)

# Extract features using FeatureExtractor
k2y_train_features = feature_extractor.create_features(
    kaggle380k, yelp, k2y_train[['id1', 'id2']], labels=k2y_train['label'], id_column='id'
)

# Prepare data for ML training
k2u_feature_columns = [col for col in k2u_train_features.columns if col not in ['id1', 'id2', 'label']]
k2u_X_train = k2u_train_features[k2u_feature_columns]
k2u_y_train = k2u_train_features['label']

k2y_feature_columns = [col for col in k2y_train_features.columns if col not in ['id1', 'id2', 'label']]
k2y_X_train = k2y_train_features[k2y_feature_columns]
k2y_y_train = k2y_train_features['label']

training_datasets = [(k2u_X_train,k2u_y_train),(k2y_X_train,k2y_y_train)]

# --------------------------------
# Select Best Model
# --------------------------------

# Define models and parameter grids
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

# Use F1 score as the scoring metric (good for imbalanced data)
scorer = make_scorer(f1_score)
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_models = []

for dataset in training_datasets:

    grid_search_results = {}
    best_overall_score = -1
    best_overall_model = None
    best_model_name = None
    
    for model_name, config in param_grids.items():
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            scoring=scorer,
            cv=cv_folds,
            n_jobs=-1,  # Use all available cores
            verbose=0
        )
        
        # Fit GridSearchCV
        grid_search.fit(dataset[0], dataset[1])
        
        # Store results
        grid_search_results[model_name] = {
            'grid_search': grid_search,
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'best_estimator': grid_search.best_estimator_
        }
        
        # Track overall best model
        if grid_search.best_score_ > best_overall_score:
            best_overall_model = grid_search.best_estimator_

    best_models.append(best_overall_model)

# --------------------------------
# Entity Matching
# --------------------------------

# define blocking
blocker_k2u = StandardBlocker(
    kaggle380k, uber_eats,
    on=['city'],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column='id'
)

blocker_k2y = StandardBlocker(
    kaggle380k, yelp,
    on=['city'],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column='id'
)

ml_matcher = MLBasedMatcher(feature_extractor)

correspondences_k2u_ml = ml_matcher.match(
    kaggle380k, uber_eats, candidates=blocker_k2u, id_column='id', trained_classifier=best_models[0]
)

correspondences_k2y_ml = ml_matcher.match(
    kaggle380k, yelp, candidates=blocker_k2y, id_column='id', trained_classifier=best_models[1]
)

# --------------------------------
# Data Fusion
# --------------------------------
# set trust scores
kaggle380k.attrs["trust_score"] = 3
uber_eats.attrs["trust_score"] = 2
yelp.attrs["trust_score"] = 1

# merge rule based correspondences
all_ml_correspondences = pd.concat([correspondences_k2u_ml_1, correspondences_k2y_ml_1], ignore_index=True)

# define data fusion strategy
strategy = DataFusionStrategy('usa_restaurant_fusion_strategy')

strategy.add_attribute_fuser('name', longest_string)
strategy.add_attribute_fuser('street', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('house_number', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('city', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('state', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('postal_code', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('country', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('latitude', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('longitude', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('categories', union)

# run fusion
engine = DataFusionEngine(strategy, debug=True, debug_format='json',debug_file="output/data_fusion/debug_fusion_ml_standard_blocker.jsonl")

# fuse rule based matches
ml_fused_standard_blocker = engine.run(
    datasets=[kaggle380k, uber_eats, yelp],
    correspondences=all_ml_correspondences,
    id_column="id",
    include_singletons=False,
)

ml_fused_standard_blocker.to_csv("output/data_fusion/fusion_ml_standard_blocker.csv", index=False)

# --------------------------------
# Evaluation
# --------------------------------

strategy.add_evaluation_function("name", tokenized_match)
strategy.add_evaluation_function("street", tokenized_match)
strategy.add_evaluation_function("house_number", tokenized_match)
strategy.add_evaluation_function("city", tokenized_match)
strategy.add_evaluation_function("state", tokenized_match)
strategy.add_evaluation_function("postal_code", tokenized_match)
strategy.add_evaluation_function("country", tokenized_match)
strategy.add_evaluation_function("latitude", tokenized_match)
strategy.add_evaluation_function("longitude", tokenized_match)
strategy.add_evaluation_function("categories", tokenized_match)

fusion_test_set = pd.read_csv("input/data_fusion/Gold_Standard_final.csv")

# Keep core evaluation columns if present in fused output
eval_cols = ['kaggle380k_id','name','street','house_number','city','state','postal_code','country','latitude','longitude','categories']
fused_eval = ml_fused_standard_blocker[eval_cols].copy()

# Create evaluator with our fusion strategy
evaluator = DataFusionEvaluator(strategy)

# Evaluate the fused results against the gold standard
print("Evaluating fusion results against gold standard...")
evaluation_results = evaluator.evaluate(
    fused_df=fused_eval,
    fused_id_column='kaggle380k_id',
    gold_df=fusion_test_set,
    gold_id_column='kaggle380k_id',
)

# Display evaluation metrics
print("\nFusion Evaluation Results:")
print("=" * 40)
for metric, value in evaluation_results.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.3f}")
    else:
        print(f"  {metric}: {value}")
        
print(f"\nOverall Accuracy: {evaluation_results.get('overall_accuracy', 0):.1%}")