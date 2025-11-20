from pathlib import Path
import pandas as pd
from PyDI.io import load_parquet
from PyDI.io import load_csv
from PyDI.entitymatching import MLBasedMatcher
from PyDI.entitymatching import MaximumBipartiteMatching
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import numpy as np
import os
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union, prefer_higher_trust
from PyDI.fusion import DataFusionEvaluator
from PyDI.fusion import tokenized_match, boolean_match,numeric_tolerance_match,set_equality_match
import numpy as np
import re, ast

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_colwidth', 100)

ROOT = Path.cwd()

DATA_DIR = ROOT / "parquet"
OUTPUT_DIR = ROOT / "output"
MLDS_DIR = ROOT / "ml-datasets"
BLOCK_EVAL_DIR = OUTPUT_DIR / "blocking_evaluation"
CORR_DIR = OUTPUT_DIR / "correspondences"

PIPELINE_DIR = OUTPUT_DIR / "data_fusion"
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

amazon_sample = load_parquet(
    DATA_DIR / "amazon_sample.parquet",
    name="amazon_sample"
)

goodreads_sample = load_parquet(
    DATA_DIR / "goodreads_sample.parquet",
    name="goodreads_sample"
)

metabooks_sample = load_parquet(
  DATA_DIR / "metabooks_sample.parquet",
  name="metabooks_sample"
)

train_m2a = load_csv(
    MLDS_DIR / "train_MA.csv",
    name="train_metabooks_amazon",
    header=0,
    names=['id1', 'id2', 'label'],
    add_index=False
)

test_m2a = load_csv(
    MLDS_DIR / "test_MA.csv",
    name="test_metabooks_amazon",
    header=0,
    names=['id1', 'id2', 'label'],
    add_index=False
)

train_m2g = load_csv(
    MLDS_DIR / "train_MG.csv",
    name="train_metabooks_goodreads",
    header=0,
    names=['id1', 'id2', 'label'],
    add_index=False
)

test_m2g = load_csv(
    MLDS_DIR / "test_MG.csv",
    name="test_metabooks_goodreads",
    header=0,
    names=['id1', 'id2', 'label'],
    add_index=False
)

from PyDI.entitymatching import EmbeddingBlocker

embedding_blocker_m2a = EmbeddingBlocker(
    metabooks_sample, amazon_sample,
    text_cols=['title', 'author'],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=200,
    output_dir=BLOCK_EVAL_DIR,
    id_column='id'
)


embedding_blocker_m2g = EmbeddingBlocker(
    metabooks_sample, goodreads_sample,
    text_cols=['title', 'author'],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=200,
    output_dir=BLOCK_EVAL_DIR,
    id_column='id'
)

embedding_candidates_m2a = embedding_blocker_m2a.materialize()
embedding_candidates_m2g = embedding_blocker_m2g.materialize()

from PyDI.entitymatching import StringComparator, NumericComparator

comparators = [
    StringComparator(column='title',similarity_function='cosine'),
    StringComparator(column='title',tokenization="char",similarity_function='jaccard'),
    StringComparator(column='title',similarity_function='jaro_winkler'),

    StringComparator(column='author',similarity_function='cosine', ),
    StringComparator(column='author',similarity_function='jaccard', ),


    StringComparator(column='publisher',similarity_function='jaro_winkler', preprocess=str.lower),
    StringComparator(column='publisher',similarity_function='cosine', preprocess=str.lower),
    

    NumericComparator(column='publish_year',max_difference=1.0),


    NumericComparator(column="page_count", max_difference=10),
]

from PyDI.entitymatching import FeatureExtractor

feature_extractor = FeatureExtractor(comparators)

train_features_m2a = feature_extractor.create_features(
    metabooks_sample, amazon_sample, train_m2a[['id1', 'id2']], labels=train_m2a['label'], id_column='id'
)

train_features_m2g = feature_extractor.create_features(
    metabooks_sample, goodreads_sample, train_m2g[['id1', 'id2']], labels=train_m2g['label'], id_column='id'
)

feature_columns_m2a = [col for col in train_features_m2a.columns if col not in ['id1', 'id2', 'label']]
X_train_m2a = train_features_m2a[feature_columns_m2a]
y_train_m2a = train_features_m2a['label']

feature_columns_m2g = [col for col in train_features_m2g.columns if col not in ['id1', 'id2', 'label']]
X_train_m2g = train_features_m2g[feature_columns_m2g]
y_train_m2g = train_features_m2g['label']

training_datasets = [(X_train_m2a, y_train_m2a),(X_train_m2g, y_train_m2g)]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# classifiers
classifiers = {
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
    'SVC': SVC(probability=True, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
}

# parameter grids
param_grids = {
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 10, 20],
        'class_weight': ['balanced', None],
        'min_samples_split': [2, 5]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'LogisticRegression': {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    }
}

scorer = make_scorer(f1_score)
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

training_datasets = [
    (X_train_m2a, y_train_m2a),
    (X_train_m2g, y_train_m2g),
]

best_models = []  # one best model per dataset

for (X_train, y_train) in training_datasets:
    grid_search_results = {}
    best_overall_score = -1
    best_overall_model = None
    best_model_name = None

    for name, model in classifiers.items():
        print(f"Running GridSearchCV for {name}...")
        
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring=scorer,
            cv=cv_folds,
            n_jobs=-1,
            verbose=0
        )
        
        grid.fit(X_train, y_train)
        print(
            f"{name}: best F1 = {grid.best_score_:.4f} "
            f"with params {grid.best_params_}"
        )
        
        grid_search_results[name] = {
            'grid_search': grid,
            'best_score': grid.best_score_,
            'best_params': grid.best_params_,
            'best_estimator': grid.best_estimator_
        }

        if grid.best_score_ > best_overall_score:
            best_overall_score = grid.best_score_   
            best_overall_model = grid.best_estimator_
            best_model_name = name

    print(f"Best model for this dataset: {best_model_name} with F1={best_overall_score:.4f}")
    best_models.append(best_overall_model)



ml_matcher = MLBasedMatcher(feature_extractor)

correspondences_m2a = ml_matcher.match(
    metabooks_sample, amazon_sample,
    candidates=embedding_blocker_m2a,
    id_column='id',
    trained_classifier=best_models[0],
)

correspondences_m2g = ml_matcher.match(
    metabooks_sample, goodreads_sample,
    candidates=embedding_blocker_m2g,
    id_column='id',
    trained_classifier=best_models[1]
)

debug_output_dir = OUTPUT_DIR / "debug_results_entity_matching"
debug_output_dir.mkdir(parents=True, exist_ok=True)

# We are using Maxmimum Bipartite Matching to refine results to 1:1 matches
clusterer = MaximumBipartiteMatching()
mbm_correspondences_m2a = clusterer.cluster(correspondences_m2a)
mbm_correspondences_m2g = clusterer.cluster(correspondences_m2g)
all_correspondences = pd.concat([mbm_correspondences_m2a, mbm_correspondences_m2g], ignore_index=True)



metabooks_sample.attrs["trust_score"] = 3
goodreads_sample.attrs["trust_score"] = 2
amazon_sample.attrs["trust_score"] = 1
strategy = DataFusionStrategy('books_fusion_strategy')

strategy.add_attribute_fuser('title', longest_string)
strategy.add_attribute_fuser('author', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('publish_year', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('publisher', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('language', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('price', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('page_count', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('numratings', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('genres',union)

engine = DataFusionEngine(strategy, debug=True, debug_format='json',
                          debug_file= PIPELINE_DIR / "debug_fusion_ml_embedding_blocker.jsonl")

fused_ml_emblocker = engine.run(
    datasets=[amazon_sample, metabooks_sample, goodreads_sample],
    correspondences=all_correspondences,
    id_column="id",
    include_singletons=False
)
fused_ml_emblocker.to_parquet(PIPELINE_DIR / "fused_ml_emblocker.parquet")
print(f'Fused rows: {len(fused_ml_emblocker):,}')


def categories_set_equal(a, b) -> bool:
    """Return True if a and b contain the same unique categories (order/type agnostic)."""
    def to_set(x):
        def items(v):
            # missing
            if v is None or (isinstance(v, float) and np.isnan(v)): return []
            # numpy array → recurse over elements
            if isinstance(v, np.ndarray): 
                out=[]; [out.extend(items(e)) for e in v.flatten()]; return out
            # python containers → recurse over elements
            if isinstance(v, (list, tuple, set)):
                out=[]; [out.extend(items(e)) for e in v]; return out
            # scalar/string: try parse stringified list; else split by delimiters
            s = str(v).strip()
            if s == "" or s.lower() in {"nan","none"}: return []
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)): return items(parsed)
            except Exception:
                pass
            return [p.strip() for p in re.split(r"[|,;/]", s) if p.strip()]
        return {it.lower() for it in items(x)}
    return to_set(a) == to_set(b)

strategy.add_evaluation_function("title", tokenized_match)
strategy.add_evaluation_function("author", tokenized_match)
strategy.add_evaluation_function("publisher", tokenized_match)
strategy.add_evaluation_function("publish_year", numeric_tolerance_match)
strategy.add_evaluation_function("numratings", numeric_tolerance_match)
strategy.add_evaluation_function("price", numeric_tolerance_match)
strategy.add_evaluation_function("page_count", numeric_tolerance_match)
strategy.add_evaluation_function("language", tokenized_match)
strategy.add_evaluation_function("genres", categories_set_equal)

fused_dataset = pd.read_parquet(PIPELINE_DIR / "fused_ml_emblocker.parquet")
fused_dataset["publish_year"] = fused_dataset["publish_year"].astype("Int16")
fused_dataset["page_count"] = fused_dataset["page_count"].astype("Int32")
golden_fused_dataset= pd.read_parquet(MLDS_DIR / "golden_fused_books.parquet")

fused_dataset.drop_duplicates(subset='isbn_clean', keep='first',inplace=True)
# Create evaluator with our fusion strategy
evaluator = DataFusionEvaluator(strategy, debug=True, debug_file=OUTPUT_DIR / "data_fusion" / "debug_fusion_eval.jsonl", debug_format="json")

# Evaluate the fused results against the gold standard
print("Evaluating fusion results against gold standard...")
evaluation_results = evaluator.evaluate(
    fused_df=fused_dataset,
    fused_id_column='isbn_clean',
    gold_df=golden_fused_dataset,
    gold_id_column='isbn_clean',
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