from PyDI.io import load_parquet

from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union, prefer_higher_trust
from PyDI.fusion import DataFusionEvaluator, tokenized_match

import pandas as pd
import numpy as np

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

# --------------------------------
# Perform Entity Matching
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

# define comparators
comparators = [
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

    # category similarity - supporting evidence
    StringComparator(
        column='categories',
        similarity_function='jaccard',
        preprocess=str.lower,
        list_strategy='concatenate' # Handle list attribute by concatenation
    )
]

# Initialize Rule-Based Matcher
matcher = RuleBasedMatcher()

rb_correspondences_k2u = matcher.match(
    df_left=kaggle380k,
    df_right=uber_eats, 
    candidates=blocker_k2u,
    comparators=comparators,
    weights=[0.5, 0.2, 0.2, 0.1], 
    threshold=0.7,
    id_column='id'
)

rb_correspondences_k2y = matcher.match(
    df_left=kaggle380k,
    df_right=yelp, 
    candidates=blocker_k2y,
    comparators=comparators,
    weights=[0.5, 0.2, 0.2, 0.1],  
    threshold=0.7,
    id_column='id'
)

# --------------------------------
# Data Fusion
# --------------------------------
# set trust scores
kaggle380k.attrs["trust_score"] = 3
uber_eats.attrs["trust_score"] = 2
yelp.attrs["trust_score"] = 1

# merge rule based correspondences
all_rb_correspondences = pd.concat([rb_correspondences_k2u, rb_correspondences_k2y], ignore_index=True)

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
engine = DataFusionEngine(strategy, debug=True, debug_format='json',debug_file= "output/data_fusion/debug_fusion_rb_standard_blocker.jsonl")

# fuse rule based matches
rb_fused_standard_blocker = engine.run(
    datasets=[kaggle380k, uber_eats, yelp],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_rb_standard_blocker.csv", index=False)

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
fused_eval = rb_fused_standard_blocker[eval_cols].copy()

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