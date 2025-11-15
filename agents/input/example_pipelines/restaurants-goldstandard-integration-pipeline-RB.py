from PyDI.io import load_parquet, load_csv, load_xml

from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union, prefer_higher_trust
from PyDI.fusion import DataFusionEvaluator, tokenized_match

import pandas as pd
import numpy as np

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

# Define dataset paths
DATA_DIR = "datasets/"

# Load the first dataset
good_dataset_name_1 = load_parquet(
    DATA_DIR + "<path-and-file-name-of-dataset-1>.parquet",
    name="dataset_name_1",
)

# Load Uber Eats dataset  
good_dataset_name_2 = load_parquet(
    DATA_DIR + "path-and-file-name-of-dataset-2.parquet",
    name="dataset_name_2",
)

# Load Yelp dataset
good_dataset_name_3 = load_parquet(
    DATA_DIR + "path-and-file-name-of-dataset-3.parquet",
    name="dataset_name_3",
)

# create a column called ´id´ for further processing if such a column does not already exist
good_dataset_name_1['id'] = good_dataset_name_1['kaggle380k_id']
good_dataset_name_2['id'] = good_dataset_name_2['uber_eats_id']
good_dataset_name_3['id'] = good_dataset_name_3['yelp_id']

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# --------------------------------
# Perform Entity Matching
# Employ the embedding Blocker per default. If the size of the datasets is too large, use the StandardBlocker
# Important: Perform the blocking on a o
# --------------------------------

# Example of an Embedding blocker:
# embedding_blocker_dataset1_2_dataset2 = EmbeddingBlocker(
#    good_dataset_name_1, good_dataset_name_2, # name of the datasets
#    text_cols=['city'], # column which should be used to perform the blocking on
#    model="sentence-transformers/all-MiniLM-L6-v2",
#    index_backend="sklearn",
#    top_k=20,          # Top 20 most similar
#    batch_size=500,
#    output_dir="output/blocking-evaluation",
#    id_column='id'
#)

# Example of a Standard blocker:
# blocker_k2u = StandardBlocker(
#    good_dataset_name_1, good_dataset_name_2,
#    on=['city'], # column which should be used to perform the blocking on
#    batch_size=1000,
#    output_dir="output/blocking-evaluation",
#    id_column='id'
#)

# define blocking in this example using the standard blocker
blocker_k2u = StandardBlocker(
    good_dataset_name_1, good_dataset_name_2,
    on=['city'],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column='id'
)

blocker_k2y = StandardBlocker(
    good_dataset_name_1, good_dataset_name_3,
    on=['city'],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column='id'
)

# define comparators
# Which and how many comparators to use is open to you. Use them so they make sense in the context of the data within the dataset. 
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
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2, 
    candidates=blocker_k2u,
    comparators=comparators,
    weights=[0.5, 0.2, 0.2, 0.1], # weight the different comparators. Adjust accordingly so they make sense
    threshold=0.7,
    id_column='id'
)

rb_correspondences_k2y = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3, 
    candidates=blocker_k2y,
    comparators=comparators,
    weights=[0.5, 0.2, 0.2, 0.1],  
    threshold=0.7,
    id_column='id'
)

# --------------------------------
# Data Fusion
# --------------------------------
# set trust scores. You may draw your own assumptions about which dataset should be trusted most
good_dataset_name_1.attrs["trust_score"] = 3
good_dataset_name_2.attrs["trust_score"] = 2
good_dataset_name_3.attrs["trust_score"] = 1

# merge rule based correspondences
all_rb_correspondences = pd.concat([rb_correspondences_k2u, rb_correspondences_k2y], ignore_index=True)

# define data fusion strategy. You should merge the datasets so that it makes sense. Also keep in mind to merge list attributes using union
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
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_rb_standard_blocker.csv", index=False)