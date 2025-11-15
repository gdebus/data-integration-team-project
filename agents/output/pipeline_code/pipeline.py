```python
from PyDI.io import load_parquet
from PyDI.entitymatching import StandardBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union, prefer_higher_trust
from PyDI.fusion import DataFusionEvaluator, tokenized_match

import pandas as pd
import numpy as np

# Define dataset paths
DATA_DIR = "input/datasets/"

# Load the first dataset
amazon_dataset = load_parquet(
    DATA_DIR + "amazon_small.parquet",
    name="amazon_dataset",
)

# Load Goodreads dataset  
goodreads_dataset = load_parquet(
    DATA_DIR + "goodreads_small.parquet",
    name="goodreads_dataset",
)

# Load Metabooks dataset
metabooks_dataset = load_parquet(
    DATA_DIR + "metabooks_small.parquet",
    name="metabooks_dataset",
)

# create a column called �id� for further processing if such a column does not already exist
amazon_dataset['id'] = amazon_dataset['id']
goodreads_dataset['id'] = goodreads_dataset['id']
metabooks_dataset['id'] = metabooks_dataset['id']

datasets = [amazon_dataset, goodreads_dataset, metabooks_dataset]

# Perform Entity Matching
blocker_amazon_goodreads = StandardBlocker(
    amazon_dataset, goodreads_dataset,
    on=['title'], # column which should be used to perform the blocking on
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column='id'
)

blocker_amazon_metabooks = StandardBlocker(
    amazon_dataset, metabooks_dataset,
    on=['title'],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column='id'
)

# define comparators
comparators = [
    # title similarity
    StringComparator(
        column='title',
        similarity_function='jaccard', 
        # no preprocessing needed
    ),
    
    # author similarity
    StringComparator(
        column='book-author',
        similarity_function='jaccard', 
        preprocess=str.lower,
    ),

    # publisher similarity
    StringComparator(
        column='publisher',
        similarity_function='jaccard',
        preprocess=str.lower,
    ),

    # year of publication similarity
    NumericComparator(
        column='year-of-publication',
        max_difference=2,
    ),
]

# Initialize Rule-Based Matcher
matcher = RuleBasedMatcher()

rb_correspondences_amazon_goodreads = matcher.match(
    df_left=amazon_dataset,
    df_right=goodreads_dataset, 
    candidates=blocker_amazon_goodreads,
    comparators=comparators,
    weights=[0.4, 0.3, 0.2, 0.1], 
    threshold=0.7,
    id_column='id'
)

rb_correspondences_amazon_metabooks = matcher.match(
    df_left=amazon_dataset,
    df_right=metabooks_dataset, 
    candidates=blocker_amazon_metabooks,
    comparators=comparators,
    weights=[0.4, 0.3, 0.2, 0.1],  
    threshold=0.7,
    id_column='id'
)

# Data Fusion
# set trust scores
amazon_dataset.attrs["trust_score"] = 3
goodreads_dataset.attrs["trust_score"] = 2
metabooks_dataset.attrs["trust_score"] = 1

# merge rule based correspondences
all_rb_correspondences = pd.concat([rb_correspondences_amazon_goodreads, rb_correspondences_amazon_metabooks], ignore_index=True)

# define data fusion strategy
strategy = DataFusionStrategy('book_fusion_strategy')

strategy.add_attribute_fuser('title', longest_string)
strategy.add_attribute_fuser('book-author', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('publisher', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('year-of-publication', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('isbn_clean', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('genres', union)

# run fusion
engine = DataFusionEngine(strategy, debug=True, debug_format='json',debug_file= "output/data_fusion/debug_fusion_rb_standard_blocker.jsonl")

# fuse rule based matches
rb_fused_standard_blocker = engine.run(
    datasets=[amazon_dataset, goodreads_dataset, metabooks_dataset],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_rb_standard_blocker.csv", index=False)
```