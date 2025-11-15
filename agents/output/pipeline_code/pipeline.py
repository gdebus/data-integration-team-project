```python
from PyDI.io import load_parquet, load_csv, load_xml

from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union, prefer_higher_trust
from PyDI.fusion import DataFusionEvaluator, tokenized_match

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_google_genai import ChatGoogleGenerativeAI

import pandas as pd
import numpy as np
import os

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

# Define dataset paths
DATA_DIR = "input/datasets/"

# Define API Key

os.environ["GOOGLE_API_KEY"] = "<API-KEY>"

# Load the first dataset
amazon_small = load_parquet(
    DATA_DIR + "amazon_small.parquet",
    name="amazon_small",
)

# Load goodreads dataset  
goodreads_small = load_parquet(
    DATA_DIR + "goodreads_small.parquet",
    name="goodreads_small",
)

# Load metabooks dataset
metabooks_small = load_parquet(
    DATA_DIR + "metabooks_small.parquet",
    name="metabooks_small",
)



datasets = [amazon_small, goodreads_small, metabooks_small]

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# --------------------------------

print("Matching Schema")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True
)

# match schema of amazon_small with goodreads_small and rename schema of goodreads_small
schema_correspondences = matcher.match(amazon_small, goodreads_small)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
goodreads_small = goodreads_small.rename(columns=rename_map)

# match schema of amazon_small with metabooks_small and rename schema of metabooks_small
schema_correspondences = matcher.match(amazon_small, metabooks_small)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
metabooks_small = metabooks_small.rename(columns=rename_map)

# --------------------------------
# Perform Entity Matching
# Employ the embedding Blocker per default. If the size of the datasets is too large, use the StandardBlocker
# --------------------------------

print("Performing Blocking")

# Example of an Embedding blocker:
# embedding_blocker_dataset1_2_dataset2 = EmbeddingBlocker(
#    amazon_small, goodreads_small, # name of the datasets
#    text_cols=['title'], # column which should be used to perform the blocking on
#    model="sentence-transformers/all-MiniLM-L6-v2",
#    index_backend="sklearn",
#    top_k=20,          # Top 20 most similar
#    batch_size=500,
#    output_dir="output/blocking-evaluation",
#    id_column='id'
#)

# Example of a Standard blocker:
# blocker_k2u = StandardBlocker(
#    amazon_small, goodreads_small,
#    on=['title'], # column which should be used to perform the blocking on
#    batch_size=1000,
#    output_dir="output/blocking-evaluation",
#    id_column='id'
#)

# define blocking in this example using the standard blocker
blocker_a2g = StandardBlocker(
    amazon_small, goodreads_small,
    on=['title'],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column='id'
)

blocker_a2m = StandardBlocker(
    amazon_small, metabooks_small,
    on=['title'],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column='id'
)

# define comparators
# Which and how many comparators to use is open to you. Use them so they make sense in the context of the data within the dataset. 
comparators = [
    # Name similarity
    StringComparator(
        column='title',
        similarity_function='jaccard', 
        # no preprocessing needed
    ),
    
    # author name similarity
    StringComparator(
        column='book-author',
        similarity_function='jaccard', 
        preprocess=str.lower,
    ),

    # year similarity
    NumericComparator(
        column='year-of-publication',
        max_difference=2,
    ),

    # publisher similarity - supporting evidence
    StringComparator(
        column='publisher',
        similarity_function='jaccard',
        preprocess=str.lower,
        list_strategy='concatenate' # Handle list attribute by concatenation
    )
]

print("Matching Entities")

# Initialize Rule-Based Matcher
matcher = RuleBasedMatcher()

rb_correspondences_a2g = matcher.match(
    df_left=amazon_small,
    df_right=goodreads_small, 
    candidates=blocker_a2g,
    comparators=comparators,
    weights=[0.5, 0.2, 0.2, 0.1], # weight the different comparators. Adjust accordingly so they make sense
    threshold=0.7,
    id_column='id'
)

rb_correspondences_a2m = matcher.match(
    df_left=amazon_small,
    df_right=metabooks_small, 
    candidates=blocker_a2m,
    comparators=comparators,
    weights=[0.5, 0.2, 0.2, 0.1],  
    threshold=0.7,
    id_column='id'
)

print("Fusing Data")

# --------------------------------
# Data Fusion
# --------------------------------
# set trust scores. You may draw your own assumptions about which dataset should be trusted most
amazon_small.attrs["trust_score"] = 3
goodreads_small.attrs["trust_score"] = 2
metabooks_small.attrs["trust_score"] = 1

# merge rule based correspondences
all_rb_correspondences = pd.concat([rb_correspondences_a2g, rb_correspondences_a2m], ignore_index=True)

# define data fusion strategy. You should merge the datasets so that it makes sense. Also keep in mind to merge list attributes using union
strategy = DataFusionStrategy('book_fusion_strategy')

strategy.add_attribute_fuser('title', longest_string)
strategy.add_attribute_fuser('book-author', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('year-of-publication', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('publisher', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('isbn_clean', prefer_higher_trust, trust_key="trust_score")
strategy.add_attribute_fuser('genres', union)

# run fusion
engine = DataFusionEngine(strategy, debug=True, debug_format='json',debug_file= "output/data_fusion/debug_fusion_rb_standard_blocker.jsonl")

# fuse rule based matches
rb_fused_standard_blocker = engine.run(
    datasets=[amazon_small, goodreads_small, metabooks_small],
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=False,
)

# write output
rb_fused_standard_blocker.to_csv("output/data_fusion/fusion_rb_standard_blocker.csv", index=False)
```