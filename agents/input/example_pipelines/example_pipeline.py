# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_parquet, load_csv, load_xml

from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union, prefer_higher_trust
from PyDI.fusion import DataFusionEvaluator, tokenized_match

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

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

# Load Uber Eats dataset  
good_dataset_name_2 = load_parquet(
    DATA_DIR + "<path-and-file-name-of-dataset-2>.parquet",
    name="dataset_name_2",
)

# Load Yelp dataset
good_dataset_name_3 = load_parquet(
    DATA_DIR + "<path-and-file-name-of-dataset-3>.parquet",
    name="dataset_name_3",
)



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

# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# YOU MUST!!! USE THE PRECOMPUTED BLOCKERS PROVIEDED TO YOU UNDER "5. **BLOCKING CONFIGURATION**"!
# --------------------------------

print("Performing Blocking")

# Embedding blocker example:
#embedding_blocker_dataset1_2_dataset2 = EmbeddingBlocker(
#    good_dataset_name_1, good_dataset_name_2, # name of the datasets
#    text_cols=['city'], # column which should be used to perform the blocking on
#    model="sentence-transformers/all-MiniLM-L6-v2",
#    index_backend="sklearn",
#    top_k=20,          # Top 20 most similar
#    batch_size=1000,
#    output_dir="output/blocking-evaluation",
#    id_column='id'
#)

# Sorted NeighbourhoodBlocker example
#blocker = SortedNeighbourhoodBlocker(
#    df_left, df_right,
#    key="name",
#    window=20,
#    id_column="id",
#    output_dir="output/blocking"
#)

# TokenBlocker example
#blocker = TokenBlocker(
#    df_left, df_right,
#    column="name",
#    min_token_len=3,
#    ngram_size=2,
#    ngram_type="word",
#    id_column="id",
#    output_dir="output/blocking"
#)

# Standard blocker example:
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

blocker_u2y = StandardBlocker(
    good_dataset_name_2, good_dataset_name_3,
    on=['city'],
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column='id'
)

# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# You MUST use the matching configuration supplied to you under "6. **MATCHING CONFIGURATION**" to set the correct comparators in the following.
# --------------------------------
comparators_k2u = [
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

comparators_k2y = [
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

comparators_u2y = [
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

print("Matching Entities")

# Initialize Rule-Based Matcher
matcher = RuleBasedMatcher()

rb_correspondences_k2u = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_2, 
    candidates=blocker_k2u,
    comparators=comparators_k2u,
    weights=[0.5, 0.2, 0.2, 0.1], # weight the different comparators. Adjust accordingly so they make sense
    threshold=0.7,
    id_column='id'
)

rb_correspondences_k2y = matcher.match(
    df_left=good_dataset_name_1,
    df_right=good_dataset_name_3, 
    candidates=blocker_k2y,
    comparators=comparators_k2y,
    weights=[0.5, 0.2, 0.2, 0.1],  
    threshold=0.7,
    id_column='id'
)

rb_correspondences_u2y = matcher.match(
    df_left=good_dataset_name_2,
    df_right=good_dataset_name_3, 
    candidates=blocker_u2y,
    comparators=comparators_u2y,
    weights=[0.5, 0.2, 0.2, 0.1],  
    threshold=0.7,
    id_column='id'
)

print("Fusing Data")

# --------------------------------
# Data Fusion
# There are following conflict resolution functions available:
# For strings: longest_string, shortest_string, most_complete
# For numerics: average, median, maximum, minimum, sum_values
# For dates: most_recent, earliest
# For lists/sets: union
# --------------------------------

# merge rule based correspondences
all_rb_correspondences = pd.concat([rb_correspondences_k2u, rb_correspondences_k2y, rb_correspondences_u2y], ignore_index=True)

# define data fusion strategy
strategy = DataFusionStrategy('rule_based_fusion_strategy')

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