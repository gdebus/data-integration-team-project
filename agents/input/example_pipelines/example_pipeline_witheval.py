from PyDI.io import load_parquet, load_csv, load_xml

from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker, SortedNeighbourhoodBlocker, TokenBlocker
from PyDI.entitymatching import EntityMatchingEvaluator
from PyDI.entitymatching import StringComparator, NumericComparator
from PyDI.entitymatching import RuleBasedMatcher

from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union, prefer_higher_trust
from PyDI.fusion import DataFusionEvaluator, tokenized_match, exact_match
from PyDI.fusion import DataFusionEvaluator, tokenized_match

from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_google_genai import ChatGoogleGenerativeAI

import pandas as pd
import numpy as np
import os

def evaluate_and_select_best_blocker(blockers, ground_truth, output_dir):
    """
    Evaluates multiple blocking methods and selects the best one based on Pair Completeness and Reduction Ratio.
    
    Args:
        blockers (dict): Dictionary of {name: blocker_instance}
        ground_truth (pd.DataFrame): DataFrame containing ground truth pairs (id1, id2, label)
        output_dir (str): Directory to save evaluation results
        
    Returns:
        tuple: (best_blocker_instance, best_candidates)
    """
    evaluator = EntityMatchingEvaluator()
    results = []
    
    print("\n--- Evaluating Blocking Algorithms ---")
    
    for name, blocker in blockers.items():
        print(f"Evaluating {name}...")
        # Materialize candidates for evaluation
        candidates = blocker.materialize()
        
        # Evaluate against ground truth
        result = evaluator.evaluate_blocking(
            candidate_pairs=candidates,
            blocker=blocker,
            test_pairs=ground_truth,
            out_dir=output_dir
        )
        
        # Store results
        result['method'] = name
        result['candidates'] = candidates
        result['blocker'] = blocker
        results.append(result)
        
        print(f"  {name}: PC={result['pair_completeness']:.3f}, RR={result['reduction_ratio']:.3f}")
        
    # Select best: prioritize pair_completeness, then reduction_ratio
    # You might want to adjust this logic depending on whether recall (PC) or efficiency (RR) is more important
    best_result = max(results, key=lambda x: (x['pair_completeness'], x['reduction_ratio']))
    
    print(f"\n🏆 Best blocking method: {best_result['method']} "
          f"(PC: {best_result['pair_completeness']:.3f}, RR: {best_result['reduction_ratio']:.3f})")
          
    return best_result['blocker'], best_result['candidates']

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

# Define dataset paths
DATA_DIR = "datasets/"

# Define API Key

os.environ["GOOGLE_API_KEY"] = "<API-KEY>"

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



datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

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
# Perform Entity Matching
# Employ the embedding Blocker per default. If the size of the datasets is too large, use the StandardBlocker
# --------------------------------

print("Performing Blocking")

# -------------------------------------------------------------------------
# BLOCKING STRATEGY & COLUMN SELECTION
# -------------------------------------------------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# Picking the right columns for blocking is the MOST VITAL step in Entity Matching.
# 
# 1.  **Selectivity**: Choose columns that are distinctive enough to separate non-matches but broad enough to group potential matches.
#     - BAD: 'gender', 'country' (too large blocks, too many comparisons)
#     - GOOD: 'city', 'zip_code', 'last_name', 'product_code'
#
# 2.  **Data Quality**: The blocking column must be relatively clean. If 'city' has many typos ("New York" vs "Nw York"), 
#     StandardBlocker will fail to group them. Use TokenBlocker or EmbeddingBlocker for dirty data.
#
# 3.  **Completeness**: Avoid columns with many NULL values.
#
# 4.  **Multiple Keys**: Sometimes a single column isn't enough. You can block on multiple columns (e.g., ['city', 'zip_code']).
# -------------------------------------------------------------------------

# Define output directory for blocking evaluation
blocking_output_dir = "output/blocking-evaluation"

# --- Define Candidate Blockers for Dataset 1 <-> Dataset 2 ---

# 1. Standard Blocker: Exact match on 'city'. Fast, but requires clean data.
standard_blocker_k2u = StandardBlocker(
    good_dataset_name_1, good_dataset_name_2,
    on=['city'], 
    batch_size=1000,
    output_dir=blocking_output_dir,
    id_column='id'
)

# 2. Sorted Neighbourhood Blocker: Sorts by 'city' and compares neighbors. Good for slight typos.
sn_blocker_k2u = SortedNeighbourhoodBlocker(
    good_dataset_name_1, good_dataset_name_2,
    key='city',
    window=5, # Check 5 neighbors
    batch_size=1000,
    output_dir=blocking_output_dir,
    id_column='id'
)

# 3. Token Blocker: Tokenizes 'name' and blocks on shared tokens (e.g., 3-grams). Robust to word swaps/typos.
token_blocker_k2u = TokenBlocker(
    good_dataset_name_1, good_dataset_name_2,
    column='name', # Assuming 'name' column exists, adjust as needed
    ngram_size=3,
    batch_size=1000,
    output_dir=blocking_output_dir,
    id_column='id'
)

# 4. Embedding Blocker: Semantic similarity. Best for unstructured text or when names vary significantly.
#    Note: Requires a sentence-transformer model.
embedding_blocker_k2u = EmbeddingBlocker(
    good_dataset_name_1, good_dataset_name_2,
    text_cols=['name', 'city'], # Combine columns for embedding
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=500,
    output_dir=blocking_output_dir,
    id_column='id'
)

# --- Evaluate and Select Best Blocker ---

# Dictionary of blockers to evaluate
blockers_to_evaluate = {
    'Standard (City)': standard_blocker_k2u,
    'SortedNeighbourhood (City)': sn_blocker_k2u,
    'Token (Name)': token_blocker_k2u,
    'Embedding (Name+City)': embedding_blocker_k2u
}

# Load Ground Truth for Evaluation (If available)
# Ideally, you should have a small labeled dataset to evaluate blocking performance.
# ground_truth_df = load_csv("path/to/ground_truth.csv", header=None, names=['id1', 'id2', 'label'])

# IF ground truth is available, uncomment the following lines:
# best_blocker_k2u, candidates_k2u = evaluate_and_select_best_blocker(
#     blockers_to_evaluate, 
#     ground_truth_df, 
#     blocking_output_dir
# )
# blocker_k2u = best_blocker_k2u

# ELSE, manually select the one you think is best based on data profiling:
print("Selecting StandardBlocker as default (No ground truth provided for auto-selection)")
blocker_k2u = standard_blocker_k2u


# --- Define Blocker for Dataset 1 <-> Dataset 3 (Repeat process) ---
# For brevity, using StandardBlocker here, but you should apply the same selection logic.
blocker_k2y = StandardBlocker(
    good_dataset_name_1, good_dataset_name_3,
    on=['city'],
    batch_size=1000,
    output_dir=blocking_output_dir,
    id_column='id'
)

# -------------------------------------------------------------------------
# COMPARATOR SELECTION & CONFIGURATION
# -------------------------------------------------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# Choosing the right comparators and similarity functions is crucial for accurate matching.
#
# 1. String Attributes (Names, Cities, Descriptions):
#    - 'levenshtein': Best for short strings with typos (e.g., "Apple" vs "Aple").
#    - 'jaccard': Best for token-based overlap, good for multi-word strings where order doesn't matter (e.g., "Burger King" vs "King Burger").
#    - 'jaro_winkler': Good for names, gives higher weight to matching prefixes.
#    - ALWAYS use `preprocess=str.lower` to ignore case differences.
#
# 2. Numeric Attributes (Prices, Years, Coordinates):
#    - Use NumericComparator.
#    - Set `max_difference` based on domain knowledge (e.g., year +/- 1 is okay, price +/- 0.05 is okay).
#
# 3. List/Set Attributes (Categories, Authors):
#    - Use StringComparator with `list_strategy='concatenate'` or specific list comparators if available.
#    - 'jaccard' is usually the best metric here.
#
# 4. Weights (in RuleBasedMatcher):
#    - Assign higher weights (e.g., 0.5 - 0.8) to distinctive attributes (Name, ID).
#    - Assign lower weights (e.g., 0.1 - 0.3) to common attributes (City, Category).
# -------------------------------------------------------------------------

# define comparators
comparators = [
    # Name similarity - High importance
    # Using Jaccard here as word order might vary, but Levenshtein is also a strong candidate for names.
    StringComparator(
        column='name_norm',
        similarity_function='jaccard', 
        preprocess=str.lower,
    ),
    
    # Street name similarity - Medium importance
    StringComparator(
        column='street',
        similarity_function='jaccard', 
        preprocess=str.lower,
    ),

    # House number similarity - High precision required
    NumericComparator(
        column='house_number',
        max_difference=2, # Allow small deviation (e.g. 10 vs 12)
    ),

    # Category similarity - Low importance / Supporting evidence
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

print("Fusing Data")

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

# --------------------------------
# Data Fusion Evaluation
# --------------------------------
print("Evaluating Data Fusion")

# -------------------------------------------------------------------------
# FUSION EVALUATION INSTRUCTIONS
# -------------------------------------------------------------------------
# To evaluate the quality of the fused dataset, we compare it against a "Gold Standard".
# The Gold Standard is a manually verified dataset containing the "correct" values for the entities.
#
# 1. Load Gold Standard:
#    - Ensure the file path is correct.
#    - Use `load_csv` or `load_xml` depending on the file format.
#
# 2. Define Evaluation Functions:
#    - For each attribute you want to evaluate, assign a comparison function.
#    - `tokenized_match`: Good for strings where word order might differ (e.g. "Cafe Nero" vs "Nero Cafe").
#    - `exact_match`: Strict equality.
#    - `boolean_match`: For True/False flags.
# -------------------------------------------------------------------------

# 1. Load Gold Standard
# Replace with the actual path to your gold standard file
# gold_standard_df = load_csv(
#     "path/to/gold_standard.csv",
#     name="gold_standard",
# )

# 2. Define Evaluation Functions
# Add evaluation functions to the strategy for the attributes you want to check.
# strategy.add_evaluation_function("name", tokenized_match)
# strategy.add_evaluation_function("street", tokenized_match)
# strategy.add_evaluation_function("city", exact_match)
# strategy.add_evaluation_function("categories", tokenized_match)

# 3. Run Evaluation
# Uncomment the following lines when you have a gold standard loaded.

# evaluator = DataFusionEvaluator(strategy)

# evaluation_results = evaluator.evaluate(
#     fused_df=rb_fused_standard_blocker,
#     fused_id_column='id',      # The ID column in your fused output
#     gold_df=gold_standard_df,  # The DataFrame loaded above
#     gold_id_column='id',       # The ID column in your gold standard
# )

# print("\nFusion Evaluation Results:")
# print("=" * 40)
# for metric, value in evaluation_results.items():
#     if isinstance(value, float):
#         print(f"  {metric}: {value:.3f}")
#     else:
#         print(f"  {metric}: {value}")
