# CRITICAL: Do not adjust output file names.
# CRITICAL: Always use OUTPUT_DIR for all output paths (correspondences, fusion, debug).

from PyDI.io import load_parquet, load_csv, load_xml
from PyDI.entitymatching import (
    StandardBlocker, EmbeddingBlocker, SortedNeighbourhoodBlocker, TokenBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.entitymatching import (
    MaximumBipartiteMatching,
    StableMatching,
    GreedyOneToOneMatchingAlgorithm,
    HierarchicalClusterer,
    ConnectedComponentClusterer,
)
from PyDI.fusion import (
    DataFusionStrategy, DataFusionEngine,
    # String: voting, longest_string, shortest_string, most_complete
    # Numeric: median, average, maximum, minimum, sum_values
    # Date: most_recent, earliest
    # List: union, intersection, intersection_k_sources  (pass separator= for delimited strings)
    # Trust: prefer_higher_trust, favour_sources
    voting, longest_string, most_complete,
    median, average,
    most_recent, earliest,
    union, intersection, intersection_k_sources,
    prefer_higher_trust, favour_sources,
)
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _candidates = [Path.cwd(), Path.cwd() / "agents",
                   Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent,
                   Path(__file__).resolve().parent.parent.parent]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            if str(_path.resolve()) not in sys.path:
                sys.path.append(str(_path.resolve()))
    from list_normalization import detect_list_like_columns, normalize_list_like_columns

# === 0. OUTPUT DIRECTORY ===
# CRITICAL: Use OUTPUT_DIR for ALL output paths. It will be provided in the prompt.
# Do NOT hardcode "output/" — always use os.path.join(OUTPUT_DIR, ...).
OUTPUT_DIR = "output"  # Will be replaced by prompt with the actual run-scoped directory

# === 1. LOAD DATA ===
# Use the correct loader: load_csv, load_parquet, load_xml
# For XML with nested elements: load_xml("path.xml", nested_handling="aggregate")

DATA_DIR = "input/datasets/"
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_parquet(DATA_DIR + "<dataset-1>.parquet", name="dataset_name_1")
good_dataset_name_2 = load_parquet(DATA_DIR + "<dataset-2>.parquet", name="dataset_name_2")
good_dataset_name_3 = load_parquet(DATA_DIR + "<dataset-3>.parquet", name="dataset_name_3")

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# === 2. SCHEMA MATCHING ===
# Aligns dataset2/dataset3 column names to match dataset1.

print("Matching Schema")
llm = ChatOpenAI(model="gpt-5.1")
matcher = LLMBasedSchemaMatcher(chat_model=llm, num_rows=10, debug=True)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_2)
good_dataset_name_2 = good_dataset_name_2.rename(
    columns=schema_correspondences.set_index("target_column")["source_column"].to_dict())

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_3)
good_dataset_name_3 = good_dataset_name_3.rename(
    columns=schema_correspondences.set_index("target_column")["source_column"].to_dict())

# === 3. LIST NORMALIZATION ===
# Converts delimited strings ("pop; rock; jazz") to Python lists so list_strategy
# comparators and list fusers (union, intersection) work correctly.

list_like_columns = detect_list_like_columns(
    [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
    exclude_columns={"id", "_id"},
)
if list_like_columns:
    (good_dataset_name_1, good_dataset_name_2, good_dataset_name_3,
     ) = normalize_list_like_columns(
        [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3], list_like_columns)
    print(f"Normalized list-like columns: {', '.join(list_like_columns)}")

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# === 4. BLOCKING ===
# CRITICAL: Use the precomputed blocking config from "5. **BLOCKING CONFIGURATION**".
# See example_pipeline.py for blocker type signatures.

print("Performing Blocking")

blocker_1_2 = StandardBlocker(
    good_dataset_name_1, good_dataset_name_2,
    on=["city"], id_column="id", batch_size=100000,
)
blocker_1_3 = StandardBlocker(
    good_dataset_name_1, good_dataset_name_3,
    on=["city"], id_column="id", batch_size=100000,
)
blocker_2_3 = StandardBlocker(
    good_dataset_name_2, good_dataset_name_3,
    on=["city"], id_column="id", batch_size=100000,
)

# === 5. ENTITY MATCHING ===
# CRITICAL: Use the matching config from "6. **MATCHING CONFIGURATION**".
# See example_pipeline.py for comparator signatures and list_strategy values.

print("Matching Entities")

threshold_dataset_name_1_dataset_name_2 = 0.7
threshold_dataset_name_1_dataset_name_3 = 0.7
threshold_dataset_name_2_dataset_name_3 = 0.7

comparators_1_2 = [
    StringComparator(column="name", similarity_function="jaro_winkler"),
    StringComparator(column="city", similarity_function="jaro_winkler", preprocess=str.lower),
    NumericComparator(column="house_number", method="absolute_difference", max_difference=2),
    DateComparator(column="founded", max_days_difference=365),
    StringComparator(column="categories", similarity_function="jaccard",
                     preprocess=str.lower, list_strategy="set_jaccard"),
]

comparators_1_3 = [
    StringComparator(column="name", similarity_function="jaro_winkler"),
    StringComparator(column="city", similarity_function="jaro_winkler", preprocess=str.lower),
    NumericComparator(column="house_number", max_difference=2),
    StringComparator(column="categories", similarity_function="jaccard",
                     preprocess=str.lower, list_strategy="set_jaccard"),
]

comparators_2_3 = [
    StringComparator(column="name", similarity_function="jaro_winkler"),
    StringComparator(column="city", similarity_function="jaro_winkler", preprocess=str.lower),
    NumericComparator(column="house_number", max_difference=2),
    StringComparator(column="categories", similarity_function="jaccard",
                     preprocess=str.lower, list_strategy="set_jaccard"),
]

matcher = RuleBasedMatcher()

rb_correspondences_1_2 = matcher.match(
    df_left=good_dataset_name_1, df_right=good_dataset_name_2,
    candidates=blocker_1_2, comparators=comparators_1_2,
    weights=[0.4, 0.2, 0.1, 0.15, 0.15],
    threshold=threshold_dataset_name_1_dataset_name_2, id_column="id",
)
rb_correspondences_1_3 = matcher.match(
    df_left=good_dataset_name_1, df_right=good_dataset_name_3,
    candidates=blocker_1_3, comparators=comparators_1_3,
    weights=[0.5, 0.2, 0.1, 0.2],
    threshold=threshold_dataset_name_1_dataset_name_3, id_column="id",
)
rb_correspondences_2_3 = matcher.match(
    df_left=good_dataset_name_2, df_right=good_dataset_name_3,
    candidates=blocker_2_3, comparators=comparators_2_3,
    weights=[0.5, 0.2, 0.1, 0.2],
    threshold=threshold_dataset_name_2_dataset_name_3, id_column="id",
)

# === 6. SAVE CORRESPONDENCES (MANDATORY) ===
# Save per-pair CSVs BEFORE post-clustering. Naming: correspondences_<left>_<right>.csv

CORR_DIR = os.path.join(OUTPUT_DIR, "correspondences")
os.makedirs(CORR_DIR, exist_ok=True)
rb_correspondences_1_2.to_csv(os.path.join(CORR_DIR, "correspondences_dataset_name_1_dataset_name_2.csv"), index=False)
rb_correspondences_1_3.to_csv(os.path.join(CORR_DIR, "correspondences_dataset_name_1_dataset_name_3.csv"), index=False)
rb_correspondences_2_3.to_csv(os.path.join(CORR_DIR, "correspondences_dataset_name_2_dataset_name_3.csv"), index=False)

# === 7. POST-CLUSTERING ===
# Refines raw matcher output. Apply per-pair BEFORE merging into all_correspondences.
#
# STRATEGY SELECTION GUIDE (choose based on cluster analysis evidence):
#   MaximumBipartiteMatching()           — 1:1, optimal total score (Hungarian). Best default.
#   StableMatching()                     — 1:1, stable assignment (Gale-Shapley). Best when ambiguity is high.
#   GreedyOneToOneMatchingAlgorithm()    — 1:1, greedy highest-score-first. Fast but approximate.
#   HierarchicalClusterer(linkage_mode=LinkageMode.AVG, min_similarity=0.5)
#     from PyDI.entitymatching import LinkageMode  — Many-to-many. Splits over-merged clusters.
#   ConnectedComponentClusterer()        — Baseline transitive closure.
#
# You can use DIFFERENT strategies per pair based on evidence:
#   refined_1_2 = MaximumBipartiteMatching().cluster(rb_correspondences_1_2)
#   refined_1_3 = StableMatching().cluster(rb_correspondences_1_3)

clusterer = MaximumBipartiteMatching()
refined_1_2 = clusterer.cluster(rb_correspondences_1_2)
refined_1_3 = clusterer.cluster(rb_correspondences_1_3)
refined_2_3 = clusterer.cluster(rb_correspondences_2_3)

all_correspondences = pd.concat([refined_1_2, refined_1_3, refined_2_3], ignore_index=True)

# === 8. DATA FUSION ===
# See example_pipeline.py for full resolver reference and choosing guidelines.
# DO NOT write custom fusers. Use only PyDI built-in resolvers listed above.

print("Fusing Data")

trust_map = {"dataset_name_1": 3, "dataset_name_2": 2, "dataset_name_3": 1}

strategy = DataFusionStrategy("fusion_strategy")

# String attributes
strategy.add_attribute_fuser("name", voting)
strategy.add_attribute_fuser("street", most_complete)
strategy.add_attribute_fuser("city", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("state", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("country", prefer_higher_trust, trust_map=trust_map)

# Numeric attributes
strategy.add_attribute_fuser("revenue", median)
strategy.add_attribute_fuser("latitude", average)
strategy.add_attribute_fuser("longitude", average)
strategy.add_attribute_fuser("house_number", voting)

# Date attributes
strategy.add_attribute_fuser("founded", earliest)

# List attributes — always pass separator= for delimited strings
strategy.add_attribute_fuser("categories", union, separator="; ")

# === 9. RUN FUSION ===
# CRITICAL: Always include_singletons=True for full fused dataset.
# CRITICAL: Use OUTPUT_DIR for all output paths.

FUSION_DIR = os.path.join(OUTPUT_DIR, "data_fusion")
os.makedirs(FUSION_DIR, exist_ok=True)
engine = DataFusionEngine(
    strategy, debug=True, debug_format="json",
    debug_file=os.path.join(FUSION_DIR, "debug_fusion_data.jsonl"),
)

fused_result = engine.run(
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
    correspondences=all_correspondences,
    id_column="id",
    include_singletons=True,
)

fused_result.to_csv(os.path.join(FUSION_DIR, "fusion_data.csv"), index=False)
