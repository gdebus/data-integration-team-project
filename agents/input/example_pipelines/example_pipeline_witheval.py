# CRITICAL: Do not adjust output file names.
# CRITICAL: Always use OUTPUT_DIR for all output paths (correspondences, fusion, debug).
# This example shows a complete pipeline WITH inline evaluation.
# For pipelines without evaluation, see example_pipeline.py.

from PyDI.io import load_parquet, load_csv, load_xml
from PyDI.entitymatching import (
    StandardBlocker, EmbeddingBlocker, SortedNeighbourhoodBlocker, TokenBlocker,
)
from PyDI.entitymatching import StringComparator, NumericComparator, DateComparator
from PyDI.entitymatching import RuleBasedMatcher
from PyDI.fusion import (
    DataFusionStrategy, DataFusionEngine, DataFusionEvaluator,
    tokenized_match, exact_match, year_only_match, numeric_tolerance_match,
    boolean_match, set_equality_match,
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
import json
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
DATA_DIR = "input/datasets/"
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

good_dataset_name_1 = load_parquet(DATA_DIR + "<dataset-1>.parquet", name="dataset_name_1")
good_dataset_name_2 = load_parquet(DATA_DIR + "<dataset-2>.parquet", name="dataset_name_2")
good_dataset_name_3 = load_parquet(DATA_DIR + "<dataset-3>.parquet", name="dataset_name_3")

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# === 2. SCHEMA MATCHING ===
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
print("Performing Blocking")

blocker_1_2 = StandardBlocker(
    good_dataset_name_1, good_dataset_name_2,
    on=["city"], id_column="id", batch_size=100000, )
blocker_1_3 = StandardBlocker(
    good_dataset_name_1, good_dataset_name_3,
    on=["city"], id_column="id", batch_size=100000, )
blocker_2_3 = StandardBlocker(
    good_dataset_name_2, good_dataset_name_3,
    on=["city"], id_column="id", batch_size=100000, )

# === 5. ENTITY MATCHING ===
# CRITICAL: Use the matching config from "6. **MATCHING CONFIGURATION**".
print("Matching Entities")

threshold_dataset_name_1_dataset_name_2 = 0.7
threshold_dataset_name_1_dataset_name_3 = 0.7
threshold_dataset_name_2_dataset_name_3 = 0.7

comparators_1_2 = [
    StringComparator(column="name", similarity_function="jaro_winkler"),
    StringComparator(column="city", similarity_function="jaro_winkler", preprocess=str.lower),
    NumericComparator(column="house_number", method="absolute_difference", max_difference=2),
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
    weights=[0.5, 0.2, 0.1, 0.2],
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
CORR_DIR = os.path.join(OUTPUT_DIR, "correspondences")
os.makedirs(CORR_DIR, exist_ok=True)
rb_correspondences_1_2.to_csv(os.path.join(CORR_DIR, "correspondences_dataset_name_1_dataset_name_2.csv"), index=False)
rb_correspondences_1_3.to_csv(os.path.join(CORR_DIR, "correspondences_dataset_name_1_dataset_name_3.csv"), index=False)
rb_correspondences_2_3.to_csv(os.path.join(CORR_DIR, "correspondences_dataset_name_2_dataset_name_3.csv"), index=False)

all_rb_correspondences = pd.concat(
    [rb_correspondences_1_2, rb_correspondences_1_3, rb_correspondences_2_3], ignore_index=True)

# === 7. DATA FUSION ===
# DO NOT write custom fusers. Use only PyDI built-in resolvers listed above.
print("Fusing Data")

trust_map = {"dataset_name_1": 3, "dataset_name_2": 2, "dataset_name_3": 1}

strategy = DataFusionStrategy("fusion_strategy")

strategy.add_attribute_fuser("name", voting)
strategy.add_attribute_fuser("street", most_complete)
strategy.add_attribute_fuser("city", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("state", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("country", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("revenue", median)
strategy.add_attribute_fuser("latitude", average)
strategy.add_attribute_fuser("longitude", average)
strategy.add_attribute_fuser("house_number", voting)
strategy.add_attribute_fuser("founded", earliest)
strategy.add_attribute_fuser("categories", union, separator="; ")

# === 8. RUN FUSION ===
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
    correspondences=all_rb_correspondences,
    id_column="id",
    include_singletons=True,
)

fused_result.to_csv(os.path.join(FUSION_DIR, "fusion_data.csv"), index=False)

# === 9. EVALUATION ===
# See example_evaluation.py for full evaluation function reference and ID alignment helper.
#
# EVALUATION FUNCTION QUICK REFERENCE:
#   tokenized_match(threshold=0.85)       — fuzzy string (names, titles, addresses)
#   exact_match                           — strict equality (IDs, codes)
#   year_only_match                       — year component only (dates)
#   numeric_tolerance_match(tolerance=0.01) — numeric within tolerance
#   set_equality_match                    — exact set comparison (list attributes)
#   boolean_match                         — boolean comparison

print("Evaluating Data Fusion")

gold_standard = load_csv("<path-to-gold-standard.csv>", name="gold_standard")

# Register evaluation functions per attribute
strategy.add_evaluation_function("name", tokenized_match, threshold=0.85)
strategy.add_evaluation_function("street", tokenized_match, threshold=0.75)
strategy.add_evaluation_function("city", tokenized_match, threshold=0.75)
strategy.add_evaluation_function("country", tokenized_match, threshold=0.65)
strategy.add_evaluation_function("founded", year_only_match)
strategy.add_evaluation_function("revenue", numeric_tolerance_match, tolerance=0.01)
strategy.add_evaluation_function("categories", tokenized_match, threshold=0.75)

EVAL_DIR = os.path.join(OUTPUT_DIR, "pipeline_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)
evaluator = DataFusionEvaluator(
    strategy, debug=True, debug_format="json",
    debug_file=os.path.join(EVAL_DIR, "debug_fusion_eval.jsonl"),
)

evaluation_results = evaluator.evaluate(
    fused_df=fused_result,
    fused_id_column="id",
    gold_df=gold_standard,
    gold_id_column="id",
)

with open(os.path.join(EVAL_DIR, "pipeline_evaluation.json"), "w") as f:
    json.dump(evaluation_results, f, indent=4)
