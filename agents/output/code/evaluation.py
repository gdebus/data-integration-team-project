
import pandas as pd
import json

from PyDI.io import load_xml
from PyDI.fusion import (
    DataFusionStrategy,
    longest_string,
    union,
    DataFusionEvaluator,
    tokenized_match,
    year_only_match,
    set_equality_match,
    numeric_tolerance_match,
)

# --------------------------------
# Load Fused Output and Gold Standard
# --------------------------------

fused_path = "output/data_fusion/fusion_rb_standard_blocker.csv"
gold_path = "input/datasets/music/testsets/test_set.xml"

fused = pd.read_csv(fused_path)
fusion_test_set = load_xml(
    gold_path,
    name="fusion_test_set",
    nested_handling="aggregate",
)

# --------------------------------
# Recreate Fusion Strategy (must match pipeline)
# --------------------------------

strategy = DataFusionStrategy("music_fusion_strategy")

# Attribute fusers (same as in pipeline)
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)
strategy.add_attribute_fuser("duration", longest_string)
strategy.add_attribute_fuser("label", longest_string)
strategy.add_attribute_fuser("genre", longest_string)

strategy.add_attribute_fuser("tracks_track_name", union)
strategy.add_attribute_fuser("tracks_track_position", union)
strategy.add_attribute_fuser("tracks_track_duration", union)

# --------------------------------
# Define Evaluation Functions
# --------------------------------

# These specify how to compare fused vs. gold values
strategy.add_evaluation_function("name", tokenized_match)
strategy.add_evaluation_function("artist", tokenized_match)
strategy.add_evaluation_function("duration", numeric_tolerance_match)
strategy.add_evaluation_function("release-date", year_only_match)
strategy.add_evaluation_function("release-country", tokenized_match)
strategy.add_evaluation_function("label", tokenized_match)
strategy.add_evaluation_function("tracks_track_name", set_equality_match)

# --------------------------------
# Evaluate
# --------------------------------

evaluator = DataFusionEvaluator(
    strategy,
    debug=True,
    debug_file="output/pipeline_evaluation/debug_fusion_eval.jsonl",
    debug_format="json",
)

evaluation_results = evaluator.evaluate(
    fused_df=fused,
    fused_id_column="_id",   # ID column created by DataFusionEngine
    gold_df=fusion_test_set,
    gold_id_column="id",
)

# --------------------------------
# Print Structured Metrics
# --------------------------------

print("\n=== Data Fusion Evaluation Metrics ===")
overall = evaluation_results.get("overall", {})
print(f"Overall precision: {overall.get('precision')}")
print(f"Overall recall:    {overall.get('recall')}")
print(f"Overall f1:        {overall.get('f1')}")

per_attr = evaluation_results.get("per_attribute", {})
print("\nPer-attribute metrics:")
for attr, metrics in per_attr.items():
    p = metrics.get("precision")
    r = metrics.get("recall")
    f1 = metrics.get("f1")
    print(f"  {attr}: precision={p}, recall={r}, f1={f1}")

# --------------------------------
# Print Evaluation Functions (Compact Summary)
# --------------------------------

eval_funcs = {
    "name": "tokenized_match",
    "artist": "tokenized_match",
    "duration": "numeric_tolerance_match",
    "release-date": "year_only_match",
    "release-country": "tokenized_match",
    "label": "tokenized_match",
    "tracks_track_name": "set_equality_match",
}

print("\nEvaluation functions (attribute=function):")
print(", ".join(f"{attr}={func}" for attr, func in eval_funcs.items()))

# --------------------------------
# Write Evaluation Output to JSON
# --------------------------------

evaluation_output = "output/pipeline_evaluation/pipeline_evaluation.json"
with open(evaluation_output, "w") as f:
    json.dump(evaluation_results, f, indent=4)
