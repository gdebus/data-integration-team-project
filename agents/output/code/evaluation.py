
import pandas as pd
import json

from PyDI.io import load_xml
from PyDI.fusion import (
    DataFusionStrategy,
    longest_string,
    union,
    DataFusionEvaluator,
    tokenized_match,
)

# --------------------------------
# Load fused output and gold standard
# --------------------------------

fused_path = "output/data_fusion/fusion_rb_standard_blocker.csv"
gold_path = "input/testsets/music/test_set.xml"

fused = pd.read_csv(fused_path)

# Gold / test set is XML
fusion_test_set = load_xml(
    gold_path,
    name="fusion_test_set",
    nested_handling="aggregate",  # aggregate track-level lists, as in example
)

# --------------------------------
# Define fusion strategy consistent with pipeline
# --------------------------------

strategy = DataFusionStrategy("music_release_fusion_strategy")

# Core attributes
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)

# Duration: prefer the longest non-null value
strategy.add_attribute_fuser("duration", longest_string)

# Release info
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)

# Additional metadata
strategy.add_attribute_fuser("label", longest_string)
strategy.add_attribute_fuser("genre", union)

# Track-level attributes
strategy.add_attribute_fuser("tracks_track_name", union)
strategy.add_attribute_fuser("tracks_track_position", union)
strategy.add_attribute_fuser("tracks_track_duration", union)

# --------------------------------
# Configure evaluation functions per attribute
# --------------------------------

# We use tokenized_match (set-based text match) for text-like fields
# and leave complex types (lists) to set-like comparison via tokenized_match as well.
strategy.add_evaluation_function("name", tokenized_match)
strategy.add_evaluation_function("artist", tokenized_match)
strategy.add_evaluation_function("duration", tokenized_match)
strategy.add_evaluation_function("release-date", tokenized_match)
strategy.add_evaluation_function("release-country", tokenized_match)
strategy.add_evaluation_function("label", tokenized_match)
strategy.add_evaluation_function("genre", tokenized_match)
strategy.add_evaluation_function("tracks_track_name", tokenized_match)
strategy.add_evaluation_function("tracks_track_position", tokenized_match)
strategy.add_evaluation_function("tracks_track_duration", tokenized_match)

# --------------------------------
# Run evaluation
# --------------------------------

evaluator = DataFusionEvaluator(
    strategy,
    debug=True,
    debug_file="output/pipeline_evaluation/debug_fusion_eval.jsonl",
    debug_format="json",
)

evaluation_results = evaluator.evaluate(
    fused_df=fused,
    fused_id_column="_id",  # id column produced by fusion engine
    gold_df=fusion_test_set,
    gold_id_column="id",    # id column in gold / test set
)

# --------------------------------
# Print structured metrics
# --------------------------------

print(json.dumps(evaluation_results, indent=4))

# --------------------------------
# Persist evaluation metrics to JSON
# --------------------------------

evaluation_output = "output/pipeline_evaluation/pipeline_evaluation.json"
with open(evaluation_output, "w") as f:
    json.dump(evaluation_results, f, indent=4)
