
import pandas as pd
import json

from PyDI.io import load_xml
from PyDI.fusion import (
    DataFusionStrategy,
    longest_string,
    union,
)
from PyDI.fusion import (
    DataFusionEvaluator,
    tokenized_match,
    year_only_match,
    numeric_tolerance_match,
    set_equality_match,
)

# --------------------------------
# Load fused output and gold test set
# --------------------------------

fused = pd.read_csv("output/data_fusion/fusion_rb_standard_blocker.csv")

fusion_test_set = load_xml(
    "input/testsets/music/test_set.xml",
    name="fusion_test_set",
    nested_handling="aggregate",
)

# --------------------------------
# Recreate the fusion strategy used in the pipeline
# --------------------------------

strategy = DataFusionStrategy("music_release_fusion_strategy")

# Attribute fusers (must match the integration pipeline)

# Titles & artist
strategy.add_attribute_fuser("name", longest_string)
strategy.add_attribute_fuser("artist", longest_string)

# Dates & countries
strategy.add_attribute_fuser("release-date", longest_string)
strategy.add_attribute_fuser("release-country", longest_string)

# Duration and label
strategy.add_attribute_fuser("duration", longest_string)
strategy.add_attribute_fuser("label", longest_string)

# Genre: union
strategy.add_attribute_fuser("genre", union)

# Track-level attributes
strategy.add_attribute_fuser("tracks_track_name", longest_string)
strategy.add_attribute_fuser("tracks_track_position", longest_string)
strategy.add_attribute_fuser("tracks_track_duration", longest_string)

# --------------------------------
# Configure evaluation functions
# --------------------------------

# Token-based equality for string-like attributes
strategy.add_evaluation_function("name", tokenized_match)
strategy.add_evaluation_function("artist", tokenized_match)
strategy.add_evaluation_function("release-country", tokenized_match)
strategy.add_evaluation_function("label", tokenized_match)
strategy.add_evaluation_function("genre", tokenized_match)
strategy.add_evaluation_function("tracks_track_name", set_equality_match)

# Dates: compare only the year component
strategy.add_evaluation_function("release-date", year_only_match)

# Numeric tolerance for duration (seconds)
strategy.add_evaluation_function("duration", numeric_tolerance_match)

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
    fused_id_column="_id",   # ID column in fused output
    gold_df=fusion_test_set,
    gold_id_column="id",     # ID column in gold standard
)

# Print structured metrics to stdout
print(json.dumps(evaluation_results, indent=4))

# --------------------------------
# Write evaluation results to file (REQUIRED)
# --------------------------------

evaluation_output = "output/pipeline_evaluation/pipeline_evaluation.json"
with open(evaluation_output, "w") as f:
    json.dump(evaluation_results, f, indent=4)
