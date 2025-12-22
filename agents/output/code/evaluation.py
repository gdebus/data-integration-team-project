
import pandas as pd
import json

from PyDI.io import load_xml
from PyDI.fusion import (
    DataFusionStrategy,
    DataFusionEvaluator,
    longest_string,
    union,
    prefer_higher_trust,
)
from PyDI.fusion import (
    tokenized_match,
    year_only_match,
    set_equality_match,
    numeric_tolerance_match,
)
import numpy as np

# ----------------------------------------------------
# Load fused output and gold standard test set
# ----------------------------------------------------

fused = pd.read_csv("output/data_fusion/fusion_rb_standard_blocker.csv")

fusion_test_set = load_xml(
    "input/testsets/music/test_set.xml",
    name="fusion_test_set",
    nested_handling="aggregate",
)

# ----------------------------------------------------
# Recreate the fusion strategy used in the pipeline
# ----------------------------------------------------

trust_scores = {
    "discogs": 0.9,
    "musicbrainz": 0.85,
    "lastfm": 0.7,
}

strategy = DataFusionStrategy("music_release_fusion_strategy")

# Helper fuser to prefer higher trust, then longest string
def trusted_string_fuser(values, sources):
    return prefer_higher_trust(values, sources, trust_scores, fallback=longest_string)

# String attributes (as in the pipeline)
for attr in ["name", "artist", "label", "release-country", "release-date", "genre"]:
    strategy.add_attribute_fuser(attr, trusted_string_fuser)

# Duration fuser (trust-based numeric)
def duration_fuser(values, sources):
    numeric_vals = [v for v in values if pd.notna(v)]
    if not numeric_vals:
        return np.nan
    return prefer_higher_trust(numeric_vals, sources, trust_scores)

strategy.add_attribute_fuser("duration", duration_fuser)

# Track-level attributes: union
for attr in ["tracks_track_name", "tracks_track_position", "tracks_track_duration"]:
    strategy.add_attribute_fuser(attr, union)

# ----------------------------------------------------
# Add evaluation / comparison functions
# ----------------------------------------------------

strategy.add_evaluation_function("name", tokenized_match)
strategy.add_evaluation_function("artist", tokenized_match)
strategy.add_evaluation_function("duration", numeric_tolerance_match)
strategy.add_evaluation_function("release-date", year_only_match)
strategy.add_evaluation_function("release-country", tokenized_match)
strategy.add_evaluation_function("label", tokenized_match)
strategy.add_evaluation_function("tracks_track_name", set_equality_match)

# ----------------------------------------------------
# Run evaluation
# ----------------------------------------------------

evaluator = DataFusionEvaluator(
    strategy,
    debug=True,
    debug_file="output/pipeline_evaluation/debug_fusion_eval.jsonl",
    debug_format="json",
)

evaluation_results = evaluator.evaluate(
    fused_df=fused,
    fused_id_column="_id",  # ID column from fusion output
    gold_df=fusion_test_set,
    gold_id_column="id",    # ID column in gold standard
)

# ----------------------------------------------------
# Print structured evaluation metrics
# ----------------------------------------------------

print(json.dumps(evaluation_results, indent=4))

# ----------------------------------------------------
# Save evaluation output to JSON file (as required)
# ----------------------------------------------------

evaluation_output = "output/pipeline_evaluation/pipeline_evaluation.json"
with open(evaluation_output, "w") as f:
    json.dump(evaluation_results, f, indent=4)
