
import json
import pandas as pd

from PyDI.io import load_xml
from PyDI.fusion import (
    DataFusionStrategy,
    tokenized_match,
    year_only_match,
    set_equality_match,
    numeric_tolerance_match,
)
from PyDI.fusion import DataFusionEvaluator

# --------------------------------
# Paths (do not change output file name)
# --------------------------------
FUSED_PATH = "output/data_fusion/fusion_rb_standard_blocker.csv"
GOLD_PATH = "input/testsets/music/test_set.xml"
EVALUATION_OUTPUT = "output/pipeline_evaluation/pipeline_evaluation.json"

# --------------------------------
# Load fused output + gold test set
# --------------------------------
fused = pd.read_csv(FUSED_PATH)
gold = load_xml(GOLD_PATH, name="fusion_test_set", nested_handling="aggregate")

# --------------------------------
# Evaluation fusion strategy (must match the evaluation functions expected for this task)
# --------------------------------
strategy = DataFusionStrategy("music_release_fusion_strategy")

strategy.add_evaluation_function("name", tokenized_match)
strategy.add_evaluation_function("artist", tokenized_match)
strategy.add_evaluation_function("duration", numeric_tolerance_match)
strategy.add_evaluation_function("release-date", year_only_match)
strategy.add_evaluation_function("release-country", tokenized_match)
strategy.add_evaluation_function("label", tokenized_match)
strategy.add_evaluation_function("tracks_track_name", set_equality_match)

# --------------------------------
# Evaluate + print structured metrics
# --------------------------------
evaluator = DataFusionEvaluator(
    strategy,
    debug=True,
    debug_file="output/pipeline_evaluation/debug_fusion_eval.jsonl",
    debug_format="json",
)

evaluation_results = evaluator.evaluate(
    fused_df=fused,
    fused_id_column="_id",
    gold_df=gold,
    gold_id_column="id",
)

print(json.dumps(evaluation_results, indent=2))

# --------------------------------
# CRITICAL: write results to the required JSON output file
# --------------------------------
with open(EVALUATION_OUTPUT, "w") as f:
    json.dump(evaluation_results, f, indent=4)
