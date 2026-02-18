import pandas as pd
import json

from PyDI.io import load_xml, load_parquet, load_csv
from PyDI.fusion import DataFusionStrategy, longest_string, shortest_string, union, prefer_higher_trust, voting, maximum
from PyDI.fusion import tokenized_match, exact_match, year_only_match, set_equality_match, numeric_tolerance_match, boolean_match
from PyDI.fusion import DataFusionEvaluator 

# load test set and fusion set
fused = pd.read_csv("output/data_fusion/fusion_data.csv")
fusion_test_set = load_xml('<path-to-testset>', name='fusion_test_set', nested_handling='aggregate')

# 'DataFusionStrategy' object has NO attribute 'evaluation_functions'. It has the attribute 'add_evaluation_function'
strategy = DataFusionStrategy('music_fusion_strategy')
strategy.add_evaluation_function("name", tokenized_match, threshold=0.9) # threshold goes from 0 to 1.0. 1.0 is exact match. 
strategy.add_evaluation_function("artist", tokenized_match, threshold=0.9)
strategy.add_evaluation_function("duration", numeric_tolerance_match, tolerance=5) # tolerance between numeric values
strategy.add_evaluation_function("release-date", year_only_match)
strategy.add_evaluation_function("release-country", tokenized_match, threshold=0.9)
strategy.add_evaluation_function("label", tokenized_match, threshold=0.9)
strategy.add_evaluation_function("tracks_track_name", set_equality_match, threshold=0.9)

evaluator = DataFusionEvaluator(strategy, debug=True, debug_file="output/pipeline_evaluation/debug_fusion_eval.jsonl", debug_format="json")

evaluation_results = evaluator.evaluate(
    fused_df=fused,
    fused_id_column='_id',
    gold_df=fusion_test_set,
    gold_id_column='id',
)

# CRITICAL INSTRUCTION FOR AGENTS:
# The output should ONLY be written to the output JSON file as follows
evaluation_output = "output/pipeline_evaluation/pipeline_evaluation.json"
with open(evaluation_output, 'w') as f:
    json.dump(evaluation_results, f, indent=4)