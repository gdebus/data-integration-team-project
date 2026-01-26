
import pandas as pd
import json

from PyDI.io import load_csv
from PyDI.fusion import DataFusionStrategy, longest_string, union
from PyDI.fusion import tokenized_match, set_equality_match
from PyDI.fusion import DataFusionEvaluator

# Load the fused output
fused = pd.read_csv("output/data_fusion/fusion_rb_standard_blocker.csv")

# Load the gold standard test set
fusion_test_set = load_csv('input/datasets/restaurant/testsets/Restaurant_Fusion_Test_Set.csv', name='fusion_test_set')

# Define the fusion strategy
strategy = DataFusionStrategy('ml_fusion_strategy')
strategy.add_evaluation_function("name", tokenized_match)
strategy.add_evaluation_function("street", tokenized_match)
strategy.add_evaluation_function("house_number", tokenized_match)
strategy.add_evaluation_function("city", tokenized_match)
strategy.add_evaluation_function("state", tokenized_match)
strategy.add_evaluation_function("postal_code", tokenized_match)
strategy.add_evaluation_function("country", tokenized_match)
strategy.add_evaluation_function("latitude", tokenized_match)
strategy.add_evaluation_function("longitude", tokenized_match)
strategy.add_evaluation_function("categories", set_equality_match)

# Initialize the evaluator
evaluator = DataFusionEvaluator(strategy, debug=True, debug_file="output/pipeline_evaluation/debug_fusion_eval.jsonl", debug_format="json")

# Evaluate the fusion results
evaluation_results = evaluator.evaluate(
    fused_df=fused,
    fused_id_column='_id',
    gold_df=fusion_test_set,
    gold_id_column='id',
)

# Print structured evaluation metrics
print(json.dumps(evaluation_results, indent=4))

# Print chosen evaluation functions in a compact one-line summary
print("Evaluation functions: name, street, house_number, city, state, postal_code, country, latitude, longitude, categories")
