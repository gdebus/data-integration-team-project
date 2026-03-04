import pandas as pd
import json
import ast
from collections import Counter
from pathlib import Path
import sys

from PyDI.io import load_xml, load_parquet, load_csv
from PyDI.fusion import DataFusionStrategy, longest_string, shortest_string, union, prefer_higher_trust, voting, maximum
from PyDI.fusion import tokenized_match, year_only_match, set_equality_match, numeric_tolerance_match
from PyDI.fusion import DataFusionEvaluator 

try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _candidates = [
        Path.cwd(),
        Path.cwd() / "agents",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
    ]
    for _path in _candidates:
        if (_path / "list_normalization.py").is_file():
            _path_str = str(_path.resolve())
            if _path_str not in sys.path:
                sys.path.append(_path_str)
    from list_normalization import detect_list_like_columns, normalize_list_like_columns


def _parse_source_ids(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text:
        return []
    if text[0] not in "[({":
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(v) for v in parsed if str(v)]
        except Exception:
            pass
    return []


def _detect_gold_prefix(gold_ids):
    prefixes = [g.split("_", 1)[0] for g in gold_ids if "_" in g]
    if not prefixes:
        return None
    return Counter(prefixes).most_common(1)[0][0] + "_"


def build_eval_view(fused_df, gold_df):
    gold_ids = set(gold_df["id"].dropna().astype(str).tolist()) if "id" in gold_df.columns else set()
    gold_prefix = _detect_gold_prefix(gold_ids)
    expanded_rows = []

    for _, row in fused_df.iterrows():
        row_dict = row.to_dict()
        cluster_id = str(row_dict.get("_id", ""))
        source_ids = _parse_source_ids(row_dict.get("_fusion_sources"))

        candidates = []
        if gold_prefix:
            candidates = [sid for sid in source_ids if sid.startswith(gold_prefix)]
        if not candidates:
            candidates = [sid for sid in source_ids if sid in gold_ids]
        if not candidates and cluster_id:
            candidates = [cluster_id]

        seen = set()
        for eval_id in candidates:
            if eval_id in seen:
                continue
            seen.add(eval_id)
            enriched = dict(row_dict)
            enriched["eval_id"] = eval_id
            enriched["_eval_cluster_id"] = cluster_id
            expanded_rows.append(enriched)

    if not expanded_rows:
        fallback = fused_df.copy()
        fallback["eval_id"] = fallback["_id"].astype(str)
        fallback["_eval_cluster_id"] = fallback["_id"].astype(str)
        return fallback

    fused_eval = pd.DataFrame(expanded_rows)
    if "_fusion_confidence" in fused_eval.columns:
        fused_eval["_fusion_confidence"] = pd.to_numeric(
            fused_eval["_fusion_confidence"], errors="coerce"
        ).fillna(0.0)
        fused_eval = fused_eval.sort_values("_fusion_confidence", ascending=False)
    fused_eval = fused_eval.drop_duplicates(subset=["eval_id"], keep="first")
    return fused_eval

# load test set and fusion set
fused = pd.read_csv("output/data_fusion/fusion_data.csv")
fusion_test_set = load_xml('<path-to-testset>', name='fusion_test_set', nested_handling='aggregate')
fused_eval = build_eval_view(fused, fusion_test_set)
list_eval_columns = detect_list_like_columns(
    [fused_eval, fusion_test_set],
    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id", "_fusion_confidence"},
)
list_eval_columns = [column for column in list_eval_columns if not column.startswith("_")]
if list_eval_columns:
    fused_eval, fusion_test_set = normalize_list_like_columns(
        [fused_eval, fusion_test_set], list_eval_columns
    )
    print(f"[LIST NORMALIZATION] columns: {', '.join(list_eval_columns)}")

gold_ids = set(fusion_test_set["id"].dropna().astype(str).tolist()) if "id" in fusion_test_set.columns else set()
direct_cov = len(set(fused["_id"].astype(str).tolist()) & gold_ids) if "_id" in fused.columns else 0
mapped_cov = len(set(fused_eval["eval_id"].astype(str).tolist()) & gold_ids) if "eval_id" in fused_eval.columns else 0
print(f"[ID ALIGNMENT] direct _id coverage: {direct_cov}/{len(gold_ids)}")
print(f"[ID ALIGNMENT] mapped eval_id coverage: {mapped_cov}/{len(gold_ids)}")

# 'DataFusionStrategy' object has NO attribute 'evaluation_functions'. It has the attribute 'add_evaluation_function'
strategy = DataFusionStrategy('music_fusion_strategy')
eval_funcs_summary = {}


def register_eval(column, fn, fn_name):
    strategy.add_evaluation_function(column, fn)
    eval_funcs_summary[column] = fn_name


register_eval("name", tokenized_match, "tokenized_match")
register_eval("artist", tokenized_match, "tokenized_match")
register_eval("duration", numeric_tolerance_match, "numeric_tolerance_match")
register_eval("release-date", year_only_match, "year_only_match")
register_eval("release-country", tokenized_match, "tokenized_match")
register_eval("label", tokenized_match, "tokenized_match")
register_eval("tracks_track_name", set_equality_match, "set_equality_match")

for column in list_eval_columns:
    if column not in eval_funcs_summary:
        register_eval(column, set_equality_match, "set_equality_match")

evaluator = DataFusionEvaluator(strategy, debug=True, debug_file="output/pipeline_evaluation/debug_fusion_eval.jsonl", debug_format="json")

evaluation_results = evaluator.evaluate(
    fused_df=fused_eval,
    fused_id_column='eval_id',
    gold_df=fusion_test_set,
    gold_id_column='id',
)

# CRITICAL INSTRUCTION FOR AGENTS:
# The output should ONLY be written to the output JSON file as follows
evaluation_output = "output/pipeline_evaluation/pipeline_evaluation.json"
with open(evaluation_output, 'w') as f:
    json.dump(evaluation_results, f, indent=4)