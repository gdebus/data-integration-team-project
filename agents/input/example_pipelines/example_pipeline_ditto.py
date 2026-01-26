# --------------------------------
# CRITICAL GENERAL INSTRUCTION FOR AGENTS: Do not adjust the names of the output files
# --------------------------------

from PyDI.io import load_parquet, load_csv, load_xml
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker, TokenBlocker, SortedNeighbourhoodBlocker
from PyDI.fusion import DataFusionStrategy, DataFusionEngine, longest_string, union
from PyDI.schemamatching import LLMBasedSchemaMatcher
from langchain_openai import ChatOpenAI

import pandas as pd
import json
import os
import sys
from argparse import Namespace

# Ditto repo is a local git clone; add it to sys.path
DITTO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "ditto"))
if DITTO_DIR not in sys.path:
    sys.path.insert(0, DITTO_DIR)

from ditto_light.dataset import DittoDataset
from ditto_light.ditto import train
from matcher import load_model, predict, tune_threshold

# --------------------------------
# Prepare Data
# Important: Use the correct loader (load_parquet, load_csv, load_xml)
# --------------------------------

DATA_DIR = "input/datasets/"

# Load the first dataset
good_dataset_name_1 = load_parquet(
    DATA_DIR + "<path-and-file-name-of-dataset-1>.parquet",
    name="dataset_name_1",
)

# Load the second dataset
good_dataset_name_2 = load_parquet(
    DATA_DIR + "<path-and-file-name-of-dataset-2>.parquet",
    name="dataset_name_2",
)

# Load the third dataset
good_dataset_name_3 = load_parquet(
    DATA_DIR + "<path-and-file-name-of-dataset-3>.parquet",
    name="dataset_name_3",
)

# create id columns (replace with actual id columns)
good_dataset_name_1["id"] = good_dataset_name_1["<dataset1_id_column>"]
good_dataset_name_2["id"] = good_dataset_name_2["<dataset2_id_column>"]
good_dataset_name_3["id"] = good_dataset_name_3["<dataset3_id_column>"]

datasets = [good_dataset_name_1, good_dataset_name_2, good_dataset_name_3]

# --------------------------------
# Perform Schema Matching (LLM-based matching)
# CRITICAL INSTRUCTION FOR AGENTS:
# The here implemented schema matching will match the schema of dataset2 and dataset3 to the schema of dataset1.
# --------------------------------

print("Matching Schema")

llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_tokens=None,
)

matcher = LLMBasedSchemaMatcher(
    chat_model=llm,
    num_rows=10,
    debug=True
)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_2)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
good_dataset_name_2 = good_dataset_name_2.rename(columns=rename_map)

schema_correspondences = matcher.match(good_dataset_name_1, good_dataset_name_3)
rename_map = (
    schema_correspondences
    .set_index("target_column")["source_column"]
    .to_dict()
)
good_dataset_name_3 = good_dataset_name_3.rename(columns=rename_map)

# --------------------------------
# CRITICAL INSTRUCTION FOR AGENTS:
# YOU MUST USE THE PRECOMPUTED BLOCKER TYPES AND PARAMETER SETTINGS PROVIDED TO YOU
# LATER IN JSON UNDER "5. **BLOCKING CONFIGURATION**".
# --------------------------------

print("Performing Blocking")

# Blocking examples (DO NOT use these directly; use precomputed configs instead)
blocker_1_2 = EmbeddingBlocker(
    good_dataset_name_1, good_dataset_name_2,
    text_cols=['city'],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    top_k=20,
    batch_size=1000,
    output_dir="output/blocking-evaluation",
    id_column='id'
)

blocker_1_3 = TokenBlocker(
    good_dataset_name_1, good_dataset_name_3,
    column="name",
    min_token_len=3,
    ngram_size=2,
    ngram_type="word",
    id_column="id",
    output_dir="output/blocking"
)

blocker_2_3 = SortedNeighbourhoodBlocker(
    good_dataset_name_2, good_dataset_name_3,
    key="name",
    window=20,
    id_column="id",
    output_dir="output/blocking"
)

# --------------------------------
# DITTO MATCHER (NO COMPARATORS)
# CRITICAL INSTRUCTION FOR AGENTS:
# Ditto does NOT use comparators or weights. It needs:
# 1) Materialized candidate pairs from blocking (id1, id2)
# 2) Record-pair JSONL for prediction
# 3) Training TSV (serialized) for training
# --------------------------------

def _stringify_value(val):
    if isinstance(val, list):
        return ", ".join(map(str, val))
    if pd.isna(val):
        return ""
    return str(val)

def _serialize_record(row_dict):
    # Ditto training format: "COL <attr> VAL <value> ..."
    parts = []
    for k, v in row_dict.items():
        parts.append(f"COL {k} VAL {_stringify_value(v)}")
    return " ".join(parts)

def _write_ditto_jsonl(pairs_df, df_left, df_right, id_left, id_right, output_path):
    left_idx = df_left.set_index(id_left, drop=False)
    right_idx = df_right.set_index(id_right, drop=False)
    common_cols = [c for c in df_left.columns if c in df_right.columns and c not in (id_left, id_right)]
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in pairs_df.iterrows():
            l_id, r_id = row["id1"], row["id2"]
            left_full = left_idx.loc[l_id]
            right_full = right_idx.loc[r_id]
            left = {c: left_full[c] for c in common_cols}
            right = {c: right_full[c] for c in common_cols}
            f.write(json.dumps([left, right]) + "\n")

def _write_ditto_train_tsv(gold_df, df_left, df_right, id_left, id_right, output_path):
    left_idx = df_left.set_index(id_left, drop=False)
    right_idx = df_right.set_index(id_right, drop=False)
    common_cols = [c for c in df_left.columns if c in df_right.columns and c not in (id_left, id_right)]
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in gold_df.iterrows():
            l_id, r_id, label = row["id1"], row["id2"], int(row["label"])
            left_full = left_idx.loc[l_id]
            right_full = right_idx.loc[r_id]
            left = _serialize_record({c: left_full[c] for c in common_cols})
            right = _serialize_record({c: right_full[c] for c in common_cols})
            f.write(left + "\t" + right + "\t" + str(label) + "\n")

# --------------------------------
# Train Ditto & Predict (per dataset pair)
# --------------------------------

def train_ditto_model(train_tsv, valid_tsv, test_tsv, run_tag, logdir, lm="distilbert", max_len=128):
    hp = Namespace(
        batch_size=64,
        max_len=max_len,
        lr=3e-5,
        n_epochs=20,
        lm=lm,
        fp16=False,
        alpha_aug=0.8,
        logdir=logdir,
    )
    trainset = DittoDataset(train_tsv, lm=lm, max_len=max_len)
    validset = DittoDataset(valid_tsv, lm=lm, max_len=max_len)
    testset = DittoDataset(test_tsv, lm=lm, max_len=max_len)
    train(trainset, validset, testset, run_tag, hp)

def predict_ditto(task, input_path, output_path, checkpoint_path, lm="distilbert", max_len=128):
    config, model = load_model(task, checkpoint_path, lm, use_gpu=True, fp16=False)
    threshold = tune_threshold(config, model, Namespace(task=task, summarize=False, dk=None, lm=lm, max_len=max_len))
    predict(input_path, output_path, config, model, lm=lm, max_len=max_len, threshold=threshold)

def _predictions_to_correspondences(candidate_pairs, output_path):
    labels = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            labels.append(int(obj.get("match", 0)))
    corr = candidate_pairs[["id1", "id2"]].copy()
    corr["label"] = labels
    return corr

# --------------------------------
# Run Ditto for all dataset pairs
# --------------------------------

pairs = [
    (good_dataset_name_1, good_dataset_name_2, "dataset_name_1", "dataset_name_2"),
    (good_dataset_name_1, good_dataset_name_3, "dataset_name_1", "dataset_name_3"),
    (good_dataset_name_2, good_dataset_name_3, "dataset_name_2", "dataset_name_3"),
]

all_correspondences = []

for left_df, right_df, left_name, right_name in pairs:
    # NOTE: use the precomputed blocker configuration here to create the blocker
    if left_name == "dataset_name_1" and right_name == "dataset_name_2":
        blocker = blocker_1_2
    elif left_name == "dataset_name_1" and right_name == "dataset_name_3":
        blocker = blocker_1_3
    else:
        blocker = blocker_2_3
    candidate_pairs = blocker.materialize()  # DataFrame with id1, id2

    # Build Ditto JSONL for prediction
    jsonl_path = f"output/ditto/candidates_{left_name}_{right_name}.jsonl"
    _write_ditto_jsonl(
        candidate_pairs,
        left_df,
        right_df,
        id_left="id",
        id_right="id",
        output_path=jsonl_path
    )

    # Build Ditto training TSV from labeled pairs (id1, id2, label)
    train_tsv = f"output/ditto/train_{left_name}_{right_name}.txt"
    valid_tsv = f"output/ditto/valid_{left_name}_{right_name}.txt"
    test_tsv = f"output/ditto/test_{left_name}_{right_name}.txt"
    gold_pairs = pd.read_csv(f"input/testsets/<train_pairs_{left_name}_{right_name}>.csv")
    _write_ditto_train_tsv(gold_pairs, left_df, right_df, id_left="id", id_right="id", output_path=train_tsv)
    _write_ditto_train_tsv(gold_pairs, left_df, right_df, id_left="id", id_right="id", output_path=valid_tsv)
    _write_ditto_train_tsv(gold_pairs, left_df, right_df, id_left="id", id_right="id", output_path=test_tsv)

    # Train Ditto and predict
    run_tag = f"{left_name}_{right_name}"
    train_ditto_model(train_tsv, valid_tsv, test_tsv, run_tag, logdir="output/ditto/checkpoints")

    output_path = f"output/ditto/preds_{left_name}_{right_name}.jsonl"
    predict_ditto(
        task=run_tag,
        input_path=jsonl_path,
        output_path=output_path,
        checkpoint_path="output/ditto/checkpoints",
    )

    # Convert Ditto output to DataFrame (id1, id2, score)
    corr = _predictions_to_correspondences(candidate_pairs, output_path)
    all_correspondences.append(corr)

# Merge correspondences across pairs
correspondences_df = pd.concat(all_correspondences, ignore_index=True)

# --------------------------------
# Data Fusion (same as other example pipelines)
# --------------------------------

strategy = DataFusionStrategy('ditto_fusion_strategy')
strategy.add_attribute_fuser('name', longest_string)
strategy.add_attribute_fuser('street', longest_string)
strategy.add_attribute_fuser('house_number', longest_string)
strategy.add_attribute_fuser('city', longest_string)
strategy.add_attribute_fuser('state', longest_string)
strategy.add_attribute_fuser('postal_code', longest_string)
strategy.add_attribute_fuser('country', longest_string)
strategy.add_attribute_fuser('latitude', longest_string)
strategy.add_attribute_fuser('longitude', longest_string)
strategy.add_attribute_fuser('categories', union)

engine = DataFusionEngine(strategy, debug=True, debug_format='json', debug_file="output/data_fusion/debug_fusion_ditto.jsonl")
ditto_fused = engine.run(
    datasets=[good_dataset_name_1, good_dataset_name_2, good_dataset_name_3],
    correspondences=correspondences_df,
    id_column="id",
    include_singletons=False,
)
ditto_fused.to_csv("output/data_fusion/fusion_ditto_blocker.csv", index=False)
