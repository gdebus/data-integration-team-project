"""Run the agent pipeline on the Music dataset.

Pre-populates blocking_config and matching_config from a previous successful
run so those expensive tester phases are skipped entirely.
"""
import sys; sys.dont_write_bytecode = True
import json
import os
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parent
os.chdir(str(AGENTS_DIR))

from _resolve_import import ensure_project_root
ensure_project_root()

import config
from config import INPUT_DIR, GRAPH_RECURSION_LIMIT, LLM_REQUEST_TIMEOUT, OPENAI_MAX_RETRIES
from pipeline_agent import SimpleModelAgent, ProfileDatasetTool, SearchDocumentationTool
from required_logging import attach_logging
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5.4", request_timeout=LLM_REQUEST_TIMEOUT, max_retries=OPENAI_MAX_RETRIES)

I = INPUT_DIR + "datasets/music/"
datasets = [I + "discogs.xml", I + "lastfm.xml", I + "musicbrainz.xml"]

# ── Pre-computed configs (skip blocking + matching testers) ──────────────
BLOCKING_CONFIG = {
    "blocking_strategies": {
        "discogs_lastfm": {
            "strategy": "semantic_similarity",
            "columns": ["name", "artist"],
            "params": {"top_k": 10},
            "pair_completeness": 0.9642857142857143,
            "num_candidates": 225831,
            "is_acceptable": True,
        },
        "discogs_musicbrainz": {
            "strategy": "sorted_neighbourhood",
            "columns": ["name"],
            "params": {"window": 15},
            "pair_completeness": 0.9,
            "num_candidates": 115280,
            "is_acceptable": True,
        },
        "musicbrainz_lastfm": {
            "strategy": "semantic_similarity",
            "columns": ["name", "artist", "duration"],
            "params": {"top_k": 10},
            "pair_completeness": 0.9240506329113924,
            "num_candidates": 47615,
            "is_acceptable": True,
        },
    },
    "id_columns": {"discogs": "id", "lastfm": "id", "musicbrainz": "id"},
}

MATCHING_CONFIG = {
    "id_columns": {"discogs": "id", "lastfm": "id", "musicbrainz": "id"},
    "matching_strategies": {
        "discogs_lastfm": {
            "comparators": [
                {"type": "string", "column": "artist", "similarity_function": "jaro_winkler", "preprocess": "lower_strip", "list_strategy": "concatenate"},
                {"type": "string", "column": "name", "similarity_function": "jaro_winkler", "preprocess": "lower_strip", "list_strategy": "concatenate"},
                {"type": "string", "column": "tracks_track_name", "similarity_function": "cosine", "preprocess": "lower_strip", "list_strategy": "set_jaccard"},
                {"type": "numeric", "column": "duration", "max_difference": 10.0},
            ],
            "weights": [0.3, 0.35, 0.25, 0.1],
            "threshold": 0.6,
            "f1": 0.896551724137931,
        },
        "discogs_musicbrainz": {
            "comparators": [
                {"type": "string", "column": "name", "similarity_function": "jaro_winkler", "preprocess": "lower_strip", "list_strategy": "concatenate"},
                {"type": "string", "column": "artist", "similarity_function": "jaro_winkler", "preprocess": "lower_strip", "list_strategy": "concatenate"},
                {"type": "date", "column": "release-date", "max_days_difference": 365},
                {"type": "string", "column": "release-country", "similarity_function": "cosine", "preprocess": "lower_strip", "list_strategy": "concatenate"},
                {"type": "numeric", "column": "duration", "max_difference": 60.0},
            ],
            "weights": [0.35, 0.25, 0.2, 0.1, 0.1],
            "threshold": 0.7,
            "f1": 0.8732394366197184,
        },
        "musicbrainz_lastfm": {
            "comparators": [
                {"type": "string", "column": "artist", "similarity_function": "jaro_winkler", "preprocess": "lower_strip", "list_strategy": "concatenate"},
                {"type": "string", "column": "name", "similarity_function": "cosine", "preprocess": "lower_strip", "list_strategy": "concatenate"},
                {"type": "string", "column": "duration", "similarity_function": "jaro_winkler", "preprocess": "lower_strip", "list_strategy": "concatenate"},
            ],
            "weights": [0.35, 0.45, 0.2],
            "threshold": 0.75,
            "f1": 0.8652482269503546,
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────

profile_tool = ProfileDatasetTool()
search_tool = SearchDocumentationTool()
agent = SimpleModelAgent(llm, tools={profile_tool.name: profile_tool, search_tool.name: search_tool})
attach_logging(agent, output_dir=config.OUTPUT_DIR, notebook_name="run_music", use_case="music", llm_model="gpt-5.4")

print("[*] Starting music run (blocking+matching pre-loaded)...")
result = agent.graph.invoke({
    "datasets": datasets,
    "original_datasets": list(datasets),
    "normalized_datasets": [],
    "entity_matching_testsets": {
        ("discogs", "lastfm"): I + "testsets/discogs_lastfm_goldstandard_blocking.csv",
        ("discogs", "musicbrainz"): I + "testsets/discogs_musicbrainz_goldstandard_blocking.csv",
        ("musicbrainz", "lastfm"): I + "testsets/musicbrainz_lastfm_goldstandard_blocking.csv",
    },
    "fusion_testset": I + "testsets/test_set.xml",
    "validation_fusion_testset": I + "testsets/validation_set.xml",
    "matcher_mode": "rulebased",
    "blocking_config": BLOCKING_CONFIG,
    "matching_config": MATCHING_CONFIG,
    "evaluation_attempts": 0,
    "normalization_attempts": 0,
    "normalization_execution_result": "",
    "normalization_rework_required": False,
    "normalization_rework_reasons": [],
    "normalization_directives": {},
    "investigator_decision": "",
}, config={"recursion_limit": GRAPH_RECURSION_LIMIT})

print("[*] Music run complete")
out = {
    "overall_accuracy": (result.get("evaluation_metrics") or {}).get("overall_accuracy"),
    "macro_accuracy": (result.get("evaluation_metrics") or {}).get("macro_accuracy"),
    "sealed_test_overall": (result.get("sealed_test_metrics_final") or {}).get("overall_accuracy"),
    "evaluation_attempts": result.get("evaluation_attempts"),
    "run_id": result.get("run_id"),
}
print("[SUMMARY]", json.dumps(out, indent=2, default=str))
