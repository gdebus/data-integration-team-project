"""Run the agent pipeline on the Games dataset."""
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

I = INPUT_DIR + "datasets/games/"
datasets = [I + "dbpedia.xml", I + "metacritic.xml", I + "sales.xml"]

profile_tool = ProfileDatasetTool()
search_tool = SearchDocumentationTool()
agent = SimpleModelAgent(llm, tools={profile_tool.name: profile_tool, search_tool.name: search_tool})
attach_logging(agent, output_dir=config.OUTPUT_DIR, notebook_name="run_games", use_case="games", llm_model="gpt-5.4")

print("[*] Starting games run...")
result = agent.graph.invoke({
    "datasets": datasets,
    "original_datasets": list(datasets),
    "normalized_datasets": [],
    "entity_matching_testsets": {
        ("dbpedia", "sales"): I + "testsets/dbpedia_2_sales_test.csv",
        ("metacritic", "dbpedia"): I + "testsets/metacritic_2_dbpedia_test.csv",
        ("metacritic", "sales"): I + "testsets/metacritic_2_sales_test.csv",
    },
    "fusion_testset": I + "testsets/test_set_fusion.xml",
    "validation_fusion_testset": I + "testsets/validation_set_fusion.xml",
    "matcher_mode": "rulebased",
    "evaluation_attempts": 0,
    "normalization_attempts": 0,
    "normalization_execution_result": "",
    "normalization_rework_required": False,
    "normalization_rework_reasons": [],
    "normalization_directives": {},
    "investigator_decision": "",
}, config={"recursion_limit": GRAPH_RECURSION_LIMIT})

print("[*] Games run complete")
out = {
    "overall_accuracy": (result.get("evaluation_metrics") or {}).get("overall_accuracy"),
    "macro_accuracy": (result.get("evaluation_metrics") or {}).get("macro_accuracy"),
    "sealed_test_overall": (result.get("sealed_test_metrics_final") or {}).get("overall_accuracy"),
    "evaluation_attempts": result.get("evaluation_attempts"),
    "run_id": result.get("run_id"),
}
print("[SUMMARY]", json.dumps(out, indent=2, default=str))
