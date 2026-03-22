"""Run the agent pipeline on the Companies dataset."""
import json
import os
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parent
os.chdir(str(AGENTS_DIR))

# Direct imports — no exec() bootstrapping needed
from _resolve_import import ensure_project_root
ensure_project_root()

import config
from config import INPUT_DIR, GRAPH_RECURSION_LIMIT
from pipeline_agent import SimpleModelAgent, ProfileDatasetTool, SearchDocumentationTool
from required_logging import attach_logging

from langchain_openai import ChatOpenAI
from config import LLM_REQUEST_TIMEOUT, OPENAI_MAX_RETRIES

llm = ChatOpenAI(
    model="gpt-5.4",
    request_timeout=LLM_REQUEST_TIMEOUT,
    max_retries=OPENAI_MAX_RETRIES,
)

# --- Dataset configuration ---
entity_matching_testsets = {
    ("forbes", "dbpedia"): INPUT_DIR + "datasets/companies/testsets/forbes_dbpedia_art.csv",
    ("forbes", "fullcontact"): INPUT_DIR + "datasets/companies/testsets/forbes_fullcontact_art.csv",
    ("fullcontact", "dbpedia"): INPUT_DIR + "datasets/companies/testsets/fullcontact_dbpedia_art.csv",
}
datasets = [
    INPUT_DIR + "datasets/companies/forbes.csv",
    INPUT_DIR + "datasets/companies/dbpedia.xml",
    INPUT_DIR + "datasets/companies/fullcontact.xml",
]
fusion_testset = INPUT_DIR + "datasets/companies/testsets/test_set.xml"
validation_fusion_testset = INPUT_DIR + "datasets/companies/testsets/validation_set.xml"

# --- Build agent ---
profile_tool = ProfileDatasetTool()
search_tool = SearchDocumentationTool()
all_tools = {profile_tool.name: profile_tool, search_tool.name: search_tool}
agent = SimpleModelAgent(llm, tools=all_tools)
attach_logging(agent, output_dir=config.OUTPUT_DIR, notebook_name="run_companies", use_case="companies", llm_model="gpt-5.4")

# --- Invoke ---
invoke_payload = {
    "datasets": datasets,
    "original_datasets": list(datasets),
    "normalized_datasets": [],
    "entity_matching_testsets": entity_matching_testsets,
    "fusion_testset": fusion_testset,
    "matcher_mode": "rulebased",
    "evaluation_attempts": 0,
    "normalization_attempts": 0,
    "normalization_execution_result": "",
    "normalization_rework_required": False,
    "normalization_rework_reasons": [],
    "normalization_directives": {},
    "investigator_decision": "",
}
if validation_fusion_testset:
    invoke_payload["validation_fusion_testset"] = validation_fusion_testset

print("[*] Starting companies run...")
result = agent.graph.invoke(invoke_payload, config={"recursion_limit": GRAPH_RECURSION_LIMIT})
print("[*] Companies run complete")

out = {
    "investigator_decision": result.get("investigator_decision"),
    "overall_accuracy": (result.get("evaluation_metrics") or {}).get("overall_accuracy"),
    "macro_accuracy": (result.get("evaluation_metrics") or {}).get("macro_accuracy"),
    "sealed_test_overall": (result.get("sealed_test_metrics_final") or {}).get("overall_accuracy"),
    "normalization_attempts": result.get("normalization_attempts"),
    "evaluation_attempts": result.get("evaluation_attempts"),
    "run_id": result.get("run_id"),
    "run_report_path": result.get("run_report_path"),
}
print("[SUMMARY]", json.dumps(out, indent=2, default=str))
