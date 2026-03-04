import json
import os
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parent
os.chdir(str(AGENTS_DIR))

script_path = AGENTS_DIR / "AdaptationPipeline_nblazek_setcount_exec.py.py"
source = script_path.read_text(encoding="utf-8")
marker = "# Music Dataset"
if marker not in source:
    raise RuntimeError(f"Marker not found in {script_path}")

bootstrap = source.split(marker, 1)[0]
ns = {"__name__": "__main__"}
exec(bootstrap, ns)

INPUT_DIR = ns.get("INPUT_DIR", "input/")
ProfileDatasetTool = ns["ProfileDatasetTool"]
SearchDocumentationTool = ns["SearchDocumentationTool"]
SimpleModelAgent = ns["SimpleModelAgent"]
llm = ns["llm"]

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

ns["SKIP_BLOCKING_TESTER"] = False
ns["SKIP_MATCHING_TESTER"] = False

profile_tool = ProfileDatasetTool()
search_tool = SearchDocumentationTool()
all_tools = {profile_tool.name: profile_tool, search_tool.name: search_tool}
agent = SimpleModelAgent(llm, tools=all_tools)

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
result = agent.graph.invoke(invoke_payload, config={"recursion_limit": 200})
print("[*] Companies run complete")

out = {
    "investigator_decision": result.get("investigator_decision"),
    "overall_accuracy": (result.get("evaluation_metrics") or {}).get("overall_accuracy"),
    "macro_accuracy": (result.get("evaluation_metrics") or {}).get("macro_accuracy"),
    "normalization_attempts": result.get("normalization_attempts"),
    "evaluation_attempts": result.get("evaluation_attempts"),
    "pipeline_code_path": result.get("pipeline_code_path"),
    "evaluation_code_path": result.get("evaluation_code_path"),
}
print("[SUMMARY]", json.dumps(out, indent=2, default=str))
