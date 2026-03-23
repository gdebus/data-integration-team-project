#!/usr/bin/env python
# coding: utf-8

import sys
sys.dont_write_bytecode = True  # Prevent stale .pyc cache issues

import os
import json
import getpass
import logging
from pathlib import Path
from time import sleep
import subprocess
import sys
import re
import shutil
import pandas as pd
import time
from datetime import datetime, timezone

from _resolve_import import ensure_project_root
ensure_project_root()

from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool

from cluster_tester import ClusterTester
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import traceback

from typing import TypedDict, Dict, List, Any, Optional, Tuple

from PyDI.io import load_xml, load_parquet, load_csv
try:
    from PyDI.profiling import DataProfiler
except ImportError:
    from PyDI.utils.profiler import DataProfiler
from blocking_tester import BlockingTester
from matching_tester import MatchingTester
from PyDI.entitymatching import RuleBasedMatcher
from schema_matching_node import run_schema_matching
from fusion_size_monitor import compare_estimates_with_actual
from workflow_logging import attach_logging, configure_workflow_logger, log_workflow_action

import config
from config import (
    INPUT_DIR,
    OPENAI_REQUEST_TIMEOUT,
    OPENAI_MAX_RETRIES,
    LLM_REQUEST_TIMEOUT,
    LLM_INVOKE_MAX_ATTEMPTS,
    PIPELINE_TOOLCALL_LOOP_LIMIT,
    INCLUDE_DOCS,
    USE_LLM,
    QUALITY_GATE_THRESHOLD,
    MATCHING_F1_GATE,
    MATCHING_F1_THRESHOLD,
    BLOCKING_PC_THRESHOLD,
    HUMAN_REVIEW_EXEC_TIMEOUT,
    PIPELINE_EXEC_MAX_ATTEMPTS,
    MATCHING_MAX_ATTEMPTS,
    MATCHING_MAX_ERROR_RETRIES,
    BLOCKING_MAX_CANDIDATES,
    BLOCKING_MAX_ATTEMPTS,
    BLOCKING_MAX_ERROR_RETRIES,
    STDERR_MAX_LENGTH,
    OPENAI_RATE_CARD,
    truncate_stderr,
    GRAPH_RECURSION_LIMIT,
    SKIP_BLOCKING_TESTER,
    SKIP_MATCHING_TESTER,
    ERROR_RAW_SNIPPET_LENGTH,
    configure_run_output,
)

# ── Helper imports ────────────────────────────────────────────────────────────
from helpers.evaluation import (
    extract_json_object as helper_extract_json_object,
    extract_llm_text as helper_extract_llm_text,
    extract_python_code as helper_extract_python_code,
    compute_auto_diagnostics as helper_compute_auto_diagnostics,
    compute_id_alignment as helper_compute_id_alignment,
)
from helpers.metrics import (
    is_metrics_payload,
    extract_metrics_payload,
)
from helpers.correspondence import (
    expected_dataset_pairs,
    resolve_correspondence_file,
    csv_data_row_count,
    collect_latest_correspondence_files as helper_collect_latest_correspondence_files,
    summarize_correspondence_entries as helper_summarize_correspondence_entries,
)
from helpers.context_summarizer import (
    summarize_metrics_for_llm,
    summarize_diagnostics_for_llm,
    build_focused_pipeline_context,
    build_iteration_history_section,
    build_input_data_context,
    build_correspondence_summary,
)
from helpers.snapshots import (
    run_path,
    snapshot_file,
    snapshot_patterns,
    snapshot_pipeline_attempt,
    snapshot_evaluation_attempt,
)
from helpers.token_tracking import TokenTracker
from helpers.results_writer import save_results as _save_results
from helpers.pipeline_scaffold import (
    build_scaffold,
    build_patch_prompt_context,
    extract_mutable_from_response,
    assemble_pipeline,
    needs_new_imports,
    inject_imports,
)
from helpers.code_guardrails import (
    apply_pipeline_guardrails as helper_apply_pipeline_guardrails,
    apply_evaluation_guardrails as helper_apply_evaluation_guardrails,
    static_pipeline_sanity_findings,
)
from helpers.matching_validation import (
    config_has_list_based_comparators,
    config_matches_datasets,
    matching_config_needs_refresh,
)
from helpers.error_classifier import classify_execution_error
from helpers.normalization_orchestrator import run_normalization_node
from helpers.evaluation_decision import pipeline_problem_classes, process_evaluation_decision
from helpers.evaluation_orchestrator import run_evaluation_node
from helpers.investigator_orchestrator import run_investigator_node
from helpers.script_runner import run_pipeline_subprocess, run_evaluation_subprocess

# ── Prompt imports ────────────────────────────────────────────────────────────
from prompts.review_prompt import REVIEW_SYSTEM_PROMPT, REVISION_SYSTEM_PROMPT
from prompts.diagnostics_prompt import HUMAN_REVIEW_SYSTEM_PROMPT
from prompts.profile_prompt import PROFILE_SYSTEM_PROMPT

from prompts.evaluation_prompt import EVALUATION_ROBUSTNESS_RULES_BLOCK
from prompts.pipeline_prompt import PIPELINE_NORMALIZATION_RULES_BLOCK, PIPELINE_MATCHING_SAFETY_RULES_BLOCK, build_pipeline_system_prompt

from dotenv import load_dotenv
load_dotenv()


if USE_LLM == "gemini": # or USE_LLM == "gemini_broken":
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
elif USE_LLM == "groq":
    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
elif USE_LLM == "gpt":
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")




logging.basicConfig(filename=os.path.join(config.OUTPUT_DIR, 'agent.log'),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG,
                    encoding='utf-8')


# -- Utilities -----------------------------------------------------------------



def load_dataset(path):
    # Validates that the file exists before attempting to load.
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    ext = os.path.splitext(path)[1].lower()

    # Dispatches to the appropriate PyDI loader based on file extension.
    if ext == ".parquet":
        df = load_parquet(path)
    elif ext == ".csv":
        df = load_csv(path)
    elif ext == ".xml":
        df = load_xml(path, nested_handling="aggregate")
    else:
        raise ValueError(f"Unsupported format: {ext}. Supported: .csv, .parquet, .xml")
    return df





# -- Tools ---------------------------------------------------------------------



class ProfileDatasetTool(BaseTool):
    name: str = "profile_dataset"
    description: str = """
        A tool that takes the path as a argument called `path` of type str of the dataset file as string and performs data analysis. 
        A JSON string is returned with the profile data.
    """
    
    def _run(self, path) -> str:   
        return self.create_profile(path)

    def create_profile(self, path):

        if not path or not isinstance(path, str):
                raise ValueError("ProfileDatasetTool requires a single path string argument.")

        df = load_dataset(path)    
        
        if df is None or getattr(df, "empty", False):
            # Returns a JSON-encoded error so the caller can parse it uniformly.
            return json.dumps({"error": f"Dataset at {path} loaded as empty or failed to load."})

        profiler = DataProfiler()
        profile = profiler.summary(df, print_summary=False)

        # Serializes the profile dict to a JSON string for tool output compatibility.
        try:
            profile_json = json.dumps(profile, default=str)
        except Exception:
            # Fallback for non-JSON-serializable profiles: wraps as a string representation.
            profile_json = json.dumps({"profile_str": str(profile)})

        return profile_json
        




class SearchDocumentationTool(BaseTool):
    """Searches the PyDI documentation vector database for relevant API references."""
    name: str = "search_documentation"
    description: str = (
        "Searches the PyDI library documentation for a given query. "
        "Use this to find information about functions, classes, or how-to instructions. "
        "The input should be a specific question about the library."
    )
    call_count: int = 0

    def _clean_text(self, text: str) -> str:
        """Normalizes whitespace and line endings in LLM/tool output text."""

        if text is None:
            return ""

        text = str(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.strip()

        return text


    def _run(self, query: str) -> str:
        """Runs a similarity search against the PyDI documentation vector store."""
        self.call_count += 1
        print(f"[SEARCH TOOL CALL {self.call_count}]: Asking '{query}'")
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_db_path = os.path.join(INPUT_DIR, "api_documentation/pydi_apidocs_vector_db/")

        if not os.path.exists(vector_db_path):
            error_msg = f"Error: Vector DB not found at {vector_db_path}"
            return error_msg
        try:
            db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
            docs = db.similarity_search(query, k=8)
            
            if not docs:
                response = "No relevant documentation found for your query."
            else:
                response = "\n\n-----\n\n".join([r.page_content for r in docs])
                response = self._clean_text(response)
            
            return response
        except Exception as e:
            error_response = f"An error occurred while searching documentation: {str(e)}\n{traceback.format_exc()}"
            return error_response
    
    def reset(self):
        """Zeroes the call counter so a fresh agent run starts from 0."""
        self.call_count = 0


# -- Agents --------------------------------------------------------------------



class SimpleModelAgentState(TypedDict):
    datasets: list
    entity_matching_testsets: list
    fusion_testset: str
    validation_fusion_testset: str
    blocking_config: Dict  # Blocking strategy selected by BlockingTester
    matching_config: Dict  # Matching strategy selected by MatchingTester
    matcher_mode: str  # Matcher mode: rule_based, ml, or auto
    schema_correspondences: Dict  # Schema alignment results across datasets
    
    data_profiles: Dict

    messages: List[AnyMessage] # Conversation history accumulated during research
    eval_messages: List[AnyMessage] # Conversation history accumulated during evaluation

    cluster_analysis_result: Dict
    
    integration_pipeline_code: str
    pipeline_execution_result: str
    pipeline_execution_attempts: int

    evaluation_code: str
    evaluation_execution_result: str
    evaluation_execution_attempts: int
    integration_diagnostics_code: str
    integration_diagnostics_execution_result: str
    integration_diagnostics_report: Dict
    human_review_code: str
    human_review_execution_result: str
    human_review_report: Dict
    final_test_evaluation_execution_result: str
    final_test_evaluation_metrics: Dict

    evaluation_attempts: int
    evaluation_metrics: Dict
    best_validation_metrics: Dict
    best_pipeline_code: str
    best_evaluation_code: str
    latest_validation_metrics: Dict
    evaluation_regression_guard: Dict
    evaluation_cycle_audit: List[Dict[str, Any]]
    evaluation_analysis: str
    evaluation_reasoning_brief: Dict
    fusion_size_comparison: Dict
    auto_diagnostics: Dict
    correspondence_integrity: Dict
    normalization_execution_result: str
    normalization_attempts: int
    normalization_report: Dict
    normalized_datasets: List[str]
    original_datasets: List[str]
    normalization_rework_required: bool
    normalization_rework_reasons: List[str]
    normalization_directives: Dict
    investigator_action_plan: List[Dict[str, Any]]
    fusion_guidance: Dict
    pipeline_generation_review: Dict
    investigator_decision: str
    investigator_routing_decision: Dict
    validation_metrics_final: Dict
    sealed_test_metrics_final: Dict
    run_audit_path: str
    run_report_path: str
    pipeline_snapshots: List[Dict[str, Any]]
    evaluation_snapshots: List[Dict[str, Any]]
    run_id: str
    run_output_root: str
    pipeline_run_started_at: float
    pipeline_run_finished_at: float
    evaluation_metrics_raw: Dict
    evaluation_metrics_for_adaptation: Dict
    pipeline_scaffold: Dict  # Frozen scaffold for patch-mode pipeline adaptation

def _validate_dataset_config_signature(config: dict, state: dict) -> bool:
    """Returns True when the config's dataset_names match the state's datasets, or when the config carries no names."""
    cfg_names = config.get("dataset_names", []) if isinstance(config, dict) else []
    expected = sorted([Path(p).stem for p in state.get("datasets", [])])
    got = sorted([str(x) for x in cfg_names]) if isinstance(cfg_names, list) else []
    if got and got != expected:
        return False
    return True


class SimpleModelAgent:
    
    def __init__(self, model, tools: Dict[str, BaseTool]):
        self.logger = configure_workflow_logger(output_dir=config.OUTPUT_DIR)

        self._token_tracker = TokenTracker()
        self.tools = tools
        self.model = model.bind_tools(list(self.tools.values()))
        self.base_model = model
        self.run_id = ""
        self.run_output_root = ""
        self._reset_run_context()

        # Builds the LangGraph StateGraph with all pipeline nodes.
        graph = StateGraph(SimpleModelAgentState)

        # Node registration
        graph.add_node("match_schemas", self.match_schemas)
        graph.add_node("profile_data", self.profile_data)

        graph.add_node("normalization_node", lambda state: run_normalization_node(self, state, load_dataset))
        graph.add_node("run_blocking_tester", self.run_blocking_tester)
        graph.add_node("run_matching_tester", self.run_matching_tester)
        graph.add_node("pipeline_adaption", self.pipeline_adaption)
        graph.add_node("execute_pipeline", self.execute_pipeline)
        graph.add_node("evaluation_node", lambda state: run_evaluation_node(self, state))
        graph.add_node("investigator_node", lambda state: run_investigator_node(self, state))
        graph.add_node("human_review_export", self.human_review_export)
        graph.add_node("sealed_final_test_evaluation", self.sealed_final_test_evaluation)
        graph.add_node("save_results", self.save_results)

        # Edge wiring
        graph.add_edge("match_schemas", "profile_data")
        graph.add_edge("profile_data", "normalization_node")
        graph.add_edge("normalization_node", "run_blocking_tester")
        graph.add_edge("run_blocking_tester", "run_matching_tester")
        graph.add_edge("run_matching_tester", "pipeline_adaption")
        
        graph.add_conditional_edges(
            "pipeline_adaption",
            self.should_continue_research,
            {
                "continue": "pipeline_adaption",
                "end": "execute_pipeline"
            }
        )
        
        graph.add_conditional_edges(
            "execute_pipeline",
            self._route_after_execution,
            {
                "evaluation_node": "evaluation_node",
                "pipeline_adaption": "pipeline_adaption",
                END: END,
            },
        )

        # Evaluation + investigation routing

        graph.add_conditional_edges(
            "evaluation_node",
            self._route_after_evaluation,
            {
                "investigator_node": "investigator_node",
                "evaluation_node": "evaluation_node",
                END: END,
            }
        )

        graph.add_conditional_edges(
            "investigator_node",
            lambda state: state.get("investigator_decision", "human_review_export"),
            {
                "normalization_node": "normalization_node",
                "run_blocking_tester": "run_blocking_tester",
                "run_matching_tester": "run_matching_tester",
                "pipeline_adaption": "pipeline_adaption",
                "human_review_export": "human_review_export",
                END: END
            }
        )
        graph.add_conditional_edges(
            "human_review_export",
            self._route_after_human_review,
            {
                "sealed_final_test_evaluation": "sealed_final_test_evaluation",
                "save_results": "save_results",
                END: END,
            },
        )
        graph.add_edge("sealed_final_test_evaluation", "save_results")
        graph.add_edge("save_results", END)

        graph.set_entry_point("match_schemas")
        self.graph = graph.compile()
        try:
            attach_logging(
                self,
                output_dir=config.OUTPUT_DIR,
                notebook_name="pipeline_agent",
            )
        except Exception as e:
            print(f"[!] Workflow logging attachment failed: {e}")

    def _reset_run_context(self, state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # Derive usecase from dataset paths
        usecase = "unknown"
        datasets = (state or {}).get("datasets", [])
        if datasets:
            first = datasets[0] if isinstance(datasets, list) else ""
            parts = str(first).replace("\\", "/").split("/")
            for i, p in enumerate(parts):
                if p == "datasets" and i + 1 < len(parts):
                    usecase = parts[i + 1]
                    break

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{timestamp}_{usecase}"
        run_dir = os.path.join("output", "runs", self.run_id) + "/"

        # Redirect all output paths to this run's directory
        configure_run_output(run_dir)

        # Update workflow logger output dir so node_activity.json and
        # pipelines.md are written into the run-scoped directory.
        if hasattr(self, "workflow_logger") and self.workflow_logger is not None:
            wl = self.workflow_logger
            if hasattr(wl, "relocate_output"):
                wl.relocate_output(run_dir)
            else:
                wl.output_dir = run_dir

        self.run_output_root = run_dir

        # Create standard subdirectories
        for subdir in [
            "code", "data_fusion", "pipeline_evaluation", "correspondences",
            "human_review", "blocking-evaluation", "matching-evaluation",
            "cluster-evaluation", "normalization", "schema-matching",
            "profile", "investigation", "results",
            "pipeline", "evaluation", "snapshots",
        ]:
            os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)

        self._token_tracker.reset()

        # Set up per-run logging
        run_log_path = os.path.join(run_dir, "agent.log")
        for handler in list(logging.getLogger().handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logging.getLogger().removeHandler(handler)
        file_handler = logging.FileHandler(run_log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
        ))
        file_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(file_handler)

        return {
            "run_id": self.run_id,
            "run_output_root": self.run_output_root,
        }

    def prepare_for_new_run(self, state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self._reset_run_context(state)

    @staticmethod
    def _matching_refresh_gate(state: Dict[str, Any], min_pair_f1: float = MATCHING_F1_GATE) -> Dict[str, Any]:
        cfg = state.get("matching_config", {}) if isinstance(state, dict) else {}
        strategies = cfg.get("matching_strategies", {}) if isinstance(cfg, dict) else {}
        weak_pairs: List[Dict[str, Any]] = []
        if not isinstance(strategies, dict):
            return {"triggered": False, "weak_pairs": []}
        for pair_name, pair_cfg in strategies.items():
            if not isinstance(pair_cfg, dict):
                continue
            try:
                f1 = float(pair_cfg.get("f1", 0.0) or 0.0)
            except Exception:
                f1 = 0.0
            tags = {str(tag) for tag in pair_cfg.get("failure_tags", []) if str(tag).strip()}
            if f1 < min_pair_f1 or "low_matching_quality" in tags:
                weak_pairs.append(
                    {
                        "pair": str(pair_name),
                        "f1": round(f1, 6),
                        "failure_tags": sorted(tags),
                    }
                )
        return {"triggered": bool(weak_pairs), "weak_pairs": weak_pairs, "threshold": min_pair_f1}

    # -- Token tracking (delegates to helpers.token_tracking.TokenTracker) ------

    def _invoke_model_with_usage(self, model, message, tag):
        result = self._token_tracker.invoke_model_with_usage(model, message, tag, base_model=self.base_model)
        # Sync cost to agent.token_usage so required_logging can read it
        if hasattr(self, "token_usage") and isinstance(self.token_usage, dict):
            self.token_usage.update(self._token_tracker.usage)
        return result

    def _invoke_with_usage(self, message, tag):
        return self._invoke_model_with_usage(self.model, message, tag)

    def _invoke_base_with_usage(self, message, tag):
        return self._invoke_model_with_usage(self.base_model, message, tag)

    def _print_total_usage(self):
        self._token_tracker.print_total_usage()

    def _log_action(self, step: str, action: str, why: str, improvement: str = "", details: dict | None = None):
        log_workflow_action(
            self.logger,
            step=step,
            action=action,
            why=why,
            improvement=improvement,
            details=details,
        )

    def _run_path(self, *parts: str) -> str:
        return run_path(self.run_output_root, *parts)

    def _snapshot_file(self, src_path: str, dst_rel_path: str) -> str:
        return snapshot_file(self.run_output_root, src_path, dst_rel_path)

    def _snapshot_patterns(self, patterns: List[str], dst_rel_dir: str) -> List[str]:
        return snapshot_patterns(self.run_output_root, patterns, dst_rel_dir)

    def _snapshot_pipeline_attempt(self, **kwargs) -> Dict[str, Any]:
        return snapshot_pipeline_attempt(self.run_output_root, **kwargs)

    def _snapshot_evaluation_attempt(self, **kwargs) -> Dict[str, Any]:
        return snapshot_evaluation_attempt(self.run_output_root, **kwargs)

    def _route_after_execution(self, state: SimpleModelAgentState) -> str:
        """Routes to evaluation on success, back to adaption on failure, or terminates when attempts are exhausted."""
        if state.get("pipeline_execution_result", "").lower().startswith("success"):
            return "evaluation_node"
        if state.get("pipeline_execution_attempts", 0) < PIPELINE_EXEC_MAX_ATTEMPTS:
            return "pipeline_adaption"
        return END

    def _route_after_evaluation(self, state: SimpleModelAgentState) -> str:
        """Routes to investigator on success, retries evaluation on recoverable failure, terminates on exhaustion."""
        result = state.get("evaluation_execution_result", "")
        if isinstance(result, str) and result.lower().startswith("success"):
            return "investigator_node"
        # Allow one graph-level retry of the full evaluation node before giving up
        eval_node_retries = int(state.get("_eval_node_graph_retries", 0))
        error_info = state.get("evaluation_error_classification", {})
        retryable = error_info.get("retryable", True) if isinstance(error_info, dict) else True
        if retryable and eval_node_retries < 1:
            print(f"[EVAL] Graph-level retry ({eval_node_retries + 1}/1) — resetting evaluation state")
            return "evaluation_node"
        return END

    def _route_after_human_review(self, state: SimpleModelAgentState) -> str:
        """Routes to sealed final-test evaluation when active, otherwise proceeds to save_results."""
        if self._sealed_eval_active(state):
            return "sealed_final_test_evaluation"
        return "save_results"

    def _sealed_eval_active(self, state: SimpleModelAgentState) -> bool:
        validation_path = state.get("validation_fusion_testset")
        test_path = state.get("fusion_testset")
        return bool(
            validation_path
            and test_path
            and os.path.exists(validation_path)
            and os.path.exists(test_path)
        )

    def _evaluation_testset_path(self, state: SimpleModelAgentState, force_test: bool = False) -> str:
        if force_test:
            return state.get("fusion_testset", "")
        if self._sealed_eval_active(state):
            return state.get("validation_fusion_testset", "")
        return state.get("fusion_testset", "")

    def _evaluation_stage_label(self, state: SimpleModelAgentState, force_test: bool = False) -> str:
        if force_test:
            return "sealed_test"
        return "validation" if self._sealed_eval_active(state) else "test"

    def _update_pipeline_matching_thresholds(self,
        pipeline_code: str, matching_config: Dict[str, Any]
    ) -> str:
        """Patches threshold variable assignments in generated pipeline code to match the MatchingTester-selected values."""
        if not matching_config or "matching_strategies" not in matching_config:
            return pipeline_code

        updated_code = pipeline_code
        strategies = matching_config["matching_strategies"]

        for pair_key, config in strategies.items():
            if isinstance(config, dict) and "threshold" in config:
                threshold_value = config["threshold"]

                # Variable naming convention: threshold_<pair_key> (e.g. threshold_discogs_lastfm).
                var_name = f"threshold_{pair_key}"

                # Regex matches assignments like `threshold_discogs_lastfm = 0.7`,
                # tolerating whitespace around `=` and various float formats.
                pattern_var_assignment = rf"({re.escape(var_name)}\s*=\s*)([0-9]*\.?[0-9]+)"

                if re.search(pattern_var_assignment, updated_code):
                    updated_code = re.sub(
                        pattern_var_assignment,
                        rf"\g<1>{threshold_value}",
                        updated_code,
                        count=1,
                    )
                    print(
                        f"[DEBUG] Updated variable assignment for {pair_key} to {threshold_value}"
                    )
                else:
                    print(
                        f"[DEBUG] Could not find explicit variable assignment for {var_name}. "
                        f"Relying on LLM to generate it correctly. Current code not modified for this pair."
                    )
                    pass

        return updated_code

    def _apply_pipeline_guardrails(self, pipeline_code: str, state: SimpleModelAgentState) -> str:
        """Applies dataset-agnostic guardrails to generated pipeline code."""
        return helper_apply_pipeline_guardrails(pipeline_code, state if isinstance(state, dict) else {})

    def _compute_auto_diagnostics(self, state: SimpleModelAgentState, metrics: Dict[str, Any]) -> Dict[str, Any]:
        force_test_eval = bool(state.get("_force_test_eval"))
        eval_testset_path = self._evaluation_testset_path(state, force_test=force_test_eval)
        eval_stage_label = self._evaluation_stage_label(state, force_test=force_test_eval)
        return helper_compute_auto_diagnostics(
            state=state,
            metrics=metrics,
            evaluation_testset_path=eval_testset_path,
            evaluation_stage_label=eval_stage_label,
            load_dataset_fn=load_dataset,
            compute_id_alignment_fn=helper_compute_id_alignment,
            debug_path=os.path.join(config.OUTPUT_DIR, "pipeline_evaluation/debug_fusion_eval.jsonl"),
            fused_path=os.path.join(config.OUTPUT_DIR, "data_fusion/fusion_data.csv"),
        )

    def _pipeline_problem_classes(self, state: SimpleModelAgentState) -> List[str]:
        return pipeline_problem_classes(state if isinstance(state, dict) else {})

    def _review_pipeline_candidate(self, state: SimpleModelAgentState, pipeline_code: str) -> Dict[str, Any]:
        static_findings = static_pipeline_sanity_findings(pipeline_code, correspondences_dir=config.CORRESPONDENCES_DIR)
        review_system_prompt = REVIEW_SYSTEM_PROMPT

        payload = {
            "problem_classes": self._pipeline_problem_classes(state),
            "pipeline_execution_result": state.get("pipeline_execution_result", ""),
            "evaluation_metrics": state.get("evaluation_metrics_for_adaptation", state.get("evaluation_metrics", {})),
            "auto_diagnostics": state.get("auto_diagnostics", {}),
            "evaluation_regression_guard": state.get("evaluation_regression_guard", {}),
            "investigator_action_plan": state.get("investigator_action_plan", []),
            "fusion_guidance": state.get("fusion_guidance", {}),
            "evaluation_reasoning_brief": state.get("evaluation_reasoning_brief", {}),
            "static_sanity_findings": static_findings,
        }

        message = [
            SystemMessage(content=review_system_prompt),
            HumanMessage(
                content=(
                    "Review this generated pipeline candidate.\n\n"
                    f"EVIDENCE\n{json.dumps(payload, indent=2, default=str)}\n\n"
                    f"PIPELINE_CODE\n```python\n{pipeline_code}\n```"
                )
            ),
        ]
        result = self._invoke_base_with_usage(message, "pipeline_pre_execution_review")
        parsed = helper_extract_json_object(result)
        if not isinstance(parsed, dict):
            return {
                "verdict": "pass",
                "summary": "",
                "problem_classes": payload["problem_classes"],
                "keep": [],
                "risks": [],
                "revision_instructions": [],
            }
        verdict = str(parsed.get("verdict", "pass")).strip().lower()
        if verdict not in {"pass", "revise"}:
            verdict = "pass"
        return {
            "verdict": verdict,
            "summary": str(parsed.get("summary", "") or "").strip(),
            "problem_classes": [str(x) for x in parsed.get("problem_classes", []) if str(x).strip()],
            "keep": [str(x) for x in parsed.get("keep", []) if str(x).strip()],
            "risks": [str(x) for x in parsed.get("risks", []) if str(x).strip()],
            "revision_instructions": [
                str(x) for x in parsed.get("revision_instructions", []) if str(x).strip()
            ],
            "static_sanity_findings": static_findings,
        }

    def _revise_pipeline_from_review(
        self,
        state: SimpleModelAgentState,
        pipeline_code: str,
        review: Dict[str, Any],
    ) -> str:
        revision_system_prompt = REVISION_SYSTEM_PROMPT

        evidence = {
            "problem_classes": self._pipeline_problem_classes(state),
            "review": review,
            "auto_diagnostics": state.get("auto_diagnostics", {}),
            "evaluation_metrics": state.get("evaluation_metrics_for_adaptation", state.get("evaluation_metrics", {})),
            "evaluation_regression_guard": state.get("evaluation_regression_guard", {}),
        }

        message = [
            SystemMessage(content=revision_system_prompt),
            HumanMessage(
                content=(
                    "Revise the pipeline candidate according to the review below.\n\n"
                    f"REVIEW_EVIDENCE\n{json.dumps(evidence, indent=2, default=str)}\n\n"
                    "Make the smallest possible edit set. Do not refactor unrelated sections.\n\n"
                    f"CURRENT_PIPELINE\n```python\n{pipeline_code}\n```"
                )
            ),
        ]
        result = self._invoke_base_with_usage(message, "pipeline_revision")
        revised = helper_extract_llm_text(result)
        if revised.startswith("```python") and revised.endswith("```"):
            revised = revised.strip("```python").strip("```").strip()
        return str(revised or "").strip() or pipeline_code

    def _apply_evaluation_guardrails(self, evaluation_code: str) -> str:
        """Patches common portability and runtime issues in generated evaluation code."""
        return helper_apply_evaluation_guardrails(evaluation_code)

    def should_continue_research(self, state: SimpleModelAgentState) -> str:
        """Decides whether the pipeline_adaption research loop should continue or hand off to execution."""
        messages = state['messages']
        last_message = messages[-1]
        # Continues the loop when the last message is a tool call.
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            # Caps consecutive tool-call rounds to prevent infinite loops.
            trailing_tool_call_rounds = 0
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage):
                    continue
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    trailing_tool_call_rounds += 1
                    continue
                break
            if trailing_tool_call_rounds >= PIPELINE_TOOLCALL_LOOP_LIMIT:
                print(
                    "[GUARDRAIL] pipeline_adaption tool-call loop limit reached "
                    f"({trailing_tool_call_rounds}/{PIPELINE_TOOLCALL_LOOP_LIMIT}); forcing execute_pipeline."
                )
                return "end"
            return "continue"
        # No tool call means the model produced final code; end the loop.
        else:
            return "end"

    # Aligns dataset schemas before blocking/matching, storing results in agent state.
    def match_schemas(self, state: SimpleModelAgentState):
        if state.get("run_id") == self.run_id and state.get("run_output_root") == self.run_output_root:
            run_updates = {
                "run_id": self.run_id,
                "run_output_root": self.run_output_root,
            }
        else:
            run_updates = self._reset_run_context(state)
        self._log_action("match_schemas", "start", "Align dataset schemas before blocking/matching", "Improves comparator compatibility", {"datasets": state.get("datasets")})

        self.logger.info("----------------------- Entering match_schemas -----------------------")
        if state.get("schema_correspondences"):
            print("[SCHEMA] Already present — skipping")
            return {}

        print("[SCHEMA] Aligning dataset schemas...")
        result = run_schema_matching(
            dataset_paths=state["datasets"],
            model=self.base_model,
            output_dir=os.path.join(config.OUTPUT_DIR, "schema-matching"),
        )
        self.logger.info("Leaving match_schemas")
        out = dict(result) if isinstance(result, dict) else {}
        out.update(run_updates)
        return out

    def profile_data(self, state:SimpleModelAgentState):
        self._log_action("profile_data", "start", "Profile datasets to guide feature/threshold choices", "Improves matching signal selection", {"datasets": state.get("datasets")})

        self.logger.info('----------------------- Entering profile_data -----------------------')

        print("[PROFILE] Profiling datasets...")

        system_prompt = PROFILE_SYSTEM_PROMPT
        
        datasets_list_str = "\n".join(state['datasets'])
        human_content = f"Please profile these datasets (one call per dataset):\n{datasets_list_str}"
        message = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
        self.logger.info("Input Message:" + str(message))
        
        result = self._invoke_with_usage(message, "profile_data")
        self.logger.info("RESULT:" + str(result))

        # Dispatches tool calls requested by the model.
        tool_calls = result.tool_calls

        self.logger.info("Tool Calls:" + str(tool_calls))
        results = {}
        for t in tool_calls:
            if not t['name'] in self.tools:      # Unrecognized tool name from model
                self.logger.info("adapt_pipeline: ....bad tool name....")
                result = "bad tool name, retry"  # Signals the model to retry with a valid name
            else:
                result = self.tools[t['name']].invoke(t['args'])
            
#            if USE_LLM == "groq" or USE_LLM == "gpt" or USE_LLM == "gemini":
            results[t['args']['path']] = result
#            elif USE_LLM == "gemini_broken":
#                results[t['args']['__arg1']] = result

        with open(os.path.join(config.OUTPUT_DIR, "profile", "profiles.json"), "w") as file:
            file.write(json.dumps(results, indent=2))
          
        self.logger.info('Leaving profile_data')
        return {'data_profiles': results}

    def run_blocking_tester(self, state: SimpleModelAgentState):
        self._log_action("run_blocking_tester", "start", "Select blocking strategy to reduce candidates", "Improves runtime and recall balance", {"skip": SKIP_BLOCKING_TESTER})

        self.logger.info("----------------------- Entering run_blocking_tester -----------------------")

        if state.get("blocking_config"):
            cfg = state.get("blocking_config", {})
            if not _validate_dataset_config_signature(cfg, state):
                print("[BLOCKING] Config signature mismatch — recomputing")
            else:
                print("[BLOCKING] Using existing config — skipping BlockingTester")
                return {"blocking_config": cfg}

        print("[BLOCKING] Running BlockingTester...")
        tester = BlockingTester(
            llm=self.base_model,
            datasets=state["datasets"],
            max_candidates=BLOCKING_MAX_CANDIDATES,
            blocking_testsets=state['entity_matching_testsets'],
            output_dir=os.path.join(config.OUTPUT_DIR, "blocking-evaluation"),
            pc_threshold=BLOCKING_PC_THRESHOLD,
            max_attempts=BLOCKING_MAX_ATTEMPTS,
            max_error_retries=BLOCKING_MAX_ERROR_RETRIES,
            verbose=True
        )
        _, blocking_config = tester.run_all_pairs()
        strategies = blocking_config.get("blocking_strategies", {})
        summary = ", ".join(
            f"{pair}: {cfg.get('strategy','?')} (PC={cfg.get('pair_completeness',0):.1%}, {cfg.get('num_candidates',0)} cands)"
            for pair, cfg in strategies.items() if isinstance(cfg, dict)
        )
        print(f"[BLOCKING] Done — {summary or 'no strategies'}")
        self.logger.info("Leaving run_blocking_tester")
        return {"blocking_config": blocking_config}

    def run_matching_tester(self, state: SimpleModelAgentState):
        self._log_action("run_matching_tester", "start", "Select matching strategy and thresholds", "Improves correspondence quality", {"skip": SKIP_MATCHING_TESTER, "matcher_mode": state.get("matcher_mode")})

        self.logger.info("----------------------- Entering run_matching_tester -----------------------")
        matcher_mode = str(state.get("matcher_mode", "ml")).strip().lower().replace("-", "_")
        if matcher_mode == "rulebased":
            matcher_mode = "rule_based"
        state["matcher_mode"] = matcher_mode

        if state.get("matching_config"):
            existing_cfg = state.get("matching_config", {})
            if not config_matches_datasets(existing_cfg, state.get("datasets", [])):
                print("[MATCHING] Config signature mismatch — recomputing")
            elif config_has_list_based_comparators(existing_cfg):
                # List-based comparators are valid when pre-loaded — only recompute
                # if the config also has low F1 (suggesting it needs improvement)
                needs_refresh, refresh_reason = matching_config_needs_refresh(existing_cfg)
                if needs_refresh:
                    print(f"[MATCHING] Config has list-based comparators and needs refresh ({refresh_reason}) — recomputing")
                else:
                    print("[MATCHING] Using existing config (list-based comparators OK) — skipping MatchingTester")
                    return {"matching_config": existing_cfg, "matcher_mode": matcher_mode}
            else:
                needs_refresh, refresh_reason = matching_config_needs_refresh(existing_cfg)
                if needs_refresh:
                    print(f"[MATCHING] Config needs refresh ({refresh_reason}) — recomputing")
                else:
                    print("[MATCHING] Using existing config — skipping MatchingTester")
                    return {"matching_config": existing_cfg, "matcher_mode": matcher_mode}

        # Hot-reloads matching_tester from disk so notebook sessions pick up code changes.
        try:
            import importlib
            import matching_tester as _matching_tester_module
            _matching_tester_module = importlib.reload(_matching_tester_module)
            RuntimeMatchingTester = _matching_tester_module.MatchingTester
        except Exception as e:
            print(f"[MATCHING] Could not hot-reload module: {e}")
            RuntimeMatchingTester = MatchingTester

        print(f"[MATCHING] Running MatchingTester (mode={matcher_mode})...")
        tester = RuntimeMatchingTester(
            llm=self.base_model,
            datasets=state["datasets"],
            matching_testsets=state['entity_matching_testsets'],
            blocking_config=state.get("blocking_config", {}),
            output_dir=os.path.join(config.OUTPUT_DIR, "matching-evaluation"),
            f1_threshold=MATCHING_F1_THRESHOLD,
            max_attempts=MATCHING_MAX_ATTEMPTS,
            max_error_retries=MATCHING_MAX_ERROR_RETRIES,
            verbose=True,
            matcher_mode=matcher_mode,
            disallow_list_comparators=True,
            no_gain_patience=4,
            min_blocking_pc_for_matching=0.5,
        )
        _, matching_config = tester.run_all()
        strategies = matching_config.get("matching_strategies", {})
        summary = ", ".join(
            f"{pair}: F1={cfg.get('f1',0):.3f} thr={cfg.get('threshold',0)}"
            for pair, cfg in strategies.items() if isinstance(cfg, dict)
        )
        print(f"[MATCHING] Done — {summary or 'no strategies'}")
        self.logger.info("Leaving run_matching_tester")
        return {"matching_config": matching_config, "matcher_mode": matcher_mode}

    # -- Reusable preparation helpers for pipeline / evaluation adaption ---------

    def _load_dataset_previews(self, datasets: List[str]) -> Dict[str, Any]:
        """Returns a dict mapping each dataset path to its first-row preview (or an error message)."""
        previews = {}
        for path in datasets:
            df = load_dataset(path)
            if df is not None and not df.empty:
                previews[path] = df.iloc[0].to_dict()
            else:
                previews[path] = "Failed to load or empty dataset"
        return previews

    def _load_example_pipeline(self, matcher_mode: str) -> str:
        """Reads the example pipeline template that corresponds to the active matcher mode (ml vs rule_based)."""
        example_name = "example_pipeline_ml.py" if matcher_mode == "ml" else "example_pipeline.py"
        example_path = os.path.join(INPUT_DIR, f"example_pipelines/{example_name}")
        if not os.path.exists(example_path):
            raise FileNotFoundError(f"Example pipeline not found at: {example_path}")
        with open(example_path, "r", encoding="utf-8") as f:
            return f.read()

    def pipeline_adaption(self, state: SimpleModelAgentState):
        self._log_action("pipeline_adaption", "start", "Generate integration pipeline code", "Incorporates configs and prior feedback")

        self.logger.info('----------------------- Entering pipeline_adaption -----------------------')

        # -- System prompt assembly --------------------------------------------------

        # Loads the example pipeline template matching the active matcher mode.
        matcher_mode = str(state.get("matcher_mode", "ml")).strip().lower().replace("-", "_")
        if matcher_mode == "rulebased":
            matcher_mode = "rule_based"
        state["matcher_mode"] = matcher_mode
        example_pipeline_code = self._load_example_pipeline(matcher_mode)

        # First-row previews give the model concrete column/value examples.
        dataset_previews = self._load_dataset_previews(state['datasets'])

        # Builds the entity-matching section (ML mode supplies testset paths).
        entity_matching_section = ""

        if matcher_mode == "ml":
            entity_matching_section = f"""
            2b. Entity matching testsets paths:
            {state["entity_matching_testsets"]}
            """
        
        # Normalization and config context for the system prompt.
        normalization_context = {
            "normalization_execution_result": state.get("normalization_execution_result", ""),
            "normalization_attempts": state.get("normalization_attempts", 0),
            "normalization_directives": state.get("normalization_directives", {}),
            "normalization_report": state.get("normalization_report", {}),
        }
        blocking_config = state.get('blocking_config')
        matching_config = state.get('matching_config')
        expected_pairs = [f"{left}_{right}" for left, right in expected_dataset_pairs(state)] if blocking_config else []

        # Loads the cluster-aware example pipeline when cluster analysis results exist.
        cluster_analysis = state.get("cluster_analysis_result")
        cluster_example_pipeline_code = None
        if cluster_analysis:
            cluster_example_name = "example_pipeline_ml_cluster.py" if matcher_mode == "ml" else "example_pipeline_cluster.py"
            cluster_example_pipeline_path = os.path.join(INPUT_DIR, f"example_pipelines/{cluster_example_name}")

            if not os.path.exists(cluster_example_pipeline_path):
                raise FileNotFoundError(f"Cluster example pipeline not found at: {cluster_example_pipeline_path}")

            with open(cluster_example_pipeline_path, "r", encoding="utf-8") as f:
                cluster_example_pipeline_code = f.read()

        # Build supplementary context for adaptation iterations
        focused_ctx = None
        iter_history_text = None
        input_ctx = None
        has_evaluation_feedback_early = is_metrics_payload(
            state.get("evaluation_metrics_for_adaptation", state.get("evaluation_metrics", {}))
        )
        if has_evaluation_feedback_early:
            focused_ctx = build_focused_pipeline_context(state)
            iter_history_text = build_iteration_history_section(state)
            input_ctx = build_input_data_context(state)
        corr_summary = build_correspondence_summary(state) if has_evaluation_feedback_early else None

        system_prompt = build_pipeline_system_prompt(
            example_pipeline_code=example_pipeline_code,
            datasets=state['datasets'],
            entity_matching_section=entity_matching_section,
            dataset_previews=dataset_previews,
            data_profiles=state['data_profiles'],
            normalization_context=normalization_context,
            blocking_config=blocking_config,
            matching_config=matching_config,
            matcher_mode=matcher_mode,
            expected_pairs=expected_pairs,
            evaluation_analysis=state.get("evaluation_analysis", None),
            reasoning_brief=state.get("evaluation_reasoning_brief", {}),
            auto_diagnostics=state.get("auto_diagnostics"),
            investigator_action_plan=state.get("investigator_action_plan"),
            fusion_guidance=state.get("fusion_guidance"),
            cluster_analysis=cluster_analysis,
            cluster_example_pipeline_code=cluster_example_pipeline_code,
            correspondences_dir=config.CORRESPONDENCES_DIR,
            output_dir=config.OUTPUT_DIR,
            focused_context=focused_ctx,
            iteration_history=iter_history_text,
            input_data_context=input_ctx,
            correspondence_summary=corr_summary,
        )

        broken_pipeline_path = config.PIPELINE_CODE_PATH
        has_existing_pipeline = os.path.exists(broken_pipeline_path) and os.path.getsize(broken_pipeline_path) > 0
        has_execution_feedback = bool(str(state.get("pipeline_execution_result", "") or "").strip())
        has_evaluation_feedback = is_metrics_payload(
            state.get("evaluation_metrics_for_adaptation", state.get("evaluation_metrics", {}))
        )

        # --- Scaffold+Patch mode: on cycle 2+ with evaluation feedback and a
        # working pipeline, freeze the infrastructure and only patch the mutable
        # sections (fusion strategy, post-clustering).
        pipeline_scaffold = state.get("pipeline_scaffold")
        use_patch_mode = False
        is_execution_error = (
            "pipeline_execution_result" in state
            and str(state.get("pipeline_execution_result", "") or "").lower().startswith("error")
        )

        if has_existing_pipeline and has_evaluation_feedback and not is_execution_error:
            # Try to build/reuse scaffold from the working pipeline
            if not pipeline_scaffold:
                with open(broken_pipeline_path, "r", encoding="utf-8", errors="ignore") as f:
                    existing_code = f.read()
                pipeline_scaffold = build_scaffold(existing_code)
                if pipeline_scaffold:
                    print("[PIPELINE] Built scaffold from working pipeline — freezing infrastructure")

            if pipeline_scaffold:
                use_patch_mode = True

        # Distinguishes initial generation from error-fix or evaluation-driven adaption.
        if not has_existing_pipeline or (not has_execution_feedback and not has_evaluation_feedback):
            print("[PIPELINE] Generating initial pipeline...")
            human_content = "Create the integration pipeline for the provided datasets."

        elif use_patch_mode:
            # Patch mode: show frozen pipeline as context, ask LLM to output only mutable sections
            eval_metrics = state.get("evaluation_metrics", {})
            overall = eval_metrics.get("overall_accuracy", "?") if isinstance(eval_metrics, dict) else "?"
            print(f"[PIPELINE] Patch mode — modifying only fusion/clustering (current accuracy={overall})...")

            # Build per-attribute accuracy summary with protection markers
            attr_summary_lines = []
            if isinstance(eval_metrics, dict):
                for key, val in sorted(eval_metrics.items()):
                    if key.endswith("_accuracy") and key not in ("overall_accuracy", "macro_accuracy"):
                        attr_name = key.replace("_accuracy", "")
                        if isinstance(val, (int, float)):
                            status = "PROTECTED — do NOT change" if val >= 0.8 else ("needs improvement" if val < 0.5 else "moderate")
                            attr_summary_lines.append(f"  {attr_name}: {val:.1%} — {status}")

            attr_summary = "\n".join(attr_summary_lines) if attr_summary_lines else ""

            patch_context = build_patch_prompt_context(pipeline_scaffold)
            human_content = (
                "You previously generated a working integration pipeline.\n"
                "The INFRASTRUCTURE (imports, dataset loading, blocking, matching, "
                "correspondence saving, fusion execution) is FROZEN and must NOT change.\n\n"
                f"Evaluation of the last run:\n"
                f"{json.dumps(eval_metrics, indent=2)}\n\n"
            )
            if attr_summary:
                human_content += (
                    f"Per-attribute status:\n{attr_summary}\n\n"
                    "CRITICAL: Do NOT change the fusion resolver for PROTECTED attributes.\n"
                    "Only modify resolvers for attributes that need improvement.\n\n"
                )
            human_content += (
                "Your task: output ONLY the replacement code for the MUTABLE sections:\n"
                "- The fusion strategy (DataFusionStrategy + add_attribute_fuser calls)\n"
                "- Post-clustering (optional — add/modify/remove clustering if needed)\n"
                "- Trust map definition (if used)\n\n"
                "RULES:\n"
                "- Keep the SAME resolver for any attribute with >80% accuracy\n"
                "- Only change resolvers for low-accuracy attributes\n"
                "- Do NOT add inline normalization code — infrastructure is frozen\n"
                "- Do NOT output imports, blocking, matching, or fusion engine code\n\n"
                f"{patch_context}\n\n"
                "Output ONLY the replacement mutable section code."
            )

        else:
            # Full regeneration: execution error or first adaptation without scaffold
            with open(broken_pipeline_path, "r", encoding="utf-8", errors="ignore") as f:
                broken_code = f.read()

            human_content = "You previously generated Python integration pipeline code.\n"

            # Appends execution error details when the last run failed.
            if is_execution_error:
                print("[PIPELINE] Fixing execution error...")
                human_content += f"Executing this pipeline caused the following error:\n{state['pipeline_execution_result']}\n"

            # Appends evaluation metrics when available so the model can target weak attributes.
            if "evaluation_metrics" in state:
                overall = state["evaluation_metrics"].get("overall_accuracy", "?") if isinstance(state["evaluation_metrics"], dict) else "?"
                print(f"[PIPELINE] Improving based on evaluation (current accuracy={overall})...")
                human_content += f"Evaluation of the last run shows the following metrics:\n{json.dumps(state['evaluation_metrics'], indent=2)}\n"
                human_content += "Improve the pipeline to increase overall accuracy and correct errors highlighted by the evaluation.\n"

            human_content += "Here is the current pipeline code:\n" + broken_code
            human_content += "\nOutput ONLY the corrected Python code."

        # -- Documentation tool reset --
        if "search_documentation" in self.tools:
            self.tools["search_documentation"].reset()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]
        state["messages"] = messages
        pipeline_review: Dict[str, Any] = {}
    
        self.logger.info("Input Message:" + str(messages))
    
        # Invokes the model to generate (or fix) pipeline code.
        adapted_pipeline = self._invoke_with_usage(messages, "pipeline_adaption")
        messages.append(adapted_pipeline)

        if adapted_pipeline.tool_calls:
            tool_messages = []
            for tool_call in adapted_pipeline.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id")
                try:
                    if tool_name in self.tools:
                        tool_output = self.tools[tool_name].invoke(tool_args)
                    else:
                        tool_output = (
                            f"Tool '{tool_name}' not available. "
                            f"Available tools: {list(self.tools.keys())}"
                        )
                except Exception as tool_err:
                    tool_output = f"Tool '{tool_name}' execution error: {tool_err}"
                if tool_id:
                    tool_messages.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_id)
                    )
            messages.extend(tool_messages)
            
            # Re-invokes the model after tool results so it can produce final code.
            final_response = self._invoke_with_usage(messages, "pipeline_adaption")
            messages.append(final_response)
            adapted_pipeline = final_response

        # Extracts code from the model's final response.
        if hasattr(adapted_pipeline, 'tool_calls') and adapted_pipeline.tool_calls:
            # Gives up if the model still requests tools after the second invocation.
            adapted_pipeline = "Pipeline code not available - too many tool calls"
        else:
            adapted_pipeline = helper_extract_llm_text(adapted_pipeline)
            self.logger.info("RESULT:" + str(adapted_pipeline))
            if adapted_pipeline.startswith("```python") and adapted_pipeline.endswith("```"):
                adapted_pipeline = adapted_pipeline.strip("```python").strip("```").strip()

            # --- Patch mode: splice LLM's mutable output into the frozen scaffold
            if use_patch_mode and pipeline_scaffold and not adapted_pipeline.startswith("Pipeline code not available"):
                mutable_code = extract_mutable_from_response(str(adapted_pipeline), pipeline_scaffold)
                # Check if the new mutable code needs imports not in the frozen prefix
                missing = needs_new_imports(mutable_code, pipeline_scaffold["frozen_prefix"])
                prefix = pipeline_scaffold["frozen_prefix"]
                if missing:
                    prefix = inject_imports(prefix, missing)
                    print(f"[SCAFFOLD] Injected {len(missing)} missing import(s)")
                scaffold_with_imports = {**pipeline_scaffold, "frozen_prefix": prefix}
                adapted_pipeline = assemble_pipeline(scaffold_with_imports, mutable_code)
                print(f"[SCAFFOLD] Assembled pipeline: frozen scaffold + patched mutable section")

            if not adapted_pipeline.startswith("Pipeline code not available"):
                adapted_pipeline = self._apply_pipeline_guardrails(str(adapted_pipeline), state)
                pre_review_findings = static_pipeline_sanity_findings(str(adapted_pipeline), correspondences_dir=config.CORRESPONDENCES_DIR)
                if pre_review_findings:
                    # Only run self-review when the pipeline has actual static issues to fix
                    pipeline_review = self._review_pipeline_candidate(state, adapted_pipeline)
                    if pipeline_review.get("verdict") == "revise":
                        print("[PIPELINE] Self-review requested revision — applying targeted fix")
                        original_pipeline_candidate = str(adapted_pipeline)
                        revised_pipeline = self._revise_pipeline_from_review(state, adapted_pipeline, pipeline_review)
                        revised_pipeline = self._apply_pipeline_guardrails(str(revised_pipeline), state)
                        revised_static_findings = static_pipeline_sanity_findings(str(revised_pipeline), correspondences_dir=config.CORRESPONDENCES_DIR)
                        if revised_static_findings and not pre_review_findings:
                            print("[PIPELINE] Revision rejected — introduced new sanity issues; keeping original")
                            pipeline_review["revision_rejected"] = True
                            pipeline_review["revision_rejection_reason"] = "new_static_sanity_findings"
                            pipeline_review["revision_static_sanity_findings"] = revised_static_findings
                            adapted_pipeline = original_pipeline_candidate
                        else:
                            adapted_pipeline = revised_pipeline
                else:
                    print("[PIPELINE] No static issues found — skipping self-review")
                    pipeline_review = {"verdict": "pass", "summary": "no_static_findings"}
                os.makedirs(config.CODE_DIR, exist_ok=True)
                with open(config.PIPELINE_CODE_PATH, 'w', errors="ignore") as file:
                    file.write(str(adapted_pipeline))

        self.logger.info('Leaving pipeline_adaption')
        return {
            "messages": messages,
            "integration_pipeline_code": adapted_pipeline,
            "pipeline_generation_review": pipeline_review,
            "pipeline_scaffold": pipeline_scaffold,
        }

    def execute_pipeline(self, state: SimpleModelAgentState):
        self._log_action("execute_pipeline", "start", "Run generated pipeline", "Produces correspondences for analysis", {"attempt": state.get("pipeline_execution_attempts", 0)+1})
        self.logger.info('----------------------- Entering execute_pipeline -----------------------')
        attempt = state.get("pipeline_execution_attempts", 0) + 1
        print(f"[EXECUTE] Running pipeline (attempt {attempt})...")
        result = run_pipeline_subprocess(
            state=state,
            run_output_root=self.run_output_root,
            compare_estimates_fn=compare_estimates_with_actual,
            logger=self.logger,
        )
        self.logger.info('Leaving execute_pipeline')
        return result
    
    def cluster_analysis(self, state: SimpleModelAgentState) -> Dict[str, Any]:
        self._log_action("cluster_analysis", "start", "Analyze pairwise correspondences", "Supports investigator loop")
        self.logger.info("----------------------- Entering Cluster Analysis Helper -----------------------")
        print("[CLUSTER] Analyzing pairwise correspondences...")

        correspondence_files_to_process = helper_collect_latest_correspondence_files(state)
        if not correspondence_files_to_process:
            print("[CLUSTER] No correspondence files found — skipping")
            return {"cluster_analysis_result": {"warning": "stale_or_missing_correspondence_files", "_investigation": {"files": {}}}}

        print(f"[CLUSTER] Analyzing {len(correspondence_files_to_process)} pair(s)...")

        try:
            cluster_tester = ClusterTester(verbose=True)
            report = cluster_tester.run(correspondence_files_to_process)
            if not isinstance(report, dict):
                report = {"_overall": {"recommended_strategy": "None", "parameters": {}}, "raw_report": report}
            report["_investigation"] = helper_summarize_correspondence_entries(correspondence_files_to_process)

            return {"cluster_analysis_result": report}
        except Exception as e:
            print(f"[CLUSTER] Error: {e}")
            self.logger.error(f"Cluster analysis helper failed: {traceback.format_exc()}")
            return {"cluster_analysis_result": {"error": str(e)}}
        
    def evaluation_adaption(self, state: SimpleModelAgentState):
        self._log_action("evaluation_adaption", "start", "Generate evaluation code", "Measures fusion quality")

        self.logger.info('----------------------- Entering evaluation_adaption -----------------------')

        example_eval_path = INPUT_DIR + "example_pipelines/example_evaluation.py"
        example_eval_code = open(example_eval_path).read()
    
        fused_output_path = config.FUSED_OUTPUT_PATH
        force_test_eval = bool(state.get("_force_test_eval"))
        eval_testset_path = self._evaluation_testset_path(state, force_test=force_test_eval)
        eval_stage = self._evaluation_stage_label(state, force_test=force_test_eval)

        # First-row preview of the evaluation set for the system prompt.
        eval_previews = self._load_dataset_previews([eval_testset_path] if eval_testset_path else [])
        testset_preview = eval_previews.get(eval_testset_path, "Failed to load or empty dataset")
    
        system_prompt = f"""
        You are a data scientist evaluating a data fusion pipeline.
    
        Example evaluation code:
        {example_eval_code}
    
        Generated integration pipeline:
        {state['integration_pipeline_code']}
    
        Dataset profiles:
        {json.dumps(state['data_profiles'], indent=2)}
    
        The fused output is located at (use this EXACT path, not the one from the example):
        {fused_output_path}

        The evaluation output must be saved to:
        {config.EVALUATION_JSON_PATH}

        The evaluation stage is:
        {eval_stage}

        The evaluation set is located at:
        {eval_testset_path}

        The first row of the evaluation set looks like the following:
        {testset_preview}

        Normalization directives and report from previous steps:
        {json.dumps(state.get("normalization_directives", {}), indent=2)}
        {json.dumps(state.get("normalization_report", {}), indent=2)}
    
        Create evaluation code that:
        - Uses the correct fusion strategy
        - Loads the fused output
        - Loads the gold standard evaluation set (validation in sealed mode, otherwise test)
        - Prints structured evaluation metrics
        - Prints chosen evaluation functions in a compact one-line summary
        - If helper modules are imported from local files (e.g., list_normalization), use a robust fallback import pattern that works from output/code execution context.
        - Does NOT require exact matching everywhere by default.

        {EVALUATION_ROBUSTNESS_RULES_BLOCK}
        """

        evaluation_analysis = state.get("evaluation_analysis", None)
        if evaluation_analysis:
            system_prompt += f"""
            Evaluation reasoning from prior pipeline run:
            {evaluation_analysis}
            """
        reasoning_brief = state.get("evaluation_reasoning_brief", {})
        if isinstance(reasoning_brief, dict) and reasoning_brief:
            system_prompt += f"""
            Compact summary from the prior reasoning:
            {json.dumps(reasoning_brief, indent=2)}
            """

        # -- Human prompt construction -------
        if not state.get("evaluation_execution_result"):
            print("[EVAL CODE] Generating evaluation script...")
            # Initial generation of evaluation code.
            human_content = """
            Create the evaluation code.
            """
        else:
            # Attempts to fix a previously failing evaluation script.
            attempts = state.get("evaluation_execution_attempts", 0)
            print(f"[EVAL CODE] Fixing evaluation error (attempt {attempts})...")

            eval_path = config.EVALUATION_CODE_PATH
            with open(eval_path, "r", encoding="utf-8") as f:
                broken_code = f.read()
    
            error = state["evaluation_execution_result"]
    
            human_content = f"""
            You previously generated Python evaluation code.
            Executing this code caused the following error:
    
            {error}
    
            Here is the current evaluation code:
            {broken_code}
    
            Fix the code so that it executes successfully.
            Output ONLY the corrected Python code.
            """

        if "search_documentation" in self.tools and not state.get("evaluation_execution_result"):
            self.tools["search_documentation"].reset()

        # Appends the process instructions on every attempt so the latest error context is included.
        system_prompt += """
            **PROCESS:**
            1.  **THINK**: Analyze the provided integration pipeline code, dataset profiles, and any previous error reports.
            2.  **RESEARCH**: If you are unsure how to use a PyDI function or class for evaluation, you MUST use the `search_documentation` tool. You can call it multiple times based on given information such as, fusion strategy, data profiles, error messages etc. Ask specific questions (e.g., "How to evaluate a fusion output?", "What functions does PyDI provide for evaluation?").
            3.  **CODE**: Once you have gathered enough information, write the complete, executable Python code for the evaluation. **Your final output in this process must be only the Python code itself.**"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]
        state["eval_messages"] = messages

        self.logger.info("Input Message:" + str(messages))

        result = self._invoke_with_usage(messages, "evaluation_adaption")
        messages.append(result)

        if result.tool_calls:
            tool_messages = []
            for tool_call in result.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id")
                try:
                    if tool_name in self.tools:
                        tool_output = self.tools[tool_name].invoke(tool_args)
                    else:
                        tool_output = (
                            f"Tool '{tool_name}' not available. "
                            f"Available tools: {list(self.tools.keys())}"
                        )
                except Exception as tool_err:
                    tool_output = f"Tool '{tool_name}' execution error: {tool_err}"
                if tool_id:
                    tool_messages.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_id)
                    )
            messages.extend(tool_messages)
            
            # Re-invokes after tool results to obtain final evaluation code.
            final_response = self._invoke_with_usage(messages, "evaluation_adaption")
            messages.append(final_response)
            result = final_response

        if hasattr(result, 'tool_calls') and result.tool_calls:
            print("[EVAL CODE] LLM stuck in tool loop — aborting evaluation generation")
            return {
                "evaluation_code": "",
                "eval_messages": messages,
                "evaluation_execution_result": "error: [tool_loop] LLM requested tools beyond the 2-round limit; retry with simpler evaluation approach",
            }
        else:
            code = helper_extract_llm_text(result)
            self.logger.info("RESULT:" + str(code))

            code = helper_extract_python_code(code)
            code = self._apply_evaluation_guardrails(code)
            # Pre-validate syntax before writing to disk to avoid wasting an execution attempt
            try:
                compile(code, config.EVALUATION_CODE_PATH, "exec")
            except SyntaxError as syn_err:
                print(f"[EVAL CODE] Syntax error at line {syn_err.lineno}: {syn_err.msg}")
                return {
                    "evaluation_code": code,
                    "eval_messages": messages,
                    "evaluation_execution_result": f"error: [syntax_error] SyntaxError at line {syn_err.lineno}: {syn_err.msg}",
                }
            with open(config.EVALUATION_CODE_PATH, "w") as f:
                f.write(code)

        self.logger.info('Leaving evaluation_adaption')
    
        return {"evaluation_code": code, "eval_messages": messages}

    def execute_evaluation(self, state: SimpleModelAgentState):
        self._log_action("execute_evaluation", "start", "Run evaluation", "Produces accuracy metrics", {"attempt": state.get("evaluation_execution_attempts", 0)+1})
        self.logger.info('----------------------- Entering execute_evaluation -----------------------')
        attempt = state.get("evaluation_execution_attempts", 0) + 1
        print(f"[EVAL EXEC] Running evaluation (attempt {attempt})...")
        stage_label = self._evaluation_stage_label(state, force_test=bool(state.get("_force_test_eval")))
        result = run_evaluation_subprocess(
            state=state,
            run_output_root=self.run_output_root,
            stage_label=stage_label,
            logger=self.logger,
        )
        self.logger.info('Leaving execute_evaluation')
        return result

    def investigate(self, state: SimpleModelAgentState) -> Dict[str, Any]:
        """Multi-turn investigation agent: analyzes evidence, optionally writes
        and executes diagnostic code, then decides which pipeline stage to fix.

        Returns an investigation log dict with full transcript and final decision.
        """
        from prompts.investigation_prompt import (
            INVESTIGATION_SYSTEM_PROMPT,
            build_investigation_context,
        )

        self._log_action("investigate", "start", "Investigate pipeline issues", "LLM-driven diagnosis and routing")
        overall = state.get("evaluation_metrics", {})
        acc = overall.get("overall_accuracy", "?") if isinstance(overall, dict) else "?"
        print(f"[INVESTIGATION] Starting investigation (accuracy={acc})...")

        evidence = build_investigation_context(state, state.get("investigator_probe_results", {}))
        messages = [
            SystemMessage(content=INVESTIGATION_SYSTEM_PROMPT),
            HumanMessage(content=evidence),
        ]

        max_turns = int(os.getenv("INVESTIGATION_MAX_TURNS", "3"))
        code_timeout = int(os.getenv("INVESTIGATION_CODE_TIMEOUT", "120"))
        investigation_log: Dict[str, Any] = {"turns": [], "evidence_length": len(evidence)}

        for turn in range(1, max_turns + 1):
            print(f"[INVESTIGATION] Turn {turn}/{max_turns}...")
            result = self._invoke_base_with_usage(messages, f"investigation_turn_{turn}")
            raw_content = helper_extract_llm_text(result)

            # Parse JSON response
            response = self._parse_investigation_response(raw_content)
            if not response:
                # Debug: log what the LLM actually returned
                _dbg_content = getattr(result, 'content', None)
                _dbg_addl = getattr(result, 'additional_kwargs', {})
                _dbg_tool = getattr(result, 'tool_calls', None)
                if not raw_content:
                    print(f"[INVESTIGATION] LLM returned empty content. content={repr(_dbg_content)[:100]}, "
                          f"additional_kwargs keys={list((_dbg_addl or {}).keys())}, "
                          f"tool_calls={repr(_dbg_tool)[:100]}")
                else:
                    print(f"[INVESTIGATION] Parse failed on: {raw_content[:200]}")
                print("[INVESTIGATION] Parse failed — forcing pipeline_adaption decision")
                response = {
                    "action": "decide",
                    "diagnosis": "Investigation response could not be parsed",
                    "next_node": "pipeline_adaption",
                    "reasoning": "Defaulting to pipeline adaptation",
                    "recommendations": [],
                }

            if response.get("action") == "investigate":
                hypothesis = response.get("hypothesis", "")
                code = response.get("code", "")
                print(f"[INVESTIGATION] Testing: {hypothesis[:120]}")

                execution = self._execute_investigation_code(code, timeout=code_timeout)
                turn_record = {
                    "turn": turn,
                    "action": "investigate",
                    "hypothesis": hypothesis,
                    "code": code,
                    "execution": {
                        "success": execution["success"],
                        "stdout": execution["stdout"],
                        "stderr": execution["stderr"][:2000],
                    },
                }
                investigation_log["turns"].append(turn_record)

                status = "OK" if execution["success"] else "FAIL"
                stdout_preview = execution["stdout"][:200].replace("\n", " ")
                print(f"[INVESTIGATION] Code {status}: {stdout_preview}")

                # Feed execution results back into conversation
                messages.append(AIMessage(content=raw_content))
                feedback = f"Execution {'succeeded' if execution['success'] else 'failed'}.\n\nstdout:\n{execution['stdout']}"
                if execution["stderr"]:
                    feedback += f"\n\nstderr:\n{execution['stderr'][:2000]}"
                if turn == max_turns:
                    feedback += "\n\nThis is your last turn. You MUST respond with a decision now."
                messages.append(HumanMessage(content=feedback))
                continue

            # action == "decide" (or forced)
            print(f"[INVESTIGATION] Decision: {response.get('next_node')} — {response.get('diagnosis', '')[:150]}")
            investigation_log["turns"].append({
                "turn": turn,
                "action": "decide",
                "decision": response,
            })
            investigation_log["decision"] = response
            break

        # Fallback if max turns exhausted without decision
        if "decision" not in investigation_log:
            fallback = {
                "action": "decide",
                "diagnosis": "Investigation inconclusive after max turns",
                "next_node": "pipeline_adaption",
                "reasoning": "Defaulting to pipeline adaptation after exhausting investigation budget",
                "recommendations": [],
            }
            investigation_log["turns"].append({"turn": max_turns, "action": "decide", "decision": fallback})
            investigation_log["decision"] = fallback

        recs = investigation_log["decision"].get("recommendations", [])
        rec_summary = ", ".join(r.get("attribute", "?") for r in recs[:3] if isinstance(r, dict))
        print(f"[INVESTIGATION] Complete: {len(investigation_log['turns'])} turn(s) → {investigation_log['decision']['next_node']}"
              + (f" (targets: {rec_summary})" if rec_summary else ""))
        return investigation_log

    @staticmethod
    def _parse_investigation_response(raw: str) -> Dict[str, Any] | None:
        """Extract a JSON object from the LLM's investigation response."""
        import re as _re

        text = str(raw or "").strip()
        if not text:
            return None

        # Try direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "action" in parsed:
                return parsed
        except Exception:
            pass

        # Try extracting from code fences
        fence_match = _re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if fence_match:
            try:
                parsed = json.loads(fence_match.group(1))
                if isinstance(parsed, dict) and "action" in parsed:
                    return parsed
            except Exception:
                pass

        # Try finding first { ... } block
        brace_match = _re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group(0))
                if isinstance(parsed, dict) and "action" in parsed:
                    return parsed
            except Exception:
                pass

        return None

    def _execute_investigation_code(self, code: str, timeout: int = 120) -> Dict[str, Any]:
        """Execute investigation Python code in a subprocess."""
        import re as _re

        code_path = os.path.join(config.CODE_DIR, "investigation_probe.py")
        os.makedirs(os.path.dirname(code_path), exist_ok=True)

        # Basic guardrails
        code = _re.sub(r"\bsys\.stdin\.read\(\)", "''", code)
        code = _re.sub(r"\bsys\.stdin\.buffer\.read\(\)", "b''", code)
        code = _re.sub(r"\binput\((.*?)\)", "''", code)

        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            result = subprocess.run(
                [sys.executable, code_path],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=timeout,
            )
            return {
                "success": result.returncode == 0,
                "stdout": (result.stdout or "")[:10000],
                "stderr": (result.stderr or "")[:5000],
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": f"Timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e)[:2000]}

    def human_review_export(self, state: SimpleModelAgentState):
        self._log_action(
            "human_review_export",
            "start",
            "Generate final human-review package",
            "Creates reviewer-friendly fusion/source/testset comparison artifacts",
        )
        print("[REVIEW] Building human-review package...")

        os.makedirs(config.CODE_DIR, exist_ok=True)
        os.makedirs(config.HUMAN_REVIEW_DIR, exist_ok=True)

        review_code_path = config.HUMAN_REVIEW_CODE_PATH
        review_summary_json = os.path.join(config.HUMAN_REVIEW_DIR, "human_review_summary.json")
        review_summary_md = os.path.join(config.HUMAN_REVIEW_DIR, "human_review_summary.md")
        fused_review_csv = os.path.join(config.HUMAN_REVIEW_DIR, "fused_review_table.csv")
        source_lineage_csv = os.path.join(config.HUMAN_REVIEW_DIR, "source_lineage_long.csv")
        diff_csv = os.path.join(config.HUMAN_REVIEW_DIR, "fusion_vs_testset_diff.csv")

        review_attributes: List[str] = []
        fused_csv_path = config.FUSED_OUTPUT_PATH
        if os.path.exists(fused_csv_path):
            try:
                fused_preview_df = pd.read_csv(fused_csv_path, nrows=1)
                review_attributes = [
                    c
                    for c in fused_preview_df.columns
                    if not str(c).startswith("_fusion_") and str(c) not in {"_fusion_sources", "_fusion_source_datasets", "_fusion_confidence", "_fusion_metadata"}
                ]
            except Exception:
                review_attributes = []

        context_payload = {
            "datasets": state.get("datasets", []),
            "fusion_testset": state.get("fusion_testset"),
            "review_attributes": review_attributes,
            "evaluation_metrics": state.get("evaluation_metrics", {}),
            "auto_diagnostics": state.get("auto_diagnostics", {}),
            "integration_diagnostics_report": state.get("integration_diagnostics_report", {}),
            "output_paths": {
                "fused_csv": config.FUSED_OUTPUT_PATH,
                "fused_debug_jsonl": os.path.join(config.OUTPUT_DIR, "data_fusion", "debug_fusion_data.jsonl"),
                "evaluation_json": config.EVALUATION_JSON_PATH,
                "review_summary_json": review_summary_json,
                "review_summary_md": review_summary_md,
                "fused_review_csv": fused_review_csv,
                "source_lineage_csv": source_lineage_csv,
                "diff_csv": diff_csv,
            },
        }

        system_prompt = HUMAN_REVIEW_SYSTEM_PROMPT

        def _generate_code(feedback: str | None = None) -> str:
            human_content = f"""
            Context:
            {json.dumps(context_payload, indent=2)}

            Generate a Python script that builds the required human-review outputs.
            """
            if feedback:
                human_content += f"""

                Previous script failed with:
                {feedback}

                Fix it and return corrected Python code only.
                """

            response = self._invoke_base_with_usage(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_content),
                ],
                "human_review_export_generation",
            )
            return helper_extract_python_code(response)

        execution_result = "error: not_run"
        review_code = ""
        feedback = None

        def _validate_wide_review_columns(path: str, attributes: List[str]) -> tuple[bool, str]:
            if not os.path.exists(path):
                return False, f"missing required file: {path}"
            try:
                review_df = pd.read_csv(path, nrows=1, dtype=str)
            except Exception as e:
                return False, f"could not read {path}: {e}"

            if not attributes:
                return True, "no attributes discovered; skipped strict schema validation"

            cols = set(review_df.columns.tolist())
            missing: List[str] = []
            for attr in attributes:
                expected = [
                    f"{attr}_test",
                    f"{attr}_fused",
                    f"{attr}_source_1",
                    f"{attr}_source_2",
                    f"{attr}_source_3",
                ]
                for col in expected:
                    if col not in cols:
                        missing.append(col)

            if missing:
                preview = ", ".join(missing[:40])
                if len(missing) > 40:
                    preview += f", ... and {len(missing) - 40} more"
                return False, f"missing required wide review columns: {preview}"
            return True, "ok"

        for attempt in range(1, 4):
            review_code = _generate_code(feedback)
            with open(review_code_path, "w", encoding="utf-8") as f:
                f.write(review_code)

            try:
                run = subprocess.run(
                    [sys.executable, review_code_path],
                    capture_output=True,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    timeout=HUMAN_REVIEW_EXEC_TIMEOUT,
                )
                if run.returncode == 0:
                    schema_ok, schema_msg = _validate_wide_review_columns(fused_review_csv, review_attributes)
                    if not schema_ok:
                        feedback = truncate_stderr(
                            "Script executed but fused_review_table.csv does not match required wide schema. "
                            + schema_msg
                            + " Use exact column pattern <attribute>_test/<attribute>_fused/<attribute>_source_1/<attribute>_source_2/<attribute>_source_3 for every attribute in context_payload.review_attributes."
                        )
                        execution_result = truncate_stderr(f"error: {feedback}")
                        print("[REVIEW] Schema validation failed — regenerating")
                        continue

                    execution_result = "success"
                    print("[REVIEW] Human-review package complete")
                    break
                review_error_info = classify_execution_error(
                    returncode=run.returncode,
                    stderr=run.stderr or "",
                    stdout=run.stdout or "",
                )
                feedback = f"[{review_error_info['category']}] {review_error_info['suggestion']}"
                execution_result = truncate_stderr(f"error: {feedback}")
            except subprocess.TimeoutExpired:
                review_error_info = classify_execution_error(0, "", timed_out=True)
                feedback = f"[{review_error_info['category']}] {review_error_info['suggestion']}"
                execution_result = truncate_stderr(f"error: {feedback}")
            except Exception as e:
                review_error_info = classify_execution_error(0, str(e))
                feedback = f"[{review_error_info['category']}] {review_error_info['suggestion']}"
                execution_result = truncate_stderr(f"error: {feedback}")

        report_payload: Dict[str, Any] = {}
        if os.path.exists(review_summary_json):
            try:
                with open(review_summary_json, "r", encoding="utf-8") as f:
                    report_payload = json.load(f)
            except Exception as e:
                report_payload = {"error": f"could_not_read_human_review_summary: {e}"}

        return {
            "human_review_code": review_code,
            "human_review_execution_result": execution_result,
            "human_review_report": report_payload,
        }


    def sealed_final_test_evaluation(self, state: SimpleModelAgentState):
        self._log_action(
            "sealed_final_test_evaluation",
            "start",
            "Run one-time held-out test evaluation",
            "Prevents optimization on final test while still reporting final quality",
        )

        if not self._sealed_eval_active(state):
            print("[SEALED TEST] Not active — skipping")
            return {
                "final_test_evaluation_execution_result": "skipped",
                "final_test_evaluation_metrics": {},
            }

        print("[SEALED TEST] Evaluating on held-out test set...")
        temp_state = dict(state)
        temp_state["_force_test_eval"] = True
        temp_state["evaluation_execution_result"] = ""
        temp_state["evaluation_execution_attempts"] = 0

        # Use the best pipeline code (not the latest) for sealed test
        best_code = state.get("best_pipeline_code", "")
        if best_code and best_code != state.get("integration_pipeline_code", ""):
            print("[SEALED TEST] Restoring best pipeline code for sealed evaluation")
            temp_state["integration_pipeline_code"] = best_code
            # Re-execute the best pipeline so fusion output matches
            temp_state["pipeline_execution_result"] = ""
            temp_state["pipeline_execution_attempts"] = 0
            exec_pipeline = self.execute_pipeline(temp_state)
            temp_state.update(exec_pipeline)
            if "error" in str(temp_state.get("pipeline_execution_result", "")).lower():
                print("[SEALED TEST] Best pipeline re-execution failed — falling back to latest")
                temp_state["integration_pipeline_code"] = state.get("integration_pipeline_code", "")
                temp_state.update(self.execute_pipeline(temp_state))

        eval_updates = self.evaluation_adaption(temp_state)
        temp_state.update(eval_updates)

        exec_updates = self.execute_evaluation(temp_state)
        temp_state.update(exec_updates)

        eval_path = config.EVALUATION_JSON_PATH
        final_metrics: Dict[str, Any] = {}
        metrics_from_execution = temp_state.get("evaluation_metrics_from_execution", {})
        if is_metrics_payload(metrics_from_execution):
            final_metrics = dict(metrics_from_execution)
        elif os.path.exists(eval_path):
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    final_metrics = json.load(f)
            except Exception as e:
                final_metrics = {"error": f"could_not_read_final_test_metrics: {e}"}

        final_acc = final_metrics.get("overall_accuracy") if isinstance(final_metrics, dict) else None
        if final_acc is not None:
            try:
                print(f"[SEALED TEST] Held-out accuracy: {float(final_acc):.3%}")
            except Exception:
                pass

        return {
            "final_test_evaluation_execution_result": temp_state.get("evaluation_execution_result", ""),
            "final_test_evaluation_metrics": final_metrics,
        }


    def evaluation_decision(self, state: SimpleModelAgentState):
        self._log_action("evaluation_decision", "start", "Decide whether to iterate", "Drives improvement loop")
        self.logger.info('----------------------- Entering evaluation_decision -----------------------')
        print("[EVAL DECISION] Processing evaluation metrics...")
        eval_testset_path = self._evaluation_testset_path(state, force_test=bool(state.get("_force_test_eval")))
        eval_stage_label = self._evaluation_stage_label(state, force_test=False)
        result = process_evaluation_decision(
            state=state,
            eval_testset_path=eval_testset_path,
            eval_stage_label=eval_stage_label,
            compute_auto_diagnostics_fn=self._compute_auto_diagnostics,
            print_total_usage_fn=self._print_total_usage,
            logger=self.logger,
        )
        self.logger.info('Leaving evaluation_decision')
        return result
    def save_results(self, state: SimpleModelAgentState):
        return _save_results(
            state,
            run_id=self.run_id,
            run_output_root=self.run_output_root,
            token_usage=self._token_tracker.usage.copy(),
            is_metrics_payload_fn=is_metrics_payload,
            sealed_eval_active=self._sealed_eval_active(state),
            logger=self.logger,
            log_action_fn=self._log_action,
        )


# -- Invoke Pipeline -----------------------------------------------------------

if __name__ == "__main__":
    if USE_LLM == "gemini":
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=LLM_REQUEST_TIMEOUT,
            max_retries=2,
        )
    elif USE_LLM == "groq":
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            request_timeout=LLM_REQUEST_TIMEOUT,
        )
    elif USE_LLM == "gpt":
        llm = ChatOpenAI(
            model="gpt-5.4",
            temperature=0,
            max_tokens=None,
            request_timeout=LLM_REQUEST_TIMEOUT,
            max_retries=OPENAI_MAX_RETRIES,
        )

    # Music use-case dataset configuration.
    entity_matching_testsets = {
        ("discogs", "lastfm"): INPUT_DIR + "gen_TrainSets/discogs_lastfm_goldstandard_blocking.csv",
        ("discogs", "musicbrainz"): INPUT_DIR + "gen_TrainSets/discogs_musicbrainz_goldstandard_blocking.csv",
        ("musicbrainz", "lastfm"): INPUT_DIR + "gen_TrainSets/musicbrainz_lastfm_goldstandard_blocking.csv",
    }
    datasets = [
        INPUT_DIR + "datasets/music/discogs.xml",
        INPUT_DIR + "datasets/music/lastfm.xml",
        INPUT_DIR + "datasets/music/musicbrainz.xml",
    ]
    fusion_testset = INPUT_DIR + "datasets/music/testsets/test_set.xml"
    validation_fusion_testset = INPUT_DIR + "datasets/music/testsets/validation_set.xml"

    # -- Agent initialization and invocation --

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

    # Activates sealed evaluation mode when a separate validation set exists.
    if validation_fusion_testset:
        invoke_payload["validation_fusion_testset"] = validation_fusion_testset

    result = agent.graph.invoke(
        invoke_payload,
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
    )
