#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json 
import getpass
import logging
import glob
from pathlib import Path
from time import sleep
import subprocess
import sys
import re
import shutil
import pandas as pd
import ast
import time
from datetime import datetime, timezone

from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
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
try:
    from workflow_logging import attach_logging, configure_workflow_logger, log_workflow_action
except Exception:
    try:
        from agents.workflow_logging import attach_logging, configure_workflow_logger, log_workflow_action
    except Exception:
        from helpers.workflow_logging import attach_logging, configure_workflow_logger, log_workflow_action

from dotenv import load_dotenv
load_dotenv()


# In[ ]:


# print(sys.getrecursionlimit())


# In[ ]:


# sys.setrecursionlimit(5000)
# print(sys.getrecursionlimit())


# ## Initialize

# In[ ]:


OUTPUT_DIR = "output/"
INPUT_DIR = "input/"
OPENAI_REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "600"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", str(OPENAI_REQUEST_TIMEOUT)))
PIPELINE_TOOLCALL_LOOP_LIMIT = int(os.getenv("PIPELINE_TOOLCALL_LOOP_LIMIT", "6"))

INCLUDE_DOCS = False # IMPORTANT: Use carefully since token usage is increased drastically with documentation
USE_LLM = "gpt"
#USE_LLM = "gemini"
#USE_LLM = "groq"


# In[ ]:


if USE_LLM == "gemini": # or USE_LLM == "gemini_broken":
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
elif USE_LLM == "groq":
    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
elif USE_LLM == "gpt":
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


# In[ ]:


logging.basicConfig(filename= OUTPUT_DIR + 'agent.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG,
                    encoding='utf-8')


# ## Utilities

# In[ ]:


def load_dataset(path):
    # check file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    ext = os.path.splitext(path)[1].lower()

    # load dataset according to extension
    if ext == ".parquet":
        df = load_parquet(path)
    elif ext == ".csv":
        df = load_csv(path)
    elif ext == ".xml":
        df = load_xml(path, nested_handling="aggregate")
    else:
        raise ValueError(f"Unsupported format: {ext}. Supported: .csv, .parquet, .xml")
    return df


def _clean_report_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_report_section(text: str, headings: List[str]) -> str:
    content = _clean_report_text(text)
    if not content:
        return ""
    lines = content.split("\n")
    heading_set = {h.lower() for h in headings}
    collected: List[str] = []
    active = False
    for raw_line in lines:
        line = raw_line.strip()
        normalized = re.sub(r"^[#*\\-\\s]+", "", line).strip()
        normalized = normalized.rstrip(":").strip().lower()
        if normalized in heading_set:
            active = True
            continue
        if active and line and re.match(r"^[#A-Z][A-Za-z ]{2,40}:?$", line):
            break
        if active and line:
            collected.append(line)
    return _clean_report_text("\n".join(collected))


def _first_sentences(text: str, limit: int = 2) -> str:
    cleaned = _clean_report_text(text)
    if not cleaned:
        return ""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\\s+", cleaned) if p.strip()]
    if not parts:
        return cleaned
    return " ".join(parts[:limit]).strip()


def _build_reasoning_brief(analysis: str) -> Dict[str, str]:
    cleaned = _clean_report_text(analysis)
    if not cleaned:
        return {}
    problem = _extract_report_section(cleaned, ["What went wrong", "Main problem", "Problem"])
    next_step = _extract_report_section(
        cleaned,
        ["What the agent should try next", "Next pass focus", "Next step", "What to try next"],
    )
    normalization = _extract_report_section(
        cleaned,
        ["Normalization recommendations", "Normalization recommendation"],
    )
    takeaway = _extract_report_section(cleaned, ["Report takeaway", "Takeaway"])
    if not problem:
        problem = _first_sentences(cleaned, limit=2)
    if not next_step:
        fallback = _extract_report_section(cleaned, ["Recommendations", "Recommended changes"])
        next_step = _first_sentences(fallback or cleaned, limit=2)
    if not takeaway:
        takeaway = _first_sentences(next_step or cleaned, limit=1)
    return {
        "problem": problem,
        "next_step": next_step,
        "normalization": normalization,
        "takeaway": takeaway,
    }


def _compact_report_list(items: Any, limit: int = 3) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for item in items:
        text = _clean_report_text(item)
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out
    


# ## Tools

# In[ ]:


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
            # return a structured JSON error string (LLM will see this as content)
            return json.dumps({"error": f"Dataset at {path} loaded as empty or failed to load."})

        profiler = DataProfiler()
        profile = profiler.summary(df, print_summary=False)

        # ensure to return a JSON string (your docstring promised JSON string)
        try:
            profile_json = json.dumps(profile, default=str)
        except Exception:
            # fallback: convert to str if not json-serializable
            profile_json = json.dumps({"profile_str": str(profile)})

        return profile_json
        


# In[ ]:


class SearchDocumentationTool(BaseTool):
    """
    A tool to search the PyDI documentation vector database.
    """
    name: str = "search_documentation"
    description: str = (
        "Searches the PyDI library documentation for a given query. "
        "Use this to find information about functions, classes, or how-to instructions. "
        "The input should be a specific question about the library."
    )
    call_count: int = 0

    def _clean_text(self, text: str) -> str:
        """
        Cleans and normalizes LLM or tool output text.
        Safe for LangChain AIMessage.content or ToolMessage.content.
        """

        if text is None:
            return ""

        text = str(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.strip()

        return text


    def _run(self, query: str) -> str:
        """Executes the documentation search."""
        self.call_count += 1
        print(f"[SEARCH TOOL CALL {self.call_count}]: Asking '{query}'")
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_db_path = os.path.join(INPUT_DIR, "api_documentation/pydi_apidocs_vector_db/")

        if not os.path.exists(vector_db_path):
            error_msg = f"Error: Vector DB not found at {vector_db_path}"
            # self._save_query_response(query, error_msg)
            return error_msg
        try:
            db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
            docs = db.similarity_search(query, k=4)
            
            if not docs:
                response = "No relevant documentation found for your query."
            else:
                response = "\n\n-----\n\n".join([r.page_content for r in docs])
                response = self._clean_text(response)
            
            # Save query and response to JSON file
            # self._save_query_response(query, response)
            
            return response
        except Exception as e:
            error_response = f"An error occurred while searching documentation: {str(e)}\n{traceback.format_exc()}"
            # self._save_query_response(query, error_response)
            return error_response
    
    # def _save_query_response(self, query: str, response: str) -> None:
    #     """Save query and response to JSON file."""
    #     output_file = os.path.join(OUTPUT_DIR, "searchtool_response.json")
        
    #     # Ensure output directory exists
    #     os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    #     # Load existing data or create new list
    #     if os.path.exists(output_file):
    #         with open(output_file, 'r', encoding='utf-8') as f:
    #             try:
    #                 data = json.load(f)
    #                 if isinstance(data, list) and len(data) > 50:
    #                     data = data[-50:]  # Keep only the last 50 entries to prevent file from growing indefinitely
    #                 else:
    #                     pass
    #             except json.JSONDecodeError:
    #                 data = []
    #     else:
    #         data = []
        
    #     # Append new query-response pair
    #     data.append({
    #         "call_number": self.call_count,
    #         "query": query,
    #         "response": response
    #     })
        
    #     # Write back to file
    #     with open(output_file, 'w', encoding='utf-8') as f:
    #         json.dump(data, f, indent=2, ensure_ascii=False)

    def reset(self):
        """Resets the call count for a new agent run."""
        self.call_count = 0


# ## Agents

# In[ ]:


class SimpleModelAgentState(TypedDict):
    datasets: list
    entity_matching_testsets: list
    fusion_testset: str
    validation_fusion_testset: str
    blocking_config: Dict  # NEW: Blocking config from BlockingTester
    matching_config: Dict  # NEW: Matching config from MatchingTester
    matcher_mode: str  # NEW: Matcher mode (rule_based/ml/auto)
    schema_correspondences: Dict  # NEW: Schema matching results
    
    data_profiles: Dict

    messages: List[AnyMessage] # Save conversation history during research
    eval_messages: List[AnyMessage] # Save conversation history during evaluation

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
    enable_cross_run_memory: bool

class SimpleModelAgent:
    
    def __init__(self, model, tools: Dict[str, BaseTool]):
        self.logger = configure_workflow_logger(output_dir=OUTPUT_DIR)
        
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost": 0.0}
        self.tools = tools
        self.model = model.bind_tools(list(self.tools.values()))
        self.base_model = model
        self.run_id = ""
        self.run_output_root = ""
        self._reset_run_context()

        # prepare the StateGraph
        graph = StateGraph(SimpleModelAgentState)

        # create nodes
        graph.add_node("match_schemas", self.match_schemas)
        graph.add_node("profile_data", self.profile_data)
        try:
            from helpers.normalization_orchestrator import run_normalization_node
            from helpers.evaluation_orchestrator import run_evaluation_node
            from helpers.investigator_orchestrator import run_investigator_node
        except Exception:
            from agents.helpers.normalization_orchestrator import run_normalization_node
            from agents.helpers.evaluation_orchestrator import run_evaluation_node
            from agents.helpers.investigator_orchestrator import run_investigator_node

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

        # create edges
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
            lambda state: (
                "evaluation_node"
                if state.get("pipeline_execution_result", "")
                .lower()
                .startswith("success")
                else (
                    "pipeline_adaption"
                    if state.get("pipeline_execution_attempts", 0) < 3
                    else END
                )
            ),
            {
                "evaluation_node": "evaluation_node",
                "pipeline_adaption": "pipeline_adaption",
                END: END,
            },
        )

        # Consolidated evaluation + investigation area

        graph.add_conditional_edges(
            "evaluation_node",
            lambda state: (
                "investigator_node"
                if isinstance(state.get("evaluation_execution_result", ""), str)
                   and state["evaluation_execution_result"].lower().startswith("success")
                else END
            ),
            {
                "investigator_node": "investigator_node",
                END: END,
            }
        )

        graph.add_conditional_edges(
            "investigator_node",
            lambda state: state.get("investigator_decision", "human_review_export"),
            {
                "normalization_node": "normalization_node",
                "run_matching_tester": "run_matching_tester",
                "pipeline_adaption": "pipeline_adaption",
                "human_review_export": "human_review_export",
                END: END
            }
        )
        graph.add_conditional_edges(
            "human_review_export",
            lambda state: (
                "sealed_final_test_evaluation"
                if self._sealed_eval_active(state)
                else "save_results"
            ),
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
                output_dir=OUTPUT_DIR,
                notebook_name="AdaptationPipeline_nblazek_setcount_exec",
            )
        except Exception as e:
            print(f"[!] Workflow logging attachment failed: {e}")

    def _cross_run_memory_enabled(self, state: Dict[str, Any] | None = None) -> bool:
        if isinstance(state, dict) and "enable_cross_run_memory" in state:
            return bool(state.get("enable_cross_run_memory"))
        return os.getenv("AGENT_ENABLE_CROSS_RUN_MEMORY", "").strip().lower() in {"1", "true", "yes", "on"}

    def _reset_volatile_output_dirs(self) -> None:
        volatile_dirs = [
            OUTPUT_DIR + "blocking-evaluation",
            OUTPUT_DIR + "matching-evaluation",
            OUTPUT_DIR + "correspondences",
            OUTPUT_DIR + "cluster-evaluation",
            OUTPUT_DIR + "pipeline_evaluation",
            OUTPUT_DIR + "normalization",
            OUTPUT_DIR + "human_review",
            OUTPUT_DIR + "code",
            OUTPUT_DIR + "data_fusion",
            OUTPUT_DIR + "schema-matching",
        ]
        for path in volatile_dirs:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception:
                continue
            os.makedirs(path, exist_ok=True)

    def _reset_run_context(self, state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_output_root = os.path.join(OUTPUT_DIR, "runs", self.run_id)
        os.makedirs(self.run_output_root, exist_ok=True)
        os.makedirs(os.path.join(self.run_output_root, "pipeline"), exist_ok=True)
        os.makedirs(os.path.join(self.run_output_root, "evaluation"), exist_ok=True)
        os.makedirs(os.path.join(self.run_output_root, "snapshots"), exist_ok=True)
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost": 0.0}
        self._reset_volatile_output_dirs()
        return {
            "run_id": self.run_id,
            "run_output_root": self.run_output_root,
            "enable_cross_run_memory": self._cross_run_memory_enabled(state),
        }

    def prepare_for_new_run(self, state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self._reset_run_context(state)

    @staticmethod
    def _matching_refresh_gate(state: Dict[str, Any], min_pair_f1: float = 0.65) -> Dict[str, Any]:
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

    def _extract_usage_from_result(self, result: Any) -> Dict[str, int]:
        def _to_int(value: Any) -> int:
            try:
                return int(value)
            except Exception:
                return 0

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        usage_meta = getattr(result, "usage_metadata", None)
        if isinstance(usage_meta, dict):
            prompt_tokens = _to_int(usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens"))
            completion_tokens = _to_int(usage_meta.get("output_tokens") or usage_meta.get("completion_tokens"))
            total_tokens = _to_int(usage_meta.get("total_tokens"))

        response_meta = getattr(result, "response_metadata", None)
        if isinstance(response_meta, dict):
            token_usage = response_meta.get("token_usage", {})
            if isinstance(token_usage, dict):
                if prompt_tokens <= 0:
                    prompt_tokens = _to_int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens"))
                if completion_tokens <= 0:
                    completion_tokens = _to_int(token_usage.get("completion_tokens") or token_usage.get("output_tokens"))
                if total_tokens <= 0:
                    total_tokens = _to_int(token_usage.get("total_tokens"))

        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens

        return {
            "prompt_tokens": max(0, prompt_tokens),
            "completion_tokens": max(0, completion_tokens),
            "total_tokens": max(0, total_tokens),
        }

    def _resolve_model_name(self, result: Any) -> str:
        response_meta = getattr(result, "response_metadata", None)
        if isinstance(response_meta, dict):
            model_name = str(response_meta.get("model_name") or response_meta.get("model") or "").strip()
            if model_name:
                return model_name
        model_name = str(getattr(self.base_model, "model_name", "") or getattr(self.base_model, "model", "")).strip()
        return model_name

    def _resolve_openai_rates_per_1m(self, model_name: str) -> Tuple[Optional[float], Optional[float]]:
        input_override = os.getenv("OPENAI_INPUT_COST_PER_1M")
        output_override = os.getenv("OPENAI_OUTPUT_COST_PER_1M")
        if input_override and output_override:
            try:
                return float(input_override), float(output_override)
            except Exception:
                pass

        # Local fallback rate card (USD per 1M tokens); update if you change models/pricing.
        rate_card = {
            "gpt-5.2": (1.25, 10.0),
            "gpt-5": (1.25, 10.0),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (5.0, 15.0),
            "gpt-4.1-mini": (0.40, 1.60),
            "gpt-4.1": (2.0, 8.0),
            "gpt-4.1-nano": (0.10, 0.40),
        }

        key = str(model_name or "").strip().lower()
        for prefix, rates in rate_card.items():
            if key.startswith(prefix):
                return rates

        return None, None

    def _estimate_openai_cost(self, prompt_tokens: int, completion_tokens: int, model_name: str) -> Optional[float]:
        in_rate, out_rate = self._resolve_openai_rates_per_1m(model_name)
        if in_rate is None or out_rate is None:
            return None
        return (float(prompt_tokens) * in_rate + float(completion_tokens) * out_rate) / 1_000_000.0

    def _invoke_model_with_usage(self, model, message, tag):
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                with get_openai_callback() as cb:
                    result = model.invoke(message)
                break
            except Exception as e:
                if attempt >= max_attempts:
                    raise
                wait_s = 2 * attempt
                print(f"[!] {tag} invoke failed (attempt {attempt}/{max_attempts}): {e}. Retrying in {wait_s}s")
                sleep(wait_s)

        prompt_tokens = int(cb.prompt_tokens or 0)
        completion_tokens = int(cb.completion_tokens or 0)
        total_tokens = int(cb.total_tokens or 0)
        estimated_cost = float(cb.total_cost or 0.0)

        # Fallback: extract usage directly from response metadata if callback values are unavailable.
        if prompt_tokens <= 0 and completion_tokens <= 0:
            usage = self._extract_usage_from_result(result)
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            total_tokens = int(usage.get("total_tokens", 0))

        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens

        model_name = self._resolve_model_name(result)
        if estimated_cost <= 0.0 and (prompt_tokens > 0 or completion_tokens > 0):
            estimated = self._estimate_openai_cost(prompt_tokens, completion_tokens, model_name)
            if estimated is not None:
                estimated_cost = estimated

        self.token_usage["prompt_tokens"] += prompt_tokens
        self.token_usage["completion_tokens"] += completion_tokens
        self.token_usage["total_tokens"] += total_tokens
        self.token_usage["total_cost"] += estimated_cost

        print(f"TOKEN USAGE ({tag}):")
        print(f"   Model: {model_name or 'unknown'}")
        print(f"   Prompt tokens: {prompt_tokens:,}")
        print(f"   Completion tokens: {completion_tokens:,}")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Estimated cost: ${estimated_cost:.6f}")

        return result

    def _invoke_with_usage(self, message, tag):
        return self._invoke_model_with_usage(self.model, message, tag)

    def _invoke_base_with_usage(self, message, tag):
        return self._invoke_model_with_usage(self.base_model, message, tag)

    def _print_total_usage(self):
        t = self.token_usage
        print("TOTAL TOKEN USAGE:")
        print(f"   Prompt tokens: {t['prompt_tokens']:,}")
        print(f"   Completion tokens: {t['completion_tokens']:,}")
        print(f"   Total tokens: {t['total_tokens']:,}")
        print(f"   Estimated cost: ${t['total_cost']:.6f}")

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
        return os.path.join(self.run_output_root, *parts)

    def _snapshot_file(self, src_path: str, dst_rel_path: str) -> str:
        if not src_path or not os.path.exists(src_path):
            return ""
        dst_path = self._run_path(dst_rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        try:
            shutil.copy2(src_path, dst_path)
            return dst_path
        except Exception:
            return ""

    def _snapshot_patterns(self, patterns: List[str], dst_rel_dir: str) -> List[str]:
        captured: List[str] = []
        if not patterns:
            return captured
        dst_dir = self._run_path(dst_rel_dir)
        os.makedirs(dst_dir, exist_ok=True)
        seen: set[str] = set()
        for pattern in patterns:
            for src in sorted(glob.glob(pattern)):
                if src in seen:
                    continue
                seen.add(src)
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(dst_dir, os.path.basename(src))
                try:
                    shutil.copy2(src, dst)
                    captured.append(dst)
                except Exception:
                    continue
        return captured

    def _snapshot_pipeline_attempt(
        self,
        *,
        cycle_index: int,
        exec_attempt: int,
        execution_result: str,
        stdout: str,
        stderr: str,
        fusion_size_comparison: Dict[str, Any],
    ) -> Dict[str, Any]:
        rel_dir = os.path.join("pipeline", f"cycle_{cycle_index:02d}", f"exec_{exec_attempt:02d}")
        os.makedirs(self._run_path(rel_dir), exist_ok=True)
        snapshot: Dict[str, Any] = {
            "cycle_index": cycle_index,
            "exec_attempt": exec_attempt,
            "execution_result": execution_result,
            "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "files": {},
            "fusion_size_comparison": fusion_size_comparison if isinstance(fusion_size_comparison, dict) else {},
        }

        pipeline_copy = self._snapshot_file(OUTPUT_DIR + "code/pipeline.py", os.path.join(rel_dir, "pipeline.py"))
        if pipeline_copy:
            snapshot["files"]["pipeline_code"] = pipeline_copy
        fusion_copy = self._snapshot_file("output/data_fusion/fusion_data.csv", os.path.join(rel_dir, "fusion_data.csv"))
        if fusion_copy:
            snapshot["files"]["fusion_data"] = fusion_copy
        estimate_copy = self._snapshot_file(
            "output/pipeline_evaluation/fusion_size_estimate.json",
            os.path.join(rel_dir, "fusion_size_estimate.json"),
        )
        if estimate_copy:
            snapshot["files"]["fusion_size_estimate"] = estimate_copy
        corr_copies = self._snapshot_patterns(
            ["output/correspondences/correspondences_*.csv"],
            os.path.join(rel_dir, "correspondences"),
        )
        if corr_copies:
            snapshot["files"]["correspondences"] = corr_copies

        io_path = self._run_path(rel_dir, "pipeline_execution.json")
        try:
            with open(io_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "execution_result": execution_result,
                        "stdout": stdout or "",
                        "stderr": stderr or "",
                        "captured_at": snapshot["captured_at"],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            snapshot["files"]["execution_log"] = io_path
        except Exception:
            pass
        return snapshot

    def _snapshot_evaluation_attempt(
        self,
        *,
        stage: str,
        cycle_index: int,
        exec_attempt: int,
        execution_result: str,
        stdout: str,
        stderr: str,
        metrics: Dict[str, Any],
        metrics_source: str,
    ) -> Dict[str, Any]:
        safe_stage = str(stage or "unknown").replace("/", "_")
        rel_dir = os.path.join("evaluation", safe_stage, f"cycle_{cycle_index:02d}", f"exec_{exec_attempt:02d}")
        os.makedirs(self._run_path(rel_dir), exist_ok=True)
        snapshot: Dict[str, Any] = {
            "stage": safe_stage,
            "cycle_index": cycle_index,
            "exec_attempt": exec_attempt,
            "execution_result": execution_result,
            "metrics_source": metrics_source,
            "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "metrics": metrics if isinstance(metrics, dict) else {},
            "files": {},
        }

        eval_code_copy = self._snapshot_file(OUTPUT_DIR + "code/evaluation.py", os.path.join(rel_dir, "evaluation.py"))
        if eval_code_copy:
            snapshot["files"]["evaluation_code"] = eval_code_copy
        eval_json_copy = self._snapshot_file(
            "output/pipeline_evaluation/pipeline_evaluation.json",
            os.path.join(rel_dir, "pipeline_evaluation.json"),
        )
        if eval_json_copy:
            snapshot["files"]["pipeline_evaluation"] = eval_json_copy
        debug_copy = self._snapshot_file(
            "output/pipeline_evaluation/debug_fusion_eval.jsonl",
            os.path.join(rel_dir, "debug_fusion_eval.jsonl"),
        )
        if debug_copy:
            snapshot["files"]["debug_fusion_eval"] = debug_copy

        io_path = self._run_path(rel_dir, "evaluation_execution.json")
        try:
            with open(io_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "execution_result": execution_result,
                        "metrics_source": metrics_source,
                        "metrics": metrics if isinstance(metrics, dict) else {},
                        "stdout": stdout or "",
                        "stderr": stderr or "",
                        "captured_at": snapshot["captured_at"],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            snapshot["files"]["execution_log"] = io_path
        except Exception:
            pass
        return snapshot

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

    @staticmethod
    def _is_metrics_payload(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if "overall_accuracy" in payload:
            return True
        acc_keys = [
            key
            for key in payload.keys()
            if isinstance(key, str) and key.endswith("_accuracy") and key not in {"overall_accuracy", "macro_accuracy"}
        ]
        return len(acc_keys) >= 2

    def _extract_metrics_payload(self, payload: Any) -> Dict[str, Any]:
        if self._is_metrics_payload(payload):
            return dict(payload)
        if not isinstance(payload, dict):
            return {}

        for key in (
            "evaluation_metrics",
            "metrics",
            "result",
            "results",
            "final_metrics",
            "pipeline_evaluation",
        ):
            nested = payload.get(key)
            if self._is_metrics_payload(nested):
                return dict(nested)

        for nested in payload.values():
            if isinstance(nested, dict):
                extracted = self._extract_metrics_payload(nested)
                if self._is_metrics_payload(extracted):
                    return extracted
        return {}

    def _extract_metrics_from_text(self, text: Any) -> Dict[str, Any]:
        raw = str(text or "")
        if not raw.strip():
            return {}

        decoder = json.JSONDecoder()
        candidates: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(raw):
            start = raw.find("{", idx)
            if start == -1:
                break
            try:
                parsed, offset = decoder.raw_decode(raw[start:])
            except Exception:
                idx = start + 1
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)
            idx = start + max(1, int(offset))

        if not candidates:
            try:
                parsed = ast.literal_eval(raw.strip())
                if isinstance(parsed, dict):
                    candidates.append(parsed)
            except Exception:
                pass

        def _score(candidate: Dict[str, Any]) -> Tuple[int, int, int]:
            acc_count = sum(1 for k in candidate.keys() if isinstance(k, str) and k.endswith("_accuracy"))
            return (
                1 if "overall_accuracy" in candidate else 0,
                acc_count,
                len(candidate),
            )

        ranked = sorted(candidates, key=_score, reverse=True)
        for candidate in ranked:
            extracted = self._extract_metrics_payload(candidate)
            if self._is_metrics_payload(extracted):
                return extracted
        return {}

    @staticmethod
    def _attribute_accuracy_map(metrics: Dict[str, Any]) -> Dict[str, float]:
        if not isinstance(metrics, dict):
            return {}
        out: Dict[str, float] = {}
        for key, value in metrics.items():
            if not (isinstance(key, str) and key.endswith("_accuracy")):
                continue
            if key in {"overall_accuracy", "macro_accuracy"}:
                continue
            attr = key[: -len("_accuracy")]
            try:
                count = int(float(metrics.get(f"{attr}_count", 0)))
            except Exception:
                count = 0
            if count <= 0:
                continue
            try:
                out[attr] = float(value)
            except Exception:
                continue
        return out

    def _assess_validation_regression(self, current: Dict[str, Any], best: Dict[str, Any]) -> Dict[str, Any]:
        if not (self._is_metrics_payload(current) and self._is_metrics_payload(best)):
            return {"rejected": False, "reason": "no_baseline"}

        current_overall = float(current.get("overall_accuracy", 0.0) or 0.0)
        best_overall = float(best.get("overall_accuracy", 0.0) or 0.0)
        current_macro = float(current.get("macro_accuracy", 0.0) or 0.0)
        best_macro = float(best.get("macro_accuracy", 0.0) or 0.0)
        overall_gain = current_overall - best_overall
        macro_gain = current_macro - best_macro
        overall_regression = max(0.0, best_overall - current_overall)
        macro_regression = max(0.0, best_macro - current_macro)

        current_attr = self._attribute_accuracy_map(current)
        best_attr = self._attribute_accuracy_map(best)
        drops: Dict[str, float] = {}
        gains: Dict[str, float] = {}
        for attr, best_score in best_attr.items():
            if attr not in current_attr:
                continue
            try:
                c_count = int(float(current.get(f"{attr}_count", 0)))
                b_count = int(float(best.get(f"{attr}_count", 0)))
            except Exception:
                c_count = 0
                b_count = 0
            if min(c_count, b_count) < 3:
                continue
            delta = float(current_attr[attr]) - float(best_score)
            if delta < 0.0:
                drops[attr] = round(abs(delta), 6)
            elif delta > 0.0:
                gains[attr] = round(delta, 6)

        catastrophic_drops = {k: v for k, v in drops.items() if v >= 0.2}
        severe_drops = {k: v for k, v in drops.items() if v >= 0.12}
        moderate_drops = {k: v for k, v in drops.items() if v >= 0.08}
        meaningful_gain = overall_gain >= 0.02 or macro_gain >= 0.02
        rejected = (
            overall_regression >= 0.02
            or macro_regression >= 0.02
            or len(catastrophic_drops) >= 1
            or (
                not meaningful_gain
                and (len(severe_drops) >= 1 or len(moderate_drops) >= 2)
            )
        )
        return {
            "rejected": bool(rejected),
            "overall_gain": round(overall_gain, 6),
            "macro_gain": round(macro_gain, 6),
            "overall_drop": round(overall_regression, 6),
            "macro_drop": round(macro_regression, 6),
            "meaningful_gain": bool(meaningful_gain),
            "catastrophic_attribute_drops": catastrophic_drops,
            "severe_attribute_drops": severe_drops,
            "moderate_attribute_drops": moderate_drops,
            "top_attribute_drops": dict(sorted(drops.items(), key=lambda x: x[1], reverse=True)[:8]),
            "top_attribute_gains": dict(sorted(gains.items(), key=lambda x: x[1], reverse=True)[:8]),
        }

    def _is_metrics_better(self, candidate: Dict[str, Any], baseline: Dict[str, Any]) -> bool:
        if not self._is_metrics_payload(candidate):
            return False
        if not self._is_metrics_payload(baseline):
            return True

        eps = 1e-6
        c_overall = float(candidate.get("overall_accuracy", 0.0) or 0.0)
        b_overall = float(baseline.get("overall_accuracy", 0.0) or 0.0)
        if c_overall > b_overall + eps:
            return True
        if c_overall < b_overall - eps:
            return False

        c_macro = float(candidate.get("macro_accuracy", 0.0) or 0.0)
        b_macro = float(baseline.get("macro_accuracy", 0.0) or 0.0)
        if c_macro > b_macro + eps:
            return True
        if c_macro < b_macro - eps:
            return False

        c_attr = self._attribute_accuracy_map(candidate)
        b_attr = self._attribute_accuracy_map(baseline)
        shared = [attr for attr in c_attr.keys() if attr in b_attr]
        if not shared:
            return False
        c_mean = sum(float(c_attr[attr]) for attr in shared) / max(1, len(shared))
        b_mean = sum(float(b_attr[attr]) for attr in shared) / max(1, len(shared))
        return c_mean > b_mean + eps

    def _expected_dataset_pairs(self, state: SimpleModelAgentState) -> List[Tuple[str, str]]:
        pairs: set[Tuple[str, str]] = set()
        testsets = state.get("entity_matching_testsets", {}) if isinstance(state, dict) else {}
        if isinstance(testsets, dict) and testsets:
            for key in testsets.keys():
                if isinstance(key, (tuple, list)) and len(key) == 2:
                    left = str(key[0]).strip().lower()
                    right = str(key[1]).strip().lower()
                    if left and right and left != right:
                        pairs.add(tuple(sorted((left, right))))
        if pairs:
            return sorted(list(pairs))

        dataset_names = []
        for path in state.get("datasets", []) if isinstance(state, dict) else []:
            try:
                dataset_names.append(os.path.splitext(os.path.basename(str(path)))[0].strip().lower())
            except Exception:
                continue
        dataset_names = sorted(list({n for n in dataset_names if n}))
        for i in range(len(dataset_names)):
            for j in range(i + 1, len(dataset_names)):
                pairs.add((dataset_names[i], dataset_names[j]))
        return sorted(list(pairs))

    def _resolve_correspondence_file(self, left: str, right: str) -> str:
        candidates = [
            os.path.join("output", "correspondences", f"correspondences_{left}_{right}.csv"),
            os.path.join("output", "correspondences", f"correspondences_{right}_{left}.csv"),
        ]
        existing = [path for path in candidates if os.path.exists(path)]
        if not existing:
            return ""
        if len(existing) == 1:
            return existing[0]
        existing.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return existing[0]

    @staticmethod
    def _csv_data_row_count(path: str) -> int:
        if not path or not os.path.exists(path):
            return 0
        rows = 0
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    if line.strip():
                        rows += 1
        except Exception:
            return 0
        return rows

    def _evaluate_correspondence_integrity(self, state: SimpleModelAgentState) -> Dict[str, Any]:
        pair_checks: List[Dict[str, Any]] = []
        invalid_pairs: List[str] = []
        for left, right in self._expected_dataset_pairs(state):
            pair_key = f"{left}_{right}"
            path = self._resolve_correspondence_file(left, right)
            if not path:
                pair_checks.append({"pair": pair_key, "status": "missing", "path": "", "data_rows": 0})
                invalid_pairs.append(pair_key)
                continue
            rows = self._csv_data_row_count(path)
            status = "ok" if rows > 0 else "empty"
            pair_checks.append({"pair": pair_key, "status": status, "path": path, "data_rows": rows})
            if status != "ok":
                invalid_pairs.append(pair_key)

        return {
            "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "expected_pairs": [f"{l}_{r}" for l, r in self._expected_dataset_pairs(state)],
            "pair_checks": pair_checks,
            "invalid_pairs": sorted(list(dict.fromkeys(invalid_pairs))),
            "structurally_valid": len(invalid_pairs) == 0,
        }

    def _update_pipeline_matching_thresholds(self,
        pipeline_code: str, matching_config: Dict[str, Any]
    ) -> str:
        """
        Updates the matching thresholds in the generated pipeline code based on the provided
        matching configuration. This function is designed to be called after the LLM
        generates the initial pipeline code.
        """
        if not matching_config or "matching_strategies" not in matching_config:
            return pipeline_code

        updated_code = pipeline_code
        strategies = matching_config["matching_strategies"]

        for pair_key, config in strategies.items():
            if isinstance(config, dict) and "threshold" in config:
                threshold_value = config["threshold"]

                # The LLM will be instructed to generate variable names like threshold_discogs_lastfm.
                # So, the var_name directly corresponds to the pair_key with 'threshold_' prefix.
                var_name = f"threshold_{pair_key}"

                # Pattern to find a variable assignment like 'threshold_discogs_lastfm = 0.7'
                # The regex handles potential whitespace around '=' and different float formats.
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
        """
        Apply safe, generic guardrails based on diagnostics.
        Keeps logic dataset-agnostic and only touches well-defined flags.
        """
        try:
            from helpers.code_guardrails import apply_pipeline_guardrails as helper_apply_pipeline_guardrails
        except Exception:
            try:
                from agents.helpers.code_guardrails import (
                    apply_pipeline_guardrails as helper_apply_pipeline_guardrails,
                )
            except Exception:
                helper_apply_pipeline_guardrails = None
        if callable(helper_apply_pipeline_guardrails):
            try:
                return helper_apply_pipeline_guardrails(pipeline_code, state if isinstance(state, dict) else {})
            except Exception as helper_err:
                print(f"[WARN] helper apply_pipeline_guardrails failed, using inline fallback: {helper_err}")

        updated_code = pipeline_code
        diagnostics = state.get("auto_diagnostics", {}) if isinstance(state, dict) else {}
        metrics = state.get("evaluation_metrics", {}) if isinstance(state, dict) else {}

        id_alignment = diagnostics.get("id_alignment", {}) if isinstance(diagnostics, dict) else {}
        mapped_ratio = id_alignment.get("mapped_coverage_ratio")
        debug_ratios = diagnostics.get("debug_reason_ratios", {}) if isinstance(diagnostics, dict) else {}
        missing_ratio = debug_ratios.get("missing_fused_value", 0.0)
        overall_acc = metrics.get("overall_accuracy")

        force_singletons = False
        try:
            if mapped_ratio is not None and float(mapped_ratio) < 0.85:
                force_singletons = True
        except Exception:
            pass
        try:
            if missing_ratio is not None and float(missing_ratio) > 0.20:
                force_singletons = True
        except Exception:
            pass
        try:
            if overall_acc is not None and float(overall_acc) < 0.60:
                force_singletons = True
        except Exception:
            pass

        if force_singletons:
            updated_code, replaced = re.subn(
                r"(include_singletons\s*=\s*)False",
                r"\1True",
                updated_code,
                count=1,
            )
            if replaced:
                print("[GUARDRAIL] Forced include_singletons=True due to coverage diagnostics.")

        # Ensure custom fusers are PyDI-compatible when runtime kwargs (e.g., sources) are passed.
        def _patch_custom_fuser_signature(match: re.Match) -> str:
            indent = match.group("indent")
            fn_name = match.group("name")
            params = match.group("params")
            if "fuser" not in fn_name.lower() or "**kwargs" in params:
                return match.group(0)

            param_tokens = [p.strip() for p in params.split(",") if p.strip()]
            if len(param_tokens) < 2:
                return match.group(0)
            if param_tokens[0] != "inputs" or param_tokens[1] != "context":
                return match.group(0)

            return f"{indent}def {fn_name}({params}, **kwargs):"

        updated_code, fuser_sig_updates = re.subn(
            r"(?P<indent>^[ \t]*)def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<params>[^\)]*)\)\s*:",
            _patch_custom_fuser_signature,
            updated_code,
            flags=re.MULTILINE,
        )
        if fuser_sig_updates:
            print(f"[GUARDRAIL] Updated {fuser_sig_updates} custom fuser signature(s) to accept **kwargs.")

        updated_code, lambda_updates = re.subn(
            r"lambda\s+inputs\s*,\s*context\s*:",
            "lambda inputs, **kwargs:",
            updated_code,
        )
        if lambda_updates:
            print(f"[GUARDRAIL] Updated {lambda_updates} lambda fuser signature(s) to accept **kwargs.")

        # Make local helper imports robust across execution directories.
        list_norm_import = "from list_normalization import detect_list_like_columns, normalize_list_like_columns"
        if list_norm_import in updated_code and "except ModuleNotFoundError" not in updated_code:
            robust_import_block = """
import os
import sys
try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _module_name = "list_normalization.py"
    _candidates = []
    try:
        _candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    except Exception:
        pass
    _candidates.append(os.getcwd())
    _candidates.append(os.path.abspath(os.path.join(os.getcwd(), "agents")))
    for _path in _candidates:
        if os.path.isfile(os.path.join(_path, _module_name)) and _path not in sys.path:
            sys.path.append(_path)
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
""".strip()
            updated_code = updated_code.replace(list_norm_import, robust_import_block, 1)
            print("[GUARDRAIL] Rewrote list_normalization import to robust fallback block.")

        # Ensure NumericComparator has explicit list_strategy for list-valued numeric fields.
        def _ensure_numeric_list_strategy(match: re.Match) -> str:
            block = match.group(0)
            if "list_strategy" in block:
                return block
            return re.sub(
                r"\\n([ \\t]*)\\)$",
                r"\\n\\1    list_strategy=\\\"average\\\",\\n\\1)",
                block,
                count=1,
            )

        updated_code, numeric_list_updates = re.subn(
            r"NumericComparator\\(\\n(?:[ \\t]+[^\\n]*\\n)+?[ \\t]*\\)",
            _ensure_numeric_list_strategy,
            updated_code,
            flags=re.MULTILINE,
        )
        if numeric_list_updates:
            print(f"[GUARDRAIL] Added list_strategy to {numeric_list_updates} NumericComparator call(s).")

        # Ensure StringComparator has explicit list_strategy when omitted.
        # Prevents runtime comparator failures if the compared values are list-like.
        def _ensure_string_list_strategy(match: re.Match) -> str:
            block = match.group(0)
            if "list_strategy" in block:
                return block
            strategy = "concatenate"
            try:
                if re.search(r"similarity_function\\s*=\\s*[\"\\']jaccard[\"\\']", block):
                    strategy = "set_jaccard"
            except Exception:
                strategy = "concatenate"
            return re.sub(
                r"\\n([ \\t]*)\\)$",
                rf"\\n\\1    list_strategy=\\\"{strategy}\\\",\\n\\1)",
                block,
                count=1,
            )

        updated_code, string_list_updates = re.subn(
            r"StringComparator\\(\\n(?:[ \\t]+[^\\n]*\\n)+?[ \\t]*\\)",
            _ensure_string_list_strategy,
            updated_code,
            flags=re.MULTILINE,
        )
        if string_list_updates:
            print(f"[GUARDRAIL] Added list_strategy to {string_list_updates} StringComparator call(s).")

        # Patch unsafe `if pd.isna(x):` checks that crash on array/list-like values.
        def _patch_unsafe_pd_isna(match: re.Match) -> str:
            indent = match.group("indent")
            return (
                f"{indent}try:\\n"
                f"{indent}    _is_na = pd.isna(x)\\n"
                f"{indent}    if isinstance(_is_na, (list, tuple, set, dict)) or hasattr(_is_na, '__array__'):\\n"
                f"{indent}        _is_na = False\\n"
                f"{indent}    else:\\n"
                f"{indent}        _is_na = bool(_is_na)\\n"
                f"{indent}except Exception:\\n"
                f"{indent}    _is_na = False\\n"
                f"{indent}if _is_na:\\n"
                f"{indent}    return np.nan"
            )

        updated_code, pd_isna_updates = re.subn(
            r"(?m)^(?P<indent>[ \\t]*)if pd\\.isna\\(x\\):\\n(?P=indent)[ \\t]*return np\\.nan",
            _patch_unsafe_pd_isna,
            updated_code,
        )
        if pd_isna_updates:
            print(f"[GUARDRAIL] Hardened {pd_isna_updates} unsafe pd.isna(x) check(s).")

        return updated_code

    def _compute_auto_diagnostics(self, state: SimpleModelAgentState, metrics: Dict[str, Any]) -> Dict[str, Any]:
        from helpers.evaluation_helpers import (
            compute_auto_diagnostics as helper_compute_auto_diagnostics,
            compute_id_alignment as helper_compute_id_alignment,
        )

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
        )

    def _assess_normalization_with_llm(self, state: SimpleModelAgentState) -> Dict[str, Any]:
        from helpers.evaluation_helpers import extract_json_object as helper_extract_json_object
        from helpers.normalization_policy import (
            collect_eval_debug_signals as helper_collect_eval_debug_signals,
            infer_country_output_format_from_validation as helper_infer_country_output_format_from_validation,
            infer_validation_text_case_map as helper_infer_validation_text_case_map,
        )

        default_system_prompt = """
        You are an investigation assistant deciding whether data normalization should be rerun.

        Focus on canonicalization issues such as:
        - country naming mismatches (e.g., long-form vs short-form country names)
        - list formatting inconsistencies (JSON-string lists vs scalar strings vs nested structures)
        - whitespace/casing/format inconsistencies that break exact comparisons

        Use evidence from metrics, diagnostics and reasoning text only.

        Return STRICT JSON (no markdown) with keys:
        {
          "needs_normalization": bool,
          "reasons": [str],
          "country_columns": [str],
          "list_columns": [str],
          "text_columns": [str],
          "lowercase_columns": [str],
          "country_output_format": "alpha_2|alpha_3|numeric|name|official_name",
          "list_normalization_required": bool,
          "recommendation_summary": str,
          "confidence": float
        }

        Keep lists short and only include columns that are clearly indicated by evidence.
        """
        try:
            from prompts.investigator_prompt import NORMALIZATION_ASSESSMENT_SYSTEM_PROMPT
        except Exception:
            NORMALIZATION_ASSESSMENT_SYSTEM_PROMPT = None
        system_prompt = (
            NORMALIZATION_ASSESSMENT_SYSTEM_PROMPT
            if isinstance(NORMALIZATION_ASSESSMENT_SYSTEM_PROMPT, str) and NORMALIZATION_ASSESSMENT_SYSTEM_PROMPT.strip()
            else default_system_prompt
        )

        validation_style: Dict[str, Any] = {}
        try:
            validation_path = self._evaluation_testset_path(state, force_test=False)
            validation_df = load_dataset(validation_path) if validation_path else None
            case_map = helper_infer_validation_text_case_map(validation_df)
            inferred_country_format = helper_infer_country_output_format_from_validation(validation_df)
            validation_style = {
                "validation_path": validation_path,
                "inferred_country_output_format": inferred_country_format,
                "lowercase_columns_hint": [
                    c for c, meta in case_map.items() if bool(meta.get("prefer_lowercase"))
                ],
                "case_profile": case_map,
            }
        except Exception:
            validation_style = {}

        debug_signals = helper_collect_eval_debug_signals()

        human_content = f"""
        Evaluation metrics:
        {json.dumps(state.get("evaluation_metrics", {}), indent=2)}

        Auto diagnostics:
        {json.dumps(state.get("auto_diagnostics", {}), indent=2)}

        Integration diagnostics report:
        {json.dumps(state.get("integration_diagnostics_report", {}), indent=2)}

        Evaluation reasoning summary:
        {state.get("evaluation_analysis", "")}

        Data profiles:
        {json.dumps(state.get("data_profiles", {}), indent=2)}

        Validation style hints (use these to avoid over-normalization):
        {json.dumps(validation_style, indent=2)}

        Debug mismatch style signals:
        {json.dumps(debug_signals, indent=2)}
        """

        message = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]
        result = self._invoke_with_usage(message, "investigator_normalization_assessment")
        parsed = helper_extract_json_object(result)

        if not parsed:
            inferred_country_format = str(validation_style.get("inferred_country_output_format", "name"))
            lowercase_hint = validation_style.get("lowercase_columns_hint", [])
            if not isinstance(lowercase_hint, list):
                lowercase_hint = []
            return {
                "needs_normalization": False,
                "reasons": [],
                "country_columns": [],
                "list_columns": [],
                "text_columns": [],
                "lowercase_columns": [str(c) for c in lowercase_hint if str(c).strip()],
                "country_output_format": inferred_country_format,
                "list_normalization_required": False,
                "recommendation_summary": "",
                "confidence": 0.0,
            }

        for key in ("reasons", "country_columns", "list_columns", "text_columns", "lowercase_columns"):
            value = parsed.get(key, [])
            if not isinstance(value, list):
                parsed[key] = []
            else:
                parsed[key] = [str(v) for v in value if str(v).strip()]

        parsed["needs_normalization"] = bool(parsed.get("needs_normalization", False))
        parsed["list_normalization_required"] = bool(parsed.get("list_normalization_required", False))
        allowed_country_formats = {"alpha_2", "alpha_3", "numeric", "name", "official_name"}
        inferred_country_format = str(validation_style.get("inferred_country_output_format", "name"))
        proposed_country_format = str(parsed.get("country_output_format", inferred_country_format)).strip()
        if proposed_country_format not in allowed_country_formats:
            proposed_country_format = inferred_country_format if inferred_country_format in allowed_country_formats else "name"
        parsed["country_output_format"] = proposed_country_format
        try:
            parsed["confidence"] = float(parsed.get("confidence", 0.0))
        except Exception:
            parsed["confidence"] = 0.0
        parsed["recommendation_summary"] = str(parsed.get("recommendation_summary", ""))

        return parsed

    def _apply_evaluation_guardrails(self, evaluation_code: str) -> str:
        """
        Patch common portability/runtime issues in generated evaluation code.
        """
        try:
            from helpers.code_guardrails import apply_evaluation_guardrails as helper_apply_evaluation_guardrails
        except Exception:
            helper_apply_evaluation_guardrails = None
        if callable(helper_apply_evaluation_guardrails):
            try:
                return helper_apply_evaluation_guardrails(evaluation_code)
            except Exception as helper_err:
                print(f"[WARN] helper apply_evaluation_guardrails failed, using inline fallback: {helper_err}")

        updated = evaluation_code
        list_norm_import = "from list_normalization import detect_list_like_columns, normalize_list_like_columns"
        if list_norm_import in updated and "except ModuleNotFoundError" not in updated:
            robust_import_block = """
import os
import sys
from pathlib import Path
try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _module_name = "list_normalization.py"
    _candidates = [
        Path.cwd(),
        Path.cwd() / "agents",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
    ]
    for _path in _candidates:
        if (_path / _module_name).is_file():
            _path_str = str(_path.resolve())
            if _path_str not in sys.path:
                sys.path.append(_path_str)
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
""".strip()
            updated = updated.replace(list_norm_import, robust_import_block, 1)
            print("[GUARDRAIL] Rewrote evaluation list_normalization import to robust fallback block.")
        return updated

    def should_continue_research(self, state: SimpleModelAgentState) -> str:
        """Determines whether to continue the research loop or end."""
        messages = state['messages']
        last_message = messages[-1]
        # If the LLM made a tool call, continue the loop to process it.
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            # Guardrail: avoid endless tool-call loops if the model keeps requesting tools.
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
        # If the LLM responded with code (no tool call), end the loop.
        else:
            return "end"

    # Creates tool calls to profile the data and saves it into agent state 
    def match_schemas(self, state: SimpleModelAgentState):
        if state.get("run_id") == self.run_id and state.get("run_output_root") == self.run_output_root:
            run_updates = {
                "run_id": self.run_id,
                "run_output_root": self.run_output_root,
                "enable_cross_run_memory": self._cross_run_memory_enabled(state),
            }
        else:
            run_updates = self._reset_run_context(state)
        self._log_action("match_schemas", "start", "Align dataset schemas before blocking/matching", "Improves comparator compatibility", {"datasets": state.get("datasets")})

        self.logger.info("----------------------- Entering match_schemas -----------------------")
        if state.get("schema_correspondences"):
            print("[*] Schema correspondences already present; skipping match_schemas")
            return {}

        print("[*] Running schema matching")
        result = run_schema_matching(
            dataset_paths=state["datasets"],
            model=self.base_model,
            output_dir="output/schema-matching",
        )
        self.logger.info("Leaving match_schemas")
        out = dict(result) if isinstance(result, dict) else {}
        out.update(run_updates)
        return out

    def profile_data(self, state:SimpleModelAgentState):
        self._log_action("profile_data", "start", "Profile datasets to guide feature/threshold choices", "Improves matching signal selection", {"datasets": state.get("datasets")})

        self.logger.info('----------------------- Entering profile_data -----------------------')

        print("[*] Profiling datasets")

        system_prompt = """
            You are a data scientist tasked with the integration of several datasets.
            For each dataset path provided, call the tool `profile_dataset` with the path
            (one tool call per dataset).
        """
        
        datasets_list_str = "\n".join(state['datasets'])
        human_content = f"Please profile these datasets (one call per dataset):\n{datasets_list_str}"
        message = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
        self.logger.info("Input Message:" + str(message))
        
        result = self._invoke_with_usage(message, "profile_data")
        self.logger.info("RESULT:" + str(result))

        # call tools
        tool_calls = result.tool_calls

        self.logger.info("Tool Calls:" + str(tool_calls))
        results = {}
        for t in tool_calls:
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                self.logger.info("adapt_pipeline: ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            
#            if USE_LLM == "groq" or USE_LLM == "gpt" or USE_LLM == "gemini":
            results[t['args']['path']] = result
#            elif USE_LLM == "gemini_broken":
#                results[t['args']['__arg1']] = result

        with open(OUTPUT_DIR + "profile/profiles.json", "w") as file:
            file.write(json.dumps(results, indent=2))
          
        self.logger.info('Leaving profile_data')
        return {'data_profiles': results}

    def run_blocking_tester(self, state: SimpleModelAgentState):
        self._log_action("run_blocking_tester", "start", "Select blocking strategy to reduce candidates", "Improves runtime and recall balance", {"skip": SKIP_BLOCKING_TESTER})

        self.logger.info("----------------------- Entering run_blocking_tester -----------------------")

        cross_run_memory_enabled = self._cross_run_memory_enabled(state)
        if SKIP_BLOCKING_TESTER and cross_run_memory_enabled:
            path = Path(OUTPUT_DIR + "blocking-evaluation/blocking_config.json")
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    blocking_config = json.load(f)
                    cfg_names = blocking_config.get("dataset_names", []) if isinstance(blocking_config, dict) else []
                    expected = sorted([Path(p).stem for p in state.get("datasets", [])])
                    got = sorted([str(x) for x in cfg_names]) if isinstance(cfg_names, list) else []
                    if got and got != expected:
                        print("[WARN] Saved blocking config dataset signature mismatch; recomputing blocking.")
                    else:
                        print("[+] Using saved blocking strategy: " + json.dumps(blocking_config, indent=2))
                        return {"blocking_config": blocking_config}
            else:
                print("[-] Cannot skip blocking tester. No saved blocking strategy found")
        elif SKIP_BLOCKING_TESTER:
            print("[*] Cross-run memory disabled; ignoring saved blocking strategy and recomputing.")
        
        if state.get("blocking_config"):
            cfg = state.get("blocking_config", {})
            cfg_names = cfg.get("dataset_names", []) if isinstance(cfg, dict) else []
            expected = sorted([Path(p).stem for p in state.get("datasets", [])])
            got = sorted([str(x) for x in cfg_names]) if isinstance(cfg_names, list) else []
            if got and got != expected:
                print("[WARN] State blocking config dataset signature mismatch; recomputing blocking.")
            else:
                print("[*] Blocking config already present in state; skipping BlockingTester")
                return {"blocking_config": cfg}

        print("[*] Running BlockingTester")
        tester = BlockingTester(
            llm=self.base_model,
            datasets=state["datasets"],
            max_candidates=350000,
            blocking_testsets=state['entity_matching_testsets'],
            output_dir="output/blocking-evaluation",
            pc_threshold=0.9,
            max_attempts=5,
            max_error_retries=2,
            verbose=True
        )
        _, blocking_config = tester.run_all_pairs()
        print("[*] Blocking config computed:\n" + json.dumps(blocking_config, indent=2))
        self.logger.info("Leaving run_blocking_tester")
        return {"blocking_config": blocking_config}

    def run_matching_tester(self, state: SimpleModelAgentState):
        self._log_action("run_matching_tester", "start", "Select matching strategy and thresholds", "Improves correspondence quality", {"skip": SKIP_MATCHING_TESTER, "matcher_mode": state.get("matcher_mode")})

        self.logger.info("----------------------- Entering run_matching_tester -----------------------")
        matcher_mode = str(state.get("matcher_mode", "ml")).strip().lower().replace("-", "_")
        if matcher_mode == "rulebased":
            matcher_mode = "rule_based"
        state["matcher_mode"] = matcher_mode

        def _config_has_list_based_comparators(cfg: Dict[str, Any]) -> bool:
            try:
                strategies = cfg.get("matching_strategies", {}) if isinstance(cfg, dict) else {}
                for _, pair_cfg in strategies.items():
                    comps = pair_cfg.get("comparators", []) if isinstance(pair_cfg, dict) else []
                    for comp in comps:
                        if bool((comp or {}).get("list_strategy")):
                            return True
                return False
            except Exception:
                return False

        def _config_matches_datasets(cfg: Dict[str, Any]) -> bool:
            try:
                if not isinstance(cfg, dict):
                    return False
                cfg_names = cfg.get("dataset_names", [])
                if not isinstance(cfg_names, list) or not cfg_names:
                    return True
                expected = sorted([Path(p).stem for p in state.get("datasets", [])])
                got = sorted([str(x) for x in cfg_names])
                return expected == got
            except Exception:
                return False

        def _matching_config_needs_refresh(cfg: Dict[str, Any]) -> tuple[bool, str]:
            if not isinstance(cfg, dict):
                return True, "matching config is not a dict"
            strategies = cfg.get("matching_strategies", {})
            if not isinstance(strategies, dict) or not strategies:
                return True, "matching strategies are missing"
            weak_pairs = []
            for pair_name, pair_cfg in strategies.items():
                if not isinstance(pair_cfg, dict):
                    weak_pairs.append(f"{pair_name}:invalid")
                    continue
                try:
                    f1 = float(pair_cfg.get("f1", 0.0) or 0.0)
                except Exception:
                    f1 = 0.0
                failure_tags = {str(tag) for tag in pair_cfg.get("failure_tags", []) if str(tag).strip()}
                if "low_matching_quality" in failure_tags or f1 < 0.75:
                    weak_pairs.append(f"{pair_name}:F1={f1:.3f}")
            if weak_pairs:
                return True, "low-quality matching strategies present (" + ", ".join(weak_pairs[:5]) + ")"
            return False, ""

        cross_run_memory_enabled = self._cross_run_memory_enabled(state)
        if SKIP_MATCHING_TESTER and cross_run_memory_enabled:
            path = Path(OUTPUT_DIR + "matching-evaluation/matching_config.json")
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    matching_config = json.load(f)
                    if not _config_matches_datasets(matching_config):
                        print("[WARN] Saved matching config dataset signature mismatch; recomputing matcher.")
                    elif _config_has_list_based_comparators(matching_config):
                        print("[WARN] Saved matching config contains list-based comparators; recomputing matcher.")
                    else:
                        needs_refresh, refresh_reason = _matching_config_needs_refresh(matching_config)
                        if needs_refresh:
                            print(f"[WARN] Saved matching config needs refresh; recomputing matcher ({refresh_reason}).")
                        else:
                            print("[+] Using saved matching strategy: " + json.dumps(matching_config, indent=2))
                            return {"matching_config": matching_config}
            else:
                print("[-] Cannot skip matching tester. No saved matching strategy found")
        elif SKIP_MATCHING_TESTER:
            print("[*] Cross-run memory disabled; ignoring saved matching strategy and recomputing.")
        
        if state.get("matching_config"):
            existing_cfg = state.get("matching_config", {})
            if not _config_matches_datasets(existing_cfg):
                print("[WARN] State matching config dataset signature mismatch; recomputing matcher.")
            elif _config_has_list_based_comparators(existing_cfg):
                print("[WARN] State matching config contains list-based comparators; recomputing matcher.")
            else:
                needs_refresh, refresh_reason = _matching_config_needs_refresh(existing_cfg)
                if needs_refresh:
                    print(f"[WARN] State matching config needs refresh; recomputing matcher ({refresh_reason}).")
                else:
                    print("[*] Matching config already present in state; skipping MatchingTester")
                    return {"matching_config": existing_cfg, "matcher_mode": matcher_mode}

        # Always reload matching_tester from disk so notebook sessions do not use stale class code.
        try:
            import importlib
            import matching_tester as _matching_tester_module
            _matching_tester_module = importlib.reload(_matching_tester_module)
            RuntimeMatchingTester = _matching_tester_module.MatchingTester
            if hasattr(_matching_tester_module, "__file__"):
                print(f"[*] Using MatchingTester module: {_matching_tester_module.__file__}")
        except Exception as e:
            print(f"[WARN] Could not reload matching_tester module: {e}")
            RuntimeMatchingTester = MatchingTester

        print("[*] Running MatchingTester")
        tester = RuntimeMatchingTester(
            llm=self.base_model,
            datasets=state["datasets"],
            matching_testsets=state['entity_matching_testsets'],
            blocking_config=state.get("blocking_config", {}),
            output_dir="output/matching-evaluation",
            f1_threshold=0.75,
            max_attempts=8,
            max_error_retries=2,
            verbose=True,
            matcher_mode=matcher_mode,
            disallow_list_comparators=True,
            no_gain_patience=4,
        )
        _, matching_config = tester.run_all()
        print("[*] Matching config computed:\n" + json.dumps(matching_config, indent=2))
        self.logger.info("Leaving run_matching_tester")
        return {"matching_config": matching_config, "matcher_mode": matcher_mode}


    def pipeline_adaption(self, state: SimpleModelAgentState):
        self._log_action("pipeline_adaption", "start", "Generate integration pipeline code", "Incorporates configs and prior feedback")

        self.logger.info('----------------------- Entering pipeline_adaption -----------------------')
        try:
            from prompts.pipeline_prompt import (
                PIPELINE_NORMALIZATION_RULES_BLOCK,
                PIPELINE_MATCHING_SAFETY_RULES_BLOCK,
            )
        except Exception:
            PIPELINE_NORMALIZATION_RULES_BLOCK = (
                "IMPORTANT NORMALIZATION RULES:\n"
                "- Validation set style drives normalization. Do NOT blindly lowercase text globally.\n"
                "- Keep IDs unchanged (case and content must be preserved exactly).\n"
                "- If you convert a column to list-like values, any comparator on that column MUST set list_strategy.\n"
                "- If StringComparator is used on potentially list-like columns (artist/name/tracks/etc.), set list_strategy explicitly.\n"
                "- Prefer normalization that aligns fused output with validation representation, not a random canonical style."
            )
            PIPELINE_MATCHING_SAFETY_RULES_BLOCK = (
                "MATCHING/COMPARATOR SAFETY RULES (MANDATORY):\n"
                "- Never leave StringComparator list_strategy unspecified when values may be lists.\n"
                "- Never leave NumericComparator list_strategy unspecified when values may be lists.\n"
                "- Avoid introducing preprocessing that changes representation style away from validation set conventions unless diagnostics explicitly justify it."
            )

        ####### PREPARE SYSTEM PROMT #######

        # Sync matching_config from disk only when explicitly allowed.
        matching_path = os.path.join(OUTPUT_DIR, "matching-evaluation", "matching_config.json")
        if self._cross_run_memory_enabled(state) and os.path.exists(matching_path):
            with open(matching_path, "r", encoding="utf-8") as f:
                disk_matching_config = json.load(f)
            cfg_names = disk_matching_config.get("dataset_names", []) if isinstance(disk_matching_config, dict) else []
            expected = sorted([Path(p).stem for p in state.get("datasets", [])])
            got = sorted([str(x) for x in cfg_names]) if isinstance(cfg_names, list) else []
            if not got or got == expected:
                state["matching_config"] = disk_matching_config
            else:
                print("[WARN] Ignoring disk matching_config due to dataset signature mismatch.")
    
        # Load example pipeline code
        matcher_mode = str(state.get("matcher_mode", "ml")).strip().lower().replace("-", "_")
        if matcher_mode == "rulebased":
            matcher_mode = "rule_based"
        state["matcher_mode"] = matcher_mode
        example_name = "example_pipeline_ml.py" if matcher_mode == "ml" else "example_pipeline.py"
        example_pipeline_path = os.path.join(INPUT_DIR, f"example_pipelines/{example_name}")
        if not os.path.exists(example_pipeline_path):
            raise FileNotFoundError(f"Example pipeline not found at: {example_pipeline_path}")
    
        with open(example_pipeline_path, "r", encoding="utf-8") as f:
            example_pipeline_code = f.read()

        # Prepare first-row previews for each dataset
        dataset_previews = {}
        for path in state['datasets']:
            df = load_dataset(path)
            if df is not None and not df.empty:
                # Take only first row and convert to dictionary
                dataset_previews[path] = df.iloc[0].to_dict()
            else:
                dataset_previews[path] = "Failed to load or empty dataset"
                
        # Prepare prompt for the model
        entity_matching_section = ""

        if matcher_mode == "ml":
            entity_matching_section = f"""
            2b. Entity matching testsets paths:
            {state["entity_matching_testsets"]}
            """
        
        system_prompt = f"""
        You are a data scientist tasked with the integration of several datasets.
        You are provided with the following inputs:

        1. An example integration pipeline (Python code using PyDI library). Pay
        attention to the comments within the pipeline, as they also contain important instructions:
        {example_pipeline_code}
    
        2. A list of dataset file paths:
        {json.dumps(state['datasets'], indent=2)}

        {entity_matching_section}
    
        3. The first row of each dataset to help understand the structure:
        {json.dumps(dataset_previews, indent=2)}
    
        4. A dictionary containing the profile data of the datasets
        (including number of rows, nulls_per_column and dtypes of 
        the columns):
        {json.dumps(state['data_profiles'], indent=2)}
        """

        normalization_context = {
            "normalization_execution_result": state.get("normalization_execution_result", ""),
            "normalization_attempts": state.get("normalization_attempts", 0),
            "normalization_directives": state.get("normalization_directives", {}),
            "normalization_report": state.get("normalization_report", {}),
        }
        system_prompt += f"""

        4b. NORMALIZATION CONTEXT (important for robust generation):
        {json.dumps(normalization_context, indent=2)}

        {PIPELINE_NORMALIZATION_RULES_BLOCK}
        """
        
        # Include blocking config if available (from BlockingTester)
        blocking_config = state.get('blocking_config')
        if blocking_config:
            system_prompt += f"""

        5. **BLOCKING CONFIGURATION** (pre-computed optimal blocking strategies):
        This configuration was determined by a blocking evaluation agent. Use these settings
        for your blocking step in the pipeline:
        {json.dumps(blocking_config, indent=2)}
        
        IMPORTANT: Use the id_columns and blocking_strategies from this config:
        - Use the correct id_column for each dataset as specified
        - Use the recommended strategy (exact_match_single/multi or semantic_similarity)
        - Use the recommended columns for blocking
        - Strategy to blocker mapping:
          * exact_match_single / exact_match_multi -> StandardBlocker (exact match on columns)
          * token_blocking / ngram_blocking -> TokenBlocker (token or n-gram blocking)
          * sorted_neighbourhood -> SortedNeighbourhoodBlocker (window)
          * semantic_similarity -> EmbeddingBlocker (top_k)
        """

        # Include matching config if available (from MatchingTester)
        matching_config = state.get('matching_config')
        if matching_config:
            if matcher_mode == "ml":
                system_prompt += f"""

        6. **MATCHING CONFIGURATION** (pre-computed comparator settings):
        This configuration was determined by a matching evaluation agent. Use these settings
        for your MLBasedMatcher feature extraction and training:
        {json.dumps(matching_config, indent=2)}

        IMPORTANT: Use the matching_strategies from this config:
        - Build comparators (StringComparator/NumericComparator/DateComparator) for each dataset pair
        - Use the comparators as features in FeatureExtractor
        - Train a classifier on the labeled pairs (labels in the testset) and use MLBasedMatcher
        - Do NOT set weights; ML model learns weights internally
        - Do NOT add RuleBasedMatcher fallback branches
        - Follow the example ML pipeline structure and naming closely; do not invent new variable roles
        - Gold correspondence files can use different pair-id column names (e.g., id1/id2, id_a/id_b). Infer pair columns dynamically and normalize them before feature extraction.
        - preprocess mapping: "lower" -> str.lower, "strip" -> str.strip, "lower_strip" -> lambda x: str(x).lower().strip()

        - For NumericComparator on potentially list-valued columns (e.g., duration fields), set `list_strategy` explicitly (recommended: "average") or safely scalarize before matching.
        """
            else:
                system_prompt += f"""

        6. **MATCHING CONFIGURATION** (pre-computed comparator settings):
        This configuration was determined by a matching evaluation agent. Use these settings
        for your RuleBasedMatcher step in the pipeline:
        {json.dumps(matching_config, indent=2)}

        IMPORTANT: Use the matching_strategies from this config:
        - Build comparators (StringComparator/NumericComparator/DateComparator) for each dataset pair
        - Use the specified weights and threshold.
        - For each pair in `matching_strategies`, you **MUST** define a
            variable named `threshold_[original_pair_key]` (e.g.,
            `threshold_discogs_lastfm = 0.7`) with the corresponding threshold value.
            Ensure your RuleBasedMatcher instances use these variables for their
            `threshold` parameter.
        - preprocess mapping: "lower" -> str.lower, "strip" -> str.strip, "lower_strip" -> lambda x: str(x).lower().strip()

        - For NumericComparator on potentially list-valued columns (e.g., duration fields), set `list_strategy` explicitly (recommended: "average") or safely scalarize before matching.
        """

        system_prompt += """

        Your task is to **create a similar integration pipeline** so that it works with
        the datasets provided. Output should only consist of the relevant Python code
        for the integration pipeline.

        FUSION RULES (MANDATORY):
        - Use PyDI built-in fusers as the default and preferred choice.
        - Do NOT define custom fusion resolvers or lambda resolvers unless absolutely unavoidable.
        - If unavoidable, custom fusers must be minimal, PyDI-compatible, and justified by diagnostics evidence.
        - Prefer this built-in mapping by inferred attribute type:
          * string-like scalar fields -> `voting` or `prefer_higher_trust`
          * numeric fields -> `median` (or `average` when justified)
          * date/time fields -> `earliest` or `prefer_higher_trust` unless recency is explicitly justified
          * list-like fields -> `prefer_higher_trust`, `intersection`, or `intersection_k_sources`; use `union` only when diagnostics indicate clean, consistent lists
        - For trust-based fusion, pass trust configuration only at registration time:
          `strategy.add_attribute_fuser(attr, prefer_higher_trust, trust_map=trust_map)`
        - Do NOT wrap `prefer_higher_trust` in another function that hardcodes `trust_map` again.
        - Avoid `longest_string` as a default conflict resolver for canonical scalar fields unless diagnostics explicitly show low disagreement and enrichment-only differences.
        - Avoid `most_recent` as a default temporal resolver unless diagnostics explicitly show recency is the correct target semantics.
        - Avoid `union` for list-like fields when cluster purity is weak or list-format inconsistencies are present.

        {PIPELINE_MATCHING_SAFETY_RULES_BLOCK}
        """

        evaluation_analysis = state.get("evaluation_analysis", None)
        if evaluation_analysis:
            system_prompt += f"""
            8. Evaluation reasoning from prior pipeline run:
            {evaluation_analysis}
            """
        reasoning_brief = state.get("evaluation_reasoning_brief", {})
        if isinstance(reasoning_brief, dict) and reasoning_brief:
            system_prompt += f"""
            8b. COMPACT NEXT-PASS SUMMARY:
            {json.dumps(reasoning_brief, indent=2)}

            Treat this as the highest-signal summary of what went wrong and what to try next.
            """

        auto_diagnostics = state.get("auto_diagnostics")
        if auto_diagnostics:
            system_prompt += f"""
            9. AUTO DIAGNOSTICS FROM THE LAST EXECUTION:
            {json.dumps(auto_diagnostics, indent=2)}

            You MUST react to these diagnostics:
            - If mapped ID coverage is low or missing_fused_value is high, set `include_singletons=True` and preserve source IDs for evaluation alignment.
            - If list-attribute accuracy is very low, do not blindly union noisy lists; use a robust strategy (trusted-source selection or consensus with normalization).
            - Always write correspondence files for every evaluated pair to `output/correspondences/` so cluster analysis uses current run artifacts.
            """
        integration_diagnostics_report = state.get("integration_diagnostics_report")
        if integration_diagnostics_report:
            system_prompt += f"""
            10. INTEGRATION DIAGNOSTICS REPORT FROM THE LAST EXECUTION:
            {json.dumps(integration_diagnostics_report, indent=2)}

            Use this report as hard evidence and prioritize fixes that address the highest-severity findings first.
            If present, directly use `fusion_policy_recommendations` and `source_attribute_agreement` to choose per-attribute fusion strategies (e.g., trust-based source preference where agreement clearly favors one source).
            """
        investigator_action_plan = state.get("investigator_action_plan")
        if investigator_action_plan:
            system_prompt += f"""
            11. INVESTIGATOR ACTION PLAN (RANKED):
            {json.dumps(investigator_action_plan, indent=2)}

            Implement highest-priority actions first.
            Do not implement low-confidence actions unless supported by additional evidence.
            """
        fusion_guidance = state.get("fusion_guidance")
        if fusion_guidance:
            system_prompt += f"""
            11b. GENERIC FUSION GUIDANCE FROM DIAGNOSTICS:
            {json.dumps(fusion_guidance, indent=2)}

            Apply this guidance in a dataset-agnostic way:
            - If `attribute_strategies[attr]` exists, use its `recommended_fuser` for that attribute unless the current code already implements an equally safe choice with the same rationale.
            - If `post_clustering.recommended_strategy` is present, apply that post-clustering strategy before fusion.
            - If a dominant source is indicated for an attribute, express that via `prefer_higher_trust` with a transparent `trust_map` rather than hard-coded dataset-specific logic.
            - Keep the implementation generic: decisions should depend on evidence classes such as cluster impurity, disagreement, malformed lists, and source agreement, not on dataset names.
            """
        #### Cluster Analysis ####
        cluster_analysis = state.get("cluster_analysis_result")
        cluster_recommendation = None
        cluster_parameters = {}
        cluster_recommendation_source = None

        if cluster_analysis:
            overall = cluster_analysis.get("_overall", {}) if isinstance(cluster_analysis, dict) else {}
            if isinstance(overall, dict) and overall.get("recommended_strategy") not in (None, "", "None"):
                cluster_recommendation = overall.get("recommended_strategy")
                cluster_parameters = overall.get("parameters", {})
                cluster_recommendation_source = "_overall"
            else:
                for key, value in (cluster_analysis or {}).items():
                    if not isinstance(value, dict):
                        continue
                    rec = value.get("recommended_strategy")
                    if rec and rec != "None":
                        cluster_recommendation = rec
                        cluster_parameters = value.get("parameters", {})
                        cluster_recommendation_source = key
                        break

        if cluster_analysis:
            cluster_example_name = "example_pipeline_ml_cluster.py" if matcher_mode == "ml" else "example_pipeline_cluster.py"
            cluster_example_pipeline_path = os.path.join(INPUT_DIR, f"example_pipelines/{cluster_example_name}")

            if not os.path.exists(cluster_example_pipeline_path):
                raise FileNotFoundError(f"Cluster example pipeline not found at: {cluster_example_pipeline_path}")
            
            with open(cluster_example_pipeline_path, "r", encoding="utf-8") as f:
                cluster_example_pipeline_code = f.read()

            system_prompt += f"""

        **CLUSTER ANALYSIS FEEDBACK FROM PREVIOUS RUN:**
        Report: {json.dumps(cluster_analysis, indent=2)}

        CLUSTER ANALYSIS: The report includes per-file results and may include an `_overall` summary with a `recommended_strategy`.

        **POST-CLUSTERING ACTION REQUIRED:**
        - Recommended strategy: {cluster_recommendation or "None"}
        - Parameters: {json.dumps(cluster_parameters, indent=2)}
        - Source: {cluster_recommendation_source or "n/a"}

        1.  **Analyze the feedback:** Prefer `_overall.recommended_strategy` if present; otherwise scan the per-file entries.
        2.  **CLUSTER-DRIVEN ACTIONS:**
                - Cluster analysis may only influence post-clustering after matching.
                - Do NOT change matching thresholds, comparator weights, comparator columns, or matcher type based on cluster analysis.
                - Keep pre-tested matching_config unchanged.
                - If the Recommended strategy is a post-clustering function, you **MUST** add that step to your generated code.
        3.  **FOLLOW THIS EXAMPLE FOR INCORPORATING POST-CLUSTERING:**
                - DO NOT change the earlier parts of the pipeline (data loading, blocking, matching).
                - After the entity matching step and before fusion, add the post-clustering step as shown below.
                - Use the exact function as per the Recommended strategy above.
                - Example of applying a post-clustering step:
                {cluster_example_pipeline_code}
        
        Your task is to modify the integration pipeline to incorporate the recommended post-clustering step.
        """

        broken_pipeline_path = OUTPUT_DIR + "code/pipeline.py"
        has_existing_pipeline = os.path.exists(broken_pipeline_path) and os.path.getsize(broken_pipeline_path) > 0
        has_execution_feedback = bool(str(state.get("pipeline_execution_result", "") or "").strip())
        has_evaluation_feedback = self._is_metrics_payload(
            state.get("evaluation_metrics_for_adaptation", state.get("evaluation_metrics", {}))
        )

        # Determine if this is initial generation, fix due to execution error, or fix due to poor evaluation
        if not has_existing_pipeline or (not has_execution_feedback and not has_evaluation_feedback):
            print("[*] Creating initial pipeline")
            human_content = "Create the integration pipeline for the provided datasets."
    
        else:
            # Either a pipeline execution error or evaluation-based adaption
            with open(broken_pipeline_path, "r", encoding="utf-8", errors="ignore") as f:
                broken_code = f.read()
    
            human_content = "You previously generated Python integration pipeline code.\n"
            
            # Add execution error if exists
            if "pipeline_execution_result" in state and state["pipeline_execution_result"].lower().startswith("error"):
                print("[*] Trying to fix pipeline")
                human_content += f"Executing this pipeline caused the following error:\n{state['pipeline_execution_result']}\n"
    
            # Add evaluation feedback if available
            if "evaluation_metrics" in state:
                print("[*] Trying to improve pipeline based on evaluation")
                human_content += f"Evaluation of the last run shows the following metrics:\n{json.dumps(state['evaluation_metrics'], indent=2)}\n"
                human_content += "Improve the pipeline to increase overall accuracy and correct errors highlighted by the evaluation.\n"
    
            human_content += "Here is the current pipeline code:\n" + broken_code
            human_content += "\nOutput ONLY the corrected Python code."
            
        ####### SEARCH DOCUMENTATION TOOL #######
        if "search_documentation" in self.tools:
            self.tools["search_documentation"].reset()

        # Rebuild prompt each attempt so latest failures/metrics are included.
        system_prompt += """
            **PROCESS:**
            1.  **THINK**: Analyze the provided data profiles, configurations, and any previous error reports.
            2.  **RESEARCH**: If you are unsure how to use a PyDI function or class, you MUST use the `search_documentation` tool. You can call it multiple times based on given information such as, blocking configurations, matching configurations, data profile etc. Ask specific questions (e.g., "How to use SortedNeighbourhoodBlocker?", "What are the parameters for RuleBasedMatcher?").
            3.  **CODE**: Once you have gathered enough information, write the complete, executable Python code for the pipeline. **Your final output in this process must be only the Python code itself.**
            4.  **FUSION SAFETY**: Ensure custom fusers/lambda fusers accept `**kwargs` so PyDI runtime kwargs do not fail."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]
        state["messages"] = messages
    
        self.logger.info("Input Message:" + str(messages))
    
        # Call the model
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
            
            # After tool calls, invoke the model again to get the final code
            final_response = self._invoke_with_usage(messages, "pipeline_adaption")
            messages.append(final_response)
            adapted_pipeline = final_response
        
        # Extract the code from the final response
        if hasattr(adapted_pipeline, 'tool_calls') and adapted_pipeline.tool_calls:
            # If still has tool calls after second invocation, recursively handle
            adapted_pipeline = "Pipeline code not available - too many tool calls"
        else:
            adapted_pipeline = adapted_pipeline.content if hasattr(adapted_pipeline, "content") else str(adapted_pipeline)
            self.logger.info("RESULT:" + str(adapted_pipeline))
            # if state.get("matching_config"):
                # adapted_pipeline = self._apply_matching_thresholds(adapted_pipeline, state.get("matching_config", dict()))
            if adapted_pipeline.startswith("```python") and adapted_pipeline.endswith("```"):
                adapted_pipeline = adapted_pipeline.strip("```python").strip("```").strip()
                adapted_pipeline = self._apply_pipeline_guardrails(adapted_pipeline, state)

                os.makedirs(OUTPUT_DIR + "code/", exist_ok=True)
                with open(OUTPUT_DIR + "code/pipeline.py", 'w', errors="ignore") as file:
                    file.write(str(adapted_pipeline))
                    # self._rewrite_pipeline_thresholds(OUTPUT_DIR + "code/pipeline.py", state.get("matching_config", {}))
            elif not adapted_pipeline.startswith("Pipeline code not available"):
                # Code without markdown markers - still save it
                adapted_pipeline = self._apply_pipeline_guardrails(str(adapted_pipeline), state)
                os.makedirs(OUTPUT_DIR + "code/", exist_ok=True)
                with open(OUTPUT_DIR + "code/pipeline.py", 'w', errors="ignore") as file:
                    file.write(str(adapted_pipeline))
                    # self._rewrite_pipeline_thresholds(OUTPUT_DIR + "code/pipeline.py", state.get("matching_config", {}))

        self.logger.info('Leaving pipeline_adaption')
        return {"messages": messages, "integration_pipeline_code": adapted_pipeline}

    def execute_pipeline(self, state: SimpleModelAgentState):
        self._log_action("execute_pipeline", "start", "Run generated pipeline", "Produces correspondences for analysis", {"attempt": state.get("pipeline_execution_attempts", 0)+1})

        self.logger.info('----------------------- Entering execute_pipeline -----------------------')

        print("[*] Running generated pipeline")

        attempts = state.get("pipeline_execution_attempts", 0) + 1
        state["pipeline_execution_attempts"] = attempts

        pipeline_path = OUTPUT_DIR + "code/pipeline.py"
        run_started_at = time.time()
        fusion_size_comparison = {}
        pipeline_stdout = ""
        pipeline_stderr = ""

        try:
            result = subprocess.run(
                [sys.executable, pipeline_path],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=3600
            )
            pipeline_stdout = result.stdout or ""
            pipeline_stderr = result.stderr or ""

            if result.returncode == 0:
                print("[+] Pipeline successfully completed")
                execution_result = "success"
                if result.stdout:
                    print("[PIPELINE STDOUT]")
                    print(result.stdout)
                if result.stderr:
                    print("[PIPELINE STDERR]")
                    print(result.stderr)

                # Compare estimate vs actual fused size right after successful pipeline run.
                estimate_path = "output/pipeline_evaluation/fusion_size_estimate.json"
                fused_csv_path = "output/data_fusion/fusion_data.csv"
                if os.path.exists(estimate_path) and os.path.exists(fused_csv_path):
                    try:
                        fusion_size_comparison = compare_estimates_with_actual(
                            fusion_csv_path=fused_csv_path,
                            estimate_path=estimate_path,
                        )
                        comparisons = fusion_size_comparison.get("comparisons", {})
                        if comparisons:
                            print("[FUSION SIZE COMPARISON]")
                            for stage, comp in comparisons.items():
                                print(
                                    f"  {stage}: expected_rows={comp.get('expected_rows')} "
                                    f"actual_rows={comp.get('actual_rows')} "
                                    f"rows_pct_error={comp.get('rows_pct_error')}"
                                )
                    except Exception as compare_error:
                        print(f"[WARN] Could not compare expected vs actual fusion size: {compare_error}")
            else:
                execution_result = f"error: {result.stderr}"[:10000]
                print("Error: " + str(execution_result))
                if result.stdout:
                    print("[PIPELINE STDOUT]")
                    print(result.stdout)

        except Exception as e:
            execution_result = f"error: {str(e)}"
            pipeline_stderr = str(e)

        cycle_index = int(state.get("evaluation_attempts", 0)) + 1
        pipeline_snapshot = self._snapshot_pipeline_attempt(
            cycle_index=cycle_index,
            exec_attempt=attempts,
            execution_result=execution_result,
            stdout=pipeline_stdout if isinstance(pipeline_stdout, str) else str(pipeline_stdout),
            stderr=pipeline_stderr if isinstance(pipeline_stderr, str) else str(pipeline_stderr),
            fusion_size_comparison=fusion_size_comparison if isinstance(fusion_size_comparison, dict) else {},
        )
        pipeline_snapshots = list(state.get("pipeline_snapshots", []) or [])
        pipeline_snapshots.append(pipeline_snapshot)

        self.logger.info("Pipeline execution result: " + execution_result)
        self.logger.info('Leaving execute_pipeline')

        return {
            "pipeline_execution_result": execution_result,
            "pipeline_execution_attempts": attempts,
            "pipeline_run_started_at": run_started_at,
            "pipeline_run_finished_at": time.time(),
            "fusion_size_comparison": fusion_size_comparison,
            "pipeline_snapshots": pipeline_snapshots,
        }
    
    def cluster_analysis(self, state: SimpleModelAgentState) -> Dict[str, Any]:
        from helpers.correspondence_helpers import (
            collect_latest_correspondence_files as helper_collect_latest_correspondence_files,
            summarize_correspondence_entries as helper_summarize_correspondence_entries,
        )

        self._log_action("cluster_analysis", "start", "Analyze pairwise correspondences", "Supports investigator loop")
        self.logger.info("----------------------- Entering Cluster Analysis Helper -----------------------")
        print("[*] Running cluster analysis helper...")

        correspondence_files_to_process = helper_collect_latest_correspondence_files(state)
        if not correspondence_files_to_process:
            print("[-] No relevant correspondence files found for current run, skipping cluster analysis.")
            return {"cluster_analysis_result": {"warning": "stale_or_missing_correspondence_files", "_investigation": {"files": {}}}}

        print(f"[*] Found {len(correspondence_files_to_process)} pairwise correspondence files to analyze.")

        try:
            cluster_tester = ClusterTester(llm=self.base_model, verbose=True)
            report = cluster_tester.run(correspondence_files_to_process)
            if not isinstance(report, dict):
                report = {"_overall": {"recommended_strategy": "None", "parameters": {}}, "raw_report": report}
            report["_investigation"] = helper_summarize_correspondence_entries(correspondence_files_to_process)

            return {"cluster_analysis_result": report}
        except Exception as e:
            print(f"[-] Error during cluster analysis helper: {e}")
            self.logger.error(f"Cluster analysis helper failed: {traceback.format_exc()}")
            return {"cluster_analysis_result": {"error": str(e)}}
        
    def evaluation_adaption(self, state: SimpleModelAgentState):
        from helpers.evaluation_helpers import extract_python_code as helper_extract_python_code

        self._log_action("evaluation_adaption", "start", "Generate evaluation code", "Measures fusion quality")

        self.logger.info('----------------------- Entering evaluation_adaption -----------------------')
        try:
            from prompts.evaluation_prompt import EVALUATION_ROBUSTNESS_RULES_BLOCK
        except Exception:
            EVALUATION_ROBUSTNESS_RULES_BLOCK = (
                "EVALUATION ROBUSTNESS RULES (MANDATORY):\n"
                "- Use validation-set representation as the normalization target for evaluation.\n"
                "- Normalize fused and gold columns to the SAME representation before evaluating.\n"
                "- For country-like columns, infer and apply a shared normalize_country output_format from the evaluation set (alpha_2/alpha_3/numeric/name/official_name).\n"
                "- For free-text/list-like columns, avoid strict exact comparisons unless values are canonical identifiers.\n"
                "  Prefer tokenized_match with an explicit threshold (for example 0.7-0.9) instead of exact-only semantics.\n"
                "- If using set-based comparison on textual lists, first canonicalize list elements consistently across fused and gold."
            )
        
        example_eval_path = INPUT_DIR + "example_pipelines/example_evaluation.py"
        example_eval_code = open(example_eval_path).read()
    
        fused_output_path = "output/data_fusion/fusion_data.csv"
        force_test_eval = bool(state.get("_force_test_eval"))
        eval_testset_path = self._evaluation_testset_path(state, force_test=force_test_eval)
        eval_stage = self._evaluation_stage_label(state, force_test=force_test_eval)

        # Prepare first-row preview for evaluation set
        try:
            dataset = load_dataset(eval_testset_path) if eval_testset_path else None
        except Exception:
            dataset = None
        if dataset is not None and not dataset.empty:
            testset_preview = dataset.iloc[0].to_dict()
        else:
            testset_preview = "Failed to load or empty dataset"
    
        system_prompt = f"""
        You are a data scientist evaluating a data fusion pipeline.
    
        Example evaluation code:
        {example_eval_code}
    
        Generated integration pipeline:
        {state['integration_pipeline_code']}
    
        Dataset profiles:
        {json.dumps(state['data_profiles'], indent=2)}
    
        The fused output is located at:
        {fused_output_path}

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

        ####### HUMAN PROMPT #######
        if not state.get("evaluation_execution_result"):
            print("[*] Adapting evaluation code")
            # First generation
            human_content = """
            Create the evaluation code.
            """
        else:
            # Fix broken evaluation
            print("[*] Fixing evaluation code")
            attempts = state.get("evaluation_execution_attempts", 0)
    
            if attempts >= 3:
                self.logger.info("Maximum evaluation fix attempts reached")
                return {"evaluation_execution_result": "error: max attempts reached"}
    
            eval_path = OUTPUT_DIR + "code/evaluation.py"
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

        # Rebuild prompt each attempt so latest execution error is always included.
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
            
            # After tool calls, invoke the model again to get the final code
            final_response = self._invoke_with_usage(messages, "evaluation_adaption")
            messages.append(final_response)
            result = final_response

        if hasattr(result, 'tool_calls') and result.tool_calls:
            # If still has tool calls after second invocation, recursively handle
            code = "Evaluation code not available - too many tool calls"
        else:
            code = result.content if hasattr(result, "content") else str(result)
            self.logger.info("RESULT:" + str(code))

            code = helper_extract_python_code(code)
            code = self._apply_evaluation_guardrails(code)
            with open(OUTPUT_DIR + "code/evaluation.py", "w") as f:
                f.write(code)

        self.logger.info('Leaving evaluation_adaption')
    
        return {"evaluation_code": code, "eval_messages": messages}

    def execute_evaluation(self, state: SimpleModelAgentState):
        self._log_action("execute_evaluation", "start", "Run evaluation", "Produces accuracy metrics", {"attempt": state.get("evaluation_execution_attempts", 0)+1})

        self.logger.info('----------------------- Entering execute_evaluation -----------------------')
    
        print("[*] Running generated evaluation")
    
        attempts = state.get("evaluation_execution_attempts", 0) + 1
        state["evaluation_execution_attempts"] = attempts
    
        evaluation_path = OUTPUT_DIR + "code/evaluation.py"
        evaluation_stdout = ""
        evaluation_stderr = ""
        metrics_from_execution: Dict[str, Any] = {}
        metrics_source = "none"
        stage_label = self._evaluation_stage_label(state, force_test=bool(state.get("_force_test_eval")))
        cycle_index = int(state.get("evaluation_attempts", 0)) + 1
        eval_started_at = time.time()
        eval_metrics_path = "output/pipeline_evaluation/pipeline_evaluation.json"
    
        try:
            result = subprocess.run(
                [sys.executable, evaluation_path],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=3600
            )
            evaluation_stdout = result.stdout or ""
            evaluation_stderr = result.stderr or ""
    
            if result.returncode == 0:
                print("[+] Evaluation successfully completed")
                execution_result = "success"
                if result.stdout:
                    print("[EVALUATION STDOUT]")
                    print(result.stdout)
                if result.stderr:
                    print("[EVALUATION STDERR]")
                    print(result.stderr)
            else:
                print("[-] Evaluation could not be executed")
                execution_result = f"error: {result.stderr}"[:10000]
                print("Error: " + str(execution_result))
                if result.stdout:
                    print("[EVALUATION STDOUT]")
                    print(result.stdout)

            metrics_from_stdout = self._extract_metrics_from_text(evaluation_stdout)
            metrics_from_file: Dict[str, Any] = {}
            if os.path.exists(eval_metrics_path):
                try:
                    # Avoid stale fallback from older runs.
                    if os.path.getmtime(eval_metrics_path) >= (eval_started_at - 1.0):
                        with open(eval_metrics_path, "r", encoding="utf-8") as f:
                            parsed = json.load(f)
                        if isinstance(parsed, dict):
                            metrics_from_file = parsed
                except Exception:
                    metrics_from_file = {}

            if self._is_metrics_payload(metrics_from_stdout):
                metrics_from_execution = metrics_from_stdout
                metrics_source = "stdout"
                # Keep file state aligned with accepted execution metrics.
                try:
                    os.makedirs(os.path.dirname(eval_metrics_path), exist_ok=True)
                    with open(eval_metrics_path, "w", encoding="utf-8") as f:
                        json.dump(metrics_from_execution, f, indent=4)
                except Exception:
                    pass
            elif self._is_metrics_payload(metrics_from_file):
                metrics_from_execution = metrics_from_file
                metrics_source = "pipeline_evaluation_file"
    
        except Exception as e:
            execution_result = f"error: {str(e)}"

        evaluation_snapshot = self._snapshot_evaluation_attempt(
            stage=stage_label,
            cycle_index=cycle_index,
            exec_attempt=attempts,
            execution_result=execution_result,
            stdout=evaluation_stdout if isinstance(evaluation_stdout, str) else str(evaluation_stdout),
            stderr=evaluation_stderr if isinstance(evaluation_stderr, str) else str(evaluation_stderr),
            metrics=metrics_from_execution if isinstance(metrics_from_execution, dict) else {},
            metrics_source=metrics_source,
        )
        evaluation_snapshots = list(state.get("evaluation_snapshots", []) or [])
        evaluation_snapshots.append(evaluation_snapshot)
    
        self.logger.info("Evaluation execution result: " + execution_result)
        self.logger.info('Leaving execute_evaluation')
    
        return {
            "evaluation_execution_result": execution_result,
            "evaluation_execution_attempts": attempts,
            "evaluation_execution_stdout": evaluation_stdout,
            "evaluation_execution_stderr": evaluation_stderr,
            "evaluation_metrics_from_execution": metrics_from_execution,
            "evaluation_metrics_source": metrics_source,
            "evaluation_snapshots": evaluation_snapshots,
        }

    def integration_diagnostics(self, state: SimpleModelAgentState):
        from helpers.evaluation_helpers import extract_python_code as helper_extract_python_code

        self._log_action(
            "integration_diagnostics",
            "start",
            "Generate and run free-form integration diagnostics",
            "Produces issue report and concrete next actions",
        )
        print("[*] Running free integration diagnostics")

        os.makedirs(OUTPUT_DIR + "code/", exist_ok=True)
        os.makedirs("output/pipeline_evaluation", exist_ok=True)

        diagnostics_code_path = OUTPUT_DIR + "code/integration_diagnostics.py"
        report_json_path = "output/pipeline_evaluation/integration_diagnostics.json"
        report_md_path = "output/pipeline_evaluation/integration_diagnostics.md"

        context_payload = {
            "datasets": state.get("datasets", []),
            "fusion_testset": state.get("fusion_testset"),
            "validation_fusion_testset": state.get("validation_fusion_testset"),
            "evaluation_testset": self._evaluation_testset_path(state, force_test=bool(state.get("_force_test_eval"))),
            "evaluation_stage": self._evaluation_stage_label(state, force_test=bool(state.get("_force_test_eval"))),
            "matcher_mode": state.get("matcher_mode"),
            "evaluation_metrics": state.get("evaluation_metrics", {}),
            "auto_diagnostics": state.get("auto_diagnostics", {}),
            "normalization_execution_result": state.get("normalization_execution_result", ""),
            "normalization_attempts": state.get("normalization_attempts", 0),
            "normalization_report": state.get("normalization_report", {}),
            "normalization_directives": state.get("normalization_directives", {}),
            "investigator_action_plan": state.get("investigator_action_plan", []),
            "normalized_datasets": state.get("normalized_datasets", []),
            "fusion_size_comparison": state.get("fusion_size_comparison", {}),
            "cluster_analysis_result": state.get("cluster_analysis_result", {}),
            "output_paths": {
                "fused_csv": "output/data_fusion/fusion_data.csv",
                "evaluation_json": "output/pipeline_evaluation/pipeline_evaluation.json",
                "debug_jsonl": "output/pipeline_evaluation/debug_fusion_eval.jsonl",
                "fusion_size_estimate_json": "output/pipeline_evaluation/fusion_size_estimate.json",
                "correspondences_dir": "output/correspondences",
                "report_json": report_json_path,
                "report_md": report_md_path,
            },
        }

        system_prompt = """
        You are generating a standalone Python diagnostics script for a data-integration pipeline.
        Write ONLY executable Python code.

        Hard requirements:
        - Be dataset-agnostic (no dataset-specific hardcoding).
        - The diagnostics script must be generated from reasoning and execute its own analysis logic (no placeholder output).
        - Do NOT read from stdin (no `sys.stdin.read()` / `input()`); rely only on discovered files and constants in the script.
        - Read available artifacts and detect integration issues robustly.
        - Focus on actionable findings: coverage gaps, over-clustering, noisy list fusion, and size-estimate drift.
        - Explicitly detect empty pairwise correspondence files and treat them as HIGH severity if matching config expected that pair.
        - Explicitly detect case-only mismatch patterns from debug_fusion_eval events.
        - Perform attribute-level source agreement analysis:
          * load fused output and source datasets
          * compare fused attribute values against source attribute values (for aligned rows where possible)
          * compute per-attribute/per-source agreement metrics (support_count, exact_ratio and/or similarity ratio)
          * identify attributes where one source strongly dominates agreement
        - From that analysis, generate machine-usable fusion policy suggestions (for example trust-based source preference for specific attributes).
        - Rank suggested improvements by expected impact and confidence, with concise evidence.

        - Explicitly check normalization-sensitive issues and report them as findings/recommendations:

          * country canonicalization mismatches (e.g., long-form vs short-form country names)

          * list-format inconsistencies (JSON-list strings vs scalar strings vs malformed list encodings)
        - Gracefully handle missing files (do not crash; record warnings).
        - Any index-based row sampling must be index-safe:
          * do NOT assume sampled labels are present in current dataframe index.
          * avoid unsafe patterns like df = df.loc[take] unless pre-validated.
          * prefer reset_index(drop=True)+iloc for positional samples, or filter labels to existing index before loc.
        - Always guard column access:
          * before `df[col]`, ensure `col in df.columns`.
          * if column is missing, record a warning and continue instead of raising KeyError.
        - Output TWO files:
          1) output/pipeline_evaluation/integration_diagnostics.json
          2) output/pipeline_evaluation/integration_diagnostics.md
        - JSON must contain at least:
          - summary
          - findings (list ordered by severity)
          - recommendations (list, concrete and generic)
          - source_attribute_agreement (attribute -> source metrics)
          - fusion_policy_recommendations (list of suggested strategy/trust actions with evidence)
          - each recommendation should include: action, target_attributes, evidence_summary, expected_impact, confidence

          - recommendation fields should include machine-usable hints when possible (e.g., target columns, suggested PyDI functions)
          - evidence (counts/ratios/paths)
          - created_at
        - Keep dependencies minimal (stdlib + pandas).
        """

        def _apply_diagnostics_guardrails(code: str) -> str:
            if not isinstance(code, str) or not code.strip():
                return code

            # Prevent hangs in notebook/subprocess execution: diagnostics must never block on stdin.
            code = re.sub(r"\bsys\.stdin\.read\(\)", "''", code)
            code = re.sub(r"\bsys\.stdin\.buffer\.read\(\)", "b''", code)
            code = re.sub(r"\binput\((.*?)\)", "''", code)

            # Guard common crash pattern:
            # `src_sub = src_sub.loc[take]` where `take` contains labels not in `src_sub.index`.
            risky_loc_pattern = re.compile(
                r"(?m)^(\s*)([A-Za-z_]\w*)\s*=\s*\2\.loc\[([A-Za-z_]\w*)\]\s*$"
            )

            def _repl(match: re.Match) -> str:
                indent, df_name, idx_name = match.group(1), match.group(2), match.group(3)
                idx_l = idx_name.lower()
                if not any(k in idx_l for k in ("take", "idx", "index", "sample", "selected")):
                    return match.group(0)
                return (
                    f"{indent}{df_name} = {df_name}.loc"
                    f"[[__i for __i in {idx_name} if __i in {df_name}.index]]"
                )

            return risky_loc_pattern.sub(_repl, code)

        def _compact_diagnostics_error(raw: str) -> str:
            text = str(raw or "").strip()
            if not text:
                return "unknown diagnostics execution error"
            flat = re.sub(r"\s+", " ", text)
            if "KeyError" in flat and "not in index" in flat:
                return (
                    "KeyError during diagnostics dataframe selection: sampled labels were not in dataframe index. "
                    "Use index-safe selection (filter sampled labels by existing index before `.loc`, "
                    "or use `reset_index(drop=True)` with `.iloc`)."
                )
            keyerror_column = re.search(r"KeyError:\s*'([^']+)'", flat)
            if keyerror_column:
                missing_col = keyerror_column.group(1)
                return (
                    f"KeyError: missing column '{missing_col}' in diagnostics code. "
                    f"Guard column access with `if '{missing_col}' in df.columns` and continue gracefully."
                )
            return flat[:1800]

        def _generate_code(feedback: str | None = None) -> str:
            human_content = f"""
            Context for diagnostics:
            {json.dumps(context_payload, indent=2)}

            Generate a Python script that inspects those artifacts, computes diagnostics, and writes:
            - {report_json_path}
            - {report_md_path}
            The script should include concrete computations for source-attribute agreement and not only text heuristics.
            """
            if feedback:
                human_content += f"""

                The previous diagnostics script failed with:
                {feedback}

                Fix the script and return corrected Python code only.
                """
            response = self._invoke_base_with_usage(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_content),
                ],
                "integration_diagnostics_generation",
            )
            return helper_extract_python_code(response)

        diagnostics_code = _generate_code()
        execution_result = "error: diagnostics script not executed"
        report_payload: Dict[str, Any] = {}

        for attempt in range(1, 3):
            diagnostics_code = _apply_diagnostics_guardrails(diagnostics_code)
            with open(diagnostics_code_path, "w", encoding="utf-8") as f:
                f.write(diagnostics_code)

            try:
                result = subprocess.run(
                    [sys.executable, diagnostics_code_path],
                    capture_output=True,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    timeout=1200,
                )
                if result.returncode == 0:
                    execution_result = "success"
                    if result.stdout:
                        print("[INTEGRATION DIAGNOSTICS STDOUT]")
                        print(result.stdout)
                    if result.stderr:
                        print("[INTEGRATION DIAGNOSTICS STDERR]")
                        print(result.stderr)
                    break

                execution_result = f"error: {_compact_diagnostics_error(result.stderr or result.stdout)}"[:10000]
                print("[-] Integration diagnostics failed")
                print(execution_result)
            except Exception as e:
                execution_result = f"error: {_compact_diagnostics_error(str(e))}"
                print("[-] Integration diagnostics exception")
                print(execution_result)

            if attempt < 2:
                diagnostics_code = _generate_code(feedback=execution_result)

        if os.path.exists(report_json_path):
            try:
                with open(report_json_path, "r", encoding="utf-8") as f:
                    report_payload = json.load(f)
            except Exception as e:
                report_payload = {"error": f"could_not_read_report: {e}"}

        return {
            "integration_diagnostics_code": diagnostics_code,
            "integration_diagnostics_execution_result": execution_result,
            "integration_diagnostics_report": report_payload,
        }


    def human_review_export(self, state: SimpleModelAgentState):
        from helpers.evaluation_helpers import extract_python_code as helper_extract_python_code

        self._log_action(
            "human_review_export",
            "start",
            "Generate final human-review package",
            "Creates reviewer-friendly fusion/source/testset comparison artifacts",
        )
        print("[*] Building final human-review package")

        os.makedirs(OUTPUT_DIR + "code/", exist_ok=True)
        os.makedirs("output/human_review", exist_ok=True)

        review_code_path = OUTPUT_DIR + "code/human_review_export.py"
        review_summary_json = "output/human_review/human_review_summary.json"
        review_summary_md = "output/human_review/human_review_summary.md"
        fused_review_csv = "output/human_review/fused_review_table.csv"
        source_lineage_csv = "output/human_review/source_lineage_long.csv"
        diff_csv = "output/human_review/fusion_vs_testset_diff.csv"

        review_attributes: List[str] = []
        fused_csv_path = "output/data_fusion/fusion_data.csv"
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
                "fused_csv": "output/data_fusion/fusion_data.csv",
                "fused_debug_jsonl": "output/data_fusion/debug_fusion_data.jsonl",
                "evaluation_json": "output/pipeline_evaluation/pipeline_evaluation.json",
                "review_summary_json": review_summary_json,
                "review_summary_md": review_summary_md,
                "fused_review_csv": fused_review_csv,
                "source_lineage_csv": source_lineage_csv,
                "diff_csv": diff_csv,
            },
        }

        system_prompt = """
        You are generating a standalone Python script that creates FINAL human-review outputs for a data-fusion run.
        Write ONLY executable Python code.

        Hard requirements:
        - Be dataset-agnostic (no hardcoded dataset-specific columns or IDs).
        - Use only standard library + pandas (optional: PyDI.io if available).
        - Read fused output from output/data_fusion/fusion_data.csv.
        - Load source datasets from the provided dataset paths and detect ID columns robustly.
        - Parse `_fusion_sources` robustly (JSON/list/string/empty) and map each source ID back to source rows.
        - Produce reviewer-friendly outputs:
          1) output/human_review/fused_review_table.csv
             - one row per fused entity in WIDE format
             - for EVERY attribute listed in context_payload.review_attributes, create EXACTLY these columns:
               <attribute>_test, <attribute>_fused, <attribute>_source_1, <attribute>_source_2, <attribute>_source_3
             - keep the attribute text as-is when forming column names (do not rename/sanitize)
             - if testset value is unavailable, leave <attribute>_test as empty string
             - if fewer than 3 sources are available, leave missing <attribute>_source_<n> cells empty
             - this wide-table schema is mandatory and must exist even when values are missing
          2) output/human_review/source_lineage_long.csv
             - long table: fused_id, source_id, source_dataset, source attribute values, fused attribute values
          3) output/human_review/fusion_vs_testset_diff.csv
             - if fusion_testset exists, compare overlapping columns against mapped fused rows and record per-attribute diffs
             - if unavailable, still create an empty CSV with columns and a note in summary
          4) output/human_review/human_review_summary.json
          5) output/human_review/human_review_summary.md
        - Summary JSON must include: summary, file_paths, counts, warnings, created_at.
        - Gracefully handle missing files and malformed rows; do not crash.
        """

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
                    timeout=1800,
                )
                if run.returncode == 0:
                    schema_ok, schema_msg = _validate_wide_review_columns(fused_review_csv, review_attributes)
                    if not schema_ok:
                        feedback = (
                            "Script executed but fused_review_table.csv does not match required wide schema. "
                            + schema_msg
                            + " Use exact column pattern <attribute>_test/<attribute>_fused/<attribute>_source_1/<attribute>_source_2/<attribute>_source_3 for every attribute in context_payload.review_attributes."
                        )[:10000]
                        execution_result = f"error: {feedback}"[:10000]
                        print("[-] Human-review export schema validation failed")
                        continue

                    execution_result = "success"
                    print("[*] Human-review export completed")
                    if run.stdout:
                        print("[HUMAN REVIEW STDOUT]")
                        print(run.stdout)
                    if run.stderr:
                        print("[HUMAN REVIEW STDERR]")
                        print(run.stderr)
                    break
                feedback = (run.stderr or run.stdout or "unknown error")[:10000]
                execution_result = f"error: {feedback}"[:10000]
            except Exception as e:
                feedback = str(e)
                execution_result = f"error: {feedback}"[:10000]

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
            print("[*] Sealed final-test evaluation not active (missing validation/test paths).")
            return {
                "final_test_evaluation_execution_result": "skipped",
                "final_test_evaluation_metrics": {},
            }

        print("[*] Running sealed final test evaluation on held-out test set")
        temp_state = dict(state)
        temp_state["_force_test_eval"] = True
        temp_state["evaluation_execution_result"] = ""
        temp_state["evaluation_execution_attempts"] = 0

        eval_updates = self.evaluation_adaption(temp_state)
        temp_state.update(eval_updates)

        exec_updates = self.execute_evaluation(temp_state)
        temp_state.update(exec_updates)

        eval_path = "output/pipeline_evaluation/pipeline_evaluation.json"
        final_metrics: Dict[str, Any] = {}
        metrics_from_execution = temp_state.get("evaluation_metrics_from_execution", {})
        if self._is_metrics_payload(metrics_from_execution):
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
                print(f"[*] Sealed final test overall accuracy: {float(final_acc):.3%}")
            except Exception:
                pass

        return {
            "final_test_evaluation_execution_result": temp_state.get("evaluation_execution_result", ""),
            "final_test_evaluation_metrics": final_metrics,
        }


    def evaluation_decision(self, state: SimpleModelAgentState):
        self._log_action("evaluation_decision", "start", "Decide whether to iterate", "Drives improvement loop")

        self.logger.info('----------------------- Entering evaluation_decision -----------------------')
        print("[*] Reading evaluation metrics")

        attempts = state.get("evaluation_attempts", 0) + 1
        eval_path = "output/pipeline_evaluation/pipeline_evaluation.json"

        metrics_from_execution = state.get("evaluation_metrics_from_execution", {})
        if self._is_metrics_payload(metrics_from_execution):
            metrics = dict(metrics_from_execution)
            metrics_source = state.get("evaluation_metrics_source", "execution")
        elif os.path.exists(eval_path):
            with open(eval_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            metrics_source = "pipeline_evaluation_file"
        else:
            self.logger.warning("Evaluation file missing")
            metrics = {"error": "evaluation file missing"}
            metrics_source = "missing"

        raw_metrics = dict(metrics) if isinstance(metrics, dict) else {}
        best_metrics = state.get("best_validation_metrics", {})
        correspondence_integrity = self._evaluate_correspondence_integrity(state)
        structural_invalid = not bool(correspondence_integrity.get("structurally_valid", True))
        regression = self._assess_validation_regression(
            raw_metrics if isinstance(raw_metrics, dict) else {},
            best_metrics if isinstance(best_metrics, dict) else {},
        )
        if structural_invalid:
            regression = {
                **regression,
                "rejected": True,
                "reason": "structural_invalid_correspondences",
                "invalid_pairs": correspondence_integrity.get("invalid_pairs", []),
            }
        accepted_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        if regression.get("rejected"):
            if self._is_metrics_payload(best_metrics):
                accepted_metrics = dict(best_metrics)
                print("[EVALUATION GUARD] Rejected regressive validation metrics; keeping best-known validation metrics")
        else:
            if self._is_metrics_better(raw_metrics, best_metrics if isinstance(best_metrics, dict) else {}):
                best_metrics = dict(raw_metrics)
            elif not self._is_metrics_payload(best_metrics) and self._is_metrics_payload(raw_metrics):
                best_metrics = dict(raw_metrics)

        analysis_metrics = raw_metrics if self._is_metrics_payload(raw_metrics) else accepted_metrics

        self.logger.info(f"Evaluation metrics ({metrics_source}): {raw_metrics}")
        print(f"Evaluation metrics: {json.dumps(accepted_metrics)}")
        if accepted_metrics != raw_metrics:
            print(f"[EVALUATION CANDIDATE] Raw metrics: {json.dumps(raw_metrics)}")
        if structural_invalid:
            invalid_pairs = correspondence_integrity.get("invalid_pairs", [])
            print(
                "[CORRESPONDENCE GATE] Structural invalid candidate: "
                f"missing/empty correspondences for pairs={invalid_pairs}"
            )

        state["latest_validation_metrics"] = raw_metrics
        state["best_validation_metrics"] = best_metrics if isinstance(best_metrics, dict) else {}
        state["evaluation_regression_guard"] = regression
        state["correspondence_integrity"] = correspondence_integrity
        state["evaluation_metrics"] = accepted_metrics
        state["evaluation_metrics_raw"] = raw_metrics
        state["evaluation_metrics_for_adaptation"] = analysis_metrics

        eval_stage = self._evaluation_stage_label(state, force_test=False)
        try:
            overall_acc = float(analysis_metrics.get("overall_accuracy", 0.0))
        except Exception:
            overall_acc = 0.0

        auto_diagnostics = self._compute_auto_diagnostics(state, analysis_metrics)
        state["auto_diagnostics"] = auto_diagnostics

        problems: List[str] = []
        if overall_acc < 0.85:
            problems.append(f"overall_accuracy below target: {overall_acc:.3%} < 85.000%")

        low_acc_columns: List[str] = []
        if isinstance(analysis_metrics, dict):
            for key, value in analysis_metrics.items():
                if not (isinstance(key, str) and key.endswith("_accuracy") and key != "overall_accuracy"):
                    continue
                try:
                    v = float(value)
                except Exception:
                    continue
                if v < 0.5:
                    low_acc_columns.append(f"{key[:-9]}={v:.3%}")
        if low_acc_columns:
            problems.append("low attribute accuracy: " + ", ".join(sorted(low_acc_columns)[:8]))
        if structural_invalid:
            problems.append(
                "structural invalid candidate: missing/empty correspondence files for expected pairs "
                f"{correspondence_integrity.get('invalid_pairs', [])}"
            )
        if regression.get("rejected"):
            problems.append(
                "regression guard rejected latest candidate: "
                f"overall_drop={regression.get('overall_drop')}, "
                f"macro_drop={regression.get('macro_drop')}, "
                f"top_attribute_drops={regression.get('top_attribute_drops', {})}"
            )

        id_alignment = auto_diagnostics.get("id_alignment", {}) if isinstance(auto_diagnostics, dict) else {}
        if isinstance(id_alignment, dict):
            mapped_ratio = id_alignment.get("mapped_coverage_ratio")
            direct_ratio = id_alignment.get("direct_coverage_ratio")
            try:
                if mapped_ratio is not None and float(mapped_ratio) < 0.8:
                    problems.append(f"low mapped ID coverage: {float(mapped_ratio):.3%}")
            except Exception:
                pass
            try:
                if direct_ratio is not None and float(direct_ratio) < 0.5:
                    problems.append(f"low direct ID coverage: {float(direct_ratio):.3%}")
            except Exception:
                pass

        reason_ratios = auto_diagnostics.get("debug_reason_ratios", {}) if isinstance(auto_diagnostics, dict) else {}
        if isinstance(reason_ratios, dict):
            try:
                missing_ratio = float(reason_ratios.get("missing_fused_value", 0.0))
                if missing_ratio > 0.25:
                    problems.append(f"high missing_fused_value mismatch ratio: {missing_ratio:.3%}")
            except Exception:
                pass

        print("[EVALUATION REPORT]")
        print(f"  Stage: {eval_stage}")
        print(f"  Attempt: {attempts}/3")
        print(f"  Overall Accuracy: {overall_acc:.3%}")
        print("  Problems Detected:")
        if problems:
            for p in problems[:10]:
                print(f"    - {p}")
        else:
            print("    - none")
        print("  Planned Adaptation:")
        if overall_acc >= 0.85:
            print("    - quality target reached; continue to human review")
        elif attempts >= 3:
            print("    - max automatic iterations reached; stop adaptation loop")
        else:
            print("    - investigator runs diagnostics report + cluster evidence analysis")
            print("    - reasoning node proposes targeted pipeline/code changes")
            if any("coverage" in p for p in problems):
                print("    - prioritize ID alignment and singleton handling")
            if any("missing_fused_value" in p for p in problems):
                print("    - prioritize fusion strategy changes for sparse attributes")

        if auto_diagnostics:
            print("[*] Auto diagnostics:")
            print(json.dumps(auto_diagnostics, indent=2))

        if overall_acc >= 0.85 or attempts >= 3:
            self._print_total_usage()

        cycle_audit = list(state.get("evaluation_cycle_audit", []) or [])
        cycle_audit.append(
            {
                "attempt": int(attempts),
                "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "evaluation_stage": eval_stage,
                "metrics_source": metrics_source,
                "raw_metrics": raw_metrics,
                "accepted_metrics": accepted_metrics,
                "analysis_metrics": analysis_metrics,
                "best_validation_metrics": best_metrics if isinstance(best_metrics, dict) else {},
                "regression_guard": regression,
                "correspondence_integrity": correspondence_integrity,
                "structural_valid": not structural_invalid,
                "problems": list(problems[:20]),
            }
        )
    
        self.logger.info('Leaving evaluation_decision')

        return {
            "evaluation_metrics": accepted_metrics,
            "auto_diagnostics": auto_diagnostics,
            "fusion_size_comparison": state.get("fusion_size_comparison", {}),
            "evaluation_attempts": attempts,
            "pipeline_execution_result": "",
            "pipeline_execution_attempts": 0,
            "evaluation_execution_result": "",
            "evaluation_execution_attempts": 0,
            "evaluation_regression_guard": regression,
            "latest_validation_metrics": raw_metrics,
            "best_validation_metrics": best_metrics if isinstance(best_metrics, dict) else {},
            "evaluation_metrics_raw": raw_metrics,
            "evaluation_metrics_for_adaptation": analysis_metrics,
            "correspondence_integrity": correspondence_integrity,
            "evaluation_cycle_audit": cycle_audit,
        }
    def evaluation_reasoning(self, state: SimpleModelAgentState):
        self._log_action("evaluation_reasoning", "start", "Analyze errors", "Suggests concrete fixes for next run")

        self.logger.info('----------------------- Entering evaluation_reasoning -----------------------')
    
        # Sync matching_config from disk only when explicitly allowed.
        matching_path = os.path.join(OUTPUT_DIR, "matching-evaluation", "matching_config.json")
        if self._cross_run_memory_enabled(state) and os.path.exists(matching_path):
            with open(matching_path, "r", encoding="utf-8") as f:
                state["matching_config"] = json.load(f)

        debug_path = "output/pipeline_evaluation/debug_fusion_eval.jsonl"
        debug_events = []
    
        if os.path.exists(debug_path):
            with open(debug_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        debug_events.append(json.loads(line))
                    except Exception:
                        continue
    
        # Prepare system prompt
        system_prompt = """
        You are a data integration expert analyzing the evaluation results of a data integration pipeline.
    
        You are given:
        - Aggregate evaluation metrics
        - Detailed per-record debugging events from a fusion evaluation
        - Automatic diagnostics (ID coverage, mismatch reasons, size comparison)
        - The current integration pipeline code
        - The optimal blocking and matching configurations
        - Detailed Cluster Analysis report with improvement suggestions.
    
        Your task:
        1. Identify the dominant error types (false positives, false negatives, missed blocks, bad matches)
        2. Determine whether errors are more likely caused by:
           - Blocking being too restrictive or too loose
           - Matching thresholds or comparator choices
           - Schema or preprocessing issues
           - Logic in the current pipeline code
        3. Analyze the provided cluster analysis report to suggest improvements.
        4. Suggest concrete, actionable improvements for the next pipeline iteration

        5. If issues indicate normalization problems, explicitly recommend normalization actions with target columns.
           Prefer these capabilities when appropriate:
           - PyDI country normalization: normalize_country(value, output_format="name")
           - list formatting normalization via helper functions: detect_list_like_columns / normalize_list_like_columns
        6. Phrase recommendations as generic evidence-driven patterns, not dataset-specific hacks:
           - field-shape-aware fusion
           - post-clustering cleanup when cluster impurity is high
           - consensus vs trust-based source preference when source agreement is strong
           - safer list handling when list-format noise is high

        IMPORTANT:
        - Blocking and matching have already been evaluated for the current pass, so focus primarily on fusion and preprocessing.
        - If diagnostics show pairwise matching quality remains weak, you may say that rerunning matching is justified, but do not recommend dataset-specific comparator hacks.

        Write concise reasoning that is readable in a report.
        Keep the response natural-language, but organize it with these short headings:
        - What Went Wrong
        - What The Agent Should Try Next
        - Normalization Recommendations (only when relevant)
        - Report Takeaway
        """
    
        # Prepare human content
        human_content = f"""
        Evaluation metrics:
        {json.dumps(state.get("evaluation_metrics", {}), indent=2)}
    
        Debug fusion events (JSONL):
        {json.dumps(debug_events[:50], indent=2)}
    
        Current integration pipeline code:
        {state.get("integration_pipeline_code", "Pipeline code not available")}

        Current pipeline evaluation code:
        {state.get("evaluation_code", "Pipeline evaluation code not available")}
    
        Optimal blocking configuration:
        {json.dumps(state.get("blocking_config", {}), indent=2)}
    
        Optimal matching configuration:
        {json.dumps(state.get("matching_config", {}), indent=2)}

        Cluster Analysis Report:
        {json.dumps(state.get("cluster_analysis_result", {}), indent=2)}

        Auto diagnostics:
        {json.dumps(state.get("auto_diagnostics", {}), indent=2)}

        Fusion size comparison:
        {json.dumps(state.get("fusion_size_comparison", {}), indent=2)}

        Integration diagnostics report:
        {json.dumps(state.get("integration_diagnostics_report", {}), indent=2)}
        """
    
        # Model call
        message = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]
    
        result = self._invoke_with_usage(message, "evaluation_reasoning")
        analysis = result.content if hasattr(result, "content") else str(result)
        reasoning_brief = _build_reasoning_brief(analysis)
    
        self.logger.info("Evaluation reasoning produced")
        print("[*] Evaluation reasoning:\n" + analysis)
    
        return {
            "evaluation_analysis": analysis,
            "evaluation_reasoning_brief": reasoning_brief,
        }
        
    def save_results(self, state: SimpleModelAgentState):
        self._log_action("save_results", "start", "Persist run artifacts", "Enables reproducibility")

        """Save comprehensive results to timestamped JSON file"""
        self.logger.info('----------------------- Entering save_results -----------------------')


        def _extract_query_response_pairs(messages):
            """
            Extract (query, response) pairs from LangChain message objects.
            """
            pairs = []

            # Step 1: collect tool_call_id -> query mapping
            tool_call_map = {}

            for msg in messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tool_call in msg.tool_calls:

                        call_id = tool_call.get("id")

                        query = None

                        # Case 1: args stored directly
                        if "args" in tool_call:
                            query = tool_call["args"].get("query")

                        # Case 2: arguments stored as JSON string
                        elif "function" in tool_call:
                            args_str = tool_call["function"].get("arguments", "{}")
                            try:
                                args = json.loads(args_str)
                                query = args.get("query")
                            except json.JSONDecodeError:
                                query = None

                        if call_id and query:
                            tool_call_map[call_id] = query

                # Step 2: match ToolMessage responses
                for msg in messages:
                    if isinstance(msg, ToolMessage):
                        tool_call_id = msg.tool_call_id

                        if tool_call_id in tool_call_map:
                            query = tool_call_map[tool_call_id]
                            response = msg.content

                            pairs.append({
                                "query": query,
                                "response": response
                            })

                return pairs

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        results_dir = os.path.join(OUTPUT_DIR, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Collect dataset information
        dataset_names = [os.path.splitext(os.path.basename(path))[0] for path in state.get("datasets", [])]

        # Collect blocking metrics
        blocking_metrics = {}
        blocking_strategies = state.get("blocking_config", {}).get("blocking_strategies", {})
        for pair_key, config in blocking_strategies.items():
            if isinstance(config, dict):
                blocking_metrics[pair_key] = {
                    "pair_completeness": config.get("pair_completeness", 0),
                    "num_candidates": config.get("num_candidates", 0),
                    "stategy": config.get("strategy", "unknown"),
                    "columns": config.get("columns", [])

                }

        # Collect matching metrics
        matching_metrics = {}
        matching_strategies = state.get("matching_config", {}).get("matching_strategies", {})
        for pair_key, config in matching_strategies.items():
            if isinstance(config, dict):
                matching_metrics[pair_key] = {
                    "f1_score": config.get("f1", 0)
                }

        # Collect evaluation metrics
        evaluation_metrics = state.get("evaluation_metrics", {})
        best_validation_metrics = state.get("best_validation_metrics", {})
        validation_metrics_final = (
            best_validation_metrics
            if self._is_metrics_payload(best_validation_metrics)
            else evaluation_metrics
        )
        sealed_test_metrics_final = state.get("final_test_evaluation_metrics", {})
        evaluation_analysis = state.get("evaluation_analysis", "")
        reasoning_brief = state.get("evaluation_reasoning_brief", {})
        if not isinstance(reasoning_brief, dict) or not reasoning_brief:
            reasoning_brief = _build_reasoning_brief(evaluation_analysis)
        investigator_decision = state.get("investigator_decision", "")
        routing_decision = state.get("investigator_routing_decision", {})
        cycle_audit = list(state.get("evaluation_cycle_audit", []) or [])
        latest_cycle = cycle_audit[-1] if cycle_audit else {}
        latest_problems = _compact_report_list(
            latest_cycle.get("problems", []) if isinstance(latest_cycle, dict) else [],
            limit=4,
        )
        action_plan = state.get("investigator_action_plan", [])
        top_actions: List[str] = []
        if isinstance(action_plan, list):
            for item in action_plan[:3]:
                if not isinstance(item, dict):
                    continue
                action_text = _clean_report_text(item.get("action", ""))
                if action_text:
                    top_actions.append(action_text)
        route_label_map = {
            "run_matching_tester": "rerun matching search",
            "normalization_node": "rerun normalization",
            "pipeline_adaption": "iterate pipeline adaptation",
            "human_review_export": "finish automatic passes and export for human review",
        }
        route_label = route_label_map.get(investigator_decision, investigator_decision or "not recorded")
        route_score = None
        route_threshold = None
        if isinstance(routing_decision, dict):
            route_score = routing_decision.get("score")
            route_threshold = routing_decision.get("threshold")
        agent_run_summary = {
            "agent_loop_overview": (
                "The agent profiles the datasets, optionally reruns normalization, keeps the tested "
                "blocking setup, searches or refreshes matching when needed, generates a fusion pipeline, "
                "generates and runs an evaluation script, diagnoses the result, and then routes either "
                "back to matching, back to normalization, back to pipeline adaptation, or to final human review."
            ),
            "final_route": route_label,
            "current_problem": reasoning_brief.get("problem", ""),
            "next_step_advice": reasoning_brief.get("next_step", ""),
            "normalization_note": reasoning_brief.get("normalization", ""),
            "report_takeaway": reasoning_brief.get("takeaway", ""),
            "top_detected_problems": latest_problems,
            "top_planned_actions": top_actions,
        }

        # Collect pipeline execution info
        pipeline_info = {
            "evaluation_attempts": state.get("evaluation_attempts", 0),
            "matcher_mode": state.get("matcher_mode", "unknown"),
        }

        # Collect data profiles summary
        data_profiles_summary = {}
        data_profiles = state.get("data_profiles", {})
        for path, profile in data_profiles.items():
            if isinstance(profile, dict):
                dataset_name = os.path.splitext(os.path.basename(path))[0]
                data_profiles_summary[dataset_name] = {
                    "num_rows": profile.get("num_rows", 0),
                    "num_columns": profile.get("num_columns", 0),
                    "columns": list(profile.get("dtypes", {}).keys())[:10]  # First 10 columns
                }

        # Compile comprehensive results
        results = {
            "timestamp": timestamp,
            "generated_at": generated_at,
            "run_id": self.run_id,
            "run_output_root": self.run_output_root,

            # Basic run information
            "datasets": dataset_names,
            "matcher_mode": state.get("matcher_mode", "unknown"),
            "fusion_testset": state.get("fusion_testset", "").split("/")[-1],
            "validation_fusion_testset": state.get("validation_fusion_testset", "").split("/")[-1] if state.get("validation_fusion_testset") else "",
            "sealed_evaluation_active": self._sealed_eval_active(state),

            # Performance metrics
            "blocking_metrics": blocking_metrics,
            "matching_metrics": matching_metrics,
            "evaluation_metrics": evaluation_metrics,
            "overall_accuracy": evaluation_metrics.get("overall_accuracy", 0),
            "validation_metrics_final": validation_metrics_final,
            "sealed_test_metrics_final": sealed_test_metrics_final,
            "metrics_by_split": {
                "validation_final": validation_metrics_final,
                "sealed_test_final": sealed_test_metrics_final,
            },

            # Configurations used
            "blocking_config": state.get("blocking_config", {}),
            "matching_config": state.get("matching_config", {}),
            "fusion_size_comparison": state.get("fusion_size_comparison", {}),
            "auto_diagnostics": state.get("auto_diagnostics", {}),
            "evaluation_regression_guard": state.get("evaluation_regression_guard", {}),
            "latest_validation_metrics": state.get("latest_validation_metrics", {}),
            "evaluation_metrics_raw": state.get("evaluation_metrics_raw", {}),
            "evaluation_metrics_for_adaptation": state.get("evaluation_metrics_for_adaptation", {}),
            "best_validation_metrics": state.get("best_validation_metrics", {}),
            "evaluation_cycle_audit": state.get("evaluation_cycle_audit", []),
            "correspondence_integrity": state.get("correspondence_integrity", {}),
            "evaluation_analysis": evaluation_analysis,
            "evaluation_reasoning_brief": reasoning_brief,
            "pipeline_snapshots": state.get("pipeline_snapshots", []),
            "evaluation_snapshots": state.get("evaluation_snapshots", []),
            "normalization_execution_result": state.get("normalization_execution_result", ""),
            "normalization_attempts": state.get("normalization_attempts", 0),
            "normalization_report": state.get("normalization_report", {}),
            "normalization_directives": state.get("normalization_directives", {}),
            "investigator_action_plan": state.get("investigator_action_plan", []),
            "fusion_guidance": state.get("fusion_guidance", {}),
            "investigator_decision": investigator_decision,
            "investigator_routing_decision": routing_decision,
            "matching_refresh_gate": state.get("matching_refresh_gate", {}),
            "cross_run_memory_enabled": state.get("cross_run_memory_enabled", False),
            "normalized_datasets": state.get("normalized_datasets", []),
            "integration_diagnostics_execution_result": state.get("integration_diagnostics_execution_result", ""),
            "integration_diagnostics_report": state.get("integration_diagnostics_report", {}),
            "investigator_action_plan": state.get("investigator_action_plan", []),
            "human_review_execution_result": state.get("human_review_execution_result", ""),
            "human_review_report": state.get("human_review_report", {}),
            "final_test_evaluation_execution_result": state.get("final_test_evaluation_execution_result", ""),
            "final_test_evaluation_metrics": state.get("final_test_evaluation_metrics", {}),
            "agent_run_summary": agent_run_summary,

            # Token usage
            "token_usage": self.token_usage.copy(),

            # Pipeline execution details
            "pipeline_info": pipeline_info,

            # Search tool responses
            "pipeline_search_tool_responses": _extract_query_response_pairs(state.get("messages", [])[2:-1]),
            "evaluation_search_tool_responses": _extract_query_response_pairs((state.get("eval_messages") or [])[2:-1]),

            # Code and execution
            "integration_pipeline_code_length": len(state.get("integration_pipeline_code", "")),
            "evaluation_code_length": len(state.get("evaluation_code", "")),
            "integration_diagnostics_code_length": len(state.get("integration_diagnostics_code", "")),
            "human_review_code_length": len(state.get("human_review_code", "")),

        }

        # Save to timestamped file
        results_file = os.path.join(results_dir, f"run_{timestamp}.json")
        run_audit_file = os.path.join(results_dir, f"run_audit_{timestamp}.json")
        run_report_file = os.path.join(results_dir, f"run_report_{timestamp}.md")
        results["run_audit_path"] = run_audit_file
        results["run_report_path"] = run_report_file

        run_audit = {
            "timestamp": timestamp,
            "generated_at": generated_at,
            "run_id": self.run_id,
            "run_output_root": self.run_output_root,
            "validation_metrics_final": validation_metrics_final,
            "sealed_test_metrics_final": sealed_test_metrics_final,
            "correspondence_integrity": state.get("correspondence_integrity", {}),
            "cycles": cycle_audit,
            "pipeline_snapshots": state.get("pipeline_snapshots", []),
            "evaluation_snapshots": state.get("evaluation_snapshots", []),
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        with open(run_audit_file, "w", encoding="utf-8") as f:
            json.dump(run_audit, f, indent=2, ensure_ascii=False, default=str)

        report_lines: List[str] = []
        report_lines.append(f"# Run Report ({timestamp})")
        report_lines.append("")
        report_lines.append(f"- Generated at (UTC): `{generated_at}`")
        report_lines.append(f"- Run ID: `{self.run_id}`")
        report_lines.append(f"- Run output root: `{self.run_output_root}`")
        report_lines.append(f"- Validation overall (final): `{validation_metrics_final.get('overall_accuracy', 'n/a')}`")
        report_lines.append(f"- Validation macro (final): `{validation_metrics_final.get('macro_accuracy', 'n/a')}`")
        report_lines.append(f"- Sealed test overall (final): `{sealed_test_metrics_final.get('overall_accuracy', 'n/a')}`")
        report_lines.append(f"- Sealed test macro (final): `{sealed_test_metrics_final.get('macro_accuracy', 'n/a')}`")
        report_lines.append(f"- Correspondence structurally valid: `{state.get('correspondence_integrity', {}).get('structurally_valid', 'n/a')}`")
        report_lines.append(f"- Final route: `{route_label}`")
        if route_score is not None and route_threshold is not None:
            report_lines.append(f"- Last routing score / threshold: `{route_score}` / `{route_threshold}`")
        report_lines.append("")
        report_lines.append("## Agent Overview")
        report_lines.append(agent_run_summary["agent_loop_overview"])
        report_lines.append("")
        report_lines.append("## Run Narrative")
        report_lines.append(f"- Main problem: {reasoning_brief.get('problem', 'n/a') or 'n/a'}")
        report_lines.append(f"- Next-step advice: {reasoning_brief.get('next_step', 'n/a') or 'n/a'}")
        if reasoning_brief.get("normalization"):
            report_lines.append(f"- Normalization note: {reasoning_brief.get('normalization')}")
        report_lines.append(f"- Report takeaway: {reasoning_brief.get('takeaway', 'n/a') or 'n/a'}")
        if latest_problems:
            report_lines.append(f"- Latest detected problems: {' | '.join(latest_problems)}")
        if top_actions:
            report_lines.append(f"- Top planned actions: {' | '.join(top_actions)}")
        report_lines.append("")
        report_lines.append("## Attempt Timeline")
        report_lines.append(
            "| attempt | recorded_at | raw_overall | accepted_overall | guard_rejected | structural_valid | invalid_pairs |"
        )
        report_lines.append("|---|---|---:|---:|---|---|---|")
        if cycle_audit:
            for entry in cycle_audit:
                guard = entry.get("regression_guard", {}) if isinstance(entry.get("regression_guard", {}), dict) else {}
                corr = entry.get("correspondence_integrity", {}) if isinstance(entry.get("correspondence_integrity", {}), dict) else {}
                raw = entry.get("raw_metrics", {}) if isinstance(entry.get("raw_metrics", {}), dict) else {}
                accepted = entry.get("accepted_metrics", {}) if isinstance(entry.get("accepted_metrics", {}), dict) else {}
                report_lines.append(
                    f"| {entry.get('attempt', '')} | {entry.get('recorded_at', '')} | "
                    f"{raw.get('overall_accuracy', 'n/a')} | {accepted.get('overall_accuracy', 'n/a')} | "
                    f"{guard.get('rejected', False)} | {corr.get('structurally_valid', 'n/a')} | "
                    f"{', '.join(corr.get('invalid_pairs', [])[:4])} |"
                )
        else:
            report_lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

        with open(run_report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines).strip() + "\n")

        print(f"[+] Results saved to: {results_file}")
        print(f"[+] Run audit saved to: {run_audit_file}")
        print(f"[+] Run report saved to: {run_report_file}")

        self.logger.info(f'Results saved to {results_file}')
        self.logger.info(f'Run audit saved to {run_audit_file}')
        self.logger.info(f'Run report saved to {run_report_file}')
        self.logger.info('Leaving save_results')

        return {
            "run_audit_path": run_audit_file,
            "run_report_path": run_report_file,
            "validation_metrics_final": validation_metrics_final,
            "sealed_test_metrics_final": sealed_test_metrics_final,
            "run_id": self.run_id,
            "run_output_root": self.run_output_root,
        }


# ## Invoke Pipeline

# In[ ]:


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
        model="gpt-5.2",
        temperature=0,
        max_tokens=None,
        request_timeout=LLM_REQUEST_TIMEOUT,
        max_retries=OPENAI_MAX_RETRIES,
    )


# In[ ]:


# Music use case (commented out)
# entity_matching_testsets = {
#     ("discogs", "lastfm"): INPUT_DIR + "gen_TrainSets/discogs_lastfm_goldstandard_blocking.csv",
#     ("discogs", "musicbrainz"): INPUT_DIR + "gen_TrainSets/discogs_musicbrainz_goldstandard_blocking.csv",
#     ("musicbrainz", "lastfm"): INPUT_DIR + "gen_TrainSets/musicbrainz_lastfm_goldstandard_blocking.csv",
# }
# datasets = [
#     INPUT_DIR + "datasets/music/discogs.xml",
#     INPUT_DIR + "datasets/music/lastfm.xml",
#     INPUT_DIR + "datasets/music/musicbrainz.xml"
# ]
# fusion_testset = INPUT_DIR + "datasets/music/testsets/test_set.xml"
# validation_fusion_testset = INPUT_DIR + "datasets/music/input/fusion/validation_set.xml"

# Games use case
# Using canonical games testsets (not generated train sets)
entity_matching_testsets = {
    ("dbpedia", "sales"): INPUT_DIR + "datasets/games/testsets/dpedia_games_sales_goldstandard_blocking.csv",
    ("dbpedia", "metacritic"): INPUT_DIR + "datasets/games/testsets/dpedia_games_metacritic_goldstandard_blocking.csv",
    ("sales", "metacritic"): INPUT_DIR + "datasets/games/testsets/sales_metacritic_goldstandard_blocking.csv",
}
datasets = [
    INPUT_DIR + "datasets/games/dbpedia.xml",
    INPUT_DIR + "datasets/games/metacritic.xml",
    INPUT_DIR + "datasets/games/sales.xml"
]
fusion_testset = INPUT_DIR + "datasets/games/testsets/test_set.xml"
# Optional sealed-eval input: set validation_fusion_testset when available
# validation_fusion_testset = INPUT_DIR + "datasets/games/testsets/validation_set.xml"



# In[ ]:


# # Restaurants use case
# entity_matching_testsets = {
#     ("kaggle_small", "uber_eats_small"): INPUT_DIR + "datasets/restaurant/testsets/kaggle_uber_eats_goldstandard_blocking_small.csv",
#     ("kaggle_small", "yelp_small"): INPUT_DIR + "datasets/restaurant/testsets/kaggle_yelp_goldstandard_blocking_small.csv",
#     ("uber_eats_small", "yelp_small"): INPUT_DIR + "datasets/restaurant/testsets/uber_eats_yelp_goldstandard_blocking_small.csv",
# }

# datasets = [
#     INPUT_DIR + "datasets/restaurant/kaggle_small.parquet",
#     INPUT_DIR + "datasets/restaurant/uber_eats_small.parquet",
#     INPUT_DIR + "datasets/restaurant/yelp_small.parquet"
# ]
# fusion_testset = INPUT_DIR + "datasets/restaurant/testsets/Restaurant_Fusion_Test_Set.csv"


# In[ ]:


# # Games use case
# entity_matching_testsets= {
#     ("dbpedia","sales"): INPUT_DIR + "datasets/games/testsets/dpedia_games_sales_goldstandard_blocking.csv",
#     ("dbpedia","metacritic"): INPUT_DIR + "datasets/games/testsets/dpedia_games_metacritic_goldstandard_blocking.csv",
#     ("sales","metacritic"): INPUT_DIR + "datasets/games/testsets/sales_metacritic_goldstandard_blocking.csv",
# }

# datasets= [
#     INPUT_DIR + "datasets/games/dbpedia.xml",
#     INPUT_DIR + "datasets/games/metacritic.xml",
#     INPUT_DIR + "datasets/games/sales.xml"
# ]
# fusion_testset= INPUT_DIR + "datasets/games/testsets/test_set.xml"


# In[ ]:


# # Books use case
# entity_matching_testsets= {
#     ("goodreads_small","amazon_small"): INPUT_DIR + "datasets/books/testsets/goodreads_2_amazon.csv",
#     ("metabooks_small","amazon_small"): INPUT_DIR + "datasets/books/testsets/metabooks_2_amazon.csv",
#     ("metabooks_small","goodreads_small"): INPUT_DIR + "datasets/books/testsets/metabooks_2_goodreads.csv",
# }

# datasets= [
#     INPUT_DIR + "datasets/books/amazon_small.parquet",
#     INPUT_DIR + "datasets/books/goodreads_small.parquet",
#     INPUT_DIR + "datasets/books/metabooks_small.parquet"
# ]
# fusion_testset= INPUT_DIR + "datasets/books/testsets/golden_fused_books.csv"


# In[ ]:


# Music use case (commented out)
entity_matching_testsets = {
     ("discogs", "lastfm"): INPUT_DIR + "gen_TrainSets/discogs_lastfm_goldstandard_blocking.csv",
     ("discogs", "musicbrainz"): INPUT_DIR + "gen_TrainSets/discogs_musicbrainz_goldstandard_blocking.csv",
     ("musicbrainz", "lastfm"): INPUT_DIR + "gen_TrainSets/musicbrainz_lastfm_goldstandard_blocking.csv",
 }
datasets = [
     INPUT_DIR + "datasets/music/discogs.xml",
     INPUT_DIR + "datasets/music/lastfm.xml",
     INPUT_DIR + "datasets/music/musicbrainz.xml"
 ]
fusion_testset = INPUT_DIR + "datasets/music/testsets/test_set.xml"
validation_fusion_testset = INPUT_DIR + "datasets/music/testsets/validation_set.xml"


# In[ ]:


# Skip Blocking and Matching node and use existing strategies saved in 
# "output/blocking-evaluation/blocking_config.json" and "output/matching-evaluation/matching_config.json"
SKIP_BLOCKING_TESTER=False
SKIP_MATCHING_TESTER=False


# In[ ]:


# Music Dataset
# prepare agent
# agent = SimpleModelAgent(llm, ProfileDatasetTool())
profile_tool = ProfileDatasetTool()
search_tool = SearchDocumentationTool()
all_tools = {profile_tool.name: profile_tool, search_tool.name: search_tool}

agent = SimpleModelAgent(llm, tools=all_tools)

# invoke agent
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

# Optional: enable sealed evaluation mode if validation set is provided.
# In sealed mode, adaptation/evaluation loops use validation_fusion_testset,
# and a final one-time held-out test evaluation runs on fusion_testset.
if "validation_fusion_testset" in globals() and validation_fusion_testset:
    invoke_payload["validation_fusion_testset"] = validation_fusion_testset

result = agent.graph.invoke(
    invoke_payload,
    config={"recursion_limit": 200},
)



# In[ ]:


# from IPython.display import Image, display

# display(Image(agent.graph.get_graph().draw_mermaid_png()))


# In[ ]:


entity_matching_testsets = {
    ("dbpedia", "sales"): INPUT_DIR + "datasets/games/testsets/dbpedia_2_sales_test.csv",
    ("metacritic", "dbpedia"): INPUT_DIR + "datasets/games/testsets/metacritic_2_dbpedia_test.csv",
    ("metacritic", "sales"): INPUT_DIR + "datasets/games/testsets/metacritic_2_sales_test.csv",
}
datasets = [
    INPUT_DIR + "datasets/games/dbpedia.xml",
    INPUT_DIR + "datasets/games/metacritic.xml",
    INPUT_DIR + "datasets/games/sales.xml"
]
fusion_testset = INPUT_DIR + "datasets/games/testsets/test_set_fusion.xml"
validation_fusion_testset = INPUT_DIR + "datasets/games/testsets/validation_set_fusion.xml"


# In[ ]:


# Skip Blocking and Matching node and use existing strategies saved in 
# "output/blocking-evaluation/blocking_config.json" and "output/matching-evaluation/matching_config.json"
SKIP_BLOCKING_TESTER=False
SKIP_MATCHING_TESTER=False


# In[ ]:


# Games Dataset
# prepare agent
# agent = SimpleModelAgent(llm, ProfileDatasetTool())
profile_tool = ProfileDatasetTool()
search_tool = SearchDocumentationTool()
all_tools = {profile_tool.name: profile_tool, search_tool.name: search_tool}

agent = SimpleModelAgent(llm, tools=all_tools)

# invoke agent
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

# Optional: enable sealed evaluation mode if validation set is provided.
# In sealed mode, adaptation/evaluation loops use validation_fusion_testset,
# and a final one-time held-out test evaluation runs on fusion_testset.
if "validation_fusion_testset" in globals() and validation_fusion_testset:
    invoke_payload["validation_fusion_testset"] = validation_fusion_testset

result = agent.graph.invoke(
    invoke_payload,
    config={"recursion_limit": 200},
)


# In[ ]:


# Company Datasets
entity_matching_testsets = {
    ("forbes", "dbpedia"): INPUT_DIR + "datasets/companies/testsets/forbes_dbpedia_art.csv",
    ("forbes", "fullcontact"): INPUT_DIR + "datasets/companies/testsets/forbes_fullcontact_art.csv",
    ("fullcontact", "dbpedia"): INPUT_DIR + "datasets/companies/testsets/fullcontact_dbpedia_art.csv",
}
datasets = [
    INPUT_DIR + "datasets/companies/forbes.csv",
    INPUT_DIR + "datasets/companies/dbpedia.xml",
    INPUT_DIR + "datasets/companies/fullcontact.xml"
]
fusion_testset = INPUT_DIR + "datasets/companies/testsets/test_set.xml"
validation_fusion_testset = INPUT_DIR + "datasets/companies/testsets/validation_set.xml"


# In[ ]:


# Skip Blocking and Matching node and use existing strategies saved in 
# "output/blocking-evaluation/blocking_config.json" and "output/matching-evaluation/matching_config.json"
SKIP_BLOCKING_TESTER=False
SKIP_MATCHING_TESTER=False


# In[ ]:


# Company Dataset
# prepare agent
# agent = SimpleModelAgent(llm, ProfileDatasetTool())
profile_tool = ProfileDatasetTool()
search_tool = SearchDocumentationTool()
all_tools = {profile_tool.name: profile_tool, search_tool.name: search_tool}

agent = SimpleModelAgent(llm, tools=all_tools)

# invoke agent
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

# Optional: enable sealed evaluation mode if validation set is provided.
# In sealed mode, adaptation/evaluation loops use validation_fusion_testset,
# and a final one-time held-out test evaluation runs on fusion_testset.
if "validation_fusion_testset" in globals() and validation_fusion_testset:
    invoke_payload["validation_fusion_testset"] = validation_fusion_testset

result = agent.graph.invoke(
    invoke_payload,
    config={"recursion_limit": 200},
)
