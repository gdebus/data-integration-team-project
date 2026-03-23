import ast
import difflib
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage


def configure_workflow_logger(
    *,
    output_dir: str,
    logger_name: str = "",
    filename: str = "agent_run.log",
    overwrite: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    log_path = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    abs_log_path = os.path.abspath(log_path)

    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == abs_log_path
        for h in logger.handlers
    ):
        mode = "w" if overwrite else "a"
        handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)

    return logger


def log_workflow_action(
    logger: logging.Logger,
    *,
    step: str,
    action: str,
    why: str,
    improvement: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "step": step,
        "action": action,
        "why": why,
        "improvement": improvement,
    }
    if details:
        payload["details"] = details
    logger.info(json.dumps(payload, ensure_ascii=False))


def log_agent_action(
    agent: Any,
    *,
    step: str,
    action: str,
    why: str,
    improvement: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    logger = getattr(agent, "logger", None)
    if logger is not None:
        log_workflow_action(
            logger,
            step=step,
            action=action,
            why=why,
            improvement=improvement,
            details=details,
        )
        return
    if hasattr(agent, "_log_action"):
        try:
            agent._log_action(step, action, why, improvement, details)
        except Exception:
            return


NODE_SUMMARY_EXTRACTORS: Dict[str, str] = {
    "match_schemas": "_extract_match_schemas_facts",
    "profile_data": "_extract_profile_data_facts",
    "normalization_node": "_extract_normalization_node_facts",
    "run_blocking_tester": "_extract_run_blocking_tester_facts",
    "run_matching_tester": "_extract_run_matching_tester_facts",
    "pipeline_adaption": "_extract_pipeline_adaption_facts",
    "execute_pipeline": "_extract_execute_pipeline_facts",
    "evaluation_node": "_extract_evaluation_node_facts",
    "investigator_node": "_extract_investigator_node_facts",
    "human_review_export": "_extract_human_review_export_facts",
    "sealed_final_test_evaluation": "_extract_sealed_final_test_evaluation_facts",
    "save_results": "_extract_save_results_facts",
}

SUMMARY_MODEL_PRICING_USD_PER_1M: Dict[str, Dict[str, float]] = {
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5.3-chat-latest": {"input": 1.75, "output": 14.00},
    "gpt-5.2-chat-latest": {"input": 1.75, "output": 14.00},
    "gpt-5.1-chat-latest": {"input": 1.25, "output": 10.00},
    "gpt-5-chat-latest": {"input": 1.25, "output": 10.00},
    "gpt-5.3-codex": {"input": 1.75, "output": 14.00},
    "gpt-5.2-codex": {"input": 1.75, "output": 14.00},
    "gpt-5.1-codex-max": {"input": 1.25, "output": 10.00},
    "gpt-5.1-codex": {"input": 1.25, "output": 10.00},
    "gpt-5-codex": {"input": 1.25, "output": 10.00},
    "gpt-5.2-pro": {"input": 21.00, "output": 168.00},
    "gpt-5-pro": {"input": 15.00, "output": 120.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


class WorkflowLogger:
    def __init__(
        self,
        output_dir: str,
        summary_model_name: str = "gpt-4.1-mini",
        summary_char_limit: int = 300,
        notebook_name: Optional[str] = None,
        use_case: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.summary_char_limit = summary_char_limit
        self.summary_model_name = summary_model_name
        self.notebook_name = notebook_name or "AdaptationPipeline"
        self.use_case = str(use_case).strip() if use_case is not None else None
        self.llm_model = str(llm_model).strip() if llm_model is not None else "unknown"

        self._summary_model = None
        self._activity_log_path: Optional[str] = None
        self._pipeline_archive_path: Optional[str] = None

        self._activity_records = []
        self._evaluation_runs = []
        self._run_configs: Dict[str, Any] = {}
        self._node_index = 0
        self._last_record_index: Optional[int] = None
        self._active = False

        self._tracked_file_cache = {}
        self._latest_values_state: Dict[str, Any] = {}
        self._last_token_snapshot: Dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }
        self._pipeline_snapshot_index = 0
        self._pipeline_snapshots = []
        self._pending_snapshot_indices = []
        self._last_snapshot_accuracy_index: Optional[int] = None
        self._archive_matcher_mode = "rule_based"
        self._run_started_at_ns: Optional[int] = None
        self._run_finished_at_ns: Optional[int] = None
        self._pending_node_durations_seconds: Dict[str, List[float]] = {}
        self._has_current_run_density = False
        self._density_cache_signature: Optional[tuple] = None
        self._density_cache_value: Optional[Dict[str, Any]] = None
        self._summ_prompt_tokens = 0
        self._summ_output_tokens = 0
        self._summ_total_tokens = 0
        self._summ_cost_usd = 0.0

        try:
            from langchain_openai import ChatOpenAI

            self._summary_model = ChatOpenAI(model=summary_model_name, temperature=0)
        except Exception:
            self._summary_model = None

    @property
    def active(self) -> bool:
        return self._active

    @staticmethod
    def _safe_part(value: Any, default: str) -> str:
        text = str(value or "").strip()
        text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
        return text or default

    @staticmethod
    def _normalize_matcher_mode(value: Any) -> str:
        raw = str(value or "").strip().lower().replace("-", "_")
        raw = re.sub(r"[^a-z0-9_]+", "_", raw)
        if raw in {"", "none", "null"}:
            return "rule_based"
        if raw == "rulebased":
            return "rule_based"
        return raw

    @staticmethod
    def _next_run_index(results_dir: str, base_prefix: str) -> int:
        pattern = re.compile(
            rf"^{re.escape(base_prefix)}_(\d+)_(?:node_activity\.json|pipelines\.(?:py|md))$"
        )
        max_seen = 0
        try:
            for filename in os.listdir(results_dir):
                match = pattern.match(filename)
                if match:
                    max_seen = max(max_seen, int(match.group(1)))
        except Exception:
            return 1
        return max_seen + 1

    def _write_pipeline_archive_markdown(self):
        if not self._pipeline_archive_path:
            return
        try:
            with open(self._pipeline_archive_path, "w", encoding="utf-8") as f:
                f.write("# Pipeline Snapshots\n\n")
                f.write(f"notebook_name={self.notebook_name}\n")
                f.write(f"matcher_mode={self._archive_matcher_mode}\n\n")
                for snapshot in self._pipeline_snapshots:
                    snapshot_no = int(snapshot.get("snapshot_index", 0))
                    snapshot_no_txt = f"{snapshot_no:02d}"
                    node_index = snapshot.get("node_index")
                    node_name = snapshot.get("node_name", "execute_pipeline")
                    accuracy_score = snapshot.get("accuracy_score")
                    if not accuracy_score:
                        accuracy_score = "pending"
                    pipeline_code = str(snapshot.get("pipeline_code", "")).rstrip()

                    f.write("============================================================\n")
                    f.write(f"PIPELINE SNAPSHOT {snapshot_no_txt} START\n")
                    f.write("============================================================\n")
                    f.write(f"node_index={node_index}\n")
                    f.write(f"node_name={node_name}\n")
                    f.write(f"accuracy_score={accuracy_score}\n")
                    f.write("------------------------------------------------------------\n\n")
                    f.write("```python\n")
                    f.write(pipeline_code)
                    f.write("\n```\n\n")
                    f.write("============================================================\n")
                    f.write(f"PIPELINE SNAPSHOT {snapshot_no_txt} END\n")
                    f.write("============================================================\n\n")
        except Exception:
            pass

    def _attach_accuracy_to_pending_snapshot(self, accuracy_score: Optional[str]) -> Optional[int]:
        if not self._pending_snapshot_indices:
            return None
        idx = self._pending_snapshot_indices.pop(0)
        if idx < 0 or idx >= len(self._pipeline_snapshots):
            return None
        self._pipeline_snapshots[idx]["accuracy_score"] = accuracy_score or "pending"
        self._write_pipeline_archive_markdown()
        return idx

    def _overwrite_snapshot_accuracy(self, snapshot_index: Optional[int], accuracy_score: Optional[str]) -> None:
        if snapshot_index is None:
            return
        if snapshot_index < 0 or snapshot_index >= len(self._pipeline_snapshots):
            return
        if not accuracy_score:
            return
        self._pipeline_snapshots[snapshot_index]["accuracy_score"] = accuracy_score
        self._write_pipeline_archive_markdown()

    def _build_transition_stats(self) -> Dict[str, Any]:
        counts: Dict[str, Dict[str, int]] = {}
        for record in self._activity_records:
            src = str(record.get("current_node", ""))
            dst = str(record.get("next_node", ""))
            if not src or not dst or dst == "PENDING":
                continue
            counts.setdefault(src, {})
            counts[src][dst] = counts[src].get(dst, 0) + 1
        return {
            "transition_counts": counts,
            "total_nodes_logged": len(self._activity_records),
        }

    @staticmethod
    def _round_seconds(value: float) -> float:
        return round(float(value or 0.0), 3)

    def _push_node_duration(self, node_name: str, duration_seconds: float) -> None:
        self._pending_node_durations_seconds.setdefault(node_name, []).append(float(duration_seconds or 0.0))

    def _pop_node_duration(self, node_name: str) -> float:
        queue = self._pending_node_durations_seconds.get(node_name) or []
        if not queue:
            return 0.0
        duration = float(queue.pop(0) or 0.0)
        if not queue:
            self._pending_node_durations_seconds.pop(node_name, None)
        return duration

    def _build_time_complexity(self) -> Dict[str, Any]:
        if self._run_started_at_ns is None:
            total_duration_seconds = 0.0
        else:
            end_ns = self._run_finished_at_ns if self._run_finished_at_ns is not None else time.perf_counter_ns()
            total_duration_seconds = max(0.0, (end_ns - self._run_started_at_ns) / 1_000_000_000)

        sum_logged = 0.0
        per_node: Dict[str, float] = {}
        for record in self._activity_records:
            duration = float(record.get("duration_seconds", 0.0) or 0.0)
            sum_logged += duration
            node_name = str(record.get("current_node", "") or "")
            if node_name:
                per_node[node_name] = per_node.get(node_name, 0.0) + duration

        return {
            "total_duration_seconds": self._round_seconds(total_duration_seconds),
            "sum_of_logged_node_durations_seconds": self._round_seconds(sum_logged),
            "per_node_cumulative_durations_seconds": {
                key: self._round_seconds(value) for key, value in per_node.items()
            },
        }

    def _build_token_complexity(self) -> Dict[str, Any]:
        per_node: Dict[str, Dict[str, int]] = {}
        global_prompt = 0
        global_output = 0
        global_total = 0

        for record in self._activity_records:
            node_name = str(record.get("current_node", "") or "")
            if not node_name:
                continue

            prompt_tokens = int(record.get("prompt_tokens", 0) or 0)
            output_tokens = int(record.get("completion_tokens", 0) or 0)
            total_tokens = int(record.get("total_tokens", 0) or 0)

            node_totals = per_node.setdefault(
                node_name,
                {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
            node_totals["prompt_tokens"] += prompt_tokens
            node_totals["output_tokens"] += output_tokens
            node_totals["total_tokens"] += total_tokens

            global_prompt += prompt_tokens
            global_output += output_tokens
            global_total += total_tokens

        global_cost = self._estimate_agent_call_cost_usd(global_prompt, global_output)

        return {
            "per_node_cumulative_tokens": per_node,
            "global_tokens": {
                "prompt_tokens": global_prompt,
                "output_tokens": global_output,
                "total_tokens": global_total,
                "total_cost_usd": round(float(global_cost), 8),
            },
        }

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _resolve_summary_pricing(self) -> Optional[Dict[str, float]]:
        model = str(self.summary_model_name or "").strip().lower()
        if not model:
            return None
        if model in SUMMARY_MODEL_PRICING_USD_PER_1M:
            return SUMMARY_MODEL_PRICING_USD_PER_1M[model]
        # Longest-key match first avoids collisions (e.g., gpt-5 matching gpt-5-mini).
        for known_model in sorted(SUMMARY_MODEL_PRICING_USD_PER_1M.keys(), key=len, reverse=True):
            pricing = SUMMARY_MODEL_PRICING_USD_PER_1M[known_model]
            if model.startswith(known_model) or known_model in model:
                return pricing
        return None

    def _estimate_summary_call_cost_usd(self, prompt_tokens: int, output_tokens: int) -> float:
        pricing = self._resolve_summary_pricing()
        if not pricing:
            return 0.0
        input_rate = float(pricing.get("input", 0.0) or 0.0)
        output_rate = float(pricing.get("output", 0.0) or 0.0)
        if input_rate <= 0 and output_rate <= 0:
            return 0.0
        in_cost = (max(0, int(prompt_tokens)) / 1_000_000.0) * input_rate
        out_cost = (max(0, int(output_tokens)) / 1_000_000.0) * output_rate
        return max(0.0, in_cost + out_cost)

    def _resolve_agent_pricing(self) -> Optional[Dict[str, float]]:
        model = str(self.llm_model or "").strip().lower()
        if not model:
            return None
        if model in SUMMARY_MODEL_PRICING_USD_PER_1M:
            return SUMMARY_MODEL_PRICING_USD_PER_1M[model]
        # Longest-key match first avoids collisions (e.g., gpt-5 vs gpt-5-mini).
        for known_model in sorted(SUMMARY_MODEL_PRICING_USD_PER_1M.keys(), key=len, reverse=True):
            pricing = SUMMARY_MODEL_PRICING_USD_PER_1M[known_model]
            if model.startswith(known_model) or known_model in model:
                return pricing
        return None

    def _estimate_agent_call_cost_usd(self, prompt_tokens: int, output_tokens: int) -> float:
        pricing = self._resolve_agent_pricing()
        if not pricing:
            return 0.0
        input_rate = float(pricing.get("input", 0.0) or 0.0)
        output_rate = float(pricing.get("output", 0.0) or 0.0)
        if input_rate <= 0 and output_rate <= 0:
            return 0.0
        in_cost = (max(0, int(prompt_tokens)) / 1_000_000.0) * input_rate
        out_cost = (max(0, int(output_tokens)) / 1_000_000.0) * output_rate
        return max(0.0, in_cost + out_cost)

    def _extract_usage_from_response(self, response: Any) -> tuple[int, int, int, float]:
        usage = {}
        response_meta = {}
        if hasattr(response, "usage_metadata") and isinstance(response.usage_metadata, dict):
            usage = response.usage_metadata
        if hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
            response_meta = response.response_metadata
            token_usage = response.response_metadata.get("token_usage")
            if isinstance(token_usage, dict):
                merged = dict(token_usage)
                merged.update({k: v for k, v in usage.items() if k not in merged})
                usage = merged

        prompt_tokens = self._safe_int(
            usage.get("prompt_tokens", usage.get("input_tokens", usage.get("prompt_token_count", 0)))
        )
        output_tokens = self._safe_int(
            usage.get("completion_tokens", usage.get("output_tokens", usage.get("completion_token_count", 0)))
        )
        total_tokens = self._safe_int(usage.get("total_tokens", prompt_tokens + output_tokens))
        if total_tokens <= 0:
            total_tokens = prompt_tokens + output_tokens

        cost_candidates = [
            usage.get("cost"),
            usage.get("total_cost"),
            usage.get("estimated_cost_usd"),
            response_meta.get("cost"),
            response_meta.get("total_cost"),
            response_meta.get("estimated_cost_usd"),
        ]
        cost = 0.0
        for candidate in cost_candidates:
            parsed = self._safe_float(candidate)
            if parsed > 0:
                cost = parsed
                break
        if cost <= 0 and (prompt_tokens > 0 or output_tokens > 0):
            cost = self._estimate_summary_call_cost_usd(prompt_tokens, output_tokens)
        return max(0, prompt_tokens), max(0, output_tokens), max(0, total_tokens), max(0.0, cost)

    def _invoke_summary_model(self, messages: List[Any], purpose: str):
        if self._summary_model is None:
            raise RuntimeError("summary model is not available")
        response = self._summary_model.invoke(messages)
        prompt_tokens, output_tokens, total_tokens, cost_usd = self._extract_usage_from_response(response)
        self._summ_prompt_tokens += prompt_tokens
        self._summ_output_tokens += output_tokens
        self._summ_total_tokens += total_tokens
        self._summ_cost_usd += cost_usd
        return response

    def _build_summarization_tokens(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": int(self._summ_prompt_tokens),
            "output_tokens": int(self._summ_output_tokens),
            "total_tokens": int(self._summ_total_tokens),
            "estimated_cost_usd": round(float(self._summ_cost_usd), 8),
        }

    def _stage_from_comparisons(self, comparisons: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(comparisons, dict):
            return {}
        stage = None
        payload = None
        if isinstance(comparisons.get("matching"), dict):
            stage = "matching"
            payload = comparisons.get("matching")
        elif isinstance(comparisons.get("blocking"), dict):
            stage = "blocking"
            payload = comparisons.get("blocking")
        if not isinstance(payload, dict):
            return {}

        estimated_rows = payload.get("expected_rows")
        actual_rows = payload.get("actual_rows")
        estimated = self._safe_float(estimated_rows)
        actual = self._safe_float(actual_rows)
        density = None
        if estimated > 0:
            density = round(actual / estimated, 6)
        return {
            "stage": stage,
            "estimated_rows": self._safe_int(estimated_rows) if estimated_rows is not None else None,
            "actual_rows": self._safe_int(actual_rows) if actual_rows is not None else None,
            "density": density,
        }

    def _resolve_fusion_size_artifact_paths(self) -> Dict[str, Optional[str]]:
        estimate_candidates: List[str] = []
        fusion_candidates: List[str] = []

        try:
            from fusion_size_monitor import estimate_path_for_output_dir

            monitor_estimate = estimate_path_for_output_dir(self.output_dir)
            if isinstance(monitor_estimate, str) and monitor_estimate.strip():
                estimate_candidates.append(monitor_estimate)
        except Exception:
            try:
                from agents.fusion_size_monitor import estimate_path_for_output_dir

                monitor_estimate = estimate_path_for_output_dir(self.output_dir)
                if isinstance(monitor_estimate, str) and monitor_estimate.strip():
                    estimate_candidates.append(monitor_estimate)
            except Exception:
                pass

        estimate_candidates.extend(
            [
                os.path.join(self.output_dir, "pipeline_evaluation", "fusion_size_estimate.json"),
                os.path.join("output", "pipeline_evaluation", "fusion_size_estimate.json"),
                os.path.join("agents", "output", "pipeline_evaluation", "fusion_size_estimate.json"),
            ]
        )
        fusion_candidates.extend(
            [
                os.path.join(self.output_dir, "data_fusion", "fusion_data.csv"),
                os.path.join("output", "data_fusion", "fusion_data.csv"),
                os.path.join("agents", "output", "data_fusion", "fusion_data.csv"),
            ]
        )

        def _first_existing(candidates: List[str]) -> Optional[str]:
            seen = set()
            for candidate in candidates:
                abs_candidate = os.path.abspath(candidate)
                if abs_candidate in seen:
                    continue
                seen.add(abs_candidate)
                if os.path.exists(abs_candidate):
                    return abs_candidate
            return None

        return {
            "estimate_path": _first_existing(estimate_candidates),
            "fusion_csv_path": _first_existing(fusion_candidates),
        }

    def _extract_density_from_comparisons(self, comparisons: Dict[str, Any]) -> Dict[str, Any]:
        base = {
            "stage": None,
            "estimated_rows": None,
            "actual_rows": None,
            "density": None,
        }
        stage_payload = self._stage_from_comparisons(comparisons)
        if stage_payload:
            return stage_payload
        return base

    def _load_density_payload_from_monitor(self, estimate_path: str, fusion_csv_path: Optional[str]) -> Dict[str, Any]:
        # Use fusion_size_monitor's own comparison function to avoid re-implementing estimator logic.
        if fusion_csv_path:
            try:
                from fusion_size_monitor import compare_estimates_with_actual

                compared = compare_estimates_with_actual(
                    fusion_csv_path=fusion_csv_path,
                    estimate_path=estimate_path,
                )
                if isinstance(compared, dict):
                    comparisons = compared.get("comparisons", {})
                    if isinstance(comparisons, dict):
                        return {"comparisons": comparisons}
            except Exception:
                try:
                    from agents.fusion_size_monitor import compare_estimates_with_actual

                    compared = compare_estimates_with_actual(
                        fusion_csv_path=fusion_csv_path,
                        estimate_path=estimate_path,
                    )
                    if isinstance(compared, dict):
                        comparisons = compared.get("comparisons", {})
                        if isinstance(comparisons, dict):
                            return {"comparisons": comparisons}
                except Exception:
                    pass

        # Fallback to reading the monitor artifact directly.
        try:
            with open(estimate_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {}

    def _build_density(self) -> Dict[str, Any]:
        base = {
            "stage": None,
            "estimated_rows": None,
            "actual_rows": None,
            "density": None,
        }
        artifact_paths = self._resolve_fusion_size_artifact_paths()
        estimate_path = artifact_paths.get("estimate_path")
        fusion_csv_path = artifact_paths.get("fusion_csv_path")
        if not estimate_path:
            return base

        pre_estimate_mtime = os.path.getmtime(estimate_path) if os.path.exists(estimate_path) else None
        pre_fusion_mtime = os.path.getmtime(fusion_csv_path) if fusion_csv_path and os.path.exists(fusion_csv_path) else None
        signature = (estimate_path, pre_estimate_mtime, fusion_csv_path, pre_fusion_mtime)
        if signature == self._density_cache_signature and isinstance(self._density_cache_value, dict):
            return self._density_cache_value

        try:
            payload = self._load_density_payload_from_monitor(estimate_path, fusion_csv_path)
            density = self._extract_density_from_comparisons(payload.get("comparisons", {}))
            post_estimate_mtime = os.path.getmtime(estimate_path) if os.path.exists(estimate_path) else None
            post_fusion_mtime = os.path.getmtime(fusion_csv_path) if fusion_csv_path and os.path.exists(fusion_csv_path) else None
            self._density_cache_signature = (estimate_path, post_estimate_mtime, fusion_csv_path, post_fusion_mtime)
            self._density_cache_value = density
            return density
        except Exception:
            return base

    def start_run(self, state: Dict[str, Any], token_usage: Optional[Dict[str, Any]] = None):
        matcher_mode = "rule_based"
        if isinstance(state, dict):
            matcher_mode = state.get("matcher_mode", "rule_based")
        matcher_mode = self._normalize_matcher_mode(matcher_mode)

        results_dir = os.path.join(self.output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        safe_notebook = self._safe_part(self.notebook_name, "AdaptationPipeline")
        safe_mode = self._safe_part(matcher_mode, "rule_based")
        yymmdd = datetime.now().strftime("%y%m%d")
        base_prefix = f"{safe_notebook}_{safe_mode}_{yymmdd}"
        run_index = self._next_run_index(results_dir, base_prefix)
        run_index_str = f"{run_index:02d}"

        self._activity_log_path = os.path.join(
            results_dir, f"{base_prefix}_{run_index_str}_node_activity.json"
        )
        self._pipeline_archive_path = os.path.join(
            results_dir, f"{base_prefix}_{run_index_str}_pipelines.md"
        )

        self._activity_records = []
        self._evaluation_runs = []
        self._run_configs = {}
        self._node_index = 0
        self._last_record_index = None
        self._tracked_file_cache = {}
        self._latest_values_state = {}
        if isinstance(token_usage, dict):
            self._last_token_snapshot = {
                "prompt_tokens": int(token_usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(token_usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(token_usage.get("total_tokens", 0) or 0),
                "total_cost": float(token_usage.get("total_cost", 0.0) or 0.0),
            }
        else:
            self._last_token_snapshot = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
            }
        self._pipeline_snapshot_index = 0
        self._pipeline_snapshots = []
        self._pending_snapshot_indices = []
        self._last_snapshot_accuracy_index = None
        self._archive_matcher_mode = safe_mode
        self._run_started_at_ns = time.perf_counter_ns()
        self._run_finished_at_ns = None
        self._pending_node_durations_seconds = {}
        self._has_current_run_density = False
        self._density_cache_signature = None
        self._density_cache_value = None
        self._summ_prompt_tokens = 0
        self._summ_output_tokens = 0
        self._summ_total_tokens = 0
        self._summ_cost_usd = 0.0
        self._active = True

        self._write_activity_payload()
        self._write_pipeline_archive_markdown()

    def finish_run(self, final_next_node: str = "END"):
        if self._last_record_index is not None:
            self._activity_records[self._last_record_index]["next_node"] = final_next_node
        self._run_finished_at_ns = time.perf_counter_ns()
        self._write_activity_payload()
        self._active = False

    def relocate_output(self, new_output_dir: str):
        """Re-scope activity log and pipeline archive paths to a new output directory.

        Called when configure_run_output() redirects output to a run-scoped
        directory after start_run() has already set paths using the old dir.
        """
        self.output_dir = new_output_dir
        if self._activity_log_path:
            basename = os.path.basename(self._activity_log_path)
            results_dir = os.path.join(new_output_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            self._activity_log_path = os.path.join(results_dir, basename)
        if self._pipeline_archive_path:
            basename = os.path.basename(self._pipeline_archive_path)
            results_dir = os.path.join(new_output_dir, "results")
            self._pipeline_archive_path = os.path.join(results_dir, basename)

    def mark_next_for_previous(self, next_node: str):
        if self._last_record_index is None:
            return
        prev = self._activity_records[self._last_record_index]
        if prev.get("next_node") in (None, "PENDING"):
            prev["next_node"] = next_node

    def set_run_config(self, config_key: str, config_value: Any):
        if not config_key or config_value is None:
            return
        self._run_configs[config_key] = config_value
        self._write_activity_payload()

    def append_evaluation_run(self, run_record: Dict[str, Any]):
        if not isinstance(run_record, dict):
            return
        self._evaluation_runs.append(run_record)
        self._write_activity_payload()

    def _write_activity_payload(self):
        if not self._activity_log_path:
            return
        payload: Dict[str, Any] = {}
        if self.use_case:
            payload["use_case"] = self.use_case
        payload["LLM_model"] = self.llm_model or "unknown"
        payload["node_activity"] = self._activity_records
        if self._run_configs:
            payload["run_configs"] = self._run_configs
        if self._evaluation_runs:
            payload["evaluation_runs"] = self._evaluation_runs
        payload["transition_stats"] = self._build_transition_stats()
        payload["time_complexity"] = self._build_time_complexity()
        payload["token_complexity"] = self._build_token_complexity()
        payload["summarization_tokens"] = self._build_summarization_tokens()
        payload["density"] = self._build_density()
        with open(self._activity_log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _coerce_response_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text_val = item.get("text") or item.get("content") or item.get("output_text")
                    if isinstance(text_val, str):
                        parts.append(text_val)
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join([p for p in parts if p]).strip()
        if isinstance(content, dict):
            text_val = content.get("text") or content.get("content") or content.get("output_text")
            if isinstance(text_val, str):
                return text_val
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    def _summary_limit_for_node(self, node_name: str) -> int:
        caps = {
            "pipeline_adaption": 1250,
            "evaluation_adaption": 1250,
            "run_matching_tester": 750,
            "run_blocking_tester": 750,
            "normalization_node": 900,
            "evaluation_node": 750,
            "investigator_node": 900,
            "human_review_export": 750,
            "sealed_final_test_evaluation": 500,
            "save_results": 500,
            "profile_data": 500,
            "match_schemas": 500,
            "evaluation_reasoning": 1250,
            "evaluation_decision": 350,
            "execute_pipeline": 350,
            "execute_evaluation": 300,
        }
        default_limit = 500 if node_name not in NODE_SUMMARY_EXTRACTORS else self.summary_char_limit
        return max(caps.get(node_name, default_limit), self.summary_char_limit)

    def _truncate_summary(self, text: str, limit: Optional[int] = None) -> str:
        text = str(text or "").replace("\r\n", "\n").strip()
        limit = limit or self.summary_char_limit
        if len(text) <= limit:
            return text
        if "\n" in text:
            raw_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
            lines = [line for line in raw_lines if line]
            kept = []
            current_len = 0
            for line in lines:
                extra = len(line) + (1 if kept else 0)
                if current_len + extra > limit:
                    break
                kept.append(line)
                current_len += extra
            if kept:
                return "\n".join(kept)
            first = re.sub(r"\s+", " ", lines[0] if lines else text).strip()
            if len(first) <= limit:
                return first
            clipped = first[:limit].rstrip()
            word_boundary = clipped.rfind(" ")
            if word_boundary >= 40:
                clipped = clipped[:word_boundary].rstrip()
            if clipped and clipped[-1] not in ".!?":
                clipped += "."
            return clipped
        text = re.sub(r"\s+", " ", text).strip()
        separators = ["; ", ". ", " | ", ", "]
        working = text
        for sep in separators:
            parts = working.split(sep)
            if len(parts) <= 1:
                continue
            kept = []
            current_len = 0
            for part in parts:
                candidate = part.strip()
                if not candidate:
                    continue
                extra = len(candidate) + (len(sep) if kept else 0)
                if current_len + extra > limit:
                    break
                kept.append(candidate)
                current_len += extra
            if kept:
                joined = sep.join(kept).strip()
                if joined and joined[-1] not in ".!?":
                    joined += "."
                return joined

        clipped = text[:limit].rstrip()
        word_boundary = clipped.rfind(" ")
        if word_boundary >= 40:
            clipped = clipped[:word_boundary].rstrip()
        clipped = clipped[: max(1, limit - 1)].rstrip()
        if clipped and clipped[-1] not in ".!?":
            clipped += "."
        return clipped

    def _compose_multiline_summary(self, lines: List[str], node_name: str) -> str:
        cleaned = []
        for line in lines:
            normalized = re.sub(r"[ \t]+", " ", str(line or "")).strip()
            if normalized:
                cleaned.append(normalized)
        if not cleaned:
            return ""
        return self._truncate_summary("\n".join(cleaned), self._summary_limit_for_node(node_name))

    @staticmethod
    def _summary_output_value(summary: Any) -> List[str]:
        if isinstance(summary, list):
            return [re.sub(r"[ \t]+", " ", str(line or "")).strip() for line in summary if str(line or "").strip()]
        text = str(summary or "").replace("\r\n", "\n").strip()
        if not text:
            return []
        return [line for line in (re.sub(r"[ \t]+", " ", raw).strip() for raw in text.split("\n")) if line]

    @staticmethod
    def _ensure_sentence(text: Any) -> str:
        sentence = re.sub(r"\s+", " ", str(text or "")).strip()
        if not sentence:
            return ""
        if sentence[-1] not in ".!?":
            sentence += "."
        return sentence

    def _normalize_sentence_list(self, values: Any) -> List[str]:
        if values is None:
            return []
        if isinstance(values, str):
            candidates = [values]
        elif isinstance(values, list):
            candidates = values
        else:
            candidates = [values]
        out = []
        seen = set()
        for value in candidates:
            sentence = self._ensure_sentence(value)
            if not sentence:
                continue
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(sentence)
        return out

    @staticmethod
    def _limit_context(text: Any, max_chars: int = 12000) -> str:
        content = str(text or "").strip()
        if len(content) <= max_chars:
            return content
        return content[:max_chars].rstrip() + "\n...[truncated]"

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        content = str(text or "").strip()
        if not content:
            return None
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        try:
            payload = json.loads(content)
            return payload if isinstance(payload, dict) else None
        except Exception:
            pass
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(content[start : end + 1])
                return payload if isinstance(payload, dict) else None
            except Exception:
                return None
        return None

    def _validate_structured_extractor_payload(
        self,
        payload: Any,
        expected_keys: List[str],
    ) -> Optional[Dict[str, List[str]]]:
        if not isinstance(payload, dict):
            return None
        validated: Dict[str, List[str]] = {}
        for key in expected_keys:
            validated[key] = self._normalize_sentence_list(payload.get(key, []))
        if not any(validated.values()):
            return None
        return validated

    def _run_structured_summary_extractor(
        self,
        node_name: str,
        evidence: Dict[str, Any],
        schema_name: str,
    ) -> Optional[Dict[str, List[str]]]:
        if self._summary_model is None:
            return None

        schemas = {
            "evaluation_reasoning": {
                "main_problem": ["What is going wrong in this run."],
                "evidence": ["Concrete evidence such as metrics, diagnostics, or debug signals."],
                "proposal": ["What the node proposes changing next."],
                "target": ["Whether the proposal targets pipeline logic, evaluation logic, or both."],
                "why_now": ["Why that proposal fits the current code and state."],
            },
            "pipeline_adaption": {
                "action": ["Whether this was an initial draft, repair, or refinement."],
                "main_changes": ["The most important concrete pipeline changes in this run."],
                "strategy_shift": ["How the fusion or matching behavior shifted."],
                "supporting_changes": ["Important supporting details such as singleton, lineage, helper, or threshold handling."],
                "rationale": ["Why these changes make sense for the current problem."],
            },
            "evaluation_adaption": {
                "action": ["Whether this was an initial draft, repair, or refinement."],
                "main_strategy": ["The dominant evaluation strategy used in this run."],
                "field_groups": ["Grouped evaluation-function behavior across field families."],
                "supporting_changes": ["Helper, fallback, tolerance, or execution-stability changes."],
                "rationale": ["Why these evaluation changes fit the current run."],
            },
            "normalization_node": {
                "action": ["What normalization outcome occurred in this run."],
                "main_changes": ["Key dataset-level normalization changes and transformed column behavior."],
                "transform_pattern": ["Dominant transform pattern and where it was applied."],
                "exceptions": ["Columns or datasets that deviated from the dominant pattern."],
                "quality_controls": ["Warnings, gate/ablation/acceptance outcomes, and safeguards triggered."],
                "impact": ["Direct effect on next workflow step and data readiness."],
            },
            "investigator_node": {
                "decision": ["What routing decision was made in this run and at which attempt."],
                "performance_context": ["Current accuracy context and key metric signals used for the decision."],
                "diagnostic_findings": ["Most important diagnostics or probe findings from this cycle."],
                "proposed_actions": ["Concrete high-priority fixes or action-plan items suggested in this cycle."],
                "routing_rationale": ["Why this route was chosen now, including normalization/routing pressure when relevant."],
            },
        }
        expected = schemas.get(schema_name)
        if not expected:
            return None

        system_prompt = (
            "You are extracting structured engineering summary facts for a workflow log. "
            "Analyze only the current run. Preserve material facts. Do not explain the generic purpose of the node. "
            "Do not invent evidence. Do not output markdown. Do not output prose outside JSON. "
            "Return valid JSON only. Every value must be a JSON array of complete sentences."
        )
        if schema_name == "normalization_node":
            system_prompt += (
                " For normalization_node, explicitly describe dominant transform behavior and exception columns when available. "
                "Mention gate/ablation/warnings only if present in evidence."
            )
        if schema_name == "investigator_node":
            system_prompt += (
                " For investigator_node, focus on concrete run-specific evidence driving the route decision. "
                "Do not describe generic investigator behavior."
            )
        human_prompt = (
            f"NODE\n{node_name}\n\n"
            f"SCHEMA\n{json.dumps(expected, ensure_ascii=False, indent=2)}\n\n"
            f"EVIDENCE\n{json.dumps(evidence, ensure_ascii=False, indent=2, default=str)}\n"
        )
        try:
            response = self._invoke_summary_model(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)],
                purpose=f"structured_extractor:{schema_name}:{node_name}",
            )
            text = self._coerce_response_text(response.content if hasattr(response, "content") else response)
            payload = self._extract_json_object(text)
            return self._validate_structured_extractor_payload(payload, list(expected.keys()))
        except Exception:
            return None

    def _render_reasoning_summary_lines(
        self,
        payload: Optional[Dict[str, List[str]]],
        fallback_lines: List[str],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        lines: List[str] = []
        if payload:
            for key in ["main_problem", "evidence", "proposal", "target", "why_now"]:
                lines.extend(payload.get(key, []))
        if not lines:
            lines.extend(self._normalize_sentence_list(fallback_lines))
        accuracy = self._format_accuracy(metrics or {})
        if accuracy and not any("accuracy" in line.lower() for line in lines):
            lines.append(f"This reasoning is responding to an overall accuracy of {accuracy}.")
        return self._normalize_sentence_list(lines)

    def _render_pipeline_adaption_summary_lines(
        self,
        payload: Optional[Dict[str, List[str]]],
        fallback_lines: List[str],
    ) -> List[str]:
        lines: List[str] = []
        if payload:
            for key in ["action", "main_changes", "strategy_shift", "supporting_changes", "rationale"]:
                lines.extend(payload.get(key, []))
        if not lines:
            lines.extend(self._normalize_sentence_list(fallback_lines))
        return self._normalize_sentence_list(lines)

    def _render_evaluation_adaption_summary_lines(
        self,
        payload: Optional[Dict[str, List[str]]],
        fallback_lines: List[str],
    ) -> List[str]:
        lines: List[str] = []
        if payload:
            for key in ["action", "main_strategy", "field_groups", "supporting_changes", "rationale"]:
                lines.extend(payload.get(key, []))
        if not lines:
            lines.extend(self._normalize_sentence_list(fallback_lines))
        return self._normalize_sentence_list(lines)

    def _render_normalization_summary_lines(
        self,
        payload: Optional[Dict[str, List[str]]],
        fallback_lines: List[str],
    ) -> List[str]:
        lines: List[str] = []
        if payload:
            for key in ["action", "main_changes", "transform_pattern", "exceptions", "quality_controls", "impact"]:
                lines.extend(payload.get(key, []))
        if not lines:
            lines.extend(self._normalize_sentence_list(fallback_lines))
        return self._normalize_sentence_list(lines)

    def _render_investigator_summary_lines(
        self,
        payload: Optional[Dict[str, List[str]]],
        fallback_lines: List[str],
    ) -> List[str]:
        lines: List[str] = []
        if payload:
            for key in [
                "decision",
                "performance_context",
                "diagnostic_findings",
                "proposed_actions",
                "routing_rationale",
            ]:
                lines.extend(payload.get(key, []))
        if not lines:
            lines.extend(self._normalize_sentence_list(fallback_lines))
        return self._normalize_sentence_list(lines)

    @staticmethod
    def _read_file_if_exists(path: str) -> str:
        if not os.path.exists(path):
            return ""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

    @staticmethod
    def _extract_all_changed_lines(before_text: str, after_text: str) -> List[str]:
        if before_text == after_text:
            return []
        before_lines = (before_text or "").splitlines()
        after_lines = (after_text or "").splitlines()
        diff_lines = list(
            difflib.unified_diff(before_lines, after_lines, fromfile="before", tofile="after", lineterm="")
        )
        return [
            line for line in diff_lines
            if line and (line.startswith("+") or line.startswith("-")) and not line.startswith("+++") and not line.startswith("---")
        ]

    @staticmethod
    def _safe_json_excerpt(value: Any, max_chars: int = 1600) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
        return text[:max_chars]

    @staticmethod
    def _dataset_names_from_state(state_in: Any) -> List[str]:
        if not isinstance(state_in, dict):
            return []
        out = []
        for path in state_in.get("datasets", []) or []:
            name = os.path.splitext(os.path.basename(str(path)))[0]
            if name:
                out.append(name)
        return out

    @staticmethod
    def _maybe_parse_json_text(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text or text[0] not in "{[":
            return value
        try:
            return json.loads(text)
        except Exception:
            return value

    @staticmethod
    def _compact_list(values: List[str], max_items: int = 6) -> str:
        cleaned = [str(v) for v in values if str(v)]
        if not cleaned:
            return ""
        if len(cleaned) <= max_items:
            return ", ".join(cleaned)
        return ", ".join(cleaned[:max_items]) + f", +{len(cleaned) - max_items} more"

    @staticmethod
    def _format_metric(value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.3f}"
        return str(value)

    @staticmethod
    def _diff_mapping(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        added = {k: after[k] for k in after.keys() - before.keys()}
        removed = {k: before[k] for k in before.keys() - after.keys()}
        changed = {
            k: {"before": before[k], "after": after[k]}
            for k in before.keys() & after.keys()
            if before[k] != after[k]
        }
        return {"added": added, "removed": removed, "changed": changed}

    @staticmethod
    def _extract_import_set(code: str) -> set[str]:
        imports = set()
        for line in (code or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.add(stripped)
        return imports

    @staticmethod
    def _extract_threshold_map(code: str) -> Dict[str, str]:
        return {
            match.group(1): match.group(2)
            for match in re.finditer(r"^\s*(threshold_[A-Za-z0-9_]+)\s*=\s*([0-9]*\.?[0-9]+)", code or "", flags=re.M)
        }

    @staticmethod
    def _extract_include_singletons_value(code: str) -> Optional[str]:
        match = re.search(r"include_singletons\s*=\s*(True|False)", code or "")
        return match.group(1) if match else None

    @staticmethod
    def _extract_symbol_tokens(code: str) -> set[str]:
        patterns = [
            r"\b[A-Z][A-Za-z0-9_]*(?:Blocker|Matcher|Comparator|Engine|Strategy)\b",
            r"\bload_(?:csv|xml|parquet)\b",
        ]
        tokens: set[str] = set()
        for pattern in patterns:
            tokens.update(re.findall(pattern, code or ""))
        return tokens

    @staticmethod
    def _extract_alignment_markers(code: str) -> List[str]:
        markers = []
        for marker in ["_id", "source_id", "source_dataset", "_fusion_sources"]:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(marker)}(?![A-Za-z0-9])", code or ""):
                markers.append(marker)
        return markers

    def _capture_tracked_files(self, node_name: str) -> Dict[str, str]:
        tracked = {}
        tracked_paths = {
            "pipeline.py": os.path.join(self.output_dir, "code", "pipeline.py"),
            "evaluation.py": os.path.join(self.output_dir, "code", "evaluation.py"),
        }
        cached = self._tracked_file_cache.get(node_name, {})
        for label, path in tracked_paths.items():
            text = self._read_file_if_exists(path)
            if text or label in cached:
                tracked[label] = text
        return tracked

    def _build_file_bundle(self, node_name: str, before_map: Dict[str, str], after_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        bundle: Dict[str, Dict[str, Any]] = {}
        keys = set(before_map.keys()) | set(after_map.keys())
        for key in keys:
            before_text = before_map.get(key, "")
            after_text = after_map.get(key, "")
            if before_text == after_text:
                continue
            bundle[key] = {
                "path": key,
                "before_text": before_text,
                "after_text": after_text,
                "all_changed_lines": self._extract_all_changed_lines(before_text, after_text),
                "current_file_context": after_text,
            }
        self._tracked_file_cache[node_name] = dict(after_map)
        return bundle

    def _compute_file_changes(self, node_name: str, after_map: Dict[str, str]) -> Dict[str, str]:
        before_map = self._tracked_file_cache.get(node_name, {})
        bundle = self._build_file_bundle(node_name, before_map, after_map)
        return {
            key: "\n".join(info.get("all_changed_lines", []))
            for key, info in bundle.items()
        }

    def _build_summary_context(
        self,
        node_name: str,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: Optional[str] = None,
    ) -> Dict[str, Any]:
        facts = self._build_node_facts(node_name, state_in, node_output, file_bundle, status, error_text, next_node or "PENDING")
        return {"node_facts": facts}

    def _get_node_extractor(self, node_name: str) -> Optional[Callable[..., Dict[str, Any]]]:
        extractor = NODE_SUMMARY_EXTRACTORS.get(node_name)
        if isinstance(extractor, str):
            return getattr(self, extractor, None)
        return extractor

    def _build_generic_fallback_facts(
        self,
        node_name: str,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = state_in if isinstance(state_in, dict) else {}
        changed_keys = sorted(node_output.keys()) if isinstance(node_output, dict) else []
        metrics = node_output.get("evaluation_metrics") if isinstance(node_output, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}
        upstream_error = None
        prior_exec = state.get("evaluation_execution_result")
        if isinstance(prior_exec, str) and prior_exec.lower().startswith("error"):
            upstream_error = prior_exec
        artifact_updates = []
        for key in changed_keys:
            if key.endswith("_config") or key.endswith("_metrics") or key.endswith("_report") or key.endswith("_code"):
                artifact_updates.append(key)
        return {
            "registered": False,
            "node_name": node_name,
            "next_node": next_node,
            "status": status,
            "error": (error_text or "")[:1000],
            "changed_keys": changed_keys,
            "attempt_counters": {
                "pipeline_execution_attempts": state.get("pipeline_execution_attempts"),
                "evaluation_execution_attempts": state.get("evaluation_execution_attempts"),
                "evaluation_attempts": state.get("evaluation_attempts"),
            },
            "metrics": metrics,
            "artifact_updates": artifact_updates,
            "node_output_excerpt": self._safe_json_excerpt(node_output, 2400),
            "upstream_error_excerpt": (upstream_error or "")[:1000],
            "file_bundle": file_bundle,
            "summary_clauses": [],
        }

    def _build_node_facts(
        self,
        node_name: str,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        extractor = self._get_node_extractor(node_name)
        if extractor is None:
            return self._build_generic_fallback_facts(node_name, state_in, node_output, file_bundle, status, error_text, next_node)
        try:
            facts = extractor(state_in, node_output, file_bundle, status, error_text, next_node)
            if not isinstance(facts, dict):
                raise ValueError("Extractor returned non-dict facts")
            facts.setdefault("registered", True)
            facts.setdefault("node_name", node_name)
            facts.setdefault("next_node", next_node)
            facts.setdefault("status", status)
            facts.setdefault("error", error_text or "")
            facts.setdefault("summary_clauses", [])
            facts.setdefault("file_bundle", file_bundle)
            return facts
        except Exception as exc:
            fallback = self._build_generic_fallback_facts(node_name, state_in, node_output, file_bundle, status, error_text, next_node)
            fallback["artifact_updates"].append(f"extractor_error:{type(exc).__name__}")
            return fallback

    def _compose_summary_from_facts(self, node_name: str, facts: Dict[str, Any], next_node: str) -> str:
        prioritized_lines = []
        for key in ["summary_lines_primary", "summary_lines_secondary", "summary_lines_support"]:
            prioritized_lines.extend(
                [str(line).strip() for line in facts.get(key, []) if str(line).strip()]
            )
        if prioritized_lines:
            return self._compose_multiline_summary(prioritized_lines, node_name)
        summary_lines = [str(line).strip() for line in facts.get("summary_lines", []) if str(line).strip()]
        if summary_lines:
            return self._compose_multiline_summary(summary_lines, node_name)
        clauses = [str(clause).strip() for clause in facts.get("summary_clauses", []) if str(clause).strip()]
        effect_clause = str(facts.get("effect_clause", "")).strip()
        if effect_clause:
            clauses.append(effect_clause)
        text = "; ".join(clauses).strip()
        if text and text[-1] not in ".!?":
            text += "."
        return self._truncate_summary(text, self._summary_limit_for_node(node_name)) if text else ""

    def _summarize_unknown_node_with_llm(self, facts: Dict[str, Any], node_name: str, next_node: str) -> str:
        limit = self._summary_limit_for_node(node_name)
        fallback = f"{node_name} changed this run; next step is {next_node}."
        if self._summary_model is None:
            return self._truncate_summary(fallback, limit)
        file_bundle = facts.get("file_bundle", {}) if isinstance(facts.get("file_bundle"), dict) else {}
        changed_lines = {}
        current_context = {}
        for path, info in file_bundle.items():
            changed_lines[path] = info.get("all_changed_lines", [])
            current_context[path] = info.get("current_file_context", "")
        system_prompt = (
            "You are writing an engineering run log summary for a single workflow node. "
            "Summarize only what changed in this run. Preserve every material fact from the evidence. "
            "Name exact code, config, metric, artifact, or error changes when present. "
            "Use dense semicolon-separated clauses. Mention workflow effect only after naming actual changes. "
            "Do not explain the usual purpose of the node. Do not say generic phrases like "
            "'updated the workflow', 'improved performance', or 'advanced the pipeline' unless followed by specific evidence. "
            "Do not invent changes. Do not omit material changed lines when they are clearly relevant. "
            f"Return a single plain-text string under {limit} characters."
        )
        human_prompt = (
            f"NODE\n{node_name}\n\n"
            f"NEXT_NODE\n{next_node}\n\n"
            f"STATUS\n{facts.get('status', '')}\n\n"
            f"ERROR\n{facts.get('error', '')}\n\n"
            f"CHANGED_KEYS\n{json.dumps(facts.get('changed_keys', []), ensure_ascii=False)}\n\n"
            f"ATTEMPTS\n{json.dumps(facts.get('attempt_counters', {}), ensure_ascii=False)}\n\n"
            f"METRICS\n{json.dumps(facts.get('metrics', {}), ensure_ascii=False)}\n\n"
            f"ARTIFACT_UPDATES\n{json.dumps(facts.get('artifact_updates', []), ensure_ascii=False)}\n\n"
            f"CHANGED_LINES\n{json.dumps(changed_lines, ensure_ascii=False)}\n\n"
            f"CURRENT_FILE_CONTEXT\n{json.dumps(current_context, ensure_ascii=False)}\n\n"
            f"NODE_OUTPUT_EXCERPT\n{facts.get('node_output_excerpt', '')}\n\n"
            f"UPSTREAM_ERROR_EXCERPT\n{facts.get('upstream_error_excerpt', '')}\n"
        )
        try:
            response = self._invoke_summary_model(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)],
                purpose=f"unknown_fallback:{node_name}",
            )
            text = self._coerce_response_text(response.content if hasattr(response, "content") else response)
            if not text.strip():
                text = fallback
            return self._truncate_summary(text, limit)
        except Exception:
            return self._truncate_summary(fallback, limit)

    def _summarize_step(self, current_node: str, next_node: str, facts: Dict[str, Any]) -> str:
        draft = self._compose_summary_from_facts(current_node, facts, next_node)
        if facts.get("registered") and draft:
            return draft
        if not facts.get("registered"):
            return self._summarize_unknown_node_with_llm(facts, current_node, next_node)
        if draft:
            return draft
        return self._summarize_unknown_node_with_llm(facts, current_node, next_node)

    @staticmethod
    def _normalize_expr_text(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @staticmethod
    def _extract_strategy_calls(code: str, method_name: str) -> Dict[str, str]:
        if not code:
            return {}
        try:
            tree = ast.parse(code)
            calls = {}
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Attribute) or node.func.attr != method_name:
                    continue
                if len(node.args) < 2:
                    continue
                key_arg = node.args[0]
                fn_arg = node.args[1]
                if isinstance(key_arg, ast.Constant) and isinstance(key_arg.value, str):
                    fn_text = ast.get_source_segment(code, fn_arg)
                    if not fn_text and isinstance(fn_arg, ast.Name):
                        fn_text = fn_arg.id
                    calls[key_arg.value] = WorkflowLogger._normalize_expr_text(fn_text or type(fn_arg).__name__)
            return calls
        except Exception:
            return {}

    @staticmethod
    def _extract_field_assignments(code: str, field_name: str) -> List[str]:
        hits = []
        pattern = re.compile(rf"^\s*.*{re.escape(field_name)}.*$", re.M)
        for match in pattern.finditer(code or ""):
            line = match.group(0).strip()
            if line:
                hits.append(line)
        return hits

    @staticmethod
    def _infer_config_source(state: Dict[str, Any], config_key: str, path_key: str) -> str:
        if state.get(path_key):
            return "loaded from file"
        if state.get(config_key):
            return "reused from state"
        return "fresh run"

    @staticmethod
    def _collect_numeric_metrics(payload: Any, path: str = "") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                child_path = f"{path}.{key}" if path else str(key)
                out.update(WorkflowLogger._collect_numeric_metrics(value, child_path))
        elif isinstance(payload, list):
            for idx, value in enumerate(payload):
                child_path = f"{path}[{idx}]"
                out.update(WorkflowLogger._collect_numeric_metrics(value, child_path))
        elif isinstance(payload, (int, float)):
            out[path] = payload
        return out

    @staticmethod
    def _collect_pair_scores(config: Any) -> List[str]:
        scores = []
        if not isinstance(config, dict):
            return scores
        for pair_key, pair_value in config.items():
            if not isinstance(pair_value, dict):
                continue
            snippets = []
            matcher_type = pair_value.get("matcher_type") or pair_value.get("type")
            if matcher_type:
                snippets.append(str(matcher_type))
            for metric_name in ["f1", "f1_score", "precision", "recall", "threshold"]:
                if metric_name in pair_value and isinstance(pair_value.get(metric_name), (int, float, str)):
                    snippets.append(f"{metric_name}={pair_value.get(metric_name)}")
            nested_metrics = WorkflowLogger._collect_numeric_metrics(pair_value)
            for metric_path, metric_value in nested_metrics.items():
                leaf = metric_path.split(".")[-1]
                if leaf in {"f1", "f1_score", "precision", "recall", "threshold"}:
                    rendered = f"{leaf}={metric_value}"
                    if rendered not in snippets:
                        snippets.append(rendered)
            if snippets:
                scores.append(f"{pair_key} ({', '.join(snippets[:5])})")
        return scores

    @staticmethod
    def _safe_dict(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _extract_error_excerpt(value: Any) -> str:
        text = WorkflowLogger._sanitize_error_text(str(value or "").strip())
        return text[:240]

    @staticmethod
    def _sanitize_error_text(text: str) -> str:
        if not text:
            return ""
        sanitized = str(text)
        # Collapse local absolute paths so summaries don't leak user directories.
        sanitized = re.sub(r'"/Users/[^"]+"', lambda m: f"\"{os.path.basename(m.group(0).strip('\"'))}\"", sanitized)
        sanitized = re.sub(r'"/home/[^"]+"', lambda m: f"\"{os.path.basename(m.group(0).strip('\"'))}\"", sanitized)
        sanitized = re.sub(r'"[A-Za-z]:\\[^"]+"', lambda m: f"\"{os.path.basename(m.group(0).strip('\"'))}\"", sanitized)
        # Keep line breaks out of fallback clauses.
        sanitized = sanitized.replace("\r\n", "\n").strip()
        return sanitized

    @staticmethod
    def _parse_error_details(raw_error: str) -> Dict[str, Any]:
        text = WorkflowLogger._sanitize_error_text(raw_error or "")
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        details: Dict[str, Any] = {
            "error_class": None,
            "error_message": None,
            "file_path": None,
            "file_basename": None,
            "line_no": None,
            "source_line": None,
        }
        if not lines:
            return details

        file_match = re.search(r'File "([^"]+)", line (\d+)', text)
        if file_match:
            file_path = file_match.group(1)
            details["file_path"] = file_path
            details["file_basename"] = os.path.basename(file_path)
            try:
                details["line_no"] = int(file_match.group(2))
            except Exception:
                details["line_no"] = None

        class_message_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):\s*(.+)", text)
        if class_message_match:
            details["error_class"] = class_message_match.group(1).strip()
            details["error_message"] = class_message_match.group(2).strip()
        else:
            # fallback from leading "error: ..."
            lowered = lines[0].lower()
            if lowered.startswith("error:"):
                details["error_message"] = lines[0][6:].strip()
            elif lowered.startswith("error "):
                details["error_message"] = lines[0][6:].strip()

        # Syntax traces often include the failing source line followed by caret (^).
        for idx, line in enumerate(lines):
            if line.strip() == "^" and idx > 0:
                prev = lines[idx - 1].strip()
                if prev:
                    details["source_line"] = prev
                    break
        if details["source_line"] is None and details.get("line_no") and details.get("file_basename"):
            pass
        return details

    @staticmethod
    def _read_error_context_lines(file_path: Optional[str], line_no: Optional[int], window: int = 2) -> List[str]:
        if not file_path or not line_no:
            return []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            return []
        if not lines:
            return []
        start = max(1, int(line_no) - int(window))
        end = min(len(lines), int(line_no) + int(window))
        context = []
        for idx in range(start, end + 1):
            snippet = lines[idx - 1].rstrip("\n")
            context.append(f"{idx}: {snippet.strip()}")
        return context

    @staticmethod
    def _infer_error_fix_direction(error_class: Optional[str], error_message: Optional[str]) -> str:
        cls = str(error_class or "").lower()
        msg = str(error_message or "").lower()
        if cls == "syntaxerror":
            if "line continuation" in msg or "unexpected character" in msg:
                return "Fix string escaping/quoting around the failing statement and remove malformed backslashes."
            return "Fix Python syntax around the failing line and re-run generation."
        if cls in {"modulenotfounderror", "importerror"}:
            return "Verify import paths/dependencies and add safe fallback imports where needed."
        if cls == "nameerror":
            return "Define or import the missing symbol before use."
        if cls == "typeerror" and "unexpected keyword argument" in msg:
            return "Align function call arguments with the target function signature."
        if cls in {"keyerror", "indexerror", "attributeerror"}:
            return "Add guards/validation for missing fields before accessing them."
        return "Inspect the failing statement and apply a targeted code repair before the next retry."

    def _build_error_summary_lines(
        self,
        node_label: str,
        attempts: Any,
        raw_error: str,
        code_path: Optional[str],
    ) -> List[str]:
        details = self._parse_error_details(raw_error or "")
        error_class = details.get("error_class")
        error_message = details.get("error_message")
        file_basename = details.get("file_basename") or (os.path.basename(code_path) if code_path else None)
        line_no = details.get("line_no")
        source_line = details.get("source_line")
        if source_line:
            source_line = source_line.replace('\\"', '"').strip()
        context_lines = self._read_error_context_lines(code_path, line_no, window=2)

        headline = f"{node_label} failed on attempt {attempts}"
        if error_class:
            headline += f" with {error_class}"
        headline += "."
        lines = [headline]

        if error_message:
            lines.append(f"Failure reason: {error_message}.")
        if file_basename and line_no:
            lines.append(f"Failure location: {file_basename}:{line_no}.")
        if source_line:
            lines.append(f"Failing code line: `{source_line}`.")
        if context_lines:
            lines.append("Nearby code context: " + " | ".join(context_lines[:5]) + ".")
        lines.append(self._infer_error_fix_direction(error_class, error_message))
        return [self._ensure_sentence(line) for line in lines if str(line).strip()]

    def _extract_execution_error_message(
        self,
        node_name: str,
        state_in: Any,
        node_output: Any,
    ) -> Optional[str]:
        state = self._safe_dict(state_in)
        output = self._safe_dict(node_output)
        priority_keys = [
            "pipeline_execution_result",
            "evaluation_execution_result",
            "integration_diagnostics_execution_result",
            "human_review_execution_result",
            "final_test_evaluation_execution_result",
        ]
        node_specific_keys = {
            "execute_pipeline": ["pipeline_execution_result"],
            "pipeline_adaption": ["pipeline_execution_result"],
            "execute_evaluation": ["evaluation_execution_result"],
            "evaluation_node": ["evaluation_execution_result"],
            "evaluation_adaption": ["evaluation_execution_result"],
            "integration_diagnostics": ["integration_diagnostics_execution_result"],
            "human_review_export": ["human_review_execution_result"],
            "sealed_final_test_evaluation": ["final_test_evaluation_execution_result"],
        }
        ordered_keys = node_specific_keys.get(node_name, []) + [
            k for k in priority_keys if k not in node_specific_keys.get(node_name, [])
        ]
        execution_result: Optional[str] = None

        for key in ordered_keys:
            value = output.get(key)
            if isinstance(value, str) and value.lower().startswith("error"):
                execution_result = value
                break
        if execution_result is None:
            # State fallback is restricted to node-relevant known keys to avoid stale cross-node errors.
            for key in node_specific_keys.get(node_name, []):
                value = state.get(key)
                if isinstance(value, str) and value.lower().startswith("error"):
                    execution_result = value
                    break

        if execution_result is None:
            # Generic fallback for unknown nodes: only inspect node_output to avoid stale state bleed.
            for key, value in output.items():
                if not isinstance(key, str):
                    continue
                if not key.endswith("_execution_result"):
                    continue
                if isinstance(value, str) and value.lower().startswith("error"):
                    execution_result = value
                    break

        if not execution_result:
            return None

        details = self._parse_error_details(execution_result)
        cls = str(details.get("error_class") or "").strip()
        msg = str(details.get("error_message") or "").strip()
        if cls and msg:
            return f"{cls}: {msg}"
        if msg:
            return msg
        one_line = re.sub(r"\s+", " ", self._sanitize_error_text(execution_result)).strip()
        if one_line.lower().startswith("error:"):
            one_line = one_line[6:].strip()
        return one_line or None

    def _code_path_for_node(self, node_name: str) -> Optional[str]:
        mapping = {
            "execute_pipeline": os.path.join(self.output_dir, "code", "pipeline.py"),
            "pipeline_adaption": os.path.join(self.output_dir, "code", "pipeline.py"),
            "execute_evaluation": os.path.join(self.output_dir, "code", "evaluation.py"),
            "evaluation_node": os.path.join(self.output_dir, "code", "evaluation.py"),
            "evaluation_adaption": os.path.join(self.output_dir, "code", "evaluation.py"),
        }
        return mapping.get(node_name)

    @staticmethod
    def _normalize_transform_signature(value: Any) -> str:
        if isinstance(value, list):
            parts = [str(v).strip() for v in value if str(v).strip()]
            return " + ".join(parts)
        if isinstance(value, tuple):
            parts = [str(v).strip() for v in value if str(v).strip()]
            return " + ".join(parts)
        if isinstance(value, dict):
            parts = []
            for key, val in value.items():
                parts.append(f"{key}={val}")
            return ", ".join(parts)
        return str(value).strip()

    def _analyze_normalization_transforms(self, dataset_report: Dict[str, Any]) -> Dict[str, Any]:
        signature_counts: Dict[str, int] = {}
        signature_columns: Dict[str, List[str]] = {}
        all_columns: List[str] = []

        for dataset_name, entry in dataset_report.items():
            if not isinstance(entry, dict):
                continue
            transforms = entry.get("applied_transforms", {})
            if not isinstance(transforms, dict):
                continue
            for column, transform_spec in transforms.items():
                signature = self._normalize_transform_signature(transform_spec)
                if not signature:
                    continue
                col_label = f"{dataset_name}.{column}"
                all_columns.append(col_label)
                signature_counts[signature] = signature_counts.get(signature, 0) + 1
                signature_columns.setdefault(signature, []).append(col_label)

        dominant_signature = None
        dominant_count = 0
        if signature_counts:
            dominant_signature, dominant_count = sorted(
                signature_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]

        exception_columns: List[str] = []
        if dominant_signature:
            for signature, columns in signature_columns.items():
                if signature == dominant_signature:
                    continue
                exception_columns.extend(columns)

        return {
            "signature_counts": signature_counts,
            "signature_columns": signature_columns,
            "dominant_signature": dominant_signature,
            "dominant_count": dominant_count,
            "total_transformed_columns": len(all_columns),
            "exception_columns": sorted(exception_columns),
        }

    @staticmethod
    def _classify_pipeline_adaption_mode(state: Dict[str, Any], before_code: str) -> str:
        prior_error = state.get("pipeline_execution_result")
        if isinstance(prior_error, str) and prior_error.lower().startswith("error"):
            return "execution_repair"
        if before_code.strip() and (state.get("evaluation_analysis") or state.get("evaluation_metrics")):
            return "evaluation_refinement"
        return "initial_draft"

    @staticmethod
    def _classify_evaluation_adaption_mode(state: Dict[str, Any], before_code: str) -> str:
        prior_error = state.get("evaluation_execution_result")
        if isinstance(prior_error, str) and prior_error.lower().startswith("error"):
            return "execution_repair"
        if before_code.strip() and state.get("evaluation_metrics"):
            return "evaluation_refinement"
        return "initial_draft"

    @staticmethod
    def _should_repeat_upstream_config_in_pipeline_summary(state: Dict[str, Any], before_code: str, after_code: str) -> bool:
        if not before_code.strip():
            return False
        before_tokens = WorkflowLogger._extract_symbol_tokens(before_code)
        after_tokens = WorkflowLogger._extract_symbol_tokens(after_code)
        return before_tokens != after_tokens

    @staticmethod
    def _summarize_import_capabilities(import_lines: set[str], context: str) -> List[str]:
        capabilities = []
        joined = "\n".join(sorted(import_lines))
        mapping = {
            "EmbeddingBlocker": "embedding-based blocking support",
            "RuleBasedMatcher": "rule-based matching support",
            "StringComparator": "string comparison support",
            "NumericComparator": "numeric comparison support",
            "DateComparator": "date comparison support",
            "DataFusionEngine": "fusion engine support",
            "DataFusionEvaluator": "fusion evaluation support",
            "load_xml": "XML loading support",
            "load_csv": "CSV loading support",
            "load_dotenv": "environment loading support",
        }
        for token, label in mapping.items():
            if token in joined:
                capabilities.append(label)
        if context == "evaluation" and "ImportError" in joined:
            capabilities.append("fallback import handling")
        return capabilities[:3]

    def _summarize_attribute_fuser_usage(self, pipeline_code: str) -> Dict[str, Any]:
        fusers = self._extract_attribute_fusers(pipeline_code)
        counts: Dict[str, int] = {}
        fields_by_fuser: Dict[str, List[str]] = {}
        for field, fn in fusers.items():
            counts[fn] = counts.get(fn, 0) + 1
            fields_by_fuser.setdefault(fn, []).append(field)
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        dominant = ordered[0][0] if ordered else None
        secondaries = [(fn, fields_by_fuser.get(fn, [])) for fn, _ in ordered[1:4]]
        return {
            "fusers": fusers,
            "counts": counts,
            "dominant": dominant,
            "secondaries": secondaries,
            "total_fields": len(fusers),
        }

    def _summarize_pipeline_behavior(self, pipeline_code: str) -> Dict[str, Any]:
        return {
            "include_singletons": self._extract_include_singletons_value(pipeline_code),
            "alignment_fields": self._extract_alignment_markers(pipeline_code),
            "capabilities": self._summarize_import_capabilities(self._extract_import_set(pipeline_code), "pipeline"),
        }

    def _summarize_evaluation_function_usage(self, evaluation_code: str) -> Dict[str, Any]:
        functions = self._extract_evaluation_functions(evaluation_code)
        fields_by_fn: Dict[str, List[str]] = {}
        for field, fn in functions.items():
            fields_by_fn.setdefault(fn, []).append(field)
        ordered = sorted(fields_by_fn.items(), key=lambda item: (-len(item[1]), item[0]))
        return {
            "functions": functions,
            "ordered": ordered,
            "capabilities": self._summarize_import_capabilities(self._extract_import_set(evaluation_code), "evaluation"),
        }

    @staticmethod
    def _strip_markdown_noise(text: str) -> str:
        cleaned = str(text or "")
        cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", cleaned)
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = re.sub(r"(?m)^\s*[-*]\s+", "", cleaned)
        cleaned = re.sub(r"(?m)^\s*\d+\)\s*", "", cleaned)
        cleaned = re.sub(r"\s+-\s+", ". ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _summarize_reasoning_text(self, text: str, metrics: Dict[str, Any]) -> List[str]:
        cleaned = self._strip_markdown_noise(text)
        if not cleaned:
            return []
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        problem_keywords = ("problem", "issue", "failure", "dominant error", "missing", "coverage", "accuracy")
        evidence_keywords = ("mismatch", "debug", "diagnostic", "ratio", "count", "inputs", "fused_id", "nan", "%")
        proposal_keywords = ("should", "need to", "must", "recommend", "propose", "focus on", "adjust", "preserve", "increase", "switch")
        problem_lines = []
        evidence_lines = []
        proposal_lines = []
        other_lines = []
        for sentence in sentences:
            lowered = sentence.lower()
            if any(token in lowered for token in proposal_keywords):
                proposal_lines.append(self._ensure_sentence(sentence))
            elif any(token in lowered for token in evidence_keywords) or re.search(r"\b\d+(\.\d+)?%?\b", sentence):
                evidence_lines.append(self._ensure_sentence(sentence))
            elif any(token in lowered for token in problem_keywords):
                problem_lines.append(self._ensure_sentence(sentence))
            else:
                other_lines.append(self._ensure_sentence(sentence))
        lines = problem_lines + evidence_lines + proposal_lines + other_lines
        accuracy = self._format_accuracy(metrics)
        if accuracy and not any("accuracy" in line.lower() for line in lines):
            lines.append(f"This reasoning is responding to an overall accuracy of {accuracy}.")
        return self._normalize_sentence_list(lines)

    def _extract_pipeline_adaption_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        info = file_bundle.get("pipeline.py", {})
        before_code = info.get("before_text", "")
        after_code = info.get("after_text", "") or self._read_file_if_exists(os.path.join(self.output_dir, "code", "pipeline.py"))
        threshold_diff = self._diff_mapping(self._extract_threshold_map(before_code), self._extract_threshold_map(after_code))
        fuser_diff = self._diff_mapping(self._extract_attribute_fusers(before_code), self._extract_attribute_fusers(after_code))
        mode = self._classify_pipeline_adaption_mode(state, before_code)
        behavior = self._summarize_pipeline_behavior(after_code)
        usage = self._summarize_attribute_fuser_usage(after_code)
        fallback_lines = []

        if mode == "execution_repair":
            prior_error = state.get("pipeline_execution_result")
            error_excerpt = self._extract_error_excerpt(prior_error)
            fallback_lines.append(f"Reworked the pipeline after the previous execution failure: {error_excerpt}.")
        elif mode == "evaluation_refinement":
            fallback_lines.append("Adjusted the pipeline after the last evaluation exposed weak fusion behavior.")
        else:
            fallback_lines.append("Generated a new pipeline draft using upstream blocking and matching outputs.")

        dominant = usage.get("dominant")
        if dominant:
            dominant_count = usage.get("counts", {}).get(dominant, 0)
            secondaries = usage.get("secondaries", [])
            strategy_line = f"Fusion mainly uses `{dominant}` across {dominant_count} attributes."
            if secondaries:
                details = []
                for fn, fields in secondaries[:3]:
                    if fields:
                        details.append(f"`{fn}` for {self._compact_list(fields, max_items=4)}")
                if details:
                    strategy_line += " It also uses " + "; ".join(details) + "."
            fallback_lines.append(strategy_line)

        if mode == "evaluation_refinement" and fuser_diff["changed"]:
            changed_fields = list(fuser_diff["changed"].items())[:6]
            if changed_fields:
                transitions = {}
                for field, change in changed_fields:
                    key = f"{change['before']} -> {change['after']}"
                    transitions.setdefault(key, []).append(field)
                change_bits = []
                for key, fields in transitions.items():
                    change_bits.append(f"{self._compact_list(fields, max_items=4)} moved from `{key.split(' -> ')[0]}` to `{key.split(' -> ')[1]}`")
                fallback_lines.append("This iteration shifts fusion behavior so that " + "; ".join(change_bits) + ".")

        support_bits = []
        include_singletons = behavior.get("include_singletons")
        if include_singletons == "True":
            support_bits.append("singleton retention is enabled")
        elif include_singletons == "False":
            support_bits.append("singleton retention is disabled")
        alignment_fields = behavior.get("alignment_fields") or []
        if alignment_fields:
            support_bits.append(f"lineage fields include {self._compact_list(alignment_fields, max_items=4)}")
        if support_bits:
            fallback_lines.append(self._compact_list([bit[0].upper() + bit[1:] if bit else bit for bit in support_bits], max_items=3) + ".")

        if threshold_diff["changed"] and mode != "initial_draft":
            threshold_keys = sorted(threshold_diff["changed"].keys())
            fallback_lines.append(
                "Matching thresholds were adjusted for "
                + self._compact_list(threshold_keys, max_items=4)
                + "."
            )

        capabilities = behavior.get("capabilities") or []
        if capabilities and self._should_repeat_upstream_config_in_pipeline_summary(state, before_code, after_code):
            fallback_lines.append("The revision also introduces " + self._compact_list(capabilities, max_items=3) + ".")

        evidence = {
            "node": "pipeline_adaption",
            "mode": mode,
            "evaluation_analysis": self._limit_context(state.get("evaluation_analysis", "")),
            "evaluation_metrics": state.get("evaluation_metrics", {}),
            "auto_diagnostics": state.get("auto_diagnostics", {}),
            "before_pipeline": self._limit_context(before_code),
            "after_pipeline": self._limit_context(after_code),
            "fuser_changes": fuser_diff,
            "threshold_changes": threshold_diff,
            "behavior": behavior,
            "attribute_fuser_usage": usage,
            "fallback_summary_lines": fallback_lines,
        }
        payload = self._run_structured_summary_extractor("pipeline_adaption", evidence, "pipeline_adaption")
        summary_lines = self._render_pipeline_adaption_summary_lines(payload, fallback_lines)

        return {
            "registered": True,
            "node_name": "pipeline_adaption",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_lines": summary_lines,
            "summary_clauses": [],
            "file_bundle": file_bundle,
        }

    def _extract_evaluation_adaption_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        info = file_bundle.get("evaluation.py", {})
        before_code = info.get("before_text", "")
        after_code = info.get("after_text", "") or self._read_file_if_exists(os.path.join(self.output_dir, "code", "evaluation.py"))
        mode = self._classify_evaluation_adaption_mode(state, before_code)
        usage = self._summarize_evaluation_function_usage(after_code)
        ordered = usage.get("ordered", [])
        fallback_lines = []

        if mode == "execution_repair":
            prior_error = state.get("evaluation_execution_result")
            fallback_lines.append(
                f"Reworked the evaluation script after the previous execution failure: {self._extract_error_excerpt(prior_error)}."
            )
        elif mode == "evaluation_refinement":
            fallback_lines.append("Refined the evaluation logic after reviewing the latest metrics.")
        else:
            fallback_lines.append("Generated a new evaluation script for the current fused output.")

        if ordered:
            dominant_fn, dominant_fields = ordered[0]
            line = f"Most covered fields use `{dominant_fn}`"
            if dominant_fields:
                line += f", including {self._compact_list(dominant_fields, max_items=6)}"
            line += "."
            secondary_bits = []
            for fn, fields in ordered[1:4]:
                secondary_bits.append(f"`{fn}` for {self._compact_list(fields, max_items=4)}")
            if secondary_bits:
                line += " Secondary checks use " + "; ".join(secondary_bits) + "."
            fallback_lines.append(line)

        tolerance_lines = [
            line[1:].strip()
            for line in info.get("all_changed_lines", [])
            if "tolerance" in line or "numeric_tolerance_match" in line or "tokenized_match" in line
        ]
        helper_bits = []
        if tolerance_lines:
            helper_bits.append("numeric or tokenized helper logic was updated")
        if "try:" in after_code and "except ImportError" in after_code:
            helper_bits.append("fallback import handling was kept for execution stability")
        capabilities = usage.get("capabilities") or []
        if capabilities:
            helper_bits.append(self._compact_list(capabilities, max_items=2))
        if helper_bits:
            fallback_lines.append(self._compact_list(helper_bits, max_items=3).capitalize() + ".")

        evidence = {
            "node": "evaluation_adaption",
            "mode": mode,
            "evaluation_analysis": self._limit_context(state.get("evaluation_analysis", "")),
            "evaluation_metrics": state.get("evaluation_metrics", {}),
            "before_evaluation": self._limit_context(before_code),
            "after_evaluation": self._limit_context(after_code),
            "evaluation_function_usage": usage,
            "changed_lines": info.get("all_changed_lines", []),
            "fallback_summary_lines": fallback_lines,
            "prior_error": self._extract_error_excerpt(state.get("evaluation_execution_result")),
        }
        payload = self._run_structured_summary_extractor("evaluation_adaption", evidence, "evaluation_adaption")
        summary_lines = self._render_evaluation_adaption_summary_lines(payload, fallback_lines)

        return {
            "registered": True,
            "node_name": "evaluation_adaption",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_lines": summary_lines,
            "summary_clauses": [],
            "file_bundle": file_bundle,
        }

    def _extract_run_matching_tester_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        output = self._safe_dict(node_output)
        cfg = output.get("matching_config") if output.get("matching_config") is not None else state.get("matching_config")
        matcher_mode = state.get("matcher_mode") or output.get("matcher_mode") or "rule_based"
        source = self._infer_config_source(state, "matching_config", "matching_config_path")
        pair_keys = sorted(cfg.keys()) if isinstance(cfg, dict) else []
        score_snippets = self._collect_pair_scores(cfg)
        summary_clauses = [
            f"matcher mode `{self._normalize_matcher_mode(matcher_mode)}` with config source {source}",
        ]
        if pair_keys:
            summary_clauses.append(f"evaluated pairs: {self._compact_list(pair_keys, max_items=10)}")
        if score_snippets:
            summary_clauses.append(f"selected scores: {self._compact_list(score_snippets, max_items=6)}")
        if isinstance(cfg, dict):
            summary_clauses.append(f"produced {len(cfg)} pair strategy entries")
        return {
            "registered": True,
            "node_name": "run_matching_tester",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_clauses": summary_clauses,
            "effect_clause": f"matching config is ready for `{next_node}`" if next_node and next_node != "PENDING" else "",
            "file_bundle": file_bundle,
        }

    def _extract_run_blocking_tester_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        output = self._safe_dict(node_output)
        cfg = output.get("blocking_config") if output.get("blocking_config") is not None else state.get("blocking_config")
        source = self._infer_config_source(state, "blocking_config", "blocking_config_path")
        strategies = {}
        if isinstance(cfg, dict):
            strategies = cfg.get("blocking_strategies") if isinstance(cfg.get("blocking_strategies"), dict) else cfg

        summary_lines = []
        if source == "reused from state":
            summary_lines.append("Reused the blocking configuration from state.")
        elif source == "loaded from file":
            summary_lines.append("Loaded the blocking configuration from file.")
        else:
            summary_lines.append("Computed a new blocking configuration.")

        if isinstance(strategies, dict) and strategies:
            strategy_groups: Dict[str, List[str]] = {}
            candidate_counts: List[int] = []
            completeness_values: List[float] = []
            representative_columns: Dict[str, List[str]] = {}
            for pair_key, pair_cfg in strategies.items():
                if not isinstance(pair_cfg, dict):
                    continue
                strategy = pair_cfg.get("blocker_type") or pair_cfg.get("strategy") or pair_cfg.get("blocker") or "unknown"
                strategy_groups.setdefault(str(strategy), []).append(str(pair_key))
                candidate_count = pair_cfg.get("candidate_count") or pair_cfg.get("num_candidates")
                if isinstance(candidate_count, (int, float)):
                    candidate_counts.append(int(candidate_count))
                completeness = pair_cfg.get("pair_completeness")
                if isinstance(completeness, (int, float)):
                    completeness_values.append(float(completeness))
                columns = pair_cfg.get("columns")
                if isinstance(columns, list) and columns:
                    representative_columns.setdefault(str(strategy), [str(col) for col in columns])

            strategy_bits = []
            for strategy, pairs in strategy_groups.items():
                bit = f"`{strategy}` for {self._compact_list(sorted(pairs), max_items=4)}"
                columns = representative_columns.get(strategy) or []
                if columns:
                    bit += f", blocking on {self._compact_list(columns, max_items=4)}"
                strategy_bits.append(bit)
            if strategy_bits:
                summary_lines.append("Blocking uses " + "; ".join(strategy_bits) + ".")

            performance_bits = []
            if completeness_values:
                if len(set(round(v, 4) for v in completeness_values)) == 1:
                    performance_bits.append(f"pair completeness is {completeness_values[0]:.2f} for all pairs")
                else:
                    performance_bits.append(
                        f"pair completeness ranges from {min(completeness_values):.2f} to {max(completeness_values):.2f}"
                    )
            if candidate_counts:
                performance_bits.append(
                    f"candidate counts range from {min(candidate_counts)} to {max(candidate_counts)}"
                )
            if performance_bits:
                summary_lines.append("Performance-wise, " + " and ".join(performance_bits) + ".")

        if isinstance(cfg, dict):
            extra_bits = []
            if isinstance(cfg.get("id_columns"), dict) and cfg.get("id_columns"):
                extra_bits.append("dataset id columns")
            if isinstance(cfg.get("fusion_size_estimate"), dict) and cfg.get("fusion_size_estimate"):
                estimate = cfg["fusion_size_estimate"].get("expected_rows")
                if estimate is not None:
                    extra_bits.append(f"a blocking-stage fusion estimate of {estimate} rows")
            if extra_bits:
                summary_lines.append("The config also records " + " and ".join(extra_bits) + ".")
        return {
            "registered": True,
            "node_name": "run_blocking_tester",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_lines": summary_lines,
            "summary_clauses": [],
            "file_bundle": file_bundle,
        }

    def _extract_normalization_node_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        output = self._safe_dict(node_output)
        execution_result = output.get("normalization_execution_result", state.get("normalization_execution_result", ""))
        attempts = output.get("normalization_attempts", state.get("normalization_attempts"))
        report = output.get("normalization_report")
        if not isinstance(report, dict):
            report = state.get("normalization_report", {}) if isinstance(state.get("normalization_report"), dict) else {}
        directives = output.get("normalization_directives")
        if not isinstance(directives, dict):
            directives = state.get("normalization_directives", {}) if isinstance(state.get("normalization_directives"), dict) else {}
        normalized_datasets = output.get("normalized_datasets")
        if not isinstance(normalized_datasets, list):
            normalized_datasets = state.get("normalized_datasets", []) if isinstance(state.get("normalized_datasets"), list) else []
        report_status = report.get("status") if isinstance(report, dict) else None
        warnings = report.get("warnings", []) if isinstance(report, dict) else []
        if not isinstance(warnings, list):
            warnings = []
        failure_tags = report.get("failure_tags", []) if isinstance(report, dict) else []
        if not isinstance(failure_tags, list):
            failure_tags = []
        reverted_to_original = bool(report.get("reverted_to_original")) if isinstance(report, dict) else False
        validation_style = report.get("validation_style", {}) if isinstance(report, dict) else {}
        if not isinstance(validation_style, dict):
            validation_style = {}
        ablation_report = report.get("ablation_report", {}) if isinstance(report, dict) else {}
        if not isinstance(ablation_report, dict):
            ablation_report = {}
        shadow_precheck = report.get("shadow_precheck", {}) if isinstance(report, dict) else {}
        if not isinstance(shadow_precheck, dict):
            shadow_precheck = {}
        acceptance_gate = report.get("acceptance_gate", {}) if isinstance(report, dict) else {}
        if not isinstance(acceptance_gate, dict):
            acceptance_gate = {}
        dataset_report = report.get("datasets", {}) if isinstance(report, dict) else {}
        if not isinstance(dataset_report, dict):
            dataset_report = {}

        transform_analysis = self._analyze_normalization_transforms(dataset_report)
        dominant_signature = transform_analysis.get("dominant_signature")
        dominant_count = int(transform_analysis.get("dominant_count", 0) or 0)
        total_transformed = int(transform_analysis.get("total_transformed_columns", 0) or 0)
        exception_columns = transform_analysis.get("exception_columns", [])
        if not isinstance(exception_columns, list):
            exception_columns = []

        summary_lines = []
        if str(execution_result).startswith("success"):
            summary_lines.append(
                f"Normalization succeeded on attempt {attempts} and produced {len(normalized_datasets)} normalized dataset files."
            )
        elif str(execution_result).startswith("skipped_by_shadow_gate"):
            reason = str(shadow_precheck.get("reason") or "projected gain was too low")
            summary_lines.append(
                f"Normalization was skipped by the shadow gate on attempt {attempts} because {reason}."
            )
        elif str(execution_result).startswith("fallback_to_original"):
            summary_lines.append(
                f"Normalization fell back to original datasets on attempt {attempts} due to runtime or compatibility issues."
            )
        else:
            summary_lines.append(
                f"Normalization finished with status `{execution_result}` on attempt {attempts}."
            )

        success_count = 0
        failed_count = 0
        list_norm_columns: List[str] = []
        country_norm_columns: List[str] = []
        transform_samples: List[str] = []
        for dataset_name, entry in dataset_report.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("status") == "failed":
                failed_count += 1
            else:
                success_count += 1
            list_cols = entry.get("list_normalized_columns", [])
            if isinstance(list_cols, list):
                list_norm_columns.extend([f"{dataset_name}.{c}" for c in list_cols if str(c).strip()])
            country_cols = entry.get("country_normalized_columns", [])
            if isinstance(country_cols, list):
                country_norm_columns.extend([f"{dataset_name}.{c}" for c in country_cols if str(c).strip()])
            transforms = entry.get("applied_transforms", {})
            if isinstance(transforms, dict):
                for col, spec in list(transforms.items())[:2]:
                    signature = self._normalize_transform_signature(spec)
                    if signature:
                        transform_samples.append(f"{dataset_name}.{col}:{signature}")

        if dataset_report:
            summary_lines.append(
                f"Dataset-level outcome: {success_count} succeeded and {failed_count} failed during normalization processing."
            )

        if dominant_signature and total_transformed > 0:
            if exception_columns:
                summary_lines.append(
                    f"Dominant transform pattern was `{dominant_signature}` on {dominant_count}/{total_transformed} transformed columns; exceptions include {self._compact_list(exception_columns, max_items=5)}."
                )
            else:
                summary_lines.append(
                    f"All {total_transformed} transformed columns used `{dominant_signature}`."
                )
        elif transform_samples:
            summary_lines.append(
                "Applied transform examples: " + self._compact_list(transform_samples, max_items=4) + "."
            )

        directive_bits = []
        if isinstance(directives, dict):
            list_cols = directives.get("list_columns", [])
            country_cols = directives.get("country_columns", [])
            if isinstance(list_cols, list) and list_cols:
                directive_bits.append(f"list normalization targets {self._compact_list([str(c) for c in list_cols], max_items=5)}")
            if isinstance(country_cols, list) and country_cols:
                directive_bits.append(f"country normalization targets {self._compact_list([str(c) for c in country_cols], max_items=5)}")
        if directive_bits:
            summary_lines.append("Directive focus: " + " and ".join(directive_bits) + ".")

        if list_norm_columns or country_norm_columns:
            coverage_bits = []
            if list_norm_columns:
                coverage_bits.append(f"list-normalized columns include {self._compact_list(sorted(set(list_norm_columns)), max_items=5)}")
            if country_norm_columns:
                coverage_bits.append(f"country-normalized columns include {self._compact_list(sorted(set(country_norm_columns)), max_items=5)}")
            summary_lines.append("Applied normalization coverage: " + " and ".join(coverage_bits) + ".")

        quality_bits = []
        if warnings:
            quality_bits.append(f"{len(warnings)} warning(s): {self._compact_list([str(w) for w in warnings], max_items=2)}")
        if failure_tags:
            quality_bits.append(f"failure tags={self._compact_list([str(t) for t in failure_tags], max_items=4)}")
        if reverted_to_original:
            quality_bits.append("run reverted to original datasets")
        if isinstance(shadow_precheck, dict) and shadow_precheck:
            if shadow_precheck.get("projected_delta") is not None:
                quality_bits.append(
                    f"shadow projected_delta={shadow_precheck.get('projected_delta')} (allow={shadow_precheck.get('allow')})"
                )
        if isinstance(ablation_report, dict) and ablation_report.get("selected_keys"):
            selected = ablation_report.get("selected_keys", [])
            if isinstance(selected, list) and selected:
                quality_bits.append(f"ablation selected keys={self._compact_list([str(k) for k in selected], max_items=5)}")
        if isinstance(acceptance_gate, dict) and acceptance_gate:
            requested = acceptance_gate.get("requested")
            if requested is not None:
                quality_bits.append(f"acceptance gate requested={requested}")
        if quality_bits:
            summary_lines.append("Quality controls and safeguards: " + "; ".join(quality_bits) + ".")

        style_bits = []
        used_country_format = validation_style.get("used_country_output_format")
        if used_country_format:
            style_bits.append(f"country output format `{used_country_format}`")
        list_hint = validation_style.get("validation_list_like_columns_hint")
        if isinstance(list_hint, list) and list_hint:
            style_bits.append(f"validation list-like hints {self._compact_list([str(c) for c in list_hint], max_items=4)}")
        if style_bits:
            summary_lines.append("Validation-style alignment used " + " and ".join(style_bits) + ".")

        evidence = {
            "node": "normalization_node",
            "execution_result": execution_result,
            "attempts": attempts,
            "report_status": report_status,
            "reverted_to_original": reverted_to_original,
            "normalization_directives": directives,
            "validation_style": validation_style,
            "ablation_report": ablation_report,
            "shadow_precheck": shadow_precheck,
            "acceptance_gate": acceptance_gate,
            "dataset_report": dataset_report,
            "transform_analysis": transform_analysis,
            "warnings": warnings,
            "failure_tags": failure_tags,
            "next_node": next_node,
            "fallback_summary_lines": summary_lines,
        }
        payload = self._run_structured_summary_extractor("normalization_node", evidence, "normalization_node")
        summary_lines = self._render_normalization_summary_lines(payload, summary_lines)

        return {
            "registered": True,
            "node_name": "normalization_node",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_lines": summary_lines,
            "summary_clauses": [],
            "file_bundle": file_bundle,
        }

    def _extract_evaluation_node_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        output = self._safe_dict(node_output)
        execution_result = output.get("evaluation_execution_result", state.get("evaluation_execution_result", ""))
        attempts = output.get("evaluation_execution_attempts", state.get("evaluation_execution_attempts"))
        source = output.get("evaluation_metrics_source", state.get("evaluation_metrics_source"))
        metrics = output.get("evaluation_metrics_from_execution")
        if not isinstance(metrics, dict):
            metrics = state.get("evaluation_metrics_from_execution", {})
        if not isinstance(metrics, dict) or not metrics:
            metrics = state.get("evaluation_metrics", {}) if isinstance(state.get("evaluation_metrics"), dict) else {}

        summary_lines = []
        error_mode = isinstance(execution_result, str) and execution_result.lower().startswith("error")
        if error_mode:
            evaluation_code_path = self._code_path_for_node("evaluation_node")
            summary_lines.extend(
                self._build_error_summary_lines(
                    node_label="Evaluation execution",
                    attempts=attempts,
                    raw_error=execution_result,
                    code_path=evaluation_code_path,
                )
            )
        elif str(execution_result).startswith("success"):
            summary_lines.append(
                f"The consolidated evaluation node succeeded after {attempts} execution attempt(s)."
            )
        else:
            summary_lines.append(
                f"The consolidated evaluation node completed with status `{execution_result}` after {attempts} attempt(s)."
            )

        if not error_mode:
            accuracy = self._format_accuracy(metrics)
            if accuracy:
                summary_lines.append(f"Observed overall accuracy from this evaluation pass is {accuracy}.")
            if source:
                summary_lines.append(f"Metrics source for this pass is `{source}`.")
            if isinstance(metrics, dict):
                macro = metrics.get("macro_accuracy")
                if isinstance(macro, (int, float)):
                    macro_pct = float(macro) * 100.0 if float(macro) <= 1.0 else float(macro)
                    summary_lines.append(f"Macro accuracy is {macro_pct:.2f}% for this evaluation stage.")

        return {
            "registered": True,
            "node_name": "evaluation_node",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_lines": summary_lines,
            "summary_clauses": [],
            "file_bundle": file_bundle,
        }

    def _extract_investigator_node_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        output = self._safe_dict(node_output)
        decision = output.get("investigator_decision", state.get("investigator_decision", next_node))
        metrics = output.get("evaluation_metrics")
        if not isinstance(metrics, dict):
            metrics = state.get("evaluation_metrics", {}) if isinstance(state.get("evaluation_metrics"), dict) else {}
        attempts = output.get("evaluation_attempts", state.get("evaluation_attempts"))
        diagnostics_result = output.get("integration_diagnostics_execution_result", state.get("integration_diagnostics_execution_result", ""))
        diagnostics_report = output.get("integration_diagnostics_report")
        if not isinstance(diagnostics_report, dict):
            diagnostics_report = state.get("integration_diagnostics_report", {}) if isinstance(state.get("integration_diagnostics_report"), dict) else {}
        auto_diagnostics = output.get("auto_diagnostics")
        if not isinstance(auto_diagnostics, dict):
            auto_diagnostics = state.get("auto_diagnostics", {}) if isinstance(state.get("auto_diagnostics"), dict) else {}
        normalization_required = output.get(
            "normalization_rework_required",
            state.get("normalization_rework_required"),
        )
        reasons = output.get("normalization_rework_reasons")
        if not isinstance(reasons, list):
            reasons = state.get("normalization_rework_reasons", []) if isinstance(state.get("normalization_rework_reasons"), list) else []
        action_plan = output.get("investigator_action_plan")
        if not isinstance(action_plan, list):
            action_plan = state.get("investigator_action_plan", []) if isinstance(state.get("investigator_action_plan"), list) else []
        probe_results = output.get("investigator_probe_results")
        if not isinstance(probe_results, dict):
            probe_results = state.get("investigator_probe_results", {}) if isinstance(state.get("investigator_probe_results"), dict) else {}
        routing_objective = output.get("investigator_routing_objective")
        if not isinstance(routing_objective, dict):
            routing_objective = state.get("investigator_routing_objective", {}) if isinstance(state.get("investigator_routing_objective"), dict) else {}
        routing = output.get("investigator_routing_decision")
        if not isinstance(routing, dict):
            routing = state.get("investigator_routing_decision", {}) if isinstance(state.get("investigator_routing_decision"), dict) else {}
        evaluation_analysis = output.get("evaluation_analysis")
        if not isinstance(evaluation_analysis, str):
            evaluation_analysis = str(state.get("evaluation_analysis", "") or "")

        summary_lines = []
        summary_lines.append(
            f"Investigator decided to route the workflow to `{decision}` at evaluation attempt {attempts}."
        )
        accuracy = self._format_accuracy(metrics)
        if accuracy:
            summary_lines.append(f"This routing decision is based on current overall accuracy {accuracy}.")
        if isinstance(diagnostics_result, str) and diagnostics_result:
            summary_lines.append(f"Integration diagnostics status for this cycle is `{diagnostics_result}`.")
        if isinstance(normalization_required, bool):
            if normalization_required:
                summary_lines.append(
                    "The investigator flagged normalization rework as required for the current failure pattern."
                )
            else:
                summary_lines.append(
                    "The investigator did not require additional normalization rework in this cycle."
                )
        if reasons:
            summary_lines.append(
                "Normalization rationale includes: " + self._compact_list([str(r) for r in reasons], max_items=3) + "."
            )
        if action_plan:
            top = action_plan[0] if isinstance(action_plan[0], dict) else {}
            action = str(top.get("action", "")).strip()
            targets = top.get("target_attributes", [])
            target_txt = ""
            if isinstance(targets, list) and targets:
                target_txt = self._compact_list([str(t) for t in targets], max_items=4)
            if action:
                line = f"Top proposed fix is `{action}`"
                if target_txt:
                    line += f" targeting {target_txt}"
                line += "."
                summary_lines.append(line)
        if isinstance(routing, dict) and routing:
            score = routing.get("score")
            threshold = routing.get("threshold")
            if score is not None or threshold is not None:
                summary_lines.append(
                    f"Routing score details: score={score}, threshold={threshold}, route_to_normalization={routing.get('route_to_normalization')}."
                )
        if isinstance(probe_results, dict) and probe_results:
            pressure = probe_results.get("normalization_pressure")
            best_repair = probe_results.get("best_repair_action")
            if pressure is not None:
                probe_line = f"Probe-estimated normalization pressure is {pressure}"
                if best_repair:
                    probe_line += f", with best repair action `{best_repair}`"
                probe_line += "."
                summary_lines.append(probe_line)
        if isinstance(auto_diagnostics, dict) and auto_diagnostics:
            reason_ratios = auto_diagnostics.get("debug_reason_ratios", {})
            if isinstance(reason_ratios, dict) and reason_ratios:
                dominant_reason = sorted(
                    reason_ratios.items(),
                    key=lambda item: float(item[1]) if isinstance(item[1], (int, float)) else -1.0,
                    reverse=True,
                )[0]
                summary_lines.append(
                    f"Dominant mismatch signal is `{dominant_reason[0]}` with ratio {dominant_reason[1]}."
                )
        if isinstance(diagnostics_report, dict) and diagnostics_report:
            issue_count = diagnostics_report.get("issue_count")
            severity = diagnostics_report.get("severity")
            if issue_count is not None or severity is not None:
                summary_lines.append(
                    f"Diagnostics report indicates issue_count={issue_count} and severity={severity}."
                )

        evidence = {
            "node": "investigator_node",
            "decision": decision,
            "attempts": attempts,
            "metrics": metrics,
            "diagnostics_result": diagnostics_result,
            "diagnostics_report": diagnostics_report,
            "auto_diagnostics": auto_diagnostics,
            "normalization_rework_required": normalization_required,
            "normalization_rework_reasons": reasons,
            "action_plan": action_plan,
            "probe_results": probe_results,
            "routing_objective": routing_objective,
            "routing_decision": routing,
            "evaluation_analysis": self._limit_context(evaluation_analysis, 5000),
            "next_node": next_node,
            "fallback_summary_lines": summary_lines,
        }
        payload = self._run_structured_summary_extractor("investigator_node", evidence, "investigator_node")
        summary_lines = self._render_investigator_summary_lines(payload, summary_lines)

        return {
            "registered": True,
            "node_name": "investigator_node",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_lines": summary_lines,
            "summary_clauses": [],
            "file_bundle": file_bundle,
        }

    def _extract_human_review_export_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        output = self._safe_dict(node_output)
        execution_result = output.get("human_review_execution_result", state.get("human_review_execution_result", ""))
        report = output.get("human_review_report")
        if not isinstance(report, dict):
            report = state.get("human_review_report", {}) if isinstance(state.get("human_review_report"), dict) else {}

        summary_lines = []
        if str(execution_result).startswith("success"):
            summary_lines.append("Human review export completed successfully and created reviewer-facing artifacts.")
        elif str(execution_result).startswith("error"):
            summary_lines.append(
                f"Human review export failed with: {self._extract_error_excerpt(execution_result)}."
            )
        else:
            summary_lines.append(f"Human review export finished with status `{execution_result}`.")

        if isinstance(report, dict) and report:
            file_paths = report.get("file_paths")
            if isinstance(file_paths, dict) and file_paths:
                known_files = sorted([str(k) for k in file_paths.keys()])
                summary_lines.append(
                    "Generated human-review files include: "
                    + self._compact_list(known_files, max_items=6)
                    + "."
                )
            counts = report.get("counts")
            if isinstance(counts, dict) and counts:
                metric_bits = []
                for key in ["fused_rows", "review_rows", "lineage_rows", "diff_rows"]:
                    if counts.get(key) is not None:
                        metric_bits.append(f"{key}={counts.get(key)}")
                if metric_bits:
                    summary_lines.append("Review package counts: " + ", ".join(metric_bits) + ".")
            warnings = report.get("warnings")
            if isinstance(warnings, list) and warnings:
                summary_lines.append(
                    f"Human-review generation recorded {len(warnings)} warning(s), including {self._compact_list([str(w) for w in warnings], max_items=2)}."
                )

        return {
            "registered": True,
            "node_name": "human_review_export",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_lines": summary_lines,
            "summary_clauses": [],
            "file_bundle": file_bundle,
        }

    def _extract_sealed_final_test_evaluation_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        output = self._safe_dict(node_output)
        execution_result = output.get(
            "final_test_evaluation_execution_result",
            state.get("final_test_evaluation_execution_result", ""),
        )
        metrics = output.get("final_test_evaluation_metrics")
        if not isinstance(metrics, dict):
            metrics = state.get("final_test_evaluation_metrics", {}) if isinstance(state.get("final_test_evaluation_metrics"), dict) else {}

        summary_lines = []
        if execution_result == "skipped":
            summary_lines.append("Sealed final-test evaluation was skipped because held-out test mode was not active.")
        elif str(execution_result).startswith("success"):
            summary_lines.append("Sealed final-test evaluation completed successfully on the held-out test split.")
        elif str(execution_result).startswith("error"):
            summary_lines.append(
                f"Sealed final-test evaluation failed with: {self._extract_error_excerpt(execution_result)}."
            )
        else:
            summary_lines.append(f"Sealed final-test evaluation finished with status `{execution_result}`.")

        accuracy = self._format_accuracy(metrics)
        if accuracy:
            summary_lines.append(f"Held-out test overall accuracy for this final run is {accuracy}.")
        if isinstance(metrics, dict):
            macro = metrics.get("macro_accuracy")
            if isinstance(macro, (int, float)):
                macro_pct = float(macro) * 100.0 if float(macro) <= 1.0 else float(macro)
                summary_lines.append(f"Held-out macro accuracy is {macro_pct:.2f}%.")

        return {
            "registered": True,
            "node_name": "sealed_final_test_evaluation",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_lines": summary_lines,
            "summary_clauses": [],
            "file_bundle": file_bundle,
        }

    def _extract_save_results_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        output = self._safe_dict(node_output)
        run_audit_path = output.get("run_audit_path", state.get("run_audit_path", ""))
        run_report_path = output.get("run_report_path", state.get("run_report_path", ""))
        run_id = output.get("run_id", state.get("run_id", ""))
        run_output_root = output.get("run_output_root", state.get("run_output_root", ""))
        pipeline_snapshots = state.get("pipeline_snapshots", [])
        evaluation_snapshots = state.get("evaluation_snapshots", [])

        summary_lines = []
        summary_lines.append("Run results were persisted for reproducibility and offline analysis.")
        if run_id:
            summary_lines.append(f"Run identifier is `{run_id}`.")
        if run_output_root:
            summary_lines.append(f"Run artifacts root directory is `{run_output_root}`.")
        if run_audit_path or run_report_path:
            paths = [str(p) for p in [run_audit_path, run_report_path] if str(p).strip()]
            summary_lines.append("Saved report artifacts: " + self._compact_list(paths, max_items=4) + ".")
        if isinstance(pipeline_snapshots, list) or isinstance(evaluation_snapshots, list):
            summary_lines.append(
                f"Snapshot counts captured in state: pipeline={len(pipeline_snapshots) if isinstance(pipeline_snapshots, list) else 0}, "
                f"evaluation={len(evaluation_snapshots) if isinstance(evaluation_snapshots, list) else 0}."
            )

        return {
            "registered": True,
            "node_name": "save_results",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_lines": summary_lines,
            "summary_clauses": [],
            "file_bundle": file_bundle,
        }

    def _extract_profile_data_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        output = self._safe_dict(node_output)
        profiles = output.get("data_profiles") if output.get("data_profiles") is not None else state.get("data_profiles")
        dataset_names = self._dataset_names_from_state(state)
        summary_clauses = []
        if dataset_names:
            summary_clauses.append(f"profiled datasets: {self._compact_list(dataset_names, max_items=12)}")
        if isinstance(profiles, dict):
            dataset_bits = []
            anomaly_bits = []
            for name, profile in profiles.items():
                if not isinstance(profile, dict):
                    continue
                row_count = (
                    profile.get("num_rows")
                    or profile.get("row_count")
                    or profile.get("rows")
                    or profile.get("n_rows")
                )
                col_count = (
                    profile.get("num_columns")
                    or profile.get("column_count")
                    or profile.get("cols")
                    or profile.get("n_columns")
                )
                bit = str(name)
                extras = []
                if row_count is not None:
                    extras.append(f"rows={row_count}")
                if col_count is not None:
                    extras.append(f"cols={col_count}")
                if extras:
                    bit += f" ({', '.join(extras)})"
                dataset_bits.append(bit)
                for anomaly_key in ["missing_columns", "high_null_columns", "empty_columns", "type_issues"]:
                    anomaly = profile.get(anomaly_key)
                    if anomaly:
                        anomaly_bits.append(f"{name}:{anomaly_key}")
            if dataset_bits:
                summary_clauses.append(f"profile details: {self._compact_list(dataset_bits, max_items=8)}")
            if anomaly_bits:
                summary_clauses.append(f"notable data quality issues: {self._compact_list(anomaly_bits, max_items=8)}")
        return {
            "registered": True,
            "node_name": "profile_data",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_clauses": summary_clauses or ["profile output captured for downstream generation"],
            "effect_clause": f"profile context is ready for `{next_node}`" if next_node and next_node != "PENDING" else "",
            "file_bundle": file_bundle,
        }

    def _extract_match_schemas_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        output = self._safe_dict(node_output)
        correspondences = (
            output.get("schema_correspondences")
            if output.get("schema_correspondences") is not None
            else state.get("schema_correspondences")
        )
        dataset_names = self._dataset_names_from_state(state)
        summary_clauses = []
        if dataset_names:
            summary_clauses.append(f"aligned schemas for datasets: {self._compact_list(dataset_names, max_items=12)}")
        if isinstance(correspondences, dict):
            pair_count = len(correspondences)
            summary_clauses.append(f"generated {pair_count} schema correspondence groups")
            unmatched = []
            for pair_key, value in correspondences.items():
                if isinstance(value, dict):
                    if value.get("unmatched_columns"):
                        unmatched.append(f"{pair_key}:{len(value.get('unmatched_columns', []))} unmatched")
                    elif value.get("matches"):
                        summary_clauses.append(f"{pair_key} mapped {len(value.get('matches', []))} fields")
                        break
            if unmatched:
                summary_clauses.append(f"unmatched schema hints: {self._compact_list(unmatched, max_items=6)}")
        elif isinstance(correspondences, list):
            summary_clauses.append(f"generated {len(correspondences)} schema correspondences")
        return {
            "registered": True,
            "node_name": "match_schemas",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_clauses": summary_clauses or ["schema correspondence output updated"],
            "effect_clause": f"schema mapping is ready for `{next_node}`" if next_node and next_node != "PENDING" else "",
            "file_bundle": file_bundle,
        }

    def _extract_execute_pipeline_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        output = self._safe_dict(node_output)
        execution_result = output.get("pipeline_execution_result")
        if execution_result is None:
            execution_result = state.get("pipeline_execution_result")
        attempts = output.get("pipeline_execution_attempts", state.get("pipeline_execution_attempts"))
        fusion_size = output.get("fusion_size_comparison") or state.get("fusion_size_comparison") or {}
        fusion_csv = os.path.join(self.output_dir, "data_fusion", "fusion_data.csv")
        summary_clauses = []
        if isinstance(execution_result, str) and execution_result.lower().startswith("error"):
            pipeline_code_path = self._code_path_for_node("execute_pipeline")
            error_lines = self._build_error_summary_lines(
                node_label="Pipeline execution",
                attempts=attempts,
                raw_error=execution_result,
                code_path=pipeline_code_path,
            )
            return {
                "registered": True,
                "node_name": "execute_pipeline",
                "next_node": next_node,
                "status": status,
                "error": error_text or "",
                "summary_lines": error_lines,
                "summary_clauses": [],
                "file_bundle": file_bundle,
            }
        else:
            summary_clauses.append(f"pipeline execution succeeded on attempt {attempts}")
        if os.path.exists(fusion_csv):
            summary_clauses.append("fusion_data.csv was produced")
        if isinstance(fusion_size, dict) and fusion_size:
            estimate = fusion_size.get("estimated_size") or fusion_size.get("estimated_rows")
            actual = fusion_size.get("actual_size") or fusion_size.get("actual_rows")
            if estimate is not None or actual is not None:
                summary_clauses.append(f"fusion size estimate vs actual: {estimate} vs {actual}")
        return {
            "registered": True,
            "node_name": "execute_pipeline",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_clauses": summary_clauses,
            "effect_clause": f"next run proceeds to `{next_node}`" if next_node and next_node != "PENDING" else "",
            "file_bundle": file_bundle,
        }

    def _extract_execute_evaluation_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        output = self._safe_dict(node_output)
        execution_result = output.get("evaluation_execution_result")
        if execution_result is None:
            execution_result = state.get("evaluation_execution_result")
        attempts = output.get("evaluation_execution_attempts", state.get("evaluation_execution_attempts"))
        eval_json = os.path.join(self.output_dir, "pipeline_evaluation", "pipeline_evaluation.json")
        summary_clauses = []
        if isinstance(execution_result, str) and execution_result.lower().startswith("error"):
            evaluation_code_path = self._code_path_for_node("execute_evaluation")
            error_lines = self._build_error_summary_lines(
                node_label="Evaluation execution",
                attempts=attempts,
                raw_error=execution_result,
                code_path=evaluation_code_path,
            )
            return {
                "registered": True,
                "node_name": "execute_evaluation",
                "next_node": next_node,
                "status": status,
                "error": error_text or "",
                "summary_lines": error_lines,
                "summary_clauses": [],
                "file_bundle": file_bundle,
            }
        else:
            summary_clauses.append(f"evaluation execution succeeded on attempt {attempts}")
        if os.path.exists(eval_json):
            summary_clauses.append("pipeline_evaluation.json is available")
        return {
            "registered": True,
            "node_name": "execute_evaluation",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_clauses": summary_clauses,
            "effect_clause": f"next run proceeds to `{next_node}`" if next_node and next_node != "PENDING" else "",
            "file_bundle": file_bundle,
        }

    def _extract_evaluation_decision_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        output = self._safe_dict(node_output)
        metrics = output.get("evaluation_metrics") if isinstance(output.get("evaluation_metrics"), dict) else state.get("evaluation_metrics", {})
        accuracy = self._format_accuracy(metrics)
        previous_accuracy = self._evaluation_runs[-1].get("accuracy_score") if self._evaluation_runs else None
        attempts = output.get("evaluation_attempts", state.get("evaluation_attempts"))
        summary_clauses = []
        if accuracy is not None:
            summary_clauses.append(f"overall accuracy is {accuracy}")
        if previous_accuracy and accuracy and previous_accuracy != accuracy:
            summary_clauses.append(f"previous logged accuracy was {previous_accuracy}")
        if attempts is not None:
            summary_clauses.append(f"evaluation attempt counter is {attempts}")
        if next_node and next_node != "PENDING":
            summary_clauses.append(f"decision routes to `{next_node}`")
        return {
            "registered": True,
            "node_name": "evaluation_decision",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_clauses": summary_clauses or ["evaluation decision updated workflow state"],
            "effect_clause": "",
            "file_bundle": file_bundle,
        }

    def _extract_evaluation_reasoning_facts(
        self,
        state_in: Any,
        node_output: Any,
        file_bundle: Dict[str, Dict[str, Any]],
        status: str,
        error_text: Optional[str],
        next_node: str,
    ) -> Dict[str, Any]:
        state = self._safe_dict(state_in)
        output = self._safe_dict(node_output)
        analysis = output.get("evaluation_analysis")
        if analysis is None:
            analysis = state.get("evaluation_analysis")
        analysis_text = self._coerce_response_text(analysis)
        metrics = output.get("evaluation_metrics") if isinstance(output.get("evaluation_metrics"), dict) else state.get("evaluation_metrics", {})
        pipeline_code = self._read_file_if_exists(os.path.join(self.output_dir, "code", "pipeline.py"))
        evaluation_code = self._read_file_if_exists(os.path.join(self.output_dir, "code", "evaluation.py"))
        fallback_lines = self._summarize_reasoning_text(analysis_text, metrics)
        evidence = {
            "node": "evaluation_reasoning",
            "analysis": self._limit_context(analysis_text),
            "evaluation_metrics": metrics if isinstance(metrics, dict) else {},
            "auto_diagnostics": state.get("auto_diagnostics", {}),
            "current_pipeline": self._limit_context(pipeline_code),
            "current_evaluation": self._limit_context(evaluation_code),
            "pipeline_execution_result": self._extract_error_excerpt(state.get("pipeline_execution_result")),
            "evaluation_execution_result": self._extract_error_excerpt(state.get("evaluation_execution_result")),
            "fallback_summary_lines": fallback_lines,
        }
        payload = self._run_structured_summary_extractor("evaluation_reasoning", evidence, "evaluation_reasoning")
        lines = self._render_reasoning_summary_lines(payload, fallback_lines, metrics if isinstance(metrics, dict) else {})
        return {
            "registered": True,
            "node_name": "evaluation_reasoning",
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_lines": lines or ["Evaluation reasoning produced the next repair direction."],
            "summary_clauses": [],
            "file_bundle": file_bundle,
        }

    def _extract_attribute_fusers(self, pipeline_code: str) -> Dict[str, str]:
        if not pipeline_code:
            return {}
        try:
            tree = ast.parse(pipeline_code)
            fusers = {}
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_attribute_fuser":
                    continue
                if len(node.args) < 2:
                    continue
                key_arg = node.args[0]
                fn_arg = node.args[1]
                if isinstance(key_arg, ast.Constant) and isinstance(key_arg.value, str):
                    fn_text = ast.get_source_segment(pipeline_code, fn_arg)
                    if not fn_text and isinstance(fn_arg, ast.Name):
                        fn_text = fn_arg.id
                    if not fn_text:
                        fn_text = type(fn_arg).__name__
                    fusers[key_arg.value] = self._normalize_expr_text(fn_text)
            return fusers
        except Exception:
            return {}

    def _extract_evaluation_functions(self, evaluation_code: str) -> Dict[str, str]:
        if not evaluation_code:
            return {}
        try:
            tree = ast.parse(evaluation_code)
            out = {}

            def fn_text(expr):
                text = ast.get_source_segment(evaluation_code, expr)
                if not text and isinstance(expr, ast.Name):
                    text = expr.id
                if not text:
                    text = type(expr).__name__
                return self._normalize_expr_text(text)

            def walk(node, loop_bindings):
                if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and isinstance(
                    node.iter, (ast.List, ast.Tuple)
                ):
                    values = []
                    ok = True
                    for elt in node.iter.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            values.append(elt.value)
                        else:
                            ok = False
                            break
                    local_bindings = dict(loop_bindings)
                    if ok:
                        local_bindings[node.target.id] = values
                    for child in node.body:
                        walk(child, local_bindings)
                    for child in node.orelse:
                        walk(child, loop_bindings)
                    return

                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_evaluation_function":
                    if len(node.args) >= 2:
                        key_arg = node.args[0]
                        fn_arg = node.args[1]
                        func_name = fn_text(fn_arg)
                        if isinstance(key_arg, ast.Constant) and isinstance(key_arg.value, str):
                            out[key_arg.value] = func_name
                        elif isinstance(key_arg, ast.Name) and key_arg.id in loop_bindings:
                            for col in loop_bindings[key_arg.id]:
                                out[col] = func_name

                for child in ast.iter_child_nodes(node):
                    walk(child, loop_bindings)

            walk(tree, {})
            return out
        except Exception:
            return {}

    @staticmethod
    def _format_accuracy(metrics: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(metrics, dict):
            return None
        accuracy = metrics.get("overall_accuracy")
        if isinstance(accuracy, (int, float)):
            value = float(accuracy)
            if value <= 1.0:
                value *= 100.0
            return f"{value:.2f}%"
        if isinstance(accuracy, str):
            return accuracy.strip()
        return None

    def _extract_accepted_accuracy(
        self,
        state_in: Any,
        node_output: Any,
    ) -> Optional[str]:
        metrics_candidates: List[Dict[str, Any]] = []
        if isinstance(node_output, dict):
            out_eval = node_output.get("evaluation_metrics")
            if isinstance(out_eval, dict):
                metrics_candidates.append(out_eval)
            out_exec = node_output.get("evaluation_metrics_from_execution")
            if isinstance(out_exec, dict):
                metrics_candidates.append(out_exec)
        if isinstance(state_in, dict):
            state_eval = state_in.get("evaluation_metrics")
            if isinstance(state_eval, dict):
                metrics_candidates.append(state_eval)
            state_exec = state_in.get("evaluation_metrics_from_execution")
            if isinstance(state_exec, dict):
                metrics_candidates.append(state_exec)

        for metrics in metrics_candidates:
            acc = self._format_accuracy(metrics)
            if acc:
                return acc

        eval_path = os.path.join(self.output_dir, "pipeline_evaluation", "pipeline_evaluation.json")
        if os.path.exists(eval_path):
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    acc = self._format_accuracy(payload)
                    if acc:
                        return acc
            except Exception:
                pass
        return None

    def _update_density_gate(
        self,
        node_name: str,
        state_in: Any,
        node_output: Any,
    ) -> None:
        if self._has_current_run_density:
            return
        if node_name != "execute_pipeline":
            return

        execution_result = None
        if isinstance(node_output, dict):
            execution_result = node_output.get("pipeline_execution_result")
        if execution_result is None and isinstance(state_in, dict):
            execution_result = state_in.get("pipeline_execution_result")
        if not (isinstance(execution_result, str) and execution_result.lower().startswith("success")):
            return

        fusion_size = {}
        if isinstance(node_output, dict):
            fusion_size = node_output.get("fusion_size_comparison", {})
        if not isinstance(fusion_size, dict) and isinstance(state_in, dict):
            fusion_size = state_in.get("fusion_size_comparison", {})
        if not isinstance(fusion_size, dict):
            return

        comparisons = fusion_size.get("comparisons", {})
        if self._stage_from_comparisons(comparisons):
            self._has_current_run_density = True

    def _maybe_attach_snapshot_accuracy(
        self,
        node_name: str,
        state_in: Any,
        node_output: Any,
    ) -> None:
        if node_name not in {"evaluation_node", "evaluation_decision", "investigator_node"}:
            return
        accuracy_score = self._extract_accepted_accuracy(state_in, node_output)
        if not accuracy_score:
            return

        if node_name == "evaluation_node":
            if not self._pending_snapshot_indices:
                return
            attached_index = self._attach_accuracy_to_pending_snapshot(accuracy_score)
            if attached_index is not None:
                self._last_snapshot_accuracy_index = attached_index
            return

        # Fallback compatibility for architectures that expose decision-stage nodes:
        # - attach from pending queue when evaluation_node is absent
        # - otherwise overwrite latest evaluation-attached snapshot with newer decision accuracy
        if self._pending_snapshot_indices:
            attached_index = self._attach_accuracy_to_pending_snapshot(accuracy_score)
            if attached_index is not None:
                self._last_snapshot_accuracy_index = attached_index
            return

        self._overwrite_snapshot_accuracy(self._last_snapshot_accuracy_index, accuracy_score)

    def _build_evaluation_run_record(self, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pipeline_code = self._read_file_if_exists(os.path.join(self.output_dir, "code", "pipeline.py"))
        evaluation_code = self._read_file_if_exists(os.path.join(self.output_dir, "code", "evaluation.py"))

        acc = self._format_accuracy(metrics)
        if acc is None:
            # fallback to output file
            path = os.path.join(self.output_dir, "pipeline_evaluation", "pipeline_evaluation.json")
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    acc = self._format_accuracy(payload)
                except Exception:
                    acc = None

        return {
            "attribute_fusers": self._extract_attribute_fusers(pipeline_code),
            "evaluation_functions": self._extract_evaluation_functions(evaluation_code),
            "accuracy_score": acc,
        }

    def append_pipeline_snapshot(self, node_name: str, state_in: Any, node_output: Any):
        if node_name != "execute_pipeline":
            return
        if not self._pipeline_archive_path:
            return

        execution_result = None
        if isinstance(node_output, dict):
            execution_result = node_output.get("pipeline_execution_result")
        if execution_result is None and isinstance(state_in, dict):
            execution_result = state_in.get("pipeline_execution_result")
        if isinstance(execution_result, str) and execution_result.lower().startswith("error"):
            return

        pipeline_path = os.path.join(self.output_dir, "code", "pipeline.py")
        pipeline_code = self._read_file_if_exists(pipeline_path).rstrip()
        if not pipeline_code:
            return

        self._pipeline_snapshot_index += 1
        try:
            snapshot = {
                "snapshot_index": self._pipeline_snapshot_index,
                "node_index": self._node_index,
                "node_name": node_name,
                "accuracy_score": None,
                "pipeline_code": pipeline_code,
            }
            self._pipeline_snapshots.append(snapshot)
            self._pending_snapshot_indices.append(len(self._pipeline_snapshots) - 1)
            self._write_pipeline_archive_markdown()
        except Exception:
            # Snapshot archiving should never break workflow execution.
            pass

    def log_node(
        self,
        node_name: str,
        state_in: Any,
        call_fn: Callable[[], Any],
        token_usage: Dict[str, Any],
    ) -> Any:
        self.mark_next_for_previous(node_name)
        if isinstance(state_in, dict):
            self._latest_values_state = dict(state_in)

        before_tokens = dict(token_usage or {})
        start_ns = time.perf_counter_ns()
        before_files = self._capture_tracked_files(node_name)

        node_output = {}
        status = "ok"
        error_text = None
        raised_exception = False

        try:
            result = call_fn()
            if isinstance(result, dict):
                node_output = result
            elif result is None:
                node_output = {}
            else:
                node_output = {"result": str(result)}
        except Exception as e:
            status = "error"
            error_text = f"{type(e).__name__}: {str(e)}"
            result = None
            raised_exception = True

        after_files = self._capture_tracked_files(node_name)
        file_bundle = self._build_file_bundle(node_name, before_files, after_files)
        if isinstance(node_output, dict):
            self._latest_values_state.update(node_output)
        self._update_density_gate(node_name, state_in, node_output)

        prompt_delta = max(0, int((token_usage or {}).get("prompt_tokens", 0) - before_tokens.get("prompt_tokens", 0)))
        completion_delta = max(0, int((token_usage or {}).get("completion_tokens", 0) - before_tokens.get("completion_tokens", 0)))
        total_delta = max(0, int((token_usage or {}).get("total_tokens", 0) - before_tokens.get("total_tokens", 0)))
        cost_delta = max(0.0, float((token_usage or {}).get("total_cost", 0.0) - before_tokens.get("total_cost", 0.0)))

        execution_error = self._extract_execution_error_message(node_name, state_in, node_output)
        record_status = status
        if execution_error and not raised_exception:
            record_status = "error"

        facts = self._build_node_facts(
            node_name,
            state_in,
            node_output,
            file_bundle,
            record_status,
            error_text,
            "PENDING",
        )
        summary = self._summarize_step(node_name, "PENDING", facts)

        upstream_error = None
        if node_name == "evaluation_adaption" and isinstance(state_in, dict):
            prior_exec = state_in.get("evaluation_execution_result")
            if isinstance(prior_exec, str) and prior_exec.lower().startswith("error"):
                upstream_error = prior_exec

        logged_error = error_text or execution_error or upstream_error

        self._node_index += 1
        record = {
            "node_index": self._node_index,
            "current_node": node_name,
            "next_node": "__EXCEPTION__" if raised_exception else "PENDING",
            "output_summary": self._summary_output_value(summary),
            "prompt_tokens": prompt_delta,
            "completion_tokens": completion_delta,
            "total_tokens": total_delta,
            "estimated_cost_usd": round(cost_delta, 8),
            "duration_seconds": self._round_seconds((time.perf_counter_ns() - start_ns) / 1_000_000_000),
            "status": record_status,
            "error": self._truncate_summary(re.sub(r"\s+", " ", str(logged_error))) if logged_error else None,
        }
        self._activity_records.append(record)
        self._last_record_index = len(self._activity_records) - 1

        # Enrich top-level sections
        if node_name == "run_blocking_tester":
            cfg = node_output.get("blocking_config") if isinstance(node_output, dict) else None
            if cfg is None and isinstance(state_in, dict):
                cfg = state_in.get("blocking_config")
            self.set_run_config("blocking_config", cfg)

        if node_name == "run_matching_tester":
            cfg = node_output.get("matching_config") if isinstance(node_output, dict) else None
            if cfg is None and isinstance(state_in, dict):
                cfg = state_in.get("matching_config")
            self.set_run_config("matching_config", cfg)

        if node_name == "evaluation_decision":
            metrics = node_output.get("evaluation_metrics") if isinstance(node_output, dict) else None
            if isinstance(metrics, dict):
                self.append_evaluation_run(self._build_evaluation_run_record(metrics))
        self._maybe_attach_snapshot_accuracy(node_name, state_in, node_output)

        self.append_pipeline_snapshot(node_name, state_in, node_output)
        self._write_activity_payload()

        if raised_exception:
            raise RuntimeError(error_text)
        return result

    def _consume_token_deltas(self, token_usage: Dict[str, Any]) -> tuple[int, int, int, float]:
        current = token_usage if isinstance(token_usage, dict) else {}
        prompt_now = int(current.get("prompt_tokens", 0) or 0)
        completion_now = int(current.get("completion_tokens", 0) or 0)
        total_now = int(current.get("total_tokens", 0) or 0)
        cost_now = float(current.get("total_cost", 0.0) or 0.0)

        prompt_delta = max(0, prompt_now - int(self._last_token_snapshot.get("prompt_tokens", 0) or 0))
        completion_delta = max(0, completion_now - int(self._last_token_snapshot.get("completion_tokens", 0) or 0))
        total_delta = max(0, total_now - int(self._last_token_snapshot.get("total_tokens", 0) or 0))
        cost_delta = max(0.0, cost_now - float(self._last_token_snapshot.get("total_cost", 0.0) or 0.0))

        self._last_token_snapshot = {
            "prompt_tokens": prompt_now,
            "completion_tokens": completion_now,
            "total_tokens": total_now,
            "total_cost": cost_now,
        }
        return prompt_delta, completion_delta, total_delta, cost_delta

    def log_stream_update(self, node_name: str, state_in: Any, node_output: Any, token_usage: Dict[str, Any]):
        self.mark_next_for_previous(node_name)
        if isinstance(state_in, dict):
            self._latest_values_state = dict(state_in)
        if isinstance(node_output, dict):
            self._latest_values_state.update(node_output)
        self._update_density_gate(node_name, state_in, node_output)

        before_files = dict(self._tracked_file_cache.get(node_name, {}))
        after_files = self._capture_tracked_files(node_name)
        file_bundle = self._build_file_bundle(node_name, before_files, after_files)

        prompt_delta, completion_delta, total_delta, cost_delta = self._consume_token_deltas(token_usage)

        execution_error = self._extract_execution_error_message(node_name, state_in, node_output)
        record_status = "error" if execution_error else "ok"

        facts = self._build_node_facts(
            node_name,
            state_in,
            node_output,
            file_bundle,
            record_status,
            execution_error,
            "PENDING",
        )
        summary = self._summarize_step(node_name, "PENDING", facts)

        upstream_error = None
        if node_name == "evaluation_adaption" and isinstance(state_in, dict):
            prior_exec = state_in.get("evaluation_execution_result")
            if isinstance(prior_exec, str) and prior_exec.lower().startswith("error"):
                upstream_error = prior_exec

        duration_seconds = self._round_seconds(self._pop_node_duration(node_name))

        self._node_index += 1
        record = {
            "node_index": self._node_index,
            "current_node": node_name,
            "next_node": "PENDING",
            "output_summary": self._summary_output_value(summary),
            "prompt_tokens": prompt_delta,
            "completion_tokens": completion_delta,
            "total_tokens": total_delta,
            "estimated_cost_usd": round(cost_delta, 8),
            "duration_seconds": duration_seconds,
            "status": record_status,
            "error": self._truncate_summary(re.sub(r"\s+", " ", str(execution_error or upstream_error))) if (execution_error or upstream_error) else None,
        }
        self._activity_records.append(record)
        self._last_record_index = len(self._activity_records) - 1

        # Enrich top-level sections.
        if node_name == "run_blocking_tester":
            cfg = node_output.get("blocking_config") if isinstance(node_output, dict) else None
            if cfg is None and isinstance(state_in, dict):
                cfg = state_in.get("blocking_config")
            self.set_run_config("blocking_config", cfg)

        if node_name == "run_matching_tester":
            cfg = node_output.get("matching_config") if isinstance(node_output, dict) else None
            if cfg is None and isinstance(state_in, dict):
                cfg = state_in.get("matching_config")
            self.set_run_config("matching_config", cfg)

        if node_name == "evaluation_decision":
            metrics = node_output.get("evaluation_metrics") if isinstance(node_output, dict) else None
            if isinstance(metrics, dict):
                self.append_evaluation_run(self._build_evaluation_run_record(metrics))
        self._maybe_attach_snapshot_accuracy(node_name, state_in, node_output)

        self.append_pipeline_snapshot(node_name, state_in, node_output)
        self._write_activity_payload()


class _TokenTracker:
    def __init__(self, token_usage: Dict[str, Any]):
        self.token_usage = token_usage
        self._suppress_depth = 0

    @contextmanager
    def suppressed(self):
        self._suppress_depth += 1
        try:
            yield
        finally:
            self._suppress_depth = max(0, self._suppress_depth - 1)

    def add_from_response(self, response: Any):
        if self._suppress_depth > 0:
            return

        usage = None
        if hasattr(response, "usage_metadata") and isinstance(response.usage_metadata, dict):
            usage = response.usage_metadata
        elif hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
            usage = response.response_metadata.get("token_usage")

        if not isinstance(usage, dict):
            return

        prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)

        self.token_usage["prompt_tokens"] = int(self.token_usage.get("prompt_tokens", 0)) + prompt_tokens
        self.token_usage["completion_tokens"] = int(self.token_usage.get("completion_tokens", 0)) + completion_tokens
        self.token_usage["total_tokens"] = int(self.token_usage.get("total_tokens", 0)) + total_tokens
        self.token_usage["total_cost"] = float(self.token_usage.get("total_cost", 0.0))


class _InvokeTokenProxy:
    """Fallback wrapper for objects where `.invoke` cannot be monkey-patched (e.g., pydantic models)."""

    def __init__(self, target: Any, tracker: _TokenTracker):
        self._target = target
        self._tracker = tracker
        self.__workflow_token_proxy__ = True

    def invoke(self, *args, **kwargs):
        resp = self._target.invoke(*args, **kwargs)
        try:
            self._tracker.add_from_response(resp)
        except Exception:
            pass
        return resp

    async def ainvoke(self, *args, **kwargs):
        if not hasattr(self._target, "ainvoke"):
            raise AttributeError(f"{type(self._target).__name__} has no attribute 'ainvoke'")
        resp = await self._target.ainvoke(*args, **kwargs)
        try:
            self._tracker.add_from_response(resp)
        except Exception:
            pass
        return resp

    def __getattr__(self, item):
        return getattr(self._target, item)


def _maybe_wrap_invoke_for_tokens(obj: Any, tracker: _TokenTracker):
    if obj is None or not hasattr(obj, "invoke"):
        return obj

    original = getattr(obj, "invoke")
    if getattr(original, "__workflow_token_wrapped__", False):
        return obj
    if getattr(obj, "__workflow_token_proxy__", False):
        return obj

    def wrapped_invoke(*args, **kwargs):
        resp = original(*args, **kwargs)
        try:
            tracker.add_from_response(resp)
        except Exception:
            pass
        return resp

    wrapped_invoke.__workflow_token_wrapped__ = True
    try:
        setattr(obj, "invoke", wrapped_invoke)
    except Exception:
        # Some langchain objects (e.g., RunnableBinding) are pydantic models and disallow setattr.
        return _InvokeTokenProxy(obj, tracker)

    original_ainvoke = getattr(obj, "ainvoke", None)
    if callable(original_ainvoke) and not getattr(original_ainvoke, "__workflow_token_awrapped__", False):
        async def wrapped_ainvoke(*args, **kwargs):
            resp = await original_ainvoke(*args, **kwargs)
            try:
                tracker.add_from_response(resp)
            except Exception:
                pass
            return resp

        wrapped_ainvoke.__workflow_token_awrapped__ = True
        try:
            setattr(obj, "ainvoke", wrapped_ainvoke)
        except Exception:
            # If ainvoke cannot be patched but invoke already can, keep invoke patch.
            pass

    return obj


def _maybe_wrap_invoke_with_usage(agent: Any, tracker: _TokenTracker):
    original = getattr(agent, "_invoke_with_usage", None)
    if not callable(original):
        return
    if getattr(original, "__workflow_usage_guard_wrapped__", False):
        return

    def wrapped_invoke_with_usage(*args, **kwargs):
        # _invoke_with_usage already records tokens via callbacks;
        # suppress response-metadata tracking here to avoid double counting.
        with tracker.suppressed():
            return original(*args, **kwargs)

    wrapped_invoke_with_usage.__workflow_usage_guard_wrapped__ = True
    try:
        setattr(agent, "_invoke_with_usage", wrapped_invoke_with_usage)
    except Exception:
        pass


def _disable_existing_notebook_logger(agent: Any):
    # Avoid duplicate logging when notebook already includes internal wrap logger.
    for name in ["_append_activity_record", "_ensure_activity_log", "_write_activity_payload", "_write_transition_stats"]:
        if hasattr(agent, name):
            setattr(agent, name, lambda *args, **kwargs: None)
    if hasattr(agent, "_summary_model"):
        setattr(agent, "_summary_model", None)


def _attach_node_timing(graph: Any, logger: WorkflowLogger):
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict):
        return

    for node_name, pregel_node in nodes.items():
        if not node_name or str(node_name).startswith("__"):
            continue

        runnable = getattr(pregel_node, "node", None)
        if runnable is None:
            continue

        original_invoke = getattr(runnable, "invoke", None)
        if callable(original_invoke) and not getattr(original_invoke, "__workflow_timing_wrapped__", False):
            def wrapped_invoke(*args, __orig=original_invoke, __node_name=node_name, **kwargs):
                start_ns = time.perf_counter_ns()
                try:
                    return __orig(*args, **kwargs)
                finally:
                    duration_seconds = (time.perf_counter_ns() - start_ns) / 1_000_000_000
                    logger._push_node_duration(__node_name, duration_seconds)

            wrapped_invoke.__workflow_timing_wrapped__ = True
            setattr(runnable, "invoke", wrapped_invoke)

        original_ainvoke = getattr(runnable, "ainvoke", None)
        if callable(original_ainvoke) and not getattr(original_ainvoke, "__workflow_timing_awrapped__", False):
            async def wrapped_ainvoke(*args, __orig=original_ainvoke, __node_name=node_name, **kwargs):
                start_ns = time.perf_counter_ns()
                try:
                    return await __orig(*args, **kwargs)
                finally:
                    duration_seconds = (time.perf_counter_ns() - start_ns) / 1_000_000_000
                    logger._push_node_duration(__node_name, duration_seconds)

            wrapped_ainvoke.__workflow_timing_awrapped__ = True
            setattr(runnable, "ainvoke", wrapped_ainvoke)


def attach_logging(
    agent: Any,
    output_dir: str,
    summary_model_name: str = "gpt-4.1-mini",
    summary_char_limit: int = 300,
    notebook_name: Optional[str] = None,
    use_case: Optional[str] = None,
    llm_model: Optional[str] = None,
):
    if agent is None or not hasattr(agent, "graph"):
        raise ValueError("attach_logging expects an initialized agent with a compiled graph.")

    if not hasattr(agent, "token_usage") or not isinstance(getattr(agent, "token_usage"), dict):
        agent.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }

    def _resolve_llm_model() -> str:
        explicit = str(llm_model).strip() if llm_model is not None else ""
        if explicit:
            return explicit
        for obj_name in ("base_model", "model"):
            obj = getattr(agent, obj_name, None)
            if obj is None:
                continue
            for attr in ("model_name", "model"):
                value = getattr(obj, attr, None)
                text = str(value).strip() if value is not None else ""
                if text:
                    return text
        return "unknown"

    logger = WorkflowLogger(
        output_dir=output_dir,
        summary_model_name=summary_model_name,
        summary_char_limit=summary_char_limit,
        notebook_name=notebook_name,
        use_case=use_case,
        llm_model=_resolve_llm_model(),
    )
    tracker = _TokenTracker(agent.token_usage)

    # If notebook already has its own logger internals, neutralize duplicate writes.
    _disable_existing_notebook_logger(agent)

    # Always capture usage directly from model responses so direct invokes inside any node are tracked.
    wrapped_model = _maybe_wrap_invoke_for_tokens(getattr(agent, "model", None), tracker)
    wrapped_base_model = _maybe_wrap_invoke_for_tokens(getattr(agent, "base_model", None), tracker)
    if wrapped_model is not None:
        agent.model = wrapped_model
    if wrapped_base_model is not None:
        agent.base_model = wrapped_base_model

    # Guard callback-based token accounting paths to avoid double counting.
    _maybe_wrap_invoke_with_usage(agent, tracker)

    graph = agent.graph
    _attach_node_timing(graph, logger)
    original_graph_stream = getattr(graph, "stream")
    if not getattr(original_graph_stream, "__workflow_stream_wrapped__", False):
        def wrapped_graph_stream(input_data, *args, **kwargs):
            for chunk in original_graph_stream(input_data, *args, **kwargs):
                if logger.active:
                    try:
                        mode = None
                        payload = None
                        if isinstance(chunk, tuple):
                            if len(chunk) == 2:
                                mode, payload = chunk
                            elif len(chunk) == 3:
                                _, mode, payload = chunk

                        if mode == "values" and isinstance(payload, dict):
                            logger._latest_values_state = payload

                        if mode == "updates" and isinstance(payload, dict):
                            for node_name, node_output in payload.items():
                                state_snapshot = (
                                    dict(logger._latest_values_state)
                                    if isinstance(logger._latest_values_state, dict)
                                    else {}
                                )
                                if isinstance(node_output, dict):
                                    state_snapshot.update(node_output)
                                logger.log_stream_update(node_name, state_snapshot, node_output, agent.token_usage)
                    except Exception:
                        # Logging must never break the workflow execution path.
                        pass
                yield chunk

        wrapped_graph_stream.__workflow_stream_wrapped__ = True
        setattr(graph, "stream", wrapped_graph_stream)

    original_graph_invoke = getattr(graph, "invoke")
    if getattr(original_graph_invoke, "__workflow_invoke_wrapped__", False):
        return agent

    def wrapped_graph_invoke(input_data, *args, **kwargs):
        logger.start_run(input_data if isinstance(input_data, dict) else {}, token_usage=agent.token_usage)
        try:
            result = original_graph_invoke(input_data, *args, **kwargs)
            logger.finish_run("END")
            return result
        except Exception:
            logger.finish_run("__EXCEPTION__")
            raise

    wrapped_graph_invoke.__workflow_invoke_wrapped__ = True
    setattr(graph, "invoke", wrapped_graph_invoke)

    # Expose logger for debugging.
    agent.workflow_logger = logger
    return agent
