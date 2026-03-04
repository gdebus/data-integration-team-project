import ast
import difflib
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage


# Compatibility helpers used by current agent/orchestrators.
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
    "run_blocking_tester": "_extract_run_blocking_tester_facts",
    "run_matching_tester": "_extract_run_matching_tester_facts",
    "pipeline_adaption": "_extract_pipeline_adaption_facts",
    "execute_pipeline": "_extract_execute_pipeline_facts",
    "evaluation_adaption": "_extract_evaluation_adaption_facts",
    "execute_evaluation": "_extract_execute_evaluation_facts",
    "evaluation_decision": "_extract_evaluation_decision_facts",
    "evaluation_reasoning": "_extract_evaluation_reasoning_facts",
    "normalization_node": "_extract_generic_known_node_facts",
    "evaluation_node": "_extract_generic_known_node_facts",
    "investigator_node": "_extract_generic_known_node_facts",
    "human_review_export": "_extract_generic_known_node_facts",
    "sealed_final_test_evaluation": "_extract_generic_known_node_facts",
    "save_results": "_extract_generic_known_node_facts",
}


class WorkflowLogger:
    def __init__(
        self,
        output_dir: str,
        summary_model_name: str = "gpt-4.1-mini",
        summary_char_limit: int = 300,
        notebook_name: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.summary_char_limit = summary_char_limit
        self.summary_model_name = summary_model_name
        self.notebook_name = notebook_name or "AdaptationPipeline"

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
        self._archive_matcher_mode = "rule_based"
        self._run_started_at_ns: Optional[int] = None
        self._run_finished_at_ns: Optional[int] = None
        self._pending_node_durations_seconds: Dict[str, List[float]] = {}

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

    def _attach_accuracy_to_pending_snapshot(self, accuracy_score: Optional[str]):
        if not self._pending_snapshot_indices:
            return
        idx = self._pending_snapshot_indices.pop(0)
        if idx < 0 or idx >= len(self._pipeline_snapshots):
            return
        self._pipeline_snapshots[idx]["accuracy_score"] = accuracy_score or "pending"
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
        self._archive_matcher_mode = safe_mode
        self._run_started_at_ns = time.perf_counter_ns()
        self._run_finished_at_ns = None
        self._pending_node_durations_seconds = {}
        self._active = True

        self._write_activity_payload()
        self._write_pipeline_archive_markdown()

    def finish_run(self, final_next_node: str = "END"):
        if self._last_record_index is not None:
            self._activity_records[self._last_record_index]["next_node"] = final_next_node
        self._run_finished_at_ns = time.perf_counter_ns()
        self._write_activity_payload()
        self._active = False

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
        payload = {"node_activity": self._activity_records}
        if self._run_configs:
            payload["run_configs"] = self._run_configs
        if self._evaluation_runs:
            payload["evaluation_runs"] = self._evaluation_runs
        payload["transition_stats"] = self._build_transition_stats()
        payload["time_complexity"] = self._build_time_complexity()
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
            "profile_data": 500,
            "match_schemas": 500,
            "evaluation_reasoning": 1250,
            "evaluation_decision": 350,
            "execute_pipeline": 350,
            "execute_evaluation": 300,
            "normalization_node": 500,
            "evaluation_node": 500,
            "investigator_node": 500,
            "human_review_export": 500,
            "sealed_final_test_evaluation": 500,
            "save_results": 500,
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
        clipped = text[: max(1, limit - 1)].rstrip()
        word_boundary = clipped.rfind(" ")
        if word_boundary >= 40:
            clipped = clipped[:word_boundary].rstrip()
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
        human_prompt = (
            f"NODE\n{node_name}\n\n"
            f"SCHEMA\n{json.dumps(expected, ensure_ascii=False, indent=2)}\n\n"
            f"EVIDENCE\n{json.dumps(evidence, ensure_ascii=False, indent=2, default=str)}\n"
        )
        try:
            response = self._summary_model.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
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
            response = self._summary_model.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
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
        text = str(value or "").strip()
        return text[:240]

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

    # Detailed extractors preserved from your logging design for core nodes.
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
            summary_clauses.append(f"pipeline execution failed on attempt {attempts}: {self._extract_error_excerpt(execution_result)}")
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
            summary_clauses.append(f"evaluation execution failed on attempt {attempts}: {self._extract_error_excerpt(execution_result)}")
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

    def _extract_generic_known_node_facts(
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
        changed_keys = sorted(output.keys())
        summary_clauses = []
        if changed_keys:
            summary_clauses.append(f"updated keys: {self._compact_list(changed_keys, max_items=8)}")
        if state.get("evaluation_metrics") and isinstance(state.get("evaluation_metrics"), dict):
            acc = self._format_accuracy(state.get("evaluation_metrics"))
            if acc:
                summary_clauses.append(f"current overall accuracy context is {acc}")
        if not summary_clauses:
            summary_clauses.append("state updated for next workflow step")
        return {
            "registered": True,
            "node_name": str(output.get("node_name") or "generic_node"),
            "next_node": next_node,
            "status": status,
            "error": error_text or "",
            "summary_clauses": summary_clauses,
            "effect_clause": f"next node is `{next_node}`" if next_node and next_node != "PENDING" else "",
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

    def _build_evaluation_run_record(self, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pipeline_code = self._read_file_if_exists(os.path.join(self.output_dir, "code", "pipeline.py"))
        evaluation_code = self._read_file_if_exists(os.path.join(self.output_dir, "code", "evaluation.py"))

        acc = self._format_accuracy(metrics)
        if acc is None:
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
            pass

    def log_node(
        self,
        node_name: str,
        state_in: Any,
        call_fn: Callable[[], Any],
        token_usage: Dict[str, Any],
    ) -> Any:
        self.mark_next_for_previous(node_name)

        before_tokens = dict(token_usage or {})
        start_ns = time.perf_counter_ns()
        before_files = self._capture_tracked_files(node_name)

        node_output = {}
        status = "ok"
        error_text = None

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

        after_files = self._capture_tracked_files(node_name)
        file_bundle = self._build_file_bundle(node_name, before_files, after_files)

        prompt_delta = max(0, int((token_usage or {}).get("prompt_tokens", 0) - before_tokens.get("prompt_tokens", 0)))
        completion_delta = max(0, int((token_usage or {}).get("completion_tokens", 0) - before_tokens.get("completion_tokens", 0)))
        total_delta = max(0, int((token_usage or {}).get("total_tokens", 0) - before_tokens.get("total_tokens", 0)))
        cost_delta = max(0.0, float((token_usage or {}).get("total_cost", 0.0) - before_tokens.get("total_cost", 0.0)))

        facts = self._build_node_facts(node_name, state_in, node_output, file_bundle, status, error_text, "PENDING")
        summary = self._summarize_step(node_name, "PENDING", facts)

        upstream_error = None
        if node_name == "evaluation_adaption" and isinstance(state_in, dict):
            prior_exec = state_in.get("evaluation_execution_result")
            if isinstance(prior_exec, str) and prior_exec.lower().startswith("error"):
                upstream_error = prior_exec

        logged_error = error_text or upstream_error

        self._node_index += 1
        record = {
            "node_index": self._node_index,
            "current_node": node_name,
            "next_node": "PENDING" if status == "ok" else "__EXCEPTION__",
            "output_summary": self._summary_output_value(summary),
            "prompt_tokens": prompt_delta,
            "completion_tokens": completion_delta,
            "total_tokens": total_delta,
            "estimated_cost_usd": round(cost_delta, 8),
            "duration_seconds": self._round_seconds((time.perf_counter_ns() - start_ns) / 1_000_000_000),
            "status": status,
            "error": self._truncate_summary(logged_error) if logged_error else None,
        }
        self._activity_records.append(record)
        self._last_record_index = len(self._activity_records) - 1 if status == "ok" else None

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
                self._attach_accuracy_to_pending_snapshot(self._format_accuracy(metrics))

        self.append_pipeline_snapshot(node_name, state_in, node_output)
        self._write_activity_payload()

        if status == "error":
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

        before_files = dict(self._tracked_file_cache.get(node_name, {}))
        after_files = self._capture_tracked_files(node_name)
        file_bundle = self._build_file_bundle(node_name, before_files, after_files)

        prompt_delta, completion_delta, total_delta, cost_delta = self._consume_token_deltas(token_usage)

        facts = self._build_node_facts(node_name, state_in, node_output, file_bundle, "ok", None, "PENDING")
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
            "status": "ok",
            "error": self._truncate_summary(upstream_error) if upstream_error else None,
        }
        self._activity_records.append(record)
        self._last_record_index = len(self._activity_records) - 1

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
                self._attach_accuracy_to_pending_snapshot(self._format_accuracy(metrics))

        self.append_pipeline_snapshot(node_name, state_in, node_output)
        self._write_activity_payload()


class _TokenTracker:
    def __init__(self, token_usage: Dict[str, Any]):
        self.token_usage = token_usage

    def add_from_response(self, response: Any):
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
        return obj
    except Exception:
        return _InvokeTokenProxy(obj, tracker)


def _disable_existing_notebook_logger(agent: Any):
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

    logger = WorkflowLogger(
        output_dir=output_dir,
        summary_model_name=summary_model_name,
        summary_char_limit=summary_char_limit,
        notebook_name=notebook_name,
    )
    tracker = _TokenTracker(agent.token_usage)

    _disable_existing_notebook_logger(agent)

    if not hasattr(agent, "_invoke_with_usage"):
        wrapped_model = _maybe_wrap_invoke_for_tokens(getattr(agent, "model", None), tracker)
        wrapped_base_model = _maybe_wrap_invoke_for_tokens(getattr(agent, "base_model", None), tracker)
        if wrapped_model is not None:
            agent.model = wrapped_model
        if wrapped_base_model is not None:
            agent.base_model = wrapped_base_model

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

    agent.workflow_logger = logger
    return agent
