import ast
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage


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
        self._active = True

        self._write_activity_payload()
        self._write_pipeline_archive_markdown()

    def finish_run(self, final_next_node: str = "END"):
        if self._last_record_index is not None:
            self._activity_records[self._last_record_index]["next_node"] = final_next_node
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

    def _truncate_summary(self, text: str) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        limit = self.summary_char_limit
        if len(text) <= limit:
            return text

        clipped = text[:limit].rstrip()
        boundary = max(
            clipped.rfind(". "),
            clipped.rfind("! "),
            clipped.rfind("? "),
            clipped.rfind("."),
            clipped.rfind("!"),
            clipped.rfind("?"),
        )
        if boundary >= 40:
            sentence = clipped[: boundary + 1].strip()
            if sentence:
                return sentence

        word_boundary = clipped.rfind(" ")
        if word_boundary >= 40:
            clipped = clipped[:word_boundary].rstrip()

        clipped = clipped[: max(1, limit - 1)].rstrip()
        if clipped and clipped[-1] not in ".!?":
            clipped += "."
        return clipped

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
    def _diff_excerpt(before_text: str, after_text: str, max_lines: int = 30, max_chars: int = 1200) -> str:
        if before_text == after_text:
            return ""
        import difflib

        before_lines = (before_text or "").splitlines()
        after_lines = (after_text or "").splitlines()
        diff_lines = list(
            difflib.unified_diff(before_lines, after_lines, fromfile="before", tofile="after", lineterm="")
        )
        if not diff_lines:
            return ""
        excerpt = "\n".join(diff_lines[:max_lines])
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars]
        return excerpt

    def _capture_tracked_files(self, node_name: str) -> Dict[str, str]:
        tracked = {}
        if node_name == "pipeline_adaption":
            tracked["pipeline.py"] = self._read_file_if_exists(os.path.join(self.output_dir, "code", "pipeline.py"))
        elif node_name == "evaluation_adaption":
            tracked["evaluation.py"] = self._read_file_if_exists(os.path.join(self.output_dir, "code", "evaluation.py"))
        return tracked

    def _compute_file_changes(self, node_name: str, after_map: Dict[str, str]) -> Dict[str, str]:
        before_map = self._tracked_file_cache.get(node_name, {})
        changes = {}
        keys = set(before_map.keys()) | set(after_map.keys())
        for key in keys:
            diff = self._diff_excerpt(before_map.get(key, ""), after_map.get(key, ""))
            if diff:
                changes[key] = diff
        self._tracked_file_cache[node_name] = after_map
        return changes

    def _build_summary_context(
        self,
        node_name: str,
        state_in: Any,
        node_output: Any,
        file_changes: Dict[str, str],
        status: str,
        error_text: Optional[str],
    ) -> Dict[str, Any]:
        if isinstance(node_output, dict):
            changed_keys = list(node_output.keys())
            try:
                output_text = json.dumps(node_output, ensure_ascii=False, default=str)
            except Exception:
                output_text = str(node_output)
        else:
            changed_keys = []
            output_text = str(node_output)

        state = state_in if isinstance(state_in, dict) else {}
        upstream_error = None
        prior_exec = state.get("evaluation_execution_result")
        if isinstance(prior_exec, str) and prior_exec.lower().startswith("error"):
            upstream_error = prior_exec

        attempts = {
            "pipeline_execution_attempts": state.get("pipeline_execution_attempts"),
            "evaluation_execution_attempts": state.get("evaluation_execution_attempts"),
            "evaluation_attempts": state.get("evaluation_attempts"),
        }

        accuracy_current = None
        delta_accuracy = None
        if isinstance(node_output, dict):
            metrics = node_output.get("evaluation_metrics")
            if isinstance(metrics, dict):
                accuracy_current = metrics.get("overall_accuracy")

        if isinstance(accuracy_current, (int, float)) and self._evaluation_runs:
            prev_raw = self._evaluation_runs[-1].get("accuracy_score")
            prev = None
            if isinstance(prev_raw, str) and prev_raw.endswith("%"):
                try:
                    prev = float(prev_raw[:-1]) / 100.0
                except Exception:
                    prev = None
            elif isinstance(prev_raw, (int, float)):
                prev = float(prev_raw)
                if prev > 1.0:
                    prev = prev / 100.0
            if isinstance(prev, (int, float)):
                delta_accuracy = float(accuracy_current) - float(prev)

        return {
            "node_name": node_name,
            "changed_keys": changed_keys,
            "node_output_excerpt": output_text[:1400],
            "file_changes": file_changes,
            "status": status,
            "error": (error_text or "")[:600],
            "upstream_error_excerpt": (upstream_error or "")[:600],
            "attempt_counters": attempts,
            "overall_accuracy": accuracy_current,
            "delta_accuracy": delta_accuracy,
        }

    def _summarize_step(self, current_node: str, next_node: str, summary_context: Dict[str, Any]) -> str:
        fallback = f"{current_node} changed this iteration and moved flow to {next_node}."

        if self._summary_model is None:
            return self._truncate_summary(fallback)

        system_prompt = (
            "Write one concise engineering log line for this iteration. "
            "Describe only what changed in this step and its direct effect on pipeline/evaluation progress. "
            "Do not describe generic node purpose. Use plain text and complete sentences. "
            f"Maximum {self.summary_char_limit} characters."
        )
        human_prompt = (
            f"current_node: {current_node}\n"
            f"next_node: {next_node}\n"
            f"context: {json.dumps(summary_context, ensure_ascii=False)}"
        )

        try:
            response = self._summary_model.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
            text = self._coerce_response_text(response.content if hasattr(response, "content") else response)
            if not text.strip():
                text = fallback
            return self._truncate_summary(text)
        except Exception:
            return self._truncate_summary(fallback)

    @staticmethod
    def _normalize_expr_text(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

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

        before_tokens = dict(token_usage or {})
        start = time.time()
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
        file_changes = {}
        keys = set(before_files.keys()) | set(after_files.keys())
        for key in keys:
            diff = self._diff_excerpt(before_files.get(key, ""), after_files.get(key, ""))
            if diff:
                file_changes[key] = diff

        prompt_delta = max(0, int((token_usage or {}).get("prompt_tokens", 0) - before_tokens.get("prompt_tokens", 0)))
        completion_delta = max(0, int((token_usage or {}).get("completion_tokens", 0) - before_tokens.get("completion_tokens", 0)))
        total_delta = max(0, int((token_usage or {}).get("total_tokens", 0) - before_tokens.get("total_tokens", 0)))
        cost_delta = max(0.0, float((token_usage or {}).get("total_cost", 0.0) - before_tokens.get("total_cost", 0.0)))

        summary_context = self._build_summary_context(node_name, state_in, node_output, file_changes, status, error_text)
        summary = self._summarize_step(node_name, "PENDING", summary_context)

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
            "output_summary": summary,
            "prompt_tokens": prompt_delta,
            "completion_tokens": completion_delta,
            "total_tokens": total_delta,
            "estimated_cost_usd": round(cost_delta, 8),
            "duration_ms": int((time.time() - start) * 1000),
            "status": status,
            "error": self._truncate_summary(logged_error) if logged_error else None,
        }
        self._activity_records.append(record)
        self._last_record_index = len(self._activity_records) - 1 if status == "ok" else None

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

        after_files = self._capture_tracked_files(node_name)
        file_changes = self._compute_file_changes(node_name, after_files)

        prompt_delta, completion_delta, total_delta, cost_delta = self._consume_token_deltas(token_usage)

        summary_context = self._build_summary_context(node_name, state_in, node_output, file_changes, "ok", None)
        summary = self._summarize_step(node_name, "PENDING", summary_context)

        upstream_error = None
        if node_name == "evaluation_adaption" and isinstance(state_in, dict):
            prior_exec = state_in.get("evaluation_execution_result")
            if isinstance(prior_exec, str) and prior_exec.lower().startswith("error"):
                upstream_error = prior_exec

        self._node_index += 1
        record = {
            "node_index": self._node_index,
            "current_node": node_name,
            "next_node": "PENDING",
            "output_summary": summary,
            "prompt_tokens": prompt_delta,
            "completion_tokens": completion_delta,
            "total_tokens": total_delta,
            "estimated_cost_usd": round(cost_delta, 8),
            "duration_ms": 0,
            "status": "ok",
            "error": self._truncate_summary(upstream_error) if upstream_error else None,
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
        # Some langchain objects (e.g., RunnableBinding) are pydantic models and disallow setattr.
        return _InvokeTokenProxy(obj, tracker)


def _disable_existing_notebook_logger(agent: Any):
    # Avoid duplicate logging when notebook already includes internal wrap logger.
    for name in ["_append_activity_record", "_ensure_activity_log", "_write_activity_payload", "_write_transition_stats"]:
        if hasattr(agent, name):
            setattr(agent, name, lambda *args, **kwargs: None)
    if hasattr(agent, "_summary_model"):
        setattr(agent, "_summary_model", None)


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

    # If notebook already has its own logger internals, neutralize duplicate writes.
    _disable_existing_notebook_logger(agent)

    # For notebooks without _invoke_with_usage, capture usage directly from model responses.
    if not hasattr(agent, "_invoke_with_usage"):
        wrapped_model = _maybe_wrap_invoke_for_tokens(getattr(agent, "model", None), tracker)
        wrapped_base_model = _maybe_wrap_invoke_for_tokens(getattr(agent, "base_model", None), tracker)
        if wrapped_model is not None:
            agent.model = wrapped_model
        if wrapped_base_model is not None:
            agent.base_model = wrapped_base_model

    graph = agent.graph
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
