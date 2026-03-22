"""Run results extraction, audit files, and markdown report generation."""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

import config as _config_module
from langchain_core.messages import AIMessage, ToolMessage


# ── report-text helpers ──────────────────────────────────────────────

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


def _brief_from_parsed_json(parsed: dict) -> Dict[str, Any]:
    """Extract a reasoning brief from a parsed JSON response."""
    brief: Dict[str, Any] = {
        "problem": str(parsed.get("what_went_wrong", "")),
        "next_step": str(parsed.get("next_strategy", "")),
        "normalization": str(parsed.get("normalization_recommendations", "") or ""),
        "takeaway": str(parsed.get("takeaway", "")),
    }
    # Preserve structured fields from the enhanced reasoning prompt
    if parsed.get("focus_attributes"):
        brief["focus_attributes"] = parsed["focus_attributes"]
    if parsed.get("confidence") is not None:
        brief["confidence"] = parsed["confidence"]
    if parsed.get("root_cause"):
        brief["root_cause"] = parsed["root_cause"]
    if parsed.get("root_cause_evidence"):
        brief["root_cause_evidence"] = parsed["root_cause_evidence"]
    if parsed.get("per_attribute_actions"):
        brief["per_attribute_actions"] = parsed["per_attribute_actions"]
    if parsed.get("post_clustering_recommendation"):
        brief["post_clustering_recommendation"] = parsed["post_clustering_recommendation"]
    if parsed.get("avoid_actions"):
        brief["avoid_actions"] = parsed["avoid_actions"]
    return brief


def _build_reasoning_brief(analysis: str) -> Dict[str, Any]:
    """Builds a reasoning brief from the evaluation analysis output.

    Handles both structured JSON output (preferred) and free-text fallback.
    """
    cleaned = _clean_report_text(analysis)
    if not cleaned:
        return {}

    # Try JSON parse first (structured output)
    for text in (cleaned, re.sub(r'^```(?:json)?\s*\n?', '', re.sub(r'\n?```\s*$', '', cleaned))):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return _brief_from_parsed_json(parsed)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # Fallback: free-text regex extraction
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


# ── message extraction ───────────────────────────────────────────────

def _extract_query_response_pairs(messages: list) -> list:
    pairs: list = []
    tool_call_map: dict = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                call_id = tool_call.get("id")
                query = None
                if "args" in tool_call:
                    query = tool_call["args"].get("query")
                elif "function" in tool_call:
                    args_str = tool_call["function"].get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                        query = args.get("query")
                    except json.JSONDecodeError:
                        query = None
                if call_id and query:
                    tool_call_map[call_id] = query

    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_call_id = msg.tool_call_id
            if tool_call_id in tool_call_map:
                pairs.append({
                    "query": tool_call_map[tool_call_id],
                    "response": msg.content,
                })
    return pairs


# ── public API ───────────────────────────────────────────────────────

def save_results(
    state: Dict[str, Any],
    *,
    run_id: str,
    run_output_root: str,
    token_usage: Dict[str, Any],
    is_metrics_payload_fn,
    sealed_eval_active: bool,
    logger,
    log_action_fn,
) -> Dict[str, Any]:
    """Persists run artifacts: JSON results, audit file, and markdown report.

    Parameters
    ----------
    state : agent state dict
    run_id, run_output_root : identity of this run
    token_usage : copy of token tracker usage dict
    is_metrics_payload_fn : callable(payload) -> bool
    sealed_eval_active : whether sealed evaluation is active
    logger : standard logger
    log_action_fn : callable(action, phase, detail, reason)
    """
    log_action_fn("save_results", "start", "Persist run artifacts", "Enables reproducibility")
    logger.info("----------------------- Entering save_results -----------------------")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    results_dir = os.path.join(_config_module.OUTPUT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    dataset_names = [os.path.splitext(os.path.basename(path))[0] for path in state.get("datasets", [])]

    # Blocking metrics extraction
    blocking_metrics: Dict[str, Any] = {}
    blocking_strategies = state.get("blocking_config", {}).get("blocking_strategies", {})
    for pair_key, blk_cfg in blocking_strategies.items():
        if isinstance(blk_cfg, dict):
            blocking_metrics[pair_key] = {
                "pair_completeness": blk_cfg.get("pair_completeness", 0),
                "num_candidates": blk_cfg.get("num_candidates", 0),
                "stategy": blk_cfg.get("strategy", "unknown"),
                "columns": blk_cfg.get("columns", []),
            }

    # Matching metrics extraction
    matching_metrics: Dict[str, Any] = {}
    matching_strategies = state.get("matching_config", {}).get("matching_strategies", {})
    for pair_key, mtch_cfg in matching_strategies.items():
        if isinstance(mtch_cfg, dict):
            matching_metrics[pair_key] = {"f1_score": mtch_cfg.get("f1", 0)}

    # Evaluation metrics and report assembly
    evaluation_metrics = state.get("evaluation_metrics", {})
    best_validation_metrics = state.get("best_validation_metrics", {})
    validation_metrics_final = (
        best_validation_metrics
        if is_metrics_payload_fn(best_validation_metrics)
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

    pipeline_info = {
        "evaluation_attempts": state.get("evaluation_attempts", 0),
        "matcher_mode": state.get("matcher_mode", "unknown"),
    }

    # Data profiles summary extraction
    data_profiles_summary: Dict[str, Any] = {}
    data_profiles = state.get("data_profiles", {})
    for path, profile in data_profiles.items():
        if isinstance(profile, dict):
            ds_name = os.path.splitext(os.path.basename(path))[0]
            data_profiles_summary[ds_name] = {
                "num_rows": profile.get("num_rows", 0),
                "num_columns": profile.get("num_columns", 0),
                "columns": list(profile.get("dtypes", {}).keys())[:10],
            }

    results = {
        "timestamp": timestamp,
        "generated_at": generated_at,
        "run_id": run_id,
        "run_output_root": run_output_root,
        "datasets": dataset_names,
        "matcher_mode": state.get("matcher_mode", "unknown"),
        "fusion_testset": state.get("fusion_testset", "").split("/")[-1],
        "validation_fusion_testset": (
            state.get("validation_fusion_testset", "").split("/")[-1]
            if state.get("validation_fusion_testset")
            else ""
        ),
        "sealed_evaluation_active": sealed_eval_active,
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
        "pipeline_generation_review": state.get("pipeline_generation_review", {}),
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
        "per_run_output_dir": _config_module.OUTPUT_DIR,
        "normalized_datasets": state.get("normalized_datasets", []),
        "integration_diagnostics_execution_result": state.get("integration_diagnostics_execution_result", ""),
        "integration_diagnostics_report": state.get("integration_diagnostics_report", {}),
        "human_review_execution_result": state.get("human_review_execution_result", ""),
        "human_review_report": state.get("human_review_report", {}),
        "final_test_evaluation_execution_result": state.get("final_test_evaluation_execution_result", ""),
        "final_test_evaluation_metrics": state.get("final_test_evaluation_metrics", {}),
        "agent_run_summary": agent_run_summary,
        "token_usage": token_usage,
        "pipeline_info": pipeline_info,
        "pipeline_search_tool_responses": _extract_query_response_pairs(state.get("messages", [])[2:-1]),
        "evaluation_search_tool_responses": _extract_query_response_pairs((state.get("eval_messages") or [])[2:-1]),
        "integration_pipeline_code_length": len(state.get("integration_pipeline_code", "")),
        "evaluation_code_length": len(state.get("evaluation_code", "")),
        "integration_diagnostics_code_length": len(state.get("integration_diagnostics_code", "")),
        "human_review_code_length": len(state.get("human_review_code", "")),
    }

    results_file = os.path.join(results_dir, f"run_{timestamp}.json")
    run_audit_file = os.path.join(results_dir, f"run_audit_{timestamp}.json")
    run_report_file = os.path.join(results_dir, f"run_report_{timestamp}.md")
    results["run_audit_path"] = run_audit_file
    results["run_report_path"] = run_report_file

    run_audit = {
        "timestamp": timestamp,
        "generated_at": generated_at,
        "run_id": run_id,
        "run_output_root": run_output_root,
        "validation_metrics_final": validation_metrics_final,
        "sealed_test_metrics_final": sealed_test_metrics_final,
        "correspondence_integrity": state.get("correspondence_integrity", {}),
        "cycles": cycle_audit,
        "pipeline_snapshots": state.get("pipeline_snapshots", []),
        "evaluation_snapshots": state.get("evaluation_snapshots", []),
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    with open(run_audit_file, "w", encoding="utf-8") as f:
        json.dump(run_audit, f, indent=2, ensure_ascii=False, default=str)

    # Markdown report generation
    report_lines: List[str] = []
    report_lines.append(f"# Run Report ({timestamp})")
    report_lines.append("")
    report_lines.append(f"- Generated at (UTC): `{generated_at}`")
    report_lines.append(f"- Run ID: `{run_id}`")
    report_lines.append(f"- Run output root: `{run_output_root}`")
    report_lines.append(f"- Validation overall (final): `{validation_metrics_final.get('overall_accuracy', 'n/a')}`")
    report_lines.append(f"- Validation macro (final): `{validation_metrics_final.get('macro_accuracy', 'n/a')}`")
    report_lines.append(f"- Sealed test overall (final): `{sealed_test_metrics_final.get('overall_accuracy', 'n/a')}`")
    report_lines.append(f"- Sealed test macro (final): `{sealed_test_metrics_final.get('macro_accuracy', 'n/a')}`")
    report_lines.append(
        f"- Correspondence structurally valid: "
        f"`{state.get('correspondence_integrity', {}).get('structurally_valid', 'n/a')}`"
    )
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

    logger.info(f"Results saved to {results_file}")
    logger.info(f"Run audit saved to {run_audit_file}")
    logger.info(f"Run report saved to {run_report_file}")
    logger.info("Leaving save_results")

    return {
        "run_audit_path": run_audit_file,
        "run_report_path": run_report_file,
        "validation_metrics_final": validation_metrics_final,
        "sealed_test_metrics_final": sealed_test_metrics_final,
        "run_id": run_id,
        "run_output_root": run_output_root,
    }
