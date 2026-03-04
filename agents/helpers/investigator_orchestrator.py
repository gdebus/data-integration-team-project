import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from helpers.investigation_helpers import detect_normalization_issue as helper_detect_normalization_issue
from helpers.investigator_acceptance import (
    DEFAULT_MIN_DELTA,
    build_normalization_gate_request,
    evaluate_pending_normalization_acceptance,
)
from helpers.investigator_learning import (
    apply_learning_decay_for_drift,
    append_jsonl,
    dataset_signature_from_state,
    learning_routing_signals,
    load_learning_state,
    record_learning_checkpoint,
    save_learning_state,
    to_float,
    update_learning_from_observation,
)
from helpers.investigator_probe_runner import run_investigator_probes
from helpers.investigator_routing import ROUTING_BASE_THRESHOLD, compute_normalization_routing
from helpers.normalization_policy import (
    collect_eval_debug_signals,
    infer_country_output_format_from_validation,
    infer_validation_text_case_map,
)
try:
    from workflow_logging import log_agent_action
except Exception:
    try:
        from agents.workflow_logging import log_agent_action
    except Exception:
        from helpers.workflow_logging import log_agent_action

MAX_INVESTIGATION_ATTEMPTS = 3
INVESTIGATION_HISTORY_PATH = "output/profile/investigator_history.jsonl"


def _learning_context_key(state: Dict[str, Any]) -> str:
    metrics = state.get("evaluation_metrics", {}) if isinstance(state, dict) else {}
    metric_attrs = sorted(
        [
            str(k).replace("_accuracy", "")
            for k in (metrics.keys() if isinstance(metrics, dict) else [])
            if isinstance(k, str) and k.endswith("_accuracy") and k not in {"overall_accuracy", "macro_accuracy"}
        ]
    )
    eval_path = str(
        state.get("auto_diagnostics", {}).get("evaluation_testset_path", "")
        if isinstance(state.get("auto_diagnostics", {}), dict)
        else ""
    )
    reason_keys = sorted(
        list(
            (
                state.get("auto_diagnostics", {}).get("debug_reason_ratios", {})
                if isinstance(state.get("auto_diagnostics", {}), dict)
                else {}
            ).keys()
        )
    )
    return f"{eval_path}|attrs:{','.join(metric_attrs)}|reasons:{','.join([str(r) for r in reason_keys])}"


def _build_routing_objective(
    metrics: Dict[str, Any],
    auto_diagnostics: Dict[str, Any],
    probe_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    overall = max(0.0, min(1.0, to_float(metrics.get("overall_accuracy"), 0.0)))
    worst_acc = 1.0
    worst_count = 0
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if not (isinstance(k, str) and k.endswith("_accuracy")):
                continue
            if k in {"overall_accuracy", "macro_accuracy"}:
                continue
            attr = k[: -len("_accuracy")]
            count = int(to_float(metrics.get(f"{attr}_count"), 0.0))
            acc = max(0.0, min(1.0, to_float(v, 0.0)))
            if count <= 0:
                continue
            if acc < worst_acc:
                worst_acc = acc
                worst_count = count
    if worst_count <= 0:
        worst_acc = overall
    worst_penalty = 1.0 - worst_acc
    worst_weight = 1.0 if worst_count >= 10 else max(0.35, worst_count / 10.0)
    missing_ratio = max(
        0.0,
        min(
            1.0,
            to_float(
                (
                    auto_diagnostics.get("debug_reason_ratios", {}).get("missing_fused_value", 0.0)
                    if isinstance(auto_diagnostics, dict)
                    else 0.0
                ),
                0.0,
            ),
        ),
    )
    objective_penalty = (0.55 * worst_penalty * worst_weight) + (0.45 * missing_ratio)
    objective_penalty = min(1.0, objective_penalty)
    composite = max(0.0, min(1.0, overall - objective_penalty))
    return {
        "overall_accuracy": round(overall, 6),
        "worst_attribute_accuracy": round(worst_acc, 6),
        "worst_attribute_weight": round(worst_weight, 6),
        "worst_attribute_pressure": round(min(1.0, worst_penalty * worst_weight), 6),
        "missing_value_penalty": round(missing_ratio, 6),
        "probe_actionability": round(max(0.0, min(1.0, to_float(probe_outputs.get("actionability_pressure"), 0.0))), 6),
        "composite_score": round(composite, 6),
    }


def _normalize_string_list(values: Any, allowed: set[str] | None = None) -> List[str]:
    if not isinstance(values, list):
        return []
    kept: List[str] = []
    seen = set()
    for v in values:
        s = str(v).strip()
        if not s:
            continue
        s_l = s.lower()
        if allowed and s_l not in allowed:
            continue
        if s_l in seen:
            continue
        seen.add(s_l)
        kept.append(s)
    return kept


def _load_dataset_for_style(path: str):
    if not path or not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".csv":
            from PyDI.io import load_csv

            return load_csv(path)
        if ext == ".parquet":
            from PyDI.io import load_parquet

            return load_parquet(path)
        if ext == ".xml":
            from PyDI.io import load_xml

            return load_xml(path, nested_handling="aggregate")
    except Exception:
        return None
    return None


def _sanitize_directives_to_validation_style(
    directives: Dict[str, Any],
    state: Dict[str, Any],
    validation_style: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not isinstance(directives, dict):
        directives = {}
    out = dict(directives)

    report_validation_style = {}
    try:
        report_validation_style = (
            state.get("normalization_report", {}).get("validation_style", {})
            if isinstance(state.get("normalization_report", {}), dict)
            else {}
        )
    except Exception:
        report_validation_style = {}
    validation_style = (
        validation_style if isinstance(validation_style, dict) and validation_style else report_validation_style
    )

    allowed_list_cols = {
        str(c).lower()
        for c in (validation_style.get("validation_list_like_columns_hint", []) or [])
        if str(c).strip()
    }
    allowed_lower_cols = {
        str(c).lower()
        for c in (validation_style.get("lowercase_columns_hint", []) or [])
        if str(c).strip()
    }
    allowed_country_format = str(validation_style.get("inferred_country_output_format", "")).strip()

    proposed_list = _normalize_string_list(out.get("list_columns", []))
    for col in sorted(allowed_list_cols):
        if col not in {c.lower() for c in proposed_list}:
            proposed_list.append(col)
    out["list_columns"] = proposed_list
    out["lowercase_columns"] = _normalize_string_list(out.get("lowercase_columns", []), allowed_lower_cols)
    out["country_columns"] = _normalize_string_list(
        [c for c in out.get("country_columns", []) if "country" in str(c).lower()]
    )
    if allowed_country_format:
        out["country_output_format"] = allowed_country_format
    out["list_normalization_required"] = bool(out.get("list_columns", []))
    return out


def _build_investigator_action_plan(report: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    if not isinstance(report, dict):
        return []

    candidates: List[Dict[str, Any]] = []
    for source_name in ("fusion_policy_recommendations", "recommendations"):
        recs = report.get(source_name, [])
        if not isinstance(recs, list):
            continue
        for rec in recs:
            if isinstance(rec, str):
                text = rec.strip()
                if not text:
                    continue
                candidates.append(
                    {
                        "source": source_name,
                        "action": text,
                        "target_attributes": [],
                        "confidence": 0.45,
                        "priority_score": 0.45,
                    }
                )
                continue
            if not isinstance(rec, dict):
                continue
            action = (
                rec.get("action")
                or rec.get("recommendation")
                or rec.get("title")
                or rec.get("summary")
                or ""
            )
            action = str(action).strip()
            if not action:
                continue
            action_lower = action.lower()
            if "threshold" in action_lower and ("matching" in action_lower or "matcher" in action_lower):
                continue

            targets = rec.get("target_attributes", rec.get("columns", []))
            if not isinstance(targets, list):
                targets = [str(targets)] if str(targets).strip() else []
            targets = [str(t) for t in targets if str(t).strip()]

            confidence = to_float(rec.get("confidence"), 0.0)
            if confidence > 1.0:
                confidence = confidence / 100.0
            confidence = max(0.0, min(1.0, confidence))
            impact = to_float(
                rec.get("expected_impact_score", rec.get("expected_gain", rec.get("impact"))),
                0.0,
            )
            if impact > 1.0:
                impact = impact / 100.0
            impact = max(0.0, min(1.0, impact))
            if impact == 0.0:
                text_blob = " ".join(
                    [
                        str(rec.get("expected_impact", "")),
                        str(rec.get("evidence", "")),
                        str(rec.get("rationale", "")),
                    ]
                ).lower()
                if any(k in text_blob for k in ("high", "dominant", "catastrophic", "severe")):
                    impact = 0.7
                elif any(k in text_blob for k in ("medium", "moderate")):
                    impact = 0.45
                elif any(k in text_blob for k in ("low", "minor")):
                    impact = 0.2
            if confidence == 0.0:
                confidence = 0.55 if impact >= 0.5 else 0.4

            priority_score = (0.65 * impact) + (0.35 * confidence)
            candidates.append(
                {
                    "source": source_name,
                    "action": action,
                    "target_attributes": targets,
                    "evidence_summary": str(
                        rec.get("evidence_summary", rec.get("evidence", rec.get("rationale", "")))
                    ).strip(),
                    "expected_impact": str(
                        rec.get("expected_impact", rec.get("expected_gain", rec.get("impact", "")))
                    ).strip(),
                    "confidence": round(confidence, 3),
                    "priority_score": round(priority_score, 3),
                }
            )

    deduped: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        key = c["action"].strip().lower()
        prev = deduped.get(key)
        if prev is None or c.get("priority_score", 0.0) > prev.get("priority_score", 0.0):
            deduped[key] = c
    ranked = sorted(
        deduped.values(),
        key=lambda x: (x.get("priority_score", 0.0), x.get("confidence", 0.0)),
        reverse=True,
    )
    return ranked[:limit]


def _apply_acceptance_feedback(local_state: Dict[str, Any], acceptance_feedback: Dict[str, Any]) -> tuple[bool, bool]:
    force_skip_normalization = False
    rollback_applied = False
    if not isinstance(acceptance_feedback, dict):
        return force_skip_normalization, rollback_applied

    if acceptance_feedback.get("decision_ready"):
        local_state["normalization_pending_acceptance"] = {}
        if acceptance_feedback.get("status") == "rejected":
            rollback = list(acceptance_feedback.get("rollback_datasets", []) or [])
            if rollback:
                local_state["datasets"] = rollback
                local_state["normalized_datasets"] = []
                rollback_applied = True
            force_skip_normalization = True
    else:
        force_skip_normalization = True
    return force_skip_normalization, rollback_applied


def _resolve_validation_style_for_clamp(agent, state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        validation_path = (
            agent._evaluation_testset_path(state, force_test=False)
            if hasattr(agent, "_evaluation_testset_path")
            else ""
        )
        validation_df = _load_dataset_for_style(validation_path) if validation_path else None
        if validation_df is None or validation_df.empty:
            return {}

        case_map = infer_validation_text_case_map(validation_df)
        validation_list_like_columns: List[str] = []
        try:
            try:
                from list_normalization import detect_list_like_columns
            except ModuleNotFoundError:
                from agents.list_normalization import detect_list_like_columns
            validation_list_like_columns = [
                str(c)
                for c in detect_list_like_columns(
                    [validation_df],
                    exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id"},
                )
            ]
        except Exception:
            validation_list_like_columns = []
        return {
            "inferred_country_output_format": infer_country_output_format_from_validation(validation_df),
            "lowercase_columns_hint": [
                c for c, meta in case_map.items() if isinstance(meta, dict) and bool(meta.get("prefer_lowercase"))
            ],
            "validation_list_like_columns_hint": validation_list_like_columns,
        }
    except Exception:
        return {}


def _default_routing_decision(learning_signals: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "route_to_normalization": False,
        "score": 0.0,
        "threshold": ROUTING_BASE_THRESHOLD,
        "blocked_by_action_plan": False,
        "blocked_by_learning_regret": False,
        "components": {},
        "learning_signals": learning_signals if isinstance(learning_signals, dict) else {},
    }


def _history_failure_tags(acceptance_feedback: Dict[str, Any], routing_decision: Dict[str, Any]) -> List[str]:
    tags = [
        "normalization_regression"
        if isinstance(acceptance_feedback, dict) and acceptance_feedback.get("status") == "rejected"
        else "",
        "routing_blocked_by_regret" if routing_decision.get("blocked_by_learning_regret") else "",
        "routing_blocked_by_action_plan" if routing_decision.get("blocked_by_action_plan") else "",
    ]
    return [tag for tag in tags if tag]


def _print_investigation_report(
    *,
    overall_acc: float,
    attempts: int,
    max_attempts: int,
    probe_outputs: Dict[str, Any],
    routing_objective: Dict[str, Any],
    acceptance_feedback: Dict[str, Any],
    rollback_applied: bool,
    investigator_decision: str,
    routing_decision: Dict[str, Any],
    action_plan: List[Dict[str, Any]],
) -> None:
    print("[INVESTIGATION REPORT]")
    print(f"  - overall_accuracy={overall_acc:.3%}, evaluation_attempt={attempts}/{max_attempts}")
    print(f"  - probes: {probe_outputs.get('summary', 'none')}")
    print(
        f"  - objective: composite={routing_objective.get('composite_score')} "
        f"worst_pressure={routing_objective.get('worst_attribute_pressure')} "
        f"missing_penalty={routing_objective.get('missing_value_penalty')}"
    )
    if isinstance(acceptance_feedback, dict):
        print(
            f"  - normalization acceptance: {acceptance_feedback.get('status')} "
            f"(delta={acceptance_feedback.get('observed_delta')}, required={acceptance_feedback.get('required_delta')})"
        )
        if acceptance_feedback.get("decision_ready"):
            print(
                "    key_attrs_ok="
                f"{acceptance_feedback.get('key_attrs_ok')} "
                f"(mean={acceptance_feedback.get('focus_delta_mean')}, "
                f"min={acceptance_feedback.get('focus_delta_min')}, "
                f"required={acceptance_feedback.get('min_key_attr_delta')})"
            )
            print(
                "    heldout_proxy_ok="
                f"{acceptance_feedback.get('heldout_proxy_ok')} "
                f"(delta={acceptance_feedback.get('heldout_proxy_delta_mean')}, "
                f"max_drop={acceptance_feedback.get('heldout_max_drop')})"
            )
    if rollback_applied:
        print("  - rollback: applied rejected-normalization dataset rollback")
    if investigator_decision == "normalization_node":
        print(
            f"  - routing score={routing_decision.get('score')} threshold={routing_decision.get('threshold')} "
            f"blocked_action_plan={routing_decision.get('blocked_by_action_plan')} "
            f"blocked_regret={routing_decision.get('blocked_by_learning_regret')}"
        )
    elif investigator_decision == "pipeline_adaption" and action_plan:
        top = action_plan[0]
        print(
            f"  - top action [{top.get('priority_score', 0):.3f}]: {top.get('action')} "
            f"(targets: {', '.join(top.get('target_attributes', [])[:4]) or 'n/a'})"
        )


def run_investigator_node(agent, state: Dict[str, Any]) -> Dict[str, Any]:
    log_agent_action(
        agent,
        step="investigator_node",
        action="start",
        why="Run diagnostics and decide next step",
        improvement="Can route back to normalization",
    )
    print("[*] Running consolidated investigator node")

    local_state = dict(state)
    diagnostics_updates = agent.integration_diagnostics(local_state)
    local_state.update(diagnostics_updates)
    decision_updates = agent.evaluation_decision(local_state)
    local_state.update(decision_updates)

    metrics = local_state.get("evaluation_metrics", {}) if isinstance(local_state, dict) else {}
    overall_acc = to_float(metrics.get("overall_accuracy", 0.0), 0.0)
    attempts = int(local_state.get("evaluation_attempts", 0))
    max_attempts = MAX_INVESTIGATION_ATTEMPTS

    acceptance_feedback = evaluate_pending_normalization_acceptance(
        local_state.get("normalization_pending_acceptance", {}),
        current_accuracy=overall_acc,
        current_evaluation_attempt=attempts,
        current_total_evaluations=int(to_float(metrics.get("total_evaluations"), 0.0)),
        current_total_correct=int(to_float(metrics.get("total_correct"), 0.0)),
        current_metrics=metrics if isinstance(metrics, dict) else {},
    )
    force_skip_normalization, rollback_applied = _apply_acceptance_feedback(local_state, acceptance_feedback)

    learning_state = load_learning_state(local_state)
    learning_state = apply_learning_decay_for_drift(learning_state, _learning_context_key(local_state))
    dataset_signature = dataset_signature_from_state(local_state)
    learning_state, learning_observation = update_learning_from_observation(
        learning_state,
        current_accuracy=overall_acc,
        current_eval_attempt=attempts,
        dataset_signature=dataset_signature,
    )
    learning_signals = learning_routing_signals(learning_state)

    diagnostics_report = local_state.get("integration_diagnostics_report", {})
    action_plan = _build_investigator_action_plan(diagnostics_report if isinstance(diagnostics_report, dict) else {})
    probe_outputs = run_investigator_probes(
        state=local_state,
        action_plan=action_plan,
        load_dataset_fn=_load_dataset_for_style,
    )
    local_state["investigator_probe_results"] = probe_outputs
    routing_objective = _build_routing_objective(
        metrics=metrics if isinstance(metrics, dict) else {},
        auto_diagnostics=local_state.get("auto_diagnostics", {}) if isinstance(local_state, dict) else {},
        probe_outputs=probe_outputs if isinstance(probe_outputs, dict) else {},
    )

    investigator_decision = "human_review_export"
    reasoning_updates: Dict[str, Any] = {}
    cluster_updates: Dict[str, Any] = {}
    normalization_assessment: Dict[str, Any] = {}
    normalization_directives: Dict[str, Any] = (
        state.get("normalization_directives", {}) if isinstance(state, dict) else {}
    )
    if not isinstance(normalization_directives, dict):
        normalization_directives = {}
    normalization_reasons: List[str] = []
    normalization_needed = False
    normalization_rerun = False
    normalization_gate_request: Dict[str, Any] = {}
    routing_decision: Dict[str, Any] = _default_routing_decision(learning_signals)
    validation_style_for_clamp: Dict[str, Any] = _resolve_validation_style_for_clamp(agent, local_state)

    if overall_acc < 0.85 and attempts < max_attempts:
        cluster_updates = agent.cluster_analysis(local_state)
        local_state.update(cluster_updates)
        reasoning_updates = agent.evaluation_reasoning(local_state)
        local_state.update(reasoning_updates)

        normalization_assessment = agent._assess_normalization_with_llm(local_state)
        if isinstance(normalization_assessment, dict):
            normalization_directives = _sanitize_directives_to_validation_style(
                normalization_assessment,
                local_state,
                validation_style_for_clamp,
            )
            local_state["normalization_directives"] = normalization_directives

        llm_needs_normalization = bool(normalization_assessment.get("needs_normalization", False))
        llm_reasons = (
            [str(r) for r in normalization_assessment.get("reasons", []) if str(r).strip()]
            if isinstance(normalization_assessment, dict)
            else []
        )
        debug_signals = collect_eval_debug_signals()
        fallback_needed, fallback_reasons = helper_detect_normalization_issue(
            diagnostics_report=local_state.get("integration_diagnostics_report", {}),
            evaluation_analysis=local_state.get("evaluation_analysis", ""),
            metrics=metrics,
            debug_signals=debug_signals,
        )
        normalization_needed = llm_needs_normalization or fallback_needed or (
            to_float(probe_outputs.get("normalization_pressure"), 0.0) >= 0.5
        )
        normalization_reasons = llm_reasons if llm_reasons else fallback_reasons

        routing_decision = compute_normalization_routing(
            overall_acc=overall_acc,
            attempts=attempts,
            max_attempts=max_attempts,
            llm_assessment=normalization_assessment if isinstance(normalization_assessment, dict) else {},
            fallback_needed=fallback_needed,
            diagnostics=local_state.get("auto_diagnostics", {}) if isinstance(local_state, dict) else {},
            action_plan=action_plan,
            learning_signals=learning_signals,
            probe_results=probe_outputs,
            route_objective=routing_objective,
            force_skip_normalization=force_skip_normalization,
        )
        normalization_rerun = bool(routing_decision.get("route_to_normalization", False))

        if normalization_rerun and int(local_state.get("normalization_attempts", 0)) < 3:
            investigator_decision = "normalization_node"
            normalization_gate_request = build_normalization_gate_request(
                state=local_state,
                baseline_accuracy=overall_acc,
                min_delta=DEFAULT_MIN_DELTA,
            )
            print("[*] Investigator decision: rerun normalization before next pipeline iteration")
        else:
            investigator_decision = "pipeline_adaption"
            print("[*] Investigator decision: iterate pipeline adaptation")
    else:
        investigator_decision = "human_review_export"
        print("[*] Investigator decision: proceed to human review")

    if investigator_decision in {"normalization_node", "pipeline_adaption"}:
        learning_state = record_learning_checkpoint(
            learning_state,
            decision=investigator_decision,
            accuracy=overall_acc,
            evaluation_attempt=attempts,
            dataset_signature=dataset_signature,
        )
    else:
        learning_state["pending_observation"] = {}
    save_learning_state(local_state, learning_state)

    history_tags = _history_failure_tags(acceptance_feedback, routing_decision)
    history_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_signature": dataset_signature,
        "overall_accuracy": round(float(overall_acc), 6),
        "evaluation_attempt": attempts,
        "decision": investigator_decision,
        "normalization_needed": bool(normalization_needed),
        "normalization_rerun": bool(normalization_rerun),
        "routing_decision": routing_decision,
        "probe_summary": probe_outputs,
        "routing_objective": routing_objective,
        "acceptance_feedback": acceptance_feedback,
        "rollback_applied": rollback_applied,
        "learning_state": learning_state,
        "learning_observation": learning_observation,
        "failure_tags": history_tags,
    }
    append_jsonl(INVESTIGATION_HISTORY_PATH, history_payload)

    _print_investigation_report(
        overall_acc=overall_acc,
        attempts=attempts,
        max_attempts=max_attempts,
        probe_outputs=probe_outputs if isinstance(probe_outputs, dict) else {},
        routing_objective=routing_objective if isinstance(routing_objective, dict) else {},
        acceptance_feedback=acceptance_feedback if isinstance(acceptance_feedback, dict) else {},
        rollback_applied=rollback_applied,
        investigator_decision=investigator_decision,
        routing_decision=routing_decision if isinstance(routing_decision, dict) else {},
        action_plan=action_plan if isinstance(action_plan, list) else [],
    )

    out: Dict[str, Any] = {
        **diagnostics_updates,
        **decision_updates,
        **cluster_updates,
        **reasoning_updates,
        "investigator_decision": investigator_decision,
        "normalization_rework_required": normalization_needed,
        "normalization_rework_reasons": normalization_reasons,
        "normalization_directives": normalization_directives,
        "normalization_gate_request": normalization_gate_request,
        "normalization_pending_acceptance": local_state.get("normalization_pending_acceptance", {}),
        "normalization_acceptance_feedback": acceptance_feedback,
        "normalization_rollback_applied": rollback_applied,
        "investigator_action_plan": action_plan,
        "investigator_probe_results": probe_outputs,
        "investigator_routing_objective": routing_objective,
        "investigator_learning_state": learning_state,
        "investigator_learning_observation": learning_observation,
        "investigator_routing_decision": routing_decision,
    }
    if rollback_applied:
        out["datasets"] = local_state.get("datasets", [])
        out["normalized_datasets"] = []
    return out
