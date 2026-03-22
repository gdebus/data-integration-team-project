"""Investigator orchestrator — coordinates evidence gathering and LLM-driven
investigation to decide which pipeline stage to fix next.

Flow:
  1. evaluation_decision  (metric bookkeeping, regression guards)
  2. acceptance feedback   (normalization rollback if rejected)
  3. learning state        (cross-run memory)
  4. run probes            (pre-computed evidence)
  5. cluster analysis      (if accuracy < 0.85)
  6. agent.investigate()   (LLM investigation loop — diagnoses + decides)
  7. safety overrides      (structural invalidity, budget, normalization blocks)
  8. save transcript       (full investigation log for traceability)
  9. learning updates      (record checkpoint for cross-run memory)
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from _resolve_import import ensure_project_root
ensure_project_root()

from helpers.investigator_acceptance import (
    DEFAULT_MIN_DELTA,
    build_normalization_gate_request,
    evaluate_pending_normalization_acceptance,
)
from helpers.investigator_learning import (
    append_jsonl,
    dataset_signature_from_state,
    to_float,
)
from helpers.investigator_probe_runner import run_investigator_probes
from workflow_logging import log_agent_action
import config
from config import (
    MAX_INVESTIGATION_ATTEMPTS,
    NORMALIZATION_MAX_CONSECUTIVE_REJECTIONS,
    QUALITY_GATE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _learning_context_key(state: Dict[str, Any]) -> str:
    datasets = state.get("datasets", []) if isinstance(state, dict) else []
    dataset_names = sorted(os.path.basename(str(p)) for p in datasets if str(p).strip())
    eval_path = str(
        state.get("auto_diagnostics", {}).get("evaluation_testset_path", "")
        if isinstance(state.get("auto_diagnostics", {}), dict)
        else ""
    )
    testset_name = os.path.basename(eval_path) if eval_path else ""
    return f"ds:{','.join(dataset_names)}|ts:{testset_name}"


def _apply_acceptance_feedback(
    local_state: Dict[str, Any],
    acceptance_feedback: Dict[str, Any],
) -> tuple[bool, bool]:
    """Process normalization acceptance/rejection. Returns (force_skip, rollback_applied)."""
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
            streak = int(local_state.get("normalization_consecutive_rejections", 0)) + 1
            local_state["normalization_consecutive_rejections"] = streak
            if streak >= NORMALIZATION_MAX_CONSECUTIVE_REJECTIONS:
                local_state["normalization_blocked_by_rejection_streak"] = True
                print(f"[INVESTIGATION] Normalization blocked: {streak} consecutive rejections")
        else:
            local_state["normalization_consecutive_rejections"] = 0
    else:
        force_skip_normalization = True
    return force_skip_normalization, rollback_applied


def _load_dataset_for_probes(path: str):
    """Lightweight dataset loader for probe functions."""
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


def _build_fusion_guidance(
    probe_outputs: Dict[str, Any],
    cluster_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Build fusion guidance from probe mismatch data and cluster analysis.

    This is pure data transformation — no decisions, no scoring.
    The investigation LLM provides strategy recommendations separately.
    """
    guidance: Dict[str, Any] = {
        "mismatch_classifications": {},
        "post_clustering": {},
        "global_hints": [],
    }

    # Extract mismatch classifications from mismatch_sampler probe
    results_list = probe_outputs.get("results", []) if isinstance(probe_outputs, dict) else []
    for probe in results_list:
        if not isinstance(probe, dict) or probe.get("name") != "mismatch_sampler":
            continue
        for attr, samples in probe.get("samples_by_attribute", {}).items():
            if not isinstance(samples, list):
                continue
            reason_counts: Dict[str, int] = {}
            for sample in samples:
                reason = sample.get("reason", "unknown") if isinstance(sample, dict) else "unknown"
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            guidance["mismatch_classifications"][attr] = reason_counts

    # Derive global hints from mismatch patterns
    if guidance["mismatch_classifications"]:
        all_reasons: Dict[str, int] = {}
        for attr_reasons in guidance["mismatch_classifications"].values():
            for reason, count in attr_reasons.items():
                all_reasons[reason] = all_reasons.get(reason, 0) + count
        total = sum(all_reasons.values()) or 1
        norm_reasons = sum(
            all_reasons.get(r, 0)
            for r in ("case_mismatch", "whitespace_mismatch", "format_mismatch", "country_format")
        )
        if norm_reasons / total >= 0.5:
            guidance["global_hints"].append(
                f"Normalization dominates ({norm_reasons}/{total} mismatches are case/format/country)"
            )
        list_reasons = all_reasons.get("list_format_mismatch", 0)
        if list_reasons / total >= 0.2:
            guidance["global_hints"].append(
                f"List format mismatches significant ({list_reasons}/{total})"
            )

    # Cluster analysis post-clustering recommendation
    if isinstance(cluster_analysis, dict):
        overall = cluster_analysis.get("_overall", {})
        if isinstance(overall, dict):
            recommended = str(overall.get("recommended_strategy", "")).strip()
            if recommended and recommended != "None":
                guidance["post_clustering"] = {
                    "recommended_strategy": recommended,
                    "reason": "cluster analysis evidence",
                }

    return guidance


def _enrich_fusion_guidance_with_strategies(
    fusion_guidance: Dict[str, Any],
    probe_outputs: Dict[str, Any],
    investigation_log: Dict[str, Any],
) -> None:
    """Populate attribute_strategies from source_attribution probe + investigation recommendations.

    This bridges the gap between the investigation LLM's recommendations and
    the code guardrails that enforce them.  Without this, the guardrails see
    an empty attribute_strategies and cannot enforce or preserve the correct
    resolver/trust_map choices across scaffold reuse cycles.
    """
    attribute_strategies: Dict[str, Any] = {}

    # 1. Seed from source_attribution probe (data-driven trust ordering)
    results_list = probe_outputs.get("results", []) if isinstance(probe_outputs, dict) else []
    for probe in results_list:
        if not isinstance(probe, dict) or probe.get("name") != "source_attribution":
            continue
        for attr, info in probe.get("per_attribute", {}).items():
            if not isinstance(info, dict):
                continue
            recommended = info.get("recommended_resolver", "")
            trust_order = info.get("recommended_trust_order", [])
            if not recommended:
                continue
            trust_map = {}
            if trust_order:
                # Convert ordered list to trust scores (highest first)
                for rank, ds in enumerate(trust_order):
                    trust_map[ds] = len(trust_order) - rank
            attribute_strategies[attr] = {
                "recommended_fuser": recommended,
                "trust_map": trust_map,
                "field_shape": "scalar",
                "confidence": 0.7,
                "source": "source_attribution_probe",
            }

    # 2. Override/supplement with investigation LLM recommendations (higher confidence)
    decision = investigation_log.get("decision", {})
    recommendations = decision.get("recommendations", []) if isinstance(decision, dict) else []
    for rec in recommendations:
        if not isinstance(rec, dict):
            continue
        attr = rec.get("attribute", "")
        fix_text = str(rec.get("fix", rec.get("action", "")))
        if not attr or not fix_text:
            continue

        # Parse resolver name from the fix text (all 17 PyDI built-in resolvers)
        resolver = ""
        for candidate in ("prefer_higher_trust", "favour_sources", "voting",
                          "most_complete", "longest_string", "shortest_string",
                          "median", "average", "maximum", "minimum", "sum_values",
                          "earliest", "most_recent",
                          "union", "intersection", "intersection_k_sources",
                          "random_value", "weighted_voting"):
            if candidate in fix_text.lower():
                resolver = candidate
                break

        if not resolver:
            continue

        # Parse trust_map from fix text if present
        trust_map = {}
        import re
        tm_match = re.search(r"trust_map\s*=\s*\{([^}]+)\}", fix_text)
        if tm_match:
            for pair in tm_match.group(1).split(","):
                parts = pair.strip().split(":")
                if len(parts) == 2:
                    key = parts[0].strip().strip("'\"")
                    try:
                        val = int(parts[1].strip())
                    except ValueError:
                        try:
                            val = float(parts[1].strip())
                        except ValueError:
                            continue
                    trust_map[key] = val

        # If we already have probe data for this attr, merge trust_map
        existing = attribute_strategies.get(attr, {})
        if not trust_map and existing.get("trust_map"):
            trust_map = existing["trust_map"]

        attribute_strategies[attr] = {
            "recommended_fuser": resolver,
            "trust_map": trust_map,
            "field_shape": existing.get("field_shape", "scalar"),
            "confidence": 0.9,
            "source": "investigation",
        }

    fusion_guidance["attribute_strategies"] = attribute_strategies


def _reasoning_brief_from_investigation(investigation_log: Dict[str, Any]) -> Dict[str, Any]:
    """Map investigation decision to evaluation_reasoning_brief for downstream consumers."""
    decision = investigation_log.get("decision", {})
    if not isinstance(decision, dict):
        return {}
    return {
        "root_cause": decision.get("diagnosis", ""),
        "next_step": decision.get("reasoning", ""),
        "takeaway": decision.get("diagnosis", ""),
        "recommendations": decision.get("recommendations", []),
        "next_node": decision.get("next_node", "pipeline_adaption"),
        "ceiling_assessment": decision.get("ceiling_assessment", ""),
    }


def _action_plan_from_investigation(investigation_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract action plan from investigation recommendations."""
    decision = investigation_log.get("decision", {})
    recommendations = decision.get("recommendations", []) if isinstance(decision, dict) else []
    plan: List[Dict[str, Any]] = []
    for rec in recommendations:
        if not isinstance(rec, dict):
            continue
        plan.append({
            "action": rec.get("fix", rec.get("action", "")),
            "target_attributes": [rec["attribute"]] if rec.get("attribute") else [],
            "priority_score": rec.get("expected_impact", 0.8),
            "confidence": 0.8,
            "source": "investigation",
            "attribute_class": rec.get("class", "unknown"),
        })
    return plan


def _save_investigation_transcript(
    investigation_log: Dict[str, Any],
    attempt: int,
) -> str:
    """Save full investigation transcript for traceability. Returns file path."""
    os.makedirs(config.INVESTIGATION_TRANSCRIPT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(config.INVESTIGATION_TRANSCRIPT_DIR, f"investigation_{attempt}_{timestamp}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(investigation_log, f, indent=2, default=str)
        print(f"[INVESTIGATION] Transcript saved → {os.path.basename(path)}")
    except Exception as e:
        print(f"[INVESTIGATION] Could not save transcript: {e}")
        path = ""
    return path


def _print_investigation_report(
    *,
    overall_acc: float,
    attempts: int,
    max_attempts: int,
    probe_outputs: Dict[str, Any],
    investigator_decision: str,
    investigation_log: Dict[str, Any],
    acceptance_feedback: Dict[str, Any],
    rollback_applied: bool,
) -> None:
    turns = len(investigation_log.get("turns", []))
    decision = investigation_log.get("decision", {})
    diagnosis = decision.get("diagnosis", "none")[:200] if isinstance(decision, dict) else "none"

    print(f"[INVESTIGATION] Attempt {attempts}/{max_attempts} | accuracy={overall_acc:.3%} | {turns} turn(s) → {investigator_decision}")
    print(f"[INVESTIGATION] Diagnosis: {diagnosis}")

    recs = decision.get("recommendations", []) if isinstance(decision, dict) else []
    for rec in recs[:3]:
        if isinstance(rec, dict):
            print(f"  → {rec.get('attribute', '?')}: {rec.get('fix', rec.get('issue', ''))[:120]}")

    if isinstance(acceptance_feedback, dict) and acceptance_feedback.get("decision_ready"):
        print(f"[INVESTIGATION] Normalization acceptance: {acceptance_feedback.get('status')} (delta={acceptance_feedback.get('observed_delta')})")
    if rollback_applied:
        print("[INVESTIGATION] Rolled back rejected normalization")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_investigator_node(agent, state: Dict[str, Any]) -> Dict[str, Any]:
    log_agent_action(
        agent,
        step="investigator_node",
        action="start",
        why="Investigate pipeline issues and decide next step",
        improvement="LLM-driven investigation with optional code execution",
    )
    print("[INVESTIGATION] Starting investigator node")

    local_state = dict(state)

    # ── 1. Evaluation decision (metric bookkeeping, regression guards) ──
    decision_updates = agent.evaluation_decision(local_state)
    local_state.update(decision_updates)

    accepted_metrics = local_state.get("evaluation_metrics", {})
    metrics = local_state.get(
        "evaluation_metrics_for_adaptation",
        local_state.get("latest_validation_metrics", accepted_metrics),
    )
    if not isinstance(metrics, dict):
        metrics = {}
    overall_acc = to_float(metrics.get("overall_accuracy", 0.0), 0.0)
    accepted_overall_acc = to_float(
        accepted_metrics.get("overall_accuracy", 0.0) if isinstance(accepted_metrics, dict) else 0.0,
        0.0,
    )
    attempts = int(local_state.get("evaluation_attempts", 0))
    max_attempts = MAX_INVESTIGATION_ATTEMPTS

    # ── 2. Acceptance feedback (normalization rollback) ──
    acceptance_feedback = evaluate_pending_normalization_acceptance(
        local_state.get("normalization_pending_acceptance", {}),
        current_accuracy=overall_acc,
        current_evaluation_attempt=attempts,
        current_total_evaluations=int(to_float(metrics.get("total_evaluations"), 0.0)),
        current_total_correct=int(to_float(metrics.get("total_correct"), 0.0)),
        current_metrics=accepted_metrics if isinstance(accepted_metrics, dict) else {},
    )
    force_skip_normalization, rollback_applied = _apply_acceptance_feedback(local_state, acceptance_feedback)
    if local_state.get("normalization_blocked_by_rejection_streak"):
        force_skip_normalization = True

    # ── 3. Dataset signature (for history tracking) ──
    dataset_signature = dataset_signature_from_state(local_state)

    # ── 4. Run probes (pre-computed evidence) ──
    prev_action_plan = list(local_state.get("investigator_action_plan", []) or [])
    probe_outputs = run_investigator_probes(
        state=local_state,
        action_plan=prev_action_plan,
        load_dataset_fn=_load_dataset_for_probes,
    )
    local_state["investigator_probe_results"] = probe_outputs

    # ── 5. Early exit: quality gate reached or budget exhausted ──
    if overall_acc >= QUALITY_GATE_THRESHOLD or attempts >= max_attempts:
        investigator_decision = "human_review_export"
        print(f"[INVESTIGATION] Early exit → {investigator_decision} (accuracy={overall_acc:.1%}, attempt {attempts}/{max_attempts})")

        existing_history = list(local_state.get("iteration_history", []) or [])
        existing_history.append({
            "attempt": attempts,
            "accuracy": round(float(overall_acc), 4),
            "delta": round(float(overall_acc) - float(existing_history[-1]["accuracy"]), 4)
                   if existing_history and existing_history[-1].get("accuracy") is not None else None,
            "decision": investigator_decision,
            "description": "quality gate reached" if overall_acc >= QUALITY_GATE_THRESHOLD else "max attempts reached",
        })

        return {
            **decision_updates,
            "investigator_decision": investigator_decision,
            "investigator_probe_results": probe_outputs,
            "normalization_pending_acceptance": local_state.get("normalization_pending_acceptance", {}),
            "normalization_acceptance_feedback": acceptance_feedback,
            "normalization_rollback_applied": rollback_applied,
            "iteration_history": existing_history,
        }

    # ── 6. Cluster analysis (evidence, not decision) ──
    cluster_updates: Dict[str, Any] = {}
    if overall_acc < 0.85:
        cluster_updates = agent.cluster_analysis(local_state)
        local_state.update(cluster_updates)

    # ── 7. Build fusion guidance from probes + cluster (data transformation) ──
    fusion_guidance = _build_fusion_guidance(
        probe_outputs,
        local_state.get("cluster_analysis_result", {}),
    )
    local_state["fusion_guidance"] = fusion_guidance

    # ── 8. LLM investigation loop ──
    investigation_log = agent.investigate(local_state)

    # ── 8b. Enrich fusion guidance with attribute strategies ──
    _enrich_fusion_guidance_with_strategies(fusion_guidance, probe_outputs, investigation_log)
    local_state["fusion_guidance"] = fusion_guidance

    # ── 9. Safety overrides ──
    decision = investigation_log.get("decision", {})
    investigator_decision = decision.get("next_node", "pipeline_adaption") if isinstance(decision, dict) else "pipeline_adaption"

    correspondence_integrity = local_state.get("correspondence_integrity", {})
    if isinstance(correspondence_integrity, dict) and not correspondence_integrity.get("structurally_valid", True):
        investigator_decision = "pipeline_adaption"
        investigation_log.setdefault("safety_overrides", []).append("structural_invalid_correspondences")
        print("[INVESTIGATION] Safety override: invalid correspondences → pipeline_adaption")

    if investigator_decision == "normalization_node" and force_skip_normalization:
        investigator_decision = "pipeline_adaption"
        investigation_log.setdefault("safety_overrides", []).append("normalization_blocked")
        print("[INVESTIGATION] Safety override: normalization blocked → pipeline_adaption")

    if investigator_decision == "normalization_node" and int(local_state.get("normalization_attempts", 0)) >= 3:
        investigator_decision = "pipeline_adaption"
        investigation_log.setdefault("safety_overrides", []).append("normalization_max_attempts")
        print("[INVESTIGATION] Safety override: normalization max attempts → pipeline_adaption")

    # When blocking/matching configs already exist with strategies, re-running
    # testers risks overwriting good configs. Let the tester node itself decide
    # whether to skip (it checks config signature). No override needed here —
    # the graph edge now includes run_blocking_tester as a valid target.

    # Validate routing target
    valid_targets = {"normalization_node", "run_blocking_tester", "run_matching_tester", "pipeline_adaption", "human_review_export"}
    if investigator_decision not in valid_targets:
        print(f"[INVESTIGATION] Invalid target '{investigator_decision}' → defaulting to pipeline_adaption")
        investigator_decision = "pipeline_adaption"

    # Build normalization gate request if routing to normalization
    normalization_gate_request: Dict[str, Any] = {}
    if investigator_decision == "normalization_node":
        normalization_gate_request = build_normalization_gate_request(
            state=local_state,
            baseline_accuracy=overall_acc,
            min_delta=DEFAULT_MIN_DELTA,
        )

    # ── 10. Save investigation transcript ──
    investigation_log["final_decision"] = investigator_decision
    investigation_log["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    transcript_path = _save_investigation_transcript(investigation_log, attempts)

    # ── 11. History ──
    # Extract ceiling estimate from probe results if available
    _probe_results_list = probe_outputs.get("results", []) if isinstance(probe_outputs, dict) else []
    _improv_probe = next((p for p in _probe_results_list if isinstance(p, dict) and p.get("name") == "attribute_improvability"), None)
    _ceiling_estimate = _improv_probe.get("structural_ceiling_estimate") if _improv_probe else None

    from helpers.context_summarizer import build_stagnation_analysis
    _stagnation_detected = bool(build_stagnation_analysis(local_state))

    history_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_signature": dataset_signature,
        "overall_accuracy": round(float(overall_acc), 6),
        "evaluation_attempt": attempts,
        "decision": investigator_decision,
        "accepted_overall_accuracy": round(float(accepted_overall_acc), 6),
        "investigation_turns": len(investigation_log.get("turns", [])),
        "diagnosis": (decision.get("diagnosis", "") if isinstance(decision, dict) else "")[:300],
        "acceptance_feedback": acceptance_feedback,
        "rollback_applied": rollback_applied,
        "safety_overrides": investigation_log.get("safety_overrides", []),
        "structural_ceiling_estimate": _ceiling_estimate,
        "stagnation_detected": _stagnation_detected,
    }
    history_path = os.path.join(config.INVESTIGATION_TRANSCRIPT_DIR, "investigator_history.jsonl")
    append_jsonl(history_path, history_payload)

    _print_investigation_report(
        overall_acc=overall_acc,
        attempts=attempts,
        max_attempts=max_attempts,
        probe_outputs=probe_outputs,
        investigator_decision=investigator_decision,
        investigation_log=investigation_log,
        acceptance_feedback=acceptance_feedback,
        rollback_applied=rollback_applied,
    )

    # Build iteration history entry
    existing_history = list(local_state.get("iteration_history", []) or [])
    reasoning_brief = _reasoning_brief_from_investigation(investigation_log)
    action_plan = _action_plan_from_investigation(investigation_log)

    history_entry = {
        "attempt": attempts,
        "accuracy": round(float(overall_acc), 4),
        "delta": round(float(overall_acc) - float(existing_history[-1]["accuracy"]), 4)
                 if existing_history and existing_history[-1].get("accuracy") is not None else None,
        "decision": investigator_decision,
        "description": reasoning_brief.get("next_step", reasoning_brief.get("takeaway", ""))[:150],
    }
    # Include per-attribute accuracy so the pipeline LLM can see what improved/regressed
    eval_metrics = local_state.get("evaluation_metrics_for_adaptation", local_state.get("evaluation_metrics", {}))
    if isinstance(eval_metrics, dict):
        attr_accs = {}
        for key, val in eval_metrics.items():
            if key.endswith("_accuracy") and not key.startswith("overall") and not key.startswith("macro"):
                attr_name = key.replace("_accuracy", "")
                try:
                    attr_accs[attr_name] = round(float(val), 3)
                except (ValueError, TypeError):
                    pass
        if attr_accs:
            history_entry["attribute_accuracies"] = attr_accs
    previous_code = str(local_state.get("integration_pipeline_code", ""))[:3000]
    if previous_code and previous_code != "Pipeline code not available":
        history_entry["previous_pipeline_snippet"] = previous_code[:500]
    existing_history.append(history_entry)

    # ── 13. Return state updates ──
    out: Dict[str, Any] = {
        **decision_updates,
        **cluster_updates,
        "investigator_decision": investigator_decision,
        "investigation_log": investigation_log,
        "investigation_transcript_path": transcript_path,
        "evaluation_reasoning_brief": reasoning_brief,
        "evaluation_analysis": json.dumps(investigation_log.get("decision", {}), indent=2),
        "investigator_action_plan": action_plan,
        "fusion_guidance": fusion_guidance,
        "investigator_probe_results": probe_outputs,
        "normalization_gate_request": normalization_gate_request,
        "normalization_pending_acceptance": local_state.get("normalization_pending_acceptance", {}),
        "normalization_acceptance_feedback": acceptance_feedback,
        "normalization_rollback_applied": rollback_applied,
        "iteration_history": existing_history,
    }
    if rollback_applied:
        out["datasets"] = local_state.get("datasets", [])
        out["normalized_datasets"] = []
    return out
