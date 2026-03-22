"""Evaluation decision logic — processes metrics, applies regression guards,
computes auto diagnostics, and returns state updates for the next iteration.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import config
from config import (
    ID_DIRECT_COVERAGE_LOW,
    ID_DIRECT_COVERAGE_THRESHOLD,
    ID_MAPPED_COVERAGE_LOW,
    ID_MAPPED_COVERAGE_THRESHOLD,
    MISSING_FUSED_VALUE_THRESHOLD,
    MISMATCH_RATIO_THRESHOLD,
    MISSING_VALUE_THRESHOLD,
    PIPELINE_EXEC_MAX_ATTEMPTS,
    QUALITY_GATE_THRESHOLD,
)
# NOTE: EVALUATION_JSON_PATH and FUSED_OUTPUT_PATH are accessed via config.X
# (not imported directly) because configure_run_output() mutates them at runtime.
from helpers.correspondence import evaluate_correspondence_integrity
from helpers.evaluation_sanity import check_evaluation_sanity
from helpers.metrics import (
    assess_validation_regression,
    is_metrics_better,
    is_metrics_payload,
)


def pipeline_problem_classes(state: Dict[str, Any]) -> List[str]:
    """Classifies current pipeline problems from state evidence."""
    problem_classes: List[str] = []

    pipeline_exec = str(state.get("pipeline_execution_result", "") or "").strip().lower()
    if pipeline_exec.startswith("error"):
        problem_classes.append("execution_failure")

    metrics = state.get("evaluation_metrics_for_adaptation", state.get("evaluation_metrics", {}))
    if isinstance(metrics, dict):
        try:
            overall = float(metrics.get("overall_accuracy", 0.0) or 0.0)
            if overall < QUALITY_GATE_THRESHOLD:
                problem_classes.append("quality_below_target")
        except (ValueError, TypeError) as e:
            print(f"[EVAL DECISION] Could not parse overall_accuracy: {e}")

    corr = state.get("correspondence_integrity", {})
    if isinstance(corr, dict) and corr and not bool(corr.get("structurally_valid", True)):
        problem_classes.append("structural_invalidity")

    auto_diagnostics = state.get("auto_diagnostics", {})
    if isinstance(auto_diagnostics, dict):
        id_alignment = auto_diagnostics.get("id_alignment", {})
        if isinstance(id_alignment, dict):
            try:
                direct_ratio = float(id_alignment.get("direct_coverage_ratio", 0.0) or 0.0)
                mapped_ratio = float(id_alignment.get("mapped_coverage_ratio", 0.0) or 0.0)
                if mapped_ratio >= ID_MAPPED_COVERAGE_THRESHOLD and direct_ratio < ID_DIRECT_COVERAGE_THRESHOLD:
                    problem_classes.append("identity_representation_mismatch")
            except (ValueError, TypeError) as e:
                print(f"[EVAL DECISION] Could not parse ID alignment ratios: {e}")

        debug_ratios = auto_diagnostics.get("debug_reason_ratios", {})
        if isinstance(debug_ratios, dict):
            try:
                missing = float(debug_ratios.get("missing_fused_value", 0.0) or 0.0)
                mismatch = float(debug_ratios.get("mismatch", 0.0) or 0.0)
                if mismatch >= MISMATCH_RATIO_THRESHOLD and missing <= MISSING_VALUE_THRESHOLD:
                    problem_classes.append("representation_or_selection_mismatch")
            except (ValueError, TypeError) as e:
                print(f"[EVAL DECISION] Could not parse debug reason ratios: {e}")

    regression = state.get("evaluation_regression_guard", {})
    if isinstance(regression, dict) and bool(regression.get("rejected")):
        problem_classes.append("regression_guard_rejection")

    seen = set()
    ordered: List[str] = []
    for item in problem_classes:
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def process_evaluation_decision(
    state: Dict[str, Any],
    eval_testset_path: str,
    eval_stage_label: str,
    compute_auto_diagnostics_fn,
    print_total_usage_fn=None,
    logger=None,
) -> Dict[str, Any]:
    """Core evaluation decision logic — extracted from SimpleModelAgent.evaluation_decision.

    Returns the state update dict.
    """
    attempts = state.get("evaluation_attempts", 0) + 1
    eval_path = config.EVALUATION_JSON_PATH

    metrics_from_execution = state.get("evaluation_metrics_from_execution", {})
    if is_metrics_payload(metrics_from_execution):
        metrics = dict(metrics_from_execution)
        metrics_source = state.get("evaluation_metrics_source", "execution")
    elif os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        metrics_source = "pipeline_evaluation_file"
    else:
        if logger:
            logger.warning("Evaluation file missing")
        metrics = {"error": "evaluation file missing"}
        metrics_source = "missing"

    raw_metrics = dict(metrics) if isinstance(metrics, dict) else {}

    # ── Sanity-check metrics ──────────────────────────────────────────────
    sanity_passed, sanity_warnings = check_evaluation_sanity(
        metrics=raw_metrics,
        fused_path=config.FUSED_OUTPUT_PATH,
        eval_testset_path=eval_testset_path,
    )
    if sanity_warnings:
        print(f"[EVAL DECISION] Sanity warnings: {'; '.join(sanity_warnings[:3])}")

    if not sanity_passed:
        print("[EVAL DECISION] Sanity check failed — treating as execution failure")
        return {
            "evaluation_execution_result": f"error: evaluation sanity check failed: {'; '.join(sanity_warnings)}",
            "evaluation_execution_attempts": int(state.get("evaluation_execution_attempts", 0)),
            "evaluation_metrics": state.get("evaluation_metrics", {}),
            "evaluation_metrics_raw": raw_metrics,
            "evaluation_attempts": attempts - 1,
            "evaluation_sanity_warnings": sanity_warnings,
            "best_validation_metrics": state.get("best_validation_metrics", {}),
            "auto_diagnostics": state.get("auto_diagnostics", {}),
            "evaluation_regression_guard": {},
            "correspondence_integrity": {},
            "evaluation_cycle_audit": list(state.get("evaluation_cycle_audit", []) or []),
        }

    # ── Regression guard ──────────────────────────────────────────────────
    best_metrics = state.get("best_validation_metrics", {})
    correspondence_integrity = evaluate_correspondence_integrity(state)
    structural_invalid = not bool(correspondence_integrity.get("structurally_valid", True))
    regression = assess_validation_regression(
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
    best_pipeline_code = state.get("best_pipeline_code", "")
    best_evaluation_code = state.get("best_evaluation_code", "")
    if regression.get("rejected"):
        if is_metrics_payload(best_metrics):
            accepted_metrics = dict(best_metrics)
            print(f"[EVAL DECISION] Regression rejected — keeping best accuracy={best_metrics.get('overall_accuracy')}")
    else:
        if is_metrics_better(raw_metrics, best_metrics if isinstance(best_metrics, dict) else {}):
            best_metrics = dict(raw_metrics)
            # Snapshot the pipeline + evaluation code that produced the best metrics
            best_pipeline_code = state.get("integration_pipeline_code", "")
            best_evaluation_code = state.get("integration_evaluation_code", "")
            print(f"[EVAL DECISION] New best accuracy: {raw_metrics.get('overall_accuracy')}")
        elif not is_metrics_payload(best_metrics) and is_metrics_payload(raw_metrics):
            best_metrics = dict(raw_metrics)
            best_pipeline_code = state.get("integration_pipeline_code", "")
            best_evaluation_code = state.get("integration_evaluation_code", "")

    analysis_metrics = raw_metrics if is_metrics_payload(raw_metrics) else accepted_metrics

    if logger:
        logger.info(f"Evaluation metrics ({metrics_source}): {raw_metrics}")
    if structural_invalid:
        invalid_pairs = correspondence_integrity.get("invalid_pairs", [])
        print(f"[EVAL DECISION] Structural invalid correspondences for pairs: {invalid_pairs}")

    # Mutate state for downstream consumers within same node
    state["latest_validation_metrics"] = raw_metrics
    state["best_validation_metrics"] = best_metrics if isinstance(best_metrics, dict) else {}
    state["evaluation_regression_guard"] = regression
    state["correspondence_integrity"] = correspondence_integrity
    state["evaluation_metrics"] = accepted_metrics
    state["evaluation_metrics_raw"] = raw_metrics
    state["evaluation_metrics_for_adaptation"] = analysis_metrics

    try:
        overall_acc = float(analysis_metrics.get("overall_accuracy", 0.0))
    except (ValueError, TypeError) as e:
        print(f"[EVAL DECISION] Could not parse overall_accuracy from analysis_metrics: {e}")
        overall_acc = 0.0

    auto_diagnostics = compute_auto_diagnostics_fn(state, analysis_metrics)
    state["auto_diagnostics"] = auto_diagnostics

    # ── Problem enumeration ───────────────────────────────────────────────
    problems: List[str] = []
    if overall_acc < QUALITY_GATE_THRESHOLD:
        problems.append(f"overall_accuracy below target: {overall_acc:.3%} < {QUALITY_GATE_THRESHOLD:.3%}")

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
            if mapped_ratio is not None and float(mapped_ratio) < ID_MAPPED_COVERAGE_LOW:
                problems.append(f"low mapped ID coverage: {float(mapped_ratio):.3%}")
        except (ValueError, TypeError) as e:
            print(f"[EVAL DECISION] Could not parse mapped_coverage_ratio: {e}")
        try:
            if direct_ratio is not None and float(direct_ratio) < ID_DIRECT_COVERAGE_LOW:
                problems.append(f"low direct ID coverage: {float(direct_ratio):.3%}")
        except (ValueError, TypeError) as e:
            print(f"[EVAL DECISION] Could not parse direct_coverage_ratio: {e}")

    reason_ratios = auto_diagnostics.get("debug_reason_ratios", {}) if isinstance(auto_diagnostics, dict) else {}
    if isinstance(reason_ratios, dict):
        try:
            missing_ratio = float(reason_ratios.get("missing_fused_value", 0.0))
            if missing_ratio > MISSING_FUSED_VALUE_THRESHOLD:
                problems.append(f"high missing_fused_value mismatch ratio: {missing_ratio:.3%}")
        except (ValueError, TypeError) as e:
            print(f"[EVAL DECISION] Could not parse missing_fused_value ratio: {e}")

    # ── Report ────────────────────────────────────────────────────────────
    problem_summary = f"{len(problems)} problem(s)" if problems else "none"
    next_action = ("quality gate reached" if overall_acc >= QUALITY_GATE_THRESHOLD
                   else "max attempts reached" if attempts >= 3
                   else "investigating")
    print(f"[EVAL DECISION] Attempt {attempts} | accuracy={overall_acc:.3%} | problems={problem_summary} | {next_action}")
    for p in problems[:5]:
        print(f"  - {p}")

    if overall_acc >= QUALITY_GATE_THRESHOLD or attempts >= PIPELINE_EXEC_MAX_ATTEMPTS:
        if print_total_usage_fn:
            print_total_usage_fn()

    cycle_audit = list(state.get("evaluation_cycle_audit", []) or [])
    cycle_audit.append(
        {
            "attempt": int(attempts),
            "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "evaluation_stage": eval_stage_label,
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
        "best_pipeline_code": best_pipeline_code,
        "best_evaluation_code": best_evaluation_code,
        "evaluation_metrics_raw": raw_metrics,
        "evaluation_metrics_for_adaptation": analysis_metrics,
        "correspondence_integrity": correspondence_integrity,
        "evaluation_cycle_audit": cycle_audit,
        "evaluation_sanity_warnings": sanity_warnings,
    }
