from typing import Any, Dict, List

from helpers.investigator_learning import LEARNING_MIN_OBSERVATIONS, to_float

ROUTING_WEIGHTS = {
    "llm_base": 0.45,
    "llm_confidence": 0.35,
    "fallback": 0.30,
    "low_accuracy": 0.12,
    "missing_fused": 0.08,
    "learning_bias": 0.10,
    "stagnation_penalty": 0.14,
    "exploration_bonus": 0.06,
    "probe_normalization_pressure": 0.18,
    "objective_low_score": 0.15,
    "objective_worst_attribute_pressure": 0.12,
}
ROUTING_BASE_THRESHOLD = 0.72
LEARNING_REGRET_OVERRIDE_MARGIN = 0.18


def compute_normalization_routing(
    *,
    overall_acc: float,
    attempts: int,
    max_attempts: int,
    llm_assessment: Dict[str, Any],
    fallback_needed: bool,
    diagnostics: Dict[str, Any],
    action_plan: List[Dict[str, Any]],
    learning_signals: Dict[str, Any],
    probe_results: Dict[str, Any] | None = None,
    route_objective: Dict[str, Any] | None = None,
    force_skip_normalization: bool = False,
) -> Dict[str, Any]:
    if attempts >= max_attempts or force_skip_normalization:
        return {
            "route_to_normalization": False,
            "score": 0.0,
            "threshold": ROUTING_BASE_THRESHOLD,
            "blocked_by_action_plan": False,
            "blocked_by_learning_regret": False,
            "components": {
                "attempt_budget_exhausted": 1.0 if attempts >= max_attempts else 0.0,
                "forced_skip_normalization": 1.0 if force_skip_normalization else 0.0,
            },
            "learning_signals": learning_signals,
        }

    llm_flag = bool(llm_assessment.get("needs_normalization", False))
    llm_conf = max(0.0, min(1.0, to_float(llm_assessment.get("confidence"), 0.0)))
    debug_ratios = diagnostics.get("debug_reason_ratios", {}) if isinstance(diagnostics, dict) else {}
    missing_ratio = to_float(debug_ratios.get("missing_fused_value"), 0.0)
    probe_norm_pressure = max(0.0, min(1.0, to_float((probe_results or {}).get("normalization_pressure"), 0.0)))

    score = 0.0
    components: Dict[str, float] = {}
    if llm_flag:
        llm_component = ROUTING_WEIGHTS["llm_base"] + (ROUTING_WEIGHTS["llm_confidence"] * llm_conf)
        score += llm_component
        components["llm"] = round(llm_component, 6)
    if fallback_needed:
        score += ROUTING_WEIGHTS["fallback"]
        components["fallback"] = ROUTING_WEIGHTS["fallback"]
    if overall_acc < 0.60:
        score += ROUTING_WEIGHTS["low_accuracy"]
        components["low_accuracy"] = ROUTING_WEIGHTS["low_accuracy"]
    if missing_ratio > 0.25:
        score += ROUTING_WEIGHTS["missing_fused"]
        components["missing_fused"] = ROUTING_WEIGHTS["missing_fused"]
    if probe_norm_pressure > 0.0:
        probe_component = ROUTING_WEIGHTS["probe_normalization_pressure"] * probe_norm_pressure
        score += probe_component
        components["probe_normalization_pressure"] = round(probe_component, 6)

    objective = route_objective if isinstance(route_objective, dict) else {}
    objective_score = max(0.0, min(1.0, to_float(objective.get("composite_score"), 1.0)))
    worst_attr_pressure = max(0.0, min(1.0, to_float(objective.get("worst_attribute_pressure"), 0.0)))
    if objective_score < 0.70:
        low_score_component = ROUTING_WEIGHTS["objective_low_score"] * (0.70 - objective_score) / 0.70
        score += low_score_component
        components["objective_low_score"] = round(low_score_component, 6)
    if worst_attr_pressure > 0.0:
        worst_component = ROUTING_WEIGHTS["objective_worst_attribute_pressure"] * worst_attr_pressure
        score += worst_component
        components["objective_worst_attribute_pressure"] = round(worst_component, 6)

    learning_component = ROUTING_WEIGHTS["learning_bias"] * to_float(learning_signals.get("learning_bias"), 0.0)
    score += learning_component
    components["learning_bias"] = round(learning_component, 6)

    norm_count = int((learning_signals.get("norm") or {}).get("count", 0))
    pipe_count = int((learning_signals.get("pipe") or {}).get("count", 0))
    if norm_count < LEARNING_MIN_OBSERVATIONS and pipe_count >= LEARNING_MIN_OBSERVATIONS:
        score += ROUTING_WEIGHTS["exploration_bonus"]
        components["exploration_bonus"] = ROUTING_WEIGHTS["exploration_bonus"]
    if learning_signals.get("recent_stall"):
        stall_streak = max(1, int(learning_signals.get("normalization_stall_streak", 0)))
        stall_penalty = ROUTING_WEIGHTS["stagnation_penalty"] * min(1.75, 1.0 + (0.2 * (stall_streak - 1)))
        score -= stall_penalty
        components["stagnation_penalty"] = round(-stall_penalty, 6)

    top_action_score = 0.0
    if action_plan:
        try:
            top_action_score = max(float(x.get("priority_score", 0.0)) for x in action_plan[:3])
        except Exception:
            top_action_score = 0.0

    threshold = ROUTING_BASE_THRESHOLD
    if top_action_score >= 0.75:
        threshold += 0.08
    if top_action_score >= 0.85:
        threshold += 0.05
    blocked_by_action_plan = top_action_score >= 0.75 and score < 0.95
    blocked_by_learning_regret = bool(learning_signals.get("regret_active", False)) and (
        score < (threshold + LEARNING_REGRET_OVERRIDE_MARGIN)
    )
    route_to_normalization = (score >= threshold) and not blocked_by_action_plan and not blocked_by_learning_regret
    return {
        "route_to_normalization": route_to_normalization,
        "score": round(score, 6),
        "threshold": round(threshold, 6),
        "blocked_by_action_plan": blocked_by_action_plan,
        "blocked_by_learning_regret": blocked_by_learning_regret,
        "components": components,
        "learning_signals": learning_signals,
    }
