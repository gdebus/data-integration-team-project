from datetime import datetime, timezone
from math import sqrt
from typing import Any, Dict, List, Optional
from uuid import uuid4

from helpers.metrics import attribute_accuracy_map as _attribute_accuracy_map
from helpers.utils import to_float as _safe_float

DEFAULT_MIN_DELTA = 0.003
DEFAULT_CONFIDENCE_Z = 1.28
DEFAULT_MIN_KEY_ATTR_DELTA = 0.004
DEFAULT_HELDOUT_MAX_DROP = 0.006


def _mean_for_attributes(values: Dict[str, float], attrs: List[str]) -> float:
    if not attrs:
        return 0.0
    picked = [values[a] for a in attrs if a in values]
    if not picked:
        return 0.0
    return sum(picked) / max(1, len(picked))


def build_normalization_gate_request(
    *,
    state: Dict[str, Any],
    baseline_accuracy: float,
    min_delta: float = DEFAULT_MIN_DELTA,
    confidence_z: float = DEFAULT_CONFIDENCE_Z,
    min_key_attr_delta: float = DEFAULT_MIN_KEY_ATTR_DELTA,
    heldout_max_drop: float = DEFAULT_HELDOUT_MAX_DROP,
) -> Dict[str, Any]:
    baseline_datasets = list(state.get("datasets", []) or [])
    original_datasets = list(state.get("original_datasets", baseline_datasets) or baseline_datasets)
    metrics = state.get("evaluation_metrics", {}) if isinstance(state, dict) else {}
    attr_acc = _attribute_accuracy_map(metrics if isinstance(metrics, dict) else {})

    directives = state.get("normalization_directives", {}) if isinstance(state, dict) else {}
    focus_attrs = set()
    if isinstance(directives, dict):
        for key in ("list_columns", "lowercase_columns", "country_columns", "text_columns"):
            vals = directives.get(key, [])
            if isinstance(vals, list):
                focus_attrs.update([str(v).strip() for v in vals if str(v).strip()])
    for attr, acc in attr_acc.items():
        if acc < 0.70:
            focus_attrs.add(attr)
    focus_attr_list = sorted([a for a in focus_attrs if a in attr_acc])[:12]

    heldout_candidates = [a for a in sorted(attr_acc.keys()) if a not in set(focus_attr_list)]
    heldout_attr_list = heldout_candidates[:12]
    return {
        "gate_id": uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "baseline_accuracy": round(_safe_float(baseline_accuracy), 6),
        "min_delta": round(_safe_float(min_delta, DEFAULT_MIN_DELTA), 6),
        "min_key_attr_delta": round(_safe_float(min_key_attr_delta, DEFAULT_MIN_KEY_ATTR_DELTA), 6),
        "heldout_max_drop": round(_safe_float(heldout_max_drop, DEFAULT_HELDOUT_MAX_DROP), 6),
        "confidence_z": round(_safe_float(confidence_z, DEFAULT_CONFIDENCE_Z), 3),
        "baseline_total_evaluations": int(_safe_float(metrics.get("total_evaluations"), 0.0)),
        "baseline_total_correct": int(_safe_float(metrics.get("total_correct"), 0.0)),
        "baseline_attribute_accuracy": {k: round(v, 6) for k, v in attr_acc.items()},
        "focus_attributes": focus_attr_list,
        "heldout_proxy_attributes": heldout_attr_list,
        "baseline_focus_mean_accuracy": round(_mean_for_attributes(attr_acc, focus_attr_list), 6),
        "baseline_heldout_proxy_mean_accuracy": round(_mean_for_attributes(attr_acc, heldout_attr_list), 6),
        "baseline_datasets": baseline_datasets,
        "original_datasets": original_datasets,
        "evaluation_attempt": int(state.get("evaluation_attempts", 0)),
    }


def create_pending_normalization_acceptance(
    *,
    gate_request: Dict[str, Any],
    normalized_datasets: List[str],
    normalization_attempt: int,
    evaluation_attempt: int,
) -> Dict[str, Any]:
    if not isinstance(gate_request, dict):
        return {}
    return {
        "gate_id": gate_request.get("gate_id", uuid4().hex),
        "created_at": gate_request.get("created_at"),
        "baseline_accuracy": _safe_float(gate_request.get("baseline_accuracy"), 0.0),
        "min_delta": _safe_float(gate_request.get("min_delta"), DEFAULT_MIN_DELTA),
        "confidence_z": _safe_float(gate_request.get("confidence_z"), DEFAULT_CONFIDENCE_Z),
        "baseline_total_evaluations": int(_safe_float(gate_request.get("baseline_total_evaluations"), 0.0)),
        "baseline_total_correct": int(_safe_float(gate_request.get("baseline_total_correct"), 0.0)),
        "baseline_attribute_accuracy": dict(gate_request.get("baseline_attribute_accuracy", {}) or {}),
        "focus_attributes": list(gate_request.get("focus_attributes", []) or []),
        "heldout_proxy_attributes": list(gate_request.get("heldout_proxy_attributes", []) or []),
        "baseline_focus_mean_accuracy": _safe_float(gate_request.get("baseline_focus_mean_accuracy"), 0.0),
        "baseline_heldout_proxy_mean_accuracy": _safe_float(
            gate_request.get("baseline_heldout_proxy_mean_accuracy"), 0.0
        ),
        "min_key_attr_delta": _safe_float(gate_request.get("min_key_attr_delta"), DEFAULT_MIN_KEY_ATTR_DELTA),
        "heldout_max_drop": _safe_float(gate_request.get("heldout_max_drop"), DEFAULT_HELDOUT_MAX_DROP),
        "baseline_datasets": list(gate_request.get("baseline_datasets", []) or []),
        "original_datasets": list(gate_request.get("original_datasets", []) or []),
        "candidate_datasets": list(normalized_datasets or []),
        "normalization_attempt": int(normalization_attempt),
        "evaluation_attempt_at_creation": int(evaluation_attempt),
    }


def evaluate_pending_normalization_acceptance(
    pending: Any,
    *,
    current_accuracy: float,
    current_evaluation_attempt: int,
    current_total_evaluations: int = 0,
    current_total_correct: int = 0,
    current_metrics: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(pending, dict) or not pending:
        return None

    baseline_accuracy = _safe_float(pending.get("baseline_accuracy"), 0.0)
    min_delta = _safe_float(pending.get("min_delta"), DEFAULT_MIN_DELTA)
    creation_attempt = int(pending.get("evaluation_attempt_at_creation", -1))
    if current_evaluation_attempt <= creation_attempt:
        return {
            "status": "pending",
            "gate_id": pending.get("gate_id"),
            "baseline_accuracy": baseline_accuracy,
            "current_accuracy": round(_safe_float(current_accuracy), 6),
            "required_delta": min_delta,
            "observed_delta": round(_safe_float(current_accuracy) - baseline_accuracy, 6),
            "decision_ready": False,
        }

    observed_delta = _safe_float(current_accuracy) - baseline_accuracy
    confidence_z = _safe_float(pending.get("confidence_z"), DEFAULT_CONFIDENCE_Z)
    baseline_n = int(_safe_float(pending.get("baseline_total_evaluations"), 0.0))
    baseline_correct = int(_safe_float(pending.get("baseline_total_correct"), 0.0))
    current_n = int(_safe_float(current_total_evaluations, 0.0))
    current_correct = int(_safe_float(current_total_correct, 0.0))

    confidence_adjusted_delta = observed_delta
    confidence_supported = baseline_n > 0 and current_n > 0 and baseline_correct >= 0 and current_correct >= 0
    if confidence_supported:
        p0 = baseline_correct / max(1, baseline_n)
        p1 = current_correct / max(1, current_n)
        se = sqrt(max(1e-12, (p0 * (1.0 - p0) / max(1, baseline_n)) + (p1 * (1.0 - p1) / max(1, current_n))))
        confidence_adjusted_delta = observed_delta - (confidence_z * se)

    baseline_attr = {
        str(k): _safe_float(v, 0.0)
        for k, v in (pending.get("baseline_attribute_accuracy", {}) or {}).items()
    }
    current_attr = _attribute_accuracy_map(current_metrics if isinstance(current_metrics, dict) else {})
    focus_attrs = [str(a) for a in pending.get("focus_attributes", []) if str(a)]
    heldout_attrs = [str(a) for a in pending.get("heldout_proxy_attributes", []) if str(a)]
    focus_deltas = [current_attr[a] - baseline_attr.get(a, 0.0) for a in focus_attrs if a in current_attr]
    focus_delta_mean = (sum(focus_deltas) / len(focus_deltas)) if focus_deltas else 0.0
    focus_delta_min = min(focus_deltas) if focus_deltas else 0.0
    heldout_deltas = [current_attr[a] - baseline_attr.get(a, 0.0) for a in heldout_attrs if a in current_attr]
    heldout_delta_mean = (sum(heldout_deltas) / len(heldout_deltas)) if heldout_deltas else 0.0

    min_key_attr_delta = _safe_float(pending.get("min_key_attr_delta"), DEFAULT_MIN_KEY_ATTR_DELTA)
    heldout_max_drop = _safe_float(pending.get("heldout_max_drop"), DEFAULT_HELDOUT_MAX_DROP)
    key_attrs_ok = True if not focus_attrs else (focus_delta_mean >= min_key_attr_delta and focus_delta_min >= -0.004)
    heldout_proxy_ok = True if not heldout_attrs else (heldout_delta_mean >= -heldout_max_drop)

    accepted = (confidence_adjusted_delta >= min_delta) and key_attrs_ok and heldout_proxy_ok
    rollback_datasets = list(pending.get("baseline_datasets", []) or []) if not accepted else []
    if not rollback_datasets and not accepted:
        rollback_datasets = list(pending.get("original_datasets", []) or [])
    return {
        "status": "accepted" if accepted else "rejected",
        "gate_id": pending.get("gate_id"),
        "baseline_accuracy": baseline_accuracy,
        "current_accuracy": round(_safe_float(current_accuracy), 6),
        "required_delta": min_delta,
        "observed_delta": round(observed_delta, 6),
        "confidence_adjusted_delta": round(confidence_adjusted_delta, 6),
        "confidence_supported": confidence_supported,
        "confidence_z": confidence_z,
        "key_attrs_ok": key_attrs_ok,
        "focus_attributes": focus_attrs,
        "focus_delta_mean": round(focus_delta_mean, 6),
        "focus_delta_min": round(focus_delta_min, 6),
        "min_key_attr_delta": min_key_attr_delta,
        "heldout_proxy_ok": heldout_proxy_ok,
        "heldout_proxy_attributes": heldout_attrs,
        "heldout_proxy_delta_mean": round(heldout_delta_mean, 6),
        "heldout_max_drop": heldout_max_drop,
        "decision_ready": True,
        "rollback_datasets": rollback_datasets,
        "candidate_datasets": list(pending.get("candidate_datasets", []) or []),
    }
