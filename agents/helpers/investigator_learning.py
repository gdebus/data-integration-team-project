import json
import os
from typing import Any, Dict, Optional, Tuple

LEARNING_CACHE_PATH = "output/profile/investigator_learning.json"
LEARNING_EMA_ALPHA = 0.35
LEARNING_GAIN_NORMALIZER = 0.05
LEARNING_MIN_OBSERVATIONS = 2
LEARNING_EFFECTIVE_GAIN_EPS = 0.005
LEARNING_SUCCESS_RATE_WEIGHT = 0.02
LEARNING_REGRET_MARGIN = 0.01
LEARNING_DRIFT_DECAY = 0.82


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def dataset_signature_from_state(state: Dict[str, Any]) -> str:
    names = sorted([os.path.splitext(os.path.basename(p))[0] for p in state.get("datasets", []) or []])
    return "|".join(names)


def read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.write("\n")
    except Exception:
        pass


def default_learning_state() -> Dict[str, Any]:
    return {
        "routing": {
            "normalization_node": {"count": 0, "ema_gain": 0.0, "wins": 0, "losses": 0, "last_gain": 0.0},
            "pipeline_adaption": {"count": 0, "ema_gain": 0.0, "wins": 0, "losses": 0, "last_gain": 0.0},
        },
        "pending_observation": {},
        "normalization_stall_streak": 0,
        "last_observation": {},
        "last_context_key": "",
        "drift_events": 0,
    }


def load_learning_state(state: Dict[str, Any]) -> Dict[str, Any]:
    signature = dataset_signature_from_state(state)
    if not signature:
        return default_learning_state()

    root = read_json(LEARNING_CACHE_PATH)
    datasets = root.get("datasets", {}) if isinstance(root, dict) else {}
    learning = datasets.get(signature, {})
    if not isinstance(learning, dict):
        learning = {}

    merged = default_learning_state()
    for key, value in learning.items():
        if key == "routing" and isinstance(value, dict):
            for route_key, route_val in value.items():
                if route_key in merged["routing"] and isinstance(route_val, dict):
                    merged["routing"][route_key].update(route_val)
        else:
            merged[key] = value
    return merged


def save_learning_state(state: Dict[str, Any], learning: Dict[str, Any]) -> None:
    signature = dataset_signature_from_state(state)
    if not signature:
        return

    root = read_json(LEARNING_CACHE_PATH)
    if not isinstance(root, dict):
        root = {}
    datasets = root.get("datasets")
    if not isinstance(datasets, dict):
        datasets = {}
    datasets[signature] = learning
    root["datasets"] = datasets
    write_json(LEARNING_CACHE_PATH, root)


def _route_bucket(learning: Dict[str, Any], route_name: str) -> Dict[str, Any]:
    try:
        bucket = learning.get("routing", {}).get(route_name, {})
        return {
            "count": int(bucket.get("count", 0)),
            "ema_gain": to_float(bucket.get("ema_gain"), 0.0),
            "wins": int(bucket.get("wins", 0)),
            "losses": int(bucket.get("losses", 0)),
            "last_gain": to_float(bucket.get("last_gain"), 0.0),
        }
    except Exception:
        return {"count": 0, "ema_gain": 0.0, "wins": 0, "losses": 0, "last_gain": 0.0}


def _route_effectiveness(bucket: Dict[str, Any]) -> float:
    wins = int(bucket.get("wins", 0))
    losses = int(bucket.get("losses", 0))
    success_rate = (wins + 1.0) / (wins + losses + 2.0)
    return to_float(bucket.get("ema_gain"), 0.0) + (LEARNING_SUCCESS_RATE_WEIGHT * (success_rate - 0.5))


def learning_routing_signals(learning: Dict[str, Any]) -> Dict[str, Any]:
    norm = _route_bucket(learning, "normalization_node")
    pipe = _route_bucket(learning, "pipeline_adaption")
    norm_effect = _route_effectiveness(norm)
    pipe_effect = _route_effectiveness(pipe)

    effect_delta = clamp(
        (norm_effect - pipe_effect) / max(LEARNING_GAIN_NORMALIZER, 1e-6),
        -1.0,
        1.0,
    )
    total_observations = norm["count"] + pipe["count"]
    confidence = clamp(total_observations / 8.0, 0.0, 1.0)
    learning_bias = effect_delta * confidence
    regret_active = (
        norm["count"] >= LEARNING_MIN_OBSERVATIONS
        and pipe["count"] >= LEARNING_MIN_OBSERVATIONS
        and (norm_effect + LEARNING_REGRET_MARGIN) < pipe_effect
    )
    recent_stall = bool(learning.get("last_observation", {}).get("decision") == "normalization_node") and (
        norm["last_gain"] <= LEARNING_EFFECTIVE_GAIN_EPS
    )
    return {
        "norm": norm,
        "pipe": pipe,
        "norm_effect": round(norm_effect, 6),
        "pipe_effect": round(pipe_effect, 6),
        "learning_bias": round(learning_bias, 6),
        "regret_active": regret_active,
        "recent_stall": recent_stall,
        "normalization_stall_streak": int(learning.get("normalization_stall_streak", 0)),
        "drift_events": int(learning.get("drift_events", 0)),
    }


def apply_learning_decay_for_drift(learning: Dict[str, Any], context_key: str) -> Dict[str, Any]:
    if not isinstance(learning, dict):
        learning = default_learning_state()
    context_key = str(context_key or "").strip()
    last_context = str(learning.get("last_context_key", "")).strip()
    if not context_key:
        return learning
    if not last_context:
        learning["last_context_key"] = context_key
        return learning
    if context_key == last_context:
        return learning

    routing = learning.get("routing", {})
    if isinstance(routing, dict):
        for route_name in ("normalization_node", "pipeline_adaption"):
            bucket = routing.get(route_name, {})
            if not isinstance(bucket, dict):
                continue
            bucket["ema_gain"] = round(float(to_float(bucket.get("ema_gain"), 0.0) * LEARNING_DRIFT_DECAY), 6)
            bucket["last_gain"] = round(float(to_float(bucket.get("last_gain"), 0.0) * LEARNING_DRIFT_DECAY), 6)
            bucket["count"] = int(max(0, round(int(bucket.get("count", 0)) * LEARNING_DRIFT_DECAY)))
            bucket["wins"] = int(max(0, round(int(bucket.get("wins", 0)) * LEARNING_DRIFT_DECAY)))
            bucket["losses"] = int(max(0, round(int(bucket.get("losses", 0)) * LEARNING_DRIFT_DECAY)))
            routing[route_name] = bucket
    learning["routing"] = routing
    learning["drift_events"] = int(learning.get("drift_events", 0)) + 1
    learning["last_context_key"] = context_key
    return learning


def update_learning_from_observation(
    learning: Dict[str, Any],
    *,
    current_accuracy: float,
    current_eval_attempt: int,
    dataset_signature: str,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if not isinstance(learning, dict):
        learning = default_learning_state()

    pending = learning.get("pending_observation", {})
    if not isinstance(pending, dict):
        return learning, None

    pending_decision = str(pending.get("decision", "")).strip()
    pending_accuracy = pending.get("accuracy")
    pending_eval_attempt = int(pending.get("evaluation_attempt", -1))
    pending_signature = str(pending.get("dataset_signature", "")).strip()
    if pending_decision not in {"normalization_node", "pipeline_adaption"}:
        return learning, None
    if pending_accuracy is None:
        return learning, None
    if pending_signature and dataset_signature and pending_signature != dataset_signature:
        learning["pending_observation"] = {}
        return learning, None
    if current_eval_attempt <= pending_eval_attempt:
        return learning, None

    gain = float(current_accuracy) - to_float(pending_accuracy, 0.0)
    route = learning.get("routing", {}).get(pending_decision, {})
    count = int(route.get("count", 0)) + 1
    prev_ema = to_float(route.get("ema_gain"), 0.0)
    ema_gain = gain if count == 1 else ((LEARNING_EMA_ALPHA * gain) + ((1.0 - LEARNING_EMA_ALPHA) * prev_ema))
    wins = int(route.get("wins", 0))
    losses = int(route.get("losses", 0))
    if gain > LEARNING_EFFECTIVE_GAIN_EPS:
        wins += 1
    elif gain < -LEARNING_EFFECTIVE_GAIN_EPS:
        losses += 1
    learning["routing"][pending_decision] = {
        "count": count,
        "ema_gain": round(float(ema_gain), 6),
        "wins": wins,
        "losses": losses,
        "last_gain": round(float(gain), 6),
    }
    if pending_decision == "normalization_node":
        streak = int(learning.get("normalization_stall_streak", 0))
        learning["normalization_stall_streak"] = streak + 1 if gain <= LEARNING_EFFECTIVE_GAIN_EPS else 0

    observation = {
        "decision": pending_decision,
        "before_accuracy": round(to_float(pending_accuracy, 0.0), 6),
        "after_accuracy": round(float(current_accuracy), 6),
        "gain": round(float(gain), 6),
        "before_eval_attempt": pending_eval_attempt,
        "after_eval_attempt": int(current_eval_attempt),
        "dataset_signature": dataset_signature,
    }
    learning["last_observation"] = observation
    learning["pending_observation"] = {}
    return learning, observation


def record_learning_checkpoint(
    learning: Dict[str, Any],
    *,
    decision: str,
    accuracy: float,
    evaluation_attempt: int,
    dataset_signature: str,
) -> Dict[str, Any]:
    if not isinstance(learning, dict):
        learning = default_learning_state()

    learning["pending_observation"] = {
        "decision": str(decision).strip(),
        "accuracy": round(float(accuracy), 6),
        "evaluation_attempt": int(evaluation_attempt),
        "dataset_signature": dataset_signature,
    }
    return learning
