"""Metrics extraction, comparison, and regression detection utilities.

All functions are standalone (no class dependency) and can be used by
any pipeline component that needs to work with evaluation metrics.
"""

import ast
import json
from typing import Any, Dict, List, Tuple

from config import (
    REGRESSION_CATASTROPHIC,
    REGRESSION_MINOR,
    REGRESSION_MODERATE,
    REGRESSION_SEVERE,
)


def is_metrics_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if "overall_accuracy" in payload:
        return True
    acc_keys = [
        key
        for key in payload.keys()
        if isinstance(key, str) and key.endswith("_accuracy") and key not in {"overall_accuracy", "macro_accuracy"}
    ]
    return len(acc_keys) >= 2


def extract_metrics_payload(payload: Any) -> Dict[str, Any]:
    if is_metrics_payload(payload):
        return dict(payload)
    if not isinstance(payload, dict):
        return {}

    for key in (
        "evaluation_metrics",
        "metrics",
        "result",
        "results",
        "final_metrics",
        "pipeline_evaluation",
    ):
        nested = payload.get(key)
        if is_metrics_payload(nested):
            return dict(nested)

    for nested in payload.values():
        if isinstance(nested, dict):
            extracted = extract_metrics_payload(nested)
            if is_metrics_payload(extracted):
                return extracted
    return {}


def extract_metrics_from_text(text: Any) -> Dict[str, Any]:
    raw = str(text or "")
    if not raw.strip():
        return {}

    decoder = json.JSONDecoder()
    candidates: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(raw):
        start = raw.find("{", idx)
        if start == -1:
            break
        try:
            parsed, offset = decoder.raw_decode(raw[start:])
        except Exception:
            idx = start + 1
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)
        idx = start + max(1, int(offset))

    if not candidates:
        try:
            parsed = ast.literal_eval(raw.strip())
            if isinstance(parsed, dict):
                candidates.append(parsed)
        except Exception:
            pass

    def _score(candidate: Dict[str, Any]) -> Tuple[int, int, int]:
        acc_count = sum(1 for k in candidate.keys() if isinstance(k, str) and k.endswith("_accuracy"))
        return (
            1 if "overall_accuracy" in candidate else 0,
            acc_count,
            len(candidate),
        )

    ranked = sorted(candidates, key=_score, reverse=True)
    for candidate in ranked:
        extracted = extract_metrics_payload(candidate)
        if is_metrics_payload(extracted):
            return extracted
    return {}


def attribute_accuracy_map(metrics: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(metrics, dict):
        return {}
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if not (isinstance(key, str) and key.endswith("_accuracy")):
            continue
        if key in {"overall_accuracy", "macro_accuracy"}:
            continue
        attr = key[: -len("_accuracy")]
        try:
            count = int(float(metrics.get(f"{attr}_count", 0)))
        except Exception:
            count = 0
        if count <= 0:
            continue
        try:
            out[attr] = float(value)
        except Exception:
            continue
    return out


def assess_validation_regression(current: Dict[str, Any], best: Dict[str, Any]) -> Dict[str, Any]:
    if not (is_metrics_payload(current) and is_metrics_payload(best)):
        return {"rejected": False, "reason": "no_baseline"}

    current_overall = float(current.get("overall_accuracy", 0.0) or 0.0)
    best_overall = float(best.get("overall_accuracy", 0.0) or 0.0)
    current_macro = float(current.get("macro_accuracy", 0.0) or 0.0)
    best_macro = float(best.get("macro_accuracy", 0.0) or 0.0)

    # If macro_accuracy is missing/zero (evaluation didn't produce per-attribute
    # metrics), fall back to overall_accuracy for macro comparison to avoid
    # spurious regression rejection of a genuinely better pipeline.
    if current_macro == 0.0 and current_overall > 0.0 and "macro_accuracy" not in current:
        current_macro = current_overall
    if best_macro == 0.0 and best_overall > 0.0 and "macro_accuracy" not in best:
        best_macro = best_overall

    overall_gain = current_overall - best_overall
    macro_gain = current_macro - best_macro
    overall_regression = max(0.0, best_overall - current_overall)
    macro_regression = max(0.0, best_macro - current_macro)

    current_attr = attribute_accuracy_map(current)
    best_attr = attribute_accuracy_map(best)
    drops: Dict[str, float] = {}
    gains: Dict[str, float] = {}
    for attr, best_score in best_attr.items():
        if attr not in current_attr:
            continue
        try:
            c_count = int(float(current.get(f"{attr}_count", 0)))
            b_count = int(float(best.get(f"{attr}_count", 0)))
        except Exception:
            c_count = 0
            b_count = 0
        if min(c_count, b_count) < 3:
            continue
        delta = float(current_attr[attr]) - float(best_score)
        if delta < 0.0:
            drops[attr] = round(abs(delta), 6)
        elif delta > 0.0:
            gains[attr] = round(delta, 6)

    catastrophic_drops = {k: v for k, v in drops.items() if v >= REGRESSION_CATASTROPHIC}
    severe_drops = {k: v for k, v in drops.items() if v >= REGRESSION_SEVERE}
    moderate_drops = {k: v for k, v in drops.items() if v >= REGRESSION_MODERATE}
    meaningful_gain = overall_gain >= REGRESSION_MINOR or macro_gain >= REGRESSION_MINOR

    # Net-benefit calculation: total accuracy gained across all attributes vs lost.
    # This prevents rejecting a +22% overall jump because one attribute dropped 0.2.
    total_gain_weight = sum(gains.values()) if gains else 0.0
    total_drop_weight = sum(drops.values()) if drops else 0.0
    net_positive = total_gain_weight > total_drop_weight

    rejected = (
        overall_regression >= REGRESSION_MINOR
        or macro_regression >= REGRESSION_MINOR
        or (
            not meaningful_gain
            and (
                len(catastrophic_drops) >= 1
                or len(severe_drops) >= 1
                or len(moderate_drops) >= 2
            )
        )
        or (
            meaningful_gain
            and (
                # Extreme uncompensated drops still trigger rejection despite overall gain,
                # but allow single-attribute regressions when the net benefit is strongly positive.
                (len(catastrophic_drops) >= 2 and not net_positive)
                or (len(catastrophic_drops) >= 3)
                or (any(v >= 0.4 for v in catastrophic_drops.values()) and not net_positive)
                or (len(catastrophic_drops) >= 1 and not net_positive and not meaningful_gain)
            )
        )
    )
    return {
        "rejected": bool(rejected),
        "overall_gain": round(overall_gain, 6),
        "macro_gain": round(macro_gain, 6),
        "overall_drop": round(overall_regression, 6),
        "macro_drop": round(macro_regression, 6),
        "meaningful_gain": bool(meaningful_gain),
        "net_positive": bool(net_positive),
        "total_gain_weight": round(total_gain_weight, 6),
        "total_drop_weight": round(total_drop_weight, 6),
        "catastrophic_attribute_drops": catastrophic_drops,
        "severe_attribute_drops": severe_drops,
        "moderate_attribute_drops": moderate_drops,
        "top_attribute_drops": dict(sorted(drops.items(), key=lambda x: x[1], reverse=True)[:8]),
        "top_attribute_gains": dict(sorted(gains.items(), key=lambda x: x[1], reverse=True)[:8]),
    }


def is_metrics_better(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> bool:
    if not is_metrics_payload(candidate):
        return False
    if not is_metrics_payload(baseline):
        return True

    eps = 1e-6
    c_overall = float(candidate.get("overall_accuracy", 0.0) or 0.0)
    b_overall = float(baseline.get("overall_accuracy", 0.0) or 0.0)
    if c_overall > b_overall + eps:
        return True
    if c_overall < b_overall - eps:
        return False

    c_macro = float(candidate.get("macro_accuracy", 0.0) or 0.0)
    b_macro = float(baseline.get("macro_accuracy", 0.0) or 0.0)
    if c_macro > b_macro + eps:
        return True
    if c_macro < b_macro - eps:
        return False

    c_attr = attribute_accuracy_map(candidate)
    b_attr = attribute_accuracy_map(baseline)
    shared = [attr for attr in c_attr.keys() if attr in b_attr]
    if not shared:
        return False
    c_mean = sum(float(c_attr[attr]) for attr in shared) / max(1, len(shared))
    b_mean = sum(float(b_attr[attr]) for attr in shared) / max(1, len(shared))
    return c_mean > b_mean + eps
