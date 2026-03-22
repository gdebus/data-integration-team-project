import json
from typing import Any, Dict, List, Tuple


def detect_normalization_issue(
    diagnostics_report: Dict[str, Any],
    evaluation_analysis: str,
    metrics: Dict[str, Any],
    debug_signals: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    text_blob = ""
    try:
        text_blob += json.dumps(diagnostics_report or {}, ensure_ascii=False).lower()
    except Exception:
        pass
    text_blob += "\n" + str(evaluation_analysis or "").lower()

    keyword_hits = [
        "normaliz",
        "canonical",
        "whitespace",
        "country code",
        "country canonical",
        "unit",
        "scale",
        "format mismatch",
        "standardiz",
        "list-like",
        "multi-valued",
        "json list",
        "array format",
        "case mismatch",
        "lowercase",
        "upper/lower",
    ]
    for kw in keyword_hits:
        if kw in text_blob:
            reasons.append(f"keyword:{kw}")

    critical_columns = {
        "name",
        "country",
        "city",
        "founded",
        "assets",
        "profits",
        "sales",
        "market_value",
        "keypeople_name",
    }
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if not (isinstance(key, str) and key.endswith("_accuracy")):
                continue
            col = key[: -len("_accuracy")]
            if col not in critical_columns:
                continue
            try:
                if float(value) < 0.45:
                    reasons.append(f"low_accuracy:{col}:{float(value):.3f}")
            except Exception:
                continue

    case_by_col = debug_signals.get("case_only_mismatch_by_column", {})
    if isinstance(case_by_col, dict):
        for col, ratio in case_by_col.items():
            try:
                if float(ratio) >= 0.50:
                    reasons.append(f"case_style_mismatch:{col}:{float(ratio):.3f}")
            except Exception:
                continue

    country_by_col = debug_signals.get("country_near_mismatch_by_column", {})
    if isinstance(country_by_col, dict):
        for col, ratio in country_by_col.items():
            try:
                if float(ratio) >= 0.40:
                    reasons.append(f"country_format_mismatch:{col}:{float(ratio):.3f}")
            except Exception:
                continue

    deduped = []
    seen = set()
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)

    return (len(deduped) > 0), deduped
