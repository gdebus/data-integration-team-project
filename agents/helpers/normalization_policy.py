import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd


def infer_validation_text_case_map(validation_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Infers per-column lowercase preference from validation data."""
    case_map: Dict[str, Dict[str, Any]] = {}
    if validation_df is None or validation_df.empty:
        return case_map

    for col in validation_df.columns:
        series = validation_df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        non_null = series.dropna().astype(str).head(500)
        alpha_values: List[str] = []
        for v in non_null:
            text = str(v).strip()
            if text and re.search(r"[A-Za-z]", text):
                alpha_values.append(text)

        if not alpha_values:
            continue

        lower_like = sum(1 for v in alpha_values if v == v.lower())
        lower_ratio = lower_like / max(len(alpha_values), 1)
        prefer_lowercase = len(alpha_values) >= 8 and lower_ratio >= 0.85

        case_map[str(col)] = {
            "prefer_lowercase": prefer_lowercase,
            "lower_ratio": round(float(lower_ratio), 4),
            "sample_size": int(len(alpha_values)),
        }

    return case_map


def infer_country_output_format_from_validation(
    validation_df: Optional[pd.DataFrame],
    country_columns: Optional[List[str]] = None,
) -> str:
    """Infers target country output format from validation-set representation."""
    if validation_df is None or validation_df.empty:
        return "name"

    cols: List[str] = []
    if country_columns:
        cols = [c for c in country_columns if c in validation_df.columns]
    if not cols:
        cols = [str(c) for c in validation_df.columns if "country" in str(c).lower()]
    if not cols:
        return "name"

    values: List[str] = []
    for col in cols:
        series = validation_df[col].dropna().astype(str).head(500)
        for raw in series:
            text = str(raw).strip()
            if text:
                values.append(text)

    if not values:
        return "name"

    alpha2 = sum(1 for v in values if re.fullmatch(r"[A-Z]{2}", v) is not None)
    alpha3 = sum(1 for v in values if re.fullmatch(r"[A-Z]{3}", v) is not None)
    numeric = sum(1 for v in values if re.fullmatch(r"\d{3}", v) is not None)
    official = sum(
        1 for v in values if (" of " in v.lower() and len(v) > 18) or " and " in v.lower()
    )
    total = max(len(values), 1)

    if alpha2 / total >= 0.70:
        return "alpha_2"
    if alpha3 / total >= 0.70:
        return "alpha_3"
    if numeric / total >= 0.70:
        return "numeric"
    if official / total >= 0.55:
        return "official_name"
    return "name"


def collect_eval_debug_signals(
    debug_path: str = "output/pipeline_evaluation/debug_fusion_eval.jsonl",
    max_lines: int = 5000,
) -> Dict[str, Any]:
    """Reads evaluation debug events and derives normalization-sensitive signals."""
    signals: Dict[str, Any] = {
        "total_mismatch_events": 0,
        "case_only_mismatch_by_column": {},
        "country_near_mismatch_by_column": {},
    }
    if not os.path.exists(debug_path):
        return signals

    case_counts: Dict[str, int] = {}
    country_counts: Dict[str, int] = {}
    mismatch_counts: Dict[str, int] = {}

    def _norm_str(v: Any) -> str:
        text = str(v).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _is_case_only(a: Any, b: Any) -> bool:
        if a is None or b is None:
            return False
        if isinstance(a, list) and isinstance(b, list):
            aset = {str(x).strip().lower() for x in a if str(x).strip()}
            bset = {str(x).strip().lower() for x in b if str(x).strip()}
            if not aset and not bset:
                return False
            return aset == bset and set(map(str, a)) != set(map(str, b))
        sa = _norm_str(a)
        sb = _norm_str(b)
        if not sa or not sb:
            return False
        return sa.lower() == sb.lower() and sa != sb

    def _is_country_near_match(a: Any, b: Any) -> bool:
        sa = _norm_str(a).lower()
        sb = _norm_str(b).lower()
        if not sa or not sb:
            return False
        if sa == sb:
            return False
        atok = set(re.findall(r"[a-z0-9]+", sa))
        btok = set(re.findall(r"[a-z0-9]+", sb))
        if not atok or not btok:
            return False
        overlap = len(atok & btok) / max(len(atok | btok), 1)
        return overlap >= 0.45 or (sa in sb) or (sb in sa)

    with open(debug_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if idx >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue

            if str(event.get("reason", "")).lower() != "mismatch":
                continue
            col = str(event.get("attribute") or event.get("column") or "").strip()
            if not col:
                continue

            mismatch_counts[col] = mismatch_counts.get(col, 0) + 1
            signals["total_mismatch_events"] += 1

            fused_val = event.get("fused_value")
            expected_val = event.get("expected_value")
            if _is_case_only(fused_val, expected_val):
                case_counts[col] = case_counts.get(col, 0) + 1
            if "country" in col.lower() and _is_country_near_match(fused_val, expected_val):
                country_counts[col] = country_counts.get(col, 0) + 1

    case_ratio: Dict[str, float] = {}
    country_ratio: Dict[str, float] = {}
    for col, c in mismatch_counts.items():
        if c <= 0:
            continue
        if col in case_counts:
            case_ratio[col] = round(case_counts[col] / c, 4)
        if col in country_counts:
            country_ratio[col] = round(country_counts[col] / c, 4)

    signals["case_only_mismatch_by_column"] = case_ratio
    signals["country_near_mismatch_by_column"] = country_ratio
    return signals

