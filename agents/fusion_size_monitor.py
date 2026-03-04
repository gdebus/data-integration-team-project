import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple


def estimate_path_for_output_dir(output_dir: str) -> str:
    root_dir = os.path.dirname(os.path.abspath(output_dir))
    return os.path.join(root_dir, "pipeline_evaluation", "fusion_size_estimate.json")


def estimate_from_blocking(
    dataset_sizes: Mapping[str, int],
    blocking_strategies: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    pair_estimates = _build_blocking_pair_estimates(
        dataset_sizes=dataset_sizes,
        blocking_strategies=blocking_strategies,
    )
    return _estimate_stage_payload(
        stage="blocking",
        dataset_sizes=dataset_sizes,
        pair_estimates=pair_estimates,
        fallback_method="mean(min_pair_dataset_size * pair_completeness) across dataset pairs",
    )


def estimate_from_matching(
    dataset_sizes: Mapping[str, int],
    blocking_strategies: Mapping[str, Any],
    matching_strategies: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    pair_estimates = _build_matching_pair_estimates(
        dataset_sizes=dataset_sizes,
        blocking_strategies=blocking_strategies,
        matching_strategies=matching_strategies,
    )
    return _estimate_stage_payload(
        stage="matching",
        dataset_sizes=dataset_sizes,
        pair_estimates=pair_estimates,
        fallback_method="mean(min_pair_dataset_size * pair_completeness * f1) across dataset pairs",
    )


def _build_blocking_pair_estimates(
    dataset_sizes: Mapping[str, int],
    blocking_strategies: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    pair_estimates: list[Dict[str, Any]] = []
    for pair_key, cfg in blocking_strategies.items():
        resolved = _resolve_pair_key(pair_key, dataset_sizes)
        if not resolved:
            continue
        left_name, right_name = resolved
        min_pair_size = min(int(dataset_sizes[left_name]), int(dataset_sizes[right_name]))
        pair_completeness = _clamp(_to_float(_get(cfg, "pair_completeness"), 0.0), 0.0, 1.0)
        expected_pair_clusters = int(round(min_pair_size * pair_completeness))
        pair_estimates.append(
            {
                "pair_key": pair_key,
                "left_dataset": left_name,
                "right_dataset": right_name,
                "min_pair_size": min_pair_size,
                "pair_completeness": pair_completeness,
                "expected_cluster_count": expected_pair_clusters,
            }
        )
    return pair_estimates


def _build_matching_pair_estimates(
    dataset_sizes: Mapping[str, int],
    blocking_strategies: Mapping[str, Any],
    matching_strategies: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    pair_estimates: list[Dict[str, Any]] = []
    for pair_key, cfg in matching_strategies.items():
        resolved = _resolve_pair_key(pair_key, dataset_sizes)
        if not resolved:
            continue
        left_name, right_name = resolved
        min_pair_size = min(int(dataset_sizes[left_name]), int(dataset_sizes[right_name]))
        f1_score = _clamp(_to_float(_get(cfg, "f1"), 0.0), 0.0, 1.0)
        blocking_cfg = _lookup_pair_config(blocking_strategies, left_name, right_name)
        pair_completeness = _clamp(_to_float(_get(blocking_cfg, "pair_completeness"), 1.0), 0.0, 1.0)
        quality_factor = max(0.05, f1_score)
        expected_pair_clusters = int(round(min_pair_size * pair_completeness * quality_factor))
        pair_estimates.append(
            {
                "pair_key": pair_key,
                "left_dataset": left_name,
                "right_dataset": right_name,
                "min_pair_size": min_pair_size,
                "pair_completeness": pair_completeness,
                "f1": f1_score,
                "expected_cluster_count": expected_pair_clusters,
            }
        )
    return pair_estimates


def _estimate_stage_payload(
    *,
    stage: str,
    dataset_sizes: Mapping[str, int],
    pair_estimates: list[Dict[str, Any]],
    fallback_method: str,
) -> Optional[Dict[str, Any]]:
    if not pair_estimates:
        return None

    three_way = _estimate_three_way_union(dataset_sizes, pair_estimates)
    if three_way:
        expected_rows = three_way["expected_union_clusters"]
        rows_min = three_way["expected_union_range"][0]
        rows_max = three_way["expected_union_range"][1]
        method = (
            "3-way inclusion-exclusion over pair estimates: "
            "E=E12+E13+E23-2*E123, "
            "with E123 estimated from per-dataset overlap probabilities"
        )
    else:
        expected_rows = int(round(sum(item["expected_cluster_count"] for item in pair_estimates) / len(pair_estimates)))
        rows_min, rows_max = _range_from_pair_estimates(pair_estimates, "expected_cluster_count")
        method = fallback_method

    payload = {
        "stage": stage,
        "dataset_signature": _dataset_signature(dataset_sizes),
        "datasets": sorted([str(k) for k in dataset_sizes.keys()]),
        "method": method,
        "expected_rows": expected_rows,
        "expected_unique_ids": expected_rows,
        "expected_rows_range": [rows_min, rows_max],
        "expected_unique_ids_range": [rows_min, rows_max],
        "three_way_intersection": three_way,
        "pair_estimates": pair_estimates,
        "created_at": _now_iso(),
    }
    return _attach_singleton_aware_projection(payload, dataset_sizes)


def upsert_stage_estimate(estimate_path: str, stage: str, stage_payload: Dict[str, Any]) -> Dict[str, Any]:
    doc = _read_json(estimate_path)
    doc.setdefault("estimates", {})
    doc.setdefault("comparisons", {})
    stage_signature = str(stage_payload.get("dataset_signature", "")).strip()
    if stage_signature:
        stale_stages = []
        for existing_stage, payload in doc.get("estimates", {}).items():
            if not isinstance(payload, dict):
                stale_stages.append(existing_stage)
                continue
            existing_signature = str(payload.get("dataset_signature", "")).strip()
            if existing_signature and existing_signature != stage_signature:
                stale_stages.append(existing_stage)
        for stale in stale_stages:
            doc["estimates"].pop(stale, None)
            doc["comparisons"].pop(stale, None)
        doc["active_dataset_signature"] = stage_signature
    doc["estimates"][stage] = stage_payload
    doc["updated_at"] = _now_iso()
    _write_json(estimate_path, doc)
    return doc


def compare_estimates_with_actual(
    fusion_csv_path: str,
    estimate_path: str,
) -> Dict[str, Any]:
    actual = _read_actual_fused_size(fusion_csv_path)
    doc = _read_json(estimate_path)
    estimates = doc.get("estimates", {}) if isinstance(doc, dict) else {}

    comparisons: Dict[str, Any] = {}
    for stage in ("blocking", "matching"):
        stage_estimate = estimates.get(stage)
        if not isinstance(stage_estimate, dict):
            continue

        expected_rows = _to_int(stage_estimate.get("expected_rows"), 0)
        expected_unique_ids = _to_int(stage_estimate.get("expected_unique_ids"), 0)
        expected_rows_matched_only = _to_int(
            stage_estimate.get("expected_rows_matched_only", expected_rows), expected_rows
        )
        expected_rows_singleton_aware = _to_int(
            stage_estimate.get("expected_rows_singleton_aware", expected_rows), expected_rows
        )

        rows_abs = abs(actual["rows"] - expected_rows)
        ids_abs = abs(actual["unique_ids"] - expected_unique_ids)
        rows_pct = _pct_error(rows_abs, expected_rows)
        ids_pct = _pct_error(ids_abs, expected_unique_ids)
        rows_pct_matched_only = _pct_error(
            abs(actual["rows"] - expected_rows_matched_only), expected_rows_matched_only
        )
        rows_pct_singleton_aware = _pct_error(
            abs(actual["rows"] - expected_rows_singleton_aware), expected_rows_singleton_aware
        )

        if rows_pct_matched_only is None and rows_pct_singleton_aware is None:
            better_variant = "unknown"
        elif rows_pct_singleton_aware is None:
            better_variant = "matched_only"
        elif rows_pct_matched_only is None:
            better_variant = "singleton_aware"
        else:
            better_variant = (
                "singleton_aware"
                if rows_pct_singleton_aware <= rows_pct_matched_only
                else "matched_only"
            )

        comparisons[stage] = {
            "expected_rows": expected_rows,
            "actual_rows": actual["rows"],
            "rows_abs_error": rows_abs,
            "rows_pct_error": rows_pct,
            "expected_rows_matched_only": expected_rows_matched_only,
            "expected_rows_singleton_aware": expected_rows_singleton_aware,
            "rows_pct_error_matched_only": rows_pct_matched_only,
            "rows_pct_error_singleton_aware": rows_pct_singleton_aware,
            "better_variant": better_variant,
            "expected_unique_ids": expected_unique_ids,
            "actual_unique_ids": actual["unique_ids"],
            "unique_ids_abs_error": ids_abs,
            "unique_ids_pct_error": ids_pct,
            "reasoning": _build_reasoning(
                stage=stage,
                expected_rows=expected_rows,
                actual_rows=actual["rows"],
                expected_unique_ids=expected_unique_ids,
                actual_unique_ids=actual["unique_ids"],
                rows_pct_error=rows_pct,
                unique_ids_pct_error=ids_pct,
            ),
            "compared_at": _now_iso(),
        }

    doc.setdefault("estimates", {})
    doc["actual"] = actual
    doc["comparisons"] = comparisons
    doc["updated_at"] = _now_iso()
    _write_json(estimate_path, doc)

    return {
        "estimate_path": estimate_path,
        "actual": actual,
        "comparisons": comparisons,
    }


def _read_actual_fused_size(fusion_csv_path: str) -> Dict[str, int]:
    if not os.path.exists(fusion_csv_path):
        raise FileNotFoundError(f"Fused output not found at: {fusion_csv_path}")

    row_count = 0
    unique_ids = set()

    with open(fusion_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        id_column = "_id" if reader.fieldnames and "_id" in reader.fieldnames else None
        for row in reader:
            row_count += 1
            if id_column:
                row_id = row.get(id_column)
                if row_id not in (None, "", "nan", "NaN"):
                    unique_ids.add(str(row_id))

    unique_count = len(unique_ids) if unique_ids else row_count
    return {"rows": row_count, "unique_ids": unique_count}


def _resolve_pair_key(pair_key: str, dataset_sizes: Mapping[str, int]) -> Optional[Tuple[str, str]]:
    names = list(dataset_sizes.keys())
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            if pair_key == f"{left_name}_{right_name}" or pair_key == f"{right_name}_{left_name}":
                return left_name, right_name
    return None


def _lookup_pair_config(strategies: Mapping[str, Any], left_name: str, right_name: str) -> Dict[str, Any]:
    direct = f"{left_name}_{right_name}"
    reverse = f"{right_name}_{left_name}"
    if direct in strategies and isinstance(strategies[direct], dict):
        return strategies[direct]
    if reverse in strategies and isinstance(strategies[reverse], dict):
        return strategies[reverse]
    return {}


def _range_from_pair_estimates(pair_estimates: Any, key: str) -> Tuple[int, int]:
    values = [int(item[key]) for item in pair_estimates if key in item]
    if not values:
        return 0, 0
    return min(values), max(values)


def _estimate_three_way_union(
    dataset_sizes: Mapping[str, int],
    pair_estimates: Any,
) -> Optional[Dict[str, Any]]:
    if len(dataset_sizes) != 3:
        return None

    names = list(dataset_sizes.keys())
    a, b, c = names[0], names[1], names[2]
    n_a, n_b, n_c = int(dataset_sizes[a]), int(dataset_sizes[b]), int(dataset_sizes[c])

    pair_counts: Dict[frozenset[str], int] = {}
    for item in pair_estimates:
        left = item.get("left_dataset")
        right = item.get("right_dataset")
        count = _to_int(item.get("expected_cluster_count"), 0)
        if not left or not right:
            continue
        pair_counts[frozenset((left, right))] = count

    required = [frozenset((a, b)), frozenset((a, c)), frozenset((b, c))]
    if not all(key in pair_counts for key in required):
        return None

    n_ab = pair_counts[frozenset((a, b))]
    n_ac = pair_counts[frozenset((a, c))]
    n_bc = pair_counts[frozenset((b, c))]

    min_pair = min(n_ab, n_ac, n_bc)

    # Per-dataset triple-overlap estimates under independence of pair coverage around each anchor dataset.
    e123_a = (n_ab * n_ac / n_a) if n_a > 0 else 0.0
    e123_b = (n_ab * n_bc / n_b) if n_b > 0 else 0.0
    e123_c = (n_ac * n_bc / n_c) if n_c > 0 else 0.0
    triple_candidates = sorted([e123_a, e123_b, e123_c])
    median_e123 = triple_candidates[1]

    e123 = int(round(_clamp(median_e123, 0.0, float(min_pair))))
    union_raw = n_ab + n_ac + n_bc - 2 * e123

    # include_singletons=False => each fused record contains at least 2 source records.
    max_clusters_by_records = (n_a + n_b + n_c) // 2
    union_low = max(0, n_ab + n_ac + n_bc - 2 * min_pair)
    union_high = min(n_ab + n_ac + n_bc, max_clusters_by_records)
    expected_union = _to_int(round(_clamp(float(union_raw), float(union_low), float(union_high))), 0)

    return {
        "enabled": True,
        "datasets": [a, b, c],
        "pair_cluster_estimates": {
            f"{a}_{b}": n_ab,
            f"{a}_{c}": n_ac,
            f"{b}_{c}": n_bc,
        },
        "triple_intersection_estimates": {
            f"via_{a}": e123_a,
            f"via_{b}": e123_b,
            f"via_{c}": e123_c,
            "median_used": median_e123,
            "final_triple_intersection": e123,
            "triple_intersection_range": [0, min_pair],
        },
        "expected_union_clusters": expected_union,
        "expected_union_range": [union_low, union_high],
        "formula": "E_union = E12 + E13 + E23 - 2*E123",
    }


def _build_reasoning(
    stage: str,
    expected_rows: int,
    actual_rows: int,
    expected_unique_ids: int,
    actual_unique_ids: int,
    rows_pct_error: Optional[float],
    unique_ids_pct_error: Optional[float],
) -> str:
    pct_candidates = [p for p in (rows_pct_error, unique_ids_pct_error) if p is not None]
    max_pct_error = max(pct_candidates) if pct_candidates else None

    if max_pct_error is None:
        base = "Expected size was zero; percentage error is not defined."
    elif max_pct_error <= 0.10:
        base = "Estimate is close to actual fused size."
    elif max_pct_error <= 0.30:
        base = "Estimate is moderately off; candidate generation and matching quality likely explain the gap."
    else:
        base = "Estimate deviates strongly; blocking/matching assumptions are likely not representative of fusion behavior."

    if stage == "blocking":
        if expected_rows > actual_rows:
            detail = "Blocking likely generated broader candidate space than what matching+fusion accepted."
        elif expected_rows < actual_rows:
            detail = "Blocking estimate was conservative; downstream matching/fusion retained more entities than expected."
        else:
            detail = "Blocking estimate aligns with fused output size."
    else:
        if expected_unique_ids > actual_unique_ids:
            detail = "Matching estimate seems optimistic; thresholds or fusion constraints likely reduced surviving clusters."
        elif expected_unique_ids < actual_unique_ids:
            detail = "Matching estimate seems conservative; matcher/fusion retained more clusters than projected."
        else:
            detail = "Matching estimate aligns with fused output size."

    return f"{base} {detail}"


def _attach_singleton_aware_projection(
    payload: Dict[str, Any],
    dataset_sizes: Mapping[str, int],
) -> Dict[str, Any]:
    """
    Produce a final-row estimate under include_singletons=True.
    This treats expected matched clusters as a conservative proxy for merge count
    (minimum one merge per matched cluster).
    """
    total_input_rows = sum(max(0, _to_int(v, 0)) for v in dataset_sizes.values())

    matched_only_rows = max(0, _to_int(payload.get("expected_rows"), 0))
    matched_only_low = 0
    matched_only_high = matched_only_rows
    existing_range = payload.get("expected_rows_range")
    if isinstance(existing_range, (list, tuple)) and len(existing_range) == 2:
        matched_only_low = max(0, _to_int(existing_range[0], 0))
        matched_only_high = max(matched_only_low, _to_int(existing_range[1], matched_only_rows))
    else:
        matched_only_low = matched_only_rows
        matched_only_high = matched_only_rows

    singleton_rows = max(0, total_input_rows - matched_only_rows)
    singleton_low = max(0, total_input_rows - matched_only_high)
    singleton_high = max(singleton_low, total_input_rows - matched_only_low)

    payload["expected_rows_matched_only"] = matched_only_rows
    payload["expected_unique_ids_matched_only"] = matched_only_rows
    payload["expected_rows_matched_only_range"] = [matched_only_low, matched_only_high]
    payload["expected_unique_ids_matched_only_range"] = [matched_only_low, matched_only_high]

    payload["expected_rows_singleton_aware"] = singleton_rows
    payload["expected_unique_ids_singleton_aware"] = singleton_rows
    payload["expected_rows_singleton_aware_range"] = [singleton_low, singleton_high]
    payload["expected_unique_ids_singleton_aware_range"] = [singleton_low, singleton_high]
    payload["singleton_assumption"] = {
        "assumed_include_singletons": True,
        "total_input_rows": total_input_rows,
        "formula": "E_final_rows = total_input_rows - E_matched_clusters",
        "notes": (
            "Conservative singleton-aware estimate: each expected matched cluster "
            "reduces the final row count by at least one."
        ),
    }

    # Use singleton-aware estimate as primary expected final fused size.
    payload["expected_rows"] = singleton_rows
    payload["expected_unique_ids"] = singleton_rows
    payload["expected_rows_range"] = [singleton_low, singleton_high]
    payload["expected_unique_ids_range"] = [singleton_low, singleton_high]
    payload["method"] = (
        f"{payload.get('method', '')}; "
        "singleton-aware projection applied for final fused output "
        "(include_singletons=True assumption)"
    ).strip("; ")
    return payload


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _pct_error(abs_error: int, expected: int) -> Optional[float]:
    if expected <= 0:
        return None
    return abs_error / expected


def _get(mapping: Any, key: str) -> Any:
    if isinstance(mapping, dict):
        return mapping.get(key)
    return None


def _dataset_signature(dataset_sizes: Mapping[str, int]) -> str:
    names = sorted(str(name) for name in dataset_sizes.keys())
    return "|".join(names)
