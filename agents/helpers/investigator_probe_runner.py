import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, List

import config
from config import (
    CUSTOM_PROBE_TIMEOUT_SECONDS,
    MAX_CUSTOM_PROBES,
    MAX_PROBES,
    MISMATCH_SAMPLE_ATTRIBUTES,
    MISMATCH_SAMPLE_ROWS,
    PROBE_MAX_EVENTS,
    PROBE_MAX_RUNTIME_SECONDS,
    ROUTING_BLOCKING_CANDIDATE_MIN,
)
# NOTE: CORRESPONDENCES_DIR, DEBUG_EVAL_JSONL_PATH, FUSED_OUTPUT_PATH are
# accessed via config.X (not imported directly) because configure_run_output()
# mutates them at runtime.
from helpers.utils import to_float as _to_float

MAX_EVENTS = PROBE_MAX_EVENTS
MAX_RUNTIME_SECONDS = PROBE_MAX_RUNTIME_SECONDS


def _cap_words(text: str, limit: int = 28) -> str:
    words = str(text).strip().split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]) + "..."


def _probe_reason_distribution(state: Dict[str, Any]) -> Dict[str, Any]:
    ratios = (
        state.get("auto_diagnostics", {}).get("debug_reason_ratios", {})
        if isinstance(state.get("auto_diagnostics", {}), dict)
        else {}
    )
    if not isinstance(ratios, dict):
        ratios = {}
    sorted_items = sorted(ratios.items(), key=lambda kv: float(kv[1]), reverse=True)[:5]
    return {
        "name": "reason_distribution",
        "summary": ", ".join([f"{k}={float(v):.2f}" for k, v in sorted_items]) if sorted_items else "no debug reasons",
        "normalization_pressure": min(
            1.0,
            sum(
                float(v)
                for k, v in ratios.items()
                if any(x in str(k).lower() for x in ("mismatch", "list", "type", "format", "encoding"))
            ),
        ),
    }


def _probe_worst_attributes(state: Dict[str, Any]) -> Dict[str, Any]:
    metrics = (
        state.get("evaluation_metrics_for_adaptation", state.get("evaluation_metrics", {}))
        if isinstance(state, dict)
        else {}
    )
    rows: List[tuple[str, float]] = []
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if isinstance(key, str) and key.endswith("_accuracy"):
                rows.append((key, _to_float(value, 0.0)))
    rows.sort(key=lambda kv: kv[1])
    top = rows[:5]
    summary = ", ".join([f"{k}={v:.2f}" for k, v in top]) if top else "no attribute accuracies"
    actionability = min(1.0, max((1.0 - v) for _, v in top)) if top else 0.0
    return {
        "name": "worst_attributes",
        "summary": summary,
        "actionability_pressure": actionability,
        "normalization_pressure": min(1.0, actionability * 0.5),
    }


def _probe_recent_mismatches(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import config as _cfg
        path = os.path.join(_cfg.OUTPUT_DIR, "pipeline_evaluation/debug_fusion_eval.jsonl")
    except Exception:
        path = "output/pipeline_evaluation/debug_fusion_eval.jsonl"
    if not os.path.exists(path):
        return {"name": "recent_mismatches", "summary": "debug_fusion_eval missing", "normalization_pressure": 0.0}

    reasons = Counter()
    attrs = Counter()
    samples: deque[str] = deque(maxlen=3)
    seen = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if seen >= MAX_EVENTS:
                    break
                seen += 1
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if event.get("type") != "evaluation_mismatch":
                    continue
                reason = str(event.get("reason", "unknown")).strip()
                attr = str(event.get("attribute", "")).strip()
                if reason:
                    reasons[reason] += 1
                if attr:
                    attrs[attr] += 1
                sample = str(event.get("message", event.get("details", ""))).strip()
                if sample:
                    samples.append(_cap_words(sample))
    except Exception as e:
        return {"name": "recent_mismatches", "summary": f"read failed: {e}", "normalization_pressure": 0.0}

    reason_txt = ", ".join([f"{k}:{v}" for k, v in reasons.most_common(3)]) or "none"
    attr_txt = ", ".join([f"{k}:{v}" for k, v in attrs.most_common(3)]) or "none"
    norm_hits = 0
    for reason, count in reasons.items():
        if any(k in str(reason).lower() for k in ("format", "list", "type", "encoding", "mismatch")):
            norm_hits += count
    total = max(1, sum(reasons.values()))
    return {
        "name": "recent_mismatches",
        "summary": f"reasons[{reason_txt}] attrs[{attr_txt}]",
        "samples": list(samples),
        "normalization_pressure": min(1.0, norm_hits / total),
        "actionability_pressure": min(1.0, len(attrs) / 6.0),
    }


def _probe_directive_coverage(state: Dict[str, Any], load_dataset_fn: Callable[[str], Any]) -> Dict[str, Any]:
    directives = state.get("normalization_directives", {}) if isinstance(state, dict) else {}
    datasets = list(state.get("datasets", []) or [])
    if not isinstance(directives, dict) or not datasets:
        return {"name": "directive_coverage", "summary": "no directives or datasets", "normalization_pressure": 0.0}

    required_cols = set()
    for key in ("list_columns", "country_columns", "lowercase_columns", "text_columns"):
        values = directives.get(key, [])
        if isinstance(values, list):
            required_cols.update(str(v).strip().lower() for v in values if str(v).strip())
    if not required_cols:
        return {"name": "directive_coverage", "summary": "no directive columns", "normalization_pressure": 0.0}

    missing_by_dataset: Dict[str, List[str]] = defaultdict(list)
    for path in datasets[:4]:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            df = load_dataset_fn(path)
            cols = {str(c).strip().lower() for c in (df.columns.tolist() if df is not None else [])}
        except Exception:
            cols = set()
        missing = sorted([c for c in required_cols if c not in cols])
        if missing:
            missing_by_dataset[name].extend(missing[:8])

    missing_count = sum(len(v) for v in missing_by_dataset.values())
    summary = "all directive columns present"
    if missing_by_dataset:
        parts = [f"{k}: {', '.join(v[:4])}" for k, v in missing_by_dataset.items()]
        summary = "; ".join(parts)
    return {
        "name": "directive_coverage",
        "summary": summary,
        "missing_count": missing_count,
        "normalization_pressure": 0.1 if missing_count == 0 else 0.0,
        "actionability_pressure": min(1.0, missing_count / 10.0),
    }


def _probe_mismatch_sampler(state: Dict[str, Any]) -> Dict[str, Any]:
    """Samples concrete row-level mismatch examples grouped by worst attribute.

    Reads debug_fusion_eval.jsonl, groups mismatches by attribute, and
    returns top-N attributes with concrete {expected, fused, reason} examples.
    """
    import json as _json

    debug_path = config.DEBUG_EVAL_JSONL_PATH
    events = []
    try:
        with open(debug_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    events.append(_json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        return {"name": "mismatch_sampler", "summary": "no debug JSONL found", "samples_by_attribute": {}, "worst_attributes": []}
    except Exception as e:
        return {"name": "mismatch_sampler", "summary": f"error reading debug JSONL: {e}", "samples_by_attribute": {}, "worst_attributes": []}

    # Group by attribute
    attr_mismatches: Dict[str, list] = {}
    for evt in events:
        if not isinstance(evt, dict):
            continue
        reason = str(evt.get("reason", "")).lower()
        if reason not in ("mismatch", "missing_fused_value", "format_mismatch", "case_mismatch"):
            # Falls through to type-based mismatch detection
            if evt.get("type") != "evaluation_mismatch":
                continue

        attr = str(evt.get("attribute", evt.get("column", "unknown")))
        expected = evt.get("expected_value", evt.get("gold_value", evt.get("test_value", "")))
        fused = evt.get("fused_value", evt.get("predicted_value", ""))
        fused_id = evt.get("fused_id", evt.get("entity_id", ""))

        # Sub-classification into specific reason category
        classified_reason = _classify_mismatch_reason(expected, fused, reason)

        if attr not in attr_mismatches:
            attr_mismatches[attr] = []
        attr_mismatches[attr].append({
            "expected": str(expected)[:200] if expected is not None else "",
            "fused": str(fused)[:200] if fused is not None else "",
            "reason": classified_reason,
            "fused_id": str(fused_id)[:50] if fused_id else "",
        })

    if not attr_mismatches:
        return {"name": "mismatch_sampler", "summary": "no mismatches found in debug JSONL", "samples_by_attribute": {}, "worst_attributes": []}

    # Top N attributes by mismatch count
    sorted_attrs = sorted(attr_mismatches.keys(), key=lambda a: len(attr_mismatches[a]), reverse=True)
    worst_attrs = sorted_attrs[:MISMATCH_SAMPLE_ATTRIBUTES]

    # Row samples per attribute
    samples_by_attribute = {}
    total_sampled = 0
    norm_pressure_count = 0
    total_count = 0
    for attr in worst_attrs:
        all_examples = attr_mismatches[attr]
        sampled = all_examples[:MISMATCH_SAMPLE_ROWS]
        samples_by_attribute[attr] = sampled
        total_sampled += len(sampled)
        total_count += len(all_examples)
        norm_pressure_count += sum(1 for ex in all_examples if ex["reason"] in ("case_mismatch", "format_mismatch", "country_format"))

    normalization_pressure = norm_pressure_count / max(total_count, 1)

    return {
        "name": "mismatch_sampler",
        "summary": f"sampled {total_sampled} examples across {len(worst_attrs)} worst attributes (of {len(attr_mismatches)} total with mismatches)",
        "samples_by_attribute": samples_by_attribute,
        "worst_attributes": worst_attrs,
        "attribute_mismatch_counts": {a: len(attr_mismatches[a]) for a in worst_attrs},
        "normalization_pressure": round(normalization_pressure, 3),
    }


def _classify_mismatch_reason(expected, fused, raw_reason: str) -> str:
    """Sub-classifies a mismatch into a specific reason category."""
    if fused is None or (isinstance(fused, str) and not fused.strip()) or str(fused).lower() in ("nan", "none", ""):
        return "missing_fused"

    exp_str = str(expected).strip() if expected is not None else ""
    fused_str = str(fused).strip()

    if not exp_str:
        return "missing_expected"

    # Case-only difference
    if exp_str.lower() == fused_str.lower():
        return "case_mismatch"

    # Whitespace/strip difference
    if exp_str.lower().replace(" ", "") == fused_str.lower().replace(" ", ""):
        return "whitespace_mismatch"

    # Substring containment (partial match / format difference)
    if exp_str.lower() in fused_str.lower() or fused_str.lower() in exp_str.lower():
        return "format_mismatch"

    # List-like format discrepancy
    if (exp_str.startswith("[") or fused_str.startswith("[")):
        return "list_format_mismatch"

    # Country-like format differences (short alpha code vs full name)
    if len(exp_str) <= 3 and exp_str.isalpha() and len(fused_str) > 3:
        return "country_format"
    if len(fused_str) <= 3 and fused_str.isalpha() and len(exp_str) > 3:
        return "country_format"

    return "value_mismatch"


def _probe_null_patterns(state: Dict[str, Any]) -> Dict[str, Any]:
    """Detects columns that are heavily null in the fused output."""
    import pandas as pd

    fused_path = state.get("fused_output_path", config.FUSED_OUTPUT_PATH)
    try:
        df = pd.read_csv(fused_path, nrows=500)  # Sample for speed
    except Exception as e:
        return {"name": "null_patterns", "summary": f"could not read fused output: {e}", "null_heavy_columns": []}

    total = len(df)
    if total == 0:
        return {"name": "null_patterns", "summary": "fused output is empty", "null_heavy_columns": []}

    null_info = []
    for col in df.columns:
        if col.startswith("_fusion_"):
            continue
        null_count = int(df[col].isna().sum())
        null_ratio = null_count / total
        if null_ratio > 0.5:  # More than 50% null
            null_info.append({"column": col, "null_ratio": round(null_ratio, 3), "null_count": null_count, "total": total})

    null_info.sort(key=lambda x: x["null_ratio"], reverse=True)

    return {
        "name": "null_patterns",
        "summary": f"{len(null_info)} columns with >50% nulls out of {len([c for c in df.columns if not c.startswith('_fusion_')])} total",
        "null_heavy_columns": null_info[:10],
        "fused_row_count": total,
        "total_columns": len(df.columns),
    }


def _probe_correspondence_density(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyzes correspondence file density and matching patterns."""
    import pandas as pd
    import glob as _glob

    corr_dir = config.CORRESPONDENCES_DIR
    corr_files = sorted(_glob.glob(os.path.join(corr_dir, "correspondences_*.csv")))

    if not corr_files:
        return {"name": "correspondence_density", "summary": "no correspondence files found", "pairs": [], "empty_pairs": []}

    pairs = []
    empty_pairs = []
    for path in corr_files:
        pair_name = os.path.splitext(os.path.basename(path))[0].replace("correspondences_", "")
        try:
            df = pd.read_csv(path)
            row_count = len(df)
            if row_count == 0:
                empty_pairs.append(pair_name)
                pairs.append({"pair": pair_name, "rows": 0, "unique_id1": 0, "unique_id2": 0, "ratio": "empty"})
                continue

            id1_col = "id1" if "id1" in df.columns else df.columns[0]
            id2_col = "id2" if "id2" in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
            unique_id1 = int(df[id1_col].nunique())
            unique_id2 = int(df[id2_col].nunique())

            # Ratio > 1 indicates 1:many matching
            ratio_1 = round(row_count / max(unique_id1, 1), 2)
            ratio_2 = round(row_count / max(unique_id2, 1), 2)

            pairs.append({
                "pair": pair_name,
                "rows": row_count,
                "unique_id1": unique_id1,
                "unique_id2": unique_id2,
                "avg_matches_per_entity": round(max(ratio_1, ratio_2), 2),
                "pattern": "1:1" if max(ratio_1, ratio_2) <= 1.1 else "1:many",
            })
        except Exception as e:
            pairs.append({"pair": pair_name, "rows": -1, "error": str(e)[:100]})

    return {
        "name": "correspondence_density",
        "summary": f"{len(corr_files)} correspondence files, {len(empty_pairs)} empty",
        "pairs": pairs,
        "empty_pairs": empty_pairs,
        "has_empty_pairs": len(empty_pairs) > 0,
    }


def _probe_blocking_recall(state: Dict[str, Any]) -> Dict[str, Any]:
    """Checks whether blocking produced enough candidates relative to dataset sizes."""
    blocking_config = state.get("blocking_config", {})
    if not isinstance(blocking_config, dict) or not blocking_config:
        return {"name": "blocking_recall", "summary": "no blocking config available", "assessment": "unknown"}

    # Candidate counts from blocking config
    strategies = blocking_config.get("blocking_strategies", {})
    if not isinstance(strategies, dict):
        return {"name": "blocking_recall", "summary": "no blocking strategies found", "assessment": "unknown"}

    pair_assessments = []
    for pair_name, pair_cfg in strategies.items():
        if not isinstance(pair_cfg, dict):
            continue
        candidates = int(pair_cfg.get("num_candidates", pair_cfg.get("candidate_count", 0)) or 0)
        pc = float(pair_cfg.get("pair_completeness", pair_cfg.get("pc", 0.0)) or 0.0)

        assessment = "ok"
        if candidates == 0:
            assessment = "no_candidates"
        elif candidates < ROUTING_BLOCKING_CANDIDATE_MIN:
            assessment = "too_few_candidates"
        elif pc < 0.5:
            assessment = "low_pair_completeness"

        pair_assessments.append({
            "pair": str(pair_name),
            "candidates": candidates,
            "pair_completeness": round(pc, 3) if pc else None,
            "assessment": assessment,
        })

    problem_pairs = [p for p in pair_assessments if p["assessment"] != "ok"]

    return {
        "name": "blocking_recall",
        "summary": f"{len(pair_assessments)} pairs assessed, {len(problem_pairs)} with issues",
        "pair_assessments": pair_assessments,
        "problem_pairs": problem_pairs,
        "blocking_looks_broken": len(problem_pairs) > 0,
    }


def _probe_fusion_size(state: Dict[str, Any]) -> Dict[str, Any]:
    """Checks whether the fused output size matches blocking/matching estimates."""
    size_comparison = state.get("fusion_size_comparison", {})
    if not isinstance(size_comparison, dict) or not size_comparison:
        return {"name": "fusion_size", "summary": "no fusion size comparison available"}

    comparisons = size_comparison.get("comparisons", {})
    if not isinstance(comparisons, dict) or not comparisons:
        return {"name": "fusion_size", "summary": "no stage comparisons available"}

    actual = size_comparison.get("actual", {})
    actual_rows = int(actual.get("rows", 0)) if isinstance(actual, dict) else 0

    stage_summaries = []
    worst_pct_error = 0.0
    direction = "ok"  # "overcount", "undercount", or "ok"

    for stage_name in ("matching", "blocking"):
        comp = comparisons.get(stage_name)
        if not isinstance(comp, dict):
            continue
        expected = int(comp.get("expected_rows", 0))
        pct_error = _to_float(comp.get("rows_pct_error"), 0.0)
        reasoning = str(comp.get("reasoning", ""))

        if pct_error > worst_pct_error:
            worst_pct_error = pct_error
            if actual_rows > expected and expected > 0:
                direction = "overcount"
            elif actual_rows < expected and expected > 0:
                direction = "undercount"

        stage_summaries.append(
            f"{stage_name}: expected={expected} actual={actual_rows} error={pct_error:.0%}"
        )

    # Classify severity
    if worst_pct_error <= 0.15:
        severity = "ok"
        assessment = "fusion size within expected range"
    elif worst_pct_error <= 0.35:
        severity = "moderate"
        assessment = f"fusion size moderately off ({direction})"
    elif worst_pct_error <= 0.60:
        severity = "significant"
        assessment = f"fusion size significantly off ({direction})"
    else:
        severity = "severe"
        assessment = f"fusion size severely off ({direction})"

    # Compute pressure signals
    # Size issues suggest blocking/matching problems, not normalization
    actionability = min(1.0, worst_pct_error * 1.5) if worst_pct_error > 0.15 else 0.0

    return {
        "name": "fusion_size",
        "summary": f"{assessment}; {'; '.join(stage_summaries)}",
        "severity": severity,
        "direction": direction,
        "worst_pct_error": round(worst_pct_error, 4),
        "actual_rows": actual_rows,
        "stage_details": comparisons,
        "normalization_pressure": 0.0,  # Size issues are not normalization problems
        "actionability_pressure": round(actionability, 4),
    }


def _probe_attribute_improvability(state: Dict[str, Any], accumulated_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Classifies each attribute as improvable, structurally_limited, or at_ceiling.

    Uses mismatch reason classifications to estimate an accuracy ceiling per
    attribute and computes weighted impact scores so the investigation LLM
    can prioritise effort on attributes where improvement is actually possible.
    """
    metrics = (
        state.get("evaluation_metrics_for_adaptation", state.get("evaluation_metrics", {}))
        if isinstance(state, dict) else {}
    )
    if not isinstance(metrics, dict):
        return {"name": "attribute_improvability", "summary": "no metrics", "attribute_classifications": {}}

    # Gather per-attribute accuracies (skip aggregate metrics)
    SKIP_PREFIXES = ("overall", "macro", "mean", "weighted")
    attr_accs: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(key, str) and key.endswith("_accuracy"):
            if any(key.startswith(p) for p in SKIP_PREFIXES):
                continue
            attr_accs[key] = _to_float(value, 0.0)

    if not attr_accs:
        return {"name": "attribute_improvability", "summary": "no per-attribute accuracies", "attribute_classifications": {}}

    # Get mismatch reason classifications from mismatch_sampler probe
    mismatch_data: Dict[str, list] = {}
    mismatch_counts: Dict[str, int] = {}

    # Try accumulated results first (same run), then state (previous run)
    if accumulated_results:
        for r in accumulated_results:
            if isinstance(r, dict) and r.get("name") == "mismatch_sampler":
                mismatch_data = r.get("samples_by_attribute", {})
                mismatch_counts = r.get("attribute_mismatch_counts", {})
                break

    if not mismatch_data:
        probe_results = state.get("investigator_probe_results", {})
        if isinstance(probe_results, dict):
            for r in probe_results.get("results", []):
                if isinstance(r, dict) and r.get("name") == "mismatch_sampler":
                    mismatch_data = r.get("samples_by_attribute", {})
                    mismatch_counts = r.get("attribute_mismatch_counts", {})
                    break

    # Fixable mismatch reasons (can be resolved by normalization or fusion changes)
    FIXABLE_REASONS = {"case_mismatch", "whitespace_mismatch", "format_mismatch", "country_format", "missing_fused"}
    # Structural reasons (source data fundamentally disagrees)
    STRUCTURAL_REASONS = {"value_mismatch", "list_format_mismatch"}

    classifications: Dict[str, Dict[str, Any]] = {}
    for attr, acc in attr_accs.items():
        # Analyze mismatch reasons for this attribute
        # Try matching by attribute name (with and without _accuracy suffix)
        attr_base = attr.replace("_accuracy", "")
        samples = mismatch_data.get(attr_base, mismatch_data.get(attr, []))

        fixable_count = 0
        structural_count = 0
        total_mismatches = 0
        reason_breakdown: Dict[str, int] = {}

        for sample in samples:
            if not isinstance(sample, dict):
                continue
            reason = sample.get("reason", "value_mismatch")
            reason_breakdown[reason] = reason_breakdown.get(reason, 0) + 1
            total_mismatches += 1
            if reason in FIXABLE_REASONS:
                fixable_count += 1
            elif reason in STRUCTURAL_REASONS:
                structural_count += 1

        # Use total mismatch count from probe if available (more complete than samples)
        full_count = mismatch_counts.get(attr_base, mismatch_counts.get(attr, total_mismatches))

        # Classify
        remaining_gap = 1.0 - acc
        if acc >= 0.90:
            cls = "at_ceiling"
            fixable_fraction = 0.0
            ceiling = acc
            reasons_list = []
        elif total_mismatches > 0:
            fixable_fraction = fixable_count / max(total_mismatches, 1)
            structural_fraction = structural_count / max(total_mismatches, 1)

            if structural_fraction >= 0.5:
                cls = "structurally_limited"
                ceiling = acc + (fixable_fraction * remaining_gap)
                reasons_list = [r for r, c in sorted(reason_breakdown.items(), key=lambda x: -x[1])
                                if r in STRUCTURAL_REASONS][:3]
            else:
                cls = "improvable"
                ceiling = acc + (fixable_fraction * remaining_gap) + (0.5 * (1 - fixable_fraction) * remaining_gap)
                reasons_list = [r for r, c in sorted(reason_breakdown.items(), key=lambda x: -x[1])
                                if r in FIXABLE_REASONS][:3]
        else:
            # No mismatch samples — assume improvable if accuracy is low
            cls = "improvable" if acc < 0.85 else "at_ceiling"
            fixable_fraction = 0.7 if cls == "improvable" else 0.0
            ceiling = min(1.0, acc + 0.7 * remaining_gap) if cls == "improvable" else acc
            reasons_list = []

        classifications[attr] = {
            "class": cls,
            "accuracy": round(acc, 4),
            "ceiling": round(min(1.0, ceiling), 4),
            "fixable_fraction": round(fixable_fraction, 3),
            "reasons": reasons_list,
            "mismatch_count": full_count,
        }

    # Compute impact ranking (uniform weight — macro accuracy)
    n_attrs = max(len(classifications), 1)
    weight = 1.0 / n_attrs
    impact_ranking = []
    for attr, info in classifications.items():
        impact = weight * (info["ceiling"] - info["accuracy"])
        impact_ranking.append({
            "attribute": attr,
            "impact": round(impact, 4),
            "class": info["class"],
            "accuracy": info["accuracy"],
            "ceiling": info["ceiling"],
        })
    impact_ranking.sort(key=lambda x: x["impact"], reverse=True)

    # Structural ceiling: overall accuracy if all improvable attributes hit their ceiling
    structural_ceiling = sum(info["ceiling"] for info in classifications.values()) / n_attrs
    current_overall = sum(info["accuracy"] for info in classifications.values()) / n_attrs
    improvable_headroom = structural_ceiling - current_overall

    # Summary
    by_class = Counter(info["class"] for info in classifications.values())
    summary_parts = []
    for cls in ("improvable", "structurally_limited", "at_ceiling"):
        if by_class.get(cls):
            summary_parts.append(f"{by_class[cls]} {cls}")
    summary = ", ".join(summary_parts) + f" | ceiling≈{structural_ceiling:.1%}, headroom≈{improvable_headroom:.1%}"

    return {
        "name": "attribute_improvability",
        "summary": summary,
        "attribute_classifications": classifications,
        "impact_ranking": impact_ranking[:10],
        "structural_ceiling_estimate": round(structural_ceiling, 4),
        "improvable_headroom": round(improvable_headroom, 4),
        "normalization_pressure": 0.0,
        "actionability_pressure": 0.0,
    }


def _probe_source_attribution(state: Dict[str, Any]) -> Dict[str, Any]:
    """Per-source match rates against the validation set for low-accuracy attributes.

    For each low-accuracy attribute, loads the fused output metadata to identify
    which source contributed each value, then compares every source's value
    against the validation expected value.  Returns per-source exact/fuzzy match
    rates so the investigator can recommend the right resolver without writing
    diagnostic code.
    """
    import ast
    import pandas as pd

    from config import SOURCE_ATTRIBUTION_MAX_RECORDS

    # --- 1. Identify low-accuracy attributes ---
    metrics = state.get("evaluation_metrics_for_adaptation", state.get("evaluation_metrics", {}))
    if not isinstance(metrics, dict):
        return {"name": "source_attribution", "summary": "no metrics", "per_attribute": {}}

    attr_accs: List[tuple] = []
    for key, val in metrics.items():
        if key.endswith("_accuracy") and not key.startswith("overall") and not key.startswith("macro"):
            attr_name = key.replace("_accuracy", "")
            attr_accs.append((attr_name, _to_float(val, 1.0)))
    attr_accs.sort(key=lambda x: x[1])
    # Attributes below 80% accuracy — meaningful room for improvement
    # (attributes at 80-85% are near-ceiling and shouldn't be aggressively changed)
    target_attrs_unfiltered = [name for name, acc in attr_accs if acc < 0.80]
    if not target_attrs_unfiltered:
        return {"name": "source_attribution", "summary": "no low-accuracy attributes", "per_attribute": {}}
    target_attrs = target_attrs_unfiltered  # narrowed after validation load

    # --- 2. Load validation set ---
    validation_path = state.get("validation_fusion_testset") or state.get("fusion_testset")
    if not validation_path or not os.path.exists(str(validation_path)):
        return {"name": "source_attribution", "summary": "validation set not found", "per_attribute": {}}

    try:
        ext = os.path.splitext(validation_path)[1].lower()
        if ext == ".csv":
            val_df = pd.read_csv(validation_path)
        elif ext == ".xml":
            val_df = pd.read_xml(validation_path)
        elif ext in (".parquet", ".pq"):
            val_df = pd.read_parquet(validation_path)
        else:
            return {"name": "source_attribution", "summary": f"unsupported validation format: {ext}", "per_attribute": {}}
    except Exception as e:
        return {"name": "source_attribution", "summary": f"validation load failed: {e}", "per_attribute": {}}

    # Filter target attrs to those present in the validation set, then cap at 5
    val_columns = set(val_df.columns)
    target_attrs = [a for a in target_attrs if a in val_columns][:5]
    if not target_attrs:
        return {"name": "source_attribution", "summary": "no low-accuracy attributes match validation columns", "per_attribute": {}}

    val_by_id = {}
    id_col = "id" if "id" in val_df.columns else val_df.columns[0]
    for _, row in val_df.iterrows():
        val_by_id[str(row[id_col])] = row

    # --- 3. Load fused output with metadata ---
    fused_path = config.FUSED_OUTPUT_PATH
    if not os.path.exists(fused_path):
        return {"name": "source_attribution", "summary": "fused output not found", "per_attribute": {}}

    try:
        fused_df = pd.read_csv(fused_path)
    except Exception as e:
        return {"name": "source_attribution", "summary": f"fused load failed: {e}", "per_attribute": {}}

    has_metadata = "_fusion_metadata" in fused_df.columns
    has_sources = "_fusion_sources" in fused_df.columns

    if not has_metadata and not has_sources:
        return {"name": "source_attribution", "summary": "no _fusion_metadata or _fusion_sources columns", "per_attribute": {}}

    # --- 4. Parse source IDs to find validation-matching fused records ---
    from helpers.evaluation import parse_source_ids

    # Detect the validation ID prefix (e.g. "mbrainz_")
    val_ids = set(str(v) for v in val_by_id.keys())
    val_prefix = ""
    if val_ids:
        sample_id = next(iter(val_ids))
        parts = sample_id.split("_", 1)
        if len(parts) == 2:
            val_prefix = parts[0] + "_"

    # Map fused rows to validation IDs
    fused_to_val: Dict[int, str] = {}  # fused_df index → validation ID
    for idx, row in fused_df.iterrows():
        # Try direct ID match first
        fused_id = str(row.get("_id", row.get("id", "")))
        if fused_id in val_ids:
            fused_to_val[idx] = fused_id
            continue
        # Try via _fusion_sources
        if has_sources:
            source_ids = parse_source_ids(row.get("_fusion_sources"))
            for sid in source_ids:
                if str(sid) in val_ids:
                    fused_to_val[idx] = str(sid)
                    break

    if not fused_to_val:
        return {"name": "source_attribution", "summary": "no fused records map to validation set", "per_attribute": {}}

    # --- 5. Per-attribute source comparison ---
    per_attribute: Dict[str, Any] = {}

    for attr in target_attrs:
        if attr not in val_df.columns:
            continue

        source_stats: Dict[str, Dict[str, Any]] = {}  # dataset_name → stats
        records_analyzed = 0

        for fused_idx, val_id in fused_to_val.items():
            val_row = val_by_id.get(val_id)
            if val_row is None:
                continue

            expected = val_row.get(attr)
            if expected is None or (isinstance(expected, float) and pd.isna(expected)):
                continue

            expected_str = str(expected).strip()
            records_analyzed += 1

            # Extract per-source values from _fusion_metadata
            fused_row = fused_df.loc[fused_idx]
            source_values: Dict[str, str] = {}  # dataset → value

            if has_metadata:
                try:
                    meta_raw = fused_row.get("_fusion_metadata", "")
                    if isinstance(meta_raw, str) and meta_raw.strip() and len(meta_raw) < 50000:
                        # Metadata is Python repr (single-quoted dicts, may contain
                        # numpy calls like np.float64(...)).  Extract per-attribute
                        # inputs via targeted regex when full parsing fails.
                        meta = None
                        try:
                            meta = json.loads(meta_raw)
                        except (json.JSONDecodeError, ValueError):
                            pass
                        if meta is None:
                            try:
                                meta = ast.literal_eval(meta_raw)
                            except (ValueError, SyntaxError):
                                pass
                        if meta is None:
                            # Regex fallback: extract attr_inputs blocks
                            import re
                            pattern = re.escape(f"'{attr}_inputs'") + r"\s*:\s*(\[.*?\])\s*[,}]"
                            m = re.search(pattern, meta_raw)
                            if m:
                                try:
                                    attr_inputs_raw = m.group(1)
                                    # Sanitize numpy calls: np.float64(X) → X
                                    attr_inputs_raw = re.sub(r"np\.\w+\((.*?)\)", r"\1", attr_inputs_raw)
                                    meta = {f"{attr}_inputs": ast.literal_eval(attr_inputs_raw)}
                                except Exception:
                                    pass
                        if isinstance(meta, dict):
                            inputs_key = f"{attr}_inputs"
                            attr_inputs = meta.get(inputs_key, [])
                            if isinstance(attr_inputs, list):
                                for inp in attr_inputs:
                                    if isinstance(inp, dict):
                                        ds = str(inp.get("dataset", "")).strip()
                                        val_v = inp.get("value")
                                        if ds:
                                            source_values[ds] = str(val_v).strip() if val_v is not None and not (isinstance(val_v, float) and pd.isna(val_v)) else ""
                except Exception:
                    pass

            # Fallback: if metadata didn't yield sources, try source-specific columns
            if not source_values and has_sources:
                source_ids = parse_source_ids(fused_row.get("_fusion_sources"))
                source_datasets = fused_row.get("_fusion_source_datasets")
                if isinstance(source_datasets, str):
                    try:
                        source_datasets = ast.literal_eval(source_datasets)
                    except Exception:
                        source_datasets = []
                if isinstance(source_datasets, list) and len(source_datasets) == len(source_ids):
                    fused_val = fused_row.get(attr)
                    fused_str = str(fused_val).strip() if fused_val is not None and not (isinstance(fused_val, float) and pd.isna(fused_val)) else ""
                    for ds in source_datasets:
                        source_values[str(ds)] = fused_str  # limited: we only know the fused value, not per-source

            # Compare each source value against expected
            for ds_name, src_val in source_values.items():
                if ds_name not in source_stats:
                    source_stats[ds_name] = {
                        "exact_matches": 0, "fuzzy_matches": 0,
                        "non_null": 0, "total": 0,
                        "sample_values": [],
                    }
                stats = source_stats[ds_name]
                stats["total"] += 1

                is_null = not src_val or src_val.lower() in ("nan", "none", "")
                if not is_null:
                    stats["non_null"] += 1
                    if src_val == expected_str:
                        stats["exact_matches"] += 1
                        stats["fuzzy_matches"] += 1
                    elif src_val.lower().strip() == expected_str.lower().strip():
                        stats["fuzzy_matches"] += 1
                    elif expected_str.lower() in src_val.lower() or src_val.lower() in expected_str.lower():
                        stats["fuzzy_matches"] += 1

                if len(stats["sample_values"]) < 3 and src_val and not is_null:
                    stats["sample_values"].append(src_val[:80])

        if not source_stats or records_analyzed == 0:
            continue

        # Build per-source summary
        sources_summary = {}
        for ds_name, stats in sorted(source_stats.items()):
            total = max(stats["total"], 1)
            sources_summary[ds_name] = {
                "exact_match_rate": round(stats["exact_matches"] / total, 3),
                "fuzzy_match_rate": round(stats["fuzzy_matches"] / total, 3),
                "non_null_rate": round(stats["non_null"] / total, 3),
                "sample_values": stats["sample_values"],
            }

        # Recommend resolver based on evidence
        best_source = max(sources_summary.items(), key=lambda x: x[1]["exact_match_rate"])
        sorted_sources = sorted(sources_summary.items(), key=lambda x: x[1]["exact_match_rate"], reverse=True)
        trust_order = [s[0] for s in sorted_sources]

        # Decide resolver based on source quality patterns
        best_rate = best_source[1]["exact_match_rate"]
        second_rate = sorted_sources[1][1]["exact_match_rate"] if len(sorted_sources) > 1 else 0.0
        gap = best_rate - second_rate

        # Check source coverage patterns
        non_null_rates = [s[1].get("non_null_rate", 1.0) for s in sorted_sources]
        has_sparse_sources = any(r < 0.3 for r in non_null_rates)
        all_dense = all(r > 0.7 for r in non_null_rates)

        # Attribute-type heuristics (proven patterns from reference workflows)
        attr_lower = attr.lower()
        is_name_like = any(k in attr_lower for k in ("name", "title"))
        is_label_like = any(k in attr_lower for k in ("label", "publisher", "artist", "author"))
        is_numeric = any(k in attr_lower for k in ("duration", "price", "rating", "count", "length"))
        is_date = any(k in attr_lower for k in ("date", "year", "released"))
        is_list = any(k in attr_lower for k in ("track", "genre", "category", "tag"))

        if gap > 0.25:
            # One source is clearly dominant — trust it
            recommended = "prefer_higher_trust"
        elif has_sparse_sources and best_rate > 0.3:
            # Some sources are mostly null — pick the most complete value
            recommended = "most_complete"
        elif gap > 0.15:
            # Moderate gap — trust-based is warranted
            recommended = "prefer_higher_trust"
        elif all_dense and gap < 0.10:
            # Sources are close — use type-appropriate deterministic resolver
            if is_name_like:
                recommended = "shortest_string"  # canonical names are concise
            elif is_label_like:
                recommended = "longest_string"  # full names > abbreviations
            elif is_numeric:
                recommended = "maximum"  # largest = most complete measurement
            elif is_list:
                recommended = "union"  # preserve information from all sources
            else:
                recommended = "voting"
        elif all_dense and best_rate < 0.5:
            # All sources dense but none great — longest_string captures more info
            if is_name_like:
                recommended = "shortest_string"
            else:
                recommended = "longest_string"
        else:
            recommended = "voting"

        per_attribute[attr] = {
            "sources": sources_summary,
            "recommended_resolver": recommended,
            "recommended_trust_order": trust_order,
            "records_analyzed": records_analyzed,
        }

    # Build summary string
    summary_parts = []
    for attr, info in per_attribute.items():
        best = max(info["sources"].items(), key=lambda x: x[1]["exact_match_rate"])
        summary_parts.append(f"{attr}: {best[0]}={best[1]['exact_match_rate']:.0%} exact")
    summary = "; ".join(summary_parts) if summary_parts else "no source attribution computed"

    return {
        "name": "source_attribution",
        "summary": summary,
        "per_attribute": per_attribute,
        "records_mapped": len(fused_to_val),
        "normalization_pressure": 0.0,
        "actionability_pressure": min(1.0, len(per_attribute) / 5.0),
    }


PROBE_REGISTRY = {
    "reason_distribution": _probe_reason_distribution,
    "worst_attributes": _probe_worst_attributes,
    "recent_mismatches": _probe_recent_mismatches,
    "directive_coverage": _probe_directive_coverage,
    "mismatch_sampler": _probe_mismatch_sampler,
    "attribute_improvability": _probe_attribute_improvability,
    "null_patterns": _probe_null_patterns,
    "correspondence_density": _probe_correspondence_density,
    "blocking_recall": _probe_blocking_recall,
    "fusion_size": _probe_fusion_size,
    "source_attribution": _probe_source_attribution,
}


def _parse_json_candidate(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    return {}


def _run_custom_probe(spec: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    name = str(spec.get("name", "custom_probe")).strip() or "custom_probe"
    script = str(spec.get("script", "")).strip()
    if not script:
        return {"name": name, "summary": "missing script path", "normalization_pressure": 0.0}

    workspace = Path.cwd().resolve()
    script_path = Path(script).expanduser().resolve()
    if workspace not in script_path.parents and script_path != workspace:
        return {"name": name, "summary": "script path outside workspace", "normalization_pressure": 0.0}
    if "venv" in script_path.parts:
        return {"name": name, "summary": "script path under venv is not allowed", "normalization_pressure": 0.0}
    if not script_path.is_file():
        return {"name": name, "summary": "script file missing", "normalization_pressure": 0.0}

    args = spec.get("args", [])
    if not isinstance(args, list):
        args = []
    args = [str(x) for x in args[:8]]
    state_subset = {
        "evaluation_metrics": state.get("evaluation_metrics", {}),
        "normalization_directives": state.get("normalization_directives", {}),
        "auto_diagnostics": state.get("auto_diagnostics", {}),
    }
    cmd = [sys.executable, str(script_path), *args]
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(state_subset),
            text=True,
            capture_output=True,
            timeout=CUSTOM_PROBE_TIMEOUT_SECONDS,
            cwd=str(workspace),
        )
    except subprocess.TimeoutExpired:
        return {"name": name, "summary": f"timed out after {CUSTOM_PROBE_TIMEOUT_SECONDS:.1f}s", "normalization_pressure": 0.0}
    except Exception as e:
        return {"name": name, "summary": f"execution failed: {e}", "normalization_pressure": 0.0}

    parsed = _parse_json_candidate(proc.stdout)
    if parsed:
        parsed["name"] = name
        parsed.setdefault("summary", "custom probe completed")
        parsed.setdefault("normalization_pressure", _to_float(parsed.get("normalization_pressure"), 0.0))
        parsed.setdefault("actionability_pressure", _to_float(parsed.get("actionability_pressure"), 0.0))
        return parsed
    stdout = str(proc.stdout or "").strip().replace("\n", " ")
    stderr = str(proc.stderr or "").strip().replace("\n", " ")
    summary = stdout[:180] if stdout else (stderr[:180] if stderr else f"exit_code={proc.returncode}")
    return {
        "name": name,
        "summary": f"custom probe output: {summary}",
        "normalization_pressure": 0.0,
        "actionability_pressure": 0.0,
    }


def run_investigator_probes(
    *,
    state: Dict[str, Any],
    action_plan: List[Dict[str, Any]],
    load_dataset_fn: Callable[[str], Any],
) -> Dict[str, Any]:
    requested = state.get("investigator_probe_requests", [])
    requested_names = [str(x).strip() for x in requested] if isinstance(requested, list) else []

    keyword_boost = False
    for item in action_plan[:5]:
        text = str(item.get("action", "")).lower() if isinstance(item, dict) else str(item).lower()
        if any(k in text for k in ("list", "format", "encoding", "country", "normaliz", "type mismatch")):
            keyword_boost = True
            break

    plan: List[str] = []
    for name in requested_names:
        if name in PROBE_REGISTRY and name not in plan:
            plan.append(name)
    for name in ("reason_distribution", "worst_attributes", "mismatch_sampler", "attribute_improvability", "source_attribution", "null_patterns", "correspondence_density", "blocking_recall", "fusion_size"):
        if name not in plan:
            plan.append(name)
    if keyword_boost:
        for name in ("recent_mismatches", "directive_coverage"):
            if name not in plan:
                plan.append(name)
    plan = plan[:MAX_PROBES]

    results: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for probe_name in plan:
        elapsed = time.perf_counter() - started
        if elapsed > MAX_RUNTIME_SECONDS:
            results.append({"name": "probe_budget", "summary": "probe time budget reached"})
            break
        probe_fn = PROBE_REGISTRY.get(probe_name)
        if probe_fn is None:
            continue
        try:
            if probe_name == "directive_coverage":
                output = probe_fn(state, load_dataset_fn)
            elif probe_name == "attribute_improvability":
                output = probe_fn(state, results)
            else:
                output = probe_fn(state)
            if not isinstance(output, dict):
                output = {"name": probe_name, "summary": str(output)}
        except Exception as e:
            output = {"name": probe_name, "summary": f"probe failed: {e}"}
        results.append(output)

    custom_specs = state.get("investigator_exec_plan", [])
    if isinstance(custom_specs, list):
        for spec in custom_specs[:MAX_CUSTOM_PROBES]:
            elapsed = time.perf_counter() - started
            if elapsed > MAX_RUNTIME_SECONDS:
                results.append({"name": "probe_budget", "summary": "custom probe budget reached"})
                break
            if not isinstance(spec, dict):
                continue
            results.append(_run_custom_probe(spec, state))

    norm_pressure = min(
        1.0,
        sum(_to_float(x.get("normalization_pressure"), 0.0) for x in results if isinstance(x, dict)) / max(1, len(results)),
    )
    action_pressure = min(
        1.0,
        sum(_to_float(x.get("actionability_pressure"), 0.0) for x in results if isinstance(x, dict)) / max(1, len(results)),
    )
    lines = []
    for item in results:
        lines.append(f"{item.get('name')}: {item.get('summary')}")
    return {
        "plan": plan,
        "results": results,
        "normalization_pressure": round(norm_pressure, 6),
        "actionability_pressure": round(action_pressure, 6),
        "summary": " | ".join(lines[:4]),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }
