"""Pipeline state compression into focused LLM context.

Extracts high-signal information -- worst attributes, specific failure
reasons, concrete action items -- rather than dumping full JSON
diagnostics (500+ lines).
"""
import json
import os
from typing import Any, Dict, List

from config import MISMATCH_SAMPLE_ATTRIBUTES, STAGNATION_WINDOW, STAGNATION_DELTA_THRESHOLD


def summarize_metrics_for_llm(metrics: Dict[str, Any], max_attributes: int = MISMATCH_SAMPLE_ATTRIBUTES) -> str:
    """Summarizes evaluation metrics, highlighting only the worst attributes.

    Returns a compact text block with overall accuracy, the top N worst
    attributes with their scores, and a brief summary of healthy attributes.
    """
    if not isinstance(metrics, dict) or not metrics:
        return "No metrics available."

    overall = metrics.get("overall_accuracy", metrics.get("overall", "unknown"))

    # Per-attribute accuracies
    attr_accuracies = {}
    for key, val in metrics.items():
        if key.startswith("overall") or key.startswith("_") or key in ("evaluation_functions", "eval_stage"):
            continue
        if isinstance(val, dict) and "accuracy" in val:
            attr_accuracies[key] = float(val.get("accuracy", 0.0) or 0.0)
        elif isinstance(val, (int, float)):
            attr_accuracies[key] = float(val)

    if not attr_accuracies:
        return f"Overall accuracy: {overall}. No per-attribute breakdown available."

    # Sort: worst first
    sorted_attrs = sorted(attr_accuracies.items(), key=lambda x: x[1])
    worst = sorted_attrs[:max_attributes]
    good = [a for a, v in sorted_attrs if v >= 0.85]

    lines = [f"Overall accuracy: {overall}"]
    lines.append(f"Worst {len(worst)} attributes (need fixing):")
    for attr, acc in worst:
        lines.append(f"  - {attr}: {acc:.1%}")
    if good:
        lines.append(f"{len(good)} attributes are above 85% (no changes needed): {', '.join(good[:5])}{'...' if len(good) > 5 else ''}")

    return "\n".join(lines)


def summarize_diagnostics_for_llm(auto_diagnostics: Dict[str, Any]) -> str:
    """Compresses auto diagnostics into actionable bullet points."""
    if not isinstance(auto_diagnostics, dict) or not auto_diagnostics:
        return "No diagnostics available."

    lines = []

    # ID alignment
    id_align = auto_diagnostics.get("id_alignment", {})
    if isinstance(id_align, dict):
        direct = id_align.get("direct_coverage_ratio")
        mapped = id_align.get("mapped_coverage_ratio")
        if direct is not None:
            lines.append(f"ID alignment: direct={float(direct):.1%}, mapped={float(mapped or 0):.1%}")
            if float(direct or 0) < 0.7:
                lines.append("  ⚠ Low direct ID coverage — fused entities may use different ID format than evaluation set")

    # Debug reason ratios
    ratios = auto_diagnostics.get("debug_reason_ratios", {})
    if isinstance(ratios, dict) and ratios:
        top_reasons = sorted(ratios.items(), key=lambda x: float(x[1] or 0), reverse=True)[:5]
        lines.append("Top mismatch reasons:")
        for reason, ratio in top_reasons:
            lines.append(f"  - {reason}: {float(ratio):.1%}")

    # Size comparison
    size = auto_diagnostics.get("fusion_size_comparison", auto_diagnostics.get("size_comparison", {}))
    if isinstance(size, dict) and size:
        comparisons = size.get("comparisons", {})
        actual_info = size.get("actual", {})
        actual_rows = actual_info.get("rows") if isinstance(actual_info, dict) else None
        if isinstance(comparisons, dict) and comparisons:
            lines.append("Fusion size comparison:")
            for stage, comp in comparisons.items():
                if not isinstance(comp, dict):
                    continue
                expected = comp.get("expected_rows")
                pct_err = comp.get("rows_pct_error")
                if expected and actual_rows:
                    err_str = f" (error: {float(pct_err):.0%})" if pct_err is not None else ""
                    lines.append(f"  - {stage} estimate: {expected}, actual: {actual_rows}{err_str}")
                reasoning = comp.get("reasoning", "")
                if reasoning:
                    lines.append(f"    {reasoning}")
        elif actual_rows:
            expected = size.get("expected_rows", size.get("expected"))
            if expected:
                lines.append(f"Fusion size: expected ~{expected}, got {actual_rows}")

    return "\n".join(lines) if lines else "Diagnostics available but no significant findings."


def summarize_integration_diagnostics_for_llm(report: Dict[str, Any], max_findings: int = 5) -> str:
    """Compresses the integration diagnostics report to top findings and recommendations."""
    if not isinstance(report, dict) or not report:
        return "No integration diagnostics report available."

    lines = []

    # Summary
    summary = report.get("summary", "")
    if summary:
        lines.append(f"Summary: {str(summary)[:200]}")

    # Top findings
    findings = report.get("findings", [])
    if isinstance(findings, list) and findings:
        lines.append(f"Top {min(len(findings), max_findings)} findings:")
        for f in findings[:max_findings]:
            if isinstance(f, dict):
                sev = f.get("severity", "")
                msg = str(f.get("message", f.get("description", "")))[:150]
                lines.append(f"  [{sev}] {msg}")

    # Top recommendations
    recs = report.get("recommendations", [])
    if isinstance(recs, list) and recs:
        lines.append(f"Top recommendations:")
        for r in recs[:3]:
            if isinstance(r, dict):
                action = str(r.get("action", r.get("recommendation", "")))[:150]
                lines.append(f"  → {action}")
            elif isinstance(r, str):
                lines.append(f"  → {r[:150]}")

    # Fusion policy recommendations
    fps = report.get("fusion_policy_recommendations", [])
    if isinstance(fps, list) and fps:
        lines.append("Fusion policy suggestions:")
        for fp in fps[:3]:
            if isinstance(fp, dict):
                lines.append(f"  → {fp.get('action', '')}: target={fp.get('target_attributes', '')}, evidence={str(fp.get('evidence_summary', ''))[:100]}")

    return "\n".join(lines) if lines else "Integration diagnostics report is empty."


def build_focused_pipeline_context(state: Dict[str, Any]) -> str:
    """Builds a focused context string for the pipeline adaptation LLM prompt.

    Produces a compressed, signal-focused summary instead of raw full-state JSON.
    """
    sections = []

    # Metrics summary
    metrics = state.get("evaluation_metrics_for_adaptation", state.get("evaluation_metrics", {}))
    if metrics:
        sections.append("=== EVALUATION METRICS (focused) ===\n" + summarize_metrics_for_llm(metrics))

    # Diagnostics summary
    auto_diag = state.get("auto_diagnostics", {})
    if auto_diag:
        sections.append("=== AUTO DIAGNOSTICS (focused) ===\n" + summarize_diagnostics_for_llm(auto_diag))

    # Mismatch samples (already focused from the probe)
    probe_results = state.get("investigator_probe_results", {})
    if isinstance(probe_results, dict):
        results_list = probe_results.get("results", [])
        sampler = next((r for r in results_list if isinstance(r, dict) and r.get("name") == "mismatch_sampler"), {})
        samples = sampler.get("samples_by_attribute", {})
        if samples:
            sections.append("=== CONCRETE MISMATCH EXAMPLES ===\n" + json.dumps(samples, indent=2))

    # All other probe results (not just mismatch_sampler)
    if isinstance(probe_results, dict):
        results_list = probe_results.get("results", [])
        probe_sections = []
        for probe in results_list:
            if not isinstance(probe, dict):
                continue
            name = probe.get("name", "")
            if name == "mismatch_sampler":
                continue  # Already included above
            summary = probe.get("summary", "")
            if not summary:
                continue
            probe_detail = f"- {name}: {summary}"
            # Include key data fields for high-value probes
            if name == "worst_attributes" and probe.get("worst_attributes"):
                probe_detail += f"\n  Worst: {probe['worst_attributes'][:5]}"
            elif name == "null_patterns" and probe.get("null_heavy_columns"):
                probe_detail += f"\n  Null-heavy columns: {probe['null_heavy_columns'][:5]}"
            elif name == "blocking_recall" and probe.get("problem_pairs"):
                probe_detail += f"\n  Problem pairs: {probe['problem_pairs']}"
            elif name == "correspondence_density" and probe.get("empty_pairs"):
                probe_detail += f"\n  Empty pairs: {probe['empty_pairs']}"
            elif name == "reason_distribution" and probe.get("top_reasons"):
                probe_detail += f"\n  Top reasons: {json.dumps(probe.get('top_reasons', {}))}"
            elif name == "directive_coverage" and probe.get("missing_count", 0) > 0:
                probe_detail += f"\n  Missing directive columns: {probe.get('missing_count')}"
            probe_sections.append(probe_detail)
        if probe_sections:
            sections.append("=== PROBE ANALYSIS (all probes) ===\n" + "\n".join(probe_sections))

    # Source attribution (per-source match rates for low-accuracy attributes)
    if isinstance(probe_results, dict):
        results_list = probe_results.get("results", [])
        src_attr = next((r for r in results_list if isinstance(r, dict) and r.get("name") == "source_attribution"), None)
        if src_attr and src_attr.get("per_attribute"):
            sa_lines = []
            for attr, info in src_attr["per_attribute"].items():
                sources = info.get("sources", {})
                src_parts = []
                for ds, stats in sorted(sources.items(), key=lambda x: -x[1].get("exact_match_rate", 0)):
                    exact = stats.get("exact_match_rate", 0)
                    fuzzy = stats.get("fuzzy_match_rate", 0)
                    non_null = stats.get("non_null_rate", 0)
                    src_parts.append(f"{ds}: exact={exact:.0%}, fuzzy={fuzzy:.0%}, non_null={non_null:.0%}")
                resolver = info.get("recommended_resolver", "?")
                trust = info.get("recommended_trust_order", [])
                sa_lines.append(f"  {attr} (n={info.get('records_analyzed', '?')}):")
                for part in src_parts:
                    sa_lines.append(f"    {part}")
                sa_lines.append(f"    -> recommended: {resolver}({' > '.join(trust)})")
            sections.append("=== SOURCE VS VALIDATION COMPARISON ===\n" + "\n".join(sa_lines))

    # Attribute improvability analysis
    if isinstance(probe_results, dict):
        results_list = probe_results.get("results", [])
        improv = next((r for r in results_list if isinstance(r, dict) and r.get("name") == "attribute_improvability"), None)
        if improv and improv.get("attribute_classifications"):
            improv_lines = []
            for attr, info in sorted(improv["attribute_classifications"].items(), key=lambda x: x[1].get("accuracy", 1.0)):
                cls = info.get("class", "unknown")
                acc = info.get("accuracy", 0)
                ceil = info.get("ceiling", 1.0)
                reasons = info.get("reasons", [])
                reason_str = f" [{', '.join(reasons)}]" if reasons else ""
                improv_lines.append(f"  {attr}: {cls} (accuracy={acc:.1%}, ceiling={ceil:.1%}){reason_str}")
            if improv.get("structural_ceiling_estimate"):
                improv_lines.append(f"\nEstimated overall ceiling with current structure: {improv['structural_ceiling_estimate']:.1%}")
                improv_lines.append(f"Improvable headroom: {improv.get('improvable_headroom', 0):.1%}")
            # Impact ranking
            impact = improv.get("impact_ranking", [])
            if impact:
                top_impact = [f"{r['attribute']}({r['class']}, impact={r['impact']:.3f})" for r in impact[:5]]
                improv_lines.append(f"Priority ranking: {', '.join(top_impact)}")
            sections.append("=== ATTRIBUTE IMPROVABILITY ===\n" + "\n".join(improv_lines))

    # Investigation decision (from LLM-driven investigator)
    inv_log = state.get("investigation_log", {})
    if isinstance(inv_log, dict) and inv_log.get("decision"):
        decision = inv_log["decision"]
        sections.append(
            f"=== INVESTIGATION DECISION ===\n"
            f"Next node: {decision.get('next_node', 'unknown')}\n"
            f"Diagnosis: {decision.get('diagnosis', 'none')}\n"
            f"Reasoning: {decision.get('reasoning', 'none')}"
        )

    # Cluster analysis evidence
    cluster = state.get("cluster_analysis_result", {})
    if isinstance(cluster, dict):
        cluster_overall = cluster.get("_overall", {})
        if isinstance(cluster_overall, dict) and cluster_overall.get("recommended_strategy"):
            cluster_lines = [f"Overall recommendation: {cluster_overall.get('recommended_strategy')}"]
            evidence = cluster_overall.get("all_evidence", "")
            if evidence:
                # Truncate to first 500 chars for focused context
                cluster_lines.append(evidence[:500])
            sections.append("=== CLUSTER ANALYSIS ===\n" + "\n".join(cluster_lines))

    # Fusion guidance (mismatch classifications + attribute strategies)
    fusion_guidance = state.get("fusion_guidance", {})
    if isinstance(fusion_guidance, dict):
        fg_lines = []
        # Mismatch classification breakdown (highest signal)
        mc = fusion_guidance.get("mismatch_classifications", {})
        if mc:
            fg_lines.append("Mismatch classification per attribute:")
            for attr, reasons in list(mc.items())[:8]:
                reason_str = ", ".join(f"{r}={c}" for r, c in sorted(reasons.items(), key=lambda x: -x[1]))
                fg_lines.append(f"  {attr}: {reason_str}")
        # Global hints
        hints = fusion_guidance.get("global_hints", [])
        if hints:
            fg_lines.append("Hints: " + " | ".join(hints[:3]))
        # Post-clustering
        pc = fusion_guidance.get("post_clustering", {})
        if isinstance(pc, dict) and pc.get("recommended_strategy"):
            fg_lines.append(f"Post-clustering: {pc['recommended_strategy']}")
        if fg_lines:
            sections.append("=== FUSION GUIDANCE ===\n" + "\n".join(fg_lines))

    # Reasoning brief (already compact)
    brief = state.get("evaluation_reasoning_brief", {})
    if isinstance(brief, dict) and brief:
        sections.append("=== REASONING BRIEF ===\n" + json.dumps(brief, indent=2))

    return "\n\n".join(sections)


def build_iteration_history_section(state: Dict[str, Any]) -> str:
    """Builds a compact iteration history showing the accuracy trajectory and what was tried."""
    history = state.get("iteration_history", [])
    if not isinstance(history, list) or not history:
        return ""

    lines = ["=== ITERATION HISTORY (what was tried and what happened) ==="]
    for entry in history[-5:]:  # Last 5 iterations max
        if not isinstance(entry, dict):
            continue
        attempt = entry.get("attempt", "?")
        acc = entry.get("accuracy")
        delta = entry.get("delta")
        decision = entry.get("decision", "")
        description = entry.get("description", "")

        acc_str = f"{float(acc):.1%}" if acc is not None else "N/A"
        delta_str = f" ({float(delta):+.1%})" if delta is not None else ""
        decision_str = f" [→{decision}]" if decision else ""

        line = f"  Attempt {attempt}: {acc_str}{delta_str}{decision_str}"
        if description:
            line += f" — {description[:120]}"
        lines.append(line)

        # Show per-attribute accuracies (compact format, worst 5 only)
        attr_accs = entry.get("attribute_accuracies", {})
        if isinstance(attr_accs, dict) and attr_accs:
            worst = sorted(attr_accs.items(), key=lambda x: x[1])[:5]
            attr_parts = [f"{a}={v:.0%}" for a, v in worst]
            lines.append(f"    attrs: {', '.join(attr_parts)}")

    # Add trajectory summary
    accuracies = [float(e["accuracy"]) for e in history if isinstance(e, dict) and e.get("accuracy") is not None]
    if len(accuracies) >= 2:
        if accuracies[-1] > accuracies[-2] + 0.005:
            lines.append("  Trajectory: IMPROVING")
        elif accuracies[-1] < accuracies[-2] - 0.005:
            lines.append("  Trajectory: REGRESSING")
        else:
            lines.append("  Trajectory: STAGNATING")
        if len(accuracies) >= 3 and all(abs(a - accuracies[-1]) < 0.01 for a in accuracies[-3:]):
            lines.append("  ⚠ Accuracy has plateaued — try a fundamentally different approach")

    return "\n".join(lines)


def build_stagnation_analysis(state: Dict[str, Any]) -> str:
    """Detects accuracy stagnation and explains why further iteration may be unproductive.

    Returns a formatted section if stagnation is detected, empty string otherwise.
    """
    history = state.get("iteration_history", [])
    if not isinstance(history, list) or len(history) < STAGNATION_WINDOW + 1:
        return ""

    # Extract accuracy values from recent entries
    recent = [e for e in history if isinstance(e, dict) and e.get("accuracy") is not None]
    if len(recent) < STAGNATION_WINDOW + 1:
        return ""

    # Check if last STAGNATION_WINDOW deltas are all below threshold
    window = recent[-(STAGNATION_WINDOW + 1):]
    deltas = []
    for i in range(1, len(window)):
        delta = float(window[i]["accuracy"]) - float(window[i - 1]["accuracy"])
        deltas.append(delta)

    if not all(abs(d) < STAGNATION_DELTA_THRESHOLD for d in deltas):
        return ""

    # Stagnation detected — build explanation
    total_improvement = float(window[-1]["accuracy"]) - float(window[0]["accuracy"])
    recent_decisions = [e.get("decision", "?") for e in window[1:]]

    lines = ["=== STAGNATION ANALYSIS ==="]
    lines.append(
        f"Accuracy has improved only {total_improvement:+.1%} over the last "
        f"{STAGNATION_WINDOW} iterations (threshold: {STAGNATION_DELTA_THRESHOLD:.0%})."
    )
    lines.append(f"Recent routing decisions: {', '.join(recent_decisions)}")
    lines.append("")
    lines.append("Parameter tweaks within the current approach are unlikely to yield further gains.")
    lines.append("Consider:")

    # Suggest what hasn't been tried
    tried = set(recent_decisions)
    if "run_matching_tester" not in tried:
        lines.append("  1. Re-running the matching tester with different strategies")
    if "normalization_node" not in tried:
        lines.append("  2. Trying normalization if format mismatches remain")
    if "run_blocking_tester" not in tried:
        lines.append("  3. Re-running the blocking tester")

    # Check if remaining gap is mostly in structurally limited attributes
    probe_results = state.get("investigator_probe_results", {})
    if isinstance(probe_results, dict):
        for r in probe_results.get("results", []):
            if isinstance(r, dict) and r.get("name") == "attribute_improvability":
                headroom = r.get("improvable_headroom", 0)
                ceiling = r.get("structural_ceiling_estimate", 0)
                if headroom < 0.05:
                    lines.append(
                        f"\n⚠ Improvable headroom is only {headroom:.1%}. "
                        f"The structural ceiling is ~{ceiling:.1%}. "
                        f"The remaining accuracy gap is mostly in structurally limited attributes. "
                        f"Consider accepting current accuracy or recommending human_review_export."
                    )
                break

    lines.append(
        "\nIf all approaches have been tried, the current accuracy may be near "
        "the achievable maximum for this data. Recommend human_review_export."
    )

    return "\n".join(lines)


def build_attribute_trajectory(state: Dict[str, Any]) -> str:
    """Tracks per-attribute accuracy across iterations to identify plateaus.

    Returns a formatted section showing which attributes are improving,
    plateaued, or already good.
    """
    cycle_audit = state.get("evaluation_cycle_audit", [])
    if not isinstance(cycle_audit, list) or len(cycle_audit) < 2:
        return ""

    # Collect per-attribute accuracies across iterations
    attr_trajectories: Dict[str, List[float]] = {}
    for entry in cycle_audit[-5:]:  # Last 5 iterations
        if not isinstance(entry, dict):
            continue
        raw_metrics = entry.get("raw_metrics", {})
        if not isinstance(raw_metrics, dict):
            continue
        for key, val in raw_metrics.items():
            if not isinstance(key, str) or not key.endswith("_accuracy"):
                continue
            if key.startswith("overall") or key.startswith("macro") or key.startswith("mean"):
                continue
            fval = float(val) if val is not None else None
            if fval is not None:
                attr_trajectories.setdefault(key, []).append(fval)

    if not attr_trajectories:
        return ""

    lines = ["=== ATTRIBUTE TRAJECTORY (across iterations) ==="]
    for attr, values in sorted(attr_trajectories.items(), key=lambda x: x[1][-1] if x[1] else 1.0):
        if len(values) < 2:
            continue
        trajectory_str = " → ".join(f"{v:.0%}" for v in values)
        total_delta = values[-1] - values[0]

        if values[-1] >= 0.90:
            label = "STABLE (already good)"
        elif len(values) >= 2 and all(abs(v - values[-1]) < 0.02 for v in values[-min(3, len(values)):]):
            label = f"PLATEAUED at ~{values[-1]:.0%}"
        elif total_delta > 0.02:
            label = f"IMPROVING (+{total_delta:.0%} total)"
        elif total_delta < -0.02:
            label = f"REGRESSING ({total_delta:+.0%} total)"
        else:
            label = "FLAT"

        lines.append(f"  {attr}: {trajectory_str} ({label})")

    return "\n".join(lines) if len(lines) > 1 else ""


def build_input_data_context(state: Dict[str, Any], max_rows: int = 3) -> str:
    """Builds a preview of raw input data for worst-performing attributes.

    Shows the agent what values look like in each source dataset so it can
    make informed decisions about trust, normalization, and fusion strategy.
    """
    import pandas as pd

    # Get worst attributes from probes or metrics
    probe_results = state.get("investigator_probe_results", {})
    worst_attrs = []
    if isinstance(probe_results, dict):
        results_list = probe_results.get("results", [])
        for probe in results_list:
            if isinstance(probe, dict) and probe.get("name") == "worst_attributes":
                worst_attrs = probe.get("worst_attributes", [])[:5]
                break

    if not worst_attrs:
        # Fall back to metrics
        metrics = state.get("evaluation_metrics", {})
        if isinstance(metrics, dict):
            attr_accs = {}
            for key, val in metrics.items():
                if key.startswith("overall") or key.startswith("_"):
                    continue
                if isinstance(val, dict) and "accuracy" in val:
                    attr_accs[key] = float(val.get("accuracy", 1.0) or 1.0)
            if attr_accs:
                worst_attrs = [k for k, v in sorted(attr_accs.items(), key=lambda x: x[1])][:5]

    if not worst_attrs:
        return ""

    # Get schema correspondences to find column mappings
    schema_corr = state.get("schema_correspondences", {})
    datasets = state.get("datasets", [])

    if not datasets:
        return ""

    lines = [f"=== INPUT DATA SAMPLES FOR WORST ATTRIBUTES ==="]
    lines.append(f"Worst attributes: {worst_attrs}")
    lines.append("Use these samples to understand source data quality and decide on trust/normalization strategy.\n")

    for ds_path in datasets:
        try:
            ds_name = os.path.basename(str(ds_path))
            ext = os.path.splitext(ds_path)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(ds_path, nrows=max_rows * 3)  # Read a few extra for variety
            elif ext == ".parquet":
                df = pd.read_parquet(ds_path)
                df = df.head(max_rows * 3)
            else:
                continue

            # Find columns that map to our worst attributes (check schema correspondences)
            relevant_cols = []
            for attr in worst_attrs:
                # Direct match
                if attr in df.columns:
                    relevant_cols.append((attr, attr))
                    continue
                # Check schema correspondences for mapped names
                if isinstance(schema_corr, dict):
                    for pair_key, mappings in schema_corr.items():
                        if not isinstance(mappings, (list, dict)):
                            continue
                        mapping_list = mappings if isinstance(mappings, list) else [mappings]
                        for m in mapping_list:
                            if isinstance(m, dict):
                                target = m.get("target", m.get("fused_name", ""))
                                source = m.get("source", m.get("column", ""))
                                if target == attr and source in df.columns:
                                    relevant_cols.append((attr, source))

            if not relevant_cols:
                # Try fuzzy matching on column names
                for attr in worst_attrs:
                    for col in df.columns:
                        if attr.lower() in col.lower() or col.lower() in attr.lower():
                            relevant_cols.append((attr, col))
                            break

            if relevant_cols:
                lines.append(f"Dataset: {ds_name}")
                for fused_attr, source_col in relevant_cols[:5]:
                    sample_vals = df[source_col].dropna().head(max_rows).tolist()
                    sample_strs = [str(v)[:100] for v in sample_vals]
                    null_count = int(df[source_col].isna().sum())
                    total = len(df)
                    lines.append(f"  {source_col} (→{fused_attr}): {sample_strs}  [{null_count}/{total} null]")
                lines.append("")
        except Exception:
            continue

    return "\n".join(lines) if len(lines) > 3 else ""


def build_correspondence_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """Builds a summary of correspondence statistics from probe results and state."""
    summary = {}

    # Extract from correspondence_integrity in state
    integrity = state.get("correspondence_integrity", {})
    if isinstance(integrity, dict):
        pair_details = integrity.get("pair_details", {})
        if isinstance(pair_details, dict):
            for pair, details in pair_details.items():
                if isinstance(details, dict):
                    summary[pair] = {
                        "row_count": details.get("row_count", 0),
                        "structurally_valid": details.get("structurally_valid", True),
                    }

    # Enrich with probe data if available
    probe_results = state.get("investigator_probe_results", {})
    if isinstance(probe_results, dict):
        results_list = probe_results.get("results", [])
        for probe in results_list:
            if not isinstance(probe, dict):
                continue
            if probe.get("name") == "correspondence_density":
                pairs = probe.get("pairs", [])
                for p in pairs:
                    if isinstance(p, dict):
                        pair_name = p.get("pair", "")
                        if pair_name in summary:
                            summary[pair_name].update({
                                "unique_id1": p.get("unique_id1"),
                                "unique_id2": p.get("unique_id2"),
                                "avg_matches_per_entity": p.get("avg_matches_per_entity"),
                                "pattern": p.get("pattern", ""),
                            })
                        else:
                            summary[pair_name] = {
                                "rows": p.get("rows"),
                                "unique_id1": p.get("unique_id1"),
                                "unique_id2": p.get("unique_id2"),
                                "avg_matches_per_entity": p.get("avg_matches_per_entity"),
                                "pattern": p.get("pattern", ""),
                            }

    return summary if summary else None


def enrich_reasoning_brief_with_probes(brief: Dict[str, Any], probe_results: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Enriches the reasoning brief with structured probe data and accuracy context."""
    if not isinstance(brief, dict):
        brief = {}
    enriched = dict(brief)

    if isinstance(probe_results, dict):
        results_list = probe_results.get("results", [])

        # Add worst attributes with scores
        for probe in results_list:
            if not isinstance(probe, dict):
                continue
            if probe.get("name") == "worst_attributes":
                enriched["worst_attributes_from_probe"] = probe.get("worst_attributes", [])[:5]
            elif probe.get("name") == "reason_distribution":
                enriched["dominant_mismatch_reasons"] = probe.get("summary", "")
            elif probe.get("name") == "mismatch_sampler":
                counts = probe.get("attribute_mismatch_counts", {})
                if counts:
                    enriched["mismatch_counts"] = dict(list(counts.items())[:5])

        # Add aggregate pressures
        enriched["normalization_pressure"] = probe_results.get("normalization_pressure", 0.0)
        enriched["actionability_pressure"] = probe_results.get("actionability_pressure", 0.0)

    # Add per-attribute accuracy from metrics
    if isinstance(metrics, dict):
        attr_accs = {}
        for key, val in metrics.items():
            if key.startswith("overall") or key.startswith("_"):
                continue
            if isinstance(val, dict) and "accuracy" in val:
                attr_accs[key] = round(float(val.get("accuracy", 0.0) or 0.0), 3)
        if attr_accs:
            # Only include worst 5
            sorted_attrs = sorted(attr_accs.items(), key=lambda x: x[1])[:5]
            enriched["attribute_accuracies"] = dict(sorted_attrs)

    return enriched
