import os
import re
from typing import Any, Dict, List

from helpers.investigation_helpers import detect_normalization_issue as helper_detect_normalization_issue
from helpers.normalization_policy import (
    collect_eval_debug_signals,
    infer_country_output_format_from_validation,
    infer_validation_text_case_map,
)


def _load_dataset_for_style(path: str):
    if not path or not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".csv":
            from PyDI.io import load_csv

            return load_csv(path)
        if ext == ".parquet":
            from PyDI.io import load_parquet

            return load_parquet(path)
        if ext == ".xml":
            from PyDI.io import load_xml

            return load_xml(path, nested_handling="aggregate")
    except Exception:
        return None
    return None


def _sanitize_directives_to_validation_style(
    directives: Dict[str, Any],
    state: Dict[str, Any],
    validation_style: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Hard-guard normalization directives so they cannot diverge from validation style.
    """
    if not isinstance(directives, dict):
        directives = {}
    out = dict(directives)

    report_validation_style = {}
    try:
        report_validation_style = (
            state.get("normalization_report", {}).get("validation_style", {})
            if isinstance(state.get("normalization_report", {}), dict)
            else {}
        )
    except Exception:
        report_validation_style = {}
    validation_style = (
        validation_style if isinstance(validation_style, dict) and validation_style else report_validation_style
    )

    allowed_list_cols = {
        str(c).lower()
        for c in (validation_style.get("validation_list_like_columns_hint", []) or [])
        if str(c).strip()
    }
    allowed_lower_cols = {
        str(c).lower()
        for c in (validation_style.get("lowercase_columns_hint", []) or [])
        if str(c).strip()
    }
    allowed_country_format = str(
        validation_style.get("inferred_country_output_format", "")
    ).strip()

    def _keep_allowed(values: Any, allowed: set[str]) -> List[str]:
        if not isinstance(values, list):
            return []
        kept: List[str] = []
        seen = set()
        for v in values:
            s = str(v).strip()
            if not s:
                continue
            if allowed and s.lower() not in allowed:
                continue
            if s.lower() in seen:
                continue
            seen.add(s.lower())
            kept.append(s)
        return kept

    # Validation style wins for formatting controls.
    out["list_columns"] = _keep_allowed(out.get("list_columns", []), allowed_list_cols)
    out["lowercase_columns"] = _keep_allowed(
        out.get("lowercase_columns", []), allowed_lower_cols
    )
    if allowed_country_format:
        out["country_output_format"] = allowed_country_format
    out["list_normalization_required"] = bool(out.get("list_columns", []))

    return out


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            m = re.search(r"-?\d+(\.\d+)?", value.replace(",", "."))
            if m:
                return float(m.group(0))
            return default
        return float(value)
    except Exception:
        return default


def _build_investigator_action_plan(report: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Build a ranked, machine-usable action plan from diagnostics output.
    Keeps logic dataset-agnostic and purely evidence-driven.
    """
    if not isinstance(report, dict):
        return []

    candidates: List[Dict[str, Any]] = []
    sources = [
        ("fusion_policy_recommendations", report.get("fusion_policy_recommendations", [])),
        ("recommendations", report.get("recommendations", [])),
    ]

    for source_name, recs in sources:
        if not isinstance(recs, list):
            continue
        for rec in recs:
            if isinstance(rec, str):
                text = rec.strip()
                if not text:
                    continue
                candidates.append(
                    {
                        "source": source_name,
                        "action": text,
                        "target_attributes": [],
                        "evidence_summary": "",
                        "expected_impact": "",
                        "confidence": 0.45,
                        "priority_score": 0.45,
                    }
                )
                continue

            if not isinstance(rec, dict):
                continue
            action = (
                rec.get("action")
                or rec.get("recommendation")
                or rec.get("title")
                or rec.get("summary")
                or ""
            )
            action = str(action).strip()
            if not action:
                continue
            targets = rec.get("target_attributes", rec.get("columns", []))
            if not isinstance(targets, list):
                targets = [str(targets)] if str(targets).strip() else []
            targets = [str(t) for t in targets if str(t).strip()]

            confidence = _to_float(rec.get("confidence"), 0.0)
            if confidence > 1.0:
                confidence = confidence / 100.0
            confidence = max(0.0, min(1.0, confidence))

            impact = _to_float(
                rec.get("expected_impact_score", rec.get("expected_gain", rec.get("impact"))),
                0.0,
            )
            if impact > 1.0:
                impact = impact / 100.0
            impact = max(0.0, min(1.0, impact))

            if impact == 0.0:
                text_blob = " ".join(
                    [
                        str(rec.get("expected_impact", "")),
                        str(rec.get("evidence", "")),
                        str(rec.get("rationale", "")),
                    ]
                ).lower()
                if any(k in text_blob for k in ["high", "dominant", "catastrophic", "severe"]):
                    impact = 0.7
                elif any(k in text_blob for k in ["medium", "moderate"]):
                    impact = 0.45
                elif any(k in text_blob for k in ["low", "minor"]):
                    impact = 0.2

            if confidence == 0.0:
                confidence = 0.55 if impact >= 0.5 else 0.4

            evidence_summary = str(
                rec.get("evidence_summary", rec.get("evidence", rec.get("rationale", "")))
            ).strip()
            expected_impact = str(
                rec.get("expected_impact", rec.get("expected_gain", rec.get("impact", "")))
            ).strip()

            priority_score = (0.65 * impact) + (0.35 * confidence)
            candidates.append(
                {
                    "source": source_name,
                    "action": action,
                    "target_attributes": targets,
                    "evidence_summary": evidence_summary,
                    "expected_impact": expected_impact,
                    "confidence": round(confidence, 3),
                    "priority_score": round(priority_score, 3),
                }
            )

    # Deduplicate by action text, keep strongest variant.
    best_by_action: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        key = c["action"].strip().lower()
        prev = best_by_action.get(key)
        if prev is None or c.get("priority_score", 0.0) > prev.get("priority_score", 0.0):
            best_by_action[key] = c

    ranked = sorted(
        best_by_action.values(),
        key=lambda x: (x.get("priority_score", 0.0), x.get("confidence", 0.0)),
        reverse=True,
    )
    return ranked[:limit]


def run_investigator_node(agent, state: Dict[str, Any]) -> Dict[str, Any]:
    agent._log_action(
        "investigator_node",
        "start",
        "Run diagnostics and decide next step",
        "Can route back to normalization",
    )
    print("[*] Running consolidated investigator node")

    local_state = dict(state)

    diagnostics_updates = agent.integration_diagnostics(local_state)
    local_state.update(diagnostics_updates)

    decision_updates = agent.evaluation_decision(local_state)
    local_state.update(decision_updates)

    metrics = local_state.get("evaluation_metrics", {}) if isinstance(local_state, dict) else {}
    overall_acc = 0.0
    try:
        overall_acc = float(metrics.get("overall_accuracy", 0.0))
    except Exception:
        overall_acc = 0.0

    attempts = int(local_state.get("evaluation_attempts", 0))
    investigator_decision = "human_review_export"
    reasoning_updates: Dict[str, Any] = {}
    cluster_updates: Dict[str, Any] = {}
    normalization_assessment: Dict[str, Any] = {}
    normalization_directives: Dict[str, Any] = (
        state.get("normalization_directives", {}) if isinstance(state, dict) else {}
    )
    if not isinstance(normalization_directives, dict):
        normalization_directives = {}
    normalization_reasons: List[str] = []
    normalization_needed = False

    validation_style_for_clamp: Dict[str, Any] = {}
    try:
        validation_path = (
            agent._evaluation_testset_path(local_state, force_test=False)
            if hasattr(agent, "_evaluation_testset_path")
            else ""
        )
        validation_df = _load_dataset_for_style(validation_path) if validation_path else None
        if validation_df is not None and not validation_df.empty:
            case_map = infer_validation_text_case_map(validation_df)
            validation_list_like_columns: List[str] = []
            try:
                try:
                    from list_normalization import detect_list_like_columns
                except ModuleNotFoundError:
                    from agents.list_normalization import detect_list_like_columns

                validation_list_like_columns = [
                    str(c)
                    for c in detect_list_like_columns(
                        [validation_df],
                        exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id"},
                    )
                ]
            except Exception:
                validation_list_like_columns = []
            validation_style_for_clamp = {
                "inferred_country_output_format": infer_country_output_format_from_validation(validation_df),
                "lowercase_columns_hint": [
                    c for c, meta in case_map.items() if isinstance(meta, dict) and bool(meta.get("prefer_lowercase"))
                ],
                "validation_list_like_columns_hint": validation_list_like_columns,
            }
    except Exception:
        validation_style_for_clamp = {}

    if overall_acc < 0.85 and attempts < 3:
        cluster_updates = agent.cluster_analysis(local_state)
        local_state.update(cluster_updates)

        reasoning_updates = agent.evaluation_reasoning(local_state)
        local_state.update(reasoning_updates)

        normalization_assessment = agent._assess_normalization_with_llm(local_state)
        if isinstance(normalization_assessment, dict):
            normalization_directives = _sanitize_directives_to_validation_style(
                normalization_assessment,
                local_state,
                validation_style_for_clamp,
            )
            local_state["normalization_directives"] = normalization_directives

        llm_needs_normalization = bool(normalization_assessment.get("needs_normalization", False))
        llm_reasons = (
            [str(r) for r in normalization_assessment.get("reasons", []) if str(r).strip()]
            if isinstance(normalization_assessment, dict)
            else []
        )

        debug_signals = collect_eval_debug_signals()
        fallback_normalization_needed, fallback_reasons = helper_detect_normalization_issue(
            diagnostics_report=local_state.get("integration_diagnostics_report", {}),
            evaluation_analysis=local_state.get("evaluation_analysis", ""),
            metrics=metrics,
            debug_signals=debug_signals,
        )
        normalization_needed = llm_needs_normalization or fallback_normalization_needed
        normalization_reasons = llm_reasons if llm_reasons else fallback_reasons

        if normalization_needed and int(local_state.get("normalization_attempts", 0)) < 3:
            investigator_decision = "normalization_node"
            print("[*] Investigator decision: rerun normalization before next pipeline iteration")
        else:
            investigator_decision = "pipeline_adaption"
            print("[*] Investigator decision: iterate pipeline adaptation")
    else:
        investigator_decision = "human_review_export"
        print("[*] Investigator decision: proceed to human review")

    diagnostics_report = local_state.get("integration_diagnostics_report", {})
    action_plan = _build_investigator_action_plan(
        diagnostics_report if isinstance(diagnostics_report, dict) else {}
    )
    findings = diagnostics_report.get("findings", []) if isinstance(diagnostics_report, dict) else []
    finding_summaries: List[str] = []
    for finding in findings[:3] if isinstance(findings, list) else []:
        if isinstance(finding, dict):
            severity = finding.get("severity", "info")
            title = finding.get("title") or finding.get("issue") or finding.get("message") or str(finding)
            finding_summaries.append(f"{severity}: {str(title)[:140]}")
        else:
            finding_summaries.append(str(finding)[:140])

    cluster_report = local_state.get("cluster_analysis_result", {}) if isinstance(local_state, dict) else {}
    cluster_overall = cluster_report.get("_overall", {}) if isinstance(cluster_report, dict) else {}
    cluster_strategy = (
        cluster_overall.get("recommended_strategy", "None") if isinstance(cluster_overall, dict) else "None"
    )
    cluster_files = []
    try:
        cluster_files = sorted(list((cluster_report.get("_investigation", {}).get("files", {}) or {}).keys()))
    except Exception:
        cluster_files = []

    analysis_preview = str(local_state.get("evaluation_analysis", "")).strip().replace("\n", " ")
    if len(analysis_preview) > 220:
        analysis_preview = analysis_preview[:220] + "..."

    print("[INVESTIGATION REPORT]")
    print("  Investigation Performed:")
    print("    - Ran integration diagnostics script and loaded findings")
    print("    - Re-read evaluation metrics and auto diagnostics")
    if overall_acc < 0.85 and attempts < 3:
        print("    - Ran cluster analysis on latest pairwise correspondence files")
        print(f"    - Cluster files inspected: {len(cluster_files)}")
    else:
        print("    - Skipped cluster analysis (target reached or retry budget exhausted)")

    print("  Problem Determination:")
    print(f"    - overall_accuracy={overall_acc:.3%}, evaluation_attempt={attempts}/3")
    if finding_summaries:
        for summary in finding_summaries:
            print(f"    - diagnostics finding: {summary}")
    else:
        print("    - diagnostics finding: none reported")
    if overall_acc < 0.85 and attempts < 3:
        print(f"    - cluster recommendation: {cluster_strategy}")
    if analysis_preview:
        print(f"    - reasoning summary: {analysis_preview}")
    if isinstance(normalization_directives, dict) and normalization_directives:
        rec_country = normalization_directives.get("country_columns", [])
        rec_list = normalization_directives.get("list_columns", [])
        rec_lower = normalization_directives.get("lowercase_columns", [])
        rec_country_format = normalization_directives.get("country_output_format")
        if rec_country:
            print(
                f"    - suggested country normalization columns: {', '.join([str(c) for c in rec_country[:8]])}"
            )
        if rec_country_format:
            print(f"    - suggested country output format: {rec_country_format}")
        if rec_list:
            print(f"    - suggested list normalization columns: {', '.join([str(c) for c in rec_list[:8]])}")
        if rec_lower:
            print(
                f"    - suggested lowercase columns (validation-style): {', '.join([str(c) for c in rec_lower[:8]])}"
            )

    print("  Recommendation:")
    if investigator_decision == "normalization_node":
        print("    - rerun normalization before adapting pipeline")
        if normalization_reasons:
            for reason in normalization_reasons[:6]:
                print(f"    - normalization trigger: {reason}")
    elif investigator_decision == "pipeline_adaption":
        print("    - run pipeline_adaption with targeted fixes from reasoning/diagnostics")
        if action_plan:
            print("    - ranked action plan (top):")
            for item in action_plan[:3]:
                attrs = ", ".join(item.get("target_attributes", [])[:5]) or "n/a"
                print(
                    f"      * [{item.get('priority_score', 0):.3f}] {item.get('action')} "
                    f"(targets: {attrs}, confidence={item.get('confidence', 0):.2f})"
                )
    else:
        print("    - proceed to human_review_export")

    return {
        **diagnostics_updates,
        **decision_updates,
        **cluster_updates,
        **reasoning_updates,
        "investigator_decision": investigator_decision,
        "normalization_rework_required": normalization_needed,
        "normalization_rework_reasons": normalization_reasons,
        "normalization_directives": normalization_directives,
        "investigator_action_plan": action_plan,
    }
