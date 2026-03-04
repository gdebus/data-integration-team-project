import glob
import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_jsonl(path: str, limit: int = 5000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    out.append(item)
    except Exception:
        return []
    return out


def _find_latest_file(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        return ""
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "n/a"


def _extract_run_id(path: str) -> str:
    name = os.path.basename(path)
    if name.startswith("run_") and name.endswith(".json"):
        return name[len("run_") : -len(".json")]
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_agent_run_report(
    *,
    output_root: str = "output",
    results_file: Optional[str] = None,
) -> str:
    results_dir = os.path.join(output_root, "results")
    if not results_file:
        results_file = _find_latest_file(os.path.join(results_dir, "run_*.json"))

    run_data = _read_json(results_file) if results_file else {}
    run_id = _extract_run_id(results_file) if results_file else datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(results_dir, f"run_{run_id}_report.md")

    blocking_cfg = _read_json(os.path.join(output_root, "blocking-evaluation", "blocking_config.json"))
    matching_cfg = _read_json(os.path.join(output_root, "matching-evaluation", "matching_config.json"))
    blocking_trace = _read_jsonl(os.path.join(output_root, "blocking-evaluation", "llm_blocking_trace.jsonl"))
    matching_trace = _read_jsonl(os.path.join(output_root, "matching-evaluation", "llm_matching_trace.jsonl"))
    investigator_history = _read_jsonl(os.path.join(output_root, "profile", "investigator_history.jsonl"))

    normalization_report = {}
    norm_attempt_dir = _find_latest_file(os.path.join(output_root, "normalization", "attempt_*"))
    if norm_attempt_dir and os.path.isdir(norm_attempt_dir):
        normalization_report = _read_json(os.path.join(norm_attempt_dir, "normalization_report.json"))

    failure_tags = Counter()
    for event in blocking_trace + matching_trace:
        tags = []
        if isinstance(event.get("result"), dict):
            tags = event.get("result", {}).get("failure_tags", [])
        if not tags:
            tags = event.get("failure_tags", [])
        if isinstance(tags, list):
            failure_tags.update([str(t) for t in tags if str(t)])

    lines: List[str] = []
    lines.append(f"# Agent Run Report ({run_id})")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Results file: `{results_file or 'n/a'}`")
    lines.append(f"- Overall accuracy: {_format_pct(run_data.get('evaluation_metrics', {}).get('overall_accuracy'))}")
    lines.append(
        f"- Final held-out overall: {_format_pct(run_data.get('sealed_final_test_metrics', {}).get('overall_accuracy'))}"
    )
    lines.append(
        f"- Token usage: {int(run_data.get('token_usage', {}).get('total_tokens', 0) or 0):,}, "
        f"cost=${float(run_data.get('token_usage', {}).get('total_cost', 0.0) or 0.0):.4f}"
    )
    lines.append("")

    lines.append("## Blocking Decisions")
    blocking_strategies = blocking_cfg.get("blocking_strategies", {}) if isinstance(blocking_cfg, dict) else {}
    if blocking_strategies:
        for pair, cfg in blocking_strategies.items():
            lines.append(
                f"- `{pair}`: strategy=`{cfg.get('strategy')}`, PC={float(cfg.get('pair_completeness', 0.0) or 0.0):.4f}, "
                f"candidates={int(cfg.get('num_candidates', 0) or 0):,}, tags={cfg.get('failure_tags', [])}"
            )
    else:
        lines.append("- No blocking summary found.")
    lines.append("")

    lines.append("## Matching Decisions")
    matching_strategies = matching_cfg.get("matching_strategies", {}) if isinstance(matching_cfg, dict) else {}
    if matching_strategies:
        for pair, cfg in matching_strategies.items():
            lines.append(
                f"- `{pair}`: F1={float(cfg.get('f1', 0.0) or 0.0):.4f}, proxy={cfg.get('proxy_f1')}, "
                f"skipped={bool(cfg.get('skipped_due_to_blocking_gate', False))}, tags={cfg.get('failure_tags', [])}"
            )
    else:
        lines.append("- No matching summary found.")
    lines.append("")

    lines.append("## Failure Taxonomy")
    if failure_tags:
        for tag, count in failure_tags.most_common():
            lines.append(f"- `{tag}`: {count}")
    else:
        lines.append("- No failure tags found.")
    lines.append("")

    lines.append("## Normalization")
    if normalization_report:
        lines.append(f"- Status: `{normalization_report.get('status')}`")
        lines.append(f"- Failure tags: {normalization_report.get('failure_tags', [])}")
        ablation = normalization_report.get("ablation_report", {}) if isinstance(normalization_report, dict) else {}
        if isinstance(ablation, dict):
            lines.append(f"- Selected directive groups: {ablation.get('selected_keys', [])}")
            subsets = ablation.get("evaluated_subsets", [])
            if isinstance(subsets, list) and subsets:
                top = subsets[:3]
                for item in top:
                    lines.append(
                        f"  - `{item.get('subset')}`: projected_delta={item.get('projected_delta')}, allow={item.get('allow')}"
                    )
    else:
        lines.append("- No normalization report found.")
    lines.append("")

    lines.append("## Investigator")
    if investigator_history:
        latest = investigator_history[-1]
        lines.append(f"- Last decision: `{latest.get('decision')}`")
        lines.append(
            f"- Routing score/threshold: {latest.get('routing_decision', {}).get('score')} / "
            f"{latest.get('routing_decision', {}).get('threshold')}"
        )
        lines.append(f"- Probe summary: {latest.get('probe_summary', {}).get('summary', 'n/a')}")
    else:
        lines.append("- No investigator history found.")
    lines.append("")

    lines.append("## LLM Timeline (Recent)")
    recent = (blocking_trace + matching_trace)[-10:]
    if recent:
        for item in recent:
            pair = item.get("pair", "n/a")
            attempt = item.get("attempt", "-")
            decision = item.get("decision", "evaluate")
            result = item.get("result", {}) if isinstance(item.get("result"), dict) else {}
            pc = result.get("pair_completeness")
            tags = result.get("failure_tags", item.get("failure_tags", []))
            lines.append(f"- `{pair}` attempt {attempt}: {decision} pc={pc} tags={tags}")
    else:
        lines.append("- No LLM trace events found.")

    os.makedirs(results_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
    return report_path
