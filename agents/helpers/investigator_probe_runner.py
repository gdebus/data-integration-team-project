import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, List

MAX_PROBES = 4
MAX_CUSTOM_PROBES = 2
MAX_RUNTIME_SECONDS = 1.5
MAX_EVENTS = 500
CUSTOM_PROBE_TIMEOUT_SECONDS = 2.0


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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
    metrics = state.get("evaluation_metrics", {}) if isinstance(state, dict) else {}
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


PROBE_REGISTRY = {
    "reason_distribution": _probe_reason_distribution,
    "worst_attributes": _probe_worst_attributes,
    "recent_mismatches": _probe_recent_mismatches,
    "directive_coverage": _probe_directive_coverage,
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
    for name in ("reason_distribution", "worst_attributes"):
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
