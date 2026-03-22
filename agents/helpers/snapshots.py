"""Snapshot utilities for capturing pipeline and evaluation attempt artifacts."""

import glob
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List

import config


def run_path(run_output_root: str, *parts: str) -> str:
    return os.path.join(run_output_root, *parts)


def snapshot_file(run_output_root: str, src_path: str, dst_rel_path: str) -> str:
    if not src_path or not os.path.exists(src_path):
        return ""
    dst_path = run_path(run_output_root, dst_rel_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    try:
        shutil.copy2(src_path, dst_path)
        return dst_path
    except Exception:
        return ""


def snapshot_patterns(run_output_root: str, patterns: List[str], dst_rel_dir: str) -> List[str]:
    captured: List[str] = []
    if not patterns:
        return captured
    dst_dir = run_path(run_output_root, dst_rel_dir)
    os.makedirs(dst_dir, exist_ok=True)
    seen: set[str] = set()
    for pattern in patterns:
        for src in sorted(glob.glob(pattern)):
            if src in seen:
                continue
            seen.add(src)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(dst_dir, os.path.basename(src))
            try:
                shutil.copy2(src, dst)
                captured.append(dst)
            except Exception:
                continue
    return captured


def snapshot_pipeline_attempt(
    run_output_root: str,
    *,
    cycle_index: int,
    exec_attempt: int,
    execution_result: str,
    stdout: str,
    stderr: str,
    fusion_size_comparison: Dict[str, Any],
) -> Dict[str, Any]:
    rel_dir = os.path.join("pipeline", f"cycle_{cycle_index:02d}", f"exec_{exec_attempt:02d}")
    os.makedirs(run_path(run_output_root, rel_dir), exist_ok=True)
    snapshot: Dict[str, Any] = {
        "cycle_index": cycle_index,
        "exec_attempt": exec_attempt,
        "execution_result": execution_result,
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": {},
        "fusion_size_comparison": fusion_size_comparison if isinstance(fusion_size_comparison, dict) else {},
    }

    pipeline_copy = snapshot_file(run_output_root, config.PIPELINE_CODE_PATH, os.path.join(rel_dir, "pipeline.py"))
    if pipeline_copy:
        snapshot["files"]["pipeline_code"] = pipeline_copy
    fusion_copy = snapshot_file(run_output_root, config.FUSED_OUTPUT_PATH, os.path.join(rel_dir, "fusion_data.csv"))
    if fusion_copy:
        snapshot["files"]["fusion_data"] = fusion_copy
    estimate_copy = snapshot_file(
        run_output_root,
        os.path.join(config.OUTPUT_DIR, "pipeline_evaluation", "fusion_size_estimate.json"),
        os.path.join(rel_dir, "fusion_size_estimate.json"),
    )
    if estimate_copy:
        snapshot["files"]["fusion_size_estimate"] = estimate_copy
    corr_copies = snapshot_patterns(
        run_output_root,
        [os.path.join(config.CORRESPONDENCES_DIR, "correspondences_*.csv")],
        os.path.join(rel_dir, "correspondences"),
    )
    if corr_copies:
        snapshot["files"]["correspondences"] = corr_copies

    io_path = run_path(run_output_root, rel_dir, "pipeline_execution.json")
    try:
        with open(io_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "execution_result": execution_result,
                    "stdout": stdout or "",
                    "stderr": stderr or "",
                    "captured_at": snapshot["captured_at"],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        snapshot["files"]["execution_log"] = io_path
    except Exception:
        pass
    return snapshot


def snapshot_evaluation_attempt(
    run_output_root: str,
    *,
    stage: str,
    cycle_index: int,
    exec_attempt: int,
    execution_result: str,
    stdout: str,
    stderr: str,
    metrics: Dict[str, Any],
    metrics_source: str,
) -> Dict[str, Any]:
    safe_stage = str(stage or "unknown").replace("/", "_")
    rel_dir = os.path.join("evaluation", safe_stage, f"cycle_{cycle_index:02d}", f"exec_{exec_attempt:02d}")
    os.makedirs(run_path(run_output_root, rel_dir), exist_ok=True)
    snapshot: Dict[str, Any] = {
        "stage": safe_stage,
        "cycle_index": cycle_index,
        "exec_attempt": exec_attempt,
        "execution_result": execution_result,
        "metrics_source": metrics_source,
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "metrics": metrics if isinstance(metrics, dict) else {},
        "files": {},
    }

    eval_code_copy = snapshot_file(run_output_root, config.EVALUATION_CODE_PATH, os.path.join(rel_dir, "evaluation.py"))
    if eval_code_copy:
        snapshot["files"]["evaluation_code"] = eval_code_copy
    eval_json_copy = snapshot_file(
        run_output_root,
        config.EVALUATION_JSON_PATH,
        os.path.join(rel_dir, "pipeline_evaluation.json"),
    )
    if eval_json_copy:
        snapshot["files"]["pipeline_evaluation"] = eval_json_copy
    debug_copy = snapshot_file(
        run_output_root,
        config.DEBUG_EVAL_JSONL_PATH,
        os.path.join(rel_dir, "debug_fusion_eval.jsonl"),
    )
    if debug_copy:
        snapshot["files"]["debug_fusion_eval"] = debug_copy

    io_path = run_path(run_output_root, rel_dir, "evaluation_execution.json")
    try:
        with open(io_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "execution_result": execution_result,
                    "metrics_source": metrics_source,
                    "metrics": metrics if isinstance(metrics, dict) else {},
                    "stdout": stdout or "",
                    "stderr": stderr or "",
                    "captured_at": snapshot["captured_at"],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        snapshot["files"]["execution_log"] = io_path
    except Exception:
        pass
    return snapshot
