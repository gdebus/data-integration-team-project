"""Subprocess execution for pipeline and evaluation scripts.

Pure I/O functions with no class dependencies — they run a script,
capture output, classify errors, and return a result dict.
"""

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

import config
from config import (
    EVAL_EXEC_TIMEOUT,
    PIPELINE_EXEC_TIMEOUT,
)
from helpers.error_classifier import classify_execution_error
from helpers.metrics import extract_metrics_from_text, is_metrics_payload
from helpers.snapshots import snapshot_evaluation_attempt, snapshot_pipeline_attempt


def run_pipeline_subprocess(
    state: Dict[str, Any],
    run_output_root: str,
    compare_estimates_fn,
    logger=None,
) -> Dict[str, Any]:
    """Executes the generated pipeline script and returns state updates."""
    attempts = state.get("pipeline_execution_attempts", 0) + 1
    pipeline_path = config.PIPELINE_CODE_PATH
    run_started_at = time.time()
    fusion_size_comparison: Dict[str, Any] = {}
    pipeline_stdout = ""
    pipeline_stderr = ""
    error_info: Dict[str, Any] = {}
    execution_result = "error: not executed"

    try:
        result = subprocess.run(
            [sys.executable, pipeline_path],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=PIPELINE_EXEC_TIMEOUT,
        )
        pipeline_stdout = result.stdout or ""
        pipeline_stderr = result.stderr or ""

        if result.returncode == 0:
            execution_result = "success"
            print(f"[EXECUTE] Pipeline completed successfully")

            estimate_path = os.path.join(config.OUTPUT_DIR, "pipeline_evaluation", "fusion_size_estimate.json")
            fused_csv_path = config.FUSED_OUTPUT_PATH
            if os.path.exists(estimate_path) and os.path.exists(fused_csv_path):
                try:
                    fusion_size_comparison = compare_estimates_fn(
                        fusion_csv_path=fused_csv_path,
                        estimate_path=estimate_path,
                    )
                    # Print fusion size summary
                    _actual = fusion_size_comparison.get("actual", {})
                    _comparisons = fusion_size_comparison.get("comparisons", {})
                    _best = _comparisons.get("matching", _comparisons.get("blocking", {}))
                    _actual_rows = _actual.get("rows", 0)
                    _actual_ids = _actual.get("unique_ids", 0)
                    _expected = _best.get("expected_rows", 0)
                    _pct = _best.get("rows_pct_error", 0)
                    if _actual_rows and _expected:
                        _dir = "overcount" if _pct > 0.01 else ("undercount" if _pct < -0.01 else "on target")
                        print(f"[FUSION SIZE] {_actual_rows:,} rows ({_actual_ids:,} unique) | "
                              f"expected ~{_expected:,} | {abs(_pct):.1%} {_dir}")
                except Exception as e:
                    print(f"[EXECUTE] Fusion size comparison failed: {e}")
        else:
            error_info = classify_execution_error(
                returncode=result.returncode,
                stderr=result.stderr or "",
                stdout=result.stdout or "",
            )
            execution_result = f"error: [{error_info['category']}] {error_info['suggestion']}"
            print(f"[EXECUTE] Pipeline failed: {error_info['category']} — {error_info['suggestion']}")

    except subprocess.TimeoutExpired:
        error_info = classify_execution_error(0, "", timed_out=True)
        execution_result = f"error: [{error_info['category']}] {error_info['suggestion']}"
        pipeline_stderr = "TimeoutExpired"
        print(f"[EXECUTE] Pipeline timed out after {PIPELINE_EXEC_TIMEOUT}s")
    except Exception as e:
        error_info = classify_execution_error(0, str(e))
        execution_result = f"error: [{error_info['category']}] {error_info['suggestion']}"
        pipeline_stderr = str(e)
        print(f"[EXECUTE] Pipeline exception: {str(e)[:200]}")

    cycle_index = int(state.get("evaluation_attempts", 0)) + 1
    pipeline_snapshot = snapshot_pipeline_attempt(
        run_output_root=run_output_root,
        cycle_index=cycle_index,
        exec_attempt=attempts,
        execution_result=execution_result,
        stdout=pipeline_stdout if isinstance(pipeline_stdout, str) else str(pipeline_stdout),
        stderr=pipeline_stderr if isinstance(pipeline_stderr, str) else str(pipeline_stderr),
        fusion_size_comparison=fusion_size_comparison if isinstance(fusion_size_comparison, dict) else {},
    )
    pipeline_snapshots = list(state.get("pipeline_snapshots", []) or [])
    pipeline_snapshots.append(pipeline_snapshot)

    if logger:
        logger.info("Pipeline execution result: " + execution_result)

    return {
        "pipeline_execution_result": execution_result,
        "pipeline_execution_attempts": attempts,
        "pipeline_run_started_at": run_started_at,
        "pipeline_run_finished_at": time.time(),
        "fusion_size_comparison": fusion_size_comparison,
        "pipeline_snapshots": pipeline_snapshots,
        "pipeline_error_classification": error_info,
    }


def run_evaluation_subprocess(
    state: Dict[str, Any],
    run_output_root: str,
    stage_label: str,
    logger=None,
) -> Dict[str, Any]:
    """Executes the generated evaluation script and returns state updates."""
    attempts = state.get("evaluation_execution_attempts", 0) + 1
    evaluation_path = config.EVALUATION_CODE_PATH

    evaluation_stdout = ""
    evaluation_stderr = ""
    metrics_from_execution: Dict[str, Any] = {}
    metrics_source = "none"
    error_info: Dict[str, Any] = {}
    cycle_index = int(state.get("evaluation_attempts", 0)) + 1
    eval_started_at = time.time()
    eval_metrics_path = config.EVALUATION_JSON_PATH
    execution_result = "error: not executed"

    try:
        result = subprocess.run(
            [sys.executable, evaluation_path],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=EVAL_EXEC_TIMEOUT,
        )
        evaluation_stdout = result.stdout or ""
        evaluation_stderr = result.stderr or ""

        if result.returncode == 0:
            execution_result = "success"
            print("[EVAL EXEC] Evaluation completed successfully")
        else:
            error_info = classify_execution_error(
                returncode=result.returncode,
                stderr=result.stderr or "",
                stdout=result.stdout or "",
            )
            execution_result = f"error: [{error_info['category']}] {error_info['suggestion']}"
            print(f"[EVAL EXEC] Evaluation failed: {error_info['category']} — {error_info['suggestion']}")

        metrics_from_stdout = extract_metrics_from_text(evaluation_stdout)
        metrics_from_file: Dict[str, Any] = {}
        if os.path.exists(eval_metrics_path):
            try:
                if os.path.getmtime(eval_metrics_path) >= (eval_started_at - 1.0):
                    with open(eval_metrics_path, "r", encoding="utf-8") as f:
                        parsed = json.load(f)
                    if isinstance(parsed, dict):
                        metrics_from_file = parsed
            except Exception as e:
                print(f"[EVAL EXEC] Failed to read metrics file {eval_metrics_path}: {e}")
                metrics_from_file = {}

        if is_metrics_payload(metrics_from_stdout):
            metrics_from_execution = metrics_from_stdout
            metrics_source = "stdout"
            try:
                os.makedirs(os.path.dirname(eval_metrics_path), exist_ok=True)
                with open(eval_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics_from_execution, f, indent=4)
            except Exception as e:
                print(f"[EVAL EXEC] Failed to write metrics to {eval_metrics_path}: {e}")
        elif is_metrics_payload(metrics_from_file):
            metrics_from_execution = metrics_from_file
            metrics_source = "pipeline_evaluation_file"

    except subprocess.TimeoutExpired:
        error_info = classify_execution_error(0, "", timed_out=True)
        execution_result = f"error: [{error_info['category']}] {error_info['suggestion']}"
    except Exception as e:
        error_info = classify_execution_error(0, str(e))
        execution_result = f"error: [{error_info['category']}] {error_info['suggestion']}"

    evaluation_snapshot = snapshot_evaluation_attempt(
        run_output_root=run_output_root,
        stage=stage_label,
        cycle_index=cycle_index,
        exec_attempt=attempts,
        execution_result=execution_result,
        stdout=evaluation_stdout if isinstance(evaluation_stdout, str) else str(evaluation_stdout),
        stderr=evaluation_stderr if isinstance(evaluation_stderr, str) else str(evaluation_stderr),
        metrics=metrics_from_execution if isinstance(metrics_from_execution, dict) else {},
        metrics_source=metrics_source,
    )
    evaluation_snapshots = list(state.get("evaluation_snapshots", []) or [])
    evaluation_snapshots.append(evaluation_snapshot)

    if logger:
        logger.info("Evaluation execution result: " + execution_result)

    return {
        "evaluation_execution_result": execution_result,
        "evaluation_execution_attempts": attempts,
        "evaluation_execution_stdout": evaluation_stdout,
        "evaluation_execution_stderr": evaluation_stderr,
        "evaluation_metrics_from_execution": metrics_from_execution,
        "evaluation_metrics_source": metrics_source,
        "evaluation_snapshots": evaluation_snapshots,
        "evaluation_error_classification": error_info,
        "_eval_node_graph_retries": int(state.get("_eval_node_graph_retries", 0)) + (0 if execution_result == "success" else 1),
    }
