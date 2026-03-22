"""Sanity-check evaluation metrics before the agent trusts them.

Catches cases where the LLM-generated evaluation code:
- Evaluated wrong/missing file
- Evaluated only a subset of rows
- Produced NaN-only attribute metrics
- Used obviously broken comparators
"""
import os
import json
import pandas as pd
from typing import Any, Dict, List, Tuple

import config
# NOTE: FUSED_OUTPUT_PATH and EVALUATION_JSON_PATH are accessed via config.X
# (not imported directly) because configure_run_output() mutates them at runtime.


def check_evaluation_sanity(
    metrics: Dict[str, Any],
    fused_path: str = "",
    eval_testset_path: str = "",
) -> Tuple[bool, List[str]]:
    """Validates evaluation metrics for basic sanity.

    Returns (passed, warnings). When passed is False, the evaluation
    results are unreliable.
    """
    if not fused_path:
        fused_path = config.FUSED_OUTPUT_PATH
    warnings: List[str] = []

    if not isinstance(metrics, dict) or not metrics:
        return False, ["Metrics payload is empty or not a dict"]

    # overall_accuracy presence and numeric validity
    overall = metrics.get("overall_accuracy", metrics.get("overall"))
    if overall is None:
        warnings.append("No overall_accuracy in metrics")
    else:
        try:
            overall_f = float(overall)
            if overall_f < 0 or overall_f > 1.0:
                warnings.append(f"overall_accuracy={overall_f} is outside [0, 1] range")
        except (ValueError, TypeError):
            warnings.append(f"overall_accuracy is not a number: {overall}")

    # Per-attribute metrics presence
    attr_keys = [k for k in metrics.keys() if not k.startswith("overall") and not k.startswith("_") and k not in ("evaluation_functions", "eval_stage", "evaluated_rows", "total_rows")]
    if len(attr_keys) == 0:
        warnings.append("No per-attribute metrics found — evaluation may not have run correctly")

    # All-zero or all-NaN attribute detection
    all_zero_attrs = []
    for key in attr_keys:
        val = metrics[key]
        if isinstance(val, dict):
            acc = val.get("accuracy")
        else:
            acc = val
        try:
            acc_f = float(acc) if acc is not None else None
        except (ValueError, TypeError):
            acc_f = None
        if acc_f is not None and acc_f == 0.0:
            all_zero_attrs.append(key)

    if len(all_zero_attrs) > 0 and len(all_zero_attrs) == len(attr_keys):
        warnings.append(f"ALL {len(all_zero_attrs)} attributes have 0% accuracy — evaluation code is likely broken")
        return False, warnings
    elif len(all_zero_attrs) > len(attr_keys) * 0.8:
        warnings.append(f"{len(all_zero_attrs)}/{len(attr_keys)} attributes have 0% accuracy — evaluation may be broken")

    # Row count sanity: evaluated rows vs fused output
    evaluated_rows = metrics.get("evaluated_rows", metrics.get("total_rows", metrics.get("num_rows")))
    if evaluated_rows is not None and os.path.exists(fused_path):
        try:
            fused_df = pd.read_csv(fused_path)
            fused_rows = len(fused_df)
            eval_rows = int(evaluated_rows)

            if eval_rows == 0:
                warnings.append("Evaluation reports 0 rows evaluated")
                return False, warnings

            # Less than 10% coverage is a red flag
            if fused_rows > 0 and eval_rows < fused_rows * 0.1:
                warnings.append(f"Evaluated only {eval_rows} rows but fused output has {fused_rows} rows ({eval_rows/fused_rows:.0%} coverage)")
        except Exception as e:
            print(f"[EVAL SANITY] Row count comparison failed: {e}")

    # Testset file existence
    if eval_testset_path and not os.path.exists(eval_testset_path):
        warnings.append(f"Evaluation testset not found at: {eval_testset_path}")

    # Suspiciously perfect metrics (possible self-comparison)
    try:
        if float(overall or 0) == 1.0 and len(attr_keys) > 3:
            all_perfect = all(
                float(metrics[k].get("accuracy", 0) if isinstance(metrics[k], dict) else metrics[k]) == 1.0
                for k in attr_keys
            )
            if all_perfect:
                warnings.append("All attributes report 100% accuracy — evaluation may be comparing a file to itself")
    except Exception as e:
        print(f"[EVAL SANITY] Perfect-metrics check failed: {e}")

    # Critical warnings cause failure; non-critical ones pass with warnings
    critical = any(
        "likely broken" in w or "0 rows" in w or "ALL" in w or "comparing a file to itself" in w
        for w in warnings
    )

    return (not critical), warnings
