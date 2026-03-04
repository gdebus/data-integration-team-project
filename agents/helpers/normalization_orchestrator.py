import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from helpers.normalization_policy import (
    infer_country_output_format_from_validation,
    infer_validation_text_case_map,
)
from helpers.investigator_acceptance import create_pending_normalization_acceptance
try:
    from workflow_logging import log_agent_action
except Exception:
    try:
        from agents.workflow_logging import log_agent_action
    except Exception:
        from helpers.workflow_logging import log_agent_action

SHADOW_MIN_PROJECTED_DELTA = 0.002
ABLATION_KEYS = ("list_columns", "country_columns", "lowercase_columns", "text_columns")


def _clamp_directives_to_validation_style(
    directives: Dict[str, Any],
    validation_columns: set[str],
    validation_lowercase_columns: set[str],
    validation_list_like_columns: set[str],
    inferred_country_output_format: str,
) -> Dict[str, Any]:
    """
    Keep directives compatible with validation style without discarding useful hints.
    """
    if not isinstance(directives, dict):
        directives = {}
    out = dict(directives)

    allowed_validation_cols = {str(c).lower() for c in validation_columns if str(c).strip()}
    allowed_lower = {str(c).lower() for c in validation_lowercase_columns if str(c).strip()}
    allowed_list = {str(c).lower() for c in validation_list_like_columns if str(c).strip()}

    def _filter(values: Any, allowed: set[str]) -> List[str]:
        if not isinstance(values, list):
            return []
        kept: List[str] = []
        seen = set()
        for v in values:
            s = str(v).strip()
            if not s:
                continue
            s_l = s.lower()
            if allowed and s_l not in allowed:
                continue
            if allowed_validation_cols and s_l not in allowed_validation_cols:
                continue
            if s_l in seen:
                continue
            seen.add(s_l)
            kept.append(s)
        return kept

    hinted_list = _filter(out.get("list_columns", []), allowed_validation_cols or allowed_list)
    for col in sorted(allowed_list):
        if col not in {c.lower() for c in hinted_list}:
            hinted_list.append(col)
    out["list_columns"] = hinted_list
    out["lowercase_columns"] = _filter(out.get("lowercase_columns", []), allowed_lower)

    country_cols = out.get("country_columns", [])
    if not isinstance(country_cols, list):
        country_cols = []
    filtered_country_cols: List[str] = []
    for c in country_cols:
        s = str(c).strip()
        if not s:
            continue
        s_l = s.lower()
        if allowed_validation_cols and s_l not in allowed_validation_cols:
            continue
        if "country" not in s_l:
            continue
        filtered_country_cols.append(s)
    out["country_columns"] = filtered_country_cols

    # Validation representation is authoritative for country normalization.
    out["country_output_format"] = inferred_country_output_format
    out["list_normalization_required"] = bool(out.get("list_columns", []))
    return out


def _compute_shadow_normalization_precheck(
    *,
    dataset_paths: List[str],
    load_dataset_fn,
    directives: Dict[str, Any],
    metrics: Dict[str, Any],
    auto_diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    directive_cols: set[str] = set()
    for key in ("list_columns", "country_columns", "text_columns", "lowercase_columns"):
        values = directives.get(key, []) if isinstance(directives, dict) else []
        if isinstance(values, list):
            directive_cols.update([str(v).strip().lower() for v in values if str(v).strip()])

    low_acc_cols: set[str] = set()
    for k, v in (metrics.items() if isinstance(metrics, dict) else []):
        if not (isinstance(k, str) and k.endswith("_accuracy")):
            continue
        if k in {"overall_accuracy", "macro_accuracy"}:
            continue
        try:
            if float(v) < 0.60:
                low_acc_cols.add(k[: -len("_accuracy")].lower())
        except Exception:
            continue

    matched_problem_cols = low_acc_cols.intersection(directive_cols)
    overlap = len(matched_problem_cols) / max(1, len(low_acc_cols))

    present_hits = 0
    total_checks = 0
    sample_rows = 0
    for path in dataset_paths[:4]:
        try:
            df = load_dataset_fn(path)
            if df is None or df.empty:
                continue
            sample_rows += min(200, len(df))
            cols = {str(c).strip().lower() for c in df.columns}
            for col in directive_cols:
                total_checks += 1
                if col in cols:
                    present_hits += 1
        except Exception:
            continue
    coverage = present_hits / max(1, total_checks)

    ratios = auto_diagnostics.get("debug_reason_ratios", {}) if isinstance(auto_diagnostics, dict) else {}
    norm_pressure = 0.0
    if isinstance(ratios, dict):
        for key, value in ratios.items():
            if any(x in str(key).lower() for x in ("mismatch", "list", "type", "format", "encoding")):
                try:
                    norm_pressure += float(value)
                except Exception:
                    pass
    norm_pressure = max(0.0, min(1.0, norm_pressure))

    projected_delta = (
        (0.008 * overlap)
        + (0.005 * coverage)
        + (0.004 * norm_pressure)
        - (0.003 * (1.0 - coverage if total_checks > 0 else 1.0))
    )
    allow = projected_delta >= SHADOW_MIN_PROJECTED_DELTA
    return {
        "allow": allow,
        "projected_delta": round(projected_delta, 6),
        "min_projected_delta": SHADOW_MIN_PROJECTED_DELTA,
        "overlap_with_low_accuracy": round(overlap, 6),
        "directive_coverage": round(coverage, 6),
        "normalization_pressure": round(norm_pressure, 6),
        "matched_problem_columns": sorted(list(matched_problem_cols)),
        "sample_rows_scanned": sample_rows,
        "directive_columns": sorted(list(directive_cols)),
        "reason": "insufficient projected gain" if not allow else "projected gain acceptable",
    }


def _directive_subset(directives: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    if not isinstance(directives, dict):
        directives = {}
    out: Dict[str, Any] = {}
    for key in keys:
        value = directives.get(key, [])
        if isinstance(value, list) and value:
            out[key] = list(value)
    if directives.get("country_output_format"):
        out["country_output_format"] = directives.get("country_output_format")
    out["list_normalization_required"] = bool(out.get("list_columns"))
    return out


def _run_directive_ablation_precheck(
    *,
    dataset_paths: List[str],
    load_dataset_fn,
    directives: Dict[str, Any],
    metrics: Dict[str, Any],
    auto_diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    subset_specs: List[tuple[str, List[str]]] = []
    for key in ABLATION_KEYS:
        values = directives.get(key, []) if isinstance(directives, dict) else []
        if isinstance(values, list) and values:
            subset_specs.append((key, [key]))
    subset_specs.append(("combined", [k for k in ABLATION_KEYS if isinstance(directives.get(k, []), list)]))

    results: List[Dict[str, Any]] = []
    for name, keys in subset_specs:
        subset = _directive_subset(directives, keys)
        if not subset:
            continue
        score = _compute_shadow_normalization_precheck(
            dataset_paths=dataset_paths,
            load_dataset_fn=load_dataset_fn,
            directives=subset,
            metrics=metrics,
            auto_diagnostics=auto_diagnostics,
        )
        results.append(
            {
                "subset": name,
                "keys": keys,
                "projected_delta": score.get("projected_delta", 0.0),
                "allow": bool(score.get("allow", False)),
                "details": score,
            }
        )

    results.sort(key=lambda x: float(x.get("projected_delta", 0.0)), reverse=True)
    selected_keys: set[str] = set()
    for item in results:
        if item.get("subset") == "combined":
            continue
        if item.get("allow") and float(item.get("projected_delta", 0.0)) > 0.0:
            selected_keys.update(item.get("keys", []))
    if not selected_keys and results:
        best = results[0]
        if best.get("subset") != "combined":
            selected_keys.update(best.get("keys", []))
    if not selected_keys:
        selected_keys = set([k for k in ABLATION_KEYS if isinstance(directives.get(k, []), list) and directives.get(k)])

    pruned = _directive_subset(directives, sorted(list(selected_keys)))
    for key in directives.keys():
        if key not in pruned and key not in ABLATION_KEYS:
            pruned[key] = directives[key]
    return {
        "evaluated_subsets": results,
        "selected_keys": sorted(list(selected_keys)),
        "pruned_directives": pruned,
        "selection_reason": "selected subsets with positive projected gain",
    }


def _write_normalization_report(attempt_dir: str, report: Dict[str, Any]) -> None:
    try:
        os.makedirs(attempt_dir, exist_ok=True)
        with open(os.path.join(attempt_dir, "normalization_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return


def _import_list_normalization_tools():
    detect_fn = None
    normalize_fn = None
    warning = None
    try:
        from list_normalization import (
            detect_list_like_columns,
            normalize_list_like_columns,
        )
        detect_fn = detect_list_like_columns
        normalize_fn = normalize_list_like_columns
    except ModuleNotFoundError:
        try:
            module_name = "list_normalization.py"
            for candidate in (Path.cwd(), Path.cwd() / "agents"):
                if (candidate / module_name).is_file():
                    candidate_str = str(candidate.resolve())
                    if candidate_str not in sys.path:
                        sys.path.append(candidate_str)
            from list_normalization import (
                detect_list_like_columns,
                normalize_list_like_columns,
            )
            detect_fn = detect_list_like_columns
            normalize_fn = normalize_list_like_columns
        except Exception as e:
            warning = f"list_normalization import failed: {e}"
    except Exception as e:
        warning = f"list_normalization import failed: {e}"
    return detect_fn, normalize_fn, warning


def _detect_validation_list_like_columns(detect_fn, validation_df, existing_warning: str | None = None):
    warning = existing_warning
    validation_list_like_columns: set[str] = set()
    if callable(detect_fn) and validation_df is not None and not validation_df.empty:
        try:
            inferred_list_cols = detect_fn(
                [validation_df],
                exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id"},
            )
            validation_list_like_columns = {str(c).lower() for c in inferred_list_cols}
        except Exception as e:
            warnings_hint = f"validation list-like detection failed: {e}"
            warning = f"{warning}; {warnings_hint}" if warning else warnings_hint
    return validation_list_like_columns, warning


def _build_shadow_gate_skip_result(
    *,
    attempts: int,
    max_attempts: int,
    eval_path: str,
    directives: Dict[str, Any],
    ablation_report: Dict[str, Any],
    shadow_precheck: Dict[str, Any],
    gate_request: Dict[str, Any],
    dataset_paths: List[str],
) -> Dict[str, Any]:
    report = {
        "status": "skipped_by_shadow_gate",
        "attempt": attempts,
        "max_attempts": max_attempts,
        "validation_set": eval_path,
        "normalization_directives": directives,
        "ablation_report": ablation_report,
        "shadow_precheck": shadow_precheck,
        "acceptance_gate": {
            "requested": bool(gate_request),
            "request": gate_request,
            "pending": {},
        },
        "datasets": {},
        "warnings": ["Normalization rerun skipped by shadow precheck (low projected gain)."],
        "failure_tags": ["normalization_regression", "shadow_precheck_blocked"],
        "reverted_to_original": False,
        "created_at": datetime.now().isoformat(),
    }
    attempt_dir = os.path.join("output/normalization", f"attempt_{attempts}")
    _write_normalization_report(attempt_dir, report)
    return {
        "datasets": dataset_paths,
        "normalized_datasets": [],
        "normalization_execution_result": "skipped_by_shadow_gate",
        "normalization_attempts": attempts,
        "normalization_report": report,
        "normalization_directives": directives,
        "normalization_pending_acceptance": {},
        "normalization_gate_request": {},
        "normalization_rework_required": False,
        "normalization_rework_reasons": ["Shadow precheck projected insufficient gain."],
        "pipeline_execution_result": "",
        "pipeline_execution_attempts": 0,
        "evaluation_execution_result": "",
        "evaluation_execution_attempts": 0,
    }


def _detect_id_column(df: pd.DataFrame) -> Any:
    for candidate in ("id", "_id"):
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if "id" in str(col).lower():
            return col
    return None


def _build_column_transforms(
    *,
    df: pd.DataFrame,
    id_column: Any,
    attempts: int,
    validation_columns: set[str],
    validation_lowercase_columns: set[str],
    low_accuracy_columns: set[str],
    directive_lowercase_columns: set[str],
    directive_text_columns: set[str],
) -> Dict[Any, Any]:
    transforms: Dict[Any, Any] = {}
    for col in df.columns:
        c_lower = str(col).lower()
        if id_column is not None and col == id_column:
            continue
        if validation_columns and col not in validation_columns:
            continue

        series = df[col]
        ops: List[str] = []
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            ops.extend(["strip", "normalize_whitespace"])
            should_lower_from_validation = c_lower in validation_lowercase_columns
            should_lower_from_directive = c_lower in directive_lowercase_columns
            if should_lower_from_directive or (
                attempts > 1
                and (c_lower in low_accuracy_columns or c_lower in directive_text_columns)
                and should_lower_from_validation
            ):
                ops.append("lower")

        if c_lower in {"assets", "profits", "sales", "market_value", "market value", "revenue"}:
            ops = ["strip", "normalize_whitespace", "to_numeric"]

        deduped_ops: List[str] = []
        for op in ops:
            if op not in deduped_ops:
                deduped_ops.append(op)
        if deduped_ops:
            transforms[col] = deduped_ops
    return transforms


def _normalize_country_value(value: Any, normalize_country_fn, output_format: str):
    if value is None:
        return value
    try:
        if pd.isna(value):
            return value
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return value
    try:
        normalized = normalize_country_fn(text, output_format=output_format)
    except Exception:
        normalized = None
    return normalized if normalized else text


def run_normalization_node(agent, state: Dict[str, Any], load_dataset_fn):
    log_agent_action(
        agent,
        step="normalization_node",
        action="start",
        why="Normalize source datasets",
        improvement="Applies PyDI normalization with safe fallback",
    )
    print("[*] Running normalization node")

    attempts = int(state.get("normalization_attempts", 0)) + 1
    max_attempts = 3
    gate_request = state.get("normalization_gate_request", {}) if isinstance(state, dict) else {}
    if not isinstance(gate_request, dict):
        gate_request = {}

    dataset_paths = list(state.get("datasets", []) or [])
    original_paths = list(state.get("original_datasets", dataset_paths) or dataset_paths)
    if "original_datasets" not in state:
        state["original_datasets"] = list(original_paths)

    eval_path = agent._evaluation_testset_path(state, force_test=False)
    validation_df = None
    validation_columns: set[str] = set()
    try:
        if eval_path:
            validation_df = load_dataset_fn(eval_path)
            if validation_df is not None and not validation_df.empty:
                validation_columns = set(validation_df.columns.tolist())
    except Exception:
        validation_df = None
        validation_columns = set()

    metrics = state.get("evaluation_metrics", {}) if isinstance(state, dict) else {}
    low_accuracy_columns: set[str] = set()
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if not (isinstance(key, str) and key.endswith("_accuracy")):
                continue
            try:
                if float(value) < 0.60:
                    low_accuracy_columns.add(key[: -len("_accuracy")].lower())
            except Exception:
                continue

    directives = state.get("normalization_directives", {}) if isinstance(state, dict) else {}
    if not isinstance(directives, dict):
        directives = {}
    validation_case_map = infer_validation_text_case_map(validation_df)
    validation_lowercase_columns = {
        str(col).lower()
        for col, meta in validation_case_map.items()
        if isinstance(meta, dict) and bool(meta.get("prefer_lowercase"))
    }
    inferred_country_output_format = infer_country_output_format_from_validation(validation_df)

    directive_country_columns = {str(c).lower() for c in directives.get("country_columns", []) if str(c).strip()}
    country_output_format = inferred_country_output_format

    try:
        from PyDI.normalization import create_normalization_config, normalize_country, normalize_dataset
    except Exception as e:
        msg = f"PyDI normalization import failed: {e}"
        print(f"[WARN] {msg}")
        return {
            "datasets": original_paths,
            "normalized_datasets": [],
            "normalization_execution_result": f"fallback_to_original: {msg}",
            "normalization_attempts": attempts,
            "normalization_report": {
                "status": "fallback",
                "attempt": attempts,
                "reason": msg,
                "reverted_to_original": True,
            },
            "normalization_directives": directives,
            "normalization_pending_acceptance": {},
            "normalization_gate_request": {},
            "normalization_rework_required": False,
            "normalization_rework_reasons": [],
        }

    detect_list_like_columns, normalize_list_like_columns, pending_list_import_warning = _import_list_normalization_tools()
    validation_list_like_columns, pending_list_import_warning = _detect_validation_list_like_columns(
        detect_list_like_columns,
        validation_df,
        pending_list_import_warning,
    )

    directives = _clamp_directives_to_validation_style(
        directives=directives,
        validation_columns=validation_columns,
        validation_lowercase_columns=validation_lowercase_columns,
        validation_list_like_columns=validation_list_like_columns,
        inferred_country_output_format=inferred_country_output_format,
    )

    directive_country_columns = {str(c).lower() for c in directives.get("country_columns", []) if str(c).strip()}
    directive_list_columns = {str(c).lower() for c in directives.get("list_columns", []) if str(c).strip()}
    directive_text_columns = {str(c).lower() for c in directives.get("text_columns", []) if str(c).strip()}
    directive_lowercase_columns = {
        str(c).lower() for c in directives.get("lowercase_columns", []) if str(c).strip()
    }
    target_list_columns = directive_list_columns | validation_list_like_columns

    country_output_format = inferred_country_output_format
    directives["country_output_format"] = country_output_format
    directives["lowercase_columns"] = sorted(list(directive_lowercase_columns))
    directives["list_columns"] = sorted(list(target_list_columns))
    directives["list_normalization_required"] = bool(target_list_columns)

    ablation_report = _run_directive_ablation_precheck(
        dataset_paths=dataset_paths,
        load_dataset_fn=load_dataset_fn,
        directives=directives,
        metrics=metrics if isinstance(metrics, dict) else {},
        auto_diagnostics=state.get("auto_diagnostics", {}) if isinstance(state, dict) else {},
    )
    if isinstance(ablation_report, dict):
        directives = ablation_report.get("pruned_directives", directives) or directives
        directives["list_normalization_required"] = bool(directives.get("list_columns"))

    shadow_precheck = {}
    if gate_request:
        shadow_precheck = _compute_shadow_normalization_precheck(
            dataset_paths=dataset_paths,
            load_dataset_fn=load_dataset_fn,
            directives=directives,
            metrics=metrics if isinstance(metrics, dict) else {},
            auto_diagnostics=state.get("auto_diagnostics", {}) if isinstance(state, dict) else {},
        )
        if not shadow_precheck.get("allow", True):
            return _build_shadow_gate_skip_result(
                attempts=attempts,
                max_attempts=max_attempts,
                eval_path=eval_path,
                directives=directives,
                ablation_report=ablation_report,
                shadow_precheck=shadow_precheck,
                gate_request=gate_request,
                dataset_paths=dataset_paths,
            )

    os.makedirs("output/normalization", exist_ok=True)
    attempt_dir = os.path.join("output/normalization", f"attempt_{attempts}")
    os.makedirs(attempt_dir, exist_ok=True)

    warnings: List[str] = []
    if pending_list_import_warning:
        warnings.append(pending_list_import_warning)
    dataset_reports: Dict[str, Any] = {}
    normalized_paths: List[str] = []
    all_ok = True

    for path in dataset_paths:
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        try:
            df = load_dataset_fn(path)
            if df is None or df.empty:
                raise ValueError("dataset empty or failed to load")

            id_column = _detect_id_column(df)
            country_columns_used: List[str] = []
            list_columns_used: List[str] = []
            transforms = _build_column_transforms(
                df=df,
                id_column=id_column,
                attempts=attempts,
                validation_columns=validation_columns,
                validation_lowercase_columns=validation_lowercase_columns,
                low_accuracy_columns=low_accuracy_columns,
                directive_lowercase_columns=directive_lowercase_columns,
                directive_text_columns=directive_text_columns,
            )

            cfg = create_normalization_config(
                enable_unit_conversion=True,
                enable_quantity_scaling=True,
                normalize_text=False,
                standardize_nulls=True,
                lowercase_text=False,
                remove_extra_whitespace=True,
                add_metadata_columns=False,
                column_transformations=transforms,
                missing_transform_column_policy="warn",
            )

            normalized_df, norm_result = normalize_dataset(df, config=cfg)

            if id_column is not None and id_column in df.columns and id_column in normalized_df.columns:
                normalized_df[id_column] = df[id_column].copy()

            candidate_country_columns: List[str] = []
            for col in normalized_df.columns:
                c_lower = str(col).lower()
                if id_column is not None and col == id_column:
                    continue
                if c_lower in directive_country_columns or (
                    "country" in c_lower
                    and (
                        not validation_columns
                        or col in validation_columns
                        or c_lower in low_accuracy_columns
                    )
                ):
                    candidate_country_columns.append(col)

            if candidate_country_columns:
                for col in candidate_country_columns:
                    if col in normalized_df.columns:
                        normalized_df[col] = normalized_df[col].apply(
                            lambda value: _normalize_country_value(
                                value=value,
                                normalize_country_fn=normalize_country,
                                output_format=country_output_format,
                            )
                        )
                        country_columns_used.append(str(col))

            if (
                callable(detect_list_like_columns)
                and callable(normalize_list_like_columns)
            ):
                # Structural detection can be broad; keep explicitly requested list columns
                # and validation list columns. Do not force scalarization back.
                all_structural_list_cols: set[str] = set()
                try:
                    auto_list_cols = detect_list_like_columns(
                        [normalized_df],
                        exclude_columns={str(id_column).lower()} if id_column else set(),
                    )
                    all_structural_list_cols.update(str(c) for c in auto_list_cols)
                except Exception as e:
                    warnings.append(f"{dataset_name}: list detection failed: {e}")

                if target_list_columns:
                    for col in normalized_df.columns:
                        if str(col).lower() in target_list_columns:
                            all_structural_list_cols.add(str(col))

                keep_list_cols: set[str] = set()
                for col in all_structural_list_cols:
                    col_l = str(col).lower()
                    if col_l in target_list_columns:
                        keep_list_cols.add(str(col))

                if keep_list_cols:
                    try:
                        normalize_list_like_columns([normalized_df], sorted(keep_list_cols))
                        list_columns_used = sorted(list(keep_list_cols))
                    except Exception as e:
                        warnings.append(f"{dataset_name}: list normalization failed: {e}")

            if len(normalized_df) != len(df):
                raise ValueError(f"row count changed from {len(df)} to {len(normalized_df)}")

            if id_column is not None:
                if id_column not in normalized_df.columns:
                    raise ValueError(f"missing id column after normalization: {id_column}")
                before_non_null = int(df[id_column].notna().sum())
                after_non_null = int(normalized_df[id_column].notna().sum())
                if before_non_null > 0 and after_non_null < int(before_non_null * 0.95):
                    raise ValueError(
                        f"id non-null drop too large for {id_column}: {before_non_null} -> {after_non_null}"
                    )

            out_path = os.path.join(attempt_dir, f"{dataset_name}.csv")
            normalized_df.to_csv(out_path, index=False)
            normalized_paths.append(out_path)

            summary = {}
            try:
                summary = norm_result.get_summary()
            except Exception:
                summary = {}

            dataset_reports[dataset_name] = {
                "input_path": path,
                "output_path": out_path,
                "row_count": len(normalized_df),
                "id_column": id_column,
                "applied_transforms": {str(k): v for k, v in transforms.items()},
                "country_normalized_columns": country_columns_used,
                "list_normalized_columns": list_columns_used,
                "summary": summary,
            }
        except Exception as e:
            all_ok = False
            err = f"{dataset_name}: {e}"
            warnings.append(err)
            dataset_reports[dataset_name] = {
                "input_path": path,
                "status": "failed",
                "error": str(e),
            }

    if all_ok and len(normalized_paths) == len(dataset_paths):
        status = "success"
        datasets_for_next = normalized_paths
        reverted = False
        print("[+] Normalization completed; using normalized datasets")
    else:
        status = "fallback_to_original"
        datasets_for_next = original_paths
        reverted = True
        print("[WARN] Normalization incomplete; reverting to original datasets")
        for warning in warnings:
            print(f"[WARN] {warning}")

    failure_tags: List[str] = []
    if status == "fallback_to_original":
        failure_tags.append("runtime_error")
        failure_tags.append("normalization_regression")
    if status == "success" and warnings:
        failure_tags.append("needs_review")
    if not failure_tags and status == "success":
        failure_tags.append("ok")

    pending_acceptance = {}
    if status == "success" and gate_request:
        pending_acceptance = create_pending_normalization_acceptance(
            gate_request=gate_request,
            normalized_datasets=normalized_paths,
            normalization_attempt=attempts,
            evaluation_attempt=int(state.get("evaluation_attempts", 0)),
        )

    report = {
        "status": status,
        "attempt": attempts,
        "max_attempts": max_attempts,
        "validation_set": eval_path,
        "validation_style": {
            "inferred_country_output_format": inferred_country_output_format,
            "used_country_output_format": country_output_format,
            "lowercase_columns_hint": sorted(list(validation_lowercase_columns)),
            "validation_list_like_columns_hint": sorted(list(validation_list_like_columns)),
            "case_profile": validation_case_map,
        },
        "normalization_directives": directives,
        "ablation_report": ablation_report,
        "shadow_precheck": shadow_precheck,
        "acceptance_gate": {
            "requested": bool(gate_request),
            "request": gate_request,
            "pending": pending_acceptance,
        },
        "datasets": dataset_reports,
        "warnings": warnings,
        "failure_tags": failure_tags,
        "reverted_to_original": reverted,
        "created_at": datetime.now().isoformat(),
    }

    _write_normalization_report(attempt_dir, report)

    return {
        "datasets": datasets_for_next,
        "normalized_datasets": normalized_paths if not reverted else [],
        "normalization_execution_result": status,
        "normalization_attempts": attempts,
        "normalization_report": report,
        "normalization_directives": directives,
        "normalization_pending_acceptance": pending_acceptance,
        "normalization_gate_request": {},
        "normalization_rework_required": False,
        "normalization_rework_reasons": [],
        "pipeline_execution_result": "",
        "pipeline_execution_attempts": 0,
        "evaluation_execution_result": "",
        "evaluation_execution_attempts": 0,
    }
