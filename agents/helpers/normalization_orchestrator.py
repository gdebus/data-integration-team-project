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


def _clamp_directives_to_validation_style(
    directives: Dict[str, Any],
    validation_columns: set[str],
    validation_lowercase_columns: set[str],
    validation_list_like_columns: set[str],
    inferred_country_output_format: str,
) -> Dict[str, Any]:
    """
    Enforce validation-set style as the single source of truth for normalization format.
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

    out["list_columns"] = _filter(out.get("list_columns", []), allowed_list)
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

    # Validation representation is authoritative.
    out["country_output_format"] = inferred_country_output_format
    out["list_normalization_required"] = bool(out.get("list_columns", []))
    return out


def run_normalization_node(agent, state: Dict[str, Any], load_dataset_fn):
    agent._log_action(
        "normalization_node",
        "start",
        "Normalize source datasets",
        "Applies PyDI normalization with safe fallback",
    )
    print("[*] Running normalization node")

    attempts = int(state.get("normalization_attempts", 0)) + 1
    max_attempts = 3

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
            "normalization_rework_required": False,
            "normalization_rework_reasons": [],
        }

    detect_list_like_columns = None
    normalize_list_like_columns = None
    normalize_list_value = None
    pending_list_import_warning = None
    try:
        from list_normalization import (
            detect_list_like_columns,
            normalize_list_like_columns,
            normalize_list_value,
        )
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
                normalize_list_value,
            )
        except Exception as e:
            pending_list_import_warning = f"list_normalization import failed: {e}"
    except Exception as e:
        pending_list_import_warning = f"list_normalization import failed: {e}"

    validation_list_like_columns: set[str] = set()
    if callable(detect_list_like_columns) and validation_df is not None and not validation_df.empty:
        try:
            inferred_list_cols = detect_list_like_columns(
                [validation_df],
                exclude_columns={"id", "_id", "eval_id", "_eval_cluster_id"},
            )
            validation_list_like_columns = {str(c).lower() for c in inferred_list_cols}
        except Exception as e:
            warnings_hint = f"validation list-like detection failed: {e}"
            pending_list_import_warning = (
                f"{pending_list_import_warning}; {warnings_hint}"
                if pending_list_import_warning
                else warnings_hint
            )

    list_norm_requested = bool(
        directives.get("list_normalization_required")
        or validation_list_like_columns
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
    list_norm_requested = bool(directive_list_columns or validation_list_like_columns)

    country_output_format = inferred_country_output_format
    directives["country_output_format"] = country_output_format
    directives["lowercase_columns"] = sorted(list(validation_lowercase_columns))
    directives["list_columns"] = sorted(list(validation_list_like_columns))
    directives["list_normalization_required"] = bool(list_norm_requested)

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

            id_column = None
            for candidate in ("id", "_id"):
                if candidate in df.columns:
                    id_column = candidate
                    break
            if id_column is None:
                for col in df.columns:
                    if "id" in str(col).lower():
                        id_column = col
                        break

            transforms: Dict[Any, Any] = {}
            country_columns_used: List[str] = []
            list_columns_used: List[str] = []
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

                if ops:
                    deduped_ops: List[str] = []
                    for op in ops:
                        if op not in deduped_ops:
                            deduped_ops.append(op)
                    transforms[col] = deduped_ops

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

                def _normalize_country_safe(value: Any):
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
                        normalized = normalize_country(text, output_format=country_output_format)
                    except Exception:
                        normalized = None
                    return normalized if normalized else text

                for col in candidate_country_columns:
                    if col in normalized_df.columns:
                        normalized_df[col] = normalized_df[col].apply(_normalize_country_safe)
                        country_columns_used.append(str(col))

            if (
                callable(detect_list_like_columns)
                and callable(normalize_list_like_columns)
                and callable(normalize_list_value)
            ):
                # Structural detection can be broad (source-driven), but representation style
                # must follow validation: keep lists only where validation is list-like.
                all_structural_list_cols: set[str] = set()
                try:
                    auto_list_cols = detect_list_like_columns(
                        [normalized_df],
                        exclude_columns={str(id_column).lower()} if id_column else set(),
                    )
                    all_structural_list_cols.update(str(c) for c in auto_list_cols)
                except Exception as e:
                    warnings.append(f"{dataset_name}: list detection failed: {e}")

                if directive_list_columns:
                    for col in normalized_df.columns:
                        if str(col).lower() in directive_list_columns:
                            all_structural_list_cols.add(str(col))

                keep_list_cols: set[str] = set()
                scalarize_cols: set[str] = set()
                for col in all_structural_list_cols:
                    col_l = str(col).lower()
                    if col_l in validation_list_like_columns:
                        keep_list_cols.add(str(col))
                    else:
                        scalarize_cols.add(str(col))

                if keep_list_cols:
                    try:
                        normalize_list_like_columns([normalized_df], sorted(keep_list_cols))
                        list_columns_used = sorted(list(keep_list_cols))
                    except Exception as e:
                        warnings.append(f"{dataset_name}: list normalization failed: {e}")

                # If validation expects scalar style, collapse structural lists back to scalars.
                if scalarize_cols:
                    def _collapse_to_scalar(value: Any):
                        tokens = normalize_list_value(value, dedupe=False)
                        if not tokens:
                            return value
                        return str(tokens[0]).strip()

                    for col in sorted(scalarize_cols):
                        if col not in normalized_df.columns:
                            continue
                        try:
                            normalized_df[col] = normalized_df[col].apply(_collapse_to_scalar)
                        except Exception as e:
                            warnings.append(f"{dataset_name}: scalarization failed for {col}: {e}")

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
        "datasets": dataset_reports,
        "warnings": warnings,
        "reverted_to_original": reverted,
        "created_at": datetime.now().isoformat(),
    }

    try:
        with open(os.path.join(attempt_dir, "normalization_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass

    return {
        "datasets": datasets_for_next,
        "normalized_datasets": normalized_paths if not reverted else [],
        "normalization_execution_result": status,
        "normalization_attempts": attempts,
        "normalization_report": report,
        "normalization_directives": directives,
        "normalization_rework_required": False,
        "normalization_rework_reasons": [],
        "pipeline_execution_result": "",
        "pipeline_execution_attempts": 0,
        "evaluation_execution_result": "",
        "evaluation_execution_attempts": 0,
    }
