"""Normalization orchestrator — LLM-driven NormalizationSpec generation.

The LLM sees sample rows from each source dataset alongside sample rows from
the validation set, then generates a PyDI ``NormalizationSpec`` per source.
Execution is just ``transform_dataframe(df, spec)``.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from helpers.investigator_acceptance import create_pending_normalization_acceptance
from _resolve_import import ensure_project_root
ensure_project_root()

import config
from workflow_logging import log_agent_action

# ---------------------------------------------------------------------------
# Probing helpers
# ---------------------------------------------------------------------------

PROBE_SAMPLE_SIZE = 15
ID_COLUMN_PATTERNS = ("id", "_id")


def _detect_id_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ID_COLUMN_PATTERNS:
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if "id" in str(col).lower():
            return col
    return None


def _sample_rows(df: pd.DataFrame, n: int = PROBE_SAMPLE_SIZE) -> List[Dict[str, Any]]:
    """Return *n* representative rows as dicts (JSON-serialisable)."""
    if df is None or df.empty:
        return []
    # Mix: first few rows, last few, and some random from the middle
    indices = set()
    total = len(df)
    # First 3
    for i in range(min(3, total)):
        indices.add(i)
    # Last 2
    for i in range(max(0, total - 2), total):
        indices.add(i)
    # Fill remaining from random sample
    remaining = n - len(indices)
    if remaining > 0 and total > len(indices):
        pool = [i for i in range(total) if i not in indices]
        import random
        rng = random.Random(42)
        indices.update(rng.sample(pool, min(remaining, len(pool))))
    sampled = df.iloc[sorted(indices)].head(n)
    rows = []
    for _, row in sampled.iterrows():
        d = {}
        for col in sampled.columns:
            val = row[col]
            try:
                is_na = pd.isna(val)
                if hasattr(is_na, '__len__'):
                    is_na = False  # array-like value (list column) — not null
                else:
                    is_na = bool(is_na)
            except (ValueError, TypeError):
                is_na = False
            if is_na:
                d[col] = None
            else:
                d[col] = val
            # Ensure JSON-serialisable
            try:
                json.dumps(d[col])
            except (TypeError, ValueError):
                d[col] = str(val)
        rows.append(d)
    return rows


def build_dataset_probe(df: pd.DataFrame) -> Dict[str, Any]:
    """Build a probe dict for a single dataset."""
    return {
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "row_count": len(df),
        "sample_rows": _sample_rows(df),
    }


# ---------------------------------------------------------------------------
# Spec parsing & validation
# ---------------------------------------------------------------------------

def _parse_spec_response(raw: Any) -> Dict[str, Any]:
    """Extract the JSON dict from the LLM response."""
    from helpers.evaluation import extract_json_object

    # Try .content first (standard path)
    parsed = extract_json_object(raw)
    if isinstance(parsed, dict) and parsed:
        return parsed

    # Reasoning models may put content in additional_kwargs
    additional = getattr(raw, "additional_kwargs", {}) or {}
    reasoning = additional.get("reasoning") or additional.get("reasoning_content")
    if reasoning:
        reasoning_text = reasoning.get("content", "") if isinstance(reasoning, dict) else str(reasoning)
        if reasoning_text:
            parsed = extract_json_object(reasoning_text)
            if isinstance(parsed, dict) and parsed:
                print("[NORM] Recovered specs from reasoning content")
                return parsed

    raw_text = raw.content if hasattr(raw, "content") else str(raw)
    print(f"[NORM] Failed to parse LLM spec response (len={len(raw_text)}): {raw_text[:300]}")
    return {}


def _validate_spec_columns(spec: Dict[str, Any], df_columns: set) -> Dict[str, Any]:
    """Remove spec entries for columns that don't exist in the DataFrame."""
    return {col: cfg for col, cfg in spec.items() if col in df_columns}


ALLOWED_COUNTRY_FORMATS = {"alpha_2", "alpha_3", "numeric", "name", "official_name"}
ALLOWED_CASE = {"lower", "upper", "title", "keep", None}
ALLOWED_OUTPUT_TYPES = {"string", "float", "int", "bool", "datetime", "keep"}
ALLOWED_ON_FAILURE = {"keep", "null", "raise"}


def _sanitize_column_spec(col_spec: Any) -> Dict[str, Any]:
    """Validate individual ColumnSpec fields, dropping invalid values."""
    if not isinstance(col_spec, dict):
        return {}
    clean: Dict[str, Any] = {}

    # output_type
    ot = col_spec.get("output_type")
    if ot in ALLOWED_OUTPUT_TYPES:
        clean["output_type"] = ot

    # on_failure
    of = col_spec.get("on_failure")
    if of in ALLOWED_ON_FAILURE:
        clean["on_failure"] = of
    elif "on_failure" not in col_spec:
        clean["on_failure"] = "keep"  # safe default

    # strip_whitespace
    sw = col_spec.get("strip_whitespace")
    if isinstance(sw, bool):
        clean["strip_whitespace"] = sw

    # case
    case = col_spec.get("case")
    if case in ALLOWED_CASE:
        clean["case"] = case

    # country_format
    cf = col_spec.get("country_format")
    if cf in ALLOWED_COUNTRY_FORMATS:
        clean["country_format"] = cf

    # currency_format
    cuf = col_spec.get("currency_format")
    if cuf in {"alpha_3", "name", "keep"}:
        clean["currency_format"] = cuf

    # phone_format
    pf = col_spec.get("phone_format")
    if pf in {"e164", "international", "national", "keep"}:
        clean["phone_format"] = pf
    pdr = col_spec.get("phone_default_region")
    if isinstance(pdr, str) and pdr.strip():
        clean["phone_default_region"] = pdr.strip()

    # normalize_email
    ne = col_spec.get("normalize_email")
    if isinstance(ne, bool):
        clean["normalize_email"] = ne

    # stdnum_format
    sf = col_spec.get("stdnum_format")
    if isinstance(sf, bool):
        clean["stdnum_format"] = sf

    # date_format
    df_val = col_spec.get("date_format")
    if isinstance(df_val, str) and df_val.strip():
        clean["date_format"] = df_val.strip()

    # expand_scale_modifiers
    esm = col_spec.get("expand_scale_modifiers")
    if isinstance(esm, bool):
        clean["expand_scale_modifiers"] = esm

    # convert_percentage
    cp = col_spec.get("convert_percentage")
    if cp in {"to_decimal", "to_percent", "keep"}:
        clean["convert_percentage"] = cp

    # target_unit
    tu = col_spec.get("target_unit")
    if isinstance(tu, str) and tu.strip():
        clean["target_unit"] = tu.strip()

    return clean


# ---------------------------------------------------------------------------
# LLM spec generation
# ---------------------------------------------------------------------------

def _generate_normalization_specs(
    agent,
    *,
    source_probes: Dict[str, Dict],
    target_probe: Dict,
    mismatch_examples: Optional[Dict] = None,
    previous_attempt_feedback: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the LLM to generate NormalizationSpec dicts per source dataset."""
    from prompts.normalization_prompt import (
        NORMALIZATION_SPEC_SYSTEM_PROMPT,
        build_normalization_user_prompt,
    )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        from langchain.schema import HumanMessage, SystemMessage

    user_prompt = build_normalization_user_prompt(
        source_probes=source_probes,
        target_probe=target_probe,
        mismatch_examples=mismatch_examples,
        previous_attempt_feedback=previous_attempt_feedback,
    )

    messages = [
        SystemMessage(content=NORMALIZATION_SPEC_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    result = agent._invoke_base_with_usage(messages, "normalization_spec_generation")

    # Diagnostic: log raw result shape when content is empty
    content = result.content if hasattr(result, "content") else None
    if not content or not str(content).strip():
        additional = getattr(result, "additional_kwargs", {})
        response_meta = getattr(result, "response_metadata", {})
        finish_reason = response_meta.get("finish_reason") if isinstance(response_meta, dict) else None
        print(f"[NORM] WARNING: empty .content from LLM | finish_reason={finish_reason} "
              f"| additional_kwargs keys={list(additional.keys()) if additional else []} "
              f"| content type={type(content).__name__} repr={repr(content)[:200]}")

    parsed = _parse_spec_response(result)
    return parsed


# ---------------------------------------------------------------------------
# Apply specs via PyDI
# ---------------------------------------------------------------------------

def _apply_spec_to_dataframe(
    df: pd.DataFrame,
    spec_dict: Dict[str, Any],
    id_column: Optional[str],
    list_columns: List[str],
) -> pd.DataFrame:
    """Apply a NormalizationSpec to a DataFrame using PyDI's transform API."""
    from PyDI.normalization import NormalizationSpec, transform_dataframe

    # Remove ID columns from spec
    if id_column:
        spec_dict = {k: v for k, v in spec_dict.items() if k != id_column}

    # Remove list columns from spec — PyDI's transform_dataframe calls pd.isna()
    # on individual cell values, which crashes on Python lists (ValueError:
    # "truth value of an array with more than one element is ambiguous").
    # List columns are handled separately via list_normalization after transform.
    if list_columns:
        _skipped_list_cols = [k for k in spec_dict if k in list_columns]
        if _skipped_list_cols:
            print(f"[NORM] Excluding list columns from NormalizationSpec (handled separately): {_skipped_list_cols}")
            spec_dict = {k: v for k, v in spec_dict.items() if k not in list_columns}

    # Also detect columns that already contain list values in the DataFrame,
    # even if not in the explicit list_columns parameter.
    _detected_list_cols = []
    for col in list(spec_dict.keys()):
        if col in df.columns:
            _sample = df[col].dropna().head(5).tolist()
            if any(isinstance(v, (list, tuple, set)) for v in _sample):
                _detected_list_cols.append(col)
    if _detected_list_cols:
        print(f"[NORM] Excluding auto-detected list-valued columns from NormalizationSpec: {_detected_list_cols}")
        spec_dict = {k: v for k, v in spec_dict.items() if k not in _detected_list_cols}

    # Validate columns exist
    df_cols = set(df.columns)
    spec_dict = _validate_spec_columns(spec_dict, df_cols)

    # Sanitize each column spec
    spec_dict = {col: _sanitize_column_spec(cfg) for col, cfg in spec_dict.items()}
    # Drop empty specs
    spec_dict = {col: cfg for col, cfg in spec_dict.items() if cfg}

    # Save original ID column
    id_backup = None
    if id_column and id_column in df.columns:
        id_backup = df[id_column].copy()

    # Apply NormalizationSpec
    normalized_df = df.copy()
    if spec_dict:
        spec = NormalizationSpec.from_dict({"columns": spec_dict})
        result = transform_dataframe(normalized_df, spec)
        normalized_df = result.dataframe

    # Restore ID column
    if id_backup is not None and id_column in normalized_df.columns:
        normalized_df[id_column] = id_backup

    # Apply list normalization if requested
    if list_columns:
        actual_list_cols = [c for c in list_columns if c in normalized_df.columns]
        if actual_list_cols:
            try:
                from list_normalization import normalize_list_like_columns
                normalize_list_like_columns([normalized_df], actual_list_cols)
            except Exception as e:
                print(f"[NORM] List normalization failed for {actual_list_cols}: {e}")

    return normalized_df


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _write_normalization_report(attempt_dir: str, report: Dict[str, Any]) -> None:
    try:
        os.makedirs(attempt_dir, exist_ok=True)
        with open(os.path.join(attempt_dir, "normalization_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[NORM] Failed to write normalization report to {attempt_dir}: {e}")


def _build_previous_attempt_feedback(state: Dict[str, Any]) -> Optional[str]:
    """Build feedback string from previous normalization/evaluation attempts."""
    parts: List[str] = []

    # Previous normalization report
    prev_report = state.get("normalization_report", {})
    if isinstance(prev_report, dict) and prev_report.get("status"):
        parts.append(f"Previous normalization status: {prev_report.get('status')}")
        warnings = prev_report.get("warnings", [])
        if warnings:
            parts.append(f"Warnings: {'; '.join(str(w) for w in warnings[:5])}")

    # Evaluation metrics (what's still wrong)
    metrics = state.get("evaluation_metrics", {})
    if isinstance(metrics, dict):
        overall = metrics.get("overall_accuracy")
        if overall is not None:
            parts.append(f"Current overall accuracy: {overall}")
        low_attrs = []
        for k, v in metrics.items():
            if isinstance(k, str) and k.endswith("_accuracy") and k not in ("overall_accuracy", "macro_accuracy"):
                try:
                    if float(v) < 0.60:
                        low_attrs.append(f"{k}: {v}")
                except (ValueError, TypeError):
                    pass  # Non-numeric metric value, skip
        if low_attrs:
            parts.append(f"Low-accuracy attributes: {', '.join(low_attrs[:10])}")

    # Acceptance feedback (if rejected)
    acceptance = state.get("normalization_acceptance_feedback", {})
    if isinstance(acceptance, dict) and acceptance.get("status") == "rejected":
        parts.append(f"Previous normalization was REJECTED (delta={acceptance.get('observed_delta')})")
        if not acceptance.get("key_attrs_ok"):
            parts.append("Reason: key attributes did not improve enough")
        if not acceptance.get("heldout_proxy_ok"):
            parts.append("Reason: non-focus attributes regressed")

    return "\n".join(parts) if parts else None


def _get_mismatch_examples(state: Dict[str, Any]) -> Optional[Dict]:
    """Extract mismatch examples from investigator probes if available."""
    probe_results = state.get("investigator_probe_results", {})
    if not isinstance(probe_results, dict):
        return None
    results_list = probe_results.get("results", [])
    sampler = next((r for r in results_list if isinstance(r, dict) and r.get("name") == "mismatch_sampler"), None)
    if not sampler:
        return None
    samples = sampler.get("samples_by_attribute", {})
    return samples if samples else None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_normalization_node(agent, state: Dict[str, Any], load_dataset_fn) -> Dict[str, Any]:
    """Normalization node: probe datasets → LLM generates specs → PyDI applies them."""
    log_agent_action(
        agent,
        step="normalization_node",
        action="start",
        why="Normalize source datasets via LLM-generated NormalizationSpec",
        improvement="LLM sees source and target data samples to generate precise per-column specs",
    )
    print("[NORM] Starting normalization (LLM-driven spec generation)")

    attempts = int(state.get("normalization_attempts", 0)) + 1
    max_attempts = 3
    gate_request = state.get("normalization_gate_request", {}) if isinstance(state, dict) else {}
    if not isinstance(gate_request, dict):
        gate_request = {}

    dataset_paths = list(state.get("datasets", []) or [])
    original_paths = list(state.get("original_datasets", dataset_paths) or dataset_paths)
    if "original_datasets" not in state:
        state["original_datasets"] = list(original_paths)

    # ── Load validation/target dataset ──
    eval_path = agent._evaluation_testset_path(state, force_test=False)
    validation_df = None
    target_probe: Dict[str, Any] = {}
    try:
        if eval_path:
            validation_df = load_dataset_fn(eval_path)
            if validation_df is not None and not validation_df.empty:
                target_probe = build_dataset_probe(validation_df)
    except Exception as e:
        print(f"[NORM] Failed to load validation set '{eval_path}': {e}")
        validation_df = None

    if not target_probe:
        print(f"[NORM] No validation dataset available — skipping (eval_path={eval_path!r})")
        return _skip_result(attempts, max_attempts, dataset_paths, "no_validation_set", gate_request)

    # ── Import PyDI ──
    try:
        from PyDI.normalization import NormalizationSpec, transform_dataframe  # noqa: F401
    except Exception as e:
        msg = f"PyDI normalization import failed: {e}"
        print(f"[NORM] {msg}")
        return _skip_result(attempts, max_attempts, original_paths, f"import_error: {msg}", gate_request)

    # ── Build source probes ──
    source_probes: Dict[str, Dict] = {}
    source_dfs: Dict[str, pd.DataFrame] = {}
    for path in dataset_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            df = load_dataset_fn(path)
            if df is not None and not df.empty:
                source_probes[name] = build_dataset_probe(df)
                source_dfs[name] = df
        except Exception as e:
            print(f"[NORM] Failed to load {name}: {e}")

    if not source_probes:
        print("[NORM] No source datasets loaded — skipping")
        return _skip_result(attempts, max_attempts, dataset_paths, "no_source_datasets", gate_request)

    # ── Gather context for re-runs ──
    mismatch_examples = _get_mismatch_examples(state) if attempts > 1 else None
    previous_feedback = _build_previous_attempt_feedback(state) if attempts > 1 else None

    # ── LLM generates specs ──
    print(f"[NORM] Generating specs via LLM (attempt {attempts})")
    llm_response = _generate_normalization_specs(
        agent,
        source_probes=source_probes,
        target_probe=target_probe,
        mismatch_examples=mismatch_examples,
        previous_attempt_feedback=previous_feedback,
    )

    specs = llm_response.get("specs", {})
    list_columns = llm_response.get("list_columns", [])
    reasoning = llm_response.get("reasoning", "")

    # If the LLM returned per-dataset specs at the top level (no "specs" wrapper),
    # detect and recover: a dict whose values are all dicts of column specs.
    if not specs and llm_response:
        non_meta_keys = {k for k in llm_response if k not in ("list_columns", "reasoning", "specs")}
        if non_meta_keys and all(isinstance(llm_response.get(k), dict) for k in non_meta_keys):
            specs = {k: llm_response[k] for k in non_meta_keys}
            print(f"[NORM] Recovered specs from top-level keys: {sorted(specs.keys())}")

    if not isinstance(specs, dict):
        specs = {}
    if not isinstance(list_columns, list):
        list_columns = []

    print(f"[NORM] Specs for {len(specs)} dataset(s), {len(list_columns)} list column(s)")
    if not specs:
        # Log raw response for debugging
        raw_keys = list(llm_response.keys()) if isinstance(llm_response, dict) else type(llm_response).__name__
        print(f"[NORM] WARNING: empty specs — LLM response keys: {raw_keys}")
        print("[NORM] Skipping — LLM returned no normalization specs")
        return _skip_result(attempts, max_attempts, dataset_paths, "empty_llm_specs", gate_request)
    if reasoning:
        print(f"[NORM] Reasoning: {reasoning[:200]}")

    # ── Apply specs ──
    norm_base = os.path.join(config.OUTPUT_DIR, "normalization")
    os.makedirs(norm_base, exist_ok=True)
    attempt_dir = os.path.join(norm_base, f"attempt_{attempts}")
    os.makedirs(attempt_dir, exist_ok=True)

    warnings: List[str] = []
    dataset_reports: Dict[str, Any] = {}
    normalized_paths: List[str] = []
    all_ok = True

    for path in dataset_paths:
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        try:
            df = source_dfs.get(dataset_name)
            if df is None:
                df = load_dataset_fn(path)
            if df is None or df.empty:
                raise ValueError("dataset empty or failed to load")

            id_column = _detect_id_column(df)

            # Get spec for this dataset (try exact name, then partial match)
            spec_dict = specs.get(dataset_name, {})
            if not spec_dict:
                # Try case-insensitive match
                for spec_name, spec_val in specs.items():
                    if spec_name.lower() == dataset_name.lower():
                        spec_dict = spec_val
                        break

            if not isinstance(spec_dict, dict):
                spec_dict = {}

            # Apply the spec
            normalized_df = _apply_spec_to_dataframe(
                df=df,
                spec_dict=spec_dict,
                id_column=id_column,
                list_columns=[c for c in list_columns if c in df.columns],
            )

            # Safety checks
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

            dataset_reports[dataset_name] = {
                "input_path": path,
                "output_path": out_path,
                "row_count": len(normalized_df),
                "id_column": id_column,
                "spec_applied": spec_dict,
                "list_columns_applied": [c for c in list_columns if c in df.columns],
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

    # ── Determine outcome ──
    if all_ok and len(normalized_paths) == len(dataset_paths):
        status = "success"
        datasets_for_next = normalized_paths
        reverted = False
        print(f"[NORM] Done — {len(normalized_paths)} dataset(s) normalized")
    else:
        status = "fallback_to_original"
        datasets_for_next = original_paths
        reverted = True
        print(f"[NORM] Incomplete — reverting to originals ({'; '.join(warnings[:3])})")

    # ── Acceptance gate ──
    pending_acceptance = {}
    if status == "success" and gate_request:
        pending_acceptance = create_pending_normalization_acceptance(
            gate_request=gate_request,
            normalized_datasets=normalized_paths,
            normalization_attempt=attempts,
            evaluation_attempt=int(state.get("evaluation_attempts", 0)),
        )

    # ── Report ──
    report = {
        "status": status,
        "attempt": attempts,
        "max_attempts": max_attempts,
        "validation_set": eval_path,
        "llm_specs": specs,
        "llm_list_columns": list_columns,
        "llm_reasoning": reasoning,
        "datasets": dataset_reports,
        "warnings": warnings,
        "acceptance_gate": {
            "requested": bool(gate_request),
            "request": gate_request,
            "pending": pending_acceptance,
        },
        "reverted_to_original": reverted,
        "created_at": datetime.now().isoformat(),
    }
    _write_normalization_report(attempt_dir, report)

    # Save the full spec as a standalone JSON for reproducibility
    try:
        spec_path = os.path.join(attempt_dir, "normalization_specs.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump({"specs": specs, "list_columns": list_columns, "reasoning": reasoning},
                      f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[NORM] Failed to save normalization specs: {e}")

    return {
        "datasets": datasets_for_next,
        "normalized_datasets": normalized_paths if not reverted else [],
        "normalization_execution_result": status,
        "normalization_attempts": attempts,
        "normalization_report": report,
        "normalization_directives": {"specs": specs, "list_columns": list_columns},
        "normalization_pending_acceptance": pending_acceptance,
        "normalization_gate_request": {},
        "normalization_rework_required": False,
        "normalization_rework_reasons": [],
        "pipeline_execution_result": "",
        "pipeline_execution_attempts": 0,
        "evaluation_execution_result": "",
        "evaluation_execution_attempts": 0,
    }


def _skip_result(
    attempts: int,
    max_attempts: int,
    dataset_paths: List[str],
    reason: str,
    gate_request: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a skip/fallback result dict."""
    return {
        "datasets": dataset_paths,
        "normalized_datasets": [],
        "normalization_execution_result": f"skipped: {reason}",
        "normalization_attempts": attempts,
        "normalization_report": {
            "status": "skipped",
            "attempt": attempts,
            "max_attempts": max_attempts,
            "reason": reason,
            "reverted_to_original": True,
            "created_at": datetime.now().isoformat(),
        },
        "normalization_directives": {},
        "normalization_pending_acceptance": {},
        "normalization_gate_request": {},
        "normalization_rework_required": False,
        "normalization_rework_reasons": [],
        "pipeline_execution_result": "",
        "pipeline_execution_attempts": 0,
        "evaluation_execution_result": "",
        "evaluation_execution_attempts": 0,
    }
