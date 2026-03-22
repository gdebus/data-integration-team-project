import ast
import json
import os
import re
from typing import Any, Callable, Dict, List

import pandas as pd


def extract_llm_text(result: Any) -> str:
    """Extract visible text from an LLM result, with reasoning model fallback.

    Reasoning models (e.g. gpt-5.4, o-series) may put their output in
    additional_kwargs['reasoning'] instead of .content.  This helper
    checks both locations.
    """
    content = result.content if hasattr(result, "content") else str(result)
    content = str(content or "").strip()
    if content:
        return content

    # Reasoning model fallback
    additional = getattr(result, "additional_kwargs", {}) or {}
    for key in ("reasoning", "reasoning_content"):
        reasoning = additional.get(key)
        if reasoning:
            text = reasoning.get("content", "") if isinstance(reasoning, dict) else str(reasoning)
            text = str(text or "").strip()
            if text:
                return text
    return ""


def parse_source_ids(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text:
        return []
    if text[0] not in "[({":
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, (list, tuple, set)):
            return [str(v) for v in parsed if str(v)]
    return []


def compute_id_alignment(
    fused_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    parse_source_ids_fn: Callable[[Any], List[str]] = parse_source_ids,
) -> Dict[str, Any]:
    if "id" not in gold_df.columns:
        return {}

    gold_ids = set(gold_df["id"].dropna().astype(str).tolist())
    if not gold_ids:
        return {}

    # Support both "_id" (PyDI default) and "id" column names
    id_col = "_id" if "_id" in fused_df.columns else ("id" if "id" in fused_df.columns else None)
    direct_ids = set(fused_df[id_col].dropna().astype(str).tolist()) if id_col else set()
    direct_cov = len(direct_ids & gold_ids)

    prefixes = [x.split("_", 1)[0] for x in gold_ids if "_" in x]
    gold_prefix = max(set(prefixes), key=prefixes.count) + "_" if prefixes else None

    # Collect source IDs from _fusion_sources or per-source ID columns (e.g. discogs_id, lastfm_id)
    source_id_cols = [c for c in fused_df.columns if c.endswith("_id") and c != "_id" and c != "id"]

    mapped_ids = set()
    for _, row in fused_df.iterrows():
        sources = parse_source_ids_fn(row.get("_fusion_sources"))
        # Also gather IDs from per-source columns when _fusion_sources is absent
        if not sources and source_id_cols:
            sources = [str(row[c]) for c in source_id_cols if pd.notna(row.get(c))]
        candidates = []
        if gold_prefix:
            candidates = [sid for sid in sources if sid.startswith(gold_prefix)]
        if not candidates:
            candidates = [sid for sid in sources if sid in gold_ids]
        if not candidates and id_col and id_col in row and pd.notna(row[id_col]):
            candidates = [str(row[id_col])]
        mapped_ids.update(candidates)

    mapped_cov = len(mapped_ids & gold_ids)
    denom = max(1, len(gold_ids))
    return {
        "gold_count": len(gold_ids),
        "direct_coverage": direct_cov,
        "direct_coverage_ratio": direct_cov / denom,
        "mapped_coverage": mapped_cov,
        "mapped_coverage_ratio": mapped_cov / denom,
        "missing_gold_ids": sorted(list(gold_ids - mapped_ids))[:25],
    }


def compute_auto_diagnostics(
    state: Dict[str, Any],
    metrics: Dict[str, Any],
    evaluation_testset_path: str,
    evaluation_stage_label: str,
    load_dataset_fn: Callable[[str], pd.DataFrame],
    compute_id_alignment_fn: Callable[[pd.DataFrame, pd.DataFrame], Dict[str, Any]],
    debug_path: str = "output/pipeline_evaluation/debug_fusion_eval.jsonl",
    fused_path: str = "output/data_fusion/fusion_data.csv",
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {}

    reason_counts: Dict[str, int] = {}
    total_mismatches = 0
    if os.path.exists(debug_path):
        with open(debug_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if event.get("type") != "evaluation_mismatch":
                    continue
                total_mismatches += 1
                reason = str(event.get("reason", "unknown"))
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    diagnostics["debug_reason_counts"] = reason_counts
    diagnostics["debug_reason_ratios"] = (
        {k: v / max(1, total_mismatches) for k, v in reason_counts.items()}
        if reason_counts
        else {}
    )

    diagnostics["evaluation_stage"] = evaluation_stage_label
    diagnostics["evaluation_testset_path"] = evaluation_testset_path
    if fused_path and evaluation_testset_path and os.path.exists(fused_path) and os.path.exists(evaluation_testset_path):
        try:
            fused_df = pd.read_csv(fused_path)
            gold_df = load_dataset_fn(evaluation_testset_path)
            diagnostics["id_alignment"] = compute_id_alignment_fn(fused_df, gold_df)
        except Exception as e:
            diagnostics["id_alignment_error"] = str(e)

    if state.get("fusion_size_comparison"):
        diagnostics["fusion_size_comparison"] = state.get("fusion_size_comparison")

    diagnostics["overall_accuracy"] = metrics.get("overall_accuracy")
    return diagnostics


def extract_python_code(text: Any) -> str:
    raw = text.content if hasattr(text, "content") else str(text)
    raw = str(raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:python)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def extract_json_object(text: Any) -> Dict[str, Any]:
    raw = text.content if hasattr(text, "content") else str(text)
    raw = str(raw or "").strip()

    # If .content is empty, try reasoning content from reasoning models
    if not raw and hasattr(text, "additional_kwargs"):
        additional = getattr(text, "additional_kwargs", {}) or {}
        reasoning = additional.get("reasoning") or additional.get("reasoning_content")
        if reasoning:
            raw = reasoning.get("content", "") if isinstance(reasoning, dict) else str(reasoning)
            raw = str(raw or "").strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    try:
        candidates = re.findall(r"\{[\s\S]*\}", raw)
    except Exception:
        candidates = []

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    return {}
