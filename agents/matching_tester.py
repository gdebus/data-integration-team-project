import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import sys
import traceback

import pandas as pd
import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage

from PyDI.io import load_xml, load_parquet, load_csv
from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
    TokenBlocker,
    SortedNeighbourhoodBlocker,
    EntityMatchingEvaluator,
    RuleBasedMatcher,
    MLBasedMatcher,
    FeatureExtractor,
    StringComparator,
    NumericComparator,
    DateComparator,
)

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

PROXY_F1_MARGIN_DEFAULT = 0.015
PROXY_MAX_CANDIDATES_DEFAULT = 60_000
PROXY_MIN_F1_ATTEMPT1 = 0.03
NO_GAIN_EPSILON = 0.003
NO_GAIN_PATIENCE = 4

AGENTS_ROOT = Path(__file__).resolve().parent
if str(AGENTS_ROOT) not in sys.path:
    sys.path.append(str(AGENTS_ROOT))

from list_normalization import (
    detect_list_like_columns,
    is_list_like_value,
    normalize_list_value,
    normalize_list_like_columns,
)

try:
    from fusion_size_monitor import (
        estimate_from_matching,
        estimate_path_for_output_dir,
        upsert_stage_estimate,
    )
except ImportError:
    from .fusion_size_monitor import (
        estimate_from_matching,
        estimate_path_for_output_dir,
        upsert_stage_estimate,
    )


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return load_parquet(path)
    if ext == ".csv":
        return load_csv(path)
    if ext == ".xml":
        return load_xml(path, nested_handling="aggregate")
    raise ValueError(f"Unsupported format: {ext}. Supported: .csv, .parquet, .xml")


MATCHING_STRATEGY_DESCRIPTIONS = """
- Use RuleBasedMatcher only.
- Comparator types:
  - string: StringComparator(column, similarity_function, preprocess?, list_strategy?)
  - numeric: NumericComparator(column, max_difference, list_strategy?)
  - date: DateComparator(column, max_days_difference)
"""


class MatchingTester:
    """LLM-driven matcher tester that evaluates comparator configs on gold standards."""

    def __init__(
        self,
        llm,
        datasets: List[str],
        matching_testsets: Dict[Tuple[str, str], str],
        blocking_config: Optional[Dict[str, Any]] = None,
        output_dir: str = "output/matching-evaluation",
        f1_threshold: float = 0.75,
        max_attempts: int = 8,
        max_error_retries: int = 2,
        verbose: bool = True,
        matcher_mode: str = "ml",
        previous_failures: Optional[List[Dict[str, Any]]] = None,
        disallow_list_comparators: bool = True,
        min_blocking_pc_for_matching: Optional[float] = None,
        proxy_f1_margin: float = PROXY_F1_MARGIN_DEFAULT,
        proxy_max_candidates: int = PROXY_MAX_CANDIDATES_DEFAULT,
        no_gain_patience: int = NO_GAIN_PATIENCE,
        no_gain_epsilon: float = NO_GAIN_EPSILON,
    ):
        self.llm = llm
        self.output_dir = output_dir
        self.f1_threshold = f1_threshold
        self.max_attempts = max_attempts
        self.max_error_retries = max_error_retries
        self.verbose = verbose
        self.evaluator = EntityMatchingEvaluator()
        self.results_history: List[Dict[str, Any]] = []
        self.previous_failures = previous_failures or []
        self.matcher_mode = matcher_mode
        self.disallow_list_comparators = disallow_list_comparators

        self.blocking_config = blocking_config or {}
        self.blocking_strategies = self.blocking_config.get("blocking_strategies", {})
        self.id_columns = self.blocking_config.get("id_columns", {})
        configured_blocking_threshold = self.blocking_config.get("pc_threshold")
        if min_blocking_pc_for_matching is None:
            min_blocking_pc_for_matching = configured_blocking_threshold
        self.min_blocking_pc_for_matching = float(min_blocking_pc_for_matching) if min_blocking_pc_for_matching is not None else None
        self.proxy_f1_margin = max(0.0, float(proxy_f1_margin))
        self.proxy_max_candidates = max(1000, int(proxy_max_candidates))
        self.no_gain_patience = max(1, int(no_gain_patience))
        self.no_gain_epsilon = max(0.0, float(no_gain_epsilon))

        os.makedirs(self.output_dir, exist_ok=True)

        self.datasets_loaded: Dict[str, pd.DataFrame] = {}
        for path in datasets:
            name = os.path.splitext(os.path.basename(path))[0]
            if self.verbose:
                print(f"[*] Loading dataset: {name}")
            self.datasets_loaded[name] = load_dataset(path)
        self.list_like_columns = detect_list_like_columns(
            list(self.datasets_loaded.values()),
            exclude_columns={"id", "_id", "record_id", "__record_id__"},
        )
        if self.list_like_columns:
            normalize_list_like_columns(
                list(self.datasets_loaded.values()),
                self.list_like_columns,
            )
            if self.verbose:
                print(
                    f"[*] Normalized list-like columns for matching: "
                    f"{', '.join(self.list_like_columns)}"
                )

        self.gold_standards: Dict[Tuple[str, str], pd.DataFrame] = {}
        for pair, path in matching_testsets.items():
            if self.verbose:
                print(f"[*] Loading matching gold standard for {pair[0]} <-> {pair[1]}")
            self.gold_standards[pair] = self._load_gold_standard(path)

    @staticmethod
    def _coerce_response_text(content: Any) -> str:
        """Normalize LLM content payloads (str/list/dict) into plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text_val = item.get("text") or item.get("content") or item.get("output_text")
                    if isinstance(text_val, str):
                        parts.append(text_val)
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p).strip()
        if isinstance(content, dict):
            text_val = content.get("text") or content.get("content") or content.get("output_text")
            if isinstance(text_val, str):
                return text_val
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    def _read_gold_standard(self, path: str) -> pd.DataFrame:
        if path.lower().endswith((".txt", ".tsv")):
            return pd.read_csv(path, sep=None, engine="python")
        return pd.read_csv(path)

    def _load_gold_standard(self, path: str) -> pd.DataFrame:
        gs = self._read_gold_standard(path)
        col_mapping = {}
        for col in gs.columns:
            col_lower = col.lower()
            if "id_a" in col_lower or col_lower == "id1":
                col_mapping[col] = "id1"
            elif "id_b" in col_lower or col_lower == "id2":
                col_mapping[col] = "id2"
            elif "label" in col_lower:
                col_mapping[col] = "label"
        gs = gs.rename(columns=col_mapping)
        if "label" not in gs.columns:
            gs["label"] = 1
        if self.verbose:
            print(f"    Loaded {len(gs)} ground truth pairs")
        return gs[["id1", "id2", "label"]]

    def _detect_id_column(self, df: pd.DataFrame) -> str:
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ["id", "record_id", "row_id", "index", "key"]:
                return col
            if "id" in col_lower:
                try:
                    if df[col].nunique() == len(df):
                        return col
                except TypeError:
                    pass
        return df.columns[0]

    def _candidate_id_columns(self, df: pd.DataFrame) -> List[str]:
        candidates = []
        for col in df.columns:
            if "id" in col.lower():
                candidates.append(col)
        for col in df.columns:
            if col in candidates:
                continue
            try:
                if df[col].nunique() == len(df):
                    candidates.append(col)
            except TypeError:
                pass
        if not candidates:
            candidates = [df.columns[0]]
        return candidates[:5]

    def _sample_values(self, series: pd.Series, n: int = 3) -> List[str]:
        vals = series.dropna().astype(str).unique().tolist()
        return vals[:n]

    @staticmethod
    def _safe_nunique(series: pd.Series) -> int:
        if not hasattr(series, "nunique"):
            return -1
        try:
            return int(series.nunique(dropna=True))
        except TypeError:
            # pandas cannot hash raw list/dict objects; coerce nested values to stable scalars first.
            try:
                coerced = series.dropna().apply(
                    lambda v: tuple(v)
                    if isinstance(v, list)
                    else tuple(sorted(v))
                    if isinstance(v, set)
                    else json.dumps(v, sort_keys=True, ensure_ascii=False)
                    if isinstance(v, dict)
                    else str(v)
                )
                return int(coerced.nunique(dropna=True))
            except Exception:
                return -1
        except Exception:
            return -1

    def _align_ids_with_gold(
        self,
        name_left: str,
        name_right: str,
        gold: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, str]:
        df_left = self.datasets_loaded[name_left]
        df_right = self.datasets_loaded[name_right]

        left_id = self._detect_id_column(df_left)
        right_id = self._detect_id_column(df_right)
        gold_id1_source = "left"

        if self.llm is not None:
            try:
                left_candidates = self._candidate_id_columns(df_left)
                right_candidates = self._candidate_id_columns(df_right)
                left_samples = {c: self._sample_values(df_left[c]) for c in left_candidates}
                right_samples = {c: self._sample_values(df_right[c]) for c in right_candidates}
                gold_samples = {
                    "id1": self._sample_values(gold["id1"]),
                    "id2": self._sample_values(gold["id2"]),
                }

                system_prompt = (
                    "You map gold testset ID columns (id1/id2) to dataset ID columns. "
                    "Return ONLY JSON with keys: left_id_col, right_id_col, gold_id1_source "
                    "where gold_id1_source is 'left' or 'right'."
                )
                human_content = (
                    f"Left dataset: {name_left}\n"
                    f"Left ID candidates (sample values): {json.dumps(left_samples, indent=2)}\n\n"
                    f"Right dataset: {name_right}\n"
                    f"Right ID candidates (sample values): {json.dumps(right_samples, indent=2)}\n\n"
                    f"Gold id1/id2 samples: {json.dumps(gold_samples, indent=2)}\n"
                    "Pick the best left/right ID columns and indicate whether gold id1 belongs to left or right."
                )
                response = self.llm.invoke(
                    [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
                )
                raw_content = response.content if hasattr(response, "content") else response
                raw = self._coerce_response_text(raw_content)
                cleaned = re.sub(r"^```(?:json)?\\n?|```$", "", raw.strip())
                parsed = json.loads(cleaned)
                left_choice = parsed.get("left_id_col")
                right_choice = parsed.get("right_id_col")
                gold_id1_source = parsed.get("gold_id1_source", "left")
                if left_choice in df_left.columns:
                    left_id = left_choice
                if right_choice in df_right.columns:
                    right_id = right_choice
                if gold_id1_source == "right":
                    gold = gold.rename(columns={"id1": "id2", "id2": "id1"})
            except Exception:
                pass

        self.id_columns[name_left] = left_id
        self.id_columns[name_right] = right_id
        self.last_id_alignment = {
            "left_id": left_id,
            "right_id": right_id,
            "gold_id1_source": gold_id1_source,
        }
        if left_id != right_id:
            self.datasets_loaded[name_left]["__record_id__"] = self.datasets_loaded[name_left][left_id]
            self.datasets_loaded[name_right]["__record_id__"] = self.datasets_loaded[name_right][right_id]
            return gold, "__record_id__"
        return gold, left_id

    def _resolve_id_column(self, name_left: str, name_right: str, id_column: str = None) -> str:
        if id_column:
            return id_column

        id_left = self.id_columns.get(name_left) or self._detect_id_column(self.datasets_loaded[name_left])
        id_right = self.id_columns.get(name_right) or self._detect_id_column(self.datasets_loaded[name_right])

        if id_left != id_right:
            if self.verbose:
                print(f"    ID columns differ ({id_left}/{id_right}), renaming to 'record_id'")
            self.datasets_loaded[name_left] = self.datasets_loaded[name_left].rename(columns={id_left: "record_id"})
            self.datasets_loaded[name_right] = self.datasets_loaded[name_right].rename(columns={id_right: "record_id"})
            id_column = "record_id"
        else:
            id_column = id_left

        self.id_columns[name_left] = id_column
        self.id_columns[name_right] = id_column
        return id_column

    def _infer_column_type(self, series: pd.Series, column_name: str) -> str:
        name_lower = column_name.lower()
        if "date" in name_lower or "year" in name_lower or "time" in name_lower:
            return "date"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "date"
        sample = series.dropna().head(10).tolist()
        if any(is_list_like_value(v) for v in sample):
            return "list"
        return "string"

    def _analyze_columns_for_pair(self, name_left: str, name_right: str, id_column: str = None) -> Dict[str, Any]:
        df_left = self.datasets_loaded[name_left]
        df_right = self.datasets_loaded[name_right]

        common_cols = set(df_left.columns) & set(df_right.columns)
        if id_column:
            common_cols.discard(id_column)
        common_cols.discard("id")

        column_details = {}
        column_types = {}
        eligible_common_cols: List[str] = []
        for col in sorted(common_cols):
            left_col, right_col = df_left[col], df_right[col]
            left_type = self._infer_column_type(left_col, col)
            right_type = self._infer_column_type(right_col, col)
            if "list" in (left_type, right_type):
                col_type = "list"
            elif "numeric" in (left_type, right_type):
                col_type = "numeric"
            elif "date" in (left_type, right_type):
                col_type = "date"
            else:
                col_type = "string"

            if self.disallow_list_comparators and col_type == "list":
                continue

            eligible_common_cols.append(col)
            column_types[col] = col_type
            try:
                sample_left = [str(x)[:50] for x in left_col.dropna().head(2).tolist()]
                sample_right = [str(x)[:50] for x in right_col.dropna().head(2).tolist()]
            except Exception:
                sample_left, sample_right = [], []
            column_details[col] = {
                "dtype": str(left_col.dtype),
                "type": col_type,
                "null_pct": f"{left_col.isnull().mean()*100:.0f}%/{right_col.isnull().mean()*100:.0f}%",
                "null_rate_left": float(left_col.isnull().mean()),
                "null_rate_right": float(right_col.isnull().mean()),
                "unique_left": self._safe_nunique(left_col),
                "unique_right": self._safe_nunique(right_col),
                "samples": sample_left[:1] + sample_right[:1],
            }

        return {
            "left_dataset": name_left,
            "right_dataset": name_right,
            "common_columns": eligible_common_cols,
            "column_details": column_details,
            "column_types": column_types,
        }

    def _get_blocking_config(self, name_left: str, name_right: str) -> Tuple[Optional[Dict[str, Any]], str]:
        key = f"{name_left}_{name_right}"
        if key in self.blocking_strategies:
            return self.blocking_strategies[key], key
        reverse_key = f"{name_right}_{name_left}"
        if reverse_key in self.blocking_strategies:
            return self.blocking_strategies[reverse_key], reverse_key
        return None, key

    def _get_blocking_quality_for_pair(self, name_left: str, name_right: str) -> Dict[str, Any]:
        blocking_cfg, key = self._get_blocking_config(name_left, name_right)
        if not isinstance(blocking_cfg, dict):
            return {"pair_key": key, "available": False}
        pc = float(blocking_cfg.get("pair_completeness", 0.0) or 0.0)
        return {
            "pair_key": key,
            "available": True,
            "pair_completeness": pc,
            "is_acceptable": bool(blocking_cfg.get("is_acceptable", False)),
            "strategy": blocking_cfg.get("strategy"),
            "columns": list(blocking_cfg.get("columns", []) or []),
        }

    def _should_skip_pair_for_blocking_quality(self, name_left: str, name_right: str) -> Tuple[bool, Dict[str, Any]]:
        quality = self._get_blocking_quality_for_pair(name_left, name_right)
        threshold = self.min_blocking_pc_for_matching
        if threshold is None:
            return False, {"threshold": None, **quality}
        pair_pc = float(quality.get("pair_completeness", 0.0) or 0.0)
        skip = quality.get("available", False) and pair_pc < float(threshold)
        return skip, {"threshold": float(threshold), **quality}

    def _sample_candidate_pairs(self, candidates: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if not isinstance(candidates, pd.DataFrame):
            return candidates
        if len(candidates) <= max_rows:
            return candidates
        return candidates.sample(n=max_rows, random_state=42).reset_index(drop=True)

    def _append_llm_trace(self, payload: Dict[str, Any]) -> None:
        trace_path = os.path.join(self.output_dir, "llm_matching_trace.jsonl")
        try:
            with open(trace_path, "a", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
                f.write("\n")
        except Exception:
            return

    @staticmethod
    def _utc_now_z() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _trace_event(
        self,
        *,
        pair: str,
        decision: str,
        attempt: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "timestamp": self._utc_now_z(),
            "pair": pair,
            "decision": decision,
        }
        if attempt is not None:
            payload["attempt"] = attempt
        if isinstance(extra, dict) and extra:
            payload.update(extra)
        self._append_llm_trace(payload)

    def _emit_confidence_stop_if_needed(
        self,
        *,
        pair: str,
        attempt: int,
        f1_history: List[float],
        best_result: Optional[Dict[str, Any]],
    ) -> bool:
        confidence_stop = self._confidence_stop_for_f1(
            f1_history=f1_history,
            best_f1=float(best_result.get("f1", 0.0) if best_result else 0.0),
            target_f1=self.f1_threshold,
            epsilon=self.no_gain_epsilon,
        )
        if not confidence_stop.get("stop"):
            return False
        print(
            "⏹️ Confidence stop: "
            f"{confidence_stop.get('reason')} "
            f"(confidence={confidence_stop.get('confidence')})"
        )
        self._trace_event(
            pair=pair,
            attempt=attempt,
            decision="confidence_stop",
            extra={"details": confidence_stop},
        )
        return True

    @staticmethod
    def _confidence_stop_for_f1(
        f1_history: List[float],
        best_f1: float,
        target_f1: float,
        epsilon: float,
    ) -> Dict[str, Any]:
        # Conservative stop heuristic: only stop after enough exploration and only
        # once the search is already near a plausible plateau. Low-F1 runs should
        # continue exploring rather than converging early on a bad region.
        if len(f1_history) < 5:
            return {"stop": False, "reason": "insufficient_history", "confidence": 0.0}
        exploration_floor = max(0.6, target_f1 - 0.12)
        if best_f1 < exploration_floor:
            return {
                "stop": False,
                "reason": "best_f1_still_too_low_for_confidence_stop",
                "confidence": 0.0,
                "best_f1": round(best_f1, 6),
                "exploration_floor": round(exploration_floor, 6),
            }
        last = f1_history[-3:]
        span = max(last) - min(last)
        mean_last = sum(last) / len(last)
        gap = max(0.0, target_f1 - best_f1)
        if span <= max(epsilon, 0.002) and gap > 0.02:
            confidence = min(0.98, 0.65 + (0.10 * (len(f1_history) - 2)))
            return {
                "stop": True,
                "reason": "low_variance_below_target",
                "confidence": round(confidence, 3),
                "mean_last": round(mean_last, 6),
                "span_last": round(span, 6),
                "target_gap": round(gap, 6),
            }
        return {
            "stop": False,
            "reason": "continue_search",
            "confidence": 0.0,
            "mean_last": round(mean_last, 6),
            "span_last": round(span, 6),
            "target_gap": round(gap, 6),
        }

    @staticmethod
    def _failure_tags_for_matching_event(
        *,
        skipped_due_to_blocking_gate: bool = False,
        proxy_rejected: bool = False,
        f1: float = 0.0,
        target_f1: float = 0.75,
        error: str = "",
        fallback_blocking: bool = False,
    ) -> List[str]:
        tags: List[str] = []
        if skipped_due_to_blocking_gate:
            tags.append("low_blocking_recall")
        if proxy_rejected:
            tags.append("proxy_rejected")
        if error:
            tags.append("runtime_error")
        if fallback_blocking:
            tags.append("blocking_fallback")
        if f1 < target_f1:
            tags.append("low_matching_quality")
        if not tags:
            tags.append("ok")
        return tags

    def _build_matching_guardrails(
        self,
        analysis: Dict[str, Any],
        blocking_columns: List[str],
    ) -> Dict[str, Any]:
        common_columns = analysis.get("common_columns", []) if isinstance(analysis, dict) else []
        details = analysis.get("column_details", {}) if isinstance(analysis, dict) else {}
        column_types = analysis.get("column_types", {}) if isinstance(analysis, dict) else {}
        quality_scores: List[Tuple[float, str]] = []

        for col in common_columns:
            detail = details.get(col, {}) if isinstance(details, dict) else {}
            null_left = float(detail.get("null_rate_left", 1.0) or 1.0)
            null_right = float(detail.get("null_rate_right", 1.0) or 1.0)
            null_penalty = (null_left + null_right) / 2.0
            uniq_left = int(detail.get("unique_left", -1))
            uniq_right = int(detail.get("unique_right", -1))
            uniqueness = 0.0
            if uniq_left > 0 and uniq_right > 0:
                uniqueness = min(uniq_left, uniq_right) / max(uniq_left, uniq_right)
            score = max(0.0, min(1.0, (1.0 - null_penalty) * 0.7 + uniqueness * 0.3))
            if col in blocking_columns:
                score += 0.08
            if column_types.get(col) == "list":
                score -= 0.20
            quality_scores.append((score, col))

        quality_scores.sort(reverse=True, key=lambda x: x[0])
        preferred_columns = [c for score, c in quality_scores if score >= 0.25][:6]
        if not preferred_columns:
            preferred_columns = [c for _, c in quality_scores[:4]]
        return {
            "preferred_columns": preferred_columns,
            "blocking_columns": [c for c in blocking_columns if c in common_columns],
            "avoid_list_columns": bool(self.disallow_list_comparators),
        }

    def _apply_matcher_guardrails(
        self,
        choice: Dict[str, Any],
        guardrails: Dict[str, Any],
        valid_columns: List[str],
        column_types: Dict[str, str],
    ) -> Dict[str, Any]:
        if not isinstance(choice, dict):
            return choice
        preferred = [c for c in guardrails.get("preferred_columns", []) if c in valid_columns]
        blocking_cols = [c for c in guardrails.get("blocking_columns", []) if c in valid_columns]
        out = dict(choice)
        comparators = []
        for comp in out.get("comparators", []):
            if not isinstance(comp, dict):
                continue
            col = comp.get("column")
            if col not in valid_columns:
                continue
            if self.disallow_list_comparators and (
                column_types.get(col) == "list" or bool(comp.get("list_strategy"))
            ):
                continue
            comparators.append(comp)

        if blocking_cols and not any(c.get("column") in blocking_cols for c in comparators):
            seed = next((c for c in blocking_cols if c in preferred), blocking_cols[0])
            comparators.insert(
                0,
                {
                    "type": "string",
                    "column": seed,
                    "similarity_function": "cosine",
                    "preprocess": "lower",
                },
            )
        if not comparators:
            defaults = self._default_comparators(valid_columns, column_types, blocking_cols)
            comparators = defaults[:]
        out["comparators"] = comparators
        out["weights"] = self._normalize_weights(out.get("weights", []), len(comparators))
        return out

    def _create_blocker_from_config(
        self,
        name_left: str,
        name_right: str,
        id_column: str,
        common_columns: List[str],
        blocking_cfg: Optional[Dict[str, Any]],
    ):
        df_left = self.datasets_loaded[name_left]
        df_right = self.datasets_loaded[name_right]
        common_params = {"output_dir": self.output_dir, "id_column": id_column}

        if not blocking_cfg:
            fallback_col = common_columns[0] if common_columns else id_column
            if self.verbose:
                print(f"    No blocking config found. Falling back to StandardBlocker on {fallback_col}")
            return StandardBlocker(df_left, df_right, on=[fallback_col], batch_size=1000, **common_params)

        strategy = blocking_cfg.get("strategy")
        columns = [c for c in blocking_cfg.get("columns", []) if c in common_columns]
        if not columns:
            columns = [common_columns[0]] if common_columns else [id_column]

        # Keep blocking robust against nested/list values that can crash PyDI blockers
        # (notably EmbeddingBlocker calling pd.isna on ndarray/list objects).
        def _is_scalar_friendly(col: str) -> bool:
            return not (
                self._column_has_sequence_like_values(df_left, col)
                or self._column_has_sequence_like_values(df_right, col)
            )

        if self.disallow_list_comparators:
            scalar_columns = [c for c in columns if _is_scalar_friendly(c)]
            if scalar_columns:
                columns = scalar_columns

        for col in columns:
            self._sanitize_dataframe_column_for_comparator(
                self.datasets_loaded[name_left],
                col,
                ctype="string",
                list_strategy=None,
            )
            self._sanitize_dataframe_column_for_comparator(
                self.datasets_loaded[name_right],
                col,
                ctype="string",
                list_strategy=None,
            )

        if strategy in ["exact_match_single", "exact_match_multi"]:
            return StandardBlocker(df_left, df_right, on=columns, batch_size=1000, **common_params)
        params = blocking_cfg.get("params", {}) if isinstance(blocking_cfg.get("params"), dict) else {}
        if strategy == "semantic_similarity":
            return EmbeddingBlocker(
                df_left,
                df_right,
                text_cols=columns,
                top_k=params.get("top_k", blocking_cfg.get("top_k", 20)),
                batch_size=1000,
                **common_params,
            )
        if strategy == "token_blocking":
            return TokenBlocker(
                df_left,
                df_right,
                column=columns[0],
                min_token_len=params.get("min_token_len", 4),
                batch_size=1000,
                **common_params,
            )
        if strategy == "ngram_blocking":
            return TokenBlocker(
                df_left,
                df_right,
                column=columns[0],
                ngram_size=params.get("ngram_size", 4),
                ngram_type="character",
                batch_size=1000,
                **common_params,
            )
        if strategy == "sorted_neighbourhood":
            return SortedNeighbourhoodBlocker(
                df_left,
                df_right,
                key=columns[0],
                window=params.get("window", 15),
                batch_size=1000,
                **common_params,
            )
        return StandardBlocker(df_left, df_right, on=columns, batch_size=1000, **common_params)

    def _preprocess_from_name(self, name: Optional[str]):
        if name == "lower":
            return str.lower
        if name == "strip":
            return str.strip
        if name == "lower_strip":
            return lambda x: str(x).lower().strip()
        return None

    def _default_comparators(
        self,
        valid_columns: List[str],
        column_types: Dict[str, str],
        blocking_columns: List[str],
    ) -> List[Dict[str, Any]]:
        ordered = []
        for col in blocking_columns:
            if col in valid_columns and col not in ordered:
                ordered.append(col)
        for col in valid_columns:
            if col not in ordered:
                ordered.append(col)

        if self.disallow_list_comparators:
            scalar_cols = [c for c in ordered if column_types.get(c, "string") != "list"]
            if scalar_cols:
                ordered = scalar_cols

        comparators = []
        for col in ordered[:4]:
            ctype = column_types.get(col, "string")
            if ctype == "numeric":
                comparators.append({"type": "numeric", "column": col, "max_difference": 5})
            elif ctype == "date":
                comparators.append({"type": "date", "column": col, "max_days_difference": 365})
            else:
                comp = {
                    "type": "string",
                    "column": col,
                    "similarity_function": "cosine",
                    "preprocess": "lower",
                }
                if ctype == "list":
                    comp["list_strategy"] = "concatenate"
                comparators.append(comp)
        return comparators

    def _normalize_weights(self, weights: List[Any], count: int) -> List[float]:
        cleaned = []
        for w in weights:
            try:
                w_float = float(w)
            except (TypeError, ValueError):
                continue
            if w_float >= 0:
                cleaned.append(w_float)
        if len(cleaned) != count or count == 0:
            return [1.0 / count] * count if count else []
        total = sum(cleaned)
        return [w / total for w in cleaned] if total > 0 else [1.0 / count] * count

    def _parse_llm_response(
        self,
        response_text: Any,
        valid_columns: List[str],
        column_types: Dict[str, str],
        blocking_columns: List[str],
    ) -> Dict[str, Any]:
        response_text = self._coerce_response_text(response_text)
        try:
            cleaned = re.sub(r"^```(?:json)?\\n?|```$", "", response_text.strip())
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\\{[^{}]+\\}", response_text, re.DOTALL)
            parsed = json.loads(match.group()) if match else {}

        raw_comps = parsed.get("comparators", [])
        if not isinstance(raw_comps, list):
            raw_comps = []

        allowed_similarity = {"jaccard", "jaro_winkler", "levenshtein", "cosine"}
        allowed_preprocess = {"lower", "strip", "lower_strip", "none"}
        allowed_list_strategy = {"concatenate", "set_jaccard", "best_match", "set_overlap"}
        allowed_numeric_list_strategy = {"average", "best_match", "range_overlap", "set_jaccard"}
        cleaned_comps = []

        for comp in raw_comps:
            if not isinstance(comp, dict):
                continue
            column = comp.get("column")
            if column not in valid_columns:
                continue
            if self.disallow_list_comparators and (
                column_types.get(column) == "list" or bool(comp.get("list_strategy"))
            ):
                continue
            ctype = comp.get("type") or column_types.get(column, "string")
            if ctype not in {"string", "numeric", "date"}:
                ctype = column_types.get(column, "string")

            entry = {"type": ctype, "column": column}
            if ctype == "string":
                sim = comp.get("similarity_function", "cosine")
                if sim not in allowed_similarity:
                    sim = "cosine"
                entry["similarity_function"] = sim
                preprocess = comp.get("preprocess", "none")
                if preprocess not in allowed_preprocess:
                    preprocess = "none"
                if preprocess != "none":
                    entry["preprocess"] = preprocess
                list_strategy = comp.get("list_strategy")
                if list_strategy in allowed_list_strategy:
                    entry["list_strategy"] = list_strategy
                elif column_types.get(column) == "list":
                    entry["list_strategy"] = "concatenate"
            elif ctype == "numeric":
                max_diff = comp.get("max_difference", 5)
                try:
                    max_diff = float(max_diff)
                except (TypeError, ValueError):
                    max_diff = 5
                entry["max_difference"] = max_diff
                num_list_strategy = comp.get("list_strategy")
                if num_list_strategy in allowed_numeric_list_strategy:
                    entry["list_strategy"] = num_list_strategy
                elif column_types.get(column) == "list":
                    entry["list_strategy"] = "average"
            else:
                max_days = comp.get("max_days_difference", 365)
                try:
                    max_days = int(max_days)
                except (TypeError, ValueError):
                    max_days = 365
                entry["max_days_difference"] = max_days
            cleaned_comps.append(entry)

        if not cleaned_comps:
            cleaned_comps = self._default_comparators(valid_columns, column_types, blocking_columns)

        if blocking_columns and not any(c["column"] in blocking_columns for c in cleaned_comps):
            col = next((c for c in blocking_columns if c in valid_columns), None)
            if col:
                cleaned_comps.insert(
                    0,
                    {
                        "type": "string",
                        "column": col,
                        "similarity_function": "cosine",
                        "preprocess": "lower",
                    },
                )

        weights = parsed.get("weights", [])
        weights = self._normalize_weights(weights, len(cleaned_comps))

        threshold = parsed.get("threshold", 0.75)
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = 0.75
        threshold = max(0.0, min(1.0, threshold))

        return {
            "comparators": cleaned_comps,
            "weights": weights,
            "threshold": threshold,
            "reasoning": parsed.get("reasoning", ""),
        }

    def _default_choice(
        self,
        valid_columns: List[str],
        column_types: Dict[str, str],
        blocking_columns: List[str],
    ) -> Dict[str, Any]:
        comps = self._default_comparators(valid_columns, column_types, blocking_columns)
        return {
            "comparators": comps,
            "weights": self._normalize_weights([], len(comps)),
            "threshold": 0.75,
            "reasoning": "default heuristic",
        }

    def _ask_llm_for_matcher(
        self,
        analysis: Dict[str, Any],
        blocking_columns: List[str],
        matcher_mode: str,
        previous_attempts: List[Dict[str, Any]] = None,
        last_error: str = None,
        guardrails: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.llm is None:
            return self._default_choice(
                analysis["common_columns"], analysis["column_types"], blocking_columns
            )

        system_prompt = f"""Select the best RuleBasedMatcher configuration based on the provided schema/profile context. Respond with ONLY valid JSON:
{{
  "comparators": [
    {{"type": "string", "column": "col1", "similarity_function": "cosine", "preprocess": "lower", "list_strategy": "concatenate"}},
    {{"type": "numeric", "column": "col2", "max_difference": 5, "list_strategy": "average"}},
    {{"type": "date", "column": "col3", "max_days_difference": 365}}
  ],
  "weights": [0.5, 0.3, 0.2],
  "threshold": 0.7,
  "reasoning": "brief reason"
}}

{MATCHING_STRATEGY_DESCRIPTIONS}
Guidance:
- Use only columns in common_columns.
- Inspect column_details (dtype, null_pct, samples) and column names to infer semantics and informativeness.
- Prefer informative columns (names, addresses, titles, categories, dates, numeric measures); avoid high-null/noisy columns unless necessary.
- Choose comparator type by inferred data type (string/numeric/date). For list-like fields, set list_strategy.
- Choose similarity_function based on content:
  - jaro_winkler or levenshtein for short strings/typos
  - jaccard or cosine for token sets/longer text/list-ish strings
  - cosine is the default fallback for strings if unsure
- Use preprocess to normalize text: lower, strip, lower_strip, none.
- list_strategy options for string comparators: concatenate, set_jaccard, best_match, set_overlap.
- list_strategy options for numeric comparators: average, best_match, range_overlap, set_jaccard.
- If blocking columns are provided, include at least one comparator using them unless they are clearly low quality.
- Weights should be non-negative, sum to 1, and match the number of comparators.
- Set threshold to balance precision/recall based on data quality (higher for strong identifiers, lower for noisy text).
- If prior attempts fail or error, fall back to a simpler RuleBasedMatcher config.
- If matcher_mode is ml, prefer fewer informative columns; more columns does not always improve results.
"""
        if self.disallow_list_comparators:
            system_prompt += """
- IMPORTANT: Avoid list-like columns entirely (for example track lists and list-valued durations).
- Prefer scalar columns (name, artist, date, numeric scalar fields).
- Do NOT use list_strategy; choose comparators that operate on scalar columns.
"""

        human_content = (
            f"Dataset pair: {analysis['left_dataset']} <-> {analysis['right_dataset']}\n"
            f"Blocking columns: {blocking_columns}\n"
            f"Matcher mode: {matcher_mode}\n"
            f"Common columns: {analysis['common_columns']}\n"
            f"Column details: {json.dumps(analysis['column_details'], indent=2)}"
        )
        if self.disallow_list_comparators:
            list_like_cols = [c for c in analysis["common_columns"] if analysis["column_types"].get(c) == "list"]
            if list_like_cols:
                human_content += (
                    f"\nList-like columns to avoid for comparators: {list_like_cols}"
                )
        if isinstance(guardrails, dict) and guardrails:
            human_content += (
                f"\n\nPAIR GUARDRAILS:\n"
                f"- preferred_columns={guardrails.get('preferred_columns', [])}\n"
                f"- blocking_columns={guardrails.get('blocking_columns', [])}\n"
                f"- avoid_list_columns={guardrails.get('avoid_list_columns', False)}"
            )

        failure_summary = self._format_previous_failures(analysis['left_dataset'], analysis['right_dataset'])
        if failure_summary:
            human_content += f"\n\nPREVIOUS FAILED MATCHER CONFIGS (avoid repeating these):\n{failure_summary}"

        if last_error:
            human_content += f"\n\nERROR: {last_error[:300]}\nFix the configuration."
        elif previous_attempts:
            attempts_str = ", ".join(
                [f"{a['columns']}({a.get('f1', 0):.2f})" for a in previous_attempts]
            )
            human_content += (
                f"\n\nPrevious attempts below threshold ({self.f1_threshold}): {attempts_str}."
                " Try different comparators/weights/threshold."
            )

        response = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
        )
        response_content = response.content if hasattr(response, "content") else response
        response_text = self._coerce_response_text(response_content)
        parsed = self._parse_llm_response(
            response_text, analysis["common_columns"], analysis["column_types"], blocking_columns
        )
        return self._apply_matcher_guardrails(
            parsed,
            guardrails or {},
            analysis["common_columns"],
            analysis["column_types"],
        )

    def _format_previous_failures(self, name_left: str, name_right: str) -> str:
        if not self.previous_failures:
            return ""
        pair_key = f"{name_left}_{name_right}"
        relevant = [f for f in self.previous_failures if f.get("pair") == pair_key]
        if not relevant:
            return ""
        lines = []
        for item in relevant[-10:]:
            cols = [c.get("column") for c in item.get("comparators", [])]
            weights = item.get("weights")
            threshold = item.get("threshold")
            f1 = item.get("f1", 0)
            lines.append(f"- cols={cols} weights={weights} threshold={threshold} F1={f1:.3f}")
        return "\n".join(lines)

    def _build_comparators(self, comparator_configs: List[Dict[str, Any]]):
        comparators = []
        for comp in comparator_configs:
            ctype = comp.get("type")
            column = comp.get("column")
            if ctype == "numeric":
                kwargs = {
                    "column": column,
                    "max_difference": comp.get("max_difference", 5),
                }
                if comp.get("list_strategy"):
                    kwargs["list_strategy"] = comp["list_strategy"]
                comparators.append(NumericComparator(**kwargs))
            elif ctype == "date":
                comparators.append(
                    DateComparator(column=column, max_days_difference=comp.get("max_days_difference", 365))
                )
            else:
                kwargs = {
                    "column": column,
                    "similarity_function": comp.get("similarity_function", "cosine"),
                }
                preprocess = self._preprocess_from_name(comp.get("preprocess"))
                if preprocess:
                    kwargs["preprocess"] = preprocess
                if comp.get("list_strategy"):
                    kwargs["list_strategy"] = comp["list_strategy"]
                comparators.append(StringComparator(**kwargs))
        return comparators

    @staticmethod
    def _is_sequence_like_value(value: Any) -> bool:
        return isinstance(value, (list, tuple, set, np.ndarray, pd.Series)) or is_list_like_value(value)

    def _column_has_sequence_like_values(self, df: pd.DataFrame, column: str, sample_size: int = 200) -> bool:
        if column not in df.columns:
            return False
        sample = df[column].dropna().head(sample_size)
        return any(self._is_sequence_like_value(v) for v in sample)

    @staticmethod
    def _coerce_numeric_sequence(values: List[Any]) -> List[float]:
        out: List[float] = []
        for item in values:
            try:
                num = float(item)
            except (TypeError, ValueError):
                continue
            if pd.isna(num):
                continue
            out.append(num)
        return out

    def _sanitize_dataframe_column_for_comparator(
        self,
        df: pd.DataFrame,
        column: str,
        ctype: str,
        list_strategy: Optional[str],
    ) -> None:
        if column not in df.columns:
            return

        want_list = bool(list_strategy)

        def _to_sequence(value: Any) -> List[Any]:
            if value is None:
                return []
            if isinstance(value, float) and pd.isna(value):
                return []
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, pd.Series):
                value = value.tolist()
            if isinstance(value, (list, tuple, set)):
                return list(value)
            if is_list_like_value(value):
                return normalize_list_value(value, dedupe=False)
            return [value]

        if want_list:
            if ctype == "numeric":
                df[column] = df[column].apply(lambda v: self._coerce_numeric_sequence(_to_sequence(v)))
            else:
                def _to_str_list(v: Any) -> List[str]:
                    seq = _to_sequence(v)
                    out = []
                    for item in seq:
                        text = str(item).strip()
                        if not text or text.lower() in {"nan", "none", "null"}:
                            continue
                        out.append(text)
                    return out
                df[column] = df[column].apply(_to_str_list)
            return

        if ctype == "numeric":
            def _to_numeric_scalar(v: Any):
                seq = _to_sequence(v) if self._is_sequence_like_value(v) else [v]
                nums = self._coerce_numeric_sequence(seq)
                return nums[0] if nums else np.nan
            df[column] = df[column].apply(_to_numeric_scalar)
            return

        # String/date comparator without list strategy: collapse list-like to scalar text.
        def _to_scalar_text(v: Any):
            if self._is_sequence_like_value(v):
                seq = _to_sequence(v)
                tokens = []
                for item in seq:
                    text = str(item).strip()
                    if not text or text.lower() in {"nan", "none", "null"}:
                        continue
                    tokens.append(text)
                return " | ".join(tokens)
            return v

        df[column] = df[column].apply(_to_scalar_text)

    def _stabilize_choice_and_data(
        self,
        name_left: str,
        name_right: str,
        choice: Dict[str, Any],
    ) -> Dict[str, Any]:
        stabilized = dict(choice)
        stabilized_comparators: List[Dict[str, Any]] = []
        left_df = self.datasets_loaded[name_left]
        right_df = self.datasets_loaded[name_right]

        for comp in choice.get("comparators", []):
            comp_fixed = dict(comp)
            column = comp_fixed.get("column")
            ctype = comp_fixed.get("type", "string")
            if not column:
                continue

            has_sequence = self._column_has_sequence_like_values(left_df, column) or self._column_has_sequence_like_values(
                right_df, column
            )
            if has_sequence and not comp_fixed.get("list_strategy"):
                if ctype == "numeric":
                    comp_fixed["list_strategy"] = "average"
                elif ctype == "string":
                    comp_fixed["list_strategy"] = "concatenate"

            self._sanitize_dataframe_column_for_comparator(
                self.datasets_loaded[name_left],
                column,
                ctype=ctype,
                list_strategy=comp_fixed.get("list_strategy"),
            )
            self._sanitize_dataframe_column_for_comparator(
                self.datasets_loaded[name_right],
                column,
                ctype=ctype,
                list_strategy=comp_fixed.get("list_strategy"),
            )
            stabilized_comparators.append(comp_fixed)

        stabilized["comparators"] = stabilized_comparators
        return stabilized

    def _filter_ml_comparators(
        self,
        comparator_configs: List[Dict[str, Any]],
        common_columns: List[str],
        column_types: Dict[str, str],
        blocking_columns: List[str],
    ) -> List[Dict[str, Any]]:
        filtered = [c for c in comparator_configs if c.get("column") in common_columns]
        if not filtered:
            filtered = self._default_comparators(common_columns, column_types, blocking_columns)
        return filtered

    def _prepare_active_choice(
        self,
        *,
        choice: Dict[str, Any],
        matcher_mode: str,
        analysis: Dict[str, Any],
        blocking_columns: List[str],
    ) -> Dict[str, Any]:
        active_choice = dict(choice)
        if matcher_mode == "ml":
            active_choice["comparators"] = self._filter_ml_comparators(
                choice["comparators"],
                analysis["common_columns"],
                analysis["column_types"],
                blocking_columns,
            )
        return self._apply_execution_choice_guard(active_choice, analysis, blocking_columns)

    def _apply_execution_choice_guard(
        self,
        choice: Dict[str, Any],
        analysis: Dict[str, Any],
        blocking_columns: List[str],
    ) -> Dict[str, Any]:
        filtered_comps = [
            c
            for c in choice.get("comparators", [])
            if not (
                self.disallow_list_comparators
                and (
                    analysis["column_types"].get(c.get("column")) == "list"
                    or bool(c.get("list_strategy"))
                )
            )
        ]
        if len(filtered_comps) == len(choice.get("comparators", [])):
            return choice

        active_choice = dict(choice)
        active_choice["comparators"] = filtered_comps
        active_choice["weights"] = self._normalize_weights(
            active_choice.get("weights", []),
            len(filtered_comps),
        )
        if not active_choice["comparators"]:
            active_choice["comparators"] = self._default_comparators(
                analysis["common_columns"],
                analysis["column_types"],
                blocking_columns,
            )
            active_choice["weights"] = self._normalize_weights([], len(active_choice["comparators"]))
        return active_choice

    def _normalize_metrics(self, metrics: Any) -> Dict[str, Any]:
        if isinstance(metrics, dict):
            return metrics
        if hasattr(metrics, "iloc"):
            try:
                return metrics.iloc[0].to_dict()
            except Exception:
                pass
        if hasattr(metrics, "to_dict"):
            try:
                return metrics.to_dict()
            except Exception:
                pass
        return {"raw": str(metrics)}

    def _extract_f1(self, metrics: Dict[str, Any]) -> float:
        for key in ["f1", "f1_score", "f1-score"]:
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    pass
        precision = metrics.get("precision")
        recall = metrics.get("recall")
        try:
            precision = float(precision)
            recall = float(recall)
        except (TypeError, ValueError):
            return 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _split_gold_for_ml(
        self,
        gold: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Split labeled gold pairs into train/eval sets for ML matcher.
        Primary strategy is stratified split on label; if that fails, fall back to
        random split (non-stratified) as requested.
        """
        if "label" not in gold.columns:
            raise ValueError("ML matcher requires labeled pairs")
        if len(gold) < 2:
            raise ValueError("Not enough gold pairs to split for ML evaluation")

        # Try stratified split first.
        try:
            train_df, eval_df = train_test_split(
                gold,
                test_size=test_size,
                random_state=random_state,
                stratify=gold["label"],
            )
            method = "stratified"
        except Exception:
            # Fallback requested by user: random split without stratification.
            train_df, eval_df = train_test_split(
                gold,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )
            method = "random_fallback"

        train_df = train_df.reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)

        split_info = {
            "method": method,
            "train_size": len(train_df),
            "eval_size": len(eval_df),
            "train_pos": int((train_df["label"] == 1).sum()) if "label" in train_df.columns else None,
            "train_neg": int((train_df["label"] == 0).sum()) if "label" in train_df.columns else None,
            "eval_pos": int((eval_df["label"] == 1).sum()) if "label" in eval_df.columns else None,
            "eval_neg": int((eval_df["label"] == 0).sum()) if "label" in eval_df.columns else None,
        }
        return train_df, eval_df, split_info

    def _try_execute_matcher(
        self,
        name_left: str,
        name_right: str,
        candidates_input,
        id_column: str,
        choice: Dict[str, Any],
        gold: pd.DataFrame,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            stable_choice = self._stabilize_choice_and_data(name_left, name_right, choice)
            comparators = self._build_comparators(stable_choice["comparators"])
            matcher = RuleBasedMatcher()
            correspondences = matcher.match(
                df_left=self.datasets_loaded[name_left],
                df_right=self.datasets_loaded[name_right],
                candidates=candidates_input,
                comparators=comparators,
                weights=stable_choice["weights"],
                threshold=stable_choice["threshold"],
                id_column=id_column,
            )
            metrics = self.evaluator.evaluate_matching(
                correspondences=correspondences,
                test_pairs=gold,
            )
            metrics = self._normalize_metrics(metrics)
            result = {
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": self._extract_f1(metrics),
            }
            return result, None
        except Exception as e:
            tb = traceback.format_exc()
            return None, f"{type(e).__name__}: {str(e)}\n{tb}"

    def _try_execute_ml_matcher(
        self,
        name_left: str,
        name_right: str,
        candidates_input,
        id_column: str,
        choice: Dict[str, Any],
        gold_train: pd.DataFrame,
        gold_eval: pd.DataFrame,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if "label" not in gold_train.columns:
            return None, "ML matcher requires labeled pairs with both positive and negative examples"
        gold_train = gold_train.dropna(subset=["label", "id1", "id2"]).copy()
        gold_eval = gold_eval.dropna(subset=["label", "id1", "id2"]).copy()
        if gold_train["label"].nunique() < 2:
            return None, "ML matcher requires labeled pairs with both positive and negative examples"
        try:
            stable_choice = self._stabilize_choice_and_data(name_left, name_right, choice)
            print("    [ML] Building comparators")
            comparators = self._build_comparators(stable_choice["comparators"])
            print(f"    [ML] Comparators count: {len(comparators)}")
            feature_extractor = FeatureExtractor(comparators)
            print("    [ML] Creating features")
            left_df = self.datasets_loaded[name_left]
            right_df = self.datasets_loaded[name_right]
            if id_column not in left_df.columns or id_column not in right_df.columns:
                return None, f"ML matcher id column missing: {id_column}"
            left_ids = set(left_df[id_column].dropna().tolist())
            right_ids = set(right_df[id_column].dropna().tolist())
            before_train = len(gold_train)
            gold_train = gold_train[gold_train["id1"].isin(left_ids) & gold_train["id2"].isin(right_ids)].copy()
            gold_train = gold_train.reset_index(drop=True)
            print(f"    [ML] Gold train pairs filtered: {before_train} -> {len(gold_train)}")
            if gold_train.empty:
                return None, "ML matcher has no train gold pairs after ID filtering"
            before_eval = len(gold_eval)
            gold_eval = gold_eval[gold_eval["id1"].isin(left_ids) & gold_eval["id2"].isin(right_ids)].copy()
            gold_eval = gold_eval.reset_index(drop=True)
            print(f"    [ML] Gold eval pairs filtered: {before_eval} -> {len(gold_eval)}")
            if gold_eval.empty:
                return None, "ML matcher has no eval gold pairs after ID filtering"
            features = feature_extractor.create_features(
                left_df,
                right_df,
                gold_train[["id1", "id2"]],
                labels=gold_train["label"].reset_index(drop=True),
                id_column=id_column,
            )
            print(f"    [ML] Feature rows: {len(features)}")
            feature_cols = [c for c in features.columns if c not in ["id1", "id2", "label"]]
            if not feature_cols:
                return None, "ML matcher could not derive feature columns"
            print(f"    [ML] Feature columns: {len(feature_cols)}")
            X_train = features[feature_cols]
            y_train = features["label"]
            if y_train.nunique() < 2:
                return None, "ML matcher needs at least two label classes"
            print("    [ML] Selecting model")
            classifier, model_name, model_params = self._select_best_ml_model(X_train, y_train)
            print(f"    [ML] Found best model: {model_name} ({model_params})")
            print("    [ML] Training classifier")
            classifier.fit(X_train, y_train)
            ml_matcher = MLBasedMatcher(feature_extractor)
            print("    [ML] Matching candidates")
            correspondences = ml_matcher.match(
                self.datasets_loaded[name_left],
                self.datasets_loaded[name_right],
                candidates=candidates_input,
                id_column=id_column,
                trained_classifier=classifier,
            )
            print(f"    [ML] Correspondences: {len(correspondences)}")
            metrics = self.evaluator.evaluate_matching(
                correspondences=correspondences,
                test_pairs=gold_eval,
            )
            print("    [ML] Evaluation complete")
            metrics = self._normalize_metrics(metrics)
            result = {
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": self._extract_f1(metrics),
            }
            return result, None
        except Exception as e:
            tb = traceback.format_exc()
            return None, f"{type(e).__name__}: {str(e)}\n{tb}"

    def _select_best_ml_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        scorer = make_scorer(f1_score)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        candidates = [
            (
                "LogisticRegression",
                LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"),
                [{"C": 0.1}, {"C": 1.0}],
            ),
            (
                "RandomForestClassifier",
                RandomForestClassifier(random_state=42, class_weight="balanced"),
                [{"n_estimators": 50, "max_depth": None}, {"n_estimators": 100, "max_depth": 10}],
            ),
            (
                "SVC",
                SVC(random_state=42, probability=True, class_weight="balanced"),
                [{"C": 1.0, "kernel": "rbf"}, {"C": 1.0, "kernel": "linear"}],
            ),
        ]

        best_score = -1.0
        best_model = None
        best_name = ""
        best_params = {}

        for name, base_model, grid in candidates:
            for params in grid:
                model = clone(base_model)
                model.set_params(**params)
                try:
                    scores = cross_val_score(model, X_train, y_train, scoring=scorer, cv=cv, n_jobs=1)
                except Exception as exc:
                    print(f"    [ML] CV failed for {name} {params}: {exc}")
                    continue
                if scores is None or len(scores) == 0:
                    continue
                score = float(scores.mean())
                print(f"    [ML] CV {name} {params}: mean={score:.4f}, std={scores.std():.4f}")
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = name
                    best_params = params

        if best_model is None:
            best_model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
            best_name = "LogisticRegression"
            best_params = {}

        return best_model, best_name, best_params

    def run_pair_with_llm(self, name_left: str, name_right: str, id_column: str = None) -> Dict[str, Any]:
        pair_key = (name_left, name_right)
        reverse_key = (name_right, name_left)

        if pair_key in self.gold_standards:
            gold = self.gold_standards[pair_key]
        elif reverse_key in self.gold_standards:
            gold = self.gold_standards[reverse_key]
        else:
            raise ValueError(f"No matching gold standard found for pair: {pair_key}")

        print(f"\n{'='*50}\nMATCHING: {name_left} <-> {name_right}\n{'='*50}")
        skip_pair, gate_info = self._should_skip_pair_for_blocking_quality(name_left, name_right)
        if skip_pair:
            print(
                "⛔ Skipping matching due to blocking quality gate: "
                f"PC={gate_info.get('pair_completeness', 0.0):.4f} < {gate_info.get('threshold')}"
            )
            failure_tags = self._failure_tags_for_matching_event(
                skipped_due_to_blocking_gate=True,
                f1=0.0,
                target_f1=self.f1_threshold,
            )
            skipped = {
                "pair": f"{name_left}_{name_right}",
                "skipped_due_to_blocking_gate": True,
                "blocking_gate": gate_info,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "failure_tags": failure_tags,
            }
            self.results_history.append(skipped)
            self._trace_event(
                pair=f"{name_left}_{name_right}",
                decision="skip_pair",
                extra={
                    "stage": "matching_gate",
                    "blocking_gate": gate_info,
                    "failure_tags": failure_tags,
                },
            )
            return skipped

        if id_column is None:
            gold, id_column = self._align_ids_with_gold(name_left, name_right, gold)
        print(f"    Using ID column: '{id_column}'")
        alignment = getattr(self, "last_id_alignment", None)
        if alignment:
            if alignment.get("left_id") != alignment.get("right_id"):
                print(
                    f"    Using ID columns: left={alignment.get('left_id')}, "
                    f"right={alignment.get('right_id')} (alias {id_column})"
                )
            else:
                print(f"    Using ID column: '{alignment.get('left_id')}'")
            if alignment.get("gold_id1_source") == "right":
                print(
                    f"    Gold id1 -> right({alignment.get('right_id')}), "
                    f"id2 -> left({alignment.get('left_id')})"
                )
            else:
                print(
                    f"    Gold id1 -> left({alignment.get('left_id')}), "
                    f"id2 -> right({alignment.get('right_id')})"
                )

        analysis = self._analyze_columns_for_pair(name_left, name_right, id_column)
        blocking_cfg, _ = self._get_blocking_config(name_left, name_right)
        blocking_columns = blocking_cfg.get("columns", []) if blocking_cfg else []
        guardrails = self._build_matching_guardrails(analysis, blocking_columns)
        print(f"Blocking columns: {blocking_columns}")
        print(f"Guardrails preferred columns: {guardrails.get('preferred_columns', [])[:5]}")

        blocker = self._create_blocker_from_config(
            name_left, name_right, id_column, analysis["common_columns"], blocking_cfg
        )
        try:
            candidates_full = blocker.materialize()
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"⛔ Failed to materialize blocker candidates: {err}")
            failed = {
                "pair": f"{name_left}_{name_right}",
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "error": err,
                "blocking_strategy": blocking_cfg.get("strategy") if blocking_cfg else "fallback",
            }
            self.results_history.append(failed)
            return failed
        proxy_candidates = self._sample_candidate_pairs(candidates_full, self.proxy_max_candidates)
        print(
            f"Candidates: full={len(candidates_full):,}, proxy_sample={len(proxy_candidates):,} "
            f"(margin={self.proxy_f1_margin:.3f})"
        )

        matcher_mode = self.matcher_mode
        if matcher_mode == "auto":
            matcher_mode = "ml" if "label" in gold.columns and gold["label"].nunique() >= 2 else "rule_based"
        ml_split = None
        if matcher_mode == "ml":
            try:
                ml_train_gold, ml_eval_gold, split_info = self._split_gold_for_ml(gold)
                ml_split = {
                    "train": ml_train_gold,
                    "eval": ml_eval_gold,
                    "info": split_info,
                }
                print(
                    f"    [ML] Holdout split ({split_info['method']}): "
                    f"train={split_info['train_size']}, eval={split_info['eval_size']}"
                )
            except Exception as split_err:
                print(f"    [ML] Split setup failed: {split_err}. Falling back to RuleBasedMatcher.")
                matcher_mode = "rule_based"

        previous_attempts, best_result = [], None
        no_gain_streak = 0
        f1_history: List[float] = []

        for attempt in range(1, self.max_attempts + 1):
            print(f"\n--- Attempt {attempt}/{self.max_attempts} ---")
            choice = self._ask_llm_for_matcher(
                analysis,
                blocking_columns,
                matcher_mode,
                previous_attempts if attempt > 1 else None,
                guardrails=guardrails,
            )
            active_choice = self._prepare_active_choice(
                choice=choice,
                matcher_mode=matcher_mode,
                analysis=analysis,
                blocking_columns=blocking_columns,
            )
            if matcher_mode == "ml" and active_choice.get("comparators") != choice.get("comparators"):
                print(f"ML comparators filtered to: {[c['column'] for c in active_choice['comparators']]}")
            comparator_info = []
            for comp in active_choice["comparators"]:
                sim = comp.get("similarity_function") or comp.get("max_difference") or comp.get("max_days_difference")
                comparator_info.append(f"{comp.get('column')}:{sim}")
            print(f"Comparators: {comparator_info}")
            if matcher_mode != "ml":
                print(f"Weights: {choice['weights']}")
                print(f"Threshold: {choice['threshold']}")
            print(f"    Reasoning: {choice['reasoning']}")
            print(f"    Reasoning (100c): {choice['reasoning'][:100]}")

            last_error = None
            attempt_result: Optional[Dict[str, Any]] = None
            for error_retry in range(self.max_error_retries + 1):
                if error_retry > 0:
                    print(f"    Retry {error_retry}: fixing error...")
                    choice = self._ask_llm_for_matcher(
                        analysis,
                        blocking_columns,
                        matcher_mode,
                        last_error=last_error,
                        guardrails=guardrails,
                    )
                    active_choice = self._prepare_active_choice(
                        choice=choice,
                        matcher_mode=matcher_mode,
                        analysis=analysis,
                        blocking_columns=blocking_columns,
                    )

                proxy_metrics = None
                proxy_error = None
                proxy_f1 = None
                run_proxy = (len(candidates_full) > self.proxy_max_candidates) or (best_result is not None)
                if run_proxy:
                    if matcher_mode == "ml":
                        proxy_metrics, proxy_error = self._try_execute_ml_matcher(
                            name_left,
                            name_right,
                            proxy_candidates,
                            id_column,
                            active_choice,
                            ml_split["train"],
                            ml_split["eval"],
                        )
                    else:
                        proxy_metrics, proxy_error = self._try_execute_matcher(
                            name_left, name_right, proxy_candidates, id_column, active_choice, gold
                        )
                    if proxy_error:
                        error = proxy_error
                    else:
                        proxy_f1 = float((proxy_metrics or {}).get("f1", 0.0) or 0.0)
                        required = (
                            float(best_result.get("f1", 0.0) or 0.0) + self.proxy_f1_margin
                            if best_result is not None
                            else PROXY_MIN_F1_ATTEMPT1
                        )
                        print(
                            f"    Proxy stage: F1={proxy_f1:.4f}, required>={required:.4f} "
                            f"-> {'run full' if proxy_f1 >= required else 'skip full'}"
                        )
                        if proxy_f1 < required:
                            attempt_result = {
                                "pair": f"{name_left}_{name_right}",
                                "comparators": active_choice["comparators"],
                                "weights": choice["weights"],
                                "threshold": choice["threshold"],
                                "reasoning": choice["reasoning"],
                                "matcher_type": matcher_mode,
                                "blocking_strategy": blocking_cfg.get("strategy") if blocking_cfg else "fallback",
                                "blocking_columns": blocking_columns,
                                "attempt": attempt,
                                "error_retries": error_retry,
                                "precision": proxy_metrics.get("precision"),
                                "recall": proxy_metrics.get("recall"),
                                "f1": proxy_f1,
                                "proxy_only": True,
                                "proxy_sample_size": len(proxy_candidates),
                                "full_candidate_size": len(candidates_full),
                                "proxy_margin_required": self.proxy_f1_margin,
                                "failure_tags": self._failure_tags_for_matching_event(
                                    proxy_rejected=True,
                                    f1=proxy_f1,
                                    target_f1=self.f1_threshold,
                                    fallback_blocking=(blocking_cfg is None),
                                ),
                            }
                            self._trace_event(
                                pair=f"{name_left}_{name_right}",
                                attempt=attempt,
                                decision="skip_full_by_proxy_gate",
                                extra={
                                    "mode": matcher_mode,
                                    "proxy_f1": proxy_f1,
                                    "required_f1": required,
                                    "choice": {
                                        "comparators": active_choice["comparators"],
                                        "weights": choice["weights"],
                                        "threshold": choice["threshold"],
                                        "reasoning": choice.get("reasoning", ""),
                                    },
                                    "failure_tags": attempt_result["failure_tags"],
                                },
                            )
                            no_gain_streak += 1
                            previous_attempts.append(
                                {
                                    "columns": [c["column"] for c in choice["comparators"]],
                                    "f1": proxy_f1,
                                    "proxy_only": True,
                                }
                            )
                            break
                else:
                    proxy_f1 = None

                if attempt_result is not None:
                    break

                if matcher_mode == "ml":
                    metrics, error = self._try_execute_ml_matcher(
                        name_left,
                        name_right,
                        candidates_full,
                        id_column,
                        active_choice,
                        ml_split["train"],
                        ml_split["eval"],
                    )
                    if error:
                        print(f"    ML error: {error[:100]}... Falling back to RuleBasedMatcher.")
                        matcher_mode = "rule_based"
                        error = None
                        continue
                else:
                    metrics, error = self._try_execute_matcher(
                        name_left, name_right, candidates_full, id_column, active_choice, gold
                    )
                if error:
                    first_line = error.splitlines()[0] if isinstance(error, str) else str(error)
                    print(f"    ERROR: {first_line[:220]}...")
                    if self.verbose and isinstance(error, str) and "\n" in error:
                        tb_path = os.path.join(self.output_dir, "matching_errors.log")
                        with open(tb_path, "a", encoding="utf-8") as fh:
                            fh.write(
                                f"\n=== {name_left}_{name_right} attempt={attempt} retry={error_retry} ===\n"
                            )
                            fh.write(error)
                            fh.write("\n")
                        print(f"    Full traceback appended to: {tb_path}")
                    last_error = error
                    if error_retry >= self.max_error_retries:
                        previous_attempts.append(
                            {
                                "columns": [c["column"] for c in choice["comparators"]],
                                "f1": 0.0,
                                "error": error[:200],
                                "failure_tags": self._failure_tags_for_matching_event(
                                    error=error[:200],
                                    f1=0.0,
                                    target_f1=self.f1_threshold,
                                    fallback_blocking=(blocking_cfg is None),
                                ),
                            }
                        )
                        break
                    continue

                f1 = metrics.get("f1", 0.0)
                result = {
                    "pair": f"{name_left}_{name_right}",
                    "comparators": active_choice["comparators"],
                    "weights": choice["weights"],
                    "threshold": choice["threshold"],
                    "reasoning": choice["reasoning"],
                    "matcher_type": matcher_mode,
                    "blocking_strategy": blocking_cfg.get("strategy") if blocking_cfg else "fallback",
                    "blocking_columns": blocking_columns,
                    "attempt": attempt,
                    "error_retries": error_retry,
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": f1,
                    "proxy_f1": proxy_f1,
                    "proxy_sample_size": len(proxy_candidates),
                    "full_candidate_size": len(candidates_full),
                    "failure_tags": self._failure_tags_for_matching_event(
                        f1=f1,
                        target_f1=self.f1_threshold,
                        fallback_blocking=(blocking_cfg is None),
                    ),
                }
                if ml_split and result["matcher_type"] == "ml":
                    result["ml_split_method"] = ml_split["info"]["method"]
                    result["ml_train_size"] = ml_split["info"]["train_size"]
                    result["ml_eval_size"] = ml_split["info"]["eval_size"]
                self._trace_event(
                    pair=f"{name_left}_{name_right}",
                    attempt=attempt,
                    decision="full_evaluation",
                    extra={
                        "mode": matcher_mode,
                        "proxy_f1": proxy_f1,
                        "result_f1": f1,
                        "threshold": self.f1_threshold,
                        "choice": {
                            "comparators": active_choice["comparators"],
                            "weights": choice["weights"],
                            "threshold": choice["threshold"],
                            "reasoning": choice.get("reasoning", ""),
                        },
                    },
                )

                if best_result is None or f1 > (float(best_result.get("f1", 0.0) or 0.0) + self.no_gain_epsilon):
                    best_result = result
                    no_gain_streak = 0
                else:
                    no_gain_streak += 1
                f1_history.append(float(f1))

                if f1 >= self.f1_threshold:
                    print(f"SUCCESS: F1={f1:.4f}")
                    self.results_history.append(result)
                    return result

                print(f"F1={f1:.4f} < {self.f1_threshold}")
                previous_attempts.append(
                    {
                        "columns": [c["column"] for c in choice["comparators"]],
                        "f1": f1,
                        "failure_tags": result.get("failure_tags", []),
                    }
                )
                break

            if attempt_result is not None and attempt_result.get("proxy_only"):
                f1_history.append(float(attempt_result.get("f1", 0.0) or 0.0))
                if no_gain_streak >= self.no_gain_patience:
                    print("⏹️ Early stop: repeated no-gain proxy outcomes")
                    break
                if self._emit_confidence_stop_if_needed(
                    pair=f"{name_left}_{name_right}",
                    attempt=attempt,
                    f1_history=f1_history,
                    best_result=best_result,
                ):
                    break
                continue
            if no_gain_streak >= self.no_gain_patience:
                print("⏹️ Early stop: repeated no-gain matching attempts")
                break
            if self._emit_confidence_stop_if_needed(
                pair=f"{name_left}_{name_right}",
                attempt=attempt,
                f1_history=f1_history,
                best_result=best_result,
            ):
                break

        print(f"Max attempts reached. Best F1={best_result.get('f1', 0) if best_result else 0:.4f}")
        if best_result:
            self.results_history.append(best_result)
            return best_result
        fallback = {
            "pair": f"{name_left}_{name_right}",
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "blocking_strategy": blocking_cfg.get("strategy") if blocking_cfg else "fallback",
            "blocking_columns": blocking_columns,
            "failure_tags": self._failure_tags_for_matching_event(
                f1=0.0,
                target_f1=self.f1_threshold,
                fallback_blocking=(blocking_cfg is None),
            ),
        }
        self.results_history.append(fallback)
        return fallback

    def run_all(self, id_column: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        print(f"\n{'='*50}\nMATCHING EVALUATION\n{'='*50}")

        all_results = []
        dataset_names = sorted(str(name) for name in self.datasets_loaded.keys())
        discovered_config = {
            "id_columns": dict(self.id_columns),
            "matching_strategies": {},
            "dataset_signature": "|".join(dataset_names),
            "dataset_names": dataset_names,
            "min_blocking_pc_for_matching": self.min_blocking_pc_for_matching,
            "proxy_f1_margin": self.proxy_f1_margin,
            "proxy_max_candidates": self.proxy_max_candidates,
            "llm_trace_path": os.path.join(self.output_dir, "llm_matching_trace.jsonl"),
        }

        for pair in self.gold_standards.keys():
            name_left, name_right = pair
            result = self.run_pair_with_llm(name_left, name_right, id_column)
            if result:
                all_results.append(result)
                discovered_config["matching_strategies"][f"{name_left}_{name_right}"] = {
                    "comparators": result.get("comparators"),
                    "weights": result.get("weights"),
                    "threshold": result.get("threshold"),
                    "f1": result.get("f1", 0),
                    "skipped_due_to_blocking_gate": bool(result.get("skipped_due_to_blocking_gate", False)),
                    "proxy_f1": result.get("proxy_f1"),
                    "failure_tags": result.get("failure_tags", []),
                }

        df = pd.DataFrame(all_results)

        if not df.empty:
            df.to_csv(os.path.join(self.output_dir, "matching_tester_results.csv"), index=False)
            print(f"\n📊 Results: {len(df)} pairs evaluated")
            for _, row in df.iterrows():
                if bool(row.get("skipped_due_to_blocking_gate", False)):
                    status = "SKIP"
                    print(f"  {status} {row['pair']}: blocked by low blocking PC")
                    continue
                status = "OK" if row.get("f1", 0) >= self.f1_threshold else "WARN"
                proxy_txt = (
                    f", proxy={float(row.get('proxy_f1')):.4f}"
                    if pd.notna(row.get("proxy_f1"))
                    else ""
                )
                print(f"  {status} {row['pair']}: F1={row.get('f1', 0):.4f}{proxy_txt}")

        dataset_sizes = {name: len(df) for name, df in self.datasets_loaded.items()}
        matching_estimate = estimate_from_matching(
            dataset_sizes=dataset_sizes,
            blocking_strategies=self.blocking_strategies,
            matching_strategies=discovered_config.get("matching_strategies", {}),
        )
        if matching_estimate:
            discovered_config["fusion_size_estimate"] = matching_estimate
            estimate_path = estimate_path_for_output_dir(self.output_dir)
            upsert_stage_estimate(estimate_path, "matching", matching_estimate)
            print(
                "[*] Matching fused-size estimate: "
                f"rows={matching_estimate['expected_rows']}, "
                f"unique_ids={matching_estimate['expected_unique_ids']}"
            )
            if "expected_rows_matched_only" in matching_estimate:
                print(
                    "[*] Matching estimate breakdown: "
                    f"matched_only_rows={matching_estimate.get('expected_rows_matched_only')}, "
                    f"singleton_aware_rows={matching_estimate.get('expected_rows_singleton_aware')}"
                )
            print(f"[*] Fused-size report updated: {estimate_path}")

        with open(os.path.join(self.output_dir, "matching_config.json"), "w") as f:
            json.dump(discovered_config, f, indent=2)
        print("Config saved to matching_config.json")

        try:
            from helpers.run_report import write_agent_run_report

            output_root = os.path.dirname(self.output_dir.rstrip("/"))
            report_path = write_agent_run_report(output_root=output_root)
            print(f"[*] Run report updated: {report_path}")
            discovered_config["run_report_path"] = report_path
        except Exception as report_err:
            print(f"[WARN] Could not generate run report: {report_err}")

        return df, discovered_config
