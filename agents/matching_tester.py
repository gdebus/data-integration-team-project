import json
import os
import re
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC


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
  - numeric: NumericComparator(column, max_difference)
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
        max_attempts: int = 4,
        max_error_retries: int = 2,
        verbose: bool = True,
        matcher_mode: str = "ml",
        previous_failures: Optional[List[Dict[str, Any]]] = None,
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

        self.blocking_config = blocking_config or {}
        self.blocking_strategies = self.blocking_config.get("blocking_strategies", {})
        self.id_columns = self.blocking_config.get("id_columns", {})

        os.makedirs(self.output_dir, exist_ok=True)

        self.datasets_loaded: Dict[str, pd.DataFrame] = {}
        for path in datasets:
            name = os.path.splitext(os.path.basename(path))[0]
            if self.verbose:
                print(f"[*] Loading dataset: {name}")
            self.datasets_loaded[name] = load_dataset(path)

        self.gold_standards: Dict[Tuple[str, str], pd.DataFrame] = {}
        for pair, path in matching_testsets.items():
            if self.verbose:
                print(f"[*] Loading matching gold standard for {pair[0]} <-> {pair[1]}")
            self.gold_standards[pair] = self._load_gold_standard(path)

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
                raw = response.content if hasattr(response, "content") else str(response)
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
        if any(isinstance(v, (list, tuple, set)) for v in sample):
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
        for col in common_cols:
            left_col, right_col = df_left[col], df_right[col]
            col_type = self._infer_column_type(left_col, col)
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
                "samples": sample_left[:1] + sample_right[:1],
            }

        return {
            "left_dataset": name_left,
            "right_dataset": name_right,
            "common_columns": list(common_cols),
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
        response_text: str,
        valid_columns: List[str],
        column_types: Dict[str, str],
        blocking_columns: List[str],
    ) -> Dict[str, Any]:
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
        cleaned_comps = []

        for comp in raw_comps:
            if not isinstance(comp, dict):
                continue
            column = comp.get("column")
            if column not in valid_columns:
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
    ) -> Dict[str, Any]:
        if self.llm is None:
            return self._default_choice(
                analysis["common_columns"], analysis["column_types"], blocking_columns
            )

        system_prompt = f"""Select the best RuleBasedMatcher configuration based on the provided schema/profile context. Respond with ONLY valid JSON:
{{
  "comparators": [
    {{"type": "string", "column": "col1", "similarity_function": "cosine", "preprocess": "lower", "list_strategy": "concatenate"}},
    {{"type": "numeric", "column": "col2", "max_difference": 5}},
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
- list_strategy options: concatenate, set_jaccard, best_match, set_overlap.
- If blocking columns are provided, include at least one comparator using them unless they are clearly low quality.
- Weights should be non-negative, sum to 1, and match the number of comparators.
- Set threshold to balance precision/recall based on data quality (higher for strong identifiers, lower for noisy text).
- If prior attempts fail or error, fall back to a simpler RuleBasedMatcher config.
- If matcher_mode is ml, prefer fewer informative columns; more columns does not always improve results.
"""

        human_content = (
            f"Dataset pair: {analysis['left_dataset']} <-> {analysis['right_dataset']}\n"
            f"Blocking columns: {blocking_columns}\n"
            f"Matcher mode: {matcher_mode}\n"
            f"Common columns: {analysis['common_columns']}\n"
            f"Column details: {json.dumps(analysis['column_details'], indent=2)}"
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
        response_text = response.content if hasattr(response, "content") else str(response)
        return self._parse_llm_response(
            response_text, analysis["common_columns"], analysis["column_types"], blocking_columns
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
                comparators.append(
                    NumericComparator(column=column, max_difference=comp.get("max_difference", 5))
                )
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

    def _try_execute_matcher(
        self,
        name_left: str,
        name_right: str,
        blocker,
        id_column: str,
        choice: Dict[str, Any],
        gold: pd.DataFrame,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            comparators = self._build_comparators(choice["comparators"])
            matcher = RuleBasedMatcher()
            correspondences = matcher.match(
                df_left=self.datasets_loaded[name_left],
                df_right=self.datasets_loaded[name_right],
                candidates=blocker,
                comparators=comparators,
                weights=choice["weights"],
                threshold=choice["threshold"],
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
            return None, f"{type(e).__name__}: {str(e)}"

    def _try_execute_ml_matcher(
        self,
        name_left: str,
        name_right: str,
        blocker,
        id_column: str,
        choice: Dict[str, Any],
        gold: pd.DataFrame,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if "label" not in gold.columns:
            return None, "ML matcher requires labeled pairs with both positive and negative examples"
        gold = gold.dropna(subset=["label", "id1", "id2"]).copy()
        if gold["label"].nunique() < 2:
            return None, "ML matcher requires labeled pairs with both positive and negative examples"
        try:
            print("    [ML] Building comparators")
            comparators = self._build_comparators(choice["comparators"])
            print(f"    [ML] Comparators count: {len(comparators)}")
            feature_extractor = FeatureExtractor(comparators)
            print("    [ML] Creating features")
            left_df = self.datasets_loaded[name_left]
            right_df = self.datasets_loaded[name_right]
            if id_column not in left_df.columns or id_column not in right_df.columns:
                return None, f"ML matcher id column missing: {id_column}"
            left_ids = set(left_df[id_column].dropna().tolist())
            right_ids = set(right_df[id_column].dropna().tolist())
            before_count = len(gold)
            gold = gold[gold["id1"].isin(left_ids) & gold["id2"].isin(right_ids)].copy()
            gold = gold.reset_index(drop=True)
            print(f"    [ML] Gold pairs filtered: {before_count} -> {len(gold)}")
            if gold.empty:
                return None, "ML matcher has no gold pairs after ID filtering"
            features = feature_extractor.create_features(
                left_df,
                right_df,
                gold[["id1", "id2"]],
                labels=gold["label"].reset_index(drop=True),
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
                candidates=blocker,
                id_column=id_column,
                trained_classifier=classifier,
            )
            print(f"    [ML] Correspondences: {len(correspondences)}")
            metrics = self.evaluator.evaluate_matching(
                correspondences=correspondences,
                test_pairs=gold,
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
            return None, f"{type(e).__name__}: {str(e)}"

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
        print(f"Blocking columns: {blocking_columns}")

        blocker = self._create_blocker_from_config(
            name_left, name_right, id_column, analysis["common_columns"], blocking_cfg
        )

        matcher_mode = self.matcher_mode
        if matcher_mode == "auto":
            matcher_mode = "ml" if "label" in gold.columns and gold["label"].nunique() >= 2 else "rule_based"

        previous_attempts, best_result = [], None

        for attempt in range(1, self.max_attempts + 1):
            print(f"\n--- Attempt {attempt}/{self.max_attempts} ---")
            choice = self._ask_llm_for_matcher(
                analysis,
                blocking_columns,
                matcher_mode,
                previous_attempts if attempt > 1 else None,
            )
            active_choice = choice
            if matcher_mode == "ml":
                active_choice = dict(choice)
                active_choice["comparators"] = self._filter_ml_comparators(
                    choice["comparators"],
                    analysis["common_columns"],
                    analysis["column_types"],
                    blocking_columns,
                )
                if active_choice["comparators"] != choice["comparators"]:
                    print(
                        f"ML comparators filtered to: {[c['column'] for c in active_choice['comparators']]}"
                    )
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
            for error_retry in range(self.max_error_retries + 1):
                if error_retry > 0:
                    print(f"    Retry {error_retry}: fixing error...")
                    choice = self._ask_llm_for_matcher(
                        analysis,
                        blocking_columns,
                        matcher_mode,
                        last_error=last_error,
                    )

                if matcher_mode == "ml":
                    metrics, error = self._try_execute_ml_matcher(
                        name_left, name_right, blocker, id_column, active_choice, gold
                    )
                    if error:
                        print(f"    ML error: {error[:100]}... Falling back to RuleBasedMatcher.")
                        matcher_mode = "rule_based"
                        error = None
                        continue
                else:
                    metrics, error = self._try_execute_matcher(
                        name_left, name_right, blocker, id_column, active_choice, gold
                    )
                if error:
                    print(f"    ERROR: {error[:100]}...")
                    last_error = error
                    if error_retry >= self.max_error_retries:
                        previous_attempts.append(
                            {
                                "columns": [c["column"] for c in choice["comparators"]],
                                "f1": 0.0,
                                "error": error[:200],
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
                }

                if best_result is None or f1 > best_result.get("f1", 0):
                    best_result = result

                if f1 >= self.f1_threshold:
                    print(f"SUCCESS: F1={f1:.4f}")
                    self.results_history.append(result)
                    return result

                print(f"F1={f1:.4f} < {self.f1_threshold}")
                previous_attempts.append(
                    {
                        "columns": [c["column"] for c in choice["comparators"]],
                        "f1": f1,
                    }
                )
                break

        print(f"Max attempts reached. Best F1={best_result.get('f1', 0) if best_result else 0:.4f}")
        if best_result:
            self.results_history.append(best_result)
        return best_result

    def run_all(self, id_column: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        print(f"\n{'='*50}\nMATCHING EVALUATION\n{'='*50}")

        all_results = []
        discovered_config = {"id_columns": dict(self.id_columns), "matching_strategies": {}}

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
                }

        df = pd.DataFrame(all_results)

        if not df.empty:
            df.to_csv(os.path.join(self.output_dir, "matching_tester_results.csv"), index=False)
            print(f"\n📊 Results: {len(df)} pairs evaluated")
            for _, row in df.iterrows():
                status = "OK" if row.get("f1", 0) >= self.f1_threshold else "WARN"
                print(f"  {status} {row['pair']}: F1={row.get('f1', 0):.4f}")

        with open(os.path.join(self.output_dir, "matching_config.json"), "w") as f:
            json.dump(discovered_config, f, indent=2)
        print("Config saved to matching_config.json")

        return df, discovered_config
