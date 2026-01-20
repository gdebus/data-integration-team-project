import json
import math
import os
import re
import statistics
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

from PyDI.io import load_xml, load_parquet, load_csv
from langchain_core.messages import SystemMessage, HumanMessage
from PyDI.entitymatching import (
    StandardBlocker,
    EmbeddingBlocker,
    TokenBlocker,
    SortedNeighbourhoodBlocker,
    EntityMatchingEvaluator,
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


import pandas as pd
import re
import traceback
from typing import Dict, List, Tuple, Any, Optional
from PyDI.entitymatching import StandardBlocker, EmbeddingBlocker, TokenBlocker, SortedNeighbourhoodBlocker, EntityMatchingEvaluator

# Strategy descriptions with computational guidance for LLM
STRATEGY_DESCRIPTIONS = """
Available blocking strategies (ranked by computational cost):

1. **exact_match_single** - Block on exact match of ONE column
   - Speed: ⚡ FASTEST
   - Parameters: columns (1 column only)
   - Best for: Highly standardized values (IDs, codes, normalized names)
   - Candidate pairs: Low (only exact matches)

2. **exact_match_multi** - Block on exact match of MULTIPLE columns combined
   - Speed: ⚡ FAST
   - Parameters: columns (2+ columns)
   - Best for: When single column has too many collisions
   - Candidate pairs: Very low (more restrictive)

3. **sorted_neighbourhood** - Sliding window over sorted records
   - Speed: ⚡⚡ FAST-MEDIUM (O(n log n) sorting + linear scan)
   - Parameters: columns (1 column = sort key), window (INTEGER 5-50, default 15)
   - Best for: Data with natural ordering (dates, years, alphabetically sortable names)
   - WINDOW SIZE HEURISTIC: Use window ≈ max(10, min(50, sqrt(N_left + N_right) / 10))
   - For ~20K total rows: window=10-20 is reasonable
   - For ~100K total rows: window=20-35 is reasonable
   - Candidate pairs: Approximately (N_left + N_right) × window / 2
   - WARNING: Only effective if sort key has meaningful ordering for matching entities

4. **token_blocking** - Block on shared word tokens
   - Speed: ⚡⚡ MEDIUM
   - Parameters: columns (1 column), min_token_len (INTEGER 3-10, default 4)
   - Best for: Text fields with meaningful words (titles, names)
   - WARNING: min_token_len=2 creates HUGE blocks (common words like "the", "a", "of")
   - RECOMMENDATION: Use min_token_len >= 4 for large datasets (>10K rows)
   - Candidate pairs: Can be HIGH if min_token_len too small

5. **ngram_blocking** - Block on character n-grams (substrings)
   - Speed: ⚡⚡⚡ MEDIUM-SLOW
   - Parameters: columns (1 column), ngram_size (INTEGER 3-6, default 4)
   - Best for: Fuzzy matching, typos, abbreviations, non-English text
   - WARNING: Small ngram_size (2-3) creates MASSIVE candidate sets
   - RECOMMENDATION: Use ngram_size >= 4 for datasets > 5K rows
   - Candidate pairs: Can be VERY HIGH if ngram_size too small

6. **semantic_similarity** - Embedding-based similarity blocking
   - Speed: ⚡⚡⚡⚡ SLOWEST (requires embedding computation)
   - Parameters: columns (1+ columns), top_k (INTEGER 10-100, default 20)
   - Best for: Semantic variations, different wording for same entity
   - RECOMMENDATION: Use multiple informative columns (e.g., title + author + year) to improve PC
   - Candidate pairs: Controlled by top_k (predictable: left_size × top_k)

COMPUTATIONAL GUIDELINES:
- For datasets with N_left × N_right > 100M potential pairs: Use exact_match or token_blocking with min_token_len >= 5
- For datasets with N_left × N_right > 10M potential pairs: Avoid ngram_size < 4 or min_token_len < 4
- Target candidate pairs: Ideally < 2M for reasonable runtime
- Pair Completeness (PC) goal: >= 0.80 (find 80% of true matches)
"""

class BlockingTester:
    """LLM-driven blocking tester that evaluates strategies with full parameter control."""
    
    def __init__(
        self, 
        llm,
        datasets: List[str],
        blocking_testsets: Dict[Tuple[str, str], str],
        output_dir: str = "output/blocking-evaluation",
        pc_threshold: float = 0.95,
        max_attempts: int = 5,
        max_error_retries: int = 2,
        max_candidates: int = 2_000_000,
        candidate_tolerance: float = 0.15,
        verbose: bool = True,
        previous_failures: Optional[List[Dict[str, Any]]] = None
    ):
        self.llm = llm
        self.output_dir = output_dir
        self.pc_threshold = pc_threshold
        self.max_attempts = max_attempts
        self.max_error_retries = max_error_retries
        self.max_candidates = max_candidates
        self.candidate_tolerance = candidate_tolerance
        self.verbose = verbose
        self.evaluator = EntityMatchingEvaluator()
        self.results_history: List[Dict[str, Any]] = []
        self.previous_failures = previous_failures or []
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load datasets
        self.datasets_loaded: Dict[str, pd.DataFrame] = {}
        for path in datasets:
            name = os.path.splitext(os.path.basename(path))[0]
            if self.verbose:
                print(f"[*] Loading dataset: {name}")
            self.datasets_loaded[name] = load_dataset(path)
        
        # Load gold standards
        self.gold_standards: Dict[Tuple[str, str], pd.DataFrame] = {}
        for pair, path in blocking_testsets.items():
            if self.verbose:
                print(f"[*] Loading gold standard for {pair[0]} <-> {pair[1]}")
            self.gold_standards[pair] = self._load_gold_standard(path)

        self._set_adaptive_max_candidates()

    def _set_adaptive_max_candidates(self) -> None:
        """Set max_candidates using median-scaled pair size heuristic."""
        r = 0.0035
        pair_sizes = []
        for name_left, name_right in self.gold_standards.keys():
            if name_left not in self.datasets_loaded or name_right not in self.datasets_loaded:
                continue
            left_size = len(self.datasets_loaded[name_left])
            right_size = len(self.datasets_loaded[name_right])
            pair_sizes.append(left_size * right_size)

        if not pair_sizes:
            return

        median_pairs = statistics.median(pair_sizes)
        base = math.ceil(median_pairs * r)
        scaled = [math.ceil(base * math.sqrt(size / median_pairs)) for size in pair_sizes]
        adaptive_max = max(scaled)

        if self.verbose:
            print(
                f"[*] Adaptive max_candidates set to {adaptive_max:,} "
                f"(r={r:.2%}, median_pairs={int(median_pairs):,})"
            )

        self.max_candidates = adaptive_max
    
    def _load_gold_standard(self, path: str) -> pd.DataFrame:
        """Load and prepare gold standard CSV."""
        gs = pd.read_csv(path)
        col_mapping = {}
        for col in gs.columns:
            col_lower = col.lower()
            if 'id_a' in col_lower or col_lower == 'id1':
                col_mapping[col] = 'id1'
            elif 'id_b' in col_lower or col_lower == 'id2':
                col_mapping[col] = 'id2'
            elif 'label' in col_lower:
                col_mapping[col] = 'label'
        gs = gs.rename(columns=col_mapping)
        if 'label' in gs.columns:
            gs = gs[gs['label'] == 1.0].copy()
        if self.verbose:
            print(f"    Loaded {len(gs)} ground truth pairs")
        return gs[['id1', 'id2']]
    
    def _detect_id_column(self, df: pd.DataFrame) -> str:
        """Simple heuristic to detect ID column."""
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['id', 'record_id', 'row_id', 'index', 'key']:
                return col
            if 'id' in col_lower:
                try:
                    if df[col].nunique() == len(df):
                        return col
                except TypeError:
                    pass
        return df.columns[0]
    
    def _analyze_columns_for_pair(self, name_left: str, name_right: str, id_column: str = None) -> Dict[str, Any]:
        """Analyze columns for a dataset pair with computational estimates."""
        df_left = self.datasets_loaded[name_left]
        df_right = self.datasets_loaded[name_right]
        
        left_size = len(df_left)
        right_size = len(df_right)
        potential_pairs = left_size * right_size
        
        common_cols = set(df_left.columns) & set(df_right.columns)
        if id_column:
            common_cols.discard(id_column)
        common_cols.discard('id')
        
        column_details = {}
        for col in common_cols:
            left_col, right_col = df_left[col], df_right[col]
            try:
                sample_left = [str(x)[:50] for x in left_col.dropna().head(5).tolist()]
                sample_right = [str(x)[:50] for x in right_col.dropna().head(5).tolist()]
                avg_tokens = 0
                if left_col.dtype == 'object':
                    avg_tokens = left_col.dropna().head(100).apply(lambda x: len(str(x).split())).mean()
            except:
                sample_left, sample_right = [], []
                avg_tokens = 0
            
            try:
                unique_left = int(left_col.nunique())
            except TypeError:
                unique_left = -1
            try:
                unique_right = int(right_col.nunique())
            except TypeError:
                unique_right = -1
            
            column_details[col] = {
                "dtype": str(left_col.dtype),
                "null_pct": f"{left_col.isnull().mean()*100:.0f}%/{right_col.isnull().mean()*100:.0f}%",
                "unique_left": unique_left,
                "unique_right": unique_right,
                "avg_tokens": round(avg_tokens, 1),
                "samples": {
                    "left": sample_left,
                    "right": sample_right
                }
            }
        
        return {
            "left_dataset": name_left,
            "right_dataset": name_right,
            "left_size": left_size,
            "right_size": right_size,
            "potential_pairs": potential_pairs,
            "common_columns": list(common_cols),
            "column_details": column_details
        }
    
    def _ask_llm_for_strategy(self, analysis: Dict[str, Any], previous_attempts: List[Dict] = None, last_error: str = None) -> Dict[str, Any]:
        """Ask LLM to select a blocking strategy with full parameter control."""
        
        system_prompt = f"""You are a blocking strategy optimizer for entity matching.

{STRATEGY_DESCRIPTIONS}

Given dataset characteristics, select the BEST blocking strategy.

RESPOND WITH ONLY A JSON OBJECT:
{{
    "strategy": "one of: exact_match_single, exact_match_multi, sorted_neighbourhood, token_blocking, ngram_blocking, semantic_similarity",
    "columns": ["column_name"],
    "min_token_len": 4,
    "ngram_size": 4,
    "window": 15,
    "top_k": 20,
    "reasoning": "brief explanation of choice and parameter values"
}}

RULES:
- strategy MUST be one of: exact_match_single, exact_match_multi, sorted_neighbourhood, token_blocking, ngram_blocking, semantic_similarity
- columns MUST exist in common_columns
- For exact_match_single: use exactly 1 column
- For exact_match_multi: use 2+ columns
- For sorted_neighbourhood: use 1 column with natural ordering (dates, years, sortable names)
- For token_blocking/ngram_blocking: only first column is used
- If not used use semantic similarity as for your last attempt
- Use semantic_similarity with top_k equal to 20 by default.
- If candidate pairs are too many, reduce top_k to 15 or 10.
- For semantic_similarity: prefer multiple informative columns when available (not just a single title/name)
- STRICT: If PC threshold is not met after 4 attempts, use a final semantic_similarity attempt on 2–3 most informative columns.
- min_token_len: INTEGER between 3-10 (only for token_blocking)
- ngram_size: INTEGER between 3-6 (only for ngram_blocking)
- window: INTEGER between 5-50 (only for sorted_neighbourhood)
- top_k: INTEGER between 10-100 (only for semantic_similarity)"""

        human_content = f"""Dataset pair: {analysis['left_dataset']} <-> {analysis['right_dataset']}

DATASET SIZES (IMPORTANT FOR COMPUTATIONAL DECISIONS):
- Left dataset: {analysis['left_size']:,} rows
- Right dataset: {analysis['right_size']:,} rows  
- Potential pairs (N×M): {analysis['potential_pairs']:,}
- Max allowed candidates: {self.max_candidates:,}

Common columns: {analysis['common_columns']}

Column details:
{json.dumps(analysis['column_details'], indent=2)}"""

        failure_summary = self._format_previous_failures(analysis['left_dataset'], analysis['right_dataset'])
        if failure_summary:
            human_content += f"""

PREVIOUS FAILED STRATEGIES (avoid repeating these):
{failure_summary}"""
        
        if last_error:
            human_content += f"""

⚠️ PREVIOUS STRATEGY FAILED WITH ERROR:
{last_error[:500]}

Please fix the strategy - ensure correct parameter names and valid column references."""
        
        elif previous_attempts:
            attempts_summary = []
            for a in previous_attempts:
                status = "❌ ERROR" if a.get('error') else f"PC={a.get('pair_completeness',0):.3f}"
                candidates = a.get('num_candidates', 'N/A')
                attempts_summary.append(
                    f"  - {a['strategy']}(cols={a['columns']}, params={a.get('params',{})}) → {status}, candidates={candidates}"
                )
            
            human_content += f"""

PREVIOUS ATTEMPTS (all failed to meet PC >= {self.pc_threshold} or had errors):
{chr(10).join(attempts_summary)}

IMPORTANT: Try a DIFFERENT strategy or significantly different parameters.
- If token_blocking failed with too many candidates, increase min_token_len
- If PC was too low, try a less restrictive strategy or different column
- Consider semantic_similarity for fuzzy matching if exact strategies fail"""
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ])
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return self._parse_llm_response(response_text, analysis['common_columns'])

    def _format_previous_failures(self, name_left: str, name_right: str) -> str:
        if not self.previous_failures:
            return ""
        pair_key = f"{name_left}_{name_right}"
        relevant = [f for f in self.previous_failures if f.get("pair") == pair_key]
        if not relevant:
            return ""
        lines = []
        for item in relevant[-10:]:
            strategy = item.get("strategy", "unknown")
            columns = item.get("columns", [])
            params = item.get("params", {})
            pc = item.get("pair_completeness", 0)
            candidates = item.get("num_candidates", 0)
            lines.append(f"- {strategy} cols={columns} params={params} PC={pc:.3f} candidates={candidates}")
        return "\n".join(lines)
    
    def _parse_llm_response(self, response_text: str, valid_columns: List[str]) -> Dict[str, Any]:
        """Parse and validate LLM response with all parameters."""
        try:
            cleaned = re.sub(r'^```(?:json)?\n?|```$', '', response_text.strip(), flags=re.MULTILINE)
            cleaned = cleaned.strip()
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except:
                    parsed = {}
            else:
                parsed = {}
        
        # Validate strategy
        valid_strategies = ['exact_match_single', 'exact_match_multi', 'sorted_neighbourhood', 'token_blocking', 'ngram_blocking', 'semantic_similarity']
        strategy = parsed.get('strategy', 'exact_match_single')
        if strategy not in valid_strategies:
            strategy = 'exact_match_single'
        
        # Validate columns
        columns = parsed.get('columns', [])
        if not columns or not isinstance(columns, list):
            columns = [valid_columns[0]] if valid_columns else ['name']
        columns = [c for c in columns if c in valid_columns]
        if not columns:
            columns = [valid_columns[0]] if valid_columns else ['name']
        
        # Validate parameters with bounds
        min_token_len = parsed.get('min_token_len', 4)
        if not isinstance(min_token_len, int):
            try:
                min_token_len = int(min_token_len)
            except:
                min_token_len = 4
        min_token_len = max(2, min(10, min_token_len))
        
        ngram_size = parsed.get('ngram_size', 4)
        if not isinstance(ngram_size, int):
            try:
                ngram_size = int(ngram_size)
            except:
                ngram_size = 4
        ngram_size = max(2, min(6, ngram_size))
        
        top_k = parsed.get('top_k', 20)
        if not isinstance(top_k, int):
            try:
                top_k = int(top_k)
            except:
                top_k = 20
        top_k = max(5, min(100, top_k))
        
        window = parsed.get('window', 15)
        if not isinstance(window, int):
            try:
                window = int(window)
            except:
                window = 15
        window = max(3, min(100, window))
        
        return {
            'strategy': strategy,
            'columns': columns,
            'min_token_len': min_token_len,
            'ngram_size': ngram_size,
            'window': window,
            'top_k': top_k,
            'reasoning': parsed.get('reasoning', 'No reasoning provided')
        }
    
    def _strategy_to_config(self, choice: Dict[str, Any]) -> Dict[str, Any]:
        """Convert strategy choice to blocker config."""
        strategy, columns = choice['strategy'], choice['columns']
        
        if strategy == 'exact_match_single':
            return {'name': f"Standard ({columns[0]})", 'type': 'standard', 'on': [columns[0]]}
        elif strategy == 'exact_match_multi':
            return {'name': f"Standard ({'+'.join(columns)})", 'type': 'standard', 'on': columns}
        elif strategy == 'sorted_neighbourhood':
            return {'name': f"SortedNeighbourhood ({columns[0]}, w={choice['window']})", 'type': 'sorted_neighbourhood',
                    'key': columns[0], 'window': choice['window']}
        elif strategy == 'semantic_similarity':
            return {'name': f"Embedding ({'+'.join(columns)}, k={choice['top_k']})", 'type': 'embedding', 
                    'text_cols': columns, 'top_k': choice['top_k']}
        elif strategy == 'token_blocking':
            return {'name': f"Token ({columns[0]}, len>={choice['min_token_len']})", 'type': 'token', 
                    'column': columns[0], 'min_token_len': choice['min_token_len']}
        elif strategy == 'ngram_blocking':
            return {'name': f"NGram ({columns[0]}, n={choice['ngram_size']})", 'type': 'ngram', 
                    'column': columns[0], 'ngram_size': choice['ngram_size']}
        
        return {'name': f"Standard ({columns[0]})", 'type': 'standard', 'on': [columns[0]]}
    
    def create_blocker(self, name_left: str, name_right: str, blocker_type: str, id_column: str = "id", **kwargs):
        """Create a blocker instance."""
        df_left, df_right = self.datasets_loaded[name_left], self.datasets_loaded[name_right]
        common_params = {'output_dir': self.output_dir, 'id_column': id_column}
        
        if blocker_type == 'standard':
            return StandardBlocker(df_left, df_right, on=kwargs.get('on', []), batch_size=1000, **common_params)
        elif blocker_type == 'sorted_neighbourhood':
            return SortedNeighbourhoodBlocker(df_left, df_right, key=kwargs.get('key', 'name'),
                                              window=kwargs.get('window', 15), batch_size=1000, **common_params)
        elif blocker_type == 'embedding':
            return EmbeddingBlocker(df_left, df_right, text_cols=kwargs.get('text_cols', []), 
                                   top_k=kwargs.get('top_k', 20), batch_size=1000, **common_params)
        elif blocker_type == 'token':
            return TokenBlocker(df_left, df_right, column=kwargs.get('column', 'name'), 
                               min_token_len=kwargs.get('min_token_len', 4), batch_size=1000, **common_params)
        elif blocker_type == 'ngram':
            return TokenBlocker(df_left, df_right, column=kwargs.get('column', 'name'),
                               ngram_size=kwargs.get('ngram_size', 4), ngram_type="character",
                               batch_size=1000, **common_params)
        
        raise ValueError(f"Unknown blocker type: {blocker_type}")
    
    def _is_acceptable(self, pc: float, num_candidates: int) -> Tuple[bool, str]:
        """Check if result is acceptable, with tolerance for excellent PC."""
        hard_limit = self.max_candidates
        soft_limit = int(self.max_candidates * (1 + self.candidate_tolerance))
        
        if pc >= self.pc_threshold and num_candidates <= hard_limit:
            return True, "perfect"
        
        if pc >= self.pc_threshold + 0.05 and num_candidates <= soft_limit:
            return True, "acceptable_with_tolerance"
        
        if num_candidates > soft_limit:
            return False, "exceeds_hard_limit"
        elif num_candidates > hard_limit:
            return False, "exceeds_soft_limit"
        else:
            return False, "low_pc"
    
    def evaluate_blocker(
        self,
        blocker,
        gold_standard: pd.DataFrame,
        name: str = "blocker",
        allow_overflow: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate a blocker against ground truth."""
        candidates = blocker.materialize()
        num_candidates = len(candidates)
        
        if self.verbose:
            print(f"    {name}: {num_candidates:,} candidates")
        
        hard_limit = int(self.max_candidates * (1 + self.candidate_tolerance))
        
        if num_candidates > hard_limit and not allow_overflow:
            if self.verbose:
                print(f"    ⛔ SKIPPING EVALUATION: {num_candidates:,} > {hard_limit:,} (hard limit)")
                print(f"    💡 Try increasing min_token_len or ngram_size, or use a more selective column")
            return {
                'method': name,
                'num_candidates': num_candidates,
                'num_gold_pairs': len(gold_standard),
                'pair_completeness': 0,
                'reduction_ratio': 0,
                'is_acceptable': False,
                'acceptance_reason': 'exceeds_hard_limit',
                'skipped_evaluation': True
            }
        
        result = self.evaluator.evaluate_blocking(
            candidate_pairs=candidates, 
            blocker=blocker, 
            test_pairs=gold_standard, 
            out_dir=self.output_dir
        )
        pc = result.get('pair_completeness', 0)
        is_acceptable, reason = self._is_acceptable(pc, num_candidates)
        
        result.update({
            'method': name, 
            'num_candidates': num_candidates, 
            'num_gold_pairs': len(gold_standard),
            'is_acceptable': is_acceptable,
            'acceptance_reason': reason,
            'skipped_evaluation': False
        })
        
        if self.verbose:
            rr = result.get('reduction_ratio', 0)
            if is_acceptable:
                status = "✅" if reason == "perfect" else "✅ (within tolerance)"
            else:
                status = "⚠️"
            print(f"    {status} PC={pc:.4f}, RR={rr:.4f}")
        
        return result

    def _select_informative_columns(self, common_columns: List[str], column_details: Dict[str, Any]) -> List[str]:
        """Pick 2–3 informative columns for semantic similarity fallback."""
        if not common_columns:
            return []

        if self.llm is not None:
            try:
                system_prompt = (
                    "You select the 2 or 3 most informative columns for semantic similarity blocking. "
                    "Return ONLY a JSON list of column names. Too many columns does not mean higher PC."
                )
                human_content = (
                    f"Common columns: {common_columns}\n"
                    f"Column details: {json.dumps(column_details, indent=2)}\n"
                    "Use column names and sample values (from column_details) to choose.\n"
                    "Pick 2 or 3 columns that best identify entities."
                )
                response = self.llm.invoke(
                    [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
                )
                raw = response.content if hasattr(response, "content") else str(response)
                cleaned = re.sub(r"^```(?:json)?\\n?|```$", "", raw.strip())
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    picked = [c for c in parsed if c in common_columns]
                    if 2 <= len(picked) <= 3:
                        return picked
            except Exception:
                pass

        keywords = ["title", "name", "artist", "author", "year", "date", "publish", "album", "track"]
        scored = []
        for col in common_columns:
            score = 0
            col_lower = col.lower()
            for key in keywords:
                if key in col_lower:
                    score += 5
            details = column_details.get(col, {})
            try:
                avg_tokens = float(details.get("avg_tokens", 0) or 0)
            except (TypeError, ValueError):
                avg_tokens = 0
            score += min(avg_tokens, 5)
            scored.append((score, col))
        scored.sort(reverse=True)
        picked = [col for _, col in scored[:3]]
        return picked[:3]

    def _strategy_params(self, strategy_choice: Dict[str, Any]) -> Dict[str, Any]:
        """Return only the relevant params for the chosen strategy."""
        strategy = strategy_choice.get("strategy")
        if strategy == "token_blocking":
            return {"min_token_len": strategy_choice.get("min_token_len")}
        if strategy == "ngram_blocking":
            return {"ngram_size": strategy_choice.get("ngram_size")}
        if strategy == "sorted_neighbourhood":
            return {"window": strategy_choice.get("window")}
        if strategy == "semantic_similarity":
            return {"top_k": strategy_choice.get("top_k")}
        return {}
    
    def _try_execute_strategy(
        self,
        strategy_choice: Dict,
        name_left: str,
        name_right: str,
        gold: pd.DataFrame,
        id_column: str,
        allow_overflow: bool = False,
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """Try to execute a blocking strategy."""
        config = self._strategy_to_config(strategy_choice)
        blocker_type = config.pop('type')
        blocker_name = config.pop('name')
        
        try:
            blocker = self.create_blocker(name_left, name_right, blocker_type=blocker_type, 
                                         id_column=id_column, **config)
            result = self.evaluate_blocker(
                blocker,
                gold,
                name=blocker_name,
                allow_overflow=allow_overflow,
            )
            result.update({
                'pair': f"{name_left}_{name_right}", 
                'strategy': strategy_choice['strategy'],
                'columns': strategy_choice['columns'], 
                'reasoning': strategy_choice['reasoning'],
                'params': self._strategy_params(strategy_choice)
            })
            return result, None
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[-500:]}"
            return None, error_msg
    
    def run_pair_with_llm(self, name_left: str, name_right: str, id_column: str = None) -> Dict[str, Any]:
        """Run LLM-driven blocking evaluation for a dataset pair."""
        pair_key = (name_left, name_right)
        reverse_key = (name_right, name_left)
        
        if pair_key in self.gold_standards:
            gold = self.gold_standards[pair_key]
        elif reverse_key in self.gold_standards:
            gold = self.gold_standards[reverse_key]
        else:
            raise ValueError(f"No gold standard found for pair: {pair_key}")
        
        soft_limit = int(self.max_candidates * (1 + self.candidate_tolerance))
        print(f"\n{'='*60}")
        print(f"🤖 BLOCKING: {name_left} <-> {name_right}")
        print(f"{'='*60}")
        print(f"Gold pairs: {len(gold)}, PC threshold: {self.pc_threshold}")
        print(f"Candidate limit: {self.max_candidates:,} (hard: {soft_limit:,} with tolerance)")
        
        if id_column is None:
            id_col_left = self._detect_id_column(self.datasets_loaded[name_left])
            id_col_right = self._detect_id_column(self.datasets_loaded[name_right])
            
            if id_col_left != id_col_right:
                print(f"    ID columns differ ({id_col_left}/{id_col_right}), renaming to 'record_id'")
                self.datasets_loaded[name_left] = self.datasets_loaded[name_left].rename(columns={id_col_left: 'record_id'})
                self.datasets_loaded[name_right] = self.datasets_loaded[name_right].rename(columns={id_col_right: 'record_id'})
                id_column = 'record_id'
            else:
                id_column = id_col_left
            print(f"    Using ID column: '{id_column}'")
        
        analysis = self._analyze_columns_for_pair(name_left, name_right, id_column)
        print(f"Dataset sizes: {analysis['left_size']:,} × {analysis['right_size']:,} = {analysis['potential_pairs']:,} potential pairs")
        print(f"Common columns: {analysis['common_columns']}")
        
        previous_attempts = []
        best_result = None
        
        for attempt in range(self.max_attempts):
            print(f"\n--- Attempt {attempt + 1}/{self.max_attempts} ---")
            
            strategy_choice = self._ask_llm_for_strategy(analysis, previous_attempts if attempt > 0 else None)
            print(f"LLM chose: {strategy_choice['strategy']} on {strategy_choice['columns']}")
            params = self._strategy_params(strategy_choice)
            print(f"  Params: {params}")
            print(f"  Reasoning: {strategy_choice['reasoning'][:100]}...")
            
            error_retries = 0
            last_error = None
            result = None
            
            while error_retries <= self.max_error_retries:
                if error_retries > 0:
                    print(f"  Retry {error_retries}/{self.max_error_retries} after error...")
                    strategy_choice = self._ask_llm_for_strategy(analysis, previous_attempts, last_error=last_error)
                    print(f"  LLM retry chose: {strategy_choice['strategy']} on {strategy_choice['columns']}")
                
                result, error = self._try_execute_strategy(strategy_choice, name_left, name_right, gold, id_column)
                
                if error:
                    print(f"  ❌ Error: {error[:200]}...")
                    last_error = error
                    error_retries += 1
                    continue
                
                break
            
            if result is None:
                previous_attempts.append({
                    'strategy': strategy_choice['strategy'],
                    'columns': strategy_choice['columns'],
                    'params': self._strategy_params(strategy_choice),
                    'error': True
                })
                continue
            
            previous_attempts.append({
                'strategy': strategy_choice['strategy'],
                'columns': strategy_choice['columns'],
                'params': self._strategy_params(strategy_choice),
                'pair_completeness': result.get('pair_completeness', 0),
                'num_candidates': result.get('num_candidates', 0),
                'error': False
            })
            
            if result.get('is_acceptable', False):
                print(f"✅ Found acceptable strategy!")
                best_result = result
                break
            
            if best_result is None or result.get('pair_completeness', 0) > best_result.get('pair_completeness', 0):
                best_result = result
        
        if best_result is None:
            print(f"⚠️ No successful strategy found after {self.max_attempts} attempts")
            best_result = {
                'pair': f"{name_left}_{name_right}",
                'strategy': 'failed',
                'columns': [],
                'pair_completeness': 0,
                'num_candidates': 0,
                'is_acceptable': False
            }

        if not best_result.get('is_acceptable', False):
            fallback_cols = self._select_informative_columns(
                analysis.get("common_columns", []),
                analysis.get("column_details", {})
            )
            if fallback_cols:
                print("⚠️ Forcing final semantic_similarity fallback with informative columns")
                fallback_choice = {
                    "strategy": "semantic_similarity",
                    "columns": fallback_cols[:3],
                    "top_k": 20,
                    "reasoning": "forced fallback after 4 attempts"
                }
                fallback_result, _ = self._try_execute_strategy(
                    fallback_choice,
                    name_left,
                    name_right,
                    gold,
                    id_column,
                    allow_overflow=True,
                )
                if fallback_result:
                    if fallback_result.get('pair_completeness', 0) >= best_result.get('pair_completeness', 0):
                        best_result = fallback_result

        self.results_history.append(best_result)
        return best_result
    
    def run_all_pairs(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run blocking evaluation for all dataset pairs."""
        all_results = []
        discovered_config = {
            "blocking_strategies": {},
            "id_columns": {}
        }
        
        for (name_left, name_right) in self.gold_standards.keys():
            result = self.run_pair_with_llm(name_left, name_right)
            all_results.append(result)

            discovered_config["blocking_strategies"][f"{name_left}_{name_right}"] = {
                "strategy": result.get('strategy'),
                "columns": result.get('columns', []),
                "params": result.get('params', {}),
                "pair_completeness": result.get('pair_completeness', 0),
                "num_candidates": result.get('num_candidates', 0),
                "is_acceptable": result.get('is_acceptable', False)
            }
        
        for name, df in self.datasets_loaded.items():
            for col in df.columns:
                if 'id' in col.lower():
                    discovered_config["id_columns"][name] = col
                    break
        
        df = pd.DataFrame(all_results)
        
        if not df.empty:
            df.to_csv(os.path.join(self.output_dir, "blocking_tester_results.csv"), index=False)
            print(f"\n{'='*60}")
            print(f"📊 RESULTS SUMMARY: {len(df)} pairs evaluated")
            print(f"{'='*60}")
            for _, row in df.iterrows():
                pc = row.get('pair_completeness', 0)
                candidates = row.get('num_candidates', 0)
                is_acceptable = row.get('is_acceptable', False)
                status = "✅" if is_acceptable else "⚠️"
                print(f"  {status} {row['pair']}: PC={pc:.4f}, candidates={candidates:,}, strategy={row['strategy']}")
        
        with open(os.path.join(self.output_dir, "blocking_config.json"), "w") as f:
            json.dump(discovered_config, f, indent=2)
        print(f"\n💾 Config saved to: {self.output_dir}/blocking_config.json")
        
        return df, discovered_config
