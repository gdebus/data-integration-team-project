# Agent Pipeline -- Deep Dive: How It Works

## 1. High-Level Flow

The agent is a LangGraph state machine (`SimpleModelAgent` class in `pipeline_agent.py`, ~2080 lines) that autonomously integrates multiple datasets into a single fused output. The graph follows this path:

```
match_schemas -> profile_data -> normalization_node -> run_blocking_tester -> run_matching_tester
  -> pipeline_adaption -> execute_pipeline -> evaluation_node -> investigator_node
  -> (loops back to pipeline_adaption/normalization/matching OR -> human_review_export
     -> sealed_final_test_evaluation -> save_results)
```

Each node reads from and writes to a shared `SimpleModelAgentState` TypedDict (~60 fields). The graph uses conditional edges to route between nodes based on results. The recursion limit is 200 (`GRAPH_RECURSION_LIMIT`).

---

## 2. State Object: SimpleModelAgentState

The shared state is a TypedDict defined at `pipeline_agent.py:302`. Key field groups:

| Group | Fields | Purpose |
|-------|--------|---------|
| **Datasets** | `datasets`, `original_datasets`, `normalized_datasets` | File paths to source CSVs/XMLs/Parquets |
| **Testsets** | `entity_matching_testsets`, `fusion_testset`, `validation_fusion_testset` | Evaluation targets |
| **Configs** | `blocking_config`, `matching_config`, `matcher_mode` | Frozen configs from testers |
| **Messages** | `messages`, `eval_messages` | LLM conversation history |
| **Pipeline** | `integration_pipeline_code`, `pipeline_execution_result`, `pipeline_execution_attempts` | Generated code + exec state |
| **Evaluation** | `evaluation_code`, `evaluation_metrics`, `evaluation_metrics_raw`, `evaluation_metrics_for_adaptation`, `best_validation_metrics`, `best_pipeline_code`, `best_evaluation_code` | Metrics tracking + best snapshot |
| **Investigation** | `investigator_decision`, `investigator_action_plan`, `evaluation_reasoning_brief`, `fusion_guidance`, `investigator_probe_results` | Investigation outputs |
| **Diagnostics** | `auto_diagnostics`, `correspondence_integrity`, `evaluation_regression_guard`, `evaluation_cycle_audit` | Bookkeeping and guards |
| **History** | `iteration_history`, `evaluation_cycle_audit` | Per-attempt records for cross-attempt learning |
| **Normalization** | `normalization_attempts`, `normalization_report`, `normalization_directives`, `normalization_pending_acceptance` | Normalization state + rollback |
| **Scaffold** | `pipeline_scaffold` | Frozen/mutable code splitting dict |
| **Run** | `run_id`, `run_output_root`, `pipeline_run_started_at`, `pipeline_run_finished_at` | Per-run metadata |

---

## 3. Node-by-Node Walkthrough

### 3.1 Schema Matching

| | |
|-|-|
| **Function** | `SimpleModelAgent.match_schemas()` at `pipeline_agent.py:836` |
| **Reads** | `datasets`, `schema_correspondences` |
| **Writes** | `schema_correspondences` |
| **LLM invocation** | Delegates to `run_schema_matching()` in `schema_matching_node.py` which uses PyDI's `LLMBasedSchemaMatcher` or `LabelBasedSchemaMatcher`. The LLM receives column names + sample values and returns column mappings. |
| **Guardrails** | None -- purely LLM-driven schema alignment. |
| **Skip logic** | If `schema_correspondences` already present in state, the node returns immediately with no work done. |
| **Routing** | Unconditional edge to `profile_data`. |

**What can go wrong**: The LLM may produce incorrect column mappings (e.g., mapping `release-date` to `release-country`). These propagate downstream and cause incorrect fusion. The schema matcher output is saved to `output/runs/<run>/schema-matching/` for manual inspection.

**Example (Music)**: Aligns columns across discogs, lastfm, and musicbrainz:
```
discogs:     [id, name, artist, release-date, release-country, duration, label, genre, tracks]
lastfm:      [id, name, artist, duration]
musicbrainz: [id, name, artist, release-date, release-country, duration]
```

---

### 3.2 Data Profiling

| | |
|-|-|
| **Function** | `SimpleModelAgent.profile_data()` at `pipeline_agent.py:862` |
| **Reads** | `datasets` |
| **Writes** | `data_profiles` |
| **LLM invocation** | Uses `PROFILE_SYSTEM_PROMPT` from `prompts/profile_prompt.py`. The model is called with `bind_tools` (the `profile_dataset` tool). The LLM generates one tool call per dataset, each calling `ProfileDatasetTool._run(path)` which uses PyDI's `DataProfiler.summary()`. |
| **Guardrails** | None. |
| **Routing** | Unconditional edge to `normalization_node`. |

**How it works**: The LLM sees the dataset paths and calls the `profile_dataset` tool once per dataset. Each call loads the dataset (CSV/XML/Parquet via PyDI loaders), runs `DataProfiler().summary()`, and returns a JSON profile. Profiles include column types, null rates, cardinality, and sample values. The profiles are saved to `output/runs/<run>/profile/profiles.json`.

**Cost**: Minimal (~$0.002) -- the LLM only generates tool calls, actual profiling is done by PyDI.

---

### 3.3 Normalization

| | |
|-|-|
| **Function** | `run_normalization_node()` in `helpers/normalization_orchestrator.py`, registered as `lambda state: run_normalization_node(self, state, load_dataset)` at `pipeline_agent.py:405` |
| **Reads** | `datasets`, `validation_fusion_testset` (or `fusion_testset`), `normalization_attempts`, `normalization_blocked_by_rejection_streak` |
| **Writes** | `datasets` (overwritten with normalized paths), `normalized_datasets`, `original_datasets`, `normalization_report`, `normalization_attempts`, `normalization_execution_result` |
| **LLM invocation** | Uses `NORMALIZATION_SPEC_SYSTEM_PROMPT` + `build_normalization_user_prompt()` from `prompts/normalization_prompt.py`. The LLM sees 15-row probes from each source dataset + the validation/target set. It generates a `NormalizationSpec` dict per source dataset. |
| **Guardrails** | `_apply_spec_to_dataframe()` validates column existence, sanitizes ColumnSpec fields, preserves ID columns. Spec recovery auto-detects when LLM returns specs without the expected wrapper. |
| **Routing** | Unconditional edge to `run_blocking_tester`. |

**What can go wrong**: The LLM may generate specs that destroy data (e.g., country_format="alpha_2" when the validation expects full country names). This is detected on the next evaluation cycle -- the investigator can roll back via the normalization acceptance gate.

**Normalization acceptance gate**: When the investigator routes to normalization, it records a gate request with `baseline_accuracy`. After the next evaluation, `evaluate_pending_normalization_acceptance()` compares the new accuracy against the baseline. If the delta is below `DEFAULT_MIN_DELTA`, the normalization is rejected and datasets are rolled back to `original_datasets`. After `NORMALIZATION_MAX_CONSECUTIVE_REJECTIONS` (2) consecutive rejections, normalization is blocked entirely.

**Example (Books)**: The LLM sees that the validation set expects genres as comma-separated strings, but goodreads uses semicolons. It generates:
```python
{"columns": {"genres": {"strip_whitespace": True}}}
```

**Example (Music)**: The LLM normalizes release-country to full names and dates to ISO format:
```python
{"columns": {"release-country": {"country_format": "name"}, "release-date": {"date_format": "%Y-%m-%dT00:00:00"}}}
```

---

### 3.4 Blocking

| | |
|-|-|
| **Function** | `SimpleModelAgent.run_blocking_tester()` at `pipeline_agent.py:902` |
| **Reads** | `datasets`, `entity_matching_testsets`, `blocking_config` |
| **Writes** | `blocking_config` |
| **LLM invocation** | `BlockingTester` (in `blocking_tester.py`, ~1300 lines) uses the LLM to select which columns to block on. Tests `StandardBlocker`, `TokenBlocker`, `SortedNeighbourhoodBlocker`, and `EmbeddingBlocker` on each dataset pair. |
| **Guardrails** | Config signature validation via `_validate_dataset_config_signature()` -- if existing config's dataset names don't match current datasets, recomputes. |
| **Skip logic** | If `blocking_config` exists and signature matches, skips entirely. |
| **Routing** | Unconditional edge to `run_matching_tester`. |

**Key thresholds**:
- `BLOCKING_PC_THRESHOLD` = 0.9 (pair completeness target)
- `BLOCKING_MAX_CANDIDATES` = 350,000 (upper bound on candidate pairs)
- `BLOCKING_MAX_ATTEMPTS` = 5
- `BLOCKING_MAX_ERROR_RETRIES` = 2

**Example (Books)**: ISBN is nearly unique -- `StandardBlocker` on `isbn_clean` achieves PC=1.0 with only 576 candidates (vs 100M potential pairs).

**Example (Music)**: Names vary across sources -- `EmbeddingBlocker` on `name` with `top_k=10` produces ~3000-5000 candidates per pair.

---

### 3.5 Matching

| | |
|-|-|
| **Function** | `SimpleModelAgent.run_matching_tester()` at `pipeline_agent.py:937` |
| **Reads** | `datasets`, `entity_matching_testsets`, `blocking_config`, `matching_config`, `matcher_mode` |
| **Writes** | `matching_config`, `matcher_mode` |
| **LLM invocation** | `MatchingTester` (in `matching_tester.py`, ~2260 lines) uses the LLM to select comparators and weights. Tests `RuleBasedMatcher` and `MLBasedMatcher` with various comparator combinations. |
| **Guardrails** | Hot-reloads `matching_tester` module from disk for notebook sessions. Config validation via `config_matches_datasets()`, `config_has_list_based_comparators()`, and `matching_config_needs_refresh()`. |
| **Skip logic** | If config exists, signature matches, and F1 is sufficient, skips. If F1 is low (`needs_refresh` returns true), recomputes. |
| **Routing** | Unconditional edge to `pipeline_adaption`. |

**Key thresholds**:
- `MATCHING_F1_THRESHOLD` = 0.75 (target F1)
- `MATCHING_F1_GATE` = 0.65 (minimum F1 to proceed)
- `MATCHING_MAX_ATTEMPTS` = 8
- `disallow_list_comparators` = True (by default)

**Example (Music -- discogs to lastfm)**:
```python
comparators = [
    StringComparator("name", "cosine", preprocess=lower_strip),       # weight=0.4
    StringComparator("artist", "jaro_winkler", preprocess=lower_strip), # weight=0.25
    NumericComparator("duration", max_difference=15.0),                # weight=0.15
]
threshold = 0.7  # F1=0.955
```

---

### 3.6 Pipeline Generation

| | |
|-|-|
| **Function** | `SimpleModelAgent.pipeline_adaption()` at `pipeline_agent.py:1025` |
| **Reads** | `datasets`, `data_profiles`, `blocking_config`, `matching_config`, `matcher_mode`, `evaluation_metrics`, `evaluation_metrics_for_adaptation`, `evaluation_analysis`, `evaluation_reasoning_brief`, `auto_diagnostics`, `investigator_action_plan`, `fusion_guidance`, `cluster_analysis_result`, `pipeline_scaffold`, `integration_pipeline_code`, `pipeline_execution_result`, `iteration_history` |
| **Writes** | `messages`, `integration_pipeline_code`, `pipeline_generation_review`, `pipeline_scaffold` |
| **LLM invocation** | Uses `build_pipeline_system_prompt()` from `prompts/pipeline_prompt.py` (~415 lines). Assembles: example pipeline, dataset profiles, blocking/matching configs, PyDI API cheat-sheet, evaluation analysis, reasoning brief, fusion guidance, cluster evidence, iteration history, input data samples, correspondence summary. |
| **Tools available** | `search_documentation` (Chroma vector DB search with k=8). `profile_dataset`. |
| **Guardrails** | `apply_pipeline_guardrails()` applied to every generated pipeline. `static_pipeline_sanity_findings()` triggers self-review when issues found. |
| **Routing** | Conditional edge via `should_continue_research()` at `pipeline_agent.py:809`: if the model made tool calls (still researching), loops back to `pipeline_adaption`; otherwise routes to `execute_pipeline`. Loop limit: `PIPELINE_TOOLCALL_LOOP_LIMIT` = 6. |

**Three generation modes**:

1. **Initial generation** (no existing pipeline): LLM generates complete pipeline from scratch.
2. **Patch mode** (cycle 2+ with evaluation feedback, no execution error): Scaffold system freezes infrastructure (imports, blocking, matching). LLM only generates mutable section (fusion strategy, post-clustering, trust maps). Uses `build_patch_prompt_context()` to show frozen code with markers.
3. **Full regeneration** (execution error): Shows broken code + error message, asks LLM to fix.

**Self-review**: After guardrails, `static_pipeline_sanity_findings()` checks for issues (missing correspondence saves, unused imports, etc.). If issues found, runs a two-stage review:
1. **Review** (`REVIEW_SYSTEM_PROMPT`): critic evaluates the pipeline, returns "pass" or "revise" with problem classes.
2. **Revision** (`REVISION_SYSTEM_PROMPT`): targeted fix with minimal changes. Revision is validated -- rejected if it introduces new sanity issues.

**Example (Music, patch mode)**: LLM receives per-attribute status:
```
artist: 92.0% -- PROTECTED, do NOT change
name: 61.3% -- needs improvement
release-country: 0.0% -- needs improvement
```
And generates only the fusion strategy replacement.

---

### 3.7 Pipeline Execution

| | |
|-|-|
| **Function** | `SimpleModelAgent.execute_pipeline()` at `pipeline_agent.py:1318` |
| **Reads** | `integration_pipeline_code`, `pipeline_execution_attempts` |
| **Writes** | `pipeline_execution_result`, `pipeline_execution_attempts`, `fusion_size_comparison` |
| **LLM invocation** | None -- pure subprocess execution via `run_pipeline_subprocess()` in `helpers/script_runner.py`. |
| **Guardrails** | `apply_pipeline_guardrails()` already applied in `pipeline_adaption`. Error classification via `classify_execution_error()` in `helpers/error_classifier.py`. |
| **Routing** | Conditional edge via `_route_after_execution()` at `pipeline_agent.py:608`: |

**Routing logic**:
```python
def _route_after_execution(self, state):
    if state["pipeline_execution_result"].lower().startswith("success"):
        return "evaluation_node"
    if state["pipeline_execution_attempts"] < PIPELINE_EXEC_MAX_ATTEMPTS:  # 3
        return "pipeline_adaption"  # retry with error feedback
    return END  # give up
```

**Error classification** (`helpers/error_classifier.py`): Classifies errors into categories (timeout, memory_error, import_error, syntax_error, recursion_error, data_error, runtime_crash). Each category gets an actionable suggestion and retryable flag. RecursionError specifically suggests applying post-clustering.

**Fusion size monitoring**: `compare_estimates_with_actual()` from `fusion_size_monitor.py` compares actual row count against estimates from blocking/matching metrics. Saves `fusion_size_estimate.json`.

**Timeout**: `PIPELINE_EXEC_TIMEOUT` = 3600 seconds (1 hour).

---

### 3.8 Evaluation

| | |
|-|-|
| **Function** | `run_evaluation_node()` in `helpers/evaluation_orchestrator.py`, registered as `lambda state: run_evaluation_node(self, state)` at `pipeline_agent.py:410` |
| **Reads** | `integration_pipeline_code`, `data_profiles`, `fusion_testset`, `validation_fusion_testset`, `evaluation_execution_result`, `evaluation_execution_attempts`, `normalization_directives`, `normalization_report` |
| **Writes** | `evaluation_code`, `evaluation_execution_result`, `evaluation_execution_attempts`, `evaluation_metrics_from_execution`, `evaluation_error_classification` |
| **LLM invocation** | `evaluation_adaption()` at `pipeline_agent.py:1357` uses a system prompt with example evaluation code, pipeline code, dataset profiles, fused output path, evaluation set path + preview, and `EVALUATION_ROBUSTNESS_RULES_BLOCK`. |
| **Guardrails** | `apply_evaluation_guardrails()` applied to every generated evaluation script. |
| **Routing** | Conditional edge via `_route_after_evaluation()` at `pipeline_agent.py:616`: |

**Routing logic**:
```python
def _route_after_evaluation(self, state):
    if result.lower().startswith("success"):
        return "investigator_node"
    # Allow one graph-level retry
    if retryable and eval_node_retries < 1:
        return "evaluation_node"
    return END
```

**Evaluation functions** (per attribute type):
| Function | Used for | Example |
|----------|----------|---------|
| `tokenized_match(threshold=0.8)` | Names, titles, text | "The Beatles" ~ "Beatles, The" |
| `year_only_match` | Date columns | "2003-01-01" ~ "2003-06-15" |
| `numeric_tolerance_match(tolerance=5.0)` | Duration, scores | 2780 ~ 2781 |
| `set_equality_match` | List columns (genres) | "Rock, Pop" ~ "Pop, Rock" |
| `exact_match` | IDs, codes | "US" = "US" |

**Sealed vs validation**: When both `validation_fusion_testset` and `fusion_testset` exist, the evaluation uses the validation set during the iteration loop and the test set only in the sealed final evaluation. This prevents overfitting.

---

### 3.9 Investigation (Investigator Node)

| | |
|-|-|
| **Function** | `run_investigator_node()` in `helpers/investigator_orchestrator.py:359`, registered as `lambda state: run_investigator_node(self, state)` at `pipeline_agent.py:411` |
| **Reads** | Nearly everything: evaluation metrics, auto_diagnostics, correspondence_integrity, datasets, blocking_config, matching_config, integration_pipeline_code, iteration_history, normalization state |
| **Writes** | `investigator_decision`, `investigation_log`, `evaluation_reasoning_brief`, `evaluation_analysis`, `investigator_action_plan`, `fusion_guidance`, `investigator_probe_results`, `iteration_history`, `cluster_analysis_result`, all evaluation_decision outputs |
| **LLM invocation** | `agent.investigate()` at `pipeline_agent.py:1554` uses `INVESTIGATION_SYSTEM_PROMPT` + `build_investigation_context()` from `prompts/investigation_prompt.py`. Multi-turn loop with max `INVESTIGATION_MAX_TURNS` = 4 turns. |
| **Guardrails** | 3 hardcoded safety overrides after LLM decision. |
| **Routing** | Conditional edge reads `state["investigator_decision"]` directly. Maps to: `normalization_node`, `run_blocking_tester`, `run_matching_tester`, `pipeline_adaption`, or `human_review_export`. |

**10-step flow**:

1. **Evaluation decision** (metric bookkeeping): `process_evaluation_decision()` reads metrics, applies sanity checks, runs regression guard, tracks best metrics. Pure bookkeeping.
2. **Acceptance feedback**: `evaluate_pending_normalization_acceptance()` checks if the last normalization helped. Rollback if rejected. Consecutive rejection streak blocking.
3. **Dataset signature**: For history tracking.
4. **Run probes**: 11 built-in probes provide structured evidence (see Probe System section below).
5. **Early exit**: If `overall_accuracy >= QUALITY_GATE_THRESHOLD` (0.85) or `attempts >= MAX_INVESTIGATION_ATTEMPTS` (4), routes to `human_review_export`.
6. **Cluster analysis**: `ClusterTester.run()` analyzes correspondence files for cluster health, ambiguity, 1:many patterns. Rates all 5 PyDI post-clustering algorithms.
7. **Build fusion guidance**: `_build_fusion_guidance()` extracts mismatch classifications from probes + cluster recommendations. Pure data transformation.
8. **LLM investigation loop**: `agent.investigate()` multi-turn loop. LLM can write and execute diagnostic Python, then decides.
8b. **Enrich fusion guidance**: `_enrich_fusion_guidance_with_strategies()` populates `attribute_strategies` from source_attribution probe (confidence=0.7) + investigation recommendations (confidence=0.9).
9. **Safety overrides**: Only 3 hardcoded overrides:
   - Structural invalid correspondences -> force `pipeline_adaption`
   - Normalization blocked (rejection streak or max attempts 3) -> force `pipeline_adaption`
   - Invalid routing target -> default `pipeline_adaption`
10. **Save transcript + history**: Full investigation JSON saved to `output/investigation/`. History entry with per-attribute accuracies appended to `investigator_history.jsonl`.

**Multi-turn investigation** (inside `agent.investigate()` at `pipeline_agent.py:1554`):
```
Turn 1: LLM sees ALL evidence (32KB+ context)
         -> {"action": "investigate", "code": "import pandas as pd..."}   (test hypothesis)
         -> OR {"action": "decide", "next_node": "...", "diagnosis": "..."}  (decide immediately)
Turn 2+: LLM sees code execution stdout/stderr
         -> Writes more code or decides
Final:   If max turns reached without decision, defaults to pipeline_adaption
```

Investigation code executes in subprocess with guardrails:
- `sys.stdin.read()` -> `''` (blocks stdin)
- `input()` -> `''` (blocks interactive input)
- `subprocess.DEVNULL` for stdin
- Timeout: `INVESTIGATION_CODE_TIMEOUT` = 120 seconds

**Concrete example (Music, attempt 1, from `investigation_1_20260322T124405Z.json`)**:

The investigator decided in 1 turn (no diagnostic code needed) with this diagnosis:
> "The main remaining errors are in fusion strategy, not blocking or matching. Correspondence files are non-empty and matching F1 is strong (~0.87-0.90). The current fusion config is choosing the wrong resolvers/trust behavior."

Recommendations:
- `release-country`: musicbrainz exact=100%, discogs exact=0% -> `prefer_higher_trust(musicbrainz)` (expected_impact: 0.07)
- `genre`: discogs exact=94%, musicbrainz null -> `prefer_higher_trust(discogs)` (expected_impact: 0.032)
- `label`: discogs exact=94%, musicbrainz exact=0% -> `prefer_higher_trust(discogs)` (expected_impact: 0.022)
- `name`: musicbrainz exact=84%, discogs exact=39% -> `prefer_higher_trust(musicbrainz)` (expected_impact: 0.017)
- `duration`: musicbrainz exact=52%, discogs exact=0% -> `prefer_higher_trust(musicbrainz)` (expected_impact: 0.01)

Result: accuracy improved from 55.6% to 64.8% on the next iteration.

**Concrete example (Books, attempt 1, from `investigator_history.jsonl`)**:

Decision: `normalization_node` because genres at 0% accuracy due to source/target format mismatch. After normalization on attempt 2, isbn_clean regressed from 100% to 0% (leading-zero handling issue). Investigator detected this and routed back to `normalization_node` to fix. By attempt 3, isbn_clean restored and investigator routed to `pipeline_adaption` for fusion strategy changes.

---

### 3.10 Human Review Export

| | |
|-|-|
| **Function** | `SimpleModelAgent.human_review_export()` at `pipeline_agent.py:1740` |
| **Reads** | `integration_pipeline_code`, `data_profiles`, `fusion_testset`, `evaluation_metrics`, `best_pipeline_code` |
| **Writes** | `human_review_code`, `human_review_execution_result`, `human_review_report` |
| **LLM invocation** | Uses `HUMAN_REVIEW_SYSTEM_PROMPT` from `prompts/diagnostics_prompt.py` to generate a review export script that creates wide-format comparison tables with source lineage. |
| **Guardrails** | None beyond standard code extraction. |
| **Routing** | Conditional edge via `_route_after_human_review()` at `pipeline_agent.py:630`: routes to `sealed_final_test_evaluation` when both validation and test sets exist, otherwise to `save_results`. |

---

### 3.11 Sealed Final Test Evaluation

| | |
|-|-|
| **Function** | `SimpleModelAgent.sealed_final_test_evaluation()` at `pipeline_agent.py:1913` |
| **Reads** | `best_pipeline_code`, `integration_pipeline_code`, `validation_fusion_testset`, `fusion_testset` |
| **Writes** | `final_test_evaluation_execution_result`, `final_test_evaluation_metrics` |
| **LLM invocation** | Re-uses `evaluation_adaption()` and `execute_evaluation()` with `_force_test_eval=True` to target the held-out test set. |
| **Guardrails** | Uses best pipeline code (not latest) to prevent overfitting. If best pipeline re-execution fails, falls back to latest. |
| **Routing** | Unconditional edge to `save_results`. |

**Key behavior**: Re-executes the best pipeline (the one that produced the highest validation accuracy), generates evaluation code targeting the held-out test set, and runs it. This provides the final unbiased accuracy number.

---

### 3.12 Save Results

| | |
|-|-|
| **Function** | `SimpleModelAgent.save_results()` at `pipeline_agent.py:1996` |
| **Reads** | Nearly everything -- final metrics, pipeline code, run metadata, token usage |
| **Writes** | `run_audit_path`, `run_report_path`, `validation_metrics_final`, `sealed_test_metrics_final` |
| **LLM invocation** | None. |
| **Routing** | Unconditional edge to `END`. |

Delegates to `_save_results()` in `helpers/results_writer.py` (~460 lines). Saves:
- `run_<timestamp>.json` -- full results JSON
- `run_audit_<timestamp>.json` -- evaluation cycle audit trail
- `run_report_<timestamp>.md` -- markdown report with cost, timing, metrics trajectory

---

## 4. The Guardrail System

Two distinct guardrail functions mechanically fix LLM-generated code on every invocation:

### 4.1 Pipeline Guardrails (`apply_pipeline_guardrails`)

**File**: `helpers/code_guardrails.py:6`, ~700 lines.

Applied at `pipeline_agent.py:1284` before every pipeline execution. Takes the generated pipeline code and the full state dict. Returns corrected code.

**1. include_singletons=True enforcement**:
Forces `include_singletons=True` in `engine.run()` calls. If missing, injects it. This is a hard invariant -- the full fused dataset must always be produced.

**2. Matching threshold freezing**:
Reads thresholds from `matching_config["matching_strategies"]` and patches threshold variable assignments to match the MatchingTester-selected values. Prevents the LLM from accidentally changing matching thresholds.

Exception: When the investigator routed to `run_matching_tester`, thresholds are "unlocked" but clamped to plus or minus 0.1 of the original value.

**3. Fusion guidance enforcement**:
Reads `fusion_guidance["attribute_strategies"]` and patches `add_attribute_fuser()` calls with investigator-recommended resolvers.

Confidence-based override logic:
- Confidence >= 0.85 (from investigation): overrides the LLM's choice
- Confidence < 0.85 (from probe only): respects the LLM's choice if it picked a valid built-in resolver

Before:
```python
# LLM generated (wrong -- global trust_map can't satisfy both directions):
trust_map = {"musicbrainz": 3, "discogs": 2, "lastfm": 1}
strategy.add_attribute_fuser("label", most_complete, trust_map=trust_map)
strategy.add_attribute_fuser("duration", maximum, trust_map=trust_map)
```

After guardrails:
```python
trust_map_label = {"discogs": 3, "musicbrainz": 2, "lastfm": 1}
trust_map_duration = {"musicbrainz": 3, "lastfm": 2, "discogs": 1}
strategy.add_attribute_fuser("label", prefer_higher_trust, trust_map=trust_map_label)
strategy.add_attribute_fuser("duration", prefer_higher_trust, trust_map=trust_map_duration)
```

**4. Per-attribute trust_map injection**:
When multiple attributes need `prefer_higher_trust` with conflicting trust orderings, injects per-attribute `trust_map_<attr>` variables. Detected by comparing the sorted key orderings across all prefer_higher_trust strategies.

**5. list_strategy validation** (per comparator type):
- `StringComparator`: valid = {"best_match", "set_jaccard", "set_overlap"}. "concatenate" is upgraded to "best_match".
- `NumericComparator`: valid = {"average", "best_match", "range_overlap", "set_jaccard"}.
- `DateComparator`: valid = {"closest_dates", "range_overlap", "average_dates", "latest_dates", "earliest_dates"}.

**6. General import safety**:
Scans all `add_attribute_fuser()` calls for resolver symbols not in the import list. Auto-adds `from PyDI.fusion import <symbol>` for any missing built-in resolver.

**7. eval_id injection**:
Detects the validation set ID prefix (e.g., `mbrainz_` for music, `metacritic_` for games) and injects a `_extract_eval_id()` function + `eval_id` column creation into the pipeline code. This maps fused cluster IDs back to validation-aligned source IDs.

**8. ML post-clustering safety**:
When `matcher_mode="ml"` and no post-clustering exists in the code, injects `MaximumBipartiteMatching` on each pairwise correspondence set before concatenation. Prevents RecursionError from large connected components.

**9. Output path rewriting**:
Rewrites hardcoded `output/` paths to the run-scoped directory (`output/runs/<run_id>/`).

### 4.2 Evaluation Guardrails (`apply_evaluation_guardrails`)

**File**: `helpers/code_guardrails.py:923`, ~250 lines.

Applied at `pipeline_agent.py:1521` before every evaluation execution.

**1. Robust imports**: Adds fallback import resolution for `list_normalization` module.

**2. Output path rewriting**: Redirects hardcoded `"output/"` to run-scoped directory.

**3. fused_id_column -> eval_id**: If the pipeline created an `eval_id` column (detected by reading the fused CSV headers), patches `fused_id_column="_id"` to `fused_id_column="eval_id"`. If the pipeline did NOT create eval_id, reverts any LLM-generated eval_id references back to `_id`.

**4. Sub-column stripping**: Drops fused sub-columns (e.g., `tracks_track_name`, `tracks_track_duration`) when the gold set only has the parent column (`tracks`). Without this, evaluation would report 0% accuracy entries for columns that can never match, wasting investigation cycles.

**5. Numeric type coercion**: Injects `pd.to_numeric()` + `.astype("Int64")` casts for year/count/score/page/price columns in both fused and gold DataFrames. Prevents `274.0 != 274` false mismatches.

**6. List separator normalization**: Detects and aligns list separators between fused and gold (e.g., `"; "` -> `", "`) for genre/category/tag columns.

---

## 5. The Probe System

**File**: `helpers/investigator_probe_runner.py`, ~1100 lines.

11 built-in probes registered in `PROBE_REGISTRY` (dict at line 954). Each probe is a function that reads state and returns a dict with at least `{name, summary}` plus optional `normalization_pressure` and `actionability_pressure` scores. Probes are evidence only -- the investigation LLM interprets them.

**Execution**: `run_investigator_probes()` (line 1038) runs probes in priority order, respecting a time budget of `PROBE_MAX_RUNTIME_SECONDS` (6.0 seconds) and max `MAX_PROBES` (10). Keyword boosting adds `recent_mismatches` and `directive_coverage` when the action plan mentions normalization-related terms.

**Default probe order**: reason_distribution, worst_attributes, mismatch_sampler, attribute_improvability, source_attribution, null_patterns, correspondence_density, blocking_recall, fusion_size.

### Probe 1: `reason_distribution`
**Function**: `_probe_reason_distribution()` (line 37)
**Input**: `state["auto_diagnostics"]["debug_reason_ratios"]`
**Output**: Top 5 mismatch reasons sorted by frequency. `normalization_pressure` = sum of format/list/type/encoding reason ratios.
**Example**: `"mismatch=0.45, missing_fused_value=0.30, case_mismatch=0.15"`

### Probe 2: `worst_attributes`
**Function**: `_probe_worst_attributes()` (line 60)
**Input**: `state["evaluation_metrics_for_adaptation"]` (all `*_accuracy` keys)
**Output**: Bottom 5 attributes by accuracy. `actionability_pressure` = `1.0 - worst_accuracy`.
**Example**: `"release-country_accuracy=0.00, genre_accuracy=0.16, tracks_track_name_accuracy=0.17"`

### Probe 3: `mismatch_sampler`
**Function**: `_probe_mismatch_sampler()` (line 176)
**Input**: `debug_fusion_eval.jsonl` (reads up to `MAX_EVENTS` = 500 events)
**Output**: Concrete `{expected, fused, reason}` examples grouped by worst attribute. Top `MISMATCH_SAMPLE_ATTRIBUTES` (5) attributes, `MISMATCH_SAMPLE_ROWS` (5) samples each.
**Classification**: `_classify_mismatch_reason()` (line 258) sub-classifies each mismatch into: `missing_fused`, `missing_expected`, `case_mismatch`, `whitespace_mismatch`, `format_mismatch`, `list_format_mismatch`, `country_format`, `value_mismatch`.
**Example output**:
```json
{"release-country": [
    {"expected": "United Kingdom", "fused": "GB", "reason": "country_format"},
    {"expected": "United States", "fused": "", "reason": "missing_fused"}
]}
```

### Probe 4: `attribute_improvability`
**Function**: `_probe_attribute_improvability()` (line 491)
**Input**: Evaluation metrics + accumulated probe results (uses worst_attributes and mismatch_sampler)
**Output**: Per-attribute ceiling estimates and improvability classification: `improvable`, `structurally_limited`, or `at_ceiling`. Also computes `structural_ceiling_estimate` for the overall pipeline.
**Example**: Music tracks_track_name is classified as `structurally_limited` because list-format issues prevent matching.

### Probe 5: `source_attribution`
**Function**: `_probe_source_attribution()` (line 646)
**Input**: `evaluation_metrics`, `validation_fusion_testset`, `FUSED_OUTPUT_PATH` (reads `_fusion_metadata` column)
**Output**: Per-source exact/fuzzy match rates for each low-accuracy attribute (below 80%). Recommends resolver + trust ordering with attribute-type-aware defaults.
**How it works**: For each low-accuracy attribute, loads the fused CSV and validation set, maps fused rows to validation IDs via `_fusion_sources`, parses `_fusion_metadata` (Python repr format with JSON/ast.literal_eval/regex fallback for numpy types), extracts per-source values, and compares against validation expected values.
**Example (Music, label)**:
```
label (n=25):
  discogs: exact=100%, fuzzy=100%, non_null=100%
  musicbrainz: exact=0%, fuzzy=0%, non_null=0%
  -> recommended: prefer_higher_trust, trust_order=[discogs, musicbrainz]
```

### Probe 6: `null_patterns`
**Function**: `_probe_null_patterns()` (line 294)
**Input**: Reads fused CSV directly
**Output**: Columns with >50% null values in the fused output. Indicates potential blocking/matching failures or data sparsity.

### Probe 7: `correspondence_density`
**Function**: `_probe_correspondence_density()` (line 328)
**Input**: Correspondence CSV files from `CORRESPONDENCES_DIR`
**Output**: Per-pair correspondence stats: row count, 1:1 vs 1:many ratio, score distribution.

### Probe 8: `blocking_recall`
**Function**: `_probe_blocking_recall()` (line 380)
**Input**: `blocking_config`
**Output**: Candidate counts vs expectations. Flags pairs with fewer than `ROUTING_BLOCKING_CANDIDATE_MIN` (10) candidates.

### Probe 9: `fusion_size`
**Function**: `_probe_fusion_size()` (line 424)
**Input**: `fusion_size_estimate.json` + actual fused CSV row count
**Output**: Actual vs estimated row counts with severity rating (minor/major) and direction (over/under).

### Probe 10: `recent_mismatches`
**Function**: `_probe_recent_mismatches()` (line 83)
**Input**: `debug_fusion_eval.jsonl` (reads up to `MAX_EVENTS` events)
**Output**: Top 3 mismatch reasons and affected attributes. Keyword-boosted (only included when action plan mentions normalization terms).

### Probe 11: `directive_coverage`
**Function**: `_probe_directive_coverage()` (line 136)
**Input**: `normalization_directives` + dataset column lists
**Output**: Checks if normalization directive columns exist in each dataset. Keyword-boosted.

### Custom Probes
`_run_custom_probe()` (line 981) supports external probes via subprocess. Limited to `MAX_CUSTOM_PROBES` (2), timeout `CUSTOM_PROBE_TIMEOUT_SECONDS` (2.0s), workspace-scoped paths only.

---

## 6. The Scaffold System

**File**: `helpers/pipeline_scaffold.py`, ~315 lines.

After cycle 1 generates and successfully executes a pipeline, the scaffold system splits the code into frozen and mutable sections for efficient re-generation.

### How Splitting Works

`build_scaffold(pipeline_code)` (line 95) uses regex landmark detection:

**Frozen prefix** (everything before the mutable section):
- Imports
- Dataset loading (`load_csv`, `load_parquet`, `load_xml`)
- Blocking (`StandardBlocker`, `TokenBlocker`, etc.)
- Matching (`RuleBasedMatcher`, `MLBasedMatcher`)
- Correspondence saving (`.to_csv(...correspondences...)`, `pd.concat([...])`)

**Mutable section** (what the LLM regenerates):
- Post-clustering algorithm instantiation (`MaximumBipartiteMatching()`, etc.)
- Trust map definition (`trust_map = {...}`)
- Fusion strategy creation (`DataFusionStrategy(...)`)
- `add_attribute_fuser()` calls

**Frozen suffix** (everything after the mutable section):
- `DataFusionEngine(...)` instantiation
- `engine.run(...)` call
- `.to_csv(...)` output

Detection uses `_MUTABLE_START_PATTERNS` (line 32, 10 patterns) and `_SUFFIX_START_PATTERNS` (line 52, 4 patterns).

### Assembly

`assemble_pipeline(scaffold, new_mutable_code)` (line 177) splices the new mutable code between the frozen prefix and suffix.

### Import Detection

`needs_new_imports(new_mutable_code, frozen_prefix)` (line 243) checks if the new mutable code uses fusion/clustering symbols not imported in the frozen prefix. Returns import lines to inject.

`inject_imports(frozen_prefix, import_lines)` (line 288) appends missing imports after the last existing import line.

### Response Extraction

`extract_mutable_from_response(response_code, scaffold)` (line 214) handles three cases:
1. LLM returned only mutable code (no blocking/matching/loading) -- use as-is
2. LLM returned full pipeline despite instructions -- re-extract with `build_scaffold()`
3. Ambiguous -- fallback to full response

### Benefits
- Reduces token usage (~50% fewer tokens per iteration)
- Prevents accidental breakage of working blocking/matching code
- Focuses the LLM on what actually needs to change (fusion strategy)
- Ensures correspondence files are always saved (frozen in the scaffold)

---

## 7. Cross-Attempt Learning

### iteration_history

Built in `run_investigator_node()` (line 543 in investigator_orchestrator.py). Each entry contains:

```python
{
    "attempt": 2,
    "accuracy": 0.648,
    "delta": 0.0924,            # change from previous attempt
    "decision": "pipeline_adaption",
    "description": "Remaining errors are in fusion configuration...",
    "attribute_accuracies": {    # per-attribute accuracy snapshot
        "artist": 0.92,
        "name": 0.613,
        "release-country": 0.24,
        "genre": 0.48,
        ...
    },
    "previous_pipeline_snippet": "..."  # first 500 chars of previous code
}
```

### How the LLM Uses History

`build_iteration_history_section()` in `helpers/context_summarizer.py` formats the history for the pipeline prompt:
```
ITERATION HISTORY (3 attempts):
  Attempt 1: 55.6% [IMPROVING]
    Worst: release-country=0.0%, genre=0.16%, tracks_track_name=0.17%
    Decision: pipeline_adaption
  Attempt 2: 64.8% (+9.2%) [IMPROVING]
    Worst: release-country=0.24%, tracks_track_name=0.17%
    Decision: pipeline_adaption
  Attempt 3: 61.5% (-3.3%) [REGRESSING]
    Decision: pipeline_adaption
```

Each attempt shows IMPROVING/STAGNATING/REGRESSING classification based on delta.

### Stagnation Detection

`build_stagnation_analysis()` in `context_summarizer.py` checks the last `STAGNATION_WINDOW` (2) attempts. If the absolute accuracy change is below `STAGNATION_DELTA_THRESHOLD` (0.02), stagnation is detected. This is recorded in the investigator history and used by the investigation LLM to decide whether to try a different approach.

### Cross-Run Learning

`helpers/investigator_learning.py` (~275 lines) provides EMA gain tracking per routing decision, dataset signature matching, and drift detection. `learning_routing_signals()` produces signals that are orthogonal to the investigation -- they provide context about what worked in previous runs with similar datasets, but don't make decisions.

---

## 8. ID Alignment (eval_id Mechanism)

The fused output's `_id` column uses the first dataset's ID (e.g., `discogs_3`), but the validation set may use a different source's IDs (e.g., `mbrainz_974`). The eval_id guardrail bridges this gap.

### How It Works

1. **Detection**: `apply_pipeline_guardrails()` reads the validation set to detect ID prefixes. For example, if validation IDs start with `mbrainz_`, the prefix is `mbrainz_`.
2. **Injection**: Injects a `_extract_eval_id(row)` function into the pipeline code that:
   - Parses `_fusion_sources` (a serialized list of source record IDs)
   - Finds the source ID matching the validation prefix
   - Creates an `eval_id` column with that ID
3. **Evaluation alignment**: The evaluation guardrail patches `fused_id_column="_id"` to `fused_id_column="eval_id"` so the evaluation code uses the correct ID for matching against the validation set.

### Per Use Case

| Use Case | Validation ID Prefix | Example Validation ID | Example Fused `_id` | eval_id Maps To |
|----------|---------------------|-----------------------|---------------------|-----------------|
| **Music** | `mbrainz_` | `mbrainz_974` | `discogs_3` | `mbrainz_974` (from `_fusion_sources`) |
| **Games** | `metacritic_` | `metacritic_42` | `dbpedia_15` | `metacritic_42` (from `_fusion_sources`) |
| **Books** | ISBN-based | `0060006641` | `amazon_123` | `0060006641` (from `isbn_clean` column or `_fusion_sources`) |
| **Companies** | `forbes_` | `forbes_1` | `dbpedia_7` | `forbes_1` (from `_fusion_sources`) |
| **Restaurant** | Mixed | `fodors_1` | `zagats_3` | `fodors_1` (from `_fusion_sources`) |

The eval_id mechanism handles all prefixes detected in the validation set, not just the primary one. The injected function iterates through all source IDs in `_fusion_sources` and returns the first one matching any known validation prefix.

---

## 9. Regression Guard

**File**: `helpers/metrics.py:126`, function `assess_validation_regression()`.

Prevents accepting a pipeline that is worse than the current best. Applied in `process_evaluation_decision()` (`helpers/evaluation_decision.py:151`).

### How It Works

1. Computes overall and macro accuracy gains/drops between current and best metrics
2. Computes per-attribute drops and gains
3. Uses net-benefit calculation: total accuracy gained across all attributes vs lost
4. Applies rejection rules:

**Rejection conditions** (any triggers rejection):
- Overall regression >= `REGRESSION_MINOR` (0.02)
- Macro regression >= `REGRESSION_MINOR` (0.02)
- No meaningful gain AND (1+ catastrophic drops >= 0.20, OR 1+ severe drops >= 0.12, OR 2+ moderate drops >= 0.08)
- Even with meaningful gain: 2+ catastrophic drops without net-positive, OR 3+ catastrophic drops, OR any drop >= 0.40 without net-positive

**Macro fallback**: When `macro_accuracy` is missing (evaluation didn't produce per-attribute metrics), falls back to `overall_accuracy` for the macro comparison. This prevents false rejection of genuinely better pipelines.

**Comparison order** (`is_metrics_better()` at line 221): Overall accuracy first, then macro accuracy, then mean per-attribute accuracy across shared attributes.

### Concrete Example (Music)

Attempt 2: 64.8% (new best)
Attempt 3: 61.5% -> rejected (overall_drop=0.033, exceeds REGRESSION_MINOR=0.02). Best stays at 64.8%.

---

## 10. Results Summary

### Best results from actual runs (20260322):

| Use Case | Val Acc | Iterations | Key Finding |
|----------|---------|------------|-------------|
| Music | 64.8% | 3 | Source attribution probe drove resolver changes: musicbrainz for release-country/duration, discogs for label/genre |
| Books | 67.5% | 3 | Normalization regression on isbn_clean (leading zeros) detected and fixed; genres format mismatch required normalization |
| Restaurant | 88.4% | 2 | Hit quality gate (85%) early; simpler dataset with fewer sources |
| Games | ~75% | 4 | Large clusters (max_size=1075) required post-clustering |

### Music accuracy trajectory (from `investigator_history.jsonl`):
```
Attempt 1: 55.6% -> pipeline_adaption (fusion strategy wrong for 5 attributes)
Attempt 2: 64.8% (+9.2%) -> pipeline_adaption (further trust_map refinement)
Attempt 3: 61.5% (-3.3%) -> pipeline_adaption (regression, guardrails kept best at 64.8%)
```

### Books accuracy trajectory:
```
Attempt 1: 67.5% -> normalization_node (genres at 0% due to format mismatch)
Attempt 2: 61.3% -> normalization_node (isbn_clean regressed 100%->0%, leading zeros)
Attempt 3: 67.5% -> pipeline_adaption (isbn fixed, now targeting fusion resolvers)
```

---

## 11. Architecture Decisions and Trade-offs

### Why LLM-generated code (not hardcoded pipelines)?
Each dataset combination has unique characteristics (column names, data types, null patterns, source quality differences). A hardcoded pipeline cannot adapt. The LLM generates dataset-specific code that handles these variations, while guardrails enforce correctness constraints mechanically.

### Why mechanical guardrails over prompt instructions?
The LLM does not always follow prompt instructions (e.g., "use numeric_tolerance_match for publish_year" -- it sometimes generates exact_match anyway). Mechanical guardrails fire every time, regardless of what the LLM generates. Examples:
- Missing import detection: auto-adds `favour_sources` when used but not imported
- Trust map conflict resolution: injects per-attribute variables
- ML post-clustering: prevents RecursionError in fusion engine

### Why probes instead of hardcoded routing?
The old system used hardcoded scoring functions to decide what to fix. The probe system provides evidence that the investigation LLM interprets in context. This is more flexible -- the same probe output means different things for different datasets. The source attribution probe showing "discogs=100%, musicbrainz=0%" for labels leads to `prefer_higher_trust(discogs)`, but the same probe showing "all sources=50%" leads to `voting`.

### Why scaffold-based re-generation?
Without scaffolding, each iteration regenerates the entire pipeline (~3000 tokens). With scaffolding, only the mutable section (~500 tokens) is regenerated. This:
1. Saves ~50% tokens per iteration
2. Prevents the LLM from accidentally changing blocking thresholds or matching weights
3. Ensures correspondence files are always saved (frozen in the scaffold)

### Key design principle: LLM-first with mechanical safety
The agent leverages LLM intelligence for decisions. Deterministic code provides evidence; the LLM interprets and decides. But critical correctness properties (imports, trust maps, numeric coercion, ID alignment) are enforced mechanically by guardrails rather than relying on LLM compliance with prompt instructions. Only 3 hardcoded safety overrides exist after the LLM's investigation decision -- everything else is the LLM's call.
