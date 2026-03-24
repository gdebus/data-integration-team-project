# Agent VIII (nblazek_helper) — Claude Code Context

Bridge iteration between the earlier notebook-based agents (I–VI) and the final Agent VII (nblazek_final branch). The core pipeline logic lives in a single monolithic Python file rather than being split across modules. This agent introduced the first investigation system, normalization node, helper file extraction, and fusion size monitoring, but many of these subsystems were later completely rewritten for Agent VII.

## Quick Start

```bash
cd agents/
source venv/bin/activate
# Set OPENAI_API_KEY in agents/.env
python run_companies_only.py
# Or use the pipeline_runner_forlogging.ipynb notebook
```

Python 3.12.9. Venv at `agents/venv/`.

---

## Architecture Overview

**LangGraph state machine** in `agents/AdaptationPipeline_nblazek_setcount_exec.py.py` (SimpleModelAgent class, ~3894 lines — entire agent in one file). The graph flows:

```
match_schemas → profile_data → normalization_node → run_blocking_tester
  → run_matching_tester → pipeline_adaption → execute_pipeline → evaluation_node
  → investigator_node → (loops back to normalization_node/pipeline_adaption
                          OR → human_review_export → sealed_final_test_evaluation → save_results)
```

### Key Difference from Agent VII

The investigator can only route to **3 targets** (normalization_node, pipeline_adaption, human_review_export). It **cannot** route back to run_blocking_tester or run_matching_tester. In Agent VII, the investigator can route to 5 targets including re-running blocking and matching.

### Graph Routing (Conditional Edges)

| After Node | Condition | Routes To |
|---|---|---|
| `pipeline_adaption` | `should_continue_research` — tool calls pending | `pipeline_adaption` (loop) or `execute_pipeline` |
| `execute_pipeline` | success/failure/retry | `evaluation_node` / `pipeline_adaption` / END |
| `evaluation_node` | success or failure | `investigator_node` / END |
| `investigator_node` | `state["investigator_decision"]` | `normalization_node` / `pipeline_adaption` / `human_review_export` / END |
| `human_review_export` | sealed eval active? | `sealed_final_test_evaluation` / `save_results` |

### State Object (SimpleModelAgentState)

TypedDict with ~50 fields. Key ones:
- `datasets`, `original_datasets`, `normalized_datasets` — file paths
- `entity_matching_testsets` — dict of (pair_tuple → testset_path)
- `fusion_testset`, `validation_fusion_testset` — evaluation targets
- `blocking_config`, `matching_config` — frozen configs from testers
- `integration_pipeline_code` — LLM-generated Python code string
- `evaluation_metrics`, `evaluation_metrics_for_adaptation` — accuracy dicts
- `auto_diagnostics` — mismatch reasons, ID alignment, size comparison
- `normalization_directives` — directive-based normalization instructions (NOT NormalizationSpec)
- `evaluation_analysis` — raw LLM reasoning text
- `evaluation_reasoning_brief` — parsed structured dict from LLM text
- `investigator_decision` — routing string (normalization_node / pipeline_adaption / human_review_export)

---

## Investigation System (Deterministic Weighted Scoring)

Unlike Agent VII which uses an LLM-driven multi-turn investigation agent, this agent uses a **deterministic weighted scoring formula** to decide routing.

### Investigation Flow (`helpers/investigator_orchestrator.py`, 655 lines)

```
1. integration_diagnostics (LLM call) → structured diagnostics report
2. evaluation_decision (deterministic) → metrics bookkeeping, regression guards
3. Normalization acceptance check → did prior normalization help?
4. Cross-run learning state → EMA gain tracking
5. Build action plan from diagnostics
6. Build fusion guidance
7. Run 4 probes (pre-computed evidence)
8. evaluation_reasoning (LLM call) → reasoning text + brief
9. _assess_normalization_with_llm (LLM call) → normalization assessment
10. compute_normalization_routing (deterministic scoring) → normalization vs pipeline_adaption
```

### The Routing Score Formula (`helpers/investigator_routing.py`, 128 lines)

The routing decision is made by a weighted score formula, not by the LLM:

```python
ROUTING_WEIGHTS = {
    "llm_base": 0.45,
    "llm_confidence": 0.35,
    "fallback": 0.30,
    "low_accuracy": 0.12,
    "missing_fused": 0.08,
    "learning_bias": 0.10,
    "stagnation_penalty": 0.14,
    "exploration_bonus": 0.06,
    "probe_normalization_pressure": 0.18,
    "objective_low_score": 0.15,
    "objective_worst_attribute_pressure": 0.12,
}
ROUTING_BASE_THRESHOLD = 0.72
```

If the weighted score exceeds the threshold (0.72), the agent routes to normalization. Otherwise it routes to pipeline_adaption. The scoring uses 11 signal components with hardcoded weights.

### Three Separate LLM Calls

Unlike Agent VII's single multi-turn investigation, this agent makes 3 separate LLM calls per investigation cycle:
1. **`integration_diagnostics()`** (~line 2536) — generates a structured JSON diagnostics report with `per_attribute_analysis`, `fusion_policy_recommendations`, `source_attribute_agreement`, `normalization_assessment`
2. **`evaluation_reasoning()`** (~line 3199) — produces free-text reasoning about what went wrong and what to try next
3. **`_assess_normalization_with_llm()`** (~line 1326) — separate normalization-specific assessment

These can produce contradictory recommendations since they don't share context.

### Only 4 Probes (`helpers/investigator_probe_runner.py`, 318 lines)

| Probe | Purpose |
|---|---|
| `reason_distribution` | Top mismatch reasons from debug logs |
| `worst_attributes` | Attributes with lowest accuracy |
| `recent_mismatches` | Mismatch sample text |
| `directive_coverage` | Normalization directive column availability |

Agent VII expanded this to 11 probes, adding: mismatch_sampler (classified examples), source_attribution (per-source match rates), correspondence_density, blocking_recall, fusion_size, null_patterns, and attribute_improvability.

### No Code Execution

The investigation LLM **cannot write and execute diagnostic Python**. It can only reason passively from the pre-computed evidence. Agent VII added multi-turn code execution capability.

---

## Normalization System (Directive-Based, Not PyDI NormalizationSpec)

### How It Works (`helpers/normalization_orchestrator.py`, 800 lines)

Instead of Agent VII's LLM-generated NormalizationSpec (a structured PyDI config), this agent uses:
1. **Directives** — a dict of `normalization_directives` with keys like `country_columns`, inferred from the validation set
2. **Heuristic inference** — `infer_validation_text_case_map()` and `infer_country_output_format_from_validation()` detect target formats from the validation set
3. **PyDI's older `normalize_dataset()` API** — not the newer `NormalizationSpec` + `transform_dataframe()` approach

The normalization node:
- Loads each source dataset and the validation set
- Infers which columns need case normalization and which country format to use
- Applies PyDI's `normalize_dataset()` with inferred config
- Falls back to original datasets if normalization fails

### Key Limitation

The normalization is less flexible than Agent VII's approach because it relies on heuristic format inference rather than LLM-generated per-column specs. It handles case and country format well but cannot handle complex transforms like `expand_scale_modifiers`, `stdnum_format`, or `date_format` conversion.

---

## Code Guardrails (Simpler Than Agent VII)

### `_apply_pipeline_guardrails()` (~line 1118, inside main agent file, ~150 lines)

Handles:
- **list_strategy validation** — fixes invalid strategies per comparator type
- **"concatenate" → "best_match" upgrade** — same as Agent VII
- **include_singletons enforcement** — forces `include_singletons=True`
- **Output path rewriting** — redirects hardcoded `output/` to run-scoped directory
- **Custom fuser signature patching** — adds `**kwargs` to custom fuser functions

Missing compared to Agent VII (~1362 lines):
- No fusion guidance enforcement (no per-attribute resolver patching)
- No confidence-based override strategy
- No per-attribute trust_map injection
- No eval_id injection
- No ML post-clustering safety
- No general import safety scan

### `_apply_evaluation_guardrails()` (~line 1460, ~40 lines)

Much simpler than Agent VII's version. Only handles:
- Import fallback for `list_normalization`
- Output path rewriting

Missing: sub-column stripping, numeric type coercion, list separator normalization.

---

## File Reference

### Main Agent File

| File | Lines | Purpose |
|------|-------|---------|
| `AdaptationPipeline_nblazek_setcount_exec.py.py` | 3894 | Entire agent: graph definition, all node methods, all prompts, guardrails, tools, state definition |

### Root-Level Modules

| File | Lines | Purpose |
|------|-------|---------|
| `_resolve_import.py` | 41 | Locates `agents/` root and adds to `sys.path` |
| `config.py` | 181 | All magic numbers, thresholds, timeouts (same structure as Agent VII) |
| `blocking_tester.py` | 1327 | Tests blocking strategies, picks best per dataset pair |
| `matching_tester.py` | 2057 | Tests matching strategies, evaluates comparator combinations |
| `cluster_tester.py` | 294 | Cluster health analysis (simpler than Agent VII's 648-line version) |
| `schema_matching_node.py` | 92 | Schema alignment using PyDI matchers |
| `fusion_size_monitor.py` | 520 | Estimates expected fusion output row count |
| `list_normalization.py` | 126 | Detects and normalizes list-like columns |
| `workflow_logging.py` | 2341 | WorkflowLogger class, structured JSON action records |
| `run_companies_only.py` | 76 | Entry point for companies use case |

### Helpers (`agents/helpers/`)

| File | Lines | Purpose |
|------|-------|---------|
| `investigator_orchestrator.py` | 655 | Main investigation loop with 3 LLM calls + deterministic routing |
| `investigator_routing.py` | 128 | Weighted score formula for normalization vs pipeline routing |
| `investigator_probe_runner.py` | 318 | 4 built-in probes (reason_distribution, worst_attributes, recent_mismatches, directive_coverage) |
| `investigator_learning.py` | 278 | Cross-run EMA gain tracking, dataset signature matching |
| `investigator_acceptance.py` | 213 | Normalization acceptance gate with rejection streak blocking |
| `investigation.py` | 91 | Deprecated helper for normalization issue detection |
| `investigation_helpers.py` | 91 | Keyword-based normalization detection (scans for "normaliz", "canonical", etc.) |
| `normalization_orchestrator.py` | 800 | Directive-based normalization using PyDI's older API |
| `normalization_policy.py` | 181 | Heuristic format inference from validation set |
| `code_guardrails.py` | 310 | Simpler guardrails (list_strategy, singletons, path rewriting) |
| `context_summarizer.py` | 661 | Compresses state into focused LLM context |
| `evaluation_decision.py` | 312 | Regression guards, metric bookkeeping, problem classification |
| `evaluation.py` | 199 | Auto diagnostics, ID alignment, mismatch analysis |
| `evaluation_helpers.py` | 158 | Legacy evaluation helpers (kept for compatibility) |
| `evaluation_orchestrator.py` | 47 | Evaluation execution wrapper |
| `evaluation_sanity.py` | 114 | Sanity checks on evaluation results |
| `error_classifier.py` | 104 | Classifies pipeline errors into categories |
| `metrics.py` | 249 | Metric extraction, regression assessment, comparison |
| `correspondence.py` | 209 | Correspondence file resolution and integrity checks |
| `correspondence_helpers.py` | 117 | Legacy correspondence helpers |
| `matching_validation.py` | 55 | Validates matching config structure |
| `pipeline_scaffold.py` | 315 | Frozen/mutable code splitting after cycle 1 |
| `results_writer.py` | 463 | Final run output and reasoning brief parsing |
| `script_runner.py` | 227 | Subprocess execution for pipeline and evaluation |
| `snapshots.py` | 177 | Per-cycle artifact snapshots for audit trail |
| `token_tracking.py` | 164 | Token usage and cost tracking per LLM call |
| `run_report.py` | 189 | End-of-run markdown report |
| `utils.py` | 46 | Small utilities |

---

## Cluster Tester (294 lines)

Only computes:
- Cluster size distribution health (small_cluster_ratio, large_cluster_ratio)
- One-to-many ratio (simpler than Agent VII, no ambiguity metrics)
- No score statistics (no IQR, percentiles, or low confidence ratio)

Can only recommend:
- `"None"` (healthy) or `"MaximumBipartiteMatching"` (unhealthy)
- Never considers StableMatching, GreedyOneToOne, HierarchicalClusterer, or ConnectedComponentClusterer
- Agent VII rates all 5 algorithms as recommended/viable/unsuitable

---

## Search Documentation Tool

**SearchDocumentationTool** (~line 184 of main agent file):
- Uses `OpenAIEmbeddings(model="text-embedding-3-large")`
- Queries Chroma vector DB at `agents/input/api_documentation/pydi_apidocs_vector_db/`
- `similarity_search(query, k=4)` — only 4 results (Agent VII uses k=8)
- Bound as LangChain tool — the LLM can call it during `pipeline_adaption` and `evaluation_adaption`
- No static API cheat-sheet (Agent VII injects a curated 76-line PYDI_API_CHEATSHEET into every prompt)
- The prompt says "you MUST use the search_documentation tool" if unsure about PyDI functions

---

## Key Differences from Agent VII (nblazek_final)

| Feature | Agent VIII (this branch) | Agent VII (nblazek_final) |
|---|---|---|
| **Architecture** | Single 3894-line file | Modular: pipeline_agent.py + prompts/ + helpers/ |
| **Investigation** | 3 LLM calls + deterministic weighted scoring | Single multi-turn LLM with code execution |
| **Routing targets** | 3 (normalization, pipeline, human_review) | 5 (+blocking_tester, +matching_tester) |
| **Probes** | 4 basic probes (318 lines) | 11 probes (1119 lines) including source_attribution, mismatch_sampler, attribute_improvability |
| **Normalization** | Directive-based, heuristic inference | LLM-generated PyDI NormalizationSpec per column |
| **Code guardrails** | 310 lines, basic fixes | 1362 lines, fusion guidance enforcement, eval_id injection, ML safety |
| **Cluster tester** | 294 lines, binary MBM-only | 648 lines, rates all 5 algorithms |
| **Search tool** | k=4, mandatory use, no cheat-sheet | k=8, optional use, authoritative static API reference |
| **Code execution** | Investigation LLM cannot run code | Multi-turn code execution in subprocess |
| **Prompt files** | Embedded in main file | Separate files in prompts/ directory |
| **Routing logic** | Hardcoded weights (11 components, threshold=0.72) | LLM decides, only 3 safety overrides |

---

## Configuration (`config.py`, 181 lines)

Same structure as Agent VII. Key values:
- `QUALITY_GATE_THRESHOLD`: 0.85
- `MATCHING_F1_GATE`: 0.65
- `BLOCKING_PC_THRESHOLD`: 0.9
- `PIPELINE_EXEC_TIMEOUT`: 3600
- `PIPELINE_EXEC_MAX_ATTEMPTS`: 3
- `MAX_INVESTIGATION_ATTEMPTS`: 4
- `GRAPH_RECURSION_LIMIT`: 200

---

## Output Structure

```
output/runs/YYYYMMDD_HHMMSS_<usecase>/
├── agent.log
├── blocking-evaluation/
├── matching-evaluation/
├── cluster-evaluation/
├── correspondences/
├── data_fusion/
├── pipeline_evaluation/
├── code/
├── human_review/
├── normalization/
├── pipeline/
├── profile/
├── snapshots/
├── schema-matching/
└── results/
```

---

## What This Agent Introduced (Relative to Agents I–VI)

1. **First investigation system** — 3-LLM-call pattern with weighted routing (Agents I–VI had no investigation)
2. **First normalization node** — directive-based transforms (Agents I–VI had no normalization)
3. **Helper file extraction** — began breaking the monolithic notebook into helper modules
4. **Fusion size monitoring** — estimated vs actual fusion row counts
5. **Regression guards** — prevented accepting pipelines that regressed from prior best
6. **Cross-run learning** — EMA gain tracking to learn which routing decisions work
7. **Pipeline scaffold** — frozen/mutable code splitting to reduce re-generation scope
8. **Run-scoped output** — each run gets its own timestamped directory
9. **Sealed final test** — held-out test set evaluation after pipeline is finalized

## What Was Later Rewritten for Agent VII

1. Investigation system → replaced by single LLM-driven multi-turn agent with code execution
2. Normalization → replaced by LLM-generated PyDI NormalizationSpec
3. Routing logic → replaced by LLM decision with minimal safety overrides
4. Probe system → expanded from 4 to 11 probes with classified mismatch samples and source attribution
5. Code guardrails → expanded from 310 to 1362 lines with fusion guidance enforcement
6. Cluster tester → rewritten to rate all 5 post-clustering algorithms
7. Monolithic file → split into modular pipeline_agent.py + prompts/ + helpers/
