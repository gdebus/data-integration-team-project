# LLM Project Introduction

Use this document as the first orientation pass before changing anything in this repository.

## What This Project Is

This repository is a research/engineering project for automated multi-source data integration with LLM support.

The core problem is:
- take 3 heterogeneous datasets about the same entity type
- align schemas
- choose blocking strategies
- choose matching strategies
- generate a PyDI integration pipeline
- run fusion
- evaluate the fused result against a gold standard
- iteratively repair the pipeline when evaluation shows weaknesses

The repository is centered on `PyDI` plus an LLM-driven orchestration layer.

## The Active Architecture

The most important current entrypoint is:
- [agents/AdaptationPipeline_nblazek_setcount_exec.py.py](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/AdaptationPipeline_nblazek_setcount_exec.py.py)

This file defines `SimpleModelAgent`, which builds a `LangGraph` workflow. The effective graph is:

1. `match_schemas`
2. `profile_data`
3. `normalization_node`
4. `run_blocking_tester`
5. `run_matching_tester`
6. `pipeline_adaption`
7. `execute_pipeline`
8. `evaluation_node`
9. `investigator_node`
10. `human_review_export`
11. optional `sealed_final_test_evaluation`
12. `save_results`

This is not just a code generator. It is an evaluate-and-repair loop.

## What The Agent Actually Produces

The agent generates executable Python files and artifacts under `agents/output/`, especially:
- `output/code/` for generated pipeline/evaluation/diagnostics code
- `output/blocking-evaluation/` for blocking configs and traces
- `output/matching-evaluation/` for matching configs and traces
- `output/data_fusion/` for fused CSV outputs
- `output/pipeline_evaluation/` for evaluation metrics and debug events
- `output/normalization/` for normalization attempts
- `output/results/` for run summaries and reports
- `output/runs/` for per-run snapshots

Treat most files under `agents/output/` as generated artifacts, not canonical source.

## Key Source Files

If you need to understand the current system, read these first:

- [agents/AdaptationPipeline_nblazek_setcount_exec.py.py](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/AdaptationPipeline_nblazek_setcount_exec.py.py)
  Main orchestration script. Defines the graph, prompts, execution loop, result saving, and evaluation routing.

- [agents/blocking_tester.py](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/blocking_tester.py)
  LLM-guided selection and evaluation of blocking strategies against pairwise gold standards.

- [agents/matching_tester.py](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/matching_tester.py)
  LLM-guided selection and evaluation of matching comparators/thresholds. Uses blocking output as an input constraint.

- [agents/schema_matching_node.py](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/schema_matching_node.py)
  Schema matching and aligned dataset export.

- [agents/list_normalization.py](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/list_normalization.py)
  Detection and normalization of list-like attributes. This is important because nested/list-valued fields are a recurring failure mode.

- [agents/fusion_size_monitor.py](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/fusion_size_monitor.py)
  Heuristics for estimating fused result size from blocking/matching outputs and comparing estimates with actual fusion size.

- [agents/helpers/normalization_orchestrator.py](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/helpers/normalization_orchestrator.py)
  Runs normalization as a controlled repair action, not as unconditional preprocessing.

- [agents/helpers/investigator_orchestrator.py](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/helpers/investigator_orchestrator.py)
  Decides whether to rerun normalization, adapt the pipeline, or stop for review.

- [agents/helpers/investigator_routing.py](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/helpers/investigator_routing.py)
  The routing logic between normalization and pipeline changes.

- [agents/helpers/code_guardrails.py](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/helpers/code_guardrails.py)
  Patches common LLM-generated code issues before execution.

- [agents/prompts/pipeline_prompt.py](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/prompts/pipeline_prompt.py)
- [agents/prompts/evaluation_prompt.py](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/prompts/evaluation_prompt.py)
- [agents/prompts/investigator_prompt.py](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/prompts/investigator_prompt.py)
  Compact but important prompt fragments that encode the project’s current safety rules.

## Important Behavioral Rules In This Repo

These rules are enforced repeatedly in code and prompts:

- Validation-set style is authoritative.
  Normalization should make fused output comparable to the validation/test representation, not to some abstract canonical form.

- IDs must be preserved exactly.
  Do not normalize, lowercase, or rewrite identifier columns.

- Blocking and matching are pre-tested inputs.
  Once `blocking_config` and `matching_config` are selected, generated pipelines should reuse them rather than invent new strategies.

- List-like attributes are dangerous.
  If a column can contain lists, comparators and fusion logic need explicit list handling.

- Normalization is not always beneficial.
  The investigator loop contains acceptance logic, ablation checks, and rollback behavior to avoid over-normalization.

- Evaluation drives repair.
  The agent improves pipelines based on measured fusion accuracy and debug mismatch signals, not only on execution success.

## Main Use Cases / Datasets

The repo contains several domains under `agents/input/datasets/`:

- `music`
  `discogs.xml`, `lastfm.xml`, `musicbrainz.xml`

- `restaurant`
  `kaggle_small.parquet`, `uber_eats_small.parquet`, `yelp_small.parquet`

- `games`
  `dbpedia.xml`, `metacritic.xml`, `sales.xml`

- `books`
  `amazon_small.parquet`, `goodreads_small.parquet`, `metabooks_small.parquet`

- `companies`
  `forbes.csv`, `dbpedia.xml`, `fullcontact.xml`

Each domain has pairwise test sets for blocking/matching and a fusion gold standard for evaluating final fused records.

## How To Read The Repo

Do not start with old notebooks only. Many notebooks are experiments, forks, or historical variants.

Recommended reading order:

1. [docs/llm_project_intro.md](/Users/onurcanmemis/Desktop/data-integration-team-project/docs/llm_project_intro.md)
2. [README.md](/Users/onurcanmemis/Desktop/data-integration-team-project/README.md)
3. [agents/AdaptationPipeline_nblazek_setcount_exec.py.py](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/AdaptationPipeline_nblazek_setcount_exec.py.py)
4. [agents/helpers/](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/helpers)
5. [agents/input/example_pipelines/](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/input/example_pipelines)
6. [agents/tests/](/Users/onurcanmemis/Desktop/data-integration-team-project/agents/tests)
7. `agents/output/results/` for concrete run behavior

Use notebooks mainly for historical context and experiments:
- `agents/AdaptationPipeline*.ipynb`
- `books-integration/*.ipynb`
- `restaurant-integration/*.ipynb`

## What Is Legacy / Experimental Versus Current

Current, most actionable code:
- `agents/*.py`
- `agents/helpers/*.py`
- `agents/prompts/*.py`
- `agents/tests/*.py`

Older or more experimental material:
- notebook variants in `agents/`
- standalone domain pipelines in `books-integration/` and `restaurant-integration/`
- `agents/ditto/`, which is a vendored Ditto entity-matching baseline and not the main orchestration path

## External Dependencies

The core dependencies are:
- `uma-pydi`
- `langgraph`
- `langchain`
- `langchain-openai` style model integrations used in the main script
- `chromadb`
- `pyarrow` / `fastparquet`
- standard ML stack for matcher training/evaluation

The active orchestration expects API keys via environment variables, especially `OPENAI_API_KEY`.

## Practical Warnings

- The top-level `README.md` is incomplete; do not rely on it as the full system description.
- Generated outputs are numerous and can look source-like. Check whether a file is under `agents/output/` before treating it as maintained code.
- Notebook variants may duplicate logic that now exists in `.py` modules.
- File naming is sometimes inconsistent; prefer tracing actual imports and runtime paths over naming conventions.
- The active entrypoint filename ends with `.py.py`; this is intentional in the current repo state.

## If You Need To Modify The System

Prefer these moves:

1. Change helper modules or prompt fragments when the behavior is systematic.
2. Change the main orchestration script only when graph flow, state handling, or execution routing must change.
3. Use tests in `agents/tests/` to infer intended behavior before changing routing or guardrails.
4. Inspect recent files in `agents/output/results/` to see how the agent is currently behaving on real runs.

Avoid these moves:

1. Treating generated pipeline code as the primary source of truth.
2. Hardcoding dataset-specific fixes into supposedly generic helpers.
3. Adding normalization that changes representation style away from the validation set.
4. Ignoring list-like column handling.

## Result Artifacts For LLM Analysis

Two generated artifact types are especially useful when asking another LLM to compare runs or architectures:

- `agents/output/results/*_node_activity.json`
- `agents/output/results/*_pipelines.md`

They serve different purposes and should be read together.

### What `node_activity.json` Contains

`node_activity.json` is a structured execution trace of one run.

At the top level it usually contains:
- `use_case`
- `LLM_model`
- `node_activity`
- `run_configs`
- `transition_stats`
- `time_complexity`
- `token_complexity`
- `summarization_tokens`
- `density`

The most important field is `node_activity`, which is an ordered list of node events. Each event typically contains:
- `node_index`
- `current_node`
- `next_node`
- `output_summary`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `estimated_cost_usd`
- `duration_seconds`
- `status`
- `error`

Interpretation:

- `current_node` / `next_node`
  Shows the actual graph path taken through the orchestration workflow.

- `output_summary`
  Natural-language summary of what that node did. This is high-signal for diagnosing why a run changed direction.

- `status` and `error`
  Identifies execution failures, retries, and repair cycles.

- token and duration fields
  Show cost and latency concentration by node.

The top-level aggregates are also important:

- `run_configs`
  Usually stores the selected blocking and matching configs used for that run. This is the easiest place to compare strategy choice across runs.

- `transition_stats`
  Summarizes how often the run loop traversed each edge. This is useful for understanding retry behavior and loop depth.

- `time_complexity`
  Aggregated duration by node.

- `token_complexity`
  Aggregated token usage by node and globally.

- `density`
  A compact estimate-vs-actual fusion size indicator. High mismatch can indicate overproduction, undercoverage, or weak estimate assumptions.

### What `*_pipelines.md` Contains

`*_pipelines.md` is a pipeline snapshot log.

It usually contains:
- the originating notebook name
- the matcher mode
- one or more `PIPELINE SNAPSHOT` sections

Each snapshot typically contains:
- `node_index`
- `node_name`
- `accuracy_score`
- a fenced Python code block

Interpretation:

- each snapshot is a concrete generated pipeline version
- the `accuracy_score` is the observed score associated with that snapshot
- the code block is the implementation artifact that can be diffed against other snapshots

In practice, these snapshots usually correspond to points where the generated pipeline became runnable or was materially revised.

### Difference Between The Two

Use this distinction:

- `node_activity.json` explains run behavior
- `*_pipelines.md` shows generated pipeline structure

Or more concretely:

- `node_activity.json` answers: "What happened during the run?"
- `*_pipelines.md` answers: "What code architecture did the agent produce at each major stage?"

### What They Are Good For

`node_activity.json` is good for:
- locating failures
- counting repair loops
- seeing whether normalization was re-entered
- seeing whether the investigator routed to normalization or pipeline changes
- comparing blocking/matching configs between runs
- comparing time/token cost by stage

`*_pipelines.md` is good for:
- comparing blocking implementations
- comparing comparator sets
- comparing matcher families and training logic
- comparing clusterer choice
- comparing fusion strategy and attribute fusers
- comparing singleton handling
- comparing code-level repairs across iterations

### What They Are Not Good For

Do not treat `node_activity.json` as a full semantic description of the pipeline. The summaries are helpful, but they are compressed.

Do not treat `*_pipelines.md` as the only evidence of run quality. A pipeline snapshot without the run trace can hide whether:
- it failed before full evaluation
- it required several repairs
- it was expensive to obtain
- it only worked on one dataset because of dataset-specific assumptions

### How To Compare Architectures With These Files

When guiding another LLM, make it compare runs in two layers:

1. orchestration architecture
2. generated pipeline architecture

#### 1. Compare Orchestration Architecture From `node_activity.json`

Ask the LLM to extract:
- full node sequence
- loop counts per node
- failure points
- investigator routing decisions
- normalization re-entry count
- total duration
- total tokens
- blocking config
- matching config

This reveals whether two runs differ mainly because:
- one had more repair loops
- one depended on normalization
- one spent more effort in `pipeline_adaption` or `investigator_node`
- one reused configs while another recomputed them

#### 2. Compare Generated Pipeline Architecture From `*_pipelines.md`

Ask the LLM to extract, per snapshot:
- data loading paths and whether normalized datasets are used
- blocking strategy per pair
- comparator types and preprocessing
- list handling strategy
- matcher type (`RuleBasedMatcher` vs `MLBasedMatcher`)
- model selection and training logic
- clustering strategy
- fusion strategy and per-attribute fusers
- evaluation-sensitive choices such as `include_singletons`
- any repair-specific additions like scalarized blocking columns or import fallbacks

This reveals whether two pipeline architectures differ in:
- retrieval/blocking design
- matching model family
- handling of list-valued fields
- fusion policy
- robustness guardrails

### Recommended Comparison Procedure For Another LLM

If you want another LLM to compare two architectures, instruct it to do this in order:

1. Read both `node_activity.json` files.
2. Summarize graph path, retries, failures, investigator decisions, and cost profile.
3. Read both `*_pipelines.md` files.
4. Extract the final successful snapshot and any immediately preceding failed/repair snapshot.
5. Compare pipeline architecture along these dimensions:
   - blocking
   - matching
   - clustering
   - fusion
   - normalization assumptions
   - robustness fixes
6. Separate:
   - run-process differences
   - code-architecture differences
7. Conclude which differences are likely causal for accuracy changes.

### Comparison Heuristics

Tell the LLM to prioritize these signals:

- blocking recall vs candidate explosion
  Use `run_configs.blocking_config` plus pipeline blocker code.

- matching quality vs complexity
  Compare `matching_config` F1 values and the actual comparator/model setup in the pipeline.

- list-valued attribute handling
  This repo repeatedly fails on list-like fields, so special treatment here is often architecture-critical.

- fusion coverage
  If `node_activity.json` mentions severe missing fused values or low retention ratio, check whether the pipeline disabled singletons or used overly restrictive clustering.

- repair specificity
  Prefer architectures that fix a precise failure mode over those that add broad transformations without evidence.

- cost-to-quality ratio
  A better architecture is not only a higher-accuracy one; it may also require fewer repair loops and fewer tokens.

### Important Edge Cases

Some result files are partial.

For example, a run can have:
- a short `node_activity.json` with only `match_schemas` and `profile_data`
- an empty `*_pipelines.md` with zero snapshots

That means the run did not progress far enough to produce a comparable pipeline artifact. Do not compare such a run as if it were a full architecture candidate.

### Suggested Prompt Template For Another LLM

Use a prompt like this:

```text
You are comparing two data-integration run artifacts from the same repository.

Inputs:
- run A node activity JSON
- run A pipeline snapshots markdown
- run B node activity JSON
- run B pipeline snapshots markdown

Tasks:
1. Summarize each run’s orchestration path:
   - visited nodes
   - retries / loops
   - failures
   - investigator decisions
   - normalization reruns
   - total time and token profile
2. Summarize each run’s generated pipeline architecture:
   - blocking per pair
   - comparators / features
   - matcher family
   - clustering
   - fusion strategy
   - singleton handling
   - robustness fixes
3. Distinguish process differences from code-architecture differences.
4. Identify the smallest set of architectural differences most likely to explain accuracy differences.
5. Rank the differences by likely impact.

Constraints:
- Do not rely only on accuracy_score.
- Use node_activity.json for process evidence.
- Use pipelines.md for code evidence.
- Call out incomplete runs explicitly.
```

## Short Mental Model

This project is an LLM-supervised PyDI pipeline generator with an internal quality-control loop.

Its core research idea is not just “generate a fusion pipeline,” but:
- test blocking
- test matching
- generate pipeline code
- execute it
- evaluate it
- diagnose failures
- selectively normalize or adapt
- keep the best-performing version
