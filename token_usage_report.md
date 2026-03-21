# Token Usage Investigation

## Scope

This report investigates why `agents/AdaptationPipeline_with_new_Cluster.ipynb` is much more expensive than the other pipeline notebooks, using the run artifacts in `agents/output/results/AdaptationPipeline*_node_activity.json` and the pipeline snapshots in `agents/output/results/*_pipelines.md`.


## Executive Summary

The extreme token usage is real, but it is concentrated in the `with_new_Cluster` **rule-based** runs, not in the ML runs.

- Latest `with_new_Cluster` ML run:
  - `token_complexity.global_tokens.total_tokens`: **143,373**
  - `summarization_tokens.total_tokens`: **98,509**
  - combined tracked tokens: **241,882**
- Latest `with_new_Cluster` rule-based run:
  - `token_complexity.global_tokens.total_tokens`: **1,145,538**
  - `summarization_tokens.total_tokens`: **95,789**
  - combined tracked tokens: **1,241,327**

The latest rule-based run is therefore:

- about **8.0x** larger than the latest ML run on node-call tokens alone
- about **5.1x** larger than the latest ML run if summarization tokens are included

The root cause is not normal ML complexity. The root cause is a **prompt growth loop** in the rule-based path:

1. rule-based matching produces much noisier clusters
2. cluster analysis generates very large diagnostic artifacts and matching-weight update history
3. later nodes serialize those artifacts back into new LLM prompts
4. the same large state is passed again into `pipeline_adaption`, `evaluation_reasoning`, and `save_results`

So the cost explosion comes from **LLM prompt bloat from accumulated JSON/state**, not from the pipeline execution itself.

## Cross-Pipeline Comparison

Median node-call token usage by notebook family:

| Notebook family | Mode | Runs | Median total tokens |
| --- | --- | ---: | ---: |
| `AdaptationPipeline` | rule-based | 4 | 53,080 |
| `AdaptationPipeline_blocking_matching_extension_Final_Blocker_Matcher` | rule-based | 4 | 86,755 |
| `AdaptationPipeline_blocking_matching_extension_Final_Reasoning` | rule-based | 4 | 125,879 |
| `AdaptationPipeline_blocking_matching_extension_Final_Reasoning` | ml | 4 | 142,291 |
| `AdaptationPipeline_with_new_Cluster` | ml | 4 | 197,895 |
| `AdaptationPipeline_with_new_Cluster` | rule-based | 4 | 811,732 |

Key takeaways:

- `with_new_Cluster` is the most expensive notebook family in both modes.
- The **extreme outlier** is `with_new_Cluster` in **rule-based** mode.
- The latest rule-based `with_new_Cluster` run (**1,145,538**) is:
  - about **21.6x** the median of the basic `AdaptationPipeline` rule-based runs
  - about **9.1x** the median of `Final_Reasoning` rule-based runs
- The ML `with_new_Cluster` runs are only moderately above the `Final_Reasoning` ML runs; they are not the main anomaly.

## Latest ML vs Rule-Based Run

Files compared:

- `agents/output/results/AdaptationPipeline_with_new_Cluster_ml_260318_01_node_activity.json`
- `agents/output/results/AdaptationPipeline_with_new_Cluster_rule_based_260318_01_node_activity.json`

### Overall

| Metric | ML | Rule-based |
| --- | ---: | ---: |
| node-call total tokens | 143,373 | 1,145,538 |
| summarization tokens | 98,509 | 95,789 |
| combined tracked tokens | 241,882 | 1,241,327 |
| node count | 25 | 31 |
| `pipeline_adaption` visits | 3 | 5 |
| `cluster_analysis` visits | 3 | 5 |
| `evaluation_reasoning` visits | 2 | 2 |

The summarization overhead is roughly flat across runs at about 95k to 107k tokens. That means the major spike is coming from the actual node prompts, not from the logger summaries.

### Per-node cumulative token usage

| Node | ML | Rule-based | Rule-based / ML |
| --- | ---: | ---: | ---: |
| `pipeline_adaption` | 37,927 | 469,900 | 12.4x |
| `cluster_analysis` | 2,595 | 245,401 | 94.6x |
| `evaluation_reasoning` | 27,939 | 259,920 | 9.3x |
| `save_results` | 19,570 | 131,549 | 6.7x |
| `evaluation_adaption` | 51,768 | 35,194 | 0.7x |
| `match_schemas` | 3,209 | 3,209 | 1.0x |
| `profile_data` | 365 | 365 | 1.0x |

This shows the problem is not the shared upstream nodes. It is concentrated in:

- `pipeline_adaption`
- `cluster_analysis`
- `evaluation_reasoning`
- `save_results`

### Per-call prompt growth

Latest ML prompt tokens by node visit:

- `pipeline_adaption`: **6,857 -> 9,755 -> 12,650**
- `cluster_analysis`: **1,646 -> 0 -> 0**
- `evaluation_reasoning`: **10,650 -> 12,983**
- `save_results`: **17,919**

Latest rule-based prompt tokens by node visit:

- `pipeline_adaption`: **5,301 -> 67,230 -> 128,024 -> 129,696 -> 131,368**
- `cluster_analysis`: **60,159 -> 60,310 -> 118,780 -> 0 -> 0**
- `evaluation_reasoning`: **128,999 -> 126,569**
- `save_results`: **129,762**

This is the clearest evidence of prompt bloat. In the bad rule-based run, later prompts are repeatedly carrying around roughly 120k to 130k input tokens.

## Why ML and Rule-Based Behave Differently

The pipeline snapshots show that both modes share the same high-level shape:

- same three datasets: `dbpedia`, `metacritic`, `sales`
- same three `EmbeddingBlocker` steps
- same fusion stage over combined correspondences

The matching stage differs:

- ML pipeline:
  - uses `FeatureExtractor`
  - uses `MLBasedMatcher`
  - trains classifiers with `GridSearchCV`
  - uses `MaximumBipartiteMatching`
- Rule-based pipeline:
  - uses `RuleBasedMatcher`
  - depends on explicit comparator weights and thresholds
  - repeatedly retunes those weights based on cluster analysis

Important point: the ML pipeline is computationally more complex at execution time, but that is **not** what drives LLM token usage here.

The ML path stays token-cheap because its matching quality is already strong:

- ML matching F1s in the latest run configs:
  - `dbpedia_sales`: **0.9973**
  - `metacritic_dbpedia`: **0.9944**
  - `metacritic_sales`: **1.0000**
- Rule-based matching F1s in the latest run configs:
  - `dbpedia_sales`: **0.8644**
  - `metacritic_dbpedia`: **0.8867**
  - `metacritic_sales`: **0.8845**

That quality gap changes the downstream cluster diagnostics dramatically.

### ML cluster behavior

The ML run’s `cluster_analysis` summaries describe healthy correspondence graphs:

- `metacritic_dbpedia`: **6691 clusters**, all size 2, `max_degree=1`, `ambiguous_ratio=0.0`
- `metacritic_sales`: same zero-ambiguity profile
- recommended action stays effectively `None`

Result:

- cluster analysis stays small
- no heavy update loop is triggered
- later prompts remain compact

### Rule-based cluster behavior

The rule-based run’s `cluster_analysis` summaries are very different:

- `correspondences_metacritic_dbpedia.csv` flagged unhealthy
- `max_cluster_size` reaches **1076**, later **1268**
- `one_to_many_ratio` reaches about **0.62** to **0.64**
- `ambiguous_ratio` reaches about **0.47** to **0.53**

Result:

- the notebook keeps generating matching-weight updates
- cluster analysis outputs large reports with deep diagnostics
- those reports are injected back into later prompts
- prompt size explodes

## Why The Rule-Based Run Blows Up So Hard

The notebook code explains the token pattern.

### 1. `cluster_analysis` builds large structured artifacts

In `agents/AdaptationPipeline_with_new_Cluster.ipynb`, the `cluster_analysis` node:

- runs `ClusterTester`
- reads and writes `matching_config.json`
- stores the latest cluster report in `cluster_analysis_result`
- stores rolling history in `cluster_analysis_history`
- adds `_before` snapshots
- adds `_comparison_prev`
- adds `_matching_config_update` with:
  - weight deltas
  - per-weight explanations
  - pair update notes
  - skipped pairs
  - update attempt counts

`ClusterTester` itself produces rich deep analysis objects in `agents/cluster_tester.py`, including:

- `large_clusters`
- `top_left_hubs`
- `top_right_hubs`
- `attribute_agreement`
- `dominant_attributes`
- `hub_explanations`
- `multi_match_entities`

These are useful diagnostics, but they are verbose.

Supporting evidence from the current generated artifacts:

- `agents/output/cluster-evaluation/cluster_analysis_report.json`: **382,436 bytes**
- `agents/output/cluster-evaluation/cluster_analysis_history.json`: **836,716 bytes**

That is the kind of payload later prompts are drawing from.

### 2. `cluster_analysis` feeds large JSON back into another LLM call

Still inside `cluster_analysis`, the notebook sends this payload to the LLM for matching-weight updates:

- latest cluster analysis report
- last 3 history entries
- full `matching_config`
- pair metrics
- update attempt metadata

That explains why `cluster_analysis` itself can consume:

- **245,401** cumulative tokens in the latest rule-based run
- versus only **2,595** in the latest ML run

### 3. `pipeline_adaption` serializes full config and full cluster analysis back into the prompt

`pipeline_adaption` injects into the system prompt:

- the full `matching_config`
- prior `evaluation_analysis`
- the full `cluster_analysis_result`

In the rule-based loop, those objects become very large after cluster-analysis-driven weight retuning. That is why `pipeline_adaption` jumps from **5,301** prompt tokens on the first visit to **67,230**, then to **128,024+** on later visits.

### 4. `evaluation_reasoning` serializes almost the whole world

`evaluation_reasoning` includes all of the following in one prompt:

- aggregate evaluation metrics
- first 50 debug fusion events
- full pipeline code
- full evaluation code
- full blocking config
- full matching config
- full cluster analysis report

That is why `evaluation_reasoning` costs **259,920** tokens in the latest rule-based run even though it is only visited twice.

### 5. `save_results` performs one more LLM summary over the full results JSON

`save_results` builds a large `results` object and then sends the entire JSON to the model to produce a markdown summary.

This is not the root cause, but it amplifies the total:

- latest ML `save_results`: **19,570** tokens
- latest rule-based `save_results`: **131,549** tokens

## Why This Is Not Just “More Iterations”

The bad rule-based run does have more node visits than the ML run, but visit count alone does not explain the scale.

Compare two rule-based `with_new_Cluster` runs:

- `260317_02`: **286,755** total tokens
- `260318_01`: **1,145,538** total tokens

Both have the same control-flow shape:

- 31 nodes
- 5 visits to `pipeline_adaption`
- 5 visits to `cluster_analysis`
- 2 visits to `evaluation_reasoning`

What changed is the **prompt size per visit**:

| Node | `260317_02` avg tokens/visit | `260318_01` avg tokens/visit |
| --- | ---: | ---: |
| `pipeline_adaption` | 17,657 | 93,980 |
| `cluster_analysis` | 5,747 | 49,080 |
| `evaluation_reasoning` | 35,671 | 129,960 |
| `save_results` | 42,234 | 131,549 |

So the extreme token usage is fundamentally a **state-size problem**, not a simple retry-count problem.

## Bottom Line

The `AdaptationPipeline_with_new_Cluster.ipynb` notebook becomes extremely expensive when the rule-based matcher produces poor cluster structure.

The sequence is:

1. weaker rule-based matching creates large ambiguous clusters
2. cluster analysis emits large deep-diagnostic JSON plus update history
3. later nodes paste that JSON back into prompts
4. prompt sizes jump into the 60k to 130k range
5. repeated iterations multiply the damage

The ML pipeline does not show the same blow-up because its matching quality is much better, its cluster analysis remains small, and it avoids the large feedback artifacts that dominate the rule-based prompts.

## Final Diagnosis

The extreme token usage is caused primarily by **recursive reuse of large diagnostic artifacts in LLM prompts**, especially in the rule-based `with_new_Cluster` runs. The biggest contributors are:

- `pipeline_adaption`
- `cluster_analysis`
- `evaluation_reasoning`
- `save_results`

The notebook’s token problem is therefore a **prompt construction / state accumulation issue**, triggered by poor rule-based matching quality and the resulting oversized cluster-analysis feedback loop.

## Use-Case Breakdown

Each of the four use cases appears once in each notebook family / mode combination available in `agents/output/results`, so there are **6 runs per use case** in the current artifact set:

- `AdaptationPipeline` rule-based
- `Final_Blocker_Matcher` rule-based
- `Final_Reasoning` ml
- `Final_Reasoning` rule-based
- `with_new_Cluster` ml
- `with_new_Cluster` rule-based

### All available artifacts, grouped by use case

| Use case | Runs | Sum of node-call tokens | Sum of summarization tokens | Combined tracked tokens | Average node-call tokens / run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Games | 6 | 1,643,293 | 429,132 | 2,072,425 | 273,882 |
| Books | 6 | 1,422,698 | 441,195 | 1,863,893 | 237,116 |
| Music | 6 | 1,410,879 | 451,653 | 1,862,532 | 235,147 |
| Restaurants | 6 | 1,123,343 | 494,343 | 1,617,686 | 187,224 |

Across the full result set:

- **Games** is the most expensive use case overall.
- **Books** and **Music** are very close to each other.
- **Restaurants** is the cheapest overall, even though it has the highest total summarization tokens. Its lower node-call cost keeps the total below the others.

### `AdaptationPipeline_with_new_Cluster` by use case

This notebook is where the use-case differences are largest.

| Use case | ML node-call tokens | Rule-based node-call tokens | ML + summary | Rule-based + summary |
| --- | ---: | ---: | ---: | ---: |
| Games | 143,373 | 1,145,538 | 241,882 | 1,241,327 |
| Books | 242,156 | 804,284 | 339,500 | 911,611 |
| Music | 153,633 | 819,179 | 243,154 | 913,019 |
| Restaurants | 333,582 | 286,755 | 435,491 | 388,770 |

The ranking inside `with_new_Cluster` is:

- Rule-based: `games` > `music` ~= `books` >> `restaurants`
- ML: `restaurants` > `books` > `music` ~= `games`

That difference is important:

- In **games**, **music**, and **books**, the rule-based loop is the main source of token blow-up.
- In **restaurants**, the ML run is actually more expensive than the rule-based run.

### Per-use-case observations

#### Games

Games is the highest-cost use case overall because the latest `with_new_Cluster` rule-based run is the single biggest outlier in the repository:

- node-call tokens: **1,145,538**
- combined tracked tokens: **1,241,327**

Its dominant nodes are:

- `pipeline_adaption`: **469,900**
- `evaluation_reasoning`: **259,920**
- `cluster_analysis`: **245,401**
- `save_results`: **131,549**

The rule-based games run also shows the worst cluster pathology in the inspected artifacts:

- `max_cluster_size` grows above **1000**
- `one_to_many_ratio` is about **0.62 to 0.64**
- `ambiguous_ratio` is about **0.47 to 0.53**

So for games, the cost is driven by severe cluster-analysis feedback, which then inflates later prompts.

#### Books

Books is the second-most-expensive use case overall by node-call tokens:

- sum of node-call tokens across all six runs: **1,422,698**

The main driver is again the `with_new_Cluster` rule-based run:

- node-call tokens: **804,284**
- combined tracked tokens: **911,611**

Its largest nodes are:

- `pipeline_adaption`: **496,305**
- `evaluation_reasoning`: **112,619**
- `cluster_analysis`: **95,358**

Books therefore follows the same general pattern as games, but the cluster-analysis loop does not become quite as extreme as in games.

#### Music

Music is very close to books overall:

- sum of node-call tokens across all six runs: **1,410,879**

Its main outlier is also the `with_new_Cluster` rule-based run:

- node-call tokens: **819,179**
- combined tracked tokens: **913,019**

Its largest nodes are:

- `pipeline_adaption`: **357,305**
- `evaluation_reasoning`: **171,859**
- `cluster_analysis`: **163,417**

Compared with books:

- music spends less on `pipeline_adaption`
- music spends more on `cluster_analysis` and `evaluation_reasoning`

So the music rule-based run is expensive for the same structural reason, but with more of the cost shifted into analysis/reasoning nodes.

#### Restaurants

Restaurants is the lowest-cost use case overall:

- sum of node-call tokens across all six runs: **1,123,343**

It is also the one use case that breaks the dominant pattern in `with_new_Cluster`:

- ML node-call tokens: **333,582**
- rule-based node-call tokens: **286,755**

The restaurant ML run is expensive mainly because `pipeline_adaption` is unusually large:

- `pipeline_adaption`: **187,452**
- `evaluation_adaption`: **72,822**
- `evaluation_reasoning`: **35,259**

The restaurant rule-based run is still iterative, but it does not explode the way games/books/music do:

- `pipeline_adaption`: **88,286**
- `evaluation_reasoning`: **71,341**
- `cluster_analysis`: **28,733**

The cluster summaries support this difference. The restaurant rule-based run does show some unhealthy behavior, but at a much smaller scale:

- `max_cluster_size` reaches **11**, not hundreds or thousands
- the problematic pair is mainly `uber_eats_small ↔ yelp_small`
- the failure is more localized, so the generated artifacts stay much smaller

That is why restaurants remains the cheapest use case overall.

### What these use-case differences mean

The use-case differences are not explained by one universal “ML is more expensive” or “rule-based is more expensive” rule.

What the artifacts actually show is:

- **When cluster-analysis artifacts stay small**, token usage stays in a manageable range, even with multiple iterations.
- **When a use case creates large ambiguous clusters**, the rule-based `with_new_Cluster` loop becomes extremely expensive because it keeps feeding expanded diagnostics back into later prompts.
- **Games** is the clearest example of this failure mode.
- **Books** and **Music** show the same pattern, but less severely.
- **Restaurants** is the exception because its cluster problems remain much more local and the rule-based loop never grows into the same prompt sizes.
