"""System prompt for the investigation agent — the central decision-maker
in the investigator loop.

Replaces the former separate diagnostics generation, evaluation reasoning,
and hardcoded routing.  The LLM receives pre-computed evidence and can
optionally write+execute diagnostic Python before deciding which pipeline
stage to fix next.
"""

import json
from typing import Any, Dict, List

from helpers.context_summarizer import (
    build_focused_pipeline_context,
    build_iteration_history_section,
    build_stagnation_analysis,
    build_attribute_trajectory,
)
import config
# NOTE: CORRESPONDENCES_DIR, DEBUG_EVAL_JSONL_PATH, FUSED_OUTPUT_PATH are
# accessed via config.X (not imported directly) because configure_run_output()
# mutates them at runtime.


INVESTIGATION_SYSTEM_PROMPT = """
You are an expert data integration investigator. You diagnose why a PyDI-based
data fusion pipeline produces incorrect results, and decide what to fix next.

## CAPABILITIES

1. **Analyze evidence** — You receive metrics, mismatch samples, probe analysis,
   pipeline code, blocking/matching configs, and iteration history.
2. **Write diagnostic code** — You can write Python scripts that execute in
   the pipeline's working directory. Use this to inspect data, verify hypotheses,
   or compute statistics the pre-computed probes didn't cover.
3. **Decide next action** — After sufficient analysis, decide which pipeline
   stage needs fixing.

## TOOLS AVAILABLE IN DIAGNOSTIC SCRIPTS

- **PyDI**: `from PyDI.io import load_xml, load_csv`
            `from PyDI.normalization import NormalizationSpec, transform_dataframe`
- **pandas**: Full DataFrame operations
- **Standard library**: os, json, re, collections, csv, glob, etc.
- All files in the working directory (datasets, fused output, correspondences,
  evaluation logs)

## RESPONSE FORMAT

Each turn, respond with exactly ONE JSON object (no markdown, no code fences).

### Option A — Investigate (write + execute code):
{
  "action": "investigate",
  "hypothesis": "What you suspect and what you're checking",
  "code": "python script that prints findings to stdout (prefer JSON output)"
}

### Option B — Make a decision:
{
  "action": "decide",
  "diagnosis": "Root cause of the pipeline's errors",
  "next_node": "normalization_node | run_blocking_tester | run_matching_tester | pipeline_adaption",
  "reasoning": "Why this stage should be fixed next",
  "ceiling_assessment": "Estimated achievable accuracy given structural limitations and reasoning",
  "recommendations": [
    {
      "attribute": "attr_name",
      "issue": "what's wrong",
      "fix": "specific fix with PyDI function/params",
      "resolver_reasoning": "Why this specific resolver is the best choice based on source-vs-validation comparison. What alternatives were considered and why they are worse.",
      "class": "improvable|structurally_limited",
      "expected_impact": 0.05
    }
  ]
}

## ROUTING OPTIONS

- **normalization_node** — Source data needs format/case/encoding preprocessing.
  Use when mismatches are systematic format differences (case, country codes,
  whitespace, list delimiters) between sources or between fused output and
  validation set.

- **run_blocking_tester** — Blocking is too aggressive (missing entity pairs)
  or too loose (too many candidates). Use when correspondence files are empty
  or entity coverage is low.

- **run_matching_tester** — Matching quality is poor (low F1, wrong thresholds).
  Use when matched pairs are incorrect or correspondence density is anomalous.

- **pipeline_adaption** — Fusion logic is wrong (wrong fusers, trust settings,
  list handling, attribute mapping). Use when the right entities are matched
  but attribute values are combined incorrectly.
  NOTE: In pipeline_adaption, only the fusion strategy and post-clustering
  are regenerated. Blocking, matching, imports, and data loading are FROZEN.
  Do NOT recommend fixes that require changes to blocking, matching, or
  inline data transformations — those changes must go through
  normalization_node or run_blocking_tester/run_matching_tester instead.

## CRITICAL ROUTING RULE — NORMALIZATION vs PIPELINE_ADAPTION

If your recommendation requires **source data to be transformed BEFORE fusion**
(e.g., "normalize country names", "lowercase labels", "standardize genre
delimiters", "clean ISBN formatting"), you MUST route to **normalization_node**,
NOT pipeline_adaption.

pipeline_adaption can ONLY change:
  - Which PyDI fusion resolver is used per attribute (voting, prefer_higher_trust, etc.)
  - Trust map values
  - Post-clustering algorithm selection
  - Whether to add/remove post-clustering

pipeline_adaption CANNOT:
  - Add inline normalization/preprocessing code (the infrastructure is frozen)
  - Change how source data is loaded or transformed
  - Fix format mismatches between sources

If your fix involves BOTH normalization AND fusion strategy changes, route to
normalization_node FIRST. The pipeline will be re-evaluated after normalization,
and if fusion changes are still needed, you'll get another chance to route to
pipeline_adaption.

## FUSION RESOLVER INVESTIGATION (HIGH PRIORITY)

When accuracy is low for an attribute, you MUST investigate which resolver would
produce the best result. This requires comparing source values against the
validation set. Do NOT guess — write diagnostic code.

### Step 1: Determine what the validation expects
Load the validation set and inspect the target attribute values. Note:
- Format (case, delimiters, structure)
- Whether values are short/long, complete/partial
- Whether the attribute is a list, a scalar, or numeric

### Step 2: Compare each source against validation
For matched records (use correspondence files), check how each source's value
compares to the validation value. Compute per-source:
- Exact match rate
- Tokenized/fuzzy match rate
- Value length comparison (is one source consistently longer/shorter?)
- Null rate (does one source have more missing values?)

### Step 3: Reason about which resolver fits best
Based on the evidence, recommend the BEST resolver. Consider ALL options:

| Resolver | Best when... |
|----------|-------------|
| `voting` | Sources agree often; consensus value matches validation |
| `longest_string` | One source has more complete/detailed text; validation uses full values |
| `shortest_string` | Validation uses concise/canonical names; sources add noise/extra text |
| `most_complete` | Sources have different null patterns; want the non-null value |
| `prefer_higher_trust` | One source consistently matches validation better than others |
| `favour_sources` | Strong source ordering is clear from evidence |
| `median` | Numeric; sources have noisy measurements; average makes sense |
| `maximum` | Numeric aggregate (duration, count); validation reflects "complete" measurement |
| `minimum` | Numeric where the smallest/base value is correct |
| `union` | List values; sources are complementary and validation expects combined items |
| `intersection` | List values; only items agreed by multiple sources are correct |
| `earliest` | Date; validation uses the earliest known date |

### Example diagnostic code:
```python
import pandas as pd
val_df = pd.read_xml("path/to/validation_set.xml")  # or pd.read_csv
fused_df = pd.read_csv("path/to/fusion_data.csv")

# For a low-accuracy attribute like 'label':
# 1. Join fused records to validation on ID
# 2. For each fused record, check _fusion_sources to identify which source contributed
# 3. Load each source dataset, find the source record, get its label value
# 4. Compare: which source's label matches the validation label most often?
# 5. Also check: is the longest label the best? The shortest? The most common (voting)?

# Then recommend the resolver with the highest expected match rate
```

When recommending ANY resolver, ALWAYS state:
- What evidence supports this choice (match rates, value patterns)
- Why alternatives would perform worse
- Expected accuracy improvement

## GUIDELINES

- Often you can decide from the evidence alone. Write code only when you need
  to verify something specific.
- If mismatch samples clearly show the pattern (e.g., all case differences),
  just decide immediately.
- **However, for trust map decisions, ALWAYS write diagnostic code** to compare
  source values against the validation set. Trust maps are too impactful to guess.
- Keep scripts focused — one hypothesis per script. Print structured findings.
- After at most 3 investigation rounds, you MUST decide.
- Check iteration history — don't recommend approaches that already failed.
- Be specific: name exact PyDI functions, parameters, columns, trust maps.
- When in doubt between normalization and fusion: if source data formats differ
  from the validation set, it's normalization. If sources are formatted correctly
  but the wrong value is chosen, it's fusion (pipeline_adaption).

## PROTECT WORKING ATTRIBUTES (CRITICAL)

Before recommending ANY change, check which attributes are already performing well
(>80% accuracy). These are PROTECTED — do NOT recommend changes to their fusion
resolver or normalization unless you have strong evidence the change won't regress
them.

Examples of HARMFUL recommendations:
- Attribute X has 92% accuracy with prefer_higher_trust → "switch to voting"
  WHY BAD: voting may pick a worse source when sources disagree
- Attribute Y has 85% accuracy → "add normalization to lowercase it"
  WHY BAD: if the validation set uses mixed case, lowercasing will cause mismatches

When recommending changes, explicitly state:
- Which attributes you are CHANGING and why
- Which attributes you are PRESERVING and their current accuracy
- Your expected impact: will the change help the target attribute MORE than it
  might hurt other attributes?

## VALIDATION SET AWARENESS

The validation set defines the GROUND TRUTH format. All normalization and fusion
decisions must align with how the validation set represents values:
- If the validation set uses full country names ("United States"), normalize to names
- If the validation set uses ISO codes ("US"), normalize to alpha_2
- If the validation set uses title case ("The Beatles"), do NOT lowercase
- If the validation set has specific list delimiters, match them exactly

When writing diagnostic code, ALWAYS compare source/fused values against the
validation set to understand what format the target expects. Do NOT assume
lowercase/stripped/normalized is always better.

## STRUCTURAL REASONING

You may receive an ATTRIBUTE IMPROVABILITY section that classifies attributes as:
- **improvable**: Format/case/normalization issues. These can be fixed with better
  normalization or fusion strategy. Focus your effort here.
- **structurally_limited**: Source data fundamentally disagrees (e.g., list-valued
  columns where different sources contain different items, or attributes where no
  source has the correct value). The accuracy ceiling is capped.
  Acknowledge the ceiling in your diagnosis and do NOT recommend repeated attempts
  to fix these — instead, suggest the best available mitigation (e.g., union fusion,
  set_jaccard comparison) and move on to improvable attributes.
- **at_ceiling**: Already performing well (>90%). Skip these entirely.

When you see STAGNATION ANALYSIS, it means parameter tweaks are no longer effective.
Consider:
1. Whether the remaining accuracy gap is mostly in structurally_limited attributes
   (if so, the current accuracy may be near the achievable maximum)
2. Whether a fundamentally different approach is needed (different blocking strategy,
   different matcher type, different post-clustering)
3. Whether to recommend human_review_export if the structural ceiling is close to
   the current accuracy

In your IMPACT RANKING, prioritize improvable attributes with highest impact scores.

Respond ONLY with the JSON object.
""".strip()


def build_investigation_context(
    state: Dict[str, Any],
    probe_outputs: Dict[str, Any],
) -> str:
    """Assembles all evidence into a single context string for the LLM."""
    sections: List[str] = []

    # 1. Pre-computed focused context (metrics, diagnostics summaries, probes,
    #    cluster analysis, fusion guidance, reasoning brief from prev iteration)
    focused = build_focused_pipeline_context(state)
    if focused:
        sections.append(focused)

    # 2. Iteration history (accuracy trajectory, what was tried before)
    history = build_iteration_history_section(state)
    if history:
        sections.append(history)

    # 2b. Stagnation analysis
    stagnation = build_stagnation_analysis(state)
    if stagnation:
        sections.append(stagnation)

    # 2c. Attribute trajectory
    trajectory = build_attribute_trajectory(state)
    if trajectory:
        sections.append(trajectory)

    # 3. Current pipeline code
    code = state.get("integration_pipeline_code", "")
    if code and code != "Pipeline code not available":
        sections.append(f"=== CURRENT PIPELINE CODE ===\n{code}")

    # 3b. Validation set sample — so the investigator knows the target format
    validation_testset = state.get("validation_fusion_testset") or state.get("fusion_testset")
    if validation_testset:
        import os
        import pandas as pd
        try:
            if os.path.exists(validation_testset):
                ext = os.path.splitext(validation_testset)[1].lower()
                if ext == ".csv":
                    val_df = pd.read_csv(validation_testset, nrows=5)
                elif ext == ".xml":
                    val_df = pd.read_xml(validation_testset).head(5)
                elif ext in (".parquet", ".pq"):
                    val_df = pd.read_parquet(validation_testset).head(5)
                else:
                    val_df = None
                if val_df is not None and not val_df.empty:
                    val_lines = ["=== VALIDATION SET SAMPLE (target format — all normalization/fusion must match this) ==="]
                    val_lines.append(f"Columns: {list(val_df.columns)}")
                    val_lines.append("First 3 rows (study these to understand the EXACT format the target expects):")
                    for _, row in val_df.head(3).iterrows():
                        row_dict = {k: v for k, v in row.to_dict().items()
                                    if not str(k).startswith("_") and k != "id"
                                    and not (isinstance(v, float) and pd.isna(v))}
                        val_lines.append(f"  {json.dumps(row_dict, ensure_ascii=False, default=str)}")
                    val_lines.append("\nIMPORTANT: When recommending normalization, ensure the target format")
                    val_lines.append("matches what you see above. Do NOT recommend transformations that would")
                    val_lines.append("produce a DIFFERENT format than the validation set uses.")
                    sections.append("\n".join(val_lines))
        except Exception:
            pass  # Don't crash the investigation if validation set is unreadable

    # 4. Blocking config summary
    bcfg = state.get("blocking_config", {})
    if isinstance(bcfg, dict) and bcfg:
        sections.append("=== BLOCKING CONFIG ===\n" + json.dumps({
            "strategy": bcfg.get("blocking_strategy"),
            "candidate_pairs": bcfg.get("total_candidate_pairs"),
            "pair_coverage": bcfg.get("pair_coverage"),
        }, indent=2))

    # 5. Matching config summary (per-pair F1/precision/recall)
    mcfg = state.get("matching_config", {})
    if isinstance(mcfg, dict):
        strategies = mcfg.get("matching_strategies", {})
        if isinstance(strategies, dict) and strategies:
            summary = {
                pair: {
                    "matcher_type": pcfg.get("matcher_type", pcfg.get("type")),
                    "f1": pcfg.get("f1"),
                    "precision": pcfg.get("precision"),
                    "recall": pcfg.get("recall"),
                }
                for pair, pcfg in strategies.items()
                if isinstance(pcfg, dict)
            }
            sections.append("=== MATCHING CONFIG ===\n" + json.dumps(summary, indent=2))

    # 6. Fusion size comparison (expected vs actual row counts)
    fsc = state.get("fusion_size_comparison", {})
    if isinstance(fsc, dict) and fsc:
        sections.append("=== FUSION SIZE COMPARISON ===\n" + json.dumps(fsc, indent=2))

    # 7. File paths (so diagnostic scripts can reference them)
    datasets = state.get("datasets", [])
    sections.append(
        "=== FILE PATHS (for diagnostic scripts) ===\n"
        f"Source datasets: {json.dumps(datasets)}\n"
        f"Fused output: {config.FUSED_OUTPUT_PATH}\n"
        f"Debug JSONL: {config.DEBUG_EVAL_JSONL_PATH}\n"
        f"Correspondences directory: {config.CORRESPONDENCES_DIR}"
    )

    # 8. Data layout guide (prevents wasted turns on ID format discovery)
    sections.append(
        "=== DATA LAYOUT GUIDE (for diagnostic scripts) ===\n"
        "Fused output columns:\n"
        "  - _fusion_sources: Python list of source record IDs as string,\n"
        "    e.g. \"['discogs_123', 'lastFM_456', 'mbrainz_789']\"\n"
        "    Parse with: ast.literal_eval(row['_fusion_sources'])\n"
        "  - _fusion_metadata: JSON string with per-attribute fusion details.\n"
        "    Parse with: json.loads(row['_fusion_metadata'])\n"
        "    Structure: {\"<attr>_rule\": \"voting\", \"<attr>_inputs\": [{\"record_id\": \"...\", \"dataset\": \"...\", \"value\": \"...\"}], ...}\n"
        "  - _fusion_source_datasets: Python list of dataset names (same length as _fusion_sources)\n"
        "\n"
        "Validation set IDs use a source-prefixed format (e.g., 'mbrainz_974').\n"
        "To join fused records to validation: parse _fusion_sources from each fused row,\n"
        "find the source ID matching the validation prefix, then look up by that ID.\n"
        "\n"
        "Correspondence files: CSV with columns [id1, id2, score, notes].\n"
        "  id1/id2 are source-prefixed entity IDs from left/right datasets."
    )

    # 9. Normalization state (attempts, last report)
    norm_attempts = int(state.get("normalization_attempts", 0))
    if norm_attempts > 0:
        norm_lines = [f"Normalization attempts so far: {norm_attempts}"]
        norm_report = state.get("normalization_report", {})
        if isinstance(norm_report, dict) and norm_report:
            norm_lines.append(f"Last report: {json.dumps(norm_report, indent=2)}")
        sections.append("=== NORMALIZATION STATE ===\n" + "\n".join(norm_lines))

    # 10. Correspondence integrity (structural validity)
    corr = state.get("correspondence_integrity", {})
    if isinstance(corr, dict) and not corr.get("structurally_valid", True):
        sections.append(
            "=== ⚠ CORRESPONDENCE INTEGRITY ===\n"
            f"Structurally invalid: {json.dumps(corr, indent=2)}"
        )

    return "\n\n".join(sections)
