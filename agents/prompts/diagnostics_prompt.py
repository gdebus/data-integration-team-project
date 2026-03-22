DIAGNOSTICS_SYSTEM_PROMPT = """
You are generating a standalone Python diagnostics script for a data-integration pipeline.
Write ONLY executable Python code.

Hard requirements:
- Be dataset-agnostic (no dataset-specific hardcoding).
- The diagnostics script must be generated from reasoning and execute its own analysis logic (no placeholder output).
- Do NOT read from stdin (no `sys.stdin.read()` / `input()`); rely only on discovered files and constants in the script.
- Read available artifacts and detect integration issues robustly.
- Focus on actionable findings: coverage gaps, over-clustering, noisy list fusion, and size-estimate drift.
- Explicitly detect empty pairwise correspondence files and treat them as HIGH severity if matching config expected that pair.
- Explicitly detect case-only mismatch patterns from debug_fusion_eval events.
- Perform attribute-level source agreement analysis:
  * load fused output and source datasets
  * compare fused attribute values against source attribute values (for aligned rows where possible)
  * compute per-attribute/per-source agreement metrics (support_count, exact_ratio and/or similarity ratio)
  * identify attributes where one source strongly dominates agreement
- From that analysis, generate machine-usable fusion policy suggestions (for example trust-based source preference for specific attributes).
- Rank suggested improvements by expected impact and confidence, with concise evidence.

- Explicitly check normalization-sensitive issues and report them as findings/recommendations:

  * country canonicalization mismatches (e.g., long-form vs short-form country names)

  * list-format inconsistencies (JSON-list strings vs scalar strings vs malformed list encodings)
- Gracefully handle missing files (do not crash; record warnings).
- Any index-based row sampling must be index-safe:
  * do NOT assume sampled labels are present in current dataframe index.
  * avoid unsafe patterns like df = df.loc[take] unless pre-validated.
  * prefer reset_index(drop=True)+iloc for positional samples, or filter labels to existing index before loc.
- Always guard column access:
  * before `df[col]`, ensure `col in df.columns`.
  * if column is missing, record a warning and continue instead of raising KeyError.
- Output TWO files:
  1) output/pipeline_evaluation/integration_diagnostics.json
  2) output/pipeline_evaluation/integration_diagnostics.md
- JSON must contain at least:
  - summary
  - findings (list ordered by severity)
  - recommendations (list, concrete and generic)
  - source_attribute_agreement (attribute -> source metrics)
  - fusion_policy_recommendations (list of suggested strategy/trust actions with evidence)
  - each recommendation should include: action, target_attributes, evidence_summary, expected_impact, confidence

  - recommendation fields should include machine-usable hints when possible (e.g., target columns, suggested PyDI functions)
  - evidence (counts/ratios/paths)
  - created_at
- Keep dependencies minimal (stdlib + pandas).
""".strip()

HUMAN_REVIEW_SYSTEM_PROMPT = """
You are generating a standalone Python script that creates FINAL human-review outputs for a data-fusion run.
Write ONLY executable Python code.

Hard requirements:
- Be dataset-agnostic (no hardcoded dataset-specific columns or IDs).
- Use only standard library + pandas (optional: PyDI.io if available).
- Read fused output from output/data_fusion/fusion_data.csv.
- Load source datasets from the provided dataset paths and detect ID columns robustly.
- Parse `_fusion_sources` robustly (JSON/list/string/empty) and map each source ID back to source rows.
- Produce reviewer-friendly outputs:
  1) output/human_review/fused_review_table.csv
     - one row per fused entity in WIDE format
     - for EVERY attribute listed in context_payload.review_attributes, create EXACTLY these columns:
       <attribute>_test, <attribute>_fused, <attribute>_source_1, <attribute>_source_2, <attribute>_source_3
     - keep the attribute text as-is when forming column names (do not rename/sanitize)
     - if testset value is unavailable, leave <attribute>_test as empty string
     - if fewer than 3 sources are available, leave missing <attribute>_source_<n> cells empty
     - this wide-table schema is mandatory and must exist even when values are missing
  2) output/human_review/source_lineage_long.csv
     - long table: fused_id, source_id, source_dataset, source attribute values, fused attribute values
  3) output/human_review/fusion_vs_testset_diff.csv
     - if fusion_testset exists, compare overlapping columns against mapped fused rows and record per-attribute diffs
     - if unavailable, still create an empty CSV with columns and a note in summary
  4) output/human_review/human_review_summary.json
  5) output/human_review/human_review_summary.md
- Summary JSON must include: summary, file_paths, counts, warnings, created_at.
- Gracefully handle missing files and malformed rows; do not crash.
""".strip()
