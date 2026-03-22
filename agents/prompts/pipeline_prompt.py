import json

from helpers.context_summarizer import summarize_diagnostics_for_llm
from prompts.pydi_api_reference import PYDI_API_CHEATSHEET


PIPELINE_NORMALIZATION_RULES_BLOCK = """
IMPORTANT NORMALIZATION RULES:
- Validation set style drives normalization. Do NOT blindly lowercase text globally.
- Keep IDs unchanged (case and content must be preserved exactly).
- If you convert a column to list-like values, any comparator on that column MUST set list_strategy.
- If StringComparator is used on potentially list-like columns (artist/name/tracks/etc.), set list_strategy explicitly.
- Prefer normalization that aligns fused output with validation representation, not a random canonical style.
- You MAY apply targeted inline normalization in the pipeline code (e.g., str.strip(), format conversion) when it directly addresses mismatch patterns visible in the probe data. This is preferred over waiting for a separate normalization pass when the fix is simple and attribute-specific.

LIST HANDLING RULES (CRITICAL):
- After list normalization, columns like tracks_track_name, tracks_track_position, tracks_track_duration contain actual Python lists (e.g., ["track1", "track2"]).
- NEVER apply string cleaning operations (strip, lower, isin, notna, etc.) to list-valued columns. These operations crash with "truth value of an array is ambiguous".
- NEVER call pd.isna(x) or pd.notna(x) directly on a value that might be a list. Use isinstance(x, (list, tuple, set)) check first, or use a safe wrapper.
- NEVER include list-valued columns in your string_columns cleaning loop. Only clean scalar string columns.
- NEVER pass DataFrames with list-valued cells directly to EmbeddingBlocker. Flatten list columns to strings first.
- When writing a lower_strip or preprocess function, handle list inputs: `if isinstance(x, (list, tuple, set)): return " ".join(str(v).lower().strip() for v in x if pd.notna(v))`
- Recommended list_strategy values per comparator type:
  * StringComparator: "best_match", "set_jaccard", "set_overlap" (avoid "concatenate" — it merges all elements into one string, producing poor match quality on real lists)
  * NumericComparator: "average" (default), "best_match", "range_overlap", "set_jaccard"
  * DateComparator: "closest_dates" (default), "range_overlap", "average_dates", "latest_dates", "earliest_dates"
- When values are delimited strings (e.g., "pop; rock; jazz"), use union(separator="; ") or intersection(separator="; ") as the fuser to properly split before merging.
- For list-like fields where source quality varies, use `favour_sources(source_preferences=[...])` or `prefer_higher_trust(trust_map={...})` rather than blind union/intersection.
- `intersection_k_sources(k=2)` is useful when 2+ source agreement signals correctness.
- Ensure list normalization is consistent ACROSS sources before fusion (same delimiter, same casing, same element format).
""".strip()


PIPELINE_MATCHING_SAFETY_RULES_BLOCK = """
MATCHING/COMPARATOR SAFETY RULES (MANDATORY):
- Blocking and matching are pre-tested inputs: do NOT invent new strategies or tune thresholds.
- Use `blocking_config` and `matching_config` exactly as provided for strategy/columns/weights/thresholds.
- Never leave StringComparator list_strategy unspecified when values may be lists.
- Never leave NumericComparator list_strategy unspecified when values may be lists.
- Never leave DateComparator list_strategy unspecified when date columns may contain multiple dates.
- Avoid introducing preprocessing that changes representation style away from validation set conventions unless diagnostics explicitly justify it.
- Custom fusers must use PyDI resolver signature: `def my_fuser(inputs, **kwargs)`.
- Custom fusers should return a PyDI tuple: `(value, confidence, metadata)`.
FUSION SAFETY RULES (MANDATORY):
- Prefer PyDI built-in fusers for standard attribute types.
- NEVER write custom fusion resolvers or lambda fusers. Use ONLY PyDI built-in resolvers (voting, longest_string, most_complete, median, average, earliest, most_recent, union, intersection, intersection_k_sources, prefer_higher_trust, favour_sources). Custom fusers break PyDI's contract and cause runtime errors.
- Built-in fuser reference:
  * String attributes: longest_string, shortest_string, most_complete, voting, prefer_higher_trust
  * Numeric attributes: median, average, maximum, minimum, sum_values
  * Date attributes: most_recent, earliest
  * List/set attributes: union(separator=...), intersection(separator=...), intersection_k_sources(k=2)
  * Trust-based: prefer_higher_trust(trust_map={...}), favour_sources(source_preferences=[...])
  * General: voting, random_value, weighted_voting
- When using union or intersection on delimited-string lists, pass `separator` to split correctly (e.g., union(separator="; ")).
""".strip()


def build_pipeline_system_prompt(
    *,
    example_pipeline_code: str,
    datasets: list,
    entity_matching_section: str,
    dataset_previews: dict,
    data_profiles: dict,
    normalization_context: dict,
    blocking_config: dict | None,
    matching_config: dict | None,
    matcher_mode: str,
    expected_pairs: list,
    evaluation_analysis: str | None,
    reasoning_brief: dict | None,
    auto_diagnostics: dict | None,
    investigator_action_plan: list | None,
    fusion_guidance: dict | None,
    cluster_analysis: dict | None,
    cluster_example_pipeline_code: str | None,
    correspondences_dir: str,
    output_dir: str = "output",
    focused_context: str | None = None,
    iteration_history: str | None = None,
    input_data_context: str | None = None,
    correspondence_summary: dict | None = None,
) -> str:
    """Assembles the full system prompt for pipeline generation / adaptation."""

    system_prompt = f"""
        You are a data scientist tasked with the integration of several datasets.
        You are provided with the following inputs:

        1. An example integration pipeline (Python code using PyDI library). Pay
        attention to the comments within the pipeline, as they also contain important instructions:
        {example_pipeline_code}

        2. A list of dataset file paths (LOAD THESE EXACT PATHS — do NOT invent or assume different paths):
        {json.dumps(datasets, indent=2)}

        IMPORTANT: Set OUTPUT_DIR = "{output_dir}" at the top of your code and use it for ALL output paths:
        - Correspondences: os.path.join(OUTPUT_DIR, "correspondences")
        - Fusion output: os.path.join(OUTPUT_DIR, "data_fusion")
        - Debug file: os.path.join(OUTPUT_DIR, "data_fusion/debug_fusion_data.jsonl")
        Do NOT hardcode "output/" anywhere.

        {entity_matching_section}

        3. The first row of each dataset to help understand the structure:
        {json.dumps(dataset_previews, indent=2)}

        4. A dictionary containing the profile data of the datasets
        (including number of rows, nulls_per_column and dtypes of
        the columns):
        {json.dumps(data_profiles, indent=2)}
        """

    system_prompt += f"""

        4b. NORMALIZATION CONTEXT (important for robust generation):
        {json.dumps(normalization_context, indent=2)}

        {PIPELINE_NORMALIZATION_RULES_BLOCK}
        """

    # Blocking config section (from BlockingTester when available)
    if blocking_config:
        system_prompt += f"""

        5. **BLOCKING CONFIGURATION** (pre-computed optimal blocking strategies):
        This configuration was determined by a blocking evaluation agent. Use these settings
        for your blocking step in the pipeline:
        {json.dumps(blocking_config, indent=2)}

        IMPORTANT: Use the id_columns and blocking_strategies from this config:
        - Use the correct id_column for each dataset as specified
        - Use the recommended strategy (exact_match_single/multi or semantic_similarity)
        - Use the recommended columns for blocking
        - Strategy to blocker mapping:
          * exact_match_single / exact_match_multi -> StandardBlocker (exact match on columns)
          * token_blocking / ngram_blocking -> TokenBlocker (token or n-gram blocking)
          * sorted_neighbourhood -> SortedNeighbourhoodBlocker (window)
          * semantic_similarity -> EmbeddingBlocker (top_k)
        - Blocker API contract in this environment:
          * materialize candidate pairs with `blocker.materialize()`
          * do NOT call `.block()` on `EmbeddingBlocker`
          * pass the materialized candidate DataFrame into `RuleBasedMatcher.match(..., candidates=...)`
        - Correspondence artifact contract (mandatory):
          * for each expected pair {json.dumps(expected_pairs)}
          * save exactly one CSV file to `{correspondences_dir}/correspondences_<left>_<right>.csv`
          * each file must contain PyDI fusion-ready columns `id1` and `id2` (optional extra columns like `score` are allowed)
          * do NOT invent alternate filename schemes
          * do NOT rely on reloading or renaming correspondences later to make them usable
          * fail fast if an expected pair's correspondence DataFrame is empty before fusion
        """

    # Matching config section (from MatchingTester when available)
    if matching_config:
        if matcher_mode == "ml":
            system_prompt += f"""

        6. **MATCHING CONFIGURATION** (pre-computed comparator settings):
        This configuration was determined by a matching evaluation agent. Use these settings
        for your MLBasedMatcher feature extraction and training:
        {json.dumps(matching_config, indent=2)}

        IMPORTANT: Use the matching_strategies from this config:
        - Build comparators (StringComparator/NumericComparator/DateComparator) for each dataset pair
        - Use the comparators as features in FeatureExtractor
        - Train a classifier on the labeled pairs (labels in the testset) and use MLBasedMatcher
        - Do NOT set weights; ML model learns weights internally
        - Do NOT add RuleBasedMatcher fallback branches
        - Follow the example ML pipeline structure and naming closely; do not invent new variable roles
        - Gold correspondence files can use different pair-id column names (e.g., id1/id2, id_a/id_b). Infer pair columns dynamically and normalize them before feature extraction.
        - preprocess mapping: "lower" -> str.lower, "strip" -> str.strip, "lower_strip" -> lambda x: str(x).lower().strip()

        - For NumericComparator on potentially list-valued columns (e.g., duration fields), set `list_strategy` explicitly (recommended: "average") or safely scalarize before matching.
        - For DateComparator on multi-date columns, set `list_strategy` explicitly (recommended: "closest_dates").
        """
        else:
            system_prompt += f"""

        6. **MATCHING CONFIGURATION** (pre-computed comparator settings):
        This configuration was determined by a matching evaluation agent. Use these settings
        for your RuleBasedMatcher step in the pipeline:
        {json.dumps(matching_config, indent=2)}

        IMPORTANT: Use the matching_strategies from this config:
        - Build comparators (StringComparator/NumericComparator/DateComparator) for each dataset pair
        - Use the specified weights and threshold.
        - For each pair in `matching_strategies`, you **MUST** define a
            variable named `threshold_[original_pair_key]` (e.g.,
            `threshold_discogs_lastfm = 0.7`) with the corresponding threshold value.
            Ensure your RuleBasedMatcher instances use these variables for their
            `threshold` parameter.
        - preprocess mapping: "lower" -> str.lower, "strip" -> str.strip, "lower_strip" -> lambda x: str(x).lower().strip()

        - For NumericComparator on potentially list-valued columns (e.g., duration fields), set `list_strategy` explicitly (recommended: "average") or safely scalarize before matching.
        - For DateComparator on multi-date columns, set `list_strategy` explicitly (recommended: "closest_dates").
        """

    system_prompt += """

        Your task is to **create a similar integration pipeline** so that it works with
        the datasets provided. Output should only consist of the relevant Python code
        for the integration pipeline.

        OUTPUT PATHS (MANDATORY — use these exact paths, do NOT hardcode 'output/'):
        - Correspondences: save to `{correspondences_dir}/correspondences_<left>_<right>.csv`
        - Fusion data: save to `{output_dir}/data_fusion/fusion_data.csv`
        - Fusion debug: save to `{output_dir}/data_fusion/debug_fusion_data.jsonl`
        - Create directories with `os.makedirs(..., exist_ok=True)` before writing.

        FUSION RULES (MANDATORY):
        - The output must always be the full fused dataset. Run `DataFusionEngine.run(..., include_singletons=True)`.
        - Use PyDI built-in fusers as the default and strongly preferred choice.
        - NEVER write custom fusion resolvers, lambda resolvers, or helper functions for fusion. Use ONLY PyDI built-in resolvers.
        - The built-in resolvers cover ALL use cases: voting, longest_string, most_complete, median, average, earliest, most_recent, union, intersection, intersection_k_sources, prefer_higher_trust, favour_sources.
        - For list-valued columns: use union(separator="; ") or intersection_k_sources(k=2, separator="; ").
        - For trust-based selection: use prefer_higher_trust(trust_map={...}) or favour_sources(source_preferences=[...]).
        - Do NOT manually post-process or overwrite `_fusion_confidence` outside the normal resolver return contract.
        - ML MATCHER WARNING: When using MLBasedMatcher, clusters often contain MULTIPLE records per source dataset
          (e.g. 5 discogs records + 3 lastfm records + 2 musicbrainz records in one cluster).
          In this situation, `voting` is BIASED toward whichever source contributed more records to the cluster,
          regardless of data quality. NEVER use `voting` as a primary resolver with ML matching.
          Instead use `prefer_higher_trust` or `favour_sources` to select the most authoritative source,
          or use `shortest_string`/`longest_string` which pick by value characteristics rather than majority.
        - Choose fusion strategies based on EVIDENCE, not just data type:
          * If one source is clearly more complete/accurate for an attribute (visible in input data samples or source agreement), use `prefer_higher_trust` with appropriate trust_map
          * If sources have roughly equal quality, use `voting` for strings, `median` for numbers
          * `longest_string` is excellent for text attributes where one source has more complete/detailed values (e.g., full label names vs abbreviations, complete genre lists vs partial). Consider it as a strong default for metadata text fields.
          * `most_complete` is similar to longest_string but also considers non-null coverage. Good for attributes with many nulls in some sources.
          * For numeric aggregates (e.g., total duration, page count): consider `maximum` when the target represents a "full" measurement — the maximum value often reflects the most complete source. Use `median` only when averaging across sources makes semantic sense.
          * If mismatch samples show format differences (not value differences), fix normalization before fusion rather than switching fusers
          * For list-like fields: use `intersection(separator=...)` or `intersection_k_sources(k=2)` when sources are noisy; use `union(separator=...)` only when both sources are clean and complementary
          * For list-like fields where one source is clearly better, use `favour_sources(source_preferences=[...])` instead of union/intersection
          * For dates: use `prefer_higher_trust` with data-driven trust ordering. Avoid `earliest` as a blanket default — it picks the oldest date regardless of correctness. Only use `most_recent` when diagnostics explicitly justify recency semantics
          * IMPORTANT: Review the INPUT DATA SAMPLES section to see actual values from each source before choosing trust levels
        - For trust-based fusion, pass trust configuration only at registration time:
          `strategy.add_attribute_fuser(attr, prefer_higher_trust, trust_map=trust_map)`
        - Do NOT wrap `prefer_higher_trust` in another function that hardcodes `trust_map` again.
        - Avoid `most_recent` as a default temporal resolver unless diagnostics explicitly show recency is the correct target semantics.
        - Avoid `union` for list-like fields when cluster purity is weak or list-format inconsistencies are present.

        {PIPELINE_MATCHING_SAFETY_RULES_BLOCK}
        """

    if evaluation_analysis:
        system_prompt += f"""
            8. Evaluation reasoning from prior pipeline run:
            {evaluation_analysis}
            """
    if isinstance(reasoning_brief, dict) and reasoning_brief:
        system_prompt += f"""
            8b. COMPACT NEXT-PASS SUMMARY:
            {json.dumps(reasoning_brief, indent=2)}

            Treat this as the highest-signal summary of what went wrong and what to try next.
            """

    if auto_diagnostics:
        system_prompt += f"""
            9. AUTO DIAGNOSTICS FROM THE LAST EXECUTION (summarized):
            {summarize_diagnostics_for_llm(auto_diagnostics)}

            You MUST react to these diagnostics:
            - Always set `include_singletons=True`. The goal is the full fused dataset, never matched-only output.
            - Preserve source IDs for evaluation alignment when mapped ID coverage is low or missing_fused_value is high.
            - If list-attribute accuracy is very low, do not blindly union noisy lists; use a robust strategy (trusted-source selection or consensus with normalization).
            - Always write correspondence files for every evaluated pair to `{correspondences_dir}/` so cluster analysis uses current run artifacts.
            - When passing correspondences into PyDI fusion, the required schema is `id1`/`id2`. Do not invent alternate required columns such as `id_l`/`id_r`.
            """
    if investigator_action_plan:
        system_prompt += f"""
            11. INVESTIGATOR ACTION PLAN (RANKED):
            {json.dumps(investigator_action_plan, indent=2)}

            Implement highest-priority actions first.
            Do not implement low-confidence actions unless supported by additional evidence.

            CRITICAL — PROTECT WORKING ATTRIBUTES:
            If the action plan recommends changing a fusion resolver for an attribute that
            already has high accuracy (>80%), DO NOT change it unless the plan explicitly
            acknowledges the regression risk and provides strong evidence that the change
            will not hurt that attribute. Keeping a working resolver is always safer than
            speculative changes.
            """
    if fusion_guidance:
        system_prompt += f"""
            11b. GENERIC FUSION GUIDANCE FROM DIAGNOSTICS:
            {json.dumps(fusion_guidance, indent=2)}

            Apply this guidance in a dataset-agnostic way:
            - If `attribute_strategies[attr]` exists, use its `recommended_fuser` for that attribute unless the current code already implements an equally safe choice with the same rationale.
            - If `post_clustering.recommended_strategy` is present, apply that post-clustering strategy before fusion.
            - If a dominant source is indicated for an attribute, express that via `prefer_higher_trust` with a transparent `trust_map` rather than hard-coded dataset-specific logic.
            - Keep the implementation generic: decisions should depend on evidence classes such as cluster impurity, disagreement, malformed lists, and source agreement, not on dataset names.
            """

    # Cluster analysis section
    cluster_recommendation = None
    cluster_parameters = {}
    cluster_recommendation_source = None

    if cluster_analysis:
        overall = cluster_analysis.get("_overall", {}) if isinstance(cluster_analysis, dict) else {}
        if isinstance(overall, dict) and overall.get("recommended_strategy") not in (None, "", "None"):
            cluster_recommendation = overall.get("recommended_strategy")
            cluster_parameters = overall.get("parameters", {})
            cluster_recommendation_source = "_overall"
        else:
            for key, value in (cluster_analysis or {}).items():
                if not isinstance(value, dict):
                    continue
                rec = value.get("recommended_strategy")
                if rec and rec != "None":
                    cluster_recommendation = rec
                    cluster_parameters = value.get("parameters", {})
                    cluster_recommendation_source = key
                    break

    if cluster_analysis:
        # Build a compact evidence section from per-pair reports
        cluster_evidence_lines = []
        for key, value in (cluster_analysis or {}).items():
            if key == "_overall" or not isinstance(value, dict):
                continue
            evidence = value.get("evidence_summary", "")
            if evidence:
                cluster_evidence_lines.append(f"  [{key}]\n  {evidence}")
            # Include strategy ratings for this pair
            ratings = value.get("strategy_ratings", {})
            if ratings:
                rec_strats = [s for s, r in ratings.items() if r.get("rating") == "recommended"]
                if rec_strats:
                    cluster_evidence_lines.append(f"  Recommended for this pair: {', '.join(rec_strats)}")

        overall = cluster_analysis.get("_overall", {}) if isinstance(cluster_analysis, dict) else {}
        overall_evidence = overall.get("all_evidence", "") if isinstance(overall, dict) else ""

        system_prompt += f"""

        **CLUSTER ANALYSIS EVIDENCE (advisory — use your judgement):**

        Overall recommendation: {cluster_recommendation or "None"}
        Overall diagnosis: {overall.get("diagnosis", "n/a") if isinstance(overall, dict) else "n/a"}

        Per-pair evidence:
        {"chr(10)".join(cluster_evidence_lines) if cluster_evidence_lines else "No per-pair evidence available."}

        Available PyDI post-clustering strategies (choose based on the evidence above):
        - MaximumBipartiteMatching(): Best for enforcing 1:1 when one-to-many ratio is high. Maximises total score.
        - StableMatching(): Best when ambiguity is high (close scores). Produces stable 1:1 assignment.
        - GreedyOneToOneMatchingAlgorithm(): Fast 1:1 pruning when scores are well-separated (high IQR).
        - HierarchicalClusterer(linkage_mode=LinkageMode.AVG, min_similarity=0.5): Best for splitting over-merged clusters. Does NOT enforce 1:1. Import: from PyDI.entitymatching import LinkageMode
        - ConnectedComponentClusterer(): Baseline grouping. Rarely the best final choice for noisy data.

        CLUSTER-DRIVEN RULES:
        - Cluster analysis may ONLY influence post-clustering (after matching, before fusion).
        - Do NOT change matching thresholds, comparator weights, or matcher type based on cluster analysis.
        - If the evidence recommends a strategy, add a post-clustering step. If not, skip post-clustering.
        - The recommendation is advisory: if you see evidence that a different strategy fits better, use it.
        - Example of applying post-clustering:
        {cluster_example_pipeline_code}
        """

    if focused_context:
        system_prompt += f"""

        **FOCUSED PIPELINE CONTEXT (high-signal summary of current state):**
        {focused_context}

        This is a compressed summary of evaluation metrics, diagnostics, mismatch examples, and stage diagnosis.
        Use this as your primary reference for what needs fixing.
        """

    if iteration_history:
        system_prompt += f"""

        **ITERATION HISTORY (what was tried before and what happened):**
        {iteration_history}

        CRITICAL: Review this history before making changes.
        - Do NOT repeat approaches that already failed or caused regression.
        - If accuracy is stagnating, try a fundamentally different approach.
        - If a previous change improved accuracy, consider extending that approach.
        """

    if input_data_context:
        system_prompt += f"""

        **INPUT DATA SAMPLES FOR WORST ATTRIBUTES:**
        {input_data_context}

        Use these raw input samples to:
        - Identify which source dataset has cleaner/more complete values for each attribute
        - Choose `prefer_higher_trust` with appropriate trust_map when one source is clearly better
        - Decide normalization strategy based on actual value formats in each source
        - Detect systematic quality differences (nulls, formatting, completeness) between sources
        """

    if correspondence_summary and isinstance(correspondence_summary, dict):
        system_prompt += f"""

        **CORRESPONDENCE STATISTICS (entity matching results):**
        {json.dumps(correspondence_summary, indent=2)}

        Review these statistics to understand matching quality per dataset pair:
        - Low correspondence counts may indicate blocking/matching issues for that pair
        - Highly uneven match ratios suggest one-to-many relationships or matching noise
        - Empty correspondences mean no entities were matched for that pair — this is a critical issue
        """

    system_prompt += f"""

        {PYDI_API_CHEATSHEET}

            **PROCESS:**
            1.  **THINK**: Analyze the provided data profiles, configurations, and any previous error reports.
            2.  **RESEARCH**: If you are unsure how to use a PyDI function or class, you MAY use the `search_documentation` tool for additional detail. The API REFERENCE above covers the most important signatures — prefer it over search results when they conflict.
            3.  **CODE**: Once you have gathered enough information, write the complete, executable Python code for the pipeline. **Your final output in this process must be only the Python code itself.**
            4.  **FUSION SAFETY**: Ensure custom fusers/lambda fusers accept `**kwargs` so PyDI runtime kwargs do not fail.
            5.  **PYDI-FIRST DISCIPLINE**: Prefer PyDI built-ins and documented APIs over bespoke helper code. If you introduce a custom helper, justify why built-ins are insufficient.
            6.  **TARGETED CHANGE DISCIPLINE**: Make changes that are explicitly supported by diagnostics. Preserve working sections unless you have evidence they contribute to the failure."""

    return system_prompt
