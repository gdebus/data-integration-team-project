"""Compact PyDI API reference injected into pipeline prompts.

This supplements the vector DB search tool with accurate, always-available
function signatures for the most critical PyDI APIs.  Sourced from the
live documentation at https://wbsg-uni-mannheim.github.io/PyDI/ (2026-03).
"""

PYDI_API_CHEATSHEET = """
=== PyDI QUICK API REFERENCE (authoritative — use over search_documentation when conflicting) ===

COMPARATORS (PyDI.entitymatching.comparators):
  StringComparator(column, similarity_function="jaro_winkler", tokenization=None, preprocess=None, list_strategy=None)
    list_strategy options: "concatenate" (default, poor for real lists), "best_match", "set_jaccard", "set_overlap"
  NumericComparator(column, method="absolute_difference", max_difference=None, list_strategy=None)
    method: "absolute_difference" (default, fixed ±N tolerance) or "relative_difference" (percentage-based, e.g. 0.10 = 10%)
    Use relative_difference for measures where % tolerance is natural (durations, prices, ratings)
    list_strategy options: "average" (default), "best_match", "range_overlap", "set_jaccard"
  DateComparator(column, max_days_difference=None, list_strategy=None)
    list_strategy options: "closest_dates" (default), "range_overlap", "average_dates", "latest_dates", "earliest_dates"

MATCHERS:
  RuleBasedMatcher.match(df_left, df_right, candidates, id_column, comparators, weights=None, threshold=0.0, debug=False)
  MLBasedMatcher(feature_extractor).match(df_left, df_right, candidates, id_column, trained_classifier, threshold=0.5, use_probabilities=False, debug=False)

BLOCKERS:
  StandardBlocker(df_left, df_right, on, id_column, batch_size=100000)
  TokenBlocker(df_left, df_right, column, id_column, tokenizer=...)
  SortedNeighbourhoodBlocker(df_left, df_right, key, id_column, window=...)
  EmbeddingBlocker(df_left, df_right, text_cols=['col1','col2'], id_column=..., top_k=50, model="sentence-transformers/all-MiniLM-L6-v2", index_backend="sklearn")  → use .materialize(), NOT .block()

FUSION ENGINE:
  DataFusionEngine.run(datasets, correspondences, schema_correspondences=None, id_column=None, include_singletons=False)
  DataFusionStrategy.add_attribute_fuser(attribute, fuser_or_resolver, evaluation_function=None, **fuser_kwargs)
    Example: strategy.add_attribute_fuser("genre", union, separator="; ")
    Example: strategy.add_attribute_fuser("name", prefer_higher_trust, trust_map={"src_a": 0.9, "src_b": 0.6})

BUILT-IN FUSION RESOLVERS (from PyDI.fusion import ...):
  String:   voting(values, **kw), longest_string(values, **kw), shortest_string(values, **kw), most_complete(values, **kw)
  Numeric:  median(values, **kw), average(values, **kw), maximum(values, **kw), minimum(values, **kw), sum_values(values, **kw)
  Date:     most_recent(values, **kw), earliest(values, **kw)
  List/Set: union(values, separator=None, **kw), intersection(values, separator=None, **kw), intersection_k_sources(values, k=2, separator=None, **kw)
  Trust:    prefer_higher_trust(values, *, trust_map=None, default_trust=1.0, tie_breaker="first", **kw)
            favour_sources(values, source_preferences=None, source_column="_source", **kw)
  Other:    random_value(values, **kw), weighted_voting(values, weights=None, **kw)

RESOLVER SELECTION GUIDE (proven patterns from real-world PyDI workflows):
  Names/titles:     shortest_string often outperforms voting — canonical names tend to be concise, and longer values often contain noise or appended metadata.
  Artist/author:    longest_string captures the most complete attribution (e.g. "feat." credits).
  Labels/publishers: longest_string — one source typically has the official full name while others abbreviate.
  Duration/numeric: maximum — the largest value typically reflects the most complete measurement (full album vs partial).
                    Use prefer_higher_trust only when source_attribution probe shows one source is clearly more accurate.
  Dates:            prefer_higher_trust with data-driven trust ordering. Avoid earliest as default — it picks the oldest date regardless of correctness.
  Country/region:   prefer_higher_trust — one source is usually authoritative for geographic metadata.
  List attributes:  union(separator="; ") — preserves information from all sources. Use favour_sources only when one source is clearly better.
  General rule:     Start with simple deterministic resolvers (shortest_string, longest_string, maximum, union). Only escalate to
                    trust-based resolvers (prefer_higher_trust) when source_attribution evidence shows a clear winner (>15% gap in match rates).

POST-CLUSTERING (PyDI.entitymatching):
  MaximumBipartiteMatching(threshold=0.0)      — 1:1, optimal total score (Hungarian). Best default.
  StableMatching(threshold=0.0)                — 1:1, stable assignment (Gale-Shapley). Best for high ambiguity.
  GreedyOneToOneMatchingAlgorithm(threshold=0.0) — 1:1, greedy highest-score-first. Fast but approximate.
  HierarchicalClusterer(linkage_mode=LinkageMode.AVG, min_similarity=0.5) — Many-to-many. Import: from PyDI.entitymatching import LinkageMode
  ConnectedComponentClusterer(threshold=0.0)   — Baseline transitive closure.
  All use: clusterer.cluster(correspondences_df) → refined_correspondences_df

NORMALIZATION (PyDI.normalization):
  NormalizationConfig(enable_type_detection=True, enable_unit_conversion=True, standardize_nulls=True, normalize_text=True, lowercase_text=True, ...)
  ColumnSpec(output_type="keep", country_format=None, phone_format=None, currency_format=None, case=None, strip_whitespace=False, date_format=None, ...)
    country_format: "alpha_2", "alpha_3", "numeric", "name", "keep"
    phone_format: "e164", "international", "national", "keep"
    case: "lower", "upper", "title", "keep"
  normalize_dataset(df, config=None, output_path=None)

EVALUATION (PyDI.fusion):
  exact_match, boolean_match, numeric_tolerance_match, set_equality_match, tokenized_match, year_only_match
""".strip()
