EVALUATION_ROBUSTNESS_RULES_BLOCK = """
EVALUATION ROBUSTNESS RULES (MANDATORY):
- Use validation-set representation as the normalization target for evaluation.
- Normalize fused and gold columns to the SAME representation before evaluating.
- For country-like columns, infer and apply a shared normalize_country output_format from the evaluation set (alpha_2/alpha_3/numeric/name/official_name).
- For free-text/list-like columns, avoid strict exact comparisons unless values are canonical identifiers.
  Prefer tokenized_match with an explicit threshold (for example 0.7-0.9) instead of exact-only semantics.
- If using set-based comparison on textual lists, first canonicalize list elements consistently across fused and gold.
- Choose the right evaluation function per attribute type:
  * tokenized_match: names, titles, labels, artists, text fields (handles word reordering like "The Beatles" vs "Beatles, The")
  * year_only_match: dates where only the year matters
  * set_equality_match: list-like columns (tracks, genres, categories)
  * numeric_tolerance_match: durations, prices, ratings (use reasonable tolerance)
  * numeric_tolerance_match: also use for year/count columns stored as floats (e.g. publish_year, page_count) —
    cast both fused and gold values to int before comparison to avoid 274.0 != 274 mismatches
  * exact_match: IDs, codes, boolean fields
- If the gold set does NOT contain a column (i.e. it is absent or all-null), do NOT evaluate that column.
  Only evaluate columns that are present and meaningful in BOTH the fused and gold datasets.
- IMPORTANT: When the gold set has a parent column (e.g. "tracks") but the fused output has sub-columns
  derived from it (e.g. "tracks_track_name", "tracks_track_duration", "tracks_track_position"),
  DROP the sub-columns from evaluation unless they have explicit corresponding columns in the gold set.
  Evaluating derived sub-columns against a non-existent gold column produces zero accuracy and wastes
  investigation cycles. Only evaluate attributes that have a direct 1:1 column match in the gold set.
- IMPORTANT: Before evaluation, coerce numeric-like columns (publish_year, page_count, year, count, rating, score)
  to a consistent type in BOTH fused and gold DataFrames. Common pattern:
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].dropna().apply(lambda x: float(x).is_integer()).all():
            df[col] = df[col].astype('Int64')  # nullable int avoids 274.0 vs 274 mismatches
  This prevents false mismatches from float vs int representation differences.
- If the fused output contains an "eval_id" column, use fused_id_column="eval_id" (NOT "_id" or "id").
  The eval_id column is pre-computed to map cluster IDs back to validation-set source IDs.
""".strip()

