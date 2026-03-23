EVALUATION_ROBUSTNESS_RULES_BLOCK = """
EVALUATION ROBUSTNESS RULES (MANDATORY):
- Use validation-set representation as the normalization target for evaluation.
- Normalize fused and gold columns to the SAME representation before evaluating.
- For country-like columns: DO NOT call normalize_country or any country normalization function.
  The pipeline normalization node already ensures fused and gold values use the same country format.
  Applying normalize_country in evaluation DESTROYS correct matches by converting to a different format.
  Simply use tokenized_match for country columns, same as other text fields.
- For free-text/list-like columns, avoid strict exact comparisons unless values are canonical identifiers.
  Prefer tokenized_match with an explicit threshold (for example 0.7-0.9) instead of exact-only semantics.
- If using set-based comparison on textual lists, first canonicalize list elements consistently across fused and gold.
- Choose the right evaluation function per attribute type:
  * tokenized_match: names, titles, labels, artists, text fields (handles word reordering like "The Beatles" vs "Beatles, The")
  * year_only_match: dates and year columns (releaseYear, founded, publish_year) — keep values as STRINGS,
    do NOT convert to pd.to_datetime() or pd.to_numeric() before evaluation. year_only_match extracts
    the year component from string representations directly.
  * set_equality_match: list-like columns (tracks, genres, categories)
  * numeric_tolerance_match: durations, prices, ratings, scores (use reasonable tolerance —
    for user ratings/scores use tolerance=1.0, for critic scores use tolerance=5.0,
    for durations use tolerance appropriate to the unit)
  * numeric_tolerance_match: also use for year/count columns stored as floats (e.g. publish_year, page_count) —
    cast both fused and gold values to int before comparison to avoid 274.0 != 274 mismatches
  * exact_match: IDs, codes, boolean fields
- If the gold set does NOT contain a column (i.e. it is absent, all-null, or all-whitespace), do NOT evaluate that column.
  Only evaluate columns that are present and contain meaningful non-empty values in BOTH the fused and gold datasets.
  Before registering an evaluation function for a column, check: gold_df[col].dropna().str.strip().replace('', pd.NA).dropna()
  — if the result is empty, skip that column entirely.
- IMPORTANT: For columns that appear to be nested sub-fields (they share a repeated prefix pattern like
  "tracks_track_name", "tracks_track_duration", "tracks_track_position"), these are typically derived from
  XML nested elements and are structurally difficult to evaluate accurately. If such columns consistently
  show very low accuracy (<20%) in early evaluation attempts, DROP them from evaluation in subsequent
  attempts — they waste investigation cycles on structural artifacts rather than genuine fusion errors.
  Focus evaluation on top-level scalar attributes that directly correspond to gold set columns.
- IMPORTANT: Before evaluation, coerce numeric-like columns (publish_year, page_count, year, count, rating, score)
  to a consistent type in BOTH fused and gold DataFrames. Common pattern:
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].dropna().apply(lambda x: float(x).is_integer()).all():
            df[col] = df[col].astype('Int64')  # nullable int avoids 274.0 vs 274 mismatches
  This prevents false mismatches from float vs int representation differences.
- If the fused output contains an "eval_id" column, try fused_id_column="eval_id" first.
  However, if eval_id coverage is low (many gold IDs not found), build your own mapping:
  For each gold record ID, search the fused _fusion_sources column to find which fused row contains that ID.
  Use ast.literal_eval() to parse _fusion_sources (it's a Python list as string).
  This handles cases where clusters contain multiple records from the same source and
  the pre-computed eval_id picked a different one than the gold set expects.
""".strip()

