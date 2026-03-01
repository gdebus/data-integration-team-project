EVALUATION_ROBUSTNESS_RULES_BLOCK = """
EVALUATION ROBUSTNESS RULES (MANDATORY):
- Use validation-set representation as the normalization target for evaluation.
- Normalize fused and gold columns to the SAME representation before evaluating.
- For country-like columns, infer and apply a shared normalize_country output_format from the evaluation set (alpha_2/alpha_3/numeric/name/official_name).
- For free-text/list-like columns, avoid strict exact comparisons unless values are canonical identifiers.
  Prefer tokenized_match with an explicit threshold (for example 0.7-0.9) instead of exact-only semantics.
- If using set-based comparison on textual lists, first canonicalize list elements consistently across fused and gold.
""".strip()

