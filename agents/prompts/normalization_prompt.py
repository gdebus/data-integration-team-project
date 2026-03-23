"""Prompt for LLM-driven normalization spec generation.

The LLM sees sample rows from each source dataset alongside sample rows from
the validation set (the target representation) and generates a PyDI
NormalizationSpec per source dataset.
"""

NORMALIZATION_SPEC_SYSTEM_PROMPT = """
You are a data normalization expert.  Your job is to examine source datasets
and a validation/target dataset, then produce a PyDI NormalizationSpec for each
source so that, after normalization, column values match the target style.

## PyDI NormalizationSpec reference

A NormalizationSpec is a JSON object whose keys are column names.  Each value
is a ColumnSpec dict.  The available ColumnSpec fields (all optional) are:

| field                    | type / allowed values                                      | purpose                                                 |
|--------------------------|------------------------------------------------------------|---------------------------------------------------------|
| output_type              | "string", "float", "int", "bool", "datetime", "keep"      | Target dtype after transform                            |
| on_failure               | "keep", "null", "raise"                                    | What to do when a cell can't be converted               |
| strip_whitespace         | bool                                                       | Trim leading/trailing whitespace                        |
| case                     | "lower", "upper", "title", "keep", null                   | Text case transformation                                |
| country_format           | "alpha_2", "alpha_3", "numeric", "name", null              | Country value output format                             |
| currency_format          | "alpha_3", "name", "keep", null                            | Currency output format                                  |
| phone_format             | "e164", "international", "national", "keep", null          | Phone number format                                     |
| phone_default_region     | str (e.g. "US")                                            | Default region when phone lacks country code            |
| normalize_email          | bool                                                       | Normalize email addresses                               |
| stdnum_format            | bool                                                       | Normalize standard numbers (ISBN, IBAN, etc.)           |
| date_format              | str (strftime pattern, e.g. "%Y-%m-%d") or null            | Output date format                                      |
| expand_scale_modifiers   | bool                                                       | Expand "5 million" → 5000000                            |
| convert_percentage       | "to_decimal", "to_percent", "keep", null                   | Percentage representation                               |
| target_unit              | str (e.g. "km", "USD") or null                             | Convert quantities to this unit                         |

## Rules

1. **The validation/target dataset is AUTHORITATIVE.**  Your spec must produce
   values that look like the target, not the other way around.  Study the
   target examples carefully before writing specs.
   - LOOK AT ACTUAL VALUES in the target: what case? what country format?
     what date format? what delimiters for lists?
   - Your normalization must CONVERGE source values toward the target format.
   - If the target uses "United States" (full name), normalize to full names.
   - If the target uses "US" (alpha_2), normalize to alpha_2.
   - If the target uses mixed case ("The Beatles"), do NOT lowercase.
   - If the target uses lowercase, then lowercase.
   - NEVER guess — always base your decision on the actual target examples.

2. **Only specify columns that need transformation.**  If a source column
   already matches the target style, omit it from the spec (or set fields to
   "keep" / null).  Unnecessary transformations risk breaking values that
   already match.

   CRITICAL: On re-normalization attempts (attempt 2+), the spec is applied to the
   ORIGINAL source data, not the previously normalized version. Therefore, your new
   spec MUST include ALL transformations from previous specs that were working
   correctly, PLUS the new fixes. Do NOT produce a minimal spec that only addresses
   the new issue — that will lose all previous normalization gains. If the previous
   spec normalized country names and dates correctly, your new spec must ALSO include
   those same country and date transformations.

3. **Preserve ID and identifier columns.**  Never include columns whose name
   contains "id" or "isbn" (case-insensitive) in the spec.  Also preserve columns
   that serve as natural keys or identifiers — ISBN, ISSN, EAN, UPC, ASIN, barcode,
   postal/zip codes, phone numbers used as keys — even if they look numeric.
   These often have significant leading zeros (e.g. ISBN "0312252951") that
   would be lost if cast to int.  Leave them EXACTLY as they are — do not
   include them in any NormalizationSpec under any circumstances.

4. **Be precise about case.**  If the target uses lowercase for a column, set
   `case: "lower"`.  If the target preserves original case, set `case: "keep"`
   or omit.  LOOK AT THE TARGET SAMPLES to decide — do not default to
   lowercasing.

5. **Country columns.**  Compare source and target representations:
   - If target has "US", "DE" → `country_format: "alpha_2"`
   - If target has "USA", "DEU" → `country_format: "alpha_3"`
   - If target has "United States" → `country_format: "name"`
   Omit `country_format` for non-country columns.

6. **Numeric columns.**  If the source has "5 million" or "3.2B" and the
   target has plain numbers, set `expand_scale_modifiers: true` and
   `output_type: "float"`.

7. **Date columns.**  If the source has dates in a different format from the
   target, set `date_format` to match the target pattern.

8. **strip_whitespace: true** is almost always a good idea for text columns.

9. **on_failure: "keep"** is the safest default — it preserves original values
   when transformation fails rather than nulling them out.

10. One spec per source dataset.  Return a JSON object whose keys are the
    source dataset filenames (basename without extension) and whose values are
    the NormalizationSpec dicts.

11. **Minimalism principle.**  Every transformation you specify is a chance to
    break something.  Only transform a column when there is a VISIBLE
    mismatch between the source format and the target format.  When in doubt,
    omit the column from the spec.

## Additional transforms (beyond NormalizationSpec)

If you detect list-like columns (e.g. values like `["a", "b"]` or `a, b, c`)
that need normalizing, include a top-level `list_columns` array with those
column names.

## Output format

Return STRICT JSON (no markdown fences, no commentary) with this structure:

{
  "specs": {
    "<source_dataset_name>": {
      "<column_name>": { <ColumnSpec fields> },
      ...
    },
    ...
  },
  "list_columns": ["col1", "col2"],
  "reasoning": "Brief explanation of key decisions"
}
""".strip()


def build_normalization_user_prompt(
    *,
    source_probes: dict[str, dict],
    target_probe: dict,
    mismatch_examples: dict | None = None,
    previous_attempt_feedback: str | None = None,
) -> str:
    """Build the user/human message for normalization spec generation.

    Parameters
    ----------
    source_probes : dict
        ``{dataset_name: {"columns": [...], "dtypes": {...}, "sample_rows": [...]}}``
    target_probe : dict
        Same structure for the validation/target dataset.
    mismatch_examples : dict | None
        Per-attribute mismatch samples from the investigator probes.
    previous_attempt_feedback : str | None
        If this is a re-run, a summary of what went wrong last time.
    """
    import json

    parts: list[str] = []

    # ── Target dataset ──
    parts.append("## TARGET / VALIDATION DATASET (this is the authoritative style)\n")
    parts.append(f"Columns: {json.dumps(target_probe.get('columns', []))}")
    parts.append(f"Dtypes:  {json.dumps(target_probe.get('dtypes', {}))}")
    parts.append("Sample rows:")
    for row in target_probe.get("sample_rows", []):
        parts.append(f"  {json.dumps(row, ensure_ascii=False, default=str)}")
    parts.append("")

    # ── Source datasets ──
    for name, probe in source_probes.items():
        parts.append(f"## SOURCE DATASET: {name}\n")
        parts.append(f"Columns: {json.dumps(probe.get('columns', []))}")
        parts.append(f"Dtypes:  {json.dumps(probe.get('dtypes', {}))}")
        parts.append("Sample rows:")
        for row in probe.get("sample_rows", []):
            parts.append(f"  {json.dumps(row, ensure_ascii=False, default=str)}")
        parts.append("")

    # ── Mismatch examples (if available from a previous evaluation) ──
    if mismatch_examples:
        parts.append("## MISMATCH EXAMPLES FROM PREVIOUS EVALUATION")
        parts.append("These show concrete cases where fused output didn't match the target.")
        parts.append("Use them to identify which columns need transformation and what kind.\n")
        for attr, samples in mismatch_examples.items():
            parts.append(f"### {attr}")
            if isinstance(samples, list):
                for s in samples[:5]:
                    parts.append(f"  {json.dumps(s, ensure_ascii=False, default=str)}")
            elif isinstance(samples, dict):
                parts.append(f"  {json.dumps(samples, ensure_ascii=False, default=str)}")
            parts.append("")

    # ── Previous attempt feedback ──
    if previous_attempt_feedback:
        parts.append("## FEEDBACK FROM PREVIOUS NORMALIZATION ATTEMPT")
        parts.append("The previous normalization attempt did not fully resolve the issues.")
        parts.append("Adjust your spec to address the remaining problems:\n")
        parts.append(previous_attempt_feedback)
        parts.append("")

    return "\n".join(parts)
