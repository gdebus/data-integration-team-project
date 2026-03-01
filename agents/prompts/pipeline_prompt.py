PIPELINE_NORMALIZATION_RULES_BLOCK = """
IMPORTANT NORMALIZATION RULES:
- Validation set style drives normalization. Do NOT blindly lowercase text globally.
- Keep IDs unchanged (case and content must be preserved exactly).
- If you convert a column to list-like values, any comparator on that column MUST set list_strategy.
- If StringComparator is used on potentially list-like columns (artist/name/tracks/etc.), set list_strategy explicitly.
- Prefer normalization that aligns fused output with validation representation, not a random canonical style.
""".strip()


PIPELINE_MATCHING_SAFETY_RULES_BLOCK = """
MATCHING/COMPARATOR SAFETY RULES (MANDATORY):
- Never leave StringComparator list_strategy unspecified when values may be lists.
- Never leave NumericComparator list_strategy unspecified when values may be lists.
- Avoid introducing preprocessing that changes representation style away from validation set conventions unless diagnostics explicitly justify it.
- Custom fusers must use PyDI resolver signature: `def my_fuser(inputs, **kwargs)`.
- Custom fusers should return a PyDI tuple: `(value, confidence, metadata)`.
FUSION SAFETY RULES (MANDATORY):
- Prefer PyDI built-in fusers for standard attribute types.
- Avoid custom/lambda fusers unless diagnostics provide explicit evidence that built-ins are insufficient.
- Prefer built-in mapping by type: string -> longest_string/prefer_higher_trust, numeric -> median/average, date -> most_recent/earliest, list -> union/intersection.
""".strip()
