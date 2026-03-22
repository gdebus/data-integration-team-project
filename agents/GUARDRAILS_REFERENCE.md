# Guardrail Reference — When, Why, and How Each Guardrail Fires

The agent has two guardrail functions that mechanically fix LLM-generated code before execution. They exist because prompt instructions alone are unreliable — the LLM ignores them ~20-40% of the time. Guardrails fire every time, deterministically.

---

## Pipeline Guardrails (`apply_pipeline_guardrails`)

**File**: `helpers/code_guardrails.py:6`
**Called at**: `pipeline_agent.py:1284`, before every pipeline subprocess execution
**Input**: generated pipeline code (string) + full agent state (dict)
**Output**: corrected pipeline code (string)

---

### PG-1: include_singletons=True Enforcement

**What**: Forces `include_singletons=True` in `DataFusionEngine.run()` calls.

**Why**: The agent must always produce the **full** fused dataset (matched entities + unmatched singletons). Without singletons, the output only contains successfully matched records, which makes the fused dataset incomplete and evaluation unreliable (missing records look like missing fused values).

**When it fires**: Every run. If `include_singletons=False` is found, replaces with `True`. If `include_singletons` is missing entirely, injects it after `id_column=`.

**Example**:
```python
# Before:
engine.run(datasets, correspondences, id_column="id")
# After:
engine.run(datasets, correspondences, id_column="id", include_singletons=True)
```

---

### PG-2: Matching Threshold Freezing

**What**: Overwrites matching threshold variables in the pipeline code with the values from `matching_config`.

**Why**: The MatchingTester spent multiple LLM calls and F1 evaluations to find optimal thresholds. The pipeline LLM sometimes changes these values (rounding, guessing, or "improving" them), which degrades matching quality.

**When it fires**: Every run EXCEPT when the investigator routed to `run_matching_tester` (meaning thresholds are intentionally being re-evaluated). Even when unlocked, thresholds are clamped to +/-0.1 of the original.

**Example**:
```python
# matching_config says threshold=0.66 for goodreads_amazon
# LLM generated:
threshold_goodreads_small_amazon_small = 0.75  # wrong
# After guardrail:
threshold_goodreads_small_amazon_small = 0.66  # restored
```

---

### PG-3: Fusion Guidance Enforcement (Resolver Patching)

**What**: Patches `add_attribute_fuser()` calls to use the resolver recommended by the investigator and source attribution probe.

**Why**: The investigator spends multiple LLM turns analyzing per-source match rates and recommending specific resolvers. The pipeline LLM sometimes ignores these recommendations and uses `voting` or `most_complete` instead.

**When it fires**: Only when `fusion_guidance["attribute_strategies"]` contains recommendations. Two confidence tiers:
- **High confidence (>=0.85)** — from investigation LLM: overrides the pipeline LLM's choice
- **Low confidence (<0.85)** — from probe only: respects the LLM's choice if it picked a valid built-in resolver

**Example (Music)**:
```python
# Investigation recommends: label -> prefer_higher_trust(discogs>musicbrainz)
# LLM generated:
strategy.add_attribute_fuser("label", most_complete)
# After guardrail (confidence=0.9, overrides):
strategy.add_attribute_fuser("label", prefer_higher_trust, trust_map=trust_map)
```

---

### PG-4: Per-Attribute Trust Map Injection

**What**: When different attributes need conflicting trust orderings, injects per-attribute `trust_map_<attr>` variables instead of using a single global `trust_map`.

**Why**: A single `trust_map = {"musicbrainz": 3, "discogs": 2}` can't satisfy both `label` (wants discogs highest) and `duration` (wants musicbrainz highest). Without per-attribute maps, whichever attribute's trust map is applied last wins for all attributes.

**When it fires**: When 2+ attributes use `prefer_higher_trust` with different source orderings. Detected by comparing sorted key orderings across all trust map strategies.

**Example (Music)**:
```python
# Before (wrong — single trust_map can't satisfy both):
trust_map = {"musicbrainz": 3, "discogs": 2, "lastfm": 1}
strategy.add_attribute_fuser("label", prefer_higher_trust, trust_map=trust_map)
strategy.add_attribute_fuser("duration", prefer_higher_trust, trust_map=trust_map)

# After guardrail:
trust_map_label = {"discogs": 3, "musicbrainz": 2, "lastfm": 1}
trust_map_duration = {"musicbrainz": 3, "lastfm": 2, "discogs": 1}
strategy.add_attribute_fuser("label", prefer_higher_trust, trust_map=trust_map_label)
strategy.add_attribute_fuser("duration", prefer_higher_trust, trust_map=trust_map_duration)
```

**Log output**: `[GUARDRAIL] Conflicting trust orderings for 3 attributes — injecting per-attribute trust_maps.`

---

### PG-5: list_strategy Validation

**What**: Validates and fixes the `list_strategy` parameter on each comparator type.

**Why**: Each comparator type has a different set of valid list strategies. The LLM often uses `"concatenate"` which is technically valid for StringComparator but produces poor match quality on real list data.

**When it fires**: Every run, scans all comparator constructor calls.

**Valid strategies per type**:
| Comparator | Valid | Default upgrade |
|-----------|-------|-----------------|
| `StringComparator` | `best_match`, `set_jaccard`, `set_overlap` | `concatenate` -> `best_match` |
| `NumericComparator` | `average`, `best_match`, `range_overlap`, `set_jaccard` | invalid -> `average` |
| `DateComparator` | `closest_dates`, `range_overlap`, `average_dates`, `latest_dates`, `earliest_dates` | invalid -> `closest_dates` |

**Log output**: `[GUARDRAIL] Validated list_strategy in 12 comparator call(s).`

---

### PG-6: General Import Safety

**What**: Scans all `add_attribute_fuser()` calls for built-in resolver symbols that aren't imported, and auto-adds them to the fusion import block.

**Why**: The LLM uses a resolver (e.g., `favour_sources`) but forgets to add it to the `from PyDI.fusion import ...` statement. This causes a `NameError` at runtime that wastes an entire execution attempt.

**When it fires**: Every run, after all other fusion-related guardrails. Checks against the known set of 17 built-in resolvers.

**Example**:
```python
# Before (favour_sources used but not imported):
from PyDI.fusion import voting, prefer_higher_trust
strategy.add_attribute_fuser("tracks_track_name", favour_sources, source_preferences=[...])

# After guardrail:
from PyDI.fusion import voting, prefer_higher_trust, favour_sources
strategy.add_attribute_fuser("tracks_track_name", favour_sources, source_preferences=[...])
```

**Log output**: `[GUARDRAIL] Auto-imported 1 missing fusion symbol(s): favour_sources`

---

### PG-7: Eval ID Injection

**What**: Detects the validation set's ID prefix and injects a `_extract_eval_id()` function that creates an `eval_id` column in the fused output.

**Why**: The fused output's `_id` column uses the first dataset's ID format (e.g., `discogs_3`), but the validation set uses a different source's IDs (e.g., `mbrainz_974`). Without eval_id, the evaluation can't align fused records to gold records, producing 0% accuracy.

**When it fires**: When the validation set has source-prefixed IDs (detected by splitting the first ID on `_` or `-`). Does NOT fire when IDs are non-prefixed (e.g., plain ISBNs).

**How prefix detection works**:
1. Reads first 5 rows of the validation set
2. Splits the first ID on `_` (e.g., `mbrainz_974` -> prefix `mbrainz_`)
3. If no `_` found, tries `-` (e.g., `yelp-06497` -> prefix `yelp-`)
4. Collects ALL prefixes from the validation set (handles mixed prefixes like restaurant)

**What gets injected** (before the `to_csv()` call):
```python
import ast as _ast
_EVAL_PREFIXES = ['mbrainz_']
def _extract_eval_id(row):
    try:
        sources = _ast.literal_eval(str(row.get("_fusion_sources", "[]")))
        if isinstance(sources, (list, tuple)):
            for sid in sources:
                s = str(sid)
                if any(s.startswith(p) for p in _EVAL_PREFIXES):
                    return s
    except Exception:
        pass
    return str(row.get("_id", row.get("id", "")))
fused_result["eval_id"] = fused_result.apply(_extract_eval_id, axis=1)
```

**Per use case**:
| Use Case | Prefix(es) Detected | Fires? |
|----------|-------------------|--------|
| Music | `mbrainz_` | Yes |
| Games | `metacritic_` | Yes |
| Companies | `forbes_` | Yes |
| Restaurant | `kaggle380k-`, `yelp-`, `uber_` | Yes (multi-prefix) |
| Books (isbn_clean) | No prefix (plain ISBN) | No |

**Log output**: `[GUARDRAIL] Injected eval_id column (prefixes=['mbrainz_']) for evaluation alignment.`

---

### PG-8: ML Post-Clustering Safety

**What**: When `matcher_mode="ml"` and no post-clustering exists in the code, injects `MaximumBipartiteMatching` on each pairwise correspondence set before they are concatenated for fusion.

**Why**: ML matchers produce binary scores (0 or 1) and create much denser correspondence graphs than rule-based matchers. Without post-clustering, PyDI's DFS-based group builder hits Python's recursion limit (`RecursionError: maximum recursion depth exceeded`) on large connected components.

**When it fires**: Only when ALL three conditions are met:
1. `matcher_mode == "ml"` in state
2. No existing post-clustering in the code (no `MaximumBipartiteMatching`, `StableMatching`, etc.)
3. A `pd.concat()` call exists (correspondences are being merged)

Does NOT fire for rule-based mode, even with large clusters.

**What gets injected** (before the `pd.concat()` call):
```python
from PyDI.entitymatching import MaximumBipartiteMatching as _MBM
_post_clusterer = _MBM()
def _safe_cluster(corr_df):
    if corr_df is None or len(corr_df) == 0:
        return corr_df
    try:
        clustered = _post_clusterer.cluster(corr_df)
        if len(clustered) < len(corr_df):
            print(f"  Post-clustering: {len(corr_df)} -> {len(clustered)} correspondences")
        return clustered
    except Exception:
        return corr_df

ml_correspondences_discogs_lastfm = _safe_cluster(ml_correspondences_discogs_lastfm)
ml_correspondences_discogs_musicbrainz = _safe_cluster(ml_correspondences_discogs_musicbrainz)
ml_correspondences_musicbrainz_lastfm = _safe_cluster(ml_correspondences_musicbrainz_lastfm)
```

**Log output**: `[GUARDRAIL] Injected ML post-clustering safety (MaximumBipartiteMatching) for 3 correspondence set(s).`

---

### PG-9: Output Path Rewriting

**What**: Rewrites hardcoded `"output/"` paths to the run-scoped directory.

**Why**: Each run uses a unique directory (`output/runs/YYYYMMDD_HHMMSS_<usecase>/`). The LLM often hardcodes `"output/"` from the example pipeline, which would write to the wrong location and conflict with other runs.

**When it fires**: Every run. Scans for `"output/<subdir>"` patterns and replaces with the run-scoped path.

---

### PG-10: pd.isna() List Safety

**What**: Patches `pd.isna()` calls to handle list/array values without crashing, and injects a `_safe_scalar_isna()` helper function.

**Why**: When columns contain Python lists (e.g., after list normalization), `pd.isna([1, 2, 3])` returns an array instead of a scalar, causing `ValueError: The truth value of an array is ambiguous`. The LLM's `lower_strip()` preprocessing functions typically have unguarded `pd.isna(x)` checks.

**When it fires**: When the code contains `pd.isna` or `pd.notna` calls that don't already have list guards.

---

### PG-11: EmbeddingBlocker text_cols Flattening

**What**: Injects a `_flatten_list_cols_for_blocking()` helper that converts list-valued cells to strings before passing to EmbeddingBlocker.

**Why**: `EmbeddingBlocker` expects string values in its `text_cols` columns. After list normalization, these columns may contain Python lists, which the sentence transformer can't embed.

**When it fires**: When `EmbeddingBlocker` is used in the pipeline code.

---

## Evaluation Guardrails (`apply_evaluation_guardrails`)

**File**: `helpers/code_guardrails.py:923`
**Called at**: `pipeline_agent.py:1521`, before every evaluation subprocess execution
**Input**: generated evaluation code (string)
**Output**: corrected evaluation code (string)

---

### EG-1: Robust Import Resolution

**What**: Wraps `from list_normalization import ...` in a try/except with path fallback candidates.

**Why**: The `list_normalization.py` module is in the `agents/` directory, not in a standard package path. The evaluation script runs as a subprocess from the run directory, so the import may fail depending on working directory.

**When it fires**: When the evaluation code contains the bare `from list_normalization import` statement.

---

### EG-2: Output Path Rewriting

**What**: Same as PG-9 but for evaluation code. Rewrites hardcoded `"output/"` to run-scoped directory.

**When it fires**: Every run.

---

### EG-3: fused_id_column Alignment

**What**: Ensures the evaluation code uses the correct ID column for aligning fused records to gold records.

**Why**: The pipeline may or may not inject an `eval_id` column (see PG-7). The evaluation code must use the right column — `eval_id` when it exists, `_id` when it doesn't.

**Three scenarios**:

1. **Pipeline injected eval_id** (detected by reading fused CSV headers): patches `fused_id_column="_id"` to `fused_id_column="eval_id"`

2. **Pipeline did NOT inject eval_id, but eval code builds its own** (detected by scanning for `eval_id` assignment before `evaluator.evaluate()`): leaves the LLM's eval_id logic alone — it may be correct (e.g., restaurant with mixed prefixes)

3. **Pipeline did NOT inject eval_id, and eval code doesn't build one either**: reverts `fused_id_column="eval_id"` to `fused_id_column="_id"` to prevent broken alignment

**When it fires**: Every run, after checking the fused CSV for eval_id column presence.

**Log output**: `[GUARDRAIL] Patched fused_id_column to 'eval_id' (pipeline injected eval_id column).` or `[GUARDRAIL] Reverted fused_id_column to '_id' (no eval_id in pipeline or eval code).`

---

### EG-4: Sub-Column Stripping

**What**: Drops fused sub-columns (e.g., `tracks_track_name`, `tracks_track_duration`) when the gold set only has the parent column (`tracks`).

**Why**: When the gold set stores nested data as a single parent column but the fused output has derived sub-columns, evaluating those sub-columns produces 0% accuracy entries. These drag down the overall accuracy and waste investigation cycles trying to fix unfixable attributes.

**When it fires**: When `evaluator.evaluate(` is present and `_strip_unevaluable_subcols` is not already in the code. The injected function checks each fused column — if it contains `_` and its parent (prefix before first `_`) exists in the gold set but the full column name doesn't, it's dropped.

**Example (Music)**:
- Gold set has: `tracks` (single nested XML column)
- Fused has: `tracks_track_name`, `tracks_track_duration`, `tracks_track_position`
- These are dropped from evaluation

**What gets injected** (before `evaluator.evaluate()`):
```python
def _strip_unevaluable_subcols(fused_df, gold_df):
    gold_cols = set(gold_df.columns)
    to_drop = []
    for fc in fused_df.columns:
        if "_" in fc and fc not in gold_cols:
            parent = fc.split("_")[0]
            if parent in gold_cols:
                to_drop.append(fc)
    if to_drop:
        fused_df = fused_df.drop(columns=to_drop, errors="ignore")
    return fused_df
```

---

### EG-5: Numeric Type Coercion

**What**: Injects `pd.to_numeric()` + `Int64` cast for numeric-like columns in both fused and gold DataFrames.

**Why**: Prevents `274.0 != 274` and `2003.0 != 2003` false mismatches. Common when one source stores years/counts as floats and the gold stores them as integers. Without coercion, `numeric_tolerance_match` or `exact_match` reports these as mismatches.

**When it fires**: When `evaluator.evaluate(` is present and `_coerce_numeric_cols` is not already in the code.

**Columns affected**: Any column whose lowercased name contains: year, count, score, rating, page, sales, price, rank, assets, profits, revenue, duration, founded.

**What gets injected**:
```python
def _coerce_numeric_cols(*dfs):
    _NUMERIC_HINTS = {"year", "count", "score", "rating", "page", "sales", "price",
                      "rank", "assets", "profits", "revenue", "duration", "founded"}
    for df in dfs:
        for col in df.columns:
            if any(h in col.lower() for h in _NUMERIC_HINTS):
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.dropna().apply(lambda x: float(x).is_integer()).all():
                    df[col] = numeric.astype("Int64")
                else:
                    df[col] = numeric
```

---

### EG-6: List Separator Normalization

**What**: Aligns list-like column separators between fused and gold DataFrames.

**Why**: Different sources use different list separators (`, ` vs `; ` vs ` | `). When the fused output uses `; ` but the gold uses `, `, `set_equality_match` reports 0% because `"Rock; Pop"` != `"Rock, Pop"` even though the content is identical.

**When it fires**: When `evaluator.evaluate(` is present and `_normalize_list_separators` is not already in the code.

**Columns affected**: Any column whose lowercased name contains: genre, categor, tag, topic, keyword, subject, track.

**How it works**:
1. Samples 5 gold rows to detect the gold separator (tries `, `, `; `, ` | `, `|`, ` / `, `/`)
2. Samples 5 fused rows to detect the fused separator
3. If they differ, replaces the fused separator with the gold separator

**Example (Books)**:
```python
# Gold uses ", " for genres: "Mystery, Fiction, Thriller"
# Fused uses "; " for genres: "Mystery; Fiction; Thriller"
# After guardrail: fused genres become "Mystery, Fiction, Thriller"
```

---

## Guardrail Execution Order

### Pipeline guardrails (in order):
1. include_singletons=True (PG-1)
2. Matching threshold freezing (PG-2)
3. Fusion guidance enforcement + per-attribute trust maps (PG-3, PG-4)
4. Custom fuser signature patching
5. list_strategy validation (PG-5)
6. pd.isna() list safety (PG-10)
7. EmbeddingBlocker text_cols flattening (PG-11)
8. General import safety (PG-6)
9. Output path rewriting (PG-9)
10. Eval ID injection (PG-7)
11. ML post-clustering safety (PG-8)

### Evaluation guardrails (in order):
1. Robust import resolution (EG-1)
2. Output path rewriting (EG-2)
3. fused_id_column alignment (EG-3)
4. Sub-column stripping (EG-4)
5. Numeric type coercion (EG-5)
6. List separator normalization (EG-6)

### Idempotency
All guardrails check for their own markers before injecting (e.g., `if "_coerce_numeric_cols" not in updated`). Applying guardrails twice produces the same result as applying once.
