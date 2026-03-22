import os
import re
from typing import Any, Dict, List


def apply_pipeline_guardrails(pipeline_code: str, state: Dict[str, Any]) -> str:
    """Applies dataset-agnostic safety guardrails to generated pipeline code."""
    updated_code = pipeline_code
    # Full fused dataset is a hard invariant: always include singletons.
    updated_code, replaced_false = re.subn(
        r"(include_singletons\s*=\s*)False",
        r"\1True",
        updated_code,
    )
    if replaced_false:
        print("[GUARDRAIL] Forced include_singletons=True (full fused dataset invariant).")

    if "include_singletons" not in updated_code:
        updated_code = re.sub(
            r"(engine\.run\([^\)]*?id_column\s*=\s*[^,\)\n]+)",
            r"\1,\n    include_singletons=True",
            updated_code,
            count=1,
            flags=re.DOTALL,
        )

    # Freezes tested matching thresholds from matching_config.
    # SKIP when the investigation agent routed to matching — the LLM was given
    # permission to adjust thresholds within ±0.1 of configured values.
    inv_log = state.get("investigation_log", {}) if isinstance(state, dict) else {}
    inv_decision = inv_log.get("decision", {}) if isinstance(inv_log, dict) else {}
    matching_threshold_unlocked = (
        isinstance(inv_decision, dict)
        and inv_decision.get("next_node") == "run_matching_tester"
    )
    matching_config = state.get("matching_config", {}) if isinstance(state, dict) else {}
    strategies = matching_config.get("matching_strategies", {}) if isinstance(matching_config, dict) else {}
    if isinstance(strategies, dict) and not matching_threshold_unlocked:
        for pair_key, cfg in strategies.items():
            if not isinstance(cfg, dict) or "threshold" not in cfg:
                continue
            var_name = f"threshold_{pair_key}"
            try:
                threshold_value = float(cfg["threshold"])
            except Exception:
                continue
            updated_code = re.sub(
                rf"({re.escape(var_name)}\s*=\s*)([0-9]*\.?[0-9]+)",
                rf"\g<1>{threshold_value}",
                updated_code,
                count=1,
            )
    elif isinstance(strategies, dict) and matching_threshold_unlocked:
        # Validate that LLM adjustments are within ±0.1 bounds
        for pair_key, cfg in strategies.items():
            if not isinstance(cfg, dict) or "threshold" not in cfg:
                continue
            var_name = f"threshold_{pair_key}"
            try:
                original_threshold = float(cfg["threshold"])
            except Exception:
                continue
            match = re.search(rf"{re.escape(var_name)}\s*=\s*([0-9]*\.?[0-9]+)", updated_code)
            if match:
                try:
                    new_threshold = float(match.group(1))
                    if abs(new_threshold - original_threshold) > 0.1 + 1e-6:
                        # Clamp to ±0.1 range
                        clamped = max(original_threshold - 0.1, min(original_threshold + 0.1, new_threshold))
                        updated_code = re.sub(
                            rf"({re.escape(var_name)}\s*=\s*)[0-9]*\.?[0-9]+",
                            rf"\g<1>{clamped:.4f}",
                            updated_code,
                            count=1,
                        )
                        print(f"[GUARDRAIL] Clamped {var_name} adjustment from {new_threshold} to {clamped:.4f} (±0.1 limit from {original_threshold})")
                except (ValueError, TypeError):
                    pass
        print("[GUARDRAIL] Matching threshold adjustment permitted (stage diagnosis identified matching issue)")

    fusion_guidance = state.get("fusion_guidance", {}) if isinstance(state, dict) else {}
    attribute_strategies = (
        fusion_guidance.get("attribute_strategies", {}) if isinstance(fusion_guidance, dict) else {}
    )
    trust_map_available = bool(re.search(r"(?m)^\s*trust_map\s*=", updated_code))
    def _split_import_items(text: str) -> list[str]:
        cleaned = str(text or "").strip()
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1]
        parts = []
        for token in cleaned.split(","):
            item = token.strip()
            if item:
                parts.append(item)
        return parts

    def _collect_imported_fusion_symbols(code: str) -> set[str]:
        symbols = set()
        for match in re.finditer(
            r"from\s+PyDI\.fusion\s+import\s+(\([^\)]*\)|[^\n]+)",
            code,
            flags=re.MULTILINE | re.DOTALL,
        ):
            symbols.update(_split_import_items(match.group(1)))
        return symbols

    imported_fusion_symbols = _collect_imported_fusion_symbols(updated_code)

    def _ensure_fusion_import(code: str, symbol: str) -> str:
        if symbol in imported_fusion_symbols:
            return code
        multiline_match = re.search(
            r"from\s+PyDI\.fusion\s+import\s+\((?P<body>[^\)]*)\)",
            code,
            flags=re.MULTILINE | re.DOTALL,
        )
        if multiline_match:
            body = multiline_match.group("body")
            items = _split_import_items(body)
            if symbol not in items:
                items.append(symbol)
            indent_match = re.search(r"\n([ \t]*)\S", body)
            indent = indent_match.group(1) if indent_match else "    "
            rebuilt = "from PyDI.fusion import (\n"
            rebuilt += "".join(f"{indent}{item},\n" for item in items)
            rebuilt += ")"
            imported_fusion_symbols.add(symbol)
            return code[: multiline_match.start()] + rebuilt + code[multiline_match.end() :]

        single_match = re.search(r"(from\s+PyDI\.fusion\s+import\s+)([^\n]+)", code)
        if single_match:
            existing = _split_import_items(single_match.group(2))
            if symbol not in existing:
                existing.append(symbol)
            imported_fusion_symbols.add(symbol)
            return (
                code[: single_match.start()]
                + single_match.group(1)
                + ", ".join(existing)
                + code[single_match.end() :]
            )

        imported_fusion_symbols.add(symbol)
        return f"from PyDI.fusion import {symbol}\n" + code

    def _effective_resolver_name(resolver_name: str, trust_details: Dict[str, Any] | None = None) -> str:
        field_shape = str((trust_details or {}).get("field_shape", "")).strip().lower()
        effective = str(resolver_name or "").strip()
        if effective == "prefer_higher_trust" and not trust_map_available:
            effective = "intersection" if field_shape == "list_like" else "voting"
        return effective

    def _patch_attribute_fuser(
        code: str,
        attr_name: str,
        resolver_name: str,
        trust_details: Dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        pattern = re.compile(
            rf"(add_attribute_fuser\(\s*[\"']{re.escape(attr_name)}[\"']\s*,\s*)([A-Za-z_][A-Za-z0-9_]*)(?P<rest>[^\n]*\))"
        )

        def _repl(match: re.Match) -> str:
            rest = match.group("rest")
            replacement_resolver = _effective_resolver_name(resolver_name, trust_details)
            replacement_rest = rest
            if replacement_resolver == "prefer_higher_trust" and "trust_map=" not in replacement_rest:
                replacement_rest = replacement_rest[:-1] + ", trust_map=trust_map)"
            if replacement_resolver != "prefer_higher_trust":
                replacement_rest = re.sub(r",\s*trust_map\s*=\s*[^,\)]*", "", replacement_rest)
            return match.group(1) + replacement_resolver + replacement_rest

        return pattern.subn(_repl, code)

    built_in_resolvers = {
        "longest_string",
        "shortest_string",
        "most_complete",
        "average",
        "median",
        "maximum",
        "minimum",
        "sum_values",
        "most_recent",
        "earliest",
        "union",
        "intersection",
        "intersection_k_sources",
        "voting",
        "favour_sources",
        "random_value",
        "weighted_voting",
        "prefer_higher_trust",
    }

    guidance_updates = 0
    guidance_skipped = 0
    if isinstance(attribute_strategies, dict):
        for attr_name, strategy in attribute_strategies.items():
            if not isinstance(strategy, dict):
                continue
            recommended_fuser = str(strategy.get("recommended_fuser", "")).strip()
            field_shape = str(strategy.get("field_shape", "")).strip().lower()
            if field_shape == "list_like" and recommended_fuser == "voting":
                recommended_fuser = "prefer_higher_trust" if trust_map_available else "intersection"
            effective_fuser = _effective_resolver_name(recommended_fuser, strategy)
            if effective_fuser not in built_in_resolvers:
                continue

            # Check what the LLM already chose for this attribute
            existing_match = re.search(
                rf"add_attribute_fuser\(\s*[\"']{re.escape(str(attr_name))}[\"']\s*,\s*([A-Za-z_][A-Za-z0-9_]*)",
                updated_code,
            )
            strategy_confidence = float(strategy.get("confidence", 0.0)) if isinstance(strategy, dict) else 0.0
            if existing_match:
                current_fuser = existing_match.group(1)
                # High-confidence strategies from investigation override LLM choices
                if strategy_confidence >= 0.85:
                    if current_fuser == effective_fuser:
                        guidance_skipped += 1
                        continue  # Already correct, skip
                    # Otherwise fall through to patch
                elif current_fuser in built_in_resolvers and current_fuser != "voting":
                    # Low-confidence: respect LLM's built-in choice (unless voting on lists)
                    guidance_skipped += 1
                    continue

            updated_code = _ensure_fusion_import(updated_code, effective_fuser)
            updated_code, count = _patch_attribute_fuser(
                updated_code,
                str(attr_name),
                recommended_fuser,
                strategy if isinstance(strategy, dict) else {},
            )
            guidance_updates += count
    if guidance_skipped:
        print(f"[GUARDRAIL] Respected LLM's fuser choices for {guidance_skipped} attribute(s) (built-in already selected).")

    # Only patch the global trust_map if all prefer_higher_trust attributes agree
    # on the same trust ordering.  When attributes need different orderings (e.g.
    # label wants discogs>musicbrainz but duration wants musicbrainz>discogs),
    # a single global trust_map cannot satisfy both — skip the patch and let the
    # LLM handle per-attribute trust_maps via the prompt context.
    if trust_map_available and isinstance(attribute_strategies, dict):
        pht_trust_maps: list[dict] = []
        for attr_name, strategy in attribute_strategies.items():
            if not isinstance(strategy, dict):
                continue
            if str(strategy.get("recommended_fuser", "")).strip() != "prefer_higher_trust":
                continue
            td = strategy.get("trust_map", {})
            if isinstance(td, dict) and td:
                pht_trust_maps.append(td)

        # Check agreement: all trust_maps must rank sources in the same order
        trust_orderings_agree = True
        if len(pht_trust_maps) > 1:
            ref_ordering = sorted(pht_trust_maps[0].keys(), key=lambda k: -pht_trust_maps[0][k])
            for tm in pht_trust_maps[1:]:
                this_ordering = sorted(tm.keys(), key=lambda k: -tm.get(k, 0))
                if this_ordering != ref_ordering:
                    trust_orderings_agree = False
                    break

        if trust_orderings_agree and pht_trust_maps:
            # All agree — apply the first (or any, since they're the same ordering)
            consensus_map = pht_trust_maps[0]
            for source_name, score in consensus_map.items():
                updated_code = re.sub(
                    rf"([\"']{re.escape(str(source_name))}[\"']\s*:\s*)([0-9]*\.?[0-9]+)",
                    rf"\g<1>{float(score):.3f}",
                    updated_code,
                )
        elif pht_trust_maps:
            # Trust orderings conflict — inject per-attribute trust_map variables
            # so each attribute uses its own trust ordering instead of the shared one.
            print(f"[GUARDRAIL] Conflicting trust orderings for {len(pht_trust_maps)} attributes — injecting per-attribute trust_maps.")
            for attr_name, strategy in attribute_strategies.items():
                if not isinstance(strategy, dict):
                    continue
                if str(strategy.get("recommended_fuser", "")).strip() != "prefer_higher_trust":
                    continue
                attr_trust = strategy.get("trust_map", {})
                if not isinstance(attr_trust, dict) or not attr_trust:
                    continue

                # Build a per-attribute trust_map variable name
                safe_attr = re.sub(r"[^a-zA-Z0-9]", "_", str(attr_name))
                var_name = f"trust_map_{safe_attr}"

                # Build the dict literal
                items = ", ".join(f'"{k}": {v}' for k, v in attr_trust.items())
                var_def = f'{var_name} = {{{items}}}'

                # Insert the variable definition before the first add_attribute_fuser call
                first_fuser = re.search(r"(?m)^.*add_attribute_fuser\(", updated_code)
                if first_fuser:
                    # Only insert if variable doesn't already exist
                    if var_name not in updated_code:
                        updated_code = (
                            updated_code[: first_fuser.start()]
                            + var_def + "\n"
                            + updated_code[first_fuser.start():]
                        )

                # Rewrite this attribute's fuser call to use the per-attribute trust_map
                # Pattern: add_attribute_fuser("attr", prefer_higher_trust, trust_map=<anything>)
                updated_code = re.sub(
                    rf'(add_attribute_fuser\(\s*["\']' + re.escape(str(attr_name))
                    + rf'["\']\s*,\s*prefer_higher_trust\s*,\s*trust_map\s*=\s*)([A-Za-z_][A-Za-z0-9_]*)',
                    rf"\g<1>{var_name}",
                    updated_code,
                )

    # --- General import safety: ensure every built-in resolver used in
    #     add_attribute_fuser() calls is actually imported, even if the
    #     guardrails didn't patch it (the LLM may use a symbol it forgot
    #     to import).  --------------------------------------------------------
    all_used_resolvers = set(
        re.findall(
            r"add_attribute_fuser\(\s*[\"'][^\"']+[\"']\s*,\s*([A-Za-z_][A-Za-z0-9_]*)",
            updated_code,
        )
    )
    imported_fusion_symbols = _collect_imported_fusion_symbols(updated_code)
    missing_imports = all_used_resolvers & built_in_resolvers - imported_fusion_symbols
    for sym in sorted(missing_imports):
        updated_code = _ensure_fusion_import(updated_code, sym)
    if missing_imports:
        print(f"[GUARDRAIL] Auto-imported {len(missing_imports)} missing fusion symbol(s): {', '.join(sorted(missing_imports))}")

    # Patches custom fuser signatures for PyDI runtime compatibility.
    # Callable signature gets **kwargs support; wrappers follow the (value, confidence, metadata) contract.
    resolver_matches = []
    resolver_matches.extend(
        re.findall(
            r"add_attribute_fuser\(\s*[\"'][^\"']+[\"']\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:,|\))",
            updated_code,
        )
    )
    resolver_matches.extend(
        re.findall(
            r"add_attribute_fuser\([^)]*?\bresolver\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:,|\))",
            updated_code,
            flags=re.DOTALL,
        )
    )
    custom_resolvers = sorted(
        {
            r
            for r in resolver_matches
            if r not in built_in_resolvers and r != "_pydi_safe_fuser"
        }
    )

    def _patch_named_signature(code: str, fn_name: str) -> tuple[str, int]:
        pattern = re.compile(
            rf"(?m)^(?P<indent>[ \t]*)def\s+{re.escape(fn_name)}\s*\((?P<params>[^\)]*)\)\s*:"
        )

        def _repl(match: re.Match) -> str:
            indent = match.group("indent")
            params = match.group("params").strip()
            if not params:
                return f"{indent}def {fn_name}(**kwargs):"

            tokens = [t.strip() for t in params.split(",") if t.strip()]
            patched_tokens = []
            for tok in tokens:
                if re.match(r"^context(\s*:\s*[^=]+)?$", tok) and "=" not in tok:
                    if ":" in tok:
                        tok = f"{tok}=None"
                    else:
                        tok = "context=None"
                patched_tokens.append(tok)

            if not any(t.startswith("**kwargs") for t in patched_tokens):
                patched_tokens.append("**kwargs")

            return f"{indent}def {fn_name}({', '.join(patched_tokens)}):"

        return pattern.subn(_repl, code)

    fuser_sig_updates = 0
    for resolver in custom_resolvers:
        updated_code, count = _patch_named_signature(updated_code, resolver)
        fuser_sig_updates += count

    lambda_updates = 0
    updated_code, count = re.subn(
        r"lambda\s+inputs\s*,\s*context\s*,\s*\*\*kwargs\s*:",
        "lambda inputs, **kwargs:",
        updated_code,
    )
    lambda_updates += count
    updated_code, count = re.subn(
        r"lambda\s+inputs\s*,\s*context\s*:",
        "lambda inputs, **kwargs:",
        updated_code,
    )
    lambda_updates += count

    wrapped_resolvers = 0
    for resolver in custom_resolvers:
        updated_code, count = re.subn(
            rf"(add_attribute_fuser\(\s*[\"'][^\"']+[\"']\s*,\s*){re.escape(resolver)}(\s*(?:,|\)))",
            rf"\1_pydi_safe_fuser({resolver})\2",
            updated_code,
        )
        wrapped_resolvers += count
        updated_code, count = re.subn(
            rf"(add_attribute_fuser\([^)]*?\bresolver\s*=\s*){re.escape(resolver)}(\s*(?:,|\)))",
            rf"\1_pydi_safe_fuser({resolver})\2",
            updated_code,
            flags=re.DOTALL,
        )
        wrapped_resolvers += count

    if wrapped_resolvers and "def _pydi_safe_fuser(" not in updated_code:
        safe_fuser_block = """
def _pydi_safe_fuser(fn):
    \"\"\"Adapt scalar custom fusers to PyDI resolver contract.\"\"\"
    def _wrapped(values, **kwargs):
        try:
            try:
                result = fn(values, **kwargs)
            except TypeError:
                try:
                    result = fn(values, None, **kwargs)
                except TypeError:
                    result = fn(values)
        except Exception as e:
            fallback = values[0] if values else None
            return fallback, 0.1, {"error": str(e), "fallback": "first_value"}

        if isinstance(result, tuple) and len(result) == 3:
            return result
        return result, 1.0, {}

    return _wrapped
""".strip()
        anchor = re.search(r"(?m)^.*add_attribute_fuser\(", updated_code)
        if anchor:
            updated_code = (
                updated_code[: anchor.start()]
                + safe_fuser_block
                + "\n\n"
                + updated_code[anchor.start() :]
            )
        else:
            updated_code += "\n\n" + safe_fuser_block + "\n"

    # Upgrades low-quality list_strategy choices and fixes truly invalid values.
    # "concatenate" is technically valid for StringComparator but produces poor
    # match quality on real lists — upgrade to a set-aware strategy.
    _good_string_list_strategies = {"best_match", "set_jaccard", "set_overlap"}
    _valid_numeric_list_strategies = {"average", "best_match", "range_overlap", "set_jaccard"}
    _valid_date_list_strategies = {"closest_dates", "range_overlap", "average_dates", "latest_dates", "earliest_dates"}

    def _sanitize_comparator_list_strategy(match: re.Match, valid_set: set, default: str) -> str:
        block = match.group(0)
        ls_match = re.search(r"list_strategy\s*=\s*[\"']([^\"']+)[\"']", block)
        if ls_match:
            current = ls_match.group(1)
            if current not in valid_set:
                fixed = default
                # For StringComparator: prefer set_jaccard when jaccard similarity is in use
                if "StringComparator" in block and re.search(r"similarity_function\s*=\s*[\"']jaccard[\"']", block):
                    fixed = "set_jaccard"
                return block[:ls_match.start()] + f'list_strategy="{fixed}"' + block[ls_match.end():]
        return block

    list_strategy_updates = 0
    for comparator, valid_set, default in [
        ("StringComparator", _good_string_list_strategies, "best_match"),
        ("NumericComparator", _valid_numeric_list_strategies, "average"),
        ("DateComparator", _valid_date_list_strategies, "closest_dates"),
    ]:
        pattern = re.compile(
            rf"{comparator}\(\n(?:[ \t]+[^\n]*\n)+?[ \t]*\)",
            flags=re.MULTILINE,
        )
        updated_code, count = pattern.subn(
            lambda m, vs=valid_set, d=default: _sanitize_comparator_list_strategy(m, vs, d),
            updated_code,
        )
        list_strategy_updates += count
    if list_strategy_updates:
        print(f"[GUARDRAIL] Validated list_strategy in {list_strategy_updates} comparator call(s).")

    # Replaces bare pd.isna(x) with a safe wrapper that handles array-like values.
    def _patch_unsafe_pd_isna(match: re.Match) -> str:
        indent = match.group("indent")
        return (
            f"{indent}try:\n"
            f"{indent}    _is_na = pd.isna(x)\n"
            f"{indent}    if isinstance(_is_na, (list, tuple, set, dict)) or hasattr(_is_na, '__array__'):\n"
            f"{indent}        _is_na = False\n"
            f"{indent}    else:\n"
            f"{indent}        _is_na = bool(_is_na)\n"
            f"{indent}except Exception:\n"
            f"{indent}    _is_na = False\n"
            f"{indent}if _is_na:\n"
            f"{indent}    return np.nan"
        )

    # Only patch bare `if pd.isna(x):` that is NOT already inside a try block.
    # The LLM sometimes writes its own try/except guard — patching that creates
    # broken nested try blocks with wrong indentation.
    def _safe_patch_isna(match: re.Match) -> str:
        # Check if the line before our match is a `try:` — if so, skip
        start = match.start()
        preceding = updated_code[max(0, start - 80):start]
        preceding_lines = preceding.rstrip().rsplit("\n", 1)
        last_line = preceding_lines[-1].strip() if preceding_lines else ""
        if last_line == "try:":
            return match.group(0)  # already guarded, don't double-wrap
        return _patch_unsafe_pd_isna(match)

    updated_code = re.sub(
        r"(?m)^(?P<indent>[ \t]*)if pd\.isna\(x\):\n(?P=indent)[ \t]*return np\.nan",
        _safe_patch_isna,
        updated_code,
    )

    if fuser_sig_updates:
        print(f"[GUARDRAIL] Updated {fuser_sig_updates} custom fuser signature(s) for kwargs compatibility.")
    if lambda_updates:
        print(f"[GUARDRAIL] Updated {lambda_updates} lambda fuser signature(s).")
    if wrapped_resolvers:
        print(f"[GUARDRAIL] Wrapped {wrapped_resolvers} custom resolver usage(s) with _pydi_safe_fuser.")
    if guidance_updates:
        print(f"[GUARDRAIL] Applied {guidance_updates} evidence-driven attribute fuser update(s).")

    # EmbeddingBlocker uses text_cols=[...], NOT column=... (unlike other blockers).
    # Fix: column="X" → text_cols=["X"]  and  column=X → text_cols=[X]
    embedding_col_fixes = 0
    def _fix_embedding_column(m: re.Match) -> str:
        nonlocal embedding_col_fixes
        block = m.group(0)
        col_match = re.search(r'\bcolumn\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|(\[.*?\]))', block)
        if col_match:
            embedding_col_fixes += 1
            if col_match.group(3):
                # Already a list like column=[...], just rename to text_cols
                replacement = f"text_cols={col_match.group(3)}"
            else:
                col_val = col_match.group(1) or col_match.group(2)
                replacement = f'text_cols=["{col_val}"]'
            block = block[:col_match.start()] + replacement + block[col_match.end():]
        return block

    updated_code = re.sub(
        r"EmbeddingBlocker\([^)]*\)",
        _fix_embedding_column,
        updated_code,
        flags=re.DOTALL,
    )
    if embedding_col_fixes:
        print(f"[GUARDRAIL] Fixed {embedding_col_fixes} EmbeddingBlocker column= → text_cols= call(s).")

    # EmbeddingBlocker produces candidate pairs via .materialize(), not .block().
    embedding_vars = set(
        re.findall(
            r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*EmbeddingBlocker\s*\(",
            updated_code,
        )
    )
    embedding_block_updates = 0
    for var_name in sorted(embedding_vars):
        updated_code, count = re.subn(
            rf"\b{re.escape(var_name)}\.block\(\)",
            f"{var_name}.materialize()",
            updated_code,
        )
        embedding_block_updates += count
    if embedding_block_updates:
        print(f"[GUARDRAIL] Rewrote {embedding_block_updates} EmbeddingBlocker .block() call(s) to .materialize().")

    # EmbeddingBlocker crashes on list-valued cells (PyDI bug: pd.isna on lists).
    # Inject a helper that flattens list columns to strings before blocking.
    if 'EmbeddingBlocker' in updated_code and '_flatten_list_cols_for_blocking' not in updated_code:
        _helper = '''
def _flatten_list_cols_for_blocking(df, text_cols):
    """Flatten list-valued cells to strings so EmbeddingBlocker can embed them."""
    out = df.copy()
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: " ".join(str(v) for v in x) if isinstance(x, (list, tuple, set))
                else ("" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))
            )
    return out
'''
        # Insert after the last import/function-def before "Blocking" or first EmbeddingBlocker
        _insert_pos = updated_code.find('EmbeddingBlocker(')
        if _insert_pos > 0:
            # Find the start of the line
            _line_start = updated_code.rfind('\n', 0, _insert_pos)
            # Find the variable assignment on this line
            _assign_start = updated_code.rfind('\n', 0, _line_start)
            updated_code = updated_code[:_assign_start + 1] + _helper + updated_code[_assign_start + 1:]

        # Now wrap each EmbeddingBlocker df argument with the flattener
        # Pattern: EmbeddingBlocker(\n    <df_left>,\n    <df_right>,\n    text_cols=[...],
        def _wrap_embedding_blocker(m):
            full = m.group(0)
            # Extract text_cols value
            tc_match = re.search(r'text_cols\s*=\s*(\[[^\]]+\])', full)
            if not tc_match:
                return full
            text_cols = tc_match.group(1)
            # Find the two dataframe arguments (first two args)
            args_match = re.search(r'EmbeddingBlocker\s*\(\s*\n?\s*(\w+)\s*,\s*\n?\s*(\w+)\s*,', full)
            if not args_match:
                return full
            df_left, df_right = args_match.group(1), args_match.group(2)
            full = full.replace(
                f'{df_left},',
                f'_flatten_list_cols_for_blocking({df_left}, {text_cols}),',
                1,
            )
            full = full.replace(
                f'{df_right},',
                f'_flatten_list_cols_for_blocking({df_right}, {text_cols}),',
                1,
            )
            return full

        updated_code = re.sub(
            r'EmbeddingBlocker\s*\([^)]+\)',
            _wrap_embedding_blocker,
            updated_code,
            flags=re.DOTALL,
        )
        print("[GUARDRAIL] Injected list-flattening wrapper for EmbeddingBlocker text_cols.")

    # Fix lower_strip / preprocess functions that crash on list-valued cells.
    # pd.isna(x) raises ValueError when x is a list/array. Wrap in try/except.
    # SKIP if the LLM already wrapped pd.isna(x) in its own try/except block.
    if 'def lower_strip' in updated_code and 'pd.isna(x)' in updated_code:
        if 'if pd.isna(x):' in updated_code and 'except (ValueError' not in updated_code:
            # Check if pd.isna(x) is already inside a try block
            isna_idx = updated_code.find('if pd.isna(x):')
            preceding = updated_code[max(0, isna_idx - 80):isna_idx]
            preceding_stripped = preceding.rstrip().rsplit("\n", 1)[-1].strip()
            if preceding_stripped != "try:":
                updated_code = updated_code.replace(
                    'if pd.isna(x):',
                    'try:\n        _is_na = pd.isna(x)\n    except (ValueError, TypeError):\n        _is_na = False\n    if _is_na:',
                    1,  # only first occurrence
                )
                print("[GUARDRAIL] Fixed lower_strip() pd.isna() to handle list/array values.")

    # ── Broader list-safety: inject a safe null-check helper ──
    # After list normalization, columns like tracks_track_name contain Python lists.
    # Many patterns in LLM-generated code crash on these: pd.isna(x), pd.notna(x),
    # x is pd.NA, df[col].notna(), df[col].map(lambda x: ... if pd.notna(x) ...), etc.
    # Inject a safe helper and patch known crash patterns.
    _needs_safe_isna = (
        'pd.isna(' in updated_code or 'pd.notna(' in updated_code or 'pd.isnull(' in updated_code
    )
    if _needs_safe_isna and '_safe_scalar_isna' not in updated_code:
        _safe_isna_helper = '''
def _safe_scalar_isna(x):
    """Safe pd.isna() that handles list/array values without crashing."""
    if isinstance(x, (list, tuple, set)):
        return False  # a list is not null, even if it contains null elements
    try:
        result = pd.isna(x)
        if hasattr(result, '__len__') and not isinstance(result, str):
            return False  # array-like result from pd.isna on non-scalar
        return bool(result)
    except (ValueError, TypeError):
        return False
'''
        # Insert after imports
        _import_end = 0
        for _imp_match in re.finditer(r'^(?:import |from )', updated_code, re.MULTILINE):
            _line_end = updated_code.find('\n', _imp_match.start())
            if _line_end > _import_end:
                _import_end = _line_end
        if _import_end > 0:
            updated_code = updated_code[:_import_end + 1] + _safe_isna_helper + updated_code[_import_end + 1:]
            print("[GUARDRAIL] Injected _safe_scalar_isna() helper for list-safe null checking.")

        # Patch common patterns in lambda/map contexts:
        # `if pd.notna(x)` → `if not _safe_scalar_isna(x)`
        # `if pd.isna(x)` → `if _safe_scalar_isna(x)` (in lambdas, not in lower_strip which is already fixed)
        _lambda_notna_count = 0
        # Pattern: lambda ... pd.notna(x) ... (in .map, .apply, etc.)
        updated_code, _c = re.subn(
            r'(?<=lambda\s)([^:]*:\s*[^;\n]*?)pd\.notna\((\w+)\)',
            lambda m: m.group(0).replace(f'pd.notna({m.group(2)})', f'not _safe_scalar_isna({m.group(2)})'),
            updated_code,
        )
        _lambda_notna_count += _c
        updated_code, _c = re.subn(
            r'(?<=lambda\s)([^:]*:\s*[^;\n]*?)pd\.isna\((\w+)\)',
            lambda m: m.group(0).replace(f'pd.isna({m.group(2)})', f'_safe_scalar_isna({m.group(2)})'),
            updated_code,
        )
        _lambda_notna_count += _c
        if _lambda_notna_count:
            print(f"[GUARDRAIL] Patched {_lambda_notna_count} lambda pd.isna/pd.notna call(s) for list safety.")

    # Rewrite hardcoded "output/" paths to the run-scoped output directory.
    # Only rewrite known output subdirectories (not input paths or already-scoped paths).
    try:
        import config as _cfg
        run_output = _cfg.OUTPUT_DIR.rstrip("/")
        if run_output != "output" and run_output != "output/" and "/runs/" in run_output:
            path_rewrites = 0
            # Known output subdirectories that should be scoped to the run directory
            _OUTPUT_SUBDIRS = [
                "correspondences", "data_fusion", "pipeline_evaluation",
                "blocking-evaluation", "matching-evaluation", "cluster-evaluation",
                "normalization", "code", "profile", "human_review", "results",
                "snapshots", "evaluation", "investigation", "pipeline",
            ]
            for subdir in _OUTPUT_SUBDIRS:
                for quote in ['"', "'"]:
                    old_path = f'{quote}output/{subdir}'
                    new_path = f'{quote}{run_output}/{subdir}'
                    # Don't rewrite if it's already a runs/ path
                    if old_path in updated_code and f'{quote}{run_output}/{subdir}' not in updated_code:
                        updated_code = updated_code.replace(old_path, new_path)
                        path_rewrites += 1
            if path_rewrites:
                print(f"[GUARDRAIL] Rewrote {path_rewrites} hardcoded output/ path(s) → {run_output}/")
    except Exception:
        pass

    # ── Fix references to non-existent normalization paths ──
    # If the pipeline code references normalization output files that don't exist,
    # fall back to original dataset paths from the state dict.
    try:
        norm_path_pattern = re.compile(
            r'''(["'])((?:[^"']*?/)?normalization/attempt_\d+/[^"']+\.csv)\1'''
        )
        norm_matches = norm_path_pattern.findall(updated_code)
        if norm_matches:
            missing_norm_paths = [(q, p) for q, p in norm_matches if not os.path.exists(p)]
            if missing_norm_paths:
                # Extract original dataset paths from state
                original_datasets = state.get("original_datasets", state.get("datasets", []))
                if original_datasets:
                    # Build name → original path map
                    orig_map = {}
                    for op in original_datasets:
                        name = os.path.splitext(os.path.basename(op))[0].lower()
                        orig_map[name] = op

                    fixed = 0
                    for quote, norm_path in missing_norm_paths:
                        # Extract dataset name from the normalization path
                        norm_basename = os.path.splitext(os.path.basename(norm_path))[0].lower()
                        if norm_basename in orig_map:
                            updated_code = updated_code.replace(
                                f"{quote}{norm_path}{quote}",
                                f"{quote}{orig_map[norm_basename]}{quote}",
                            )
                            fixed += 1
                    if fixed:
                        print(f"[GUARDRAIL] Replaced {fixed} non-existent normalization path(s) with original dataset paths")
    except Exception:
        pass

    # ── Inject eval_id column extraction after fusion ──
    # The fused output uses cluster IDs (e.g. 'discogs_3') as _id, but the
    # validation set uses source-prefixed IDs (e.g. 'mbrainz_974').  The
    # evaluation LLM must map between them via _fusion_sources, but it often
    # gets this wrong.  Inject a deterministic eval_id extraction into the
    # pipeline code so the fused CSV always has a reliable eval_id column.
    # Detect validation prefix from state for eval_id extraction
    val_prefix = ""
    try:
        val_path = (state.get("validation_fusion_testset") or state.get("fusion_testset") or "")
        if val_path and os.path.exists(val_path):
            import pandas as _pd
            ext = os.path.splitext(val_path)[1].lower()
            if ext == ".xml":
                _val_df = _pd.read_xml(val_path)
            elif ext == ".csv":
                _val_df = _pd.read_csv(val_path, nrows=5)
            else:
                _val_df = None
            if _val_df is not None and "id" in _val_df.columns:
                _sample_id = str(_val_df["id"].iloc[0])
                # Try '_' separator first (e.g. 'mbrainz_974'), then '-' (e.g. 'yelp-06497')
                _parts = _sample_id.split("_", 1)
                if len(_parts) == 2 and _parts[0].isalpha():
                    val_prefix = _parts[0] + "_"
                else:
                    _parts = _sample_id.split("-", 1)
                    if len(_parts) == 2 and _parts[0].isalpha():
                        val_prefix = _parts[0] + "-"
    except Exception:
        pass

    if val_prefix and "eval_id" not in updated_code:
        # Collect ALL validation IDs (not just prefix) to handle mixed-prefix datasets
        _all_val_ids: set = set()
        _all_val_prefixes: set = set()
        try:
            _val_path_full = (state.get("validation_fusion_testset") or state.get("fusion_testset") or "")
            if _val_path_full and os.path.exists(_val_path_full):
                import pandas as _pd2
                _ext2 = os.path.splitext(_val_path_full)[1].lower()
                if _ext2 == ".xml":
                    _vdf2 = _pd2.read_xml(_val_path_full)
                elif _ext2 == ".csv":
                    _vdf2 = _pd2.read_csv(_val_path_full, dtype=str)
                else:
                    _vdf2 = None
                if _vdf2 is not None and "id" in _vdf2.columns:
                    _all_val_ids = set(_vdf2["id"].dropna().astype(str).tolist())
                    for _vid in _all_val_ids:
                        for _sep in ["_", "-"]:
                            _vparts = _vid.split(_sep, 1)
                            if len(_vparts) == 2:
                                _all_val_prefixes.add(_vparts[0] + _sep)
                                break
        except Exception:
            pass

        # Find the to_csv call that writes fusion_data.csv
        to_csv_match = re.search(
            r'(?m)^([ \t]*)(\w+)\.to_csv\([^)]*fusion_data\.csv[^)]*\)',
            updated_code,
        )
        if to_csv_match:
            indent = to_csv_match.group(1)
            var_name = to_csv_match.group(2)
            # Serialize prefixes for the injected code
            prefixes_repr = repr(sorted(_all_val_prefixes)) if _all_val_prefixes else repr([val_prefix])
            eval_id_block = (
                f'\n{indent}# --- Eval ID: extract source ID matching validation prefix(es) for reliable evaluation ---\n'
                f'{indent}import ast as _ast\n'
                f'{indent}_EVAL_PREFIXES = {prefixes_repr}\n'
                f'{indent}def _extract_eval_id(row):\n'
                f'{indent}    try:\n'
                f'{indent}        sources = _ast.literal_eval(str(row.get("_fusion_sources", "[]")))\n'
                f'{indent}        if isinstance(sources, (list, tuple)):\n'
                f'{indent}            for sid in sources:\n'
                f'{indent}                s = str(sid)\n'
                f'{indent}                if any(s.startswith(p) for p in _EVAL_PREFIXES):\n'
                f'{indent}                    return s\n'
                f'{indent}    except Exception:\n'
                f'{indent}        pass\n'
                f'{indent}    return str(row.get("_id", row.get("id", "")))\n'
                f'{indent}{var_name}["eval_id"] = {var_name}.apply(_extract_eval_id, axis=1)\n'
            )
            updated_code = (
                updated_code[:to_csv_match.start()]
                + eval_id_block
                + updated_code[to_csv_match.start():]
            )
            print(f"[GUARDRAIL] Injected eval_id column (prefixes={sorted(_all_val_prefixes) or [val_prefix]}) for evaluation alignment.")

    # --- ML matcher recursion safety: when matcher_mode is "ml", the ML matcher
    #     often produces dense correspondence graphs that cause RecursionError in
    #     PyDI's DFS-based group builder.  Inject a post-clustering step on each
    #     pairwise correspondence set BEFORE they are concatenated for fusion.
    #     Only injects if no post-clustering is already present in the code.  -----
    matcher_mode = state.get("matcher_mode", "rulebased") if isinstance(state, dict) else "rulebased"
    has_post_clustering = bool(re.search(
        r"(?:MaximumBipartiteMatching|StableMatching|GreedyOneToOneMatching|HierarchicalClusterer|ConnectedComponentClusterer)\s*\(",
        updated_code,
    ))
    has_concat = bool(re.search(r"pd\.concat\s*\(", updated_code))

    if matcher_mode == "ml" and not has_post_clustering and has_concat:
        # Find the pd.concat call that merges correspondence sets and inject
        # a MaximumBipartiteMatching step on each set just before it.
        concat_match = re.search(
            r"(?m)^(?P<indent>[ \t]*)(?P<var>\w+)\s*=\s*pd\.concat\s*\(\s*\[",
            updated_code,
        )
        if concat_match:
            indent = concat_match.group("indent")
            safety_block = (
                f"\n{indent}# [GUARDRAIL] Post-clustering safety for ML matcher — prevents RecursionError\n"
                f"{indent}from PyDI.entitymatching import MaximumBipartiteMatching as _MBM\n"
                f"{indent}_post_clusterer = _MBM()\n"
                f"{indent}def _safe_cluster(corr_df):\n"
                f"{indent}    if corr_df is None or len(corr_df) == 0:\n"
                f"{indent}        return corr_df\n"
                f"{indent}    try:\n"
                f"{indent}        clustered = _post_clusterer.cluster(corr_df)\n"
                f"{indent}        if len(clustered) < len(corr_df):\n"
                f'{indent}            print(f"  Post-clustering: {{len(corr_df)}} -> {{len(clustered)}} correspondences")\n'
                f"{indent}        return clustered\n"
                f"{indent}    except Exception:\n"
                f"{indent}        return corr_df\n"
            )

            # Find all correspondence variables in the concat list
            concat_list_match = re.search(
                r"pd\.concat\s*\(\s*\[([^\]]+)\]",
                updated_code[concat_match.start():],
            )
            if concat_list_match:
                var_list = concat_list_match.group(1)
                corr_vars = [v.strip() for v in var_list.split(",") if v.strip()]
                # Add _safe_cluster() call to each variable
                cluster_calls = ""
                for cv in corr_vars:
                    cluster_calls += f"{indent}{cv} = _safe_cluster({cv})\n"

                updated_code = (
                    updated_code[:concat_match.start()]
                    + safety_block + "\n"
                    + cluster_calls
                    + updated_code[concat_match.start():]
                )
                print(f"[GUARDRAIL] Injected ML post-clustering safety (MaximumBipartiteMatching) for {len(corr_vars)} correspondence set(s).")

    return updated_code


def apply_evaluation_guardrails(evaluation_code: str) -> str:
    """Patches common portability and runtime issues in generated evaluation code."""
    updated = evaluation_code
    list_norm_import = "from list_normalization import detect_list_like_columns, normalize_list_like_columns"
    if list_norm_import in updated and "except ModuleNotFoundError" not in updated:
        robust_import_block = """
import os
import sys
from pathlib import Path
try:
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
except ModuleNotFoundError:
    _module_name = "list_normalization.py"
    _candidates = [
        Path.cwd(),
        Path.cwd() / "agents",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
    ]
    for _path in _candidates:
        if (_path / _module_name).is_file():
            _path_str = str(_path.resolve())
            if _path_str not in sys.path:
                sys.path.append(_path_str)
    from list_normalization import detect_list_like_columns, normalize_list_like_columns
""".strip()
        updated = updated.replace(list_norm_import, robust_import_block, 1)

    # Rewrite hardcoded "output/" paths to run-scoped directory.
    try:
        import config as _cfg
        run_output = _cfg.OUTPUT_DIR.rstrip("/")
        if run_output != "output" and run_output != "output/" and "/runs/" in run_output:
            _OUTPUT_SUBDIRS = [
                "correspondences", "data_fusion", "pipeline_evaluation",
                "blocking-evaluation", "matching-evaluation", "cluster-evaluation",
                "normalization", "code", "profile", "human_review", "results",
                "snapshots", "evaluation", "investigation", "pipeline",
            ]
            for subdir in _OUTPUT_SUBDIRS:
                for quote in ['"', "'"]:
                    old_path = f'{quote}output/{subdir}'
                    new_path = f'{quote}{run_output}/{subdir}'
                    if old_path in updated:
                        updated = updated.replace(old_path, new_path)
    except Exception:
        pass

    # If the pipeline injected an eval_id column, ensure the evaluation code uses
    # fused_id_column="eval_id" instead of "_id" or "id" for reliable alignment.
    # IMPORTANT: Only patch to eval_id if the pipeline actually created that column.
    # For use cases where _id directly matches gold IDs (e.g. books using ISBN),
    # patching to eval_id breaks alignment because the eval code's own eval_id
    # builder can't map non-prefixed IDs through _fusion_sources.
    pipeline_has_eval_id = False
    try:
        import config as _cfg
        import os as _os
        _fused_path = _cfg.FUSED_OUTPUT_PATH
        if _os.path.exists(_fused_path):
            import pandas as _pd
            _fused_cols = _pd.read_csv(_fused_path, nrows=0).columns.tolist()
            pipeline_has_eval_id = "eval_id" in _fused_cols
    except Exception:
        pass

    if pipeline_has_eval_id and 'fused_id_column' in updated and 'eval_id' not in updated:
        # Replace fused_id_column="_id" or fused_id_column="id" with "eval_id"
        updated, n_replaced = re.subn(
            r'fused_id_column\s*=\s*["\'](_id|id)["\']',
            'fused_id_column="eval_id"',
            updated,
        )
        if n_replaced:
            print(f"[GUARDRAIL] Patched fused_id_column to 'eval_id' (pipeline injected eval_id column).")
        # Also handle cases where the LLM builds an eval_view but uses _id
        updated, n2 = re.subn(
            r'fused_id_column\s*=\s*["\']_fusion_id["\']',
            'fused_id_column="eval_id"',
            updated,
        )
    elif not pipeline_has_eval_id and 'fused_id_column' in updated:
        # Pipeline did NOT inject eval_id. Check if the LLM's eval code builds
        # its own eval_id column at runtime — if so, leave it alone (it may be
        # correct, e.g. for restaurants with mixed prefixes).
        eval_code_builds_eval_id = bool(re.search(
            r'(?:eval_id|"eval_id"|\'eval_id\')\s*[=\]]',
            updated.split('evaluator.evaluate')[0] if 'evaluator.evaluate' in updated else updated,
        ))
        if not eval_code_builds_eval_id:
            updated, n_reverted = re.subn(
                r'fused_id_column\s*=\s*["\']eval_id["\']',
                'fused_id_column="_id"',
                updated,
            )
            if n_reverted:
                print(f"[GUARDRAIL] Reverted fused_id_column to '_id' (no eval_id in pipeline or eval code).")

    # ── Mechanical evaluation fixes ────────────────────────────────────────
    # These inject code blocks into the evaluation script to fix systematic
    # issues that prompt-based guidance alone doesn't reliably prevent.

    # 1. Strip unevaluable sub-columns: if the gold set has a parent column
    #    (e.g. "tracks") but the fused output has derived sub-columns
    #    (e.g. "tracks_track_name"), drop those sub-columns from evaluation
    #    since they produce 0% accuracy and waste investigation cycles.
    if "evaluator.evaluate(" in updated and "_strip_unevaluable_subcols" not in updated:
        subcol_block = '''
# [GUARDRAIL] Strip unevaluable sub-columns from evaluation
def _strip_unevaluable_subcols(fused_df, gold_df):
    """Drop fused sub-columns that have no matching gold column (e.g. tracks_track_name when gold only has tracks)."""
    gold_cols = set(gold_df.columns)
    fused_cols = list(fused_df.columns)
    to_drop = []
    for fc in fused_cols:
        if "_" in fc and fc not in gold_cols:
            parent = fc.split("_")[0]
            if parent in gold_cols:
                to_drop.append(fc)
    if to_drop:
        fused_df = fused_df.drop(columns=to_drop, errors="ignore")
        print(f"  Dropped {len(to_drop)} unevaluable sub-column(s): {to_drop}")
    return fused_df
'''
        # Inject before the evaluator.evaluate() call
        eval_call = re.search(r"(?m)^.*evaluator\.evaluate\(", updated)
        if eval_call:
            # Find the fused_df variable name from the evaluate call
            fused_var_match = re.search(
                r"evaluator\.evaluate\(\s*(?:fused_df\s*=\s*)?(\w+)",
                updated[eval_call.start():]
            )
            fused_var = fused_var_match.group(1) if fused_var_match else "fused_eval"
            gold_var_match = re.search(
                r"(?:gold_df|expected_df)\s*=\s*(\w+)",
                updated[eval_call.start():]
            )
            gold_var = gold_var_match.group(1) if gold_var_match else "fusion_test_set"

            updated = (
                updated[:eval_call.start()]
                + subcol_block + "\n"
                + f"{fused_var} = _strip_unevaluable_subcols({fused_var}, {gold_var})\n"
                + updated[eval_call.start():]
            )

    # 2. Numeric type coercion: inject pd.to_numeric + Int64 cast for
    #    numeric-like columns to prevent 274.0 != 274 mismatches.
    if "evaluator.evaluate(" in updated and "_coerce_numeric_cols" not in updated:
        coerce_block = '''
# [GUARDRAIL] Coerce numeric-like columns to consistent types
def _coerce_numeric_cols(*dfs):
    """Cast numeric-like columns to Int64 in all DataFrames to prevent float vs int mismatches."""
    _NUMERIC_HINTS = {"year", "count", "score", "rating", "page", "sales", "price",
                      "rank", "assets", "profits", "revenue", "duration", "founded"}
    for df in dfs:
        for col in df.columns:
            col_l = col.lower().replace("_", "").replace("-", "")
            if any(h in col_l for h in _NUMERIC_HINTS):
                try:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    if numeric.dropna().empty:
                        continue
                    if numeric.dropna().apply(lambda x: float(x).is_integer()).all():
                        df[col] = numeric.astype("Int64")
                    else:
                        df[col] = numeric
                except Exception:
                    pass
'''
        eval_call = re.search(r"(?m)^.*evaluator\.evaluate\(", updated)
        if eval_call:
            fused_var_match = re.search(
                r"evaluator\.evaluate\(\s*(?:fused_df\s*=\s*)?(\w+)",
                updated[eval_call.start():]
            )
            fused_var = fused_var_match.group(1) if fused_var_match else "fused_eval"
            gold_var_match = re.search(
                r"(?:gold_df|expected_df)\s*=\s*(\w+)",
                updated[eval_call.start():]
            )
            gold_var = gold_var_match.group(1) if gold_var_match else "fusion_test_set"

            updated = (
                updated[:eval_call.start()]
                + coerce_block + "\n"
                + f"_coerce_numeric_cols({fused_var}, {gold_var})\n"
                + updated[eval_call.start():]
            )

    # 3. List separator normalization: standardize separators in list-like
    #    columns (genres, categories, tags) to prevent ", " vs "; " mismatches.
    if "evaluator.evaluate(" in updated and "_normalize_list_separators" not in updated:
        sep_block = '''
# [GUARDRAIL] Normalize list separators to match gold standard format
def _normalize_list_separators(fused_df, gold_df):
    """Align list-like column separators between fused and gold DataFrames."""
    _LIST_HINTS = {"genre", "categor", "tag", "topic", "keyword", "subject", "track"}
    shared = set(fused_df.columns) & set(gold_df.columns)
    for col in shared:
        col_l = col.lower()
        if not any(h in col_l for h in _LIST_HINTS):
            continue
        gold_sample = gold_df[col].dropna().astype(str).head(5).tolist()
        if not gold_sample:
            continue
        # Detect gold separator
        gold_sep = ", "
        for sep in ["; ", " | ", "|", " / ", "/"]:
            if any(sep in s for s in gold_sample):
                gold_sep = sep
                break
        # Normalize fused to match gold separator
        fused_sample = fused_df[col].dropna().astype(str).head(5).tolist()
        for sep in ["; ", " | ", "|", " / ", "/"]:
            if sep != gold_sep and any(sep in s for s in fused_sample):
                fused_df[col] = fused_df[col].astype(str).str.replace(sep, gold_sep)
                print(f"  Normalized {col} separator: '{sep}' -> '{gold_sep}'")
                break
    return fused_df
'''
        eval_call = re.search(r"(?m)^.*evaluator\.evaluate\(", updated)
        if eval_call:
            fused_var_match = re.search(
                r"evaluator\.evaluate\(\s*(?:fused_df\s*=\s*)?(\w+)",
                updated[eval_call.start():]
            )
            fused_var = fused_var_match.group(1) if fused_var_match else "fused_eval"
            gold_var_match = re.search(
                r"(?:gold_df|expected_df)\s*=\s*(\w+)",
                updated[eval_call.start():]
            )
            gold_var = gold_var_match.group(1) if gold_var_match else "fusion_test_set"

            updated = (
                updated[:eval_call.start()]
                + sep_block + "\n"
                + f"{fused_var} = _normalize_list_separators({fused_var}, {gold_var})\n"
                + updated[eval_call.start():]
            )

    return updated


def static_pipeline_sanity_findings(pipeline_code: str, correspondences_dir: str = "output/correspondences") -> List[Dict[str, str]]:
    """Runs lightweight static checks on generated pipeline code and returns findings."""
    findings: List[Dict[str, str]] = []
    candidate_vars = set()

    for pattern in (
        r"for\s+([A-Za-z_][A-Za-z0-9_]*)\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s+in\s+zip\(\s*datasets\s*,",
        r"for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+datasets\b",
    ):
        for match in re.finditer(pattern, pipeline_code):
            candidate_vars.add(match.group(1))

    for var_name in sorted(candidate_vars):
        if re.search(rf"\b{re.escape(var_name)}\.name\b", pipeline_code):
            findings.append(
                {
                    "severity": "critical",
                    "code": "dataframe_name_attribute",
                    "message": (
                        f"Variable `{var_name}` is iterating over `datasets` and accessed via "
                        f"`{var_name}.name`. For pandas DataFrames this resolves through column "
                        "attribute access and is not a safe dataset identifier."
                    ),
                }
            )

    # Check for correspondence file saves — match any path ending in correspondences/<filename>.csv
    corr_write_matches = re.findall(
        r"correspondences/([^\"'\n]+\.csv)",
        pipeline_code,
    )
    corr_write_paths = re.findall(
        r"output/correspondences/([^\"'\n]+\.csv)",
        pipeline_code,
    )
    # Also count .to_csv calls on variables named *correspondences*
    corr_to_csv_count = len(re.findall(
        r"correspondences[A-Za-z0-9_]*\.to_csv\(",
        pipeline_code,
    ))
    has_any_corr_save = bool(corr_write_matches) or corr_to_csv_count > 0

    if not has_any_corr_save:
        findings.append(
            {
                "severity": "critical",
                "code": "missing_correspondence_saves",
                "message": (
                    "Pipeline does NOT save per-pair correspondence files. "
                    f"For each dataset pair, save a CSV to `{correspondences_dir}/correspondences_<left>_<right>.csv` "
                    "so that cluster analysis and structural integrity checks can find them. "
                    "Save BEFORE merging into all_correspondences."
                ),
            }
        )
    elif corr_write_paths:
        nonstandard_corr_paths = [
            p for p in corr_write_paths
            if not re.fullmatch(r"correspondences_[A-Za-z0-9_]+_[A-Za-z0-9_]+\.csv", p)
        ]
        if nonstandard_corr_paths:
            findings.append(
                {
                    "severity": "high",
                    "code": "nonstandard_correspondence_filename",
                    "message": (
                        "Pipeline writes correspondence artifacts using nonstandard filenames. "
                        f"Use exactly `{correspondences_dir}/correspondences_<left>_<right>.csv` for every pair "
                        f"so structural checks and cluster analysis can find them. Found: {sorted(set(nonstandard_corr_paths))}."
                    ),
                }
            )

    embedding_vars = set(
        re.findall(
            r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*EmbeddingBlocker\s*\(",
            pipeline_code,
        )
    )
    for var_name in sorted(embedding_vars):
        if re.search(rf"\b{re.escape(var_name)}\.block\(\)", pipeline_code):
            findings.append(
                {
                    "severity": "critical",
                    "code": "embedding_blocker_block_call",
                    "message": (
                        f"`{var_name}` is an EmbeddingBlocker but the code calls `{var_name}.block()`. "
                        "In this PyDI environment EmbeddingBlocker must materialize candidates via `.materialize()`."
                    ),
                }
            )

    # EmbeddingBlocker uses text_cols=[...], NOT column=...
    if re.search(r"EmbeddingBlocker\([^)]*\bcolumn\s*=", pipeline_code, flags=re.DOTALL):
        findings.append(
            {
                "severity": "critical",
                "code": "embedding_blocker_wrong_param",
                "message": (
                    "EmbeddingBlocker uses `text_cols=['col1', 'col2']` (a list), NOT `column='col'`. "
                    "Replace `column=` with `text_cols=[...]`."
                ),
            }
        )

    if re.search(r"['\"]id_l['\"].*['\"]id_r['\"]|['\"]id_r['\"].*['\"]id_l['\"]", pipeline_code, flags=re.DOTALL):
        findings.append(
            {
                "severity": "critical",
                "code": "non_pydi_correspondence_schema",
                "message": (
                    "Pipeline code is asserting or depending on correspondence columns `id_l`/`id_r`. "
                    "PyDI fusion expects `id1`/`id2` for correspondence DataFrames."
                ),
            }
        )

    built_in_resolvers = {
        "longest_string",
        "shortest_string",
        "most_complete",
        "average",
        "median",
        "maximum",
        "minimum",
        "sum_values",
        "most_recent",
        "earliest",
        "union",
        "intersection",
        "intersection_k_sources",
        "voting",
        "favour_sources",
        "random_value",
        "weighted_voting",
        "prefer_higher_trust",
    }
    resolver_matches = []
    resolver_matches.extend(
        re.findall(
            r"add_attribute_fuser\(\s*[\"'][^\"']+[\"']\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:,|\))",
            pipeline_code,
        )
    )
    resolver_matches.extend(
        re.findall(
            r"add_attribute_fuser\([^)]*?\bresolver\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:,|\))",
            pipeline_code,
            flags=re.DOTALL,
        )
    )
    custom_resolvers = sorted(
        {
            r
            for r in resolver_matches
            if r not in built_in_resolvers and r != "_pydi_safe_fuser"
        }
    )
    wrapped_custom_resolver_count = len(re.findall(r"_pydi_safe_fuser\(", pipeline_code))
    if custom_resolvers or wrapped_custom_resolver_count > 1:
        findings.append(
            {
                "severity": "high",
                "code": "custom_fuser_overuse",
                "message": (
                    "Pipeline relies on custom fusion helpers instead of mainly using PyDI built-in fusers. "
                    f"Detected custom resolvers={custom_resolvers or []}, wrapped_custom_resolver_count={wrapped_custom_resolver_count}."
                ),
            }
        )

    include_singletons_false = re.search(r"include_singletons\s*=\s*False", pipeline_code)
    if include_singletons_false:
        findings.append(
            {
                "severity": "critical",
                "code": "matched_only_fusion",
                "message": (
                    "Pipeline code sets `include_singletons=False`. The agent must always produce the full "
                    "fused dataset, so fusion must run with `include_singletons=True`."
                ),
            }
        )

    return findings
