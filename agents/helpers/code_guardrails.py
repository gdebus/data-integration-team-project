import re
from typing import Any, Dict


def apply_pipeline_guardrails(pipeline_code: str, state: Dict[str, Any]) -> str:
    """
    Apply generic safety guardrails to generated pipeline code.
    This function is intentionally dataset-agnostic.
    """
    updated_code = pipeline_code
    diagnostics = state.get("auto_diagnostics", {}) if isinstance(state, dict) else {}
    metrics = state.get("evaluation_metrics", {}) if isinstance(state, dict) else {}

    id_alignment = diagnostics.get("id_alignment", {}) if isinstance(diagnostics, dict) else {}
    mapped_ratio = id_alignment.get("mapped_coverage_ratio")
    debug_ratios = diagnostics.get("debug_reason_ratios", {}) if isinstance(diagnostics, dict) else {}
    missing_ratio = debug_ratios.get("missing_fused_value", 0.0)
    overall_acc = metrics.get("overall_accuracy")

    force_singletons = False
    try:
        if mapped_ratio is not None and float(mapped_ratio) < 0.85:
            force_singletons = True
    except Exception:
        pass
    try:
        if missing_ratio is not None and float(missing_ratio) > 0.20:
            force_singletons = True
    except Exception:
        pass
    try:
        if overall_acc is not None and float(overall_acc) < 0.60:
            force_singletons = True
    except Exception:
        pass

    if force_singletons:
        updated_code, _ = re.subn(
            r"(include_singletons\s*=\s*)False",
            r"\1True",
            updated_code,
            count=1,
        )

    # Ensure custom fusers are fully PyDI-compatible:
    # - callable signature supports resolver kwargs
    # - return shape is (value, confidence, metadata)
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

    # Ensure NumericComparator has list_strategy.
    def _ensure_numeric_list_strategy(match: re.Match) -> str:
        block = match.group(0)
        if "list_strategy" in block:
            return block
        return re.sub(
            r"\n([ \t]*)\)$",
            r"\n\1    list_strategy=\"average\",\n\1)",
            block,
            count=1,
        )

    updated_code = re.sub(
        r"NumericComparator\(\n(?:[ \t]+[^\n]*\n)+?[ \t]*\)",
        _ensure_numeric_list_strategy,
        updated_code,
        flags=re.MULTILINE,
    )

    # Ensure StringComparator has list_strategy.
    def _ensure_string_list_strategy(match: re.Match) -> str:
        block = match.group(0)
        if "list_strategy" in block:
            return block
        strategy = "concatenate"
        try:
            if re.search(r"similarity_function\s*=\s*[\"']jaccard[\"']", block):
                strategy = "set_jaccard"
        except Exception:
            strategy = "concatenate"
        return re.sub(
            r"\n([ \t]*)\)$",
            rf"\n\1    list_strategy=\"{strategy}\",\n\1)",
            block,
            count=1,
        )

    updated_code = re.sub(
        r"StringComparator\(\n(?:[ \t]+[^\n]*\n)+?[ \t]*\)",
        _ensure_string_list_strategy,
        updated_code,
        flags=re.MULTILINE,
    )

    # Patch unsafe pd.isna(x) pattern.
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

    updated_code = re.sub(
        r"(?m)^(?P<indent>[ \t]*)if pd\.isna\(x\):\n(?P=indent)[ \t]*return np\.nan",
        _patch_unsafe_pd_isna,
        updated_code,
    )

    if fuser_sig_updates:
        print(f"[GUARDRAIL] Updated {fuser_sig_updates} custom fuser signature(s) for kwargs compatibility.")
    if lambda_updates:
        print(f"[GUARDRAIL] Updated {lambda_updates} lambda fuser signature(s).")
    if wrapped_resolvers:
        print(f"[GUARDRAIL] Wrapped {wrapped_resolvers} custom resolver usage(s) with _pydi_safe_fuser.")

    return updated_code


def apply_evaluation_guardrails(evaluation_code: str) -> str:
    """Patch common portability/runtime issues in generated evaluation code."""
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
    return updated
