"""Subprocess execution error classification into actionable categories."""
import re
from typing import Any, Dict

from config import ERROR_RAW_SNIPPET_LENGTH


def classify_execution_error(
    returncode: int,
    stderr: str,
    stdout: str = "",
    timed_out: bool = False,
) -> Dict[str, Any]:
    """Classifies a subprocess execution error into an actionable category.

    Returns a dict with keys:
      - category: "timeout"|"memory_error"|"import_error"|"syntax_error"|"data_error"|"runtime_crash"|"unknown"
      - suggestion: human-readable fix hint
      - retryable: bool -- whether retrying with code changes could help
      - max_retries: int -- recommended max retries for this error type
      - raw_error: str -- truncated raw error for context
      - detail: optional extra info (module name, signal number, etc.)
    """
    raw = (stderr or stdout or "").strip()
    raw_snippet = raw[:ERROR_RAW_SNIPPET_LENGTH] if raw else ""

    base = {"raw_error": raw_snippet}

    # 1. Timeout
    if timed_out or "TimeoutExpired" in raw:
        return {**base, "category": "timeout",
                "suggestion": "The script exceeded the time limit. Consider: reduce data size via sampling before fusion, optimize blocking to produce fewer candidate pairs, or increase the execution timeout.",
                "retryable": True, "max_retries": 1}

    # 2. Memory error
    if any(pat in raw for pat in ("MemoryError", "Cannot allocate memory", "signal 9", "SIGKILL")) or \
       (returncode == -9):
        return {**base, "category": "memory_error",
                "suggestion": "Out of memory. Reduce candidate pairs by tightening blocking strategy, sample data before fusion, or reduce batch sizes in matching/feature extraction.",
                "retryable": False, "max_retries": 0}

    # 3. Import error
    import_match = re.search(r"(?:ModuleNotFoundError|ImportError):\s*(?:No module named\s*['\"]?([^'\";\n]+)|cannot import name\s*['\"]([^'\"]+))", raw)
    if import_match:
        module = (import_match.group(1) or import_match.group(2) or "").strip().strip("'\"")
        return {**base, "category": "import_error", "detail": {"module": module},
                "suggestion": f"Missing import: '{module}'. Add the import statement at the top of the generated code, or use a fallback import pattern if the module path varies.",
                "retryable": True, "max_retries": 2}

    # 4. Syntax error
    if "SyntaxError" in raw:
        line_match = re.search(r"line (\d+)", raw)
        detail = {"line": int(line_match.group(1))} if line_match else {}
        return {**base, "category": "syntax_error", "detail": detail,
                "suggestion": "The generated code has a syntax error. Fix the Python syntax — check for unclosed brackets, bad indentation, or incomplete expressions.",
                "retryable": True, "max_retries": 2}

    # 5. Recursion error (large clusters from permissive matching)
    if "RecursionError" in raw or "maximum recursion depth exceeded" in raw:
        return {**base, "category": "data_error",
                "detail": {"error_type": "RecursionError"},
                "suggestion": (
                    "RecursionError during fusion — correspondences form clusters too large for "
                    "PyDI's recursive group builder. Apply post-clustering (e.g. "
                    "MaximumBipartiteMatching or GreedyOneToOneMatchingAlgorithm) to "
                    "each pairwise correspondence set BEFORE concatenating and passing to "
                    "DataFusionEngine.run(). This reduces 1:many relationships and prevents "
                    "deep recursion."
                ),
                "retryable": True, "max_retries": 2}

    # 6. Data error (various pandas/file/key errors)
    data_patterns = [
        (r"KeyError:\s*['\"]?([^'\";\n]+)", "KeyError"),
        (r"FileNotFoundError:\s*(.+?)(?:\n|$)", "FileNotFoundError"),
        (r"EmptyDataError", "EmptyDataError"),
        (r"ParserError", "ParserError"),
        (r"ValueError:\s*(.+?)(?:\n|$)", "ValueError"),
        (r"IndexError:\s*(.+?)(?:\n|$)", "IndexError"),
    ]
    for pattern, error_type in data_patterns:
        m = re.search(pattern, raw)
        if m:
            detail_val = m.group(1).strip() if m.lastindex else ""
            return {**base, "category": "data_error",
                    "detail": {"error_type": error_type, "value": detail_val[:200]},
                    "suggestion": f"{error_type}: Guard data access — check that columns exist before indexing, files exist before reading, and data types match expectations. Detail: {detail_val[:200]}",
                    "retryable": True, "max_retries": 2}

    # 7. Segfault/crash
    if returncode and returncode < 0:
        sig = abs(returncode)
        return {**base, "category": "runtime_crash", "detail": {"signal": sig},
                "suggestion": f"Process killed by signal {sig}. Possible native library crash — simplify pipeline operations, check for corrupt data files, or reduce data volume.",
                "retryable": False, "max_retries": 0}
    if any(pat in raw for pat in ("Segmentation fault", "core dumped", "SIGSEGV")):
        return {**base, "category": "runtime_crash", "detail": {"signal": "SIGSEGV"},
                "suggestion": "Segmentation fault in native code. Simplify pipeline operations or check for corrupt data files.",
                "retryable": False, "max_retries": 0}

    # 8. Unknown
    return {**base, "category": "unknown",
            "suggestion": "Unrecognized error. Examine the full error output and fix the root cause.",
            "retryable": True, "max_retries": 1}
