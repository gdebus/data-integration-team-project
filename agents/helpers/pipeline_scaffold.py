"""Pipeline scaffold: freeze infrastructure, only patch mutable sections.

After cycle 1 generates and successfully executes a pipeline, this module
extracts the frozen scaffold (imports, dataset loading, blocking, matching,
correspondence saving, fusion execution, output) and identifies the mutable
sections (post-clustering, fusion strategy + trust map).

On subsequent cycles the LLM only generates replacement code for the mutable
sections, and the agent splices them back into the frozen scaffold.

Uses CODE-LEVEL detection (regex on Python constructs like DataFusionStrategy,
engine.run, etc.) so it works even when the LLM omits section marker comments.
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Code-level landmark patterns for splitting a pipeline into frozen/mutable.
#
# The pipeline structure is always:
#   FROZEN PREFIX:  imports → dataset loading → blocking → matching →
#                   correspondence saving → pd.concat(...)
#   MUTABLE:        (optional post-clustering) → DataFusionStrategy →
#                   add_attribute_fuser calls → trust_map
#   FROZEN SUFFIX:  DataFusionEngine(...) → engine.run(...) → .to_csv(...)
# ---------------------------------------------------------------------------

# Detects where the mutable section STARTS — first line that's clearly
# about fusion strategy, trust map, or post-clustering.
_MUTABLE_START_PATTERNS = [
    # Post-clustering algorithm instantiation
    re.compile(r"^\s*(MaximumBipartiteMatching|StableMatching|GreedyOneToOneMatchingAlgorithm|HierarchicalClusterer|ConnectedComponentClusterer)\s*\("),
    re.compile(r"^\s*clusterer\s*="),
    re.compile(r"^\s*refined_"),
    # Section comment markers (if present)
    re.compile(r"^#\s*===.*(?:POST.?CLUSTER|DATA\s*FUSION|FUSION\s*STRATEGY)", re.IGNORECASE),
    # Trust map definition
    re.compile(r"^\s*trust_map\s*=\s*\{"),
    # Strategy creation
    re.compile(r"^\s*strategy\s*=\s*DataFusionStrategy\s*\("),
    re.compile(r"^\s*\w+\s*=\s*DataFusionStrategy\s*\("),
    # add_attribute_fuser calls
    re.compile(r"^\s*strategy\.add_attribute_fuser\s*\("),
    # Fusing Data print
    re.compile(r'^\s*print\s*\(\s*"Fusing Data"\s*\)'),
]

# Detects where the mutable section ENDS and frozen suffix starts —
# DataFusionEngine instantiation or the RUN FUSION section marker.
_SUFFIX_START_PATTERNS = [
    re.compile(r"^#\s*===.*RUN\s*FUSION", re.IGNORECASE),
    re.compile(r"^\s*FUSION_DIR\s*=\s*os\.path\.join\s*\("),
    re.compile(r"^\s*engine\s*=\s*DataFusionEngine\s*\("),
    re.compile(r"^\s*\w+\s*=\s*DataFusionEngine\s*\("),
]


def _find_line_matching(lines: list[str], patterns: list[re.Pattern], start: int = 0) -> int:
    """Return the index of the first line matching any pattern, or -1."""
    for i in range(start, len(lines)):
        for pat in patterns:
            if pat.search(lines[i]):
                return i
    return -1


def _find_last_correspondence_save(lines: list[str]) -> int:
    """Find the last line that's part of correspondence saving / pd.concat.

    Looks for:
    - .to_csv(...correspondences...)
    - pd.concat([...])
    - all_*correspondences = ...
    - CORR_DIR / correspondences directory setup

    Returns the index of the last such line, or -1.
    """
    corr_patterns = [
        re.compile(r"\.to_csv\s*\(.*correspondences", re.IGNORECASE),
        re.compile(r"pd\.concat\s*\(\s*\["),
        re.compile(r"^\s*all_\w*correspondences\s*="),
        re.compile(r"ignore_index\s*=\s*True"),
    ]
    last_idx = -1
    for i, line in enumerate(lines):
        for pat in corr_patterns:
            if pat.search(line):
                last_idx = i
                break
    return last_idx


def build_scaffold(pipeline_code: str) -> Optional[dict]:
    """Extract the frozen scaffold from a working pipeline.

    Uses code-level detection — does NOT require section marker comments.

    Returns a dict with:
      - "frozen_prefix": everything up to the mutable section
      - "frozen_suffix": DataFusionEngine + run + output
      - "mutable_code": the current fusion strategy / post-clustering code

    Returns None if detection fails.
    """
    lines = pipeline_code.split("\n")

    # --- Find mutable start ---
    mutable_start = _find_line_matching(lines, _MUTABLE_START_PATTERNS)

    if mutable_start == -1:
        # Fallback: look for DataFusionStrategy anywhere
        for i, line in enumerate(lines):
            if "DataFusionStrategy" in line:
                mutable_start = i
                break

    if mutable_start == -1:
        return None  # Can't find fusion strategy at all

    # Walk backwards from mutable_start to include the section comment and
    # any trust_map / print("Fusing Data") that precedes the strategy.
    while mutable_start > 0:
        prev = lines[mutable_start - 1].strip()
        if (
            prev == ""
            or prev.startswith("#")
            or prev.startswith("print")
            or prev.startswith("trust_map")
            or prev.startswith("clusterer")
            or prev.startswith("refined_")
        ):
            mutable_start -= 1
        else:
            break

    # --- Find suffix start (DataFusionEngine) ---
    suffix_start = _find_line_matching(lines, _SUFFIX_START_PATTERNS, start=mutable_start)

    if suffix_start == -1:
        # Fallback: find DataFusionEngine
        for i in range(mutable_start, len(lines)):
            if "DataFusionEngine" in lines[i]:
                suffix_start = i
                break

    if suffix_start == -1:
        return None  # Can't find fusion engine

    # Walk backwards from suffix_start to include its section comment / FUSION_DIR
    while suffix_start > mutable_start:
        prev = lines[suffix_start - 1].strip()
        if prev == "" or prev.startswith("#"):
            suffix_start -= 1
        else:
            break

    # Also walk forward from suffix to make sure we capture the FUSION_DIR/os.makedirs
    # that might precede engine = DataFusionEngine(...) but after the section comment
    # (these are part of the suffix, not the mutable section)

    frozen_prefix = "\n".join(lines[:mutable_start]).rstrip()
    mutable_code = "\n".join(lines[mutable_start:suffix_start]).rstrip()
    frozen_suffix = "\n".join(lines[suffix_start:]).rstrip()

    if not mutable_code.strip():
        return None

    return {
        "frozen_prefix": frozen_prefix,
        "frozen_suffix": frozen_suffix,
        "mutable_code": mutable_code,
    }


def assemble_pipeline(scaffold: dict, new_mutable_code: str) -> str:
    """Splice new mutable code into the frozen scaffold."""
    parts = [
        scaffold["frozen_prefix"],
        "",
        new_mutable_code.strip(),
        "",
        scaffold["frozen_suffix"],
    ]
    return "\n\n".join(p for p in parts if p.strip() or p == "")


def build_patch_prompt_context(scaffold: dict) -> str:
    """Build context string for the LLM in patch mode.

    Shows the full frozen pipeline for reference, clearly marking what the
    LLM should output (only the mutable sections).
    """
    lines = [
        "CURRENT PIPELINE CODE (for reference — DO NOT regenerate frozen sections):",
        "=" * 70,
        scaffold["frozen_prefix"],
        "",
        "# " + "=" * 66,
        "# >>>>>> MUTABLE SECTION BELOW — OUTPUT ONLY THIS PART <<<<<<",
        "# " + "=" * 66,
        scaffold["mutable_code"],
        "",
        "# " + "=" * 66,
        "# >>>>>> FROZEN SECTION BELOW — DO NOT INCLUDE IN YOUR OUTPUT <<<<<<",
        "# " + "=" * 66,
        scaffold["frozen_suffix"],
        "=" * 70,
    ]
    return "\n".join(lines)


def extract_mutable_from_response(response_code: str, scaffold: dict) -> str:
    """Extract just the mutable code from an LLM response.

    Handles three cases:
    1. LLM returned only the mutable code — return as-is.
    2. LLM returned the full pipeline — extract the mutable section.
    3. LLM returned something ambiguous — use code-level detection.
    """
    response_code = response_code.strip()

    # Quick check: does the response contain frozen infrastructure?
    has_blocking = bool(re.search(r"\b(StandardBlocker|TokenBlocker|EmbeddingBlocker|SortedNeighbourhoodBlocker)\s*\(", response_code))
    has_matching = bool(re.search(r"\b(RuleBasedMatcher|MLBasedMatcher)\s*\(\s*\)", response_code))
    has_load = bool(re.search(r"\bload_(csv|parquet|xml)\s*\(", response_code))

    if not has_blocking and not has_matching and not has_load:
        # Response doesn't contain frozen infrastructure — it's just the mutable code
        return response_code

    # LLM returned the full pipeline despite instructions.
    # Use build_scaffold to extract just the mutable part.
    response_scaffold = build_scaffold(response_code)
    if response_scaffold and response_scaffold["mutable_code"].strip():
        return response_scaffold["mutable_code"]

    # Fallback: return the full response (will be assembled but at least won't crash)
    return response_code


def needs_new_imports(new_mutable_code: str, frozen_prefix: str) -> list[str]:
    """Check if the new mutable code references symbols not imported in the frozen prefix.

    Returns a list of import lines that need to be added.
    """
    missing_imports = []

    fusion_symbols = {
        "voting": "from PyDI.fusion import voting",
        "longest_string": "from PyDI.fusion import longest_string",
        "shortest_string": "from PyDI.fusion import shortest_string",
        "most_complete": "from PyDI.fusion import most_complete",
        "median": "from PyDI.fusion import median",
        "average": "from PyDI.fusion import average",
        "maximum": "from PyDI.fusion import maximum",
        "minimum": "from PyDI.fusion import minimum",
        "sum_values": "from PyDI.fusion import sum_values",
        "most_recent": "from PyDI.fusion import most_recent",
        "earliest": "from PyDI.fusion import earliest",
        "union": "from PyDI.fusion import union",
        "intersection": "from PyDI.fusion import intersection",
        "intersection_k_sources": "from PyDI.fusion import intersection_k_sources",
        "prefer_higher_trust": "from PyDI.fusion import prefer_higher_trust",
        "favour_sources": "from PyDI.fusion import favour_sources",
        "random_value": "from PyDI.fusion import random_value",
        "weighted_voting": "from PyDI.fusion import weighted_voting",
    }

    clustering_symbols = {
        "MaximumBipartiteMatching": "from PyDI.entitymatching import MaximumBipartiteMatching",
        "StableMatching": "from PyDI.entitymatching import StableMatching",
        "GreedyOneToOneMatchingAlgorithm": "from PyDI.entitymatching import GreedyOneToOneMatchingAlgorithm",
        "HierarchicalClusterer": "from PyDI.entitymatching import HierarchicalClusterer",
        "ConnectedComponentClusterer": "from PyDI.entitymatching import ConnectedComponentClusterer",
    }

    all_symbols = {**fusion_symbols, **clustering_symbols}

    for symbol, import_line in all_symbols.items():
        if re.search(rf"\b{symbol}\b", new_mutable_code) and not re.search(rf"\b{symbol}\b", frozen_prefix):
            missing_imports.append(import_line)

    return missing_imports


def inject_imports(frozen_prefix: str, import_lines: list[str]) -> str:
    """Inject additional import lines into the frozen prefix.

    Finds the last import block and appends the new imports there.
    """
    if not import_lines:
        return frozen_prefix

    lines = frozen_prefix.split("\n")

    # Find the last import line
    last_import_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import_idx = i
        elif stripped.startswith(")"):
            # End of a multi-line import
            last_import_idx = i

    if last_import_idx == -1:
        return "\n".join(import_lines) + "\n" + frozen_prefix

    for imp in import_lines:
        last_import_idx += 1
        lines.insert(last_import_idx, imp)

    return "\n".join(lines)
