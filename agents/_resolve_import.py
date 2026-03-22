"""Puts the agents/ root on sys.path for reliable sibling imports.

Call ensure_project_root() at module level before importing siblings
like workflow_logging, list_normalization, config, etc.
"""

import os
import sys

_MARKER_FILE = "workflow_logging.py"
_resolved = False


def ensure_project_root() -> str:
    """Adds the agents/ root directory to sys.path if not already present.

    Returns the resolved root path.
    """
    global _resolved
    if _resolved:
        return _get_root()

    root = _get_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)
    _resolved = True
    return root


def _get_root() -> str:
    current = os.path.dirname(os.path.abspath(__file__))
    # Walks up at most 3 levels looking for the marker file
    for _ in range(4):
        if os.path.isfile(os.path.join(current, _MARKER_FILE)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    # Fallback to directory containing this file
    return os.path.dirname(os.path.abspath(__file__))
