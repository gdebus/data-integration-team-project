"""Compatibility shim for workflow logging.

Canonical implementation lives at agents/workflow_logging.py.
"""

from _resolve_import import ensure_project_root
ensure_project_root()

from workflow_logging import *  # noqa: F401,F403
