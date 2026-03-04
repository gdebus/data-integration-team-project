"""Compatibility shim for workflow logging.

Canonical implementation lives at agents/workflow_logging.py.
"""

try:
    from workflow_logging import *  # noqa: F401,F403
except Exception:
    from agents.workflow_logging import *  # type: ignore # noqa: F401,F403
