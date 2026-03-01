"""Compatibility package to import helper modules from ``agents/helpers``."""

from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_agents_helpers = _pkg_dir.parent / "agents" / "helpers"

if _agents_helpers.is_dir():
    __path__.append(str(_agents_helpers))  # type: ignore[name-defined]

