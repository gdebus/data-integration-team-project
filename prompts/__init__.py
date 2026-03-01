"""Compatibility package to import prompt modules from ``agents/prompts``."""

from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_agents_prompts = _pkg_dir.parent / "agents" / "prompts"

if _agents_prompts.is_dir():
    __path__.append(str(_agents_prompts))  # type: ignore[name-defined]

