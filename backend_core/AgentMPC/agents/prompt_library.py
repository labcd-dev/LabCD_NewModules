"""
================================================================================
agents/prompt_library.py
================================================================================
Single access point for AgentMPC's prompt text, which lives in
``backend_core/AgentMPC/prompts/*.yaml`` -- one file per agent -- rather than
as string constants inside the agent modules.

This mirrors AgentPlant (``promptTemplate.yaml`` + ``labcd_agents.PromptLibrary``)
and AgentAdaptive (a ``prompts/`` directory, one YAML per agent). AgentMPC has
15 prompt blocks across 12 agents, so the per-file layout is used here; the
loader is AgentPlant's.

Why lazy (``lru_cache``) rather than a module-level ``PromptLibrary(...)``:
the AgentMPC agents are plain node *functions* (``critic_node(state)``), not
classes with an ``__init__`` to hook into, so the only alternative would be
reading from disk at import time -- which turns a missing or unreadable
prompts/ directory into an ImportError for the whole package, and therefore a
blank screen in the Streamlit app. Deferring the read to first use keeps the
failure at the call site, where it can be reported.

``labcd_agents.PromptLibrary`` is used when it is installed. The fallback
below exists because AgentMPC currently treats ``labcd_agents`` as optional
(see ``llm_base._compute_cost``), and prompt loading should not be the thing
that makes it mandatory.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class _FallbackPromptLibrary:
    """Minimal stand-in for ``labcd_agents.PromptLibrary`` (same two methods
    this module uses) for installs without the shared package."""

    def __init__(self, directory: str) -> None:
        import yaml

        self.directory = directory
        self._prompts: dict = {}
        for filename in sorted(os.listdir(directory)):
            if filename.endswith((".yaml", ".yml")):
                name = os.path.splitext(filename)[0]
                with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                    self._prompts[name] = yaml.safe_load(f)

    def get(self, name: str) -> Any:
        try:
            return self._prompts[name]
        except KeyError as exc:
            raise KeyError(f"No prompt file named '{name}.yaml' in {self.directory}") from exc

    def get_key(self, name: str, key: str) -> Any:
        try:
            return self.get(name)[key]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"No key '{key}' in prompt file '{name}.yaml'") from exc


@lru_cache(maxsize=1)
def get_prompt_library():
    """The (cached) prompt library rooted at ``AgentMPC/prompts/``."""
    try:
        from labcd_agents import PromptLibrary
    except ImportError:
        return _FallbackPromptLibrary(str(PROMPTS_DIR))
    return PromptLibrary(str(PROMPTS_DIR))


def get_prompt(name: str, key: str = "prompt") -> str:
    """Return the template string under ``key`` in ``prompts/<name>.yaml``.

    ``key`` defaults to ``"prompt"`` because most agents have exactly one.
    """
    return get_prompt_library().get_key(name, key)


def reload_prompts() -> None:
    """Drop the cache so the next ``get_prompt`` re-reads from disk.

    Useful when iterating on prompt wording against a long-lived Streamlit
    process, which is the whole point of having the text outside Python.
    """
    get_prompt_library.cache_clear()
