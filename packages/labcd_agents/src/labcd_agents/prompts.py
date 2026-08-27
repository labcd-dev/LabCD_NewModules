"""YAML prompt-template loading.

``MuloDesigner``, ``Regularizer`` and ``Recommender`` each implement an
identical ``_load_all_prompts(directory)`` method: scan a directory for
``*.yaml`` files, load each into a dict keyed by filename. This module
provides that as a small, reusable, testable class instead of a copy-pasted
private method.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

from .exceptions import PromptNotFoundError

__all__ = ["PromptLibrary"]


class PromptLibrary:
    """Loads and formats YAML prompt templates from a directory.

    Example:
        Given ``templates/system_analyser.yaml``::

            analyse_system: |
              Analyze this system: {equation}

        Usage::

            >>> prompts = PromptLibrary("templates")
            >>> prompts.format("system_analyser", "analyse_system", equation="dx/dt = -x")
            'Analyze this system: dx/dt = -x'
    """

    def __init__(self, directory: str, *, auto_load: bool = True) -> None:
        self.directory = directory
        self._prompts: Dict[str, Any] = {}
        if auto_load:
            self.load()

    def load(self, directory: Optional[str] = None) -> "PromptLibrary":
        """(Re)load every ``*.yaml`` file in ``directory`` (or ``self.directory``)."""
        target_dir = directory or self.directory
        self._prompts = {}
        for filename in sorted(os.listdir(target_dir)):
            if filename.endswith((".yaml", ".yml")):
                name = os.path.splitext(filename)[0]
                with open(os.path.join(target_dir, filename), "r", encoding="utf-8") as f:
                    self._prompts[name] = yaml.safe_load(f)
        return self

    def __contains__(self, name: str) -> bool:
        return name in self._prompts

    def get(self, name: str) -> Any:
        """Return the raw parsed contents of ``<name>.yaml``."""
        try:
            return self._prompts[name]
        except KeyError as exc:
            raise PromptNotFoundError(f"No prompt file named '{name}.yaml' in {self.directory}") from exc

    def get_key(self, name: str, key: str) -> Any:
        """Return ``self.get(name)[key]``, e.g. a specific template string or schema."""
        template_file = self.get(name)
        try:
            return template_file[key]
        except (KeyError, TypeError) as exc:
            raise PromptNotFoundError(f"No key '{key}' in prompt file '{name}.yaml'") from exc

    def format(self, prompt_name: str, key: str, **format_kwargs: Any) -> str:
        """Return ``self.get_key(prompt_name, key).format(**format_kwargs)``.

        Note the parameter names ``prompt_name``/``key`` (rather than
        ``name``) so a template variable literally called ``name`` (a very
        common one, e.g. ``{name}``) can still be passed through
        ``**format_kwargs`` without colliding with this method's own
        signature.
        """
        return self.get_key(prompt_name, key).format(**format_kwargs)

    @property
    def names(self) -> list:
        return sorted(self._prompts.keys())
