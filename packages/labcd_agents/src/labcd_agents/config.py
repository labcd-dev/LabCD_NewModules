"""Environment and credential helpers.

Every module in the LabCD monorepo currently repeats the same
``load_dotenv()`` + ``os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")``
dance at import time. That pattern is centralized here so it happens exactly
once per process, and so provider builders can ask "do I have a key for
this provider?" without duplicating ``os.getenv`` calls everywhere.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

__all__ = ["ensure_env_loaded", "get_api_key", "require_api_key"]

_load_lock = threading.Lock()
_loaded = False

# Provider name -> environment variable holding its API key.
DEFAULT_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def ensure_env_loaded(dotenv_path: Optional[str] = None) -> None:
    """Load a ``.env`` file into ``os.environ`` at most once per process.

    Safe to call from every module's import path (as the legacy code does);
    subsequent calls are no-ops. Never raises if ``python-dotenv`` can't find
    a ``.env`` file - missing local env files are expected in production.
    """
    global _loaded
    if _loaded:
        return
    with _load_lock:
        if _loaded:
            return
        try:
            from dotenv import load_dotenv

            if dotenv_path:
                load_dotenv(dotenv_path)
            else:
                load_dotenv()
        except ImportError:  # pragma: no cover - python-dotenv is a hard dep, but be defensive
            pass
        _loaded = True


def get_api_key(provider: str, env_var: Optional[str] = None) -> Optional[str]:
    """Return the API key for ``provider``, or ``None`` if unset.

    Args:
        provider: Provider key, e.g. ``"openai"``, ``"groq"``, ``"nvidia"``.
        env_var: Override the environment variable name to read from.
    """
    ensure_env_loaded()
    var_name = env_var or DEFAULT_ENV_VARS.get(provider, f"{provider.upper()}_API_KEY")
    return os.getenv(var_name)


def require_api_key(provider: str, env_var: Optional[str] = None) -> str:
    """Like :func:`get_api_key`, but raises ``RuntimeError`` if unset."""
    key = get_api_key(provider, env_var)
    if not key:
        var_name = env_var or DEFAULT_ENV_VARS.get(provider, f"{provider.upper()}_API_KEY")
        raise RuntimeError(
            f"Missing API key for provider '{provider}'. Set the {var_name} "
            "environment variable (or pass api_key= explicitly)."
        )
    return key
