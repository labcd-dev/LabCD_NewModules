"""Exception types raised by :mod:`labcd_agents`.

Keeping these distinct from generic ``Exception`` / provider SDK exceptions
makes it possible for callers (and LangGraph nodes) to catch "this is a
labcd_agents problem" separately from downstream provider errors.
"""

from __future__ import annotations


class LabCDAgentsError(Exception):
    """Base class for all errors raised by labcd_agents."""


class UnknownProviderError(LabCDAgentsError):
    """Raised when a model name cannot be mapped to a registered provider.

    Mirrors the "Warning: Unknown model ... defaulting to Groq client"
    fallback that several modules silently applied. labcd_agents prefers to
    fail loudly (or fall back only when a default provider was explicitly
    configured) so misconfigured model names are caught early.
    """


class LLMInvocationError(LabCDAgentsError):
    """Raised when an LLM call fails after exhausting all retry attempts.

    Wraps the original exception in ``__cause__`` so the underlying provider
    error is never lost.
    """

    def __init__(self, message: str, *, attempts: int, last_error: Exception | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class PromptNotFoundError(LabCDAgentsError):
    """Raised when a requested prompt template/key cannot be found."""
