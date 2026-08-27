"""Token usage extraction and accumulation.

Every module extracts token counts from an LLM response slightly
differently depending on which of LangChain's usage-reporting conventions
the underlying provider integration uses:

- ``AIMessage.usage_metadata`` (``input_tokens`` / ``output_tokens``) - the
  modern LangChain convention, used by most chat model integrations.
- ``AIMessage.response_metadata["usage"]`` (``prompt_tokens`` /
  ``completion_tokens``) - an older convention some providers still emit.
- ``AIMessage.response_metadata["token_usage"]`` - LangChain's OpenAI
  integration historically nested usage under this key.
- The raw OpenAI *Responses* API (``client.responses.create(...)``), whose
  result exposes ``response.usage.input_tokens`` / ``.output_tokens`` /
  ``.total_tokens`` directly (no ``AIMessage`` wrapper at all).

:func:`extract_usage` understands all four shapes so agent code never has to
duplicate the ``hasattr`` chain again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["TokenUsage", "extract_usage", "TokenTracker"]


@dataclass(frozen=True)
class TokenUsage:
    """Input/output token counts for a single LLM call (or a running total)."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def as_dict(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }

    # Aliases matching the "prompt_tokens/completion_tokens" naming some
    # modules use for their internal state dicts (e.g. LangGraph state),
    # so callers can migrate without renaming every downstream consumer.
    def as_prompt_completion_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
        }


def _usage_from_mapping(usage: Dict[str, Any]) -> Optional[TokenUsage]:
    if "input_tokens" in usage or "output_tokens" in usage:
        return TokenUsage(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
        )
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        return TokenUsage(
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
        )
    return None


def extract_usage(response: Any) -> TokenUsage:
    """Best-effort extraction of a :class:`TokenUsage` from an LLM response.

    Supports LangChain ``AIMessage`` objects (both ``usage_metadata`` and
    the legacy ``response_metadata`` shapes) as well as raw OpenAI
    ``responses.create(...)`` results. Returns ``TokenUsage(0, 0)`` if no
    usage information could be found, rather than raising - usage reporting
    is best-effort and should never break the calling agent.
    """
    if response is None:
        return TokenUsage()

    # 1. LangChain AIMessage.usage_metadata (modern convention)
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata:
        found = _usage_from_mapping(dict(usage_metadata))
        if found:
            return found

    # 2. LangChain AIMessage.response_metadata (legacy shapes)
    response_metadata = getattr(response, "response_metadata", None)
    if response_metadata:
        if "token_usage" in response_metadata:
            found = _usage_from_mapping(dict(response_metadata["token_usage"]))
            if found:
                return found
        if "usage" in response_metadata:
            found = _usage_from_mapping(dict(response_metadata["usage"]))
            if found:
                return found

    # 3. Raw OpenAI Responses API object: response.usage.{input,output,total}_tokens
    usage_obj = getattr(response, "usage", None)
    if usage_obj is not None:
        input_tokens = getattr(usage_obj, "input_tokens", None)
        output_tokens = getattr(usage_obj, "output_tokens", None)
        if input_tokens is not None or output_tokens is not None:
            return TokenUsage(input_tokens=int(input_tokens or 0), output_tokens=int(output_tokens or 0))
        if isinstance(usage_obj, dict):
            found = _usage_from_mapping(usage_obj)
            if found:
                return found

    # 4. Plain dict response (e.g. already-normalized usage payload)
    if isinstance(response, dict):
        found = _usage_from_mapping(response)
        if found:
            return found

    return TokenUsage()


@dataclass
class _UsageRecord:
    usage: TokenUsage
    model: Optional[str] = None
    cost: float = 0.0


class TokenTracker:
    """Accumulates :class:`TokenUsage` (and optionally cost) across calls.

    Replaces the ad-hoc ``state["token_usage"]`` / ``state["total_cost"]``
    dict-mutation pattern duplicated in ``Recommender`` and ``Trimmer`` with
    a small reusable object. Because LangGraph nodes work with plain state
    dicts, :meth:`as_state_update` returns a dict in that exact shape so it
    can still be merged into LangGraph state via ``**metrics``.
    """

    def __init__(self) -> None:
        self._history: List[_UsageRecord] = []

    def record(self, usage: TokenUsage, *, model: Optional[str] = None, cost: float = 0.0) -> None:
        """Record one LLM call's usage (and optionally its dollar cost)."""
        self._history.append(_UsageRecord(usage=usage, model=model, cost=cost))

    @property
    def totals(self) -> TokenUsage:
        total = TokenUsage()
        for record in self._history:
            total = total + record.usage
        return total

    @property
    def total_cost(self) -> float:
        return sum(record.cost for record in self._history)

    @property
    def history(self) -> List[_UsageRecord]:
        return list(self._history)

    def as_dict(self) -> Dict[str, int]:
        return self.totals.as_dict()

    def as_state_update(self, *, existing_cost: float = 0.0) -> Dict[str, Any]:
        """Return a ``{"token_usage": ..., "total_cost": ...}`` payload.

        Mirrors the shape used by ``Recommender.agents.Agents._accumulate_state_metrics``
        and ``Trimmer.agenticNodes.agents.TokenCostCallback`` so state-dict-based
        LangGraph nodes can adopt this tracker as a drop-in replacement.
        """
        return {
            "token_usage": self.totals.as_dict(),
            "total_cost": existing_cost + self.total_cost,
        }

    def reset(self) -> None:
        self._history.clear()
