"""LangChain callback integration for automatic token/cost tracking.

Replaces ``Trimmer.agenticNodes.agents.TokenCostCallback`` (a
``BaseCallbackHandler`` subclass hand-written against a LangGraph state
dict) with a reusable handler that reports into a :class:`~labcd_agents.tokens.TokenTracker`
+ :class:`~labcd_agents.pricing.CostCalculator` pair. Because it's a normal
LangChain callback, it works transparently with chains built via LCEL
(``prompt | llm | parser``) as well as direct ``llm.invoke(...)`` calls -
covering the ``create_agent``/LangGraph-node style used in ``Trimmer``.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

from .pricing import CostCalculator
from .tokens import TokenTracker, extract_usage

__all__ = ["TokenCostCallbackHandler"]


class TokenCostCallbackHandler(BaseCallbackHandler):
    """A LangChain callback that records token usage and cost on every LLM call.

    Args:
        tracker: the :class:`TokenTracker` to record into. A new one is
            created if omitted.
        cost_calculator: the :class:`CostCalculator` used to price each
            call. A default-priced one is created if omitted.

    Example:
        >>> tracker = TokenTracker()
        >>> handler = TokenCostCallbackHandler(tracker)
        >>> chain.invoke(inputs, config={"callbacks": [handler]})
        >>> tracker.totals
        TokenUsage(input_tokens=..., output_tokens=...)
    """

    def __init__(self, tracker: Optional[TokenTracker] = None, cost_calculator: Optional[CostCalculator] = None):
        super().__init__()
        self.tracker = tracker if tracker is not None else TokenTracker()
        self.cost_calculator = cost_calculator if cost_calculator is not None else CostCalculator()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: D401 - LangChain signature
        try:
            generation = response.generations[0][0]
            message = getattr(generation, "message", generation)
        except (AttributeError, IndexError):
            message = response

        model_id = (kwargs.get("invocation_params") or {}).get("model_name", "")
        if not model_id:
            response_metadata = getattr(message, "response_metadata", None) or {}
            model_id = response_metadata.get("model_name", "unknown")

        usage = extract_usage(message)
        cost = self.cost_calculator.compute_cost(model_id, usage.input_tokens, usage.output_tokens)
        self.tracker.record(usage, model=model_id, cost=cost)
