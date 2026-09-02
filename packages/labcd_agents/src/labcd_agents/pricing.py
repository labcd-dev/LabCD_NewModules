"""Model pricing and cost calculation.

``MuloDesigner``, ``SiloDesigner``, ``Recommender`` and ``Trimmer`` each ship
their own ``PRICE_TABLE`` / ``PRICING_CONFIG`` dict plus near-identical
"strip vendor prefix, try exact match, fall back to a regex-matched base
model name" resolution logic in ``_compute_cost`` /
``_accumulate_state_metrics``. This module consolidates that into a single
:class:`CostCalculator` with a merged default price table (USD per 1M
tokens) and pluggable per-instance overrides/registrations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

__all__ = ["ModelPrice", "DEFAULT_PRICE_TABLE", "CostCalculator"]


@dataclass(frozen=True)
class ModelPrice:
    """USD price per 1,000,000 tokens."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: Optional[float] = None

    def cost(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
        cached = max(0, min(int(cached_input_tokens or 0), input_tokens))
        fresh = input_tokens - cached
        cached_rate = self.cached_input_per_million if self.cached_input_per_million is not None \
            else self.input_per_million
        return (self.input_per_million * fresh + cached_rate * cached
                + self.output_per_million * output_tokens) / 1_000_000


# Merged from the price tables in MuloDesigner/GaAgent, SiloDesigner,
# Recommender and Trimmer. Where modules disagreed on a price for the same
# model, the most recently observed/most specific value was kept; callers
# can always override via CostCalculator(overrides=...) or .register().
DEFAULT_PRICE_TABLE: Dict[str, ModelPrice] = {
    # OpenAI
    "gpt-5-nano": ModelPrice(0.05, 0.40, cached_input_per_million=0.005),
    "gpt-4o": ModelPrice(3.0, 10.0, cached_input_per_million=1.25),
    "gpt-4o-mini": ModelPrice(0.15, 0.60, cached_input_per_million=0.075),
    "gpt-5.4-mini": ModelPrice(0.75, 4.50, cached_input_per_million=0.075),
    "gpt-5.4": ModelPrice(2.50, 15.00, cached_input_per_million=0.25),
    "gpt-5.5": ModelPrice(5.00, 30.00, cached_input_per_million=0.50),
    # Groq / Cerebras open models
    "llama3.1-8b": ModelPrice(0.05, 0.08),
    "llama-3.3-70b": ModelPrice(0.50, 0.64),
    "llama-3.1-8b-instant": ModelPrice(0.55, 0.65),
    "llama-3.3-70b-versatile": ModelPrice(0.59, 0.79),
    "llama-3.1-70b": ModelPrice(0.59, 0.79),
    "qwen-3-32b": ModelPrice(0.40, 0.80),
    "qwen-3-max": ModelPrice(1.20, 6.00),
    "gpt-oss-120b": ModelPrice(0.25, 0.69),
    "gpt-oss-20b": ModelPrice(0.20, 0.30),
    "deepseek-r1-distill-llama-70b": ModelPrice(0.23, 0.69),
    "llama-4-maverick": ModelPrice(0.50, 0.64),
    "llama-3": ModelPrice(0.50, 0.64),
    "llama3": ModelPrice(0.50, 0.64),
    "llama-4": ModelPrice(0.60, 0.80),
    "llama4": ModelPrice(0.60, 0.80),
    "mistral-nemotron": ModelPrice(0.04, 0.17),
    "kimi-k2-instruct": ModelPrice(1.00, 3.00),
    # NVIDIA NIM
    "llama-3.3-nemotron-super-49b-v1": ModelPrice(0.05, 0.05),
}

# Regex patterns tried (in order) when an exact/prefix-stripped lookup
# misses, so e.g. "llama-4-maverick-17b-128e-instruct" still resolves to the
# "llama-4-maverick" (or failing that "llama-4") price entry. Mirrors the
# two-pattern fallback duplicated in SiloDesigner/llm_agents.py and
# MuloDesigner/GaAgent/base_agent.py.
_FALLBACK_PATTERNS = (
    re.compile(r"(llama-\d+-maverick)"),
    re.compile(r"(llama-?\d+(?:\.\d+)?)"),
)


class CostCalculator:
    """Resolves a model name to a price and computes call cost.

    Args:
        overrides: extra/overriding ``{model_name: ModelPrice}`` entries
            merged on top of :data:`DEFAULT_PRICE_TABLE`.
    """

    def __init__(self, overrides: Optional[Dict[str, ModelPrice]] = None) -> None:
        self._table: Dict[str, ModelPrice] = dict(DEFAULT_PRICE_TABLE)
        if overrides:
            self._table.update(overrides)

    def register(self, model: str, input_per_million: float, output_per_million: float) -> None:
        """Register or override the price for a specific model name."""
        self._table[model] = ModelPrice(input_per_million, output_per_million)

    @staticmethod
    def _strip_vendor_prefix(model_name: str) -> str:
        # "meta/llama-4-maverick..." -> "llama-4-maverick...", "openai/gpt-oss-120b" -> "gpt-oss-120b"
        return model_name.split("/", 1)[1] if "/" in model_name else model_name

    def resolve_price(self, model: str) -> Optional[ModelPrice]:
        """Look up the :class:`ModelPrice` for ``model``, or ``None`` if unknown."""
        model_name = self._strip_vendor_prefix((model or "").lower())

        if model_name in self._table:
            return self._table[model_name]

        for pattern in _FALLBACK_PATTERNS:
            match = pattern.match(model_name)
            if match:
                base_model = match.group(1)
                if base_model in self._table:
                    return self._table[base_model]

        return None

    def compute_cost(self, model: str, input_tokens: int, output_tokens: int,
                      cached_input_tokens: int = 0) -> float:
        """Compute the USD cost of a call. Returns ``0.0`` for unpriced models."""
        price = self.resolve_price(model)
        return price.cost(input_tokens, output_tokens, cached_input_tokens) if price else 0.0
