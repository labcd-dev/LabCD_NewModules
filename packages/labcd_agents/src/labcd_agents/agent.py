"""The shared agent base class.

:class:`BaseAgent` consolidates what ``LLMBaseAgent`` (MuloDesigner/GaAgent,
SiloDesigner) and the various module-local ``Agents`` classes
(MuloDesigner, Recommender, Regularizer, Trimmer) each reimplement:

- Provider-aware client construction (via :class:`~labcd_agents.providers.LLMFactory`).
- ``invoke_llm(system_prompt, user_prompt, ...)`` with retry logic, per-call
  timing, structured logging, and an optional ``monitor`` hook
  (``monitor.add_llm_response(agent_name, prompt, response)``), matching
  the ``LLMBaseAgent.invoke_llm`` contract used by MuloDesigner/GaAgent and
  SiloDesigner almost verbatim.
- Automatic token usage + cost tracking via :class:`~labcd_agents.tokens.TokenTracker`
  and :class:`~labcd_agents.pricing.CostCalculator`.
- A lower-level ``call`` method for LangChain-message-based invocation
  (mirroring the ``_call_llm(llm, prompt_text, system=, context_messages=,
  is_json=)`` helper duplicated in MuloDesigner/Regularizer/Recommender).

Existing modules are **not** migrated to this class - it's a foundation for
new modules. See the package README for a usage example.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Sequence

from .exceptions import LLMInvocationError
from .logging_utils import get_logger
from .messages import MessageLike, build_messages, extract_response_text
from .pricing import CostCalculator
from .providers import LLMFactory
from .tokens import TokenTracker, TokenUsage, extract_usage

__all__ = ["BaseAgent"]


class BaseAgent:
    """Common foundation for LLM-backed agents.

    Args:
        model: model name/id, e.g. ``"gpt-4o-mini"``.
        temperature: sampling temperature (default ``0.0`` for determinism,
            matching every existing module's default).
        seed: optional deterministic seed, forwarded to providers that
            support it.
        provider: force a specific provider instead of auto-detecting from
            ``model`` (see :meth:`labcd_agents.providers.LLMFactory.create`).
        monitor: optional object exposing
            ``add_llm_response(agent_name, prompt, response)``, called after
            every successful invocation. Matches the ``monitor`` parameter
            threaded through ``LLMBaseAgent`` in MuloDesigner/GaAgent and
            SiloDesigner.
        max_retries: default retry budget for :meth:`invoke_llm`.
        price_table: optional ``{model: ModelPrice}`` overrides forwarded to
            the agent's :class:`~labcd_agents.pricing.CostCalculator`.
        client: pre-built LangChain chat model to use instead of creating
            one via :class:`~labcd_agents.providers.LLMFactory` (useful for
            tests, or when a module needs bespoke client construction).
        client_kwargs: extra keyword arguments forwarded to
            :meth:`LLMFactory.create` (e.g. ``api_key=``, ``max_tokens=``).
        logger: a pre-configured logger; a default one is created if omitted.
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        provider: Optional[str] = None,
        monitor: Optional[Any] = None,
        max_retries: int = 3,
        price_table: Optional[dict] = None,
        client: Optional[Any] = None,
        client_kwargs: Optional[dict] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.monitor = monitor
        self.max_retries = max_retries
        self.agent_name = self.__class__.__name__
        self.logger = logger or get_logger(f"labcd_agents.{self.agent_name}")

        self.client = client if client is not None else LLMFactory.create(
            model, temperature=temperature, seed=seed, provider=provider, **(client_kwargs or {})
        )

        self.token_tracker = TokenTracker()
        self.cost_calculator = CostCalculator(price_table)

    # ------------------------------------------------------------------
    # High-level: system/user prompt invocation with retries + tracking
    # ------------------------------------------------------------------
    def invoke_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_retries: Optional[int] = None,
    ) -> tuple:
        """Invoke the LLM with separate system/user prompts, with retries.

        Mirrors ``LLMBaseAgent.invoke_llm`` from MuloDesigner/GaAgent and
        SiloDesigner: builds a ``[{"role": "system", ...}, {"role": "user",
        ...}]`` message list, retries on any exception up to ``max_retries``
        times, times the call, records token usage + cost, and reports to
        ``self.monitor`` if configured.

        Returns:
            A ``(response_text, usage)`` tuple, where ``usage`` is a dict of
            ``{"prompt_tokens", "completion_tokens", "llm_time"}`` - the same
            shape the legacy modules use so it's a drop-in return value.

        Raises:
            LLMInvocationError: if every retry attempt fails. The original
                exception is available as ``.last_error``.
        """
        retries = self.max_retries if max_retries is None else max_retries
        messages = build_messages(user_prompt, system_prompt=system_prompt, as_dicts=True)

        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                if attempt > 0:
                    self.logger.info("Retry attempt %d/%d for %s", attempt + 1, retries, self.agent_name)

                start = time.time()
                response = self.client.invoke(messages)
                duration = time.time() - start

                response_text = extract_response_text(response)
                usage = extract_usage(response)
                cost = self.cost_calculator.compute_cost(self.model, usage.input_tokens, usage.output_tokens)
                self.token_tracker.record(usage, model=self.model, cost=cost)

                if self.monitor is not None:
                    full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
                    truncated = full_prompt[:200] + "..." if len(full_prompt) > 200 else full_prompt
                    self.monitor.add_llm_response(self.agent_name, truncated, response_text)

                usage_dict = usage.as_prompt_completion_dict()
                usage_dict["llm_time"] = duration
                return response_text, usage_dict

            except Exception as exc:  # noqa: BLE001 - intentionally broad, matches legacy retry-on-anything behavior
                last_error = exc
                self.logger.error(
                    "Error invoking LLM (attempt %d/%d): %s: %s",
                    attempt + 1, retries, type(exc).__name__, exc,
                )

        raise LLMInvocationError(
            f"{self.agent_name} failed to invoke '{self.model}' after {retries} attempts",
            attempts=retries,
            last_error=last_error,
        )

    # ------------------------------------------------------------------
    # Lower-level: LangChain-message based invocation (chat history, tools, etc.)
    # ------------------------------------------------------------------
    def call(
        self,
        prompt_text: str,
        *,
        system: bool = False,
        context_messages: Optional[Sequence[MessageLike]] = None,
        is_json: bool = False,
        client: Optional[Any] = None,
    ) -> str:
        """Invoke the LLM with a single prompt plus optional chat history.

        Mirrors the ``_call_llm(llm, prompt_text, system=, context_messages=,
        is_json=)`` helper duplicated in MuloDesigner, Regularizer and
        Recommender. Unlike :meth:`invoke_llm`, this returns text only (no
        usage tuple) and lets the caller pass a different ``client`` (e.g. a
        secondary model configured on the same agent instance), matching how
        those modules call ``self._call_llm(self.llm_nvidia, ...)`` and
        ``self._call_llm(self.llm_think, ...)`` interchangeably.

        Args:
            prompt_text: the prompt content.
            system: if True, send as a system message; otherwise a human message.
            context_messages: prior conversation turns appended after the new message.
            is_json: if True, request JSON-mode output where the client supports it
                (``llm.bind(response_format={"type": "json_object"})``).
            client: override which chat model to invoke (defaults to ``self.client``).
        """
        target = client if client is not None else self.client
        messages = self._build_single_message(prompt_text, system=system, context_messages=context_messages)

        if is_json:
            target = target.bind(response_format={"type": "json_object"})

        response = target.invoke(messages)

        usage = extract_usage(response)
        if usage.total_tokens:
            model_name = getattr(target, "model", None) or getattr(target, "model_name", None) or self.model
            cost = self.cost_calculator.compute_cost(model_name, usage.input_tokens, usage.output_tokens)
            self.token_tracker.record(usage, model=model_name, cost=cost)

        return extract_response_text(response)

    @staticmethod
    def _build_single_message(
        prompt_text: str, *, system: bool, context_messages: Optional[Sequence[MessageLike]]
    ) -> List[MessageLike]:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages: List[MessageLike] = [SystemMessage(content=prompt_text) if system else HumanMessage(content=prompt_text)]
        if context_messages:
            messages = messages + list(context_messages)
        return messages

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    @property
    def total_cost(self) -> float:
        """Cumulative USD cost of every call made through this agent so far."""
        return self.token_tracker.total_cost

    @property
    def total_usage(self) -> TokenUsage:
        """Cumulative token usage across every call made through this agent."""
        return self.token_tracker.totals

    def get_metrics(self) -> dict:
        """Return ``{"token_usage": {...}, "total_cost": ...}`` for state merging.

        Convenience for LangGraph-style nodes that need to merge metrics
        into a state dict, e.g. ``return {"messages": [...], **agent.get_metrics()}``.
        """
        return self.token_tracker.as_state_update()
