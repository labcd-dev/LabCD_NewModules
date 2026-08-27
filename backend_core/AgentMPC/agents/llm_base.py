"""
================================================================================
agents/llm_base.py
================================================================================
Shared LLM client construction, used by actor.py / critic.py / scenarist.py /
terminator.py / juror.py so the model/provider is configured in exactly one
place instead of being duplicated (and drifting) across six agent files, as
it was in the original notebook.

This module intentionally does NOT hardcode a provider/model -- plug in
whatever you were using before (OpenRouter, Groq, the Anthropic API, etc.)
via ``configure_llm()``.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler

from ..utils.logging_utils import get_logger

log = get_logger(__name__)

_llm_factory: Optional[Any] = None


class TokenUsageTracker(BaseCallbackHandler):
    """Accumulates token usage across every LLM call it's attached to, via
    LangChain's on_llm_end hook -- this fires at the actual underlying API
    call, BEFORE any .with_structured_output() parsing happens on top of
    it, so it works uniformly for every agent in this codebase without
    needing each one to individually report its own usage.

    Extraction is defensive (tries several known field locations) because
    different LangChain versions / providers have shuffled where token
    counts live at least twice in the past (llm_output['token_usage'] vs.
    the newer per-message usage_metadata standard) -- this could not be
    verified against a live Groq API in this environment, so a call that
    doesn't match any known shape is silently skipped (counted as
    "unknown", surfaced via unparsed_calls) rather than raising and
    breaking the actual tuning run over a bookkeeping feature.
    """

    def __init__(self, default_model: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        self.unparsed_calls = 0
        self.per_model: Dict[str, Dict[str, int]] = {}
        # Fallback used when a response carries no model name of its own.
        # Without this, usage read via the newer per-message usage_metadata
        # path (where llm_output is empty, so there IS no name to find)
        # would all land under "unknown" and be excluded from costing --
        # which showed up as a correct token count next to a $0.00 total.
        self.default_model = default_model

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        with self._lock:
            self.call_count += 1
            usage = self._extract_usage(response)
            model_name = self._extract_model_name(response) or self.default_model
            if usage is None:
                self.unparsed_calls += 1
                return
            p, c, t = usage
            self.prompt_tokens += p
            self.completion_tokens += c
            self.total_tokens += t if t else (p + c)
            bucket = self.per_model.setdefault(model_name or "unknown", {"prompt": 0, "completion": 0, "total": 0})
            bucket["prompt"] += p
            bucket["completion"] += c
            bucket["total"] += t if t else (p + c)

    @staticmethod
    def _extract_usage(response: Any):
        # Location 1: the standard llm_output dict most providers populate.
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") or llm_output.get("usage")
        if token_usage:
            p = token_usage.get("prompt_tokens", 0) or 0
            c = token_usage.get("completion_tokens", 0) or 0
            t = token_usage.get("total_tokens", 0) or 0
            return p, c, t
        # Location 2: newer LangChain standard, per-message usage_metadata
        # (AIMessage.usage_metadata: input_tokens/output_tokens/total_tokens).
        try:
            for generation_list in getattr(response, "generations", []) or []:
                for gen in generation_list:
                    msg = getattr(gen, "message", None)
                    usage_meta = getattr(msg, "usage_metadata", None) if msg else None
                    if usage_meta:
                        p = usage_meta.get("input_tokens", 0) or 0
                        c = usage_meta.get("output_tokens", 0) or 0
                        t = usage_meta.get("total_tokens", 0) or 0
                        return p, c, t
        except Exception:  # noqa: BLE001
            pass
        return None

    @staticmethod
    def _extract_model_name(response: Any) -> Optional[str]:
        # Location 1: llm_output, populated by most providers.
        llm_output = getattr(response, "llm_output", None) or {}
        name = llm_output.get("model_name") or llm_output.get("model")
        if name:
            return name
        # Location 2: per-message response_metadata -- where several
        # integrations (ChatOpenAI among them) report the model the request
        # actually resolved to, and the only place it appears at all when
        # usage came through usage_metadata rather than llm_output.
        try:
            for gen_list in getattr(response, "generations", []) or []:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    meta = getattr(msg, "response_metadata", None) if msg else None
                    if meta:
                        name = meta.get("model_name") or meta.get("model")
                        if name:
                            return name
        except Exception:  # noqa: BLE001
            pass
        return None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            per_model = dict(self.per_model)
            cost, unpriced = self._compute_cost(per_model)
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "call_count": self.call_count,
                "unparsed_calls": self.unparsed_calls,
                "per_model": per_model,
                "cost_usd": cost,
                "unpriced_models": unpriced,
            }

    @staticmethod
    def _compute_cost(per_model: Dict[str, Dict[str, int]]):
        """USD cost, priced PER MODEL rather than on the token totals --
        a session can legitimately span several models, and their rates
        differ by more than an order of magnitude, so pricing the combined
        total would be meaningless.

        Returns (total_usd, unpriced_model_names). Models absent from the
        price table are reported by name instead of silently counting as
        free: CostCalculator.compute_cost returns 0.0 for anything it does
        not recognize, which would otherwise understate the real cost with
        no indication that it had.
        """
        try:
            from labcd_agents.pricing import CostCalculator
        except ImportError:
            return None, []
        calc = CostCalculator()
        total, unpriced = 0.0, []
        for model, u in per_model.items():
            if model == "unknown":
                continue
            if calc.resolve_price(model) is None:
                unpriced.append(model)
                continue
            total += calc.compute_cost(model, u.get("prompt", 0), u.get("completion", 0))
        return total, unpriced


def configure_llm(factory) -> None:
    """Register how to build a chat model. Call this once at startup:

        from langchain_openai import ChatOpenAI
        configure_llm(lambda: ChatOpenAI(model="...", api_key="...", base_url="..."))

    Keeping this as a factory (not a single shared instance) makes it trivial
    to give different agents different temperatures/models later if needed.
    """
    global _llm_factory
    _llm_factory = factory


def get_llm():
    if _llm_factory is None:
        raise RuntimeError(
            "No LLM configured. Call backend_core.AgentMPC.agents.llm_base.configure_llm(...) "
            "once at startup with a factory that returns a LangChain chat model "
            "(this mirrors the ChatOpenAI/ChatGroq setup from the original notebook)."
        )
    return _llm_factory()


def format_user_guidance(guidance: str) -> str:
    """Shared formatter for the free-text steering the user can type in the
    Streamlit UI (e.g. "only minimize control effort"). Used by both
    actor.py and critic.py so the same guidance text is presented
    consistently to every agent that reads it."""
    if not guidance or not guidance.strip():
        return (
            "User guidance: none provided -- use the default balanced objective "
            "(consider MSE, overshoot, settling time, and control effort together)."
        )
    return f"User guidance (follow this closely -- it overrides the default balanced objective): {guidance.strip()}"


def merge_last_output(state: dict, node_name: str, output_text: str) -> dict:
    """Returns an updated last_outputs dict with this node's OUTPUT (its
    reasoning/feedback/explanation text -- the same content that ends up in
    the Agent Reasoning tab, NOT the input prompt it was given) merged in
    under its own key, preserving every other node's last output. Used by
    every agent node (actor/critic/terminator/scenarist/juror) so app.py's
    live flow diagram can show "what did the Actor conclude" etc. on hover
    -- see render_agent_flow_diagram."""
    return {**(state.get("last_outputs") or {}), node_name: output_text}


def invoke_with_retry(structured_llm, prompt_text: str, max_retries: int = 1, node_name: str = "LLM",
                       tracker: Optional[TokenUsageTracker] = None):
    """Invoke a ``.with_structured_output(...)`` LLM, retrying (with the
    error fed back into the prompt) if the call fails.

    This exists because a single malformed response -- most commonly a
    provider-side schema/tool-call validation error, e.g. Groq rejecting a
    proposal that violates a Field's bounds -- otherwise propagates as an
    uncaught exception straight through ``graph.stream()`` and aborts the
    *entire* tuning run, discarding every iteration completed so far. One
    retry, with the exact validation error appended to the prompt asking the
    model to correct it, resolves the large majority of these without losing
    any progress.

    ``tracker``: optional TokenUsageTracker (see above) -- if given, this
    call's actual token usage is accumulated into it via LangChain's
    callback mechanism. Callers that don't need usage tracking can simply
    omit this; existing call sites are unaffected either way.

    Raises the last exception if every attempt fails -- callers are expected
    to catch that and apply their own safe, domain-specific fallback (e.g.
    actor_node keeps the previous parameters unchanged) rather than letting
    it kill the graph. See agents/actor.py for the reference pattern.
    """
    current_prompt = prompt_text
    last_error: Optional[Exception] = None
    invoke_kwargs = {"config": {"callbacks": [tracker]}} if tracker is not None else {}

    for attempt in range(max_retries + 1):
        try:
            return structured_llm.invoke(current_prompt, **invoke_kwargs)
        except Exception as e:  # noqa: BLE001
            last_error = e
            log.warning("[%s] call failed (attempt %d/%d): %s", node_name, attempt + 1, max_retries + 1, e)
            current_prompt = (
                f"{prompt_text}\n\n"
                f"IMPORTANT: your previous response failed with this exact error -- fix it and make sure "
                f"every value stays within the bounds stated above: {e}"
            )

    assert last_error is not None
    raise last_error
