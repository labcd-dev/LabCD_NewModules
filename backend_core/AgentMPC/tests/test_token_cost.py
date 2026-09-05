"""
Run with: pytest backend_core/AgentMPC/tests/test_token_cost.py -v

Regression guard for the "dollar cost is always $0.00 in AgentMPC" bug.

Two separate causes, both fixed here:

1. ``agent_mpc_app.py`` constructed its TokenUsageTracker with no
   ``default_model``. When a call's response carried no model name of its
   own (on_llm_end's on-response extraction), the call was bucketed under
   "unknown", and cost calculation silently skipped that bucket -- token
   counts looked right, cost stayed zero, with nothing in the UI explaining
   why.

2. Even with a model name, ``on_llm_end`` preferred whatever the API
   response echoed back over the model actually requested. OpenAI in
   particular echoes a dated, versioned name ("gpt-4o-mini-2024-07-18")
   that CostCalculator's price table (keyed on the bare "gpt-4o-mini") does
   not recognize, so the call still priced as $0.00 -- unlike AgentPlant's
   labcd_agents.BaseAgent, which prices every call against ``self.model``
   (the string it was actually constructed with), never against a name
   parsed back out of the response. TokenUsageTracker now makes the same
   choice.
"""

from types import SimpleNamespace

import pytest

from backend_core.AgentMPC.agents.llm_base import TokenUsageTracker


def _response(*, model_name=None, prompt=100, completion=50):
    """A LangChain LLMResult-shaped stub. ``model_name=None`` mimics a
    response with no model name anywhere -- the case that used to fall back
    to the "unknown" bucket."""
    llm_output = {"token_usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                                  "total_tokens": prompt + completion}}
    if model_name is not None:
        llm_output["model_name"] = model_name
    return SimpleNamespace(llm_output=llm_output, generations=[])


# --------------------------------------------------------------------------
# Cause 1: no default_model configured at all
# --------------------------------------------------------------------------

def test_without_default_model_and_no_response_name_buckets_as_unknown():
    tracker = TokenUsageTracker()  # the exact old agent_mpc_app.py call site
    tracker.on_llm_end(_response(model_name=None))
    snap = tracker.snapshot()
    assert snap["total_tokens"] == 150  # counts are still right...
    assert snap["cost_usd"] == 0.0      # ...but cost is unrecoverable without a name


def test_unknown_model_is_reported_as_unpriced_not_silently_dropped():
    """Before this fix, a call with no resolvable model name vanished from
    accounting entirely -- $0.00 with an EMPTY unpriced_models list, which
    looked identical to "everything priced at zero cost" in the UI. It must
    now show up so the UI can explain why the total is zero."""
    tracker = TokenUsageTracker()
    tracker.on_llm_end(_response(model_name=None))
    snap = tracker.snapshot()
    assert "unknown" in snap["unpriced_models"]


# --------------------------------------------------------------------------
# Cause 2: default_model configured, but the API echoes a different name
# --------------------------------------------------------------------------

def test_default_model_prices_a_call_whose_response_has_no_name():
    """This is the actual fix for agent_mpc_app.py: passing default_model=
    is what makes a real run's cost non-zero."""
    tracker = TokenUsageTracker(default_model="gpt-4o-mini")
    tracker.on_llm_end(_response(model_name=None, prompt=1000, completion=500))
    snap = tracker.snapshot()
    assert snap["cost_usd"] == pytest.approx((1000 * 0.15 + 500 * 0.60) / 1_000_000)
    assert snap["unpriced_models"] == []


def test_default_model_wins_over_a_versioned_name_the_api_echoes_back():
    """OpenAI-style dated model names ("gpt-4o-mini-2024-07-18") don't match
    the price table's bare key. The requested model name (known in advance,
    same as labcd_agents.BaseAgent's self.model) must be preferred over
    whatever the response reports, or every real call still mispriced as
    unrecognized."""
    tracker = TokenUsageTracker(default_model="gpt-4o-mini")
    tracker.on_llm_end(_response(model_name="gpt-4o-mini-2024-07-18", prompt=1000, completion=500))
    snap = tracker.snapshot()
    assert snap["cost_usd"] == pytest.approx((1000 * 0.15 + 500 * 0.60) / 1_000_000)
    assert list(snap["per_model"].keys()) == ["gpt-4o-mini"]
    assert "gpt-4o-mini-2024-07-18" not in snap["per_model"]


def test_response_derived_name_is_still_the_fallback_when_no_default_set():
    """Multi-model sessions with no single default_model still get priced
    per call using whatever name the response itself reports."""
    tracker = TokenUsageTracker()
    tracker.on_llm_end(_response(model_name="gpt-4o-mini", prompt=1000, completion=500))
    snap = tracker.snapshot()
    assert snap["cost_usd"] == pytest.approx((1000 * 0.15 + 500 * 0.60) / 1_000_000)


def test_cost_accumulates_correctly_across_several_calls():
    tracker = TokenUsageTracker(default_model="gpt-4o-mini")
    for _ in range(3):
        tracker.on_llm_end(_response(prompt=1000, completion=500))
    snap = tracker.snapshot()
    assert snap["call_count"] == 3
    assert snap["cost_usd"] == pytest.approx(3 * (1000 * 0.15 + 500 * 0.60) / 1_000_000)


def test_a_genuinely_unpriced_model_is_reported_and_excluded() -> None:
    tracker = TokenUsageTracker(default_model="some-brand-new-model-nobody-has-priced-yet")
    tracker.on_llm_end(_response(prompt=1000, completion=500))
    snap = tracker.snapshot()
    assert snap["cost_usd"] == 0.0
    assert snap["unpriced_models"] == ["some-brand-new-model-nobody-has-priced-yet"]
    # the token counts are still real and still reported -- only the cost is unknown
    assert snap["total_tokens"] == 1500
