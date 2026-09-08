"""Random mock Juror used to test the MPC benchmark without an LLM API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel

from .llm_base import merge_last_output
from .mock_random import mock_rng


class JurorVerdict(BaseModel):
    """The same public output schema exposed by the real Juror module."""

    verdict: Literal["retry_with_wider_search", "reset_to_best", "accept_and_end"]
    explanation: str


def juror_node(state: Dict[str, Any], cfg=None) -> Dict[str, Any]:
    """Return a random verdict, but enforce the real graph's iteration cap."""
    iteration = int(state.get("iteration", 0))
    max_iterations = int(state.get("max_iterations", 0))
    budget_exhausted = iteration >= max_iterations > 0

    if budget_exhausted:
        verdict = "accept_and_end"
        explanation = "Mock Juror accepted the best result because the iteration budget is exhausted."
    else:
        rng = mock_rng(state, node_offset=401)
        options = ["accept_and_end", "retry_with_wider_search"]
        weights = [0.45, 0.40]
        if state.get("best_params"):
            options.append("reset_to_best")
            weights.append(0.15)
        verdict = rng.choices(options, weights=weights, k=1)[0]
        explanation = f"Mock Juror randomly selected {verdict} to exercise graph routing."

    history: List[str] = state.get("history", []) + [
        f"[Mock Juror] {verdict}: {explanation}"
    ]
    new_state: Dict[str, Any] = {
        **state,
        "juror_verdict": verdict,
        "should_continue": verdict != "accept_and_end",
        "history": history,
        "last_outputs": merge_last_output(
            state,
            "juror",
            f"MOCK OUTPUT\n{verdict}: {explanation}",
        ),
    }
    if verdict == "reset_to_best" and state.get("best_params"):
        new_state["current_params"] = state["best_params"]
    return new_state
