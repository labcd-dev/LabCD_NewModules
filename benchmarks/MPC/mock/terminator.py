"""Random mock Terminator used to test the MPC benchmark without an LLM API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel

from backend_core.AgentMPC.agents.llm_base import merge_last_output
from .mock_random import mock_rng

Decision = Literal["critic", "juror"]


class TerminationDecision(BaseModel):
    """The same public output schema exposed by the real Terminator module."""

    decision: Decision
    reason: str


def should_continue(state: Dict[str, Any]) -> Dict[str, Any]:
    """Randomly route to Critic/Juror while always respecting hard guards."""
    iteration = int(state.get("iteration", 0))
    max_iterations = int(state.get("max_iterations", 20))
    mse_history = state.get("mse_history", [])
    target_mse = state.get("target_mse")
    has_valid_result = bool(state.get("best_params"))

    if iteration >= max_iterations:
        decision: Decision = "juror"
        reason = "max_iterations reached (mock numeric guard)"
    elif not has_valid_result:
        decision = "critic"
        reason = "no valid evaluated parameters yet (mock safety guard)"
    elif target_mse is not None and mse_history and mse_history[-1] <= float(target_mse):
        decision = "juror"
        reason = "target MSE reached (mock numeric guard)"
    else:
        rng = mock_rng(state, node_offset=307)
        decision = rng.choices(["critic", "juror"], weights=[0.75, 0.25], k=1)[0]
        reason = f"random mock routing selected {decision}"

    history: List[str] = state.get("history", []) + [
        f"[Mock Terminator] Decision: route to {decision.capitalize()} - {reason}"
    ]
    return {
        **state,
        "should_continue": True,
        "termination_reason": reason,
        "_next": decision,
        "history": history,
        "last_outputs": merge_last_output(
            state,
            "terminator",
            f"MOCK OUTPUT\nDecision: route to {decision.capitalize()} - {reason}",
        ),
    }
