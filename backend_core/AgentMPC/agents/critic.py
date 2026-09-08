"""Random mock Critic used to test the MPC benchmark without an LLM API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .llm_base import merge_last_output
from .mock_random import mock_rng


class CriticFeedback(BaseModel):
    """The same public output schema exposed by the real Critic module."""

    feedback: str = Field(description="Mock qualitative tuning feedback.")
    strategy_recommendation: str
    suggested_multipliers: Optional[Dict[str, float]] = None


def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return random qualitative feedback while preserving graph fields."""
    if state.get("eval_error"):
        feedback = (
            f"Mock feedback: the previous proposal failed: {state['eval_error']}. "
            "Try a more conservative valid proposal."
        )
        strategy = "exploit"
    else:
        rng = mock_rng(state, node_offset=211)
        strategy = rng.choice(["explore", "explore", "exploit", "aggressive_explore"])
        feedback = rng.choice(
            [
                "Mock feedback: explore different horizons and rebalance Q relative to R.",
                "Mock feedback: reduce aggressive control by increasing the input penalty.",
                "Mock feedback: emphasize the state with the largest tracking error.",
                "Mock feedback: try a broader parameter change because progress may be slow.",
            ]
        )

        min_explore = int(state.get("min_explore_iterations", 4))
        if strategy == "exploit" and int(state.get("iteration", 0)) < min_explore:
            strategy = "explore"

    history: List[str] = state.get("history", []) + [f"[Mock Critic] {feedback}"]
    return {
        **state,
        "critic_feedback": feedback,
        "strategy": strategy,
        "history": history,
        "last_outputs": merge_last_output(state, "critic", f"MOCK OUTPUT\n{feedback}"),
    }
