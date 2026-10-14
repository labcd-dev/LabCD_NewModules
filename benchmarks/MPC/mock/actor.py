"""Random mock Actor used to test the MPC benchmark without an LLM API."""

from __future__ import annotations

from typing import Any, Dict, List

from backend_core.AgentMPC.utils.logging_utils import get_logger
from backend_core.AgentMPC.agents.llm_base import merge_last_output
from .mock_random import log_uniform, mock_rng
from backend_core.AgentMPC.agents.schemas import MPCParameters

log = get_logger(__name__)


def actor_node(state: Dict[str, Any], cfg=None) -> Dict[str, Any]:
    """Return a dimensionally valid, randomized MPC parameter proposal."""
    n_states = int(state["n_states"])
    n_inputs = int(state["n_inputs"])
    iteration = int(state.get("iteration", 0))
    rng = mock_rng(state, node_offset=101)

    if iteration == 0:
        # Begin every mock run with one conservative, deterministic candidate.
        # Later calls remain random, so the mock still exercises exploration
        # while avoiding an accidental early exit before any valid result exists.
        prediction_horizon = 10
        control_horizon = 3
        q = [10.0] * n_states
        r = [1.0] * n_inputs
        p = [20.0] * n_states
        strategy = "explore"
    else:
        prediction_horizon = rng.randint(8, 30)
        control_horizon = rng.randint(2, min(12, prediction_horizon))
        q = [log_uniform(rng, -1.0, 2.0) for _ in range(n_states)]
        r = [log_uniform(rng, -1.0, 1.0) for _ in range(n_inputs)]
        p = [weight * rng.uniform(1.0, 3.0) for weight in q]
        strategy = rng.choice(["explore", "explore", "aggressive_explore", "exploit"])

    # Most real Actor responses leave dt unchanged. Occasionally changing it
    # exercises that JSON/evaluator path without choosing an unsafe value.
    dt = None
    if iteration > 0 and cfg is not None and rng.random() < 0.2:
        current_dt = float(cfg.data.dt_mpc)
        dt = max(0.001, min(2.0, current_dt * rng.choice([0.5, 1.0, 2.0])))

    params = {
        "Np": prediction_horizon,
        "Nc": control_horizon,
        "Q": q,
        "R": r,
        "P": p,
        "dt": dt,
    }
    reasoning = (
        "Mock Actor generated a random, valid parameter proposal to exercise "
        "the evaluator and benchmark serialization path."
    )
    history: List[str] = state.get("history", []) + [
        f"[Mock Actor] {strategy}: Np={prediction_horizon}, Nc={control_horizon}"
    ]
    log.info(
        "[Mock Actor] iteration=%d strategy=%s Np=%d Nc=%d",
        iteration,
        strategy,
        prediction_horizon,
        control_horizon,
    )

    return {
        **state,
        "current_params": params,
        "strategy": strategy,
        "actor_reasoning": reasoning,
        "history": history,
        "iteration": iteration + 1,
        "last_outputs": merge_last_output(
            state,
            "actor",
            f"MOCK OUTPUT\nStrategy: {strategy}\n{reasoning}",
        ),
    }
