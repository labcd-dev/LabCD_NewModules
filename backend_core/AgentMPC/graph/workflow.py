"""
================================================================================
graph/workflow.py
================================================================================
Wires the agent nodes into the LangGraph state machine:

    START -> scenarist -> actor -> evaluator -> terminator -> {critic|juror}
                              ^                        |
                              |________________________| (critic loops back to actor)
                              |
                              |___________ juror -> {actor | END} ______|
                                            (juror is the ONLY path to END --
                                             see agents/terminator.py and
                                             agents/juror.py)

Two structural changes vs. the original notebook:

  1. ``dynamics``/``cfg`` are bound into the Evaluator (and Scenarist) node
     via ``functools.partial`` at graph-build time, instead of being pulled
     from ``globals()``/``__main__`` at call time. This is what makes
     ``evaluator_node`` unit-testable in isolation (see tests/) and lets the
     same graph be reused for a different dynamics plugin just by building it
     again with different bindings -- no global mutation involved.
  2. The Terminator's routing decision is read from ``state["_next"]``
     (set in agents/terminator.py) via a conditional edge, instead of a
     separate always-LLM "should I continue" node.
"""

from __future__ import annotations

import functools
from typing import Any, Dict

try:
    from langgraph.graph import END, StateGraph
except ImportError as e:  # pragma: no cover
    raise ImportError("langgraph is required for backend_core.AgentMPC.graph (pip install langgraph)") from e

from ..dynamics.base import BaseDynamics
from ..mpc.config import Config
from ..utils.logging_utils import get_logger
from .state import MPCGraphState

log = get_logger(__name__)


def _route_after_terminator(state: Dict[str, Any]) -> str:
    # The Terminator itself never ends a run anymore (see agents/terminator.py) --
    # should_continue is always True coming out of it, so this just reads its
    # "critic" vs "juror" routing choice. The should_continue check is kept as a
    # defensive fallback in case a future change reintroduces a direct-end path.
    if not state.get("should_continue", True):
        return "end"
    return state.get("_next", "critic")


def _route_after_juror(state: Dict[str, Any]) -> str:
    # Juror is the ONLY node that can actually end a run (accept_and_end sets
    # should_continue=False) -- every other verdict (retry_with_wider_search,
    # reset_to_best, tune_dt) loops back to the Actor for more iterations.
    if not state.get("should_continue", True):
        return "end"
    return "actor"


def _register_common_nodes(workflow: "StateGraph", dynamics: BaseDynamics, cfg: Config) -> None:
    from ..agents.actor import actor_node
    from ..agents.critic import critic_node
    from ..agents.evaluator import evaluator_node
    from ..agents.juror import juror_node
    from ..agents.terminator import should_continue

    workflow.add_node("actor", functools.partial(actor_node, cfg=cfg))
    workflow.add_node("evaluator", functools.partial(evaluator_node, dynamics=dynamics, cfg=cfg))
    workflow.add_node("terminator", should_continue)
    workflow.add_node("critic", critic_node)
    workflow.add_node("juror", functools.partial(juror_node, cfg=cfg))

    workflow.add_edge("actor", "evaluator")
    workflow.add_edge("evaluator", "terminator")
    workflow.add_conditional_edges(
        "terminator",
        _route_after_terminator,
        {"critic": "critic", "juror": "juror", "end": END},
    )
    workflow.add_edge("critic", "actor")
    workflow.add_conditional_edges(
        "juror",
        _route_after_juror,
        {"actor": "actor", "end": END},
    )


def build_mpc_tuning_graph(dynamics: BaseDynamics, cfg: Config):
    """Build and compile the full Scenarist-Actor-Critic-Juror tuning graph
    (the Scenarist LLM call designs the test scenario itself). Used by
    run_agents.py."""
    from ..agents.scenarist import scenarist_node

    workflow = StateGraph(MPCGraphState)

    workflow.add_node(
        "scenarist",
        functools.partial(
            scenarist_node,
            default_initial_state=dynamics.config.default_initial_state,
            default_target=dynamics.config.default_target,
        ),
    )
    _register_common_nodes(workflow, dynamics, cfg)

    workflow.set_entry_point("scenarist")
    workflow.add_edge("scenarist", "actor")

    return workflow.compile()


def build_ui_tuning_graph(dynamics: BaseDynamics, cfg: Config, entry_node: str = "actor"):
    """Build and compile the tuning graph WITHOUT the Scenarist node --
    entry point is directly "actor" or "evaluator". Used by the Streamlit UI
    (app.py), where the scenario (Level 1/2/3) is a deterministic choice the
    user makes from a dropdown (see agents/scenario_presets.py:apply_scenario_level),
    not something an LLM should be designing on every run.

    ``entry_node="actor"`` (default): the Actor LLM proposes the very first
    parameter set with no prior context.
    ``entry_node="evaluator"``: skips the Actor for iteration 0 and evaluates
    ``state["current_params"]`` as given -- used when the user supplies their
    own initial Np/Nc/Q/R from the UI (see agents/seed_params.py:parse_seed_params
    and app.py's "Initial Parameters" panel). The Actor/Critic loop still
    takes over normally from iteration 1 onward.
    """
    if entry_node not in ("actor", "evaluator"):
        raise ValueError(f"entry_node must be 'actor' or 'evaluator', got {entry_node!r}")

    workflow = StateGraph(MPCGraphState)
    _register_common_nodes(workflow, dynamics, cfg)
    workflow.set_entry_point(entry_node)
    return workflow.compile()


def initial_state(
    dynamics: BaseDynamics,
    system_name: str,
    max_iterations: int = 20,
    ui_scenario_level: int = 1,
    seed_params: Any = None,
    user_guidance: str = "",
    min_explore_iterations: int = 4,
    cost_weights: Any = None,
    exploration_intensity: int = 50,
    dt_mpc: float = 0.02,
    token_tracker: Any = None,
) -> MPCGraphState:
    """Convenience builder for the graph's starting state.

    ``seed_params``: optional dict ({"Np":..., "Nc":..., "Q":[...], "R":[...],
    "P":[...]}) used as ``current_params`` when the graph's entry point is
    "evaluator" (see build_ui_tuning_graph) -- i.e. the user's own initial
    guess, evaluated as-is before the Actor/Critic loop starts refining it.

    ``user_guidance``: free-text steering passed into the Actor/Critic
    prompts (e.g. "only minimize control effort, overshoot doesn't matter
    for this system"). Empty string (default) means "no extra guidance" --
    the agents fall back to their normal balanced multi-objective behavior.

    ``min_explore_iterations``: the Critic cannot recommend 'exploit' before
    this many iterations have completed, regardless of what the LLM itself
    proposes -- a deterministic guard against converging to local
    fine-tuning too early, before the search has covered enough of the
    parameter space (see agents/critic.py).

    ``cost_weights``: optional weight dict (see
    agents/metrics.py:OPTIMIZATION_FOCUS_PRESETS) used by evaluator_node to
    rank "best so far" -- None means the balanced default. This is what
    makes "Optimization Focus: Minimize MSE" in the UI actually change which
    result is reported as best, not just a label.

    ``exploration_intensity``: 1-100, how bold the Actor is while in
    'explore' mode -- translated into an explicit multiplicative-change
    range in the Actor's prompt (see agents/actor.py). 50 (default) matches
    the original explore behavior; 100 pushes toward 'aggressive_explore'-
    level boldness even for a plain 'explore' step, 1 keeps changes very
    conservative.
    """
    return MPCGraphState(
        system_name=system_name,
        n_states=dynamics.n_states,
        n_inputs=dynamics.n_inputs,
        state_names=dynamics.state_names,
        input_names=dynamics.input_names,
        current_params=seed_params or {},
        strategy="manual" if seed_params else "explore",
        iteration=0,
        max_iterations=max_iterations,
        best_mse=float("inf"),
        mse_history=[],
        overshoot_history=[],
        settling_history=[],
        effort_history=[],
        params_history=[],
        history=[],
        ui_scenario_level=ui_scenario_level,
        user_guidance=user_guidance,
        min_explore_iterations=min_explore_iterations,
        cost_weights=cost_weights,
        exploration_intensity=exploration_intensity,
        last_outputs={},
        dt_mpc=dt_mpc,
        dt_tuned_by_juror=False,
        token_tracker=token_tracker,
    )
