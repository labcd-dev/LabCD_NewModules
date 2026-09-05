"""
================================================================================
agents/actor.py
================================================================================
Actor node: LLM proposes the next MPC parameter set given the Critic's
feedback and the run history.

PORTING NOTE
--------------------------------------------------------------------------------
This file reproduces the *structure* (I/O schema, dimension-safety checks,
history bookkeeping) of the Actor node from the original notebook (cell 19),
not its exact prompt wording -- that prompt is your domain-tuned asset and
lives in ``../prompts/actor.yaml``, where it can be edited without touching
this file. Everything else (structured-output parsing, dimension repair,
logging) is complete and ready to use.

Two behavioural changes worth knowing about vs. the original:
  * The Actor is called through ``get_llm()`` (agents/llm_base.py) instead of
    a module-level ``llm`` global, so it no longer needs to guess which
    module scope a `ChatOpenAI`/`ChatGroq` instance lives in.
  * ``P`` (terminal weights) now has a real effect downstream (see
    mpc/controller.py) -- the Actor's reasoning about it is no longer wasted.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.prompts import PromptTemplate

from ..utils.logging_utils import get_logger
from .formatting import round_floats
from .llm_base import format_user_guidance, get_llm, invoke_with_retry, merge_last_output
from .prompt_library import get_prompt
from .schemas import MPCParameters

log = get_logger(__name__)

# Prompt text lives in ../prompts/actor.yaml. Keep the input_variables list
# below in sync with whatever placeholders that file actually uses.
ACTOR_PROMPT_TEMPLATE = get_prompt("actor")

actor_prompt = PromptTemplate(
    input_variables=[
        "system_name", "n_states", "n_inputs", "state_names", "input_names",
        "user_guidance_block",
        "current_params", "current_dt", "critic_feedback", "strategy", "best_mse", "current_mse",
        "current_per_state_mse", "current_per_state_ise", "current_unstable",
        "exploration_intensity", "intensity_low", "intensity_high", "aggressive_low", "aggressive_high",
    ],
    template=ACTOR_PROMPT_TEMPLATE,
)


def _intensity_to_multiplier_range(intensity: int) -> tuple:
    """Maps the user's 1-100 'Exploration Intensity' slider to a concrete
    multiplicative-change range for the Actor's 'explore' proposals, via
    piecewise-linear interpolation through three anchor points:
        intensity=1   -> [1.02x, 1.10x]  (very cautious)
        intensity=50  -> [1.50x, 3.00x]  (the original, pre-slider default)
        intensity=100 -> [3.00x, 10.0x]  (as bold as plain 'explore' gets --
                                           matches where 'aggressive_explore'
                                           used to start)
    'aggressive_explore' always goes further still than whatever 'explore'
    is currently doing (roughly double the upper end), so the two concepts
    (the intensity slider, and the plateau-triggered aggressive escalation
    in critic.py) compose sensibly instead of colliding.
    """
    intensity = max(1, min(100, int(intensity)))
    if intensity <= 50:
        t = (intensity - 1) / 49.0
        low = 1.02 + t * (1.5 - 1.02)
        high = 1.10 + t * (3.0 - 1.10)
    else:
        t = (intensity - 50) / 50.0
        low = 1.5 + t * (3.0 - 1.5)
        high = 3.0 + t * (10.0 - 3.0)
    return low, high


def _repair_dimensions(params: MPCParameters, n_states: int, n_inputs: int) -> Dict[str, Any]:
    """Defensive repair in case the LLM returns a Q/R/P of the wrong length
    (kept from the original notebook's behaviour -- LLM structured output can
    still occasionally violate array-length constraints in practice)."""
    q = list(params.Q)
    if len(q) != n_states:
        log.warning("Actor returned Q of length %d, expected %d -- resetting to 1.0", len(q), n_states)
        q = [1.0] * n_states

    r = list(params.R)
    if len(r) != n_inputs:
        log.warning("Actor returned R of length %d, expected %d -- resetting to 0.1", len(r), n_inputs)
        r = [0.1] * n_inputs

    p = list(params.P) if params.P else None
    if p is not None and len(p) != n_states:
        log.warning("Actor returned P of length %d, expected %d -- resetting to Q", len(p), n_states)
        p = q

    return {"Np": params.Np, "Nc": params.Nc, "Q": q, "R": r, "P": p if p is not None else q, "dt": params.dt}


def _format_params_block(params: Dict[str, Any]) -> str:
    """Renders the numeric parameters the Actor arrived at this iteration --
    the part the Agent Reasoning panel used to omit entirely (it only ever
    showed the prose reasoning, truncated to 200 characters, never the
    actual Np/Nc/Q/R/P/dt the run is now using)."""
    dt = params.get("dt")
    dt_str = f"{dt:.4g}s" if dt is not None else "unchanged"
    return (
        f"Np={params.get('Np')}  Nc={params.get('Nc')}  dt={dt_str}\n"
        f"Q={round_floats(params.get('Q'))}\n"
        f"R={round_floats(params.get('R'))}\n"
        f"P={round_floats(params.get('P'))}"
    )


def _fallback_params(state: Dict[str, Any], n_states: int, n_inputs: int) -> Dict[str, Any]:
    """Used only if the Actor LLM call fails even after a retry (see
    invoke_with_retry) -- keep whatever the last known-good parameters were
    rather than guessing something new and rather than crashing the run.
    Falls back to a conservative flat default only if there's no prior
    proposal at all (i.e. this is iteration 0)."""
    current = state.get("current_params")
    if current and current.get("Q") and current.get("R"):
        return current
    return {"Np": 10, "Nc": 5, "Q": [1.0] * n_states, "R": [0.1] * n_inputs, "P": [1.0] * n_states, "dt": None}


def actor_node(state: Dict[str, Any], cfg=None) -> Dict[str, Any]:
    n_states, n_inputs = state["n_states"], state["n_inputs"]

    exploration_intensity = state.get("exploration_intensity", 50)
    intensity_low, intensity_high = _intensity_to_multiplier_range(exploration_intensity)
    aggressive_low = intensity_high
    aggressive_high = min(intensity_high * 2.0, 15.0)

    llm = get_llm().with_structured_output(MPCParameters)
    prompt_text = actor_prompt.format(
        system_name=state.get("system_name", "unknown"),
        n_states=n_states,
        n_inputs=n_inputs,
        state_names=state.get("state_names", []),
        input_names=state.get("input_names", []),
        user_guidance_block=format_user_guidance(state.get("user_guidance", "")),
        current_params=round_floats(state.get("current_params", {})),
        current_dt=round_floats(cfg.data.dt_mpc) if cfg is not None else "unknown",
        critic_feedback=state.get("critic_feedback", "(first iteration -- no feedback yet)"),
        strategy=state.get("strategy", "explore"),
        best_mse=round_floats(state.get("best_mse", float("inf"))),
        current_mse=round_floats(state.get("current_mse", float("nan"))),
        current_per_state_mse=round_floats(state.get("current_per_state_mse", {})),
        current_per_state_ise=round_floats(state.get("current_per_state_ise", {})),
        current_unstable=state.get("current_unstable", False),
        exploration_intensity=exploration_intensity,
        intensity_low=intensity_low,
        intensity_high=intensity_high,
        aggressive_low=aggressive_low,
        aggressive_high=aggressive_high,
    )

    try:
        proposal: MPCParameters = invoke_with_retry(llm, prompt_text, max_retries=1, node_name="Actor",
                                                       tracker=state.get("token_tracker"))
    except Exception as e:  # noqa: BLE001
        # A single bad response (most commonly a provider-side schema/bounds
        # validation error -- see the HARD BOUNDS note above and
        # llm_base.invoke_with_retry's docstring) used to propagate straight
        # through graph.stream() and abort the entire run, discarding every
        # iteration completed so far. Falling back to the last known-good
        # parameters keeps the run alive instead.
        log.error("[Actor] LLM call failed after retry, keeping previous parameters: %s", e)
        params = _fallback_params(state, n_states, n_inputs)
        history: List[str] = state.get("history", []) + [
            f"[Actor] FAILED to get a valid proposal ({e}); keeping previous parameters unchanged.\n"
            f"{_format_params_block(params)}"
        ]
        return {
            **state,
            "current_params": params,
            "strategy": "exploit",  # be conservative right after a failure
            "actor_reasoning": f"(fallback -- LLM call failed after retry: {e})",
            "history": history,
            "iteration": state.get("iteration", 0) + 1,
            "last_outputs": merge_last_output(state, "actor", f"FAILED to get a valid proposal ({e}); keeping previous parameters unchanged."),
        }

    params = _repair_dimensions(proposal, n_states, n_inputs)

    log.info("[Actor] iteration=%s strategy=%s Np=%d Nc=%d", state.get("iteration"), proposal.strategy, params["Np"], params["Nc"])

    history: List[str] = state.get("history", [])
    history = history + [
        f"[Actor] strategy={proposal.strategy}\n"
        f"{_format_params_block(params)}\n\n"
        f"{proposal.reasoning}"
    ]

    return {
        **state,
        "current_params": params,
        "strategy": proposal.strategy,
        "actor_reasoning": proposal.reasoning,
        "history": history,
        "iteration": state.get("iteration", 0) + 1,
        "last_outputs": merge_last_output(state, "actor", f"Strategy: {proposal.strategy}\n\n{proposal.reasoning}"),
    }
