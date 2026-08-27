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
should be copied as-is from the notebook into ``ACTOR_PROMPT_TEMPLATE`` below,
rather than reconstructed from memory. Everything else (structured-output
parsing, dimension repair, logging) is complete and ready to use.

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
from .schemas import MPCParameters

log = get_logger(__name__)

# TODO: paste the full prompt text from the original notebook (cell 19,
# `ACTOR_PROMPT_TEMPLATE`) here. Keep the same input_variables list below in
# sync with whatever placeholders the prompt text actually uses.
ACTOR_PROMPT_TEMPLATE = """
You are the Actor in an MPC parameter-tuning loop for the system
"{system_name}" ({n_states} states: {state_names}; {n_inputs} inputs: {input_names}).

{user_guidance_block}

Current parameters: {current_params}
Current dt_mpc (MPC sample time, seconds): {current_dt}
Critic feedback: {critic_feedback}
Strategy hint: {strategy}

Best MSE so far: {best_mse}   Current MSE: {current_mse}
Per-state MSE this iteration (use this to decide which Q weight to raise/lower,
rather than scaling all of Q uniformly): {current_per_state_mse}
Per-state integral squared error (ISE = accumulated squared tracking error over
the entire run, per state -- prefer this over per-state MSE when the trajectory
is a moving reference, since it's the more standard tracking-quality signal):
{current_per_state_ise}
Unstable this iteration: {current_unstable}

Propose the next MPC parameters (Np, Nc, Q, R, P, and optionally dt) as
structured output.

HARD BOUNDS (violating these will cause your response to be rejected):
  - Np must be an integer between 1 and 60 (inclusive).
  - Nc must be an integer between 1 and 60 (inclusive), and Nc <= Np.
  - All Q, R, P values must be positive.
  - dt, if you set it, must be strictly positive and no more than 2 seconds.

IMPORTANT: Np (prediction horizon) and Nc (control horizon) are just as much
design parameters as Q and R -- don't leave them fixed while only adjusting
weights. A too-short Np can cause myopic/oscillatory control; a too-long Np
wastes computation and can make the controller sluggish. If strategy is
'explore' or 'aggressive_explore', actually vary Np/Nc across iterations
(not just Q/R) to find a horizon that suits this system, not only the
horizon you started with -- but stay within the hard bounds above.

dt_mpc (the controller's own sample time) is ALSO one of your tunable
parameters, alongside Q/R/Np/Nc -- don't leave it untouched for the whole
run any more than you would Np/Nc. If strategy is 'explore' or
'aggressive_explore', periodically try a new dt value (roughly every few
iterations, not literally every single one) rather than only reacting to an
obvious problem -- you won't know whether a different sample time helps
until you've actually tried a few. Reasonable signals to prompt a change:
the response looks under-sampled/jerky relative to dt (try finer), or dt
looks unnecessarily fine and iteration feels slow for no accuracy benefit
(try coarser). Changing dt reshapes the whole discretization of the
problem, so prefer small, deliberate adjustments (not more than roughly
2x-5x up or down in one step) over large jumps, and expect to also revisit
Np/Nc afterward since their EFFECTIVE prediction horizon in real time is
Np * dt. Omit the "dt" field entirely on iterations where you're not
proposing a change (this leaves it exactly as it was, it does not reset to
a default).

EXPLORATION INTENSITY: {exploration_intensity}% (user-set, 1=very cautious,
50=normal, 100=very bold). If strategy is 'explore', target multiplicative
changes on individual Q/R entries roughly in the range {intensity_low:.2f}x
to {intensity_high:.2f}x per iteration (higher = bolder change) -- this
range already reflects the chosen intensity, so use it directly rather than
picking your own step size.

If strategy is 'aggressive_explore': the search has stalled -- a normal-sized
adjustment already isn't working. Propose a MUCH bolder change than a regular
'explore' step: multiply or divide individual Q/R entries by roughly
{aggressive_low:.2f}x to {aggressive_high:.2f}x, and/or try a substantially
different Np/Nc, rather than nudging the current values. The goal is to jump
to a genuinely different region of the parameter space, not to fine-tune the
current one.

If Unstable is True, propose a more conservative change than usual (the last
proposal caused the system to diverge) -- this overrides the
exploration-intensity/'aggressive_explore' boldness instructions above;
safety comes first.
""".strip()

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
            f"[Actor] FAILED to get a valid proposal ({e}); keeping previous parameters unchanged."
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
    history = history + [f"[Actor] {proposal.strategy}: {proposal.reasoning[:200]}"]

    return {
        **state,
        "current_params": params,
        "strategy": proposal.strategy,
        "actor_reasoning": proposal.reasoning,
        "history": history,
        "iteration": state.get("iteration", 0) + 1,
        "last_outputs": merge_last_output(state, "actor", f"Strategy: {proposal.strategy}\n\n{proposal.reasoning}"),
    }
