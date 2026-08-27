"""
================================================================================
agents/critic.py
================================================================================
Critic node: looks at the Evaluator's metrics + history and produces feedback
for the next Actor call. Port your prompt text from the original notebook
(cell 17) into CRITIC_PROMPT_TEMPLATE below.

Design note carried over from the review: the original Critic gives
*qualitative-only* feedback ("no numbers") and leaves the Actor to translate
that into numeric weight changes on its own -- two LLM calls independently
approximating the same numeric reasoning. If you want tighter, more
reproducible convergence, consider having the Critic emit a structured
`suggested_direction: Dict[str, float]` (e.g. relative multipliers per state)
alongside the qualitative text, and have the Actor use it as a strong prior
instead of re-deriving it from prose. The schema below supports this
(`suggested_multipliers` is optional so plain qualitative feedback still
works if you don't want to make that change).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from ..utils.logging_utils import get_logger
from .convergence import is_plateaued
from .formatting import round_floats
from .llm_base import format_user_guidance, get_llm, invoke_with_retry, merge_last_output

log = get_logger(__name__)


class CriticFeedback(BaseModel):
    feedback: str = Field(description="At most 3 sentences: what to change and why.")
    strategy_recommendation: str = Field(
        description="'explore' (moderate changes), 'exploit' (fine-tune near the best known params), "
        "or 'aggressive_explore' (large, bold changes -- recommend this instead of 'explore' only when "
        "progress has clearly stalled and a normal-sized change is unlikely to help)."
    )
    suggested_multipliers: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional: e.g. {'Q_2': 1.5, 'R_0': 0.8} -- multiplicative "
        "hints per weight index, for a tighter numeric handoff to the Actor.",
    )


# TODO: paste the full prompt text from the original notebook (cell 17) here.
# Kept the per-state / oscillation / unstable placeholders below when you do --
# they give the Critic the information to recommend *which* weight to change,
# not just "MSE is high".
CRITIC_PROMPT_TEMPLATE = """
You are the Critic in an MPC parameter-tuning loop.

{user_guidance_block}

Trajectory type: {trajectory_kind}

Current params: {current_params}
Current metrics: MSE={current_mse}, overshoot={current_overshoot}, settling={current_settling}, effort={current_effort}
Oscillation (zero-crossings of the worst-off state's error): {current_oscillation_count}
Unstable this iteration: {current_unstable}

Per-state MSE (use this to identify WHICH state needs a higher/lower Q weight,
rather than only reasoning about the aggregate MSE): {current_per_state_mse}
Per-state overshoot (fraction past target, per state): {current_per_state_overshoot}
Per-state integral squared error (ISE = accumulated squared tracking error over
the ENTIRE run, per state -- this is the most reliable per-state signal for a
MOVING reference trajectory, since overshoot/oscillation above are not
meaningful when the target itself moves): {current_per_state_ise}

{regulation_note}

Best so far: MSE={best_mse}, overshoot={best_overshoot}, settling={best_settling}, effort={best_effort}
History (MSE per iteration): {mse_history}

Give qualitative feedback on what to change and why -- if one state's error
(MSE or ISE, whichever applies -- see above) clearly dominates the others,
say so explicitly and recommend increasing that state's Q weight (or R, if
control effort/oscillation is the issue) -- and recommend 'explore',
'exploit', or 'aggressive_explore'. Recommend 'aggressive_explore' instead of
'explore' only if progress has clearly stalled over the recent history and a
small adjustment is unlikely to help -- it signals the Actor to make much
larger, bolder changes than a normal explore step. If Unstable is True, be
much more conservative (increase weights less aggressively, or propose a
smaller prediction/control horizon) since the previous proposal caused the
system to diverge.
""".strip()

critic_prompt = PromptTemplate(
    input_variables=[
        "user_guidance_block", "trajectory_kind",
        "current_params", "current_mse", "current_overshoot", "current_settling", "current_effort",
        "current_oscillation_count", "current_unstable", "current_per_state_mse", "current_per_state_overshoot",
        "current_per_state_ise", "regulation_note",
        "best_mse", "best_overshoot", "best_settling", "best_effort", "mse_history",
    ],
    template=CRITIC_PROMPT_TEMPLATE,
)


def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    iteration = state.get("iteration", 0)
    min_explore = state.get("min_explore_iterations", 4)

    if state.get("eval_error"):
        # deterministic fast-path: don't spend an LLM call reasoning about a
        # simulation that failed to run at all.
        feedback = f"Previous proposal failed to simulate: {state['eval_error']}. Propose a safer/more conservative parameter set."
        return {**state, "critic_feedback": feedback, "strategy": "exploit"}

    llm = get_llm().with_structured_output(CriticFeedback)
    is_regulation = state.get("current_is_regulation", True)
    prompt_text = critic_prompt.format(
        user_guidance_block=format_user_guidance(state.get("user_guidance", "")),
        trajectory_kind="regulation (fixed target)" if is_regulation else "tracking (moving reference)",
        current_params=round_floats(state.get("current_params", {})),
        current_mse=round_floats(state.get("current_mse")),
        current_overshoot=round_floats(state.get("current_overshoot")),
        current_settling=round_floats(state.get("current_settling")),
        current_effort=round_floats(state.get("current_effort")),
        current_oscillation_count=state.get("current_oscillation_count", 0),
        current_unstable=state.get("current_unstable", False),
        current_per_state_mse=round_floats(state.get("current_per_state_mse", {})),
        current_per_state_overshoot=round_floats(state.get("current_per_state_overshoot", {})),
        current_per_state_ise=round_floats(state.get("current_per_state_ise", {})),
        regulation_note=(
            "" if is_regulation else
            "NOTE: this is a TRACKING run (moving reference) -- the overshoot/oscillation "
            "numbers above are 0 / not meaningful by construction (the target itself moves, so the "
            "usual step-response definitions don't apply). Settling time IS still meaningful here "
            "(computed relative to the reference signal's own magnitude). Base your feedback on MSE "
            "and the per-state ISE values."
        ),
        best_mse=round_floats(state.get("best_mse", float("inf"))),
        best_overshoot=round_floats(state.get("best_overshoot")),
        best_settling=round_floats(state.get("best_settling")),
        best_effort=round_floats(state.get("best_effort")),
        mse_history=round_floats(state.get("mse_history", [])),
    )

    try:
        result: CriticFeedback = invoke_with_retry(llm, prompt_text, max_retries=1, node_name="Critic",
                                                      tracker=state.get("token_tracker"))
    except Exception as e:  # noqa: BLE001
        # Same reasoning as agents/actor.py: don't let one bad LLM response
        # abort a run that may already be many iterations deep.
        log.error("[Critic] LLM call failed after retry, using a conservative fallback: %s", e)
        feedback = f"(fallback -- Critic LLM call failed after retry: {e}. Defaulting to exploit.)"
        history: List[str] = state.get("history", []) + [f"[Critic] {feedback}"]
        return {**state, "critic_feedback": feedback, "strategy": "exploit", "history": history,
                "last_outputs": merge_last_output(state, "critic", feedback)}

    strategy = result.strategy_recommendation
    feedback = result.feedback

    # Deterministic guard: don't let the LLM converge to fine-tuning before
    # the search has had a chance to actually cover the parameter space.
    # This directly addresses "it goes to Exploit too fast" -- the LLM's own
    # judgment about when it's "found a good enough region" is qualitative
    # and can be overeager, especially with a placeholder/under-specified
    # prompt. The override still keeps the LLM's qualitative feedback text,
    # only the strategy label is forced.
    if strategy == "exploit" and iteration < min_explore:
        strategy = "explore"
        feedback = (
            f"[Overridden: forcing 'explore' until iteration {min_explore} -- "
            f"the search needs to cover more of the parameter space before fine-tuning. "
            f"Original Critic feedback: {feedback}]"
        )

    # Second deterministic guard: if the search has genuinely stalled (best
    # MSE hasn't meaningfully improved over the last several iterations),
    # escalate past normal 'explore' into 'aggressive_explore' -- this tells
    # the Actor (see actor.py) to propose much larger, bolder parameter
    # jumps instead of incremental adjustments, which converges faster out
    # of a stuck region than repeatedly nudging the same neighborhood. This
    # is a different mechanism from the Juror escalation in terminator.py:
    # the Juror handles "something looks structurally broken" (repeated
    # failures), this handles "tuning is fine, just stuck in a local
    # optimum" -- the far more common case.
    if strategy != "aggressive_explore" and is_plateaued(state.get("mse_history", []), window=5, rel_tol=0.02):
        strategy = "aggressive_explore"
        feedback = (
            f"[Overridden: MSE has plateaued over the last several iterations -- escalating to "
            f"'aggressive_explore' for a bolder parameter jump. Original Critic feedback: {feedback}]"
        )

    log.info("[Critic] recommendation=%s (llm said %s)", strategy, result.strategy_recommendation)
    history: List[str] = state.get("history", []) + [f"[Critic] {feedback}"]

    return {
        **state,
        "critic_feedback": feedback,
        "strategy": strategy,
        "history": history,
        "last_outputs": merge_last_output(state, "critic", feedback),
    }
