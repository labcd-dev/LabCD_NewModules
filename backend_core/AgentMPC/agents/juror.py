"""
================================================================================
agents/juror.py
================================================================================
Juror node. Historically this only ran on escalation (plateau/repeated
failures) and could retry-wider, reset-to-best, or accept-and-end. It is now
the MANDATORY final reviewer for every run: the Terminator no longer ends a
run directly (see agents/terminator.py) -- every path to ending, whether
triggered by max_iterations, a target-MSE hit, or a genuine plateau, goes
through the Juror first. This means the Juror is invoked far more often than
before, and in two distinct situations it needs to tell apart from
``termination_reason``:

  1. "This looks converged, sanity-check before ending" -- the common case
     once tuning is going well. Here the Juror's job is a genuine holistic
     review: do the final Np/Nc/Q/R actually make sense for this system
     (e.g. is Q dominating R the way it should, is Np/Nc proportionate to
     the system's own response speed), not just "is MSE low". If something
     looks off, the Juror can send it back for one more refinement pass
     instead of rubber-stamping a numerically-low-but-structurally-odd
     result.

  2. "This is stuck" -- plateaued or repeatedly failing. The existing
     retry_with_wider_search / reset_to_best escalation behavior.

NOTE on dt_mpc: an earlier version of this file also let the Juror tune
dt_mpc directly (gated behind a minimum-iteration/stability eligibility
check). That has been superseded, for now, by folding dt into the Actor's
own normal parameter set instead -- see agents/schemas.py's
MPCParameters.dt and agents/actor.py's prompt -- so dt is now tuned every
iteration alongside Q/R/Np/Nc by the same Actor/Critic loop, rather than
being held back for a separate, later mechanism. Kept out of the Juror's
own decision space entirely (rather than leaving both paths active) to
avoid two different agents proposing conflicting dt changes in an
unpredictable order.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from ..utils.logging_utils import get_logger
from .formatting import round_floats
from .llm_base import get_llm, invoke_with_retry, merge_last_output
from .prompt_library import get_prompt

log = get_logger(__name__)


class JurorVerdict(BaseModel):
    verdict: Literal["retry_with_wider_search", "reset_to_best", "accept_and_end"]
    explanation: str


JUROR_PROMPT_TEMPLATE = get_prompt("juror")

juror_prompt = PromptTemplate(
    input_variables=["best_params", "best_mse", "mse_history", "termination_reason",
                      "iteration", "max_iterations", "budget_note"],
    template=JUROR_PROMPT_TEMPLATE,
)


def juror_node(state: Dict[str, Any], cfg=None) -> Dict[str, Any]:
    """``cfg`` is accepted (bound via functools.partial, see
    graph/workflow.py) for consistency with the other nodes and in case a
    future version needs it again, but is not currently used -- dt tuning
    now happens in the Actor/evaluator path instead (see module docstring).
    """
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 0)
    mse_history = state.get("mse_history", [])
    budget_exhausted = iteration >= max_iterations > 0

    llm = get_llm().with_structured_output(JurorVerdict)
    prompt_text = juror_prompt.format(
        best_params=round_floats(state.get("best_params", {})),
        best_mse=round_floats(state.get("best_mse")),
        mse_history=round_floats(mse_history),
        termination_reason=state.get("termination_reason", ""),
        iteration=iteration, max_iterations=max_iterations,
        budget_note=("NOTE: the iteration budget is already exhausted -- \"retry_with_wider_search\" "
                      "is NOT an option right now no matter how the results look; choose between "
                      "\"accept_and_end\" and \"reset_to_best\" (which will also then end the run)."
                      if budget_exhausted else ""),
    )

    try:
        result: JurorVerdict = invoke_with_retry(llm, prompt_text, max_retries=1, node_name="Juror",
                                                    tracker=state.get("token_tracker"))
    except Exception as e:  # noqa: BLE001
        # Same reasoning as agents/actor.py -- the Juror only runs when
        # something's already gone sideways (plateau/repeated failures), so
        # the safest fallback here is to stop cleanly with the best result
        # found so far rather than risk looping indefinitely or crashing.
        log.error("[Juror] LLM call failed after retry, accepting the best result and ending: %s", e)
        history: List[str] = state.get("history", []) + [
            f"[Juror] verdict=accept_and_end (fallback)  iteration={iteration}/{max_iterations}\n\n"
            f"FAILED ({e}); accepting the best result found so far and ending."
        ]
        return {
            **state, "juror_verdict": "accept_and_end", "should_continue": False,
            "current_params": state.get("best_params") or state.get("current_params"), "history": history,
            "last_outputs": merge_last_output(state, "juror", f"FAILED ({e}); accepting the best result found so far and ending."),
        }

    verdict = result.verdict
    explanation = result.explanation

    # Deterministic override: max_iterations is a HARD cap, not a
    # suggestion. If the budget is already exhausted, the Juror is not
    # allowed to send the run back to the Actor for "just one more
    # iteration" -- that's exactly what let a run configured for 10
    # iterations actually run 11+ (every subsequent iteration would hit
    # this same numeric guard again with no bound on how many extra
    # iterations could pile up before the LLM happened to choose
    # accept_and_end on its own). Still honor a "reset_to_best" preference
    # by applying that reset before forcing the end, since that's still a
    # meaningful improvement to the final reported result even though we
    # can't spend another iteration searching.
    if budget_exhausted and verdict != "accept_and_end":
        log.warning("[Juror] budget exhausted (iteration %d >= max_iterations %d) -- overriding '%s' to accept_and_end.",
                    iteration, max_iterations, verdict)
        if verdict == "reset_to_best" and state.get("best_params"):
            state = {**state, "current_params": state["best_params"]}
        explanation = f"(iteration budget exhausted -- overridden to end) {explanation}"
        verdict = "accept_and_end"

    log.info("[Juror] verdict=%s", verdict)
    # explanation was previously cut to 150 characters here -- the full text
    # already reached last_outputs (for the flow-diagram hover) but never the
    # Agent Reasoning panel itself. The budget-exhausted override note (if
    # any) is already prefixed into `explanation` above, so it shows here too.
    history: List[str] = state.get("history", []) + [
        f"[Juror] verdict={verdict}  iteration={iteration}/{max_iterations}\n\n{explanation}"
    ]

    new_state: Dict[str, Any] = {
        **state, "juror_verdict": verdict, "history": history,
        "last_outputs": merge_last_output(state, "juror", f"{verdict}: {explanation}"),
    }

    if verdict == "reset_to_best":
        new_state["current_params"] = state.get("best_params")
    if verdict == "accept_and_end":
        new_state["should_continue"] = False

    return new_state
