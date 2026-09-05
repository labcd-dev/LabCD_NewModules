"""
================================================================================
agents/terminator.py
================================================================================
Terminator node: decides whether to continue tuning (route to Critic) or
hand off to the Juror. As of this version, the Terminator can NO LONGER end
a run directly -- every path to actually stopping (numeric guards below, or
the LLM's own judgment that things look done) now routes to "juror" instead
of "end". The Juror (agents/juror.py) is the sole place a run can truly end
(via its accept_and_end verdict), acting as a mandatory final quality check
-- not just an escalation handler for stuck runs -- on every single run.

Change vs. the original: termination is no longer decided by the LLM alone.
A cheap numeric guard runs first (max_iterations reached, or target MSE
reached) and can short-circuit straight to the Juror without an LLM call --
the LLM is only asked to make a judgment call in the ambiguous middle
ground. This bounds worst-case cost/latency of a tuning run and removes the
risk of the LLM indefinitely deciding "continue" on a plateaued search.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from ..utils.logging_utils import get_logger
from .convergence import is_plateaued
from .formatting import round_floats
from .llm_base import get_llm, invoke_with_retry, merge_last_output
from .prompt_library import get_prompt

log = get_logger(__name__)

Decision = Literal["critic", "juror"]


class TerminationDecision(BaseModel):
    decision: Decision
    reason: str


TERMINATOR_PROMPT_TEMPLATE = get_prompt("terminator")

terminator_prompt = PromptTemplate(
    input_variables=["iteration", "max_iterations", "current_mse", "best_mse", "mse_history", "plateaued"],
    template=TERMINATOR_PROMPT_TEMPLATE,
)


def should_continue(state: Dict[str, Any]) -> Dict[str, Any]:
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 20)
    mse_history = state.get("mse_history", [])

    # --- numeric guards: bypass the LLM entirely when the answer is obvious,
    # but ALWAYS route to juror rather than ending directly -- see module
    # docstring. ---
    if iteration >= max_iterations:
        history: List[str] = state.get("history", []) + [
            f"[Terminator] decision=juror  iteration={iteration}/{max_iterations}\n\n"
            f"max_iterations reached (numeric guard, no LLM call)"]
        return {**state, "should_continue": True, "_next": "juror", "history": history,
                "termination_reason": "max_iterations reached (numeric guard)",
                "last_outputs": merge_last_output(state, "terminator", "Decision: route to Juror (final review) \u2014 max_iterations reached")}

    if len(mse_history) >= 2 and mse_history[-1] <= (state.get("target_mse") or -1):
        history: List[str] = state.get("history", []) + [
            f"[Terminator] decision=juror  iteration={iteration}/{max_iterations}  "
            f"current_mse={round_floats(mse_history[-1])}  target_mse={round_floats(state.get('target_mse'))}\n\n"
            f"target MSE reached (numeric guard, no LLM call)"]
        return {**state, "should_continue": True, "_next": "juror", "history": history,
                "termination_reason": "target MSE reached (numeric guard)",
                "last_outputs": merge_last_output(state, "terminator", "Decision: route to Juror (final review) \u2014 target MSE reached")}

    # --- otherwise, ask the LLM to make the judgment call ---
    llm = get_llm().with_structured_output(TerminationDecision)
    prompt_text = terminator_prompt.format(
        iteration=iteration,
        max_iterations=max_iterations,
        current_mse=round_floats(state.get("current_mse")),
        best_mse=round_floats(state.get("best_mse")),
        mse_history=round_floats(mse_history),
        plateaued=is_plateaued(mse_history),
    )

    try:
        result: TerminationDecision = invoke_with_retry(llm, prompt_text, max_retries=1, node_name="Terminator",
                                                            tracker=state.get("token_tracker"))
    except Exception as e:  # noqa: BLE001
        # Same reasoning as agents/actor.py: don't let one bad LLM response
        # abort a run that may already be many iterations deep. Falling back
        # to "keep tuning" (route to critic) is the safe default -- worst
        # case it uses one more iteration of the budget, rather than ending
        # the run early or crashing it entirely.
        log.error("[Terminator] LLM call failed after retry, defaulting to 'continue': %s", e)
        reason = f"(fallback -- Terminator LLM call failed after retry: {e}. Defaulting to continue.)"
        history: List[str] = state.get("history", []) + [
            f"[Terminator] decision=critic  iteration={iteration}/{max_iterations}\n\n{reason}"
        ]
        return {**state, "should_continue": True, "termination_reason": reason, "_next": "critic", "history": history,
                "last_outputs": merge_last_output(state, "terminator", reason)}

    log.info("[Terminator] decision=%s reason=%s", result.decision, result.reason)
    # The numbers below are what the LLM actually reasoned over -- the panel
    # used to show only the "route to X" sentence with no visible context
    # for why, unless that context happened to be repeated in result.reason.
    history: List[str] = state.get("history", []) + [
        f"[Terminator] decision={result.decision}  iteration={iteration}/{max_iterations}  "
        f"current_mse={round_floats(state.get('current_mse'))}  best_mse={round_floats(state.get('best_mse'))}  "
        f"plateaued={is_plateaued(mse_history)}\n\n{result.reason}"
    ]

    return {
        **state,
        "should_continue": True,  # the Terminator itself never ends a run -- see module docstring
        "termination_reason": result.reason,
        "_next": result.decision,  # "critic" or "juror" -- consumed by graph/workflow.py's conditional edge
        "history": history,
        "last_outputs": merge_last_output(state, "terminator", f"Decision: route to {result.decision.capitalize()} \u2014 {result.reason}"),
    }
