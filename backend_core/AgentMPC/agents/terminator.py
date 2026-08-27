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

log = get_logger(__name__)

Decision = Literal["critic", "juror"]


class TerminationDecision(BaseModel):
    decision: Decision
    reason: str


TERMINATOR_PROMPT_TEMPLATE = """
You are the Terminator in an MPC tuning loop. You do NOT end runs yourself --
your only job is deciding whether normal tuning should continue, or whether
it's time for the Juror (the final reviewer) to look at things, either
because performance looks good enough to consider wrapping up, or because
something looks structurally stuck.

Iteration: {iteration} / {max_iterations}
Current MSE: {current_mse}   Best MSE: {best_mse}
MSE history: {mse_history}
Plateaued (best MSE hasn't meaningfully improved over the last several iterations): {plateaued}

Decide the next step:
  - "critic": performance can likely still be improved with normal tuning --
             there's clear room left and no sign of being stuck.
  - "juror": time for the final reviewer, for EITHER reason: (a) performance
             looks satisfactory and this could plausibly be a good place to
             stop, or (b) something looks structurally wrong (metrics not
             improving, or plateaued in a way normal tuning won't fix).
             Strongly consider this if Plateaued is True, or if the budget
             (Iteration vs Max Iterations) is running low.
""".strip()

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
            "[Terminator] Decision: route to Juror \u2014 max_iterations reached (numeric guard)"]
        return {**state, "should_continue": True, "_next": "juror", "history": history,
                "termination_reason": "max_iterations reached (numeric guard)",
                "last_outputs": merge_last_output(state, "terminator", "Decision: route to Juror (final review) \u2014 max_iterations reached")}

    if len(mse_history) >= 2 and mse_history[-1] <= (state.get("target_mse") or -1):
        history: List[str] = state.get("history", []) + [
            "[Terminator] Decision: route to Juror \u2014 target MSE reached (numeric guard)"]
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
        history: List[str] = state.get("history", []) + [f"[Terminator] Decision: route to Critic (fallback) \u2014 {reason}"]
        return {**state, "should_continue": True, "termination_reason": reason, "_next": "critic", "history": history,
                "last_outputs": merge_last_output(state, "terminator", reason)}

    log.info("[Terminator] decision=%s reason=%s", result.decision, result.reason)
    history: List[str] = state.get("history", []) + [f"[Terminator] Decision: route to {result.decision.capitalize()} \u2014 {result.reason}"]

    return {
        **state,
        "should_continue": True,  # the Terminator itself never ends a run -- see module docstring
        "termination_reason": result.reason,
        "_next": result.decision,  # "critic" or "juror" -- consumed by graph/workflow.py's conditional edge
        "history": history,
        "last_outputs": merge_last_output(state, "terminator", f"Decision: route to {result.decision.capitalize()} \u2014 {result.reason}"),
    }
