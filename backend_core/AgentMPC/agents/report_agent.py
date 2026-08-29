"""
================================================================================
agents/report_agent.py
================================================================================
Report Agent: runs once, on demand (from a button in the UI, NOT as part of
the LangGraph tuning loop), after a run has stopped or finished. Produces a
genuine, in-depth analysis -- not a template fill-in, and not a short
summary either -- of the system, the tuning process itself, the final
controller configuration, and the results achieved, which app.py then lays
out into a PDF (see report.py, built on the shared labcd_pdfmaker package)
alongside the actual charts and data table.

Kept deliberately separate from the tuning graph: report generation doesn't
need LangGraph's node/state machinery (it runs exactly once, has no routing
decisions, and doesn't feed back into anything) -- it's a single, larger LLM
call given the FULL run history as context, following the same
invoke_with_retry / fallback pattern as every other agent in this package.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from ..utils.logging_utils import get_logger
from .formatting import round_floats
from .llm_base import get_llm, invoke_with_retry

log = get_logger(__name__)


class ReportAnalysis(BaseModel):
    system_analysis: str = Field(
        description="4-7 sentences, at least one full paragraph, characterizing the SYSTEM itself in "
        "real engineering depth: what kind of dynamics it has (underactuated vs. fully actuated, "
        "open-loop stable vs. unstable, linear vs. strongly nonlinear as far as you can tell from the "
        "state/input names and how hard the tuning process had to work), how the state count and "
        "input count relate to the system's degrees of freedom, and any coupling between states that "
        "the naming suggests (e.g. position/velocity pairs, angles). Ground every claim in the actual "
        "names and numbers given -- do not write generic boilerplate that could apply to any system.")
    controller_analysis: str = Field(
        description="6-10 sentences, at least two short paragraphs, analyzing the FINAL controller "
        "configuration in depth. Cover: (1) the Q/R weight balance specifically -- which individual "
        "states got the highest weights and why that is or isn't sensible for this system, and whether "
        "Q dominates R by a reasonable margin (standard practice) or not; (2) whether Np and Nc look "
        "proportionate to the system's own response speed (inferred from dt and the settling behavior), "
        "including whether Nc is a sensible fraction of Np; (3) dt_mpc specifically -- did it change "
        "during the run, and if so what does the direction of that change (finer vs. coarser) suggest "
        "about the system's natural timescale; if it never changed, say so plainly rather than inventing "
        "a reason. Reference the actual numbers throughout, not just their general shape.")
    search_process_analysis: str = Field(
        description="5-8 sentences on HOW the tuning search itself unfolded, based on the strategy "
        "history and MSE trajectory given below: did it spend most iterations exploring vs. exploiting, "
        "was there a plateau that triggered an escalation, how many iterations did it take to reach a "
        "good result vs. how many were available, and were there any unstable/failed iterations along "
        "the way and what (if anything) the Actor seemed to learn from them (e.g. did parameters after "
        "an unstable iteration look more conservative). This section is about the PROCESS, not the "
        "final numbers -- write it as a narrative of what happened over the course of the run.")
    results_analysis: str = Field(
        description="6-9 sentences, at least two short paragraphs, on the ACHIEVED performance in real "
        "depth: the MSE/overshoot/settling/stability trend across iterations with specific numbers "
        "compared (e.g. first vs. best, expressed as a ratio or percentage where sensible), whether the "
        "run converged cleanly, plateaued, or was still improving when it stopped, and what the "
        "stability/oscillation numbers say about the QUALITY of the response beyond just the MSE scalar "
        "(e.g. a low MSE that's still oscillating is a different situation than a low MSE that's smooth "
        "and settled). Be honest about any weaknesses in the final result, not just its strengths.")
    theoretical_context: str = Field(
        description="4-6 sentences connecting the observed results to standard MPC/control theory "
        "principles relevant to what actually happened here -- e.g. the prediction-horizon/control-"
        "horizon tradeoff, the tracking-vs-regulation distinction if a moving reference was used, "
        "the role of terminal weighting, or the effect of sample time on discretization accuracy -- "
        "whichever of these are actually relevant to the numbers seen, not a generic MPC theory recap. "
        "This section should help a reader understand WHY the final parameters make sense (or don't) "
        "in terms of established practice, not just restate what they are.")
    recommendations: str = Field(
        description="6-9 sentences of concrete, specific, and varied suggestions for further improvement "
        "or things worth double-checking, each grounded in a specific number or observation from this "
        "run (e.g. 'per-state error for state X remained N times higher than the others even in the "
        "best iteration, suggesting its Q weight could go higher still', 'the run stopped exactly at the "
        "iteration budget while MSE was still trending down -- a longer budget may yield further gains', "
        "'settling time could not be confirmed within the simulation window -- consider a longer "
        "simulation_time to verify true steady-state'). Avoid generic advice that would apply to any run "
        "regardless of its actual results.")
    conclusion: str = Field(
        description="3-5 sentences summarizing the overall verdict on this tuning run -- is the final "
        "controller ready to use, usable with caveats, or in need of more work -- and the single most "
        "important thing a reader should take away from this report.")


REPORT_PROMPT_TEMPLATE = """
You are the Report Agent, producing the analysis sections of an in-depth technical
report for one completed MPC (Model Predictive Control) auto-tuning run. Write as
a controls engineer thoroughly reviewing real results for a colleague, not as a
generic summary generator -- reference the ACTUAL numbers below throughout, draw
connections between different pieces of evidence (e.g. tie the strategy history to
the MSE trend, tie the Q/R balance to the per-state error breakdown), and say
plainly if something looks off (a metric that didn't improve, an unstable
iteration, a suspiciously small Q relative to R, a run that stopped before truly
converging) rather than only praising the outcome. Each section should read as
genuine analysis with reasoning shown, not a list of facts restated -- explain
WHY the evidence supports your conclusion, not just WHAT the evidence is.

SYSTEM: "{system_name}"
States ({n_states}): {state_names}
Inputs ({n_inputs}): {input_names}

RUN SUMMARY:
Total iterations: {n_iterations}  (successful: {n_ok}, unstable: {n_unstable}, failed: {n_failed})
Stopped by user: {stopped_by_user}
Initial MSE (iteration 1): {first_mse}
Best MSE achieved: {best_mse}
MSE history (all iterations, in order): {mse_history}
Strategy per iteration (in order -- explore/exploit/aggressive_explore): {strategy_history}
dt_mpc per iteration (in order -- watch for any changes): {dt_history}

FINAL / BEST CONTROLLER CONFIGURATION:
Np (prediction horizon): {best_np}
Nc (control horizon): {best_nc}
Q (state weights, one per state, same order as the state list above): {best_q}
R (input weights, one per input, same order as the input list above): {best_r}
dt_mpc (sample time): {best_dt}s

BEST-ITERATION METRICS:
Overshoot: {best_overshoot}
Settling time: {best_settling}
Control effort: {best_effort}
Is stable: {best_is_stable}
Oscillation count: {best_oscillations}
Per-state MSE breakdown (best iteration, same order as the state list above): {best_per_state_mse}

Write all seven analysis sections as structured output. Take the space you need --
these are meant to be substantive, multi-sentence (often multi-paragraph) sections
of a real report, not one-line summaries.
""".strip()

report_prompt = PromptTemplate(
    input_variables=[
        "system_name", "n_states", "n_inputs", "state_names", "input_names",
        "n_iterations", "n_ok", "n_unstable", "n_failed", "stopped_by_user",
        "first_mse", "best_mse", "mse_history", "strategy_history", "dt_history",
        "best_np", "best_nc", "best_q", "best_r", "best_dt",
        "best_overshoot", "best_settling", "best_effort", "best_is_stable", "best_oscillations",
        "best_per_state_mse",
    ],
    template=REPORT_PROMPT_TEMPLATE,
)


def _fallback_analysis(context: Dict[str, Any]) -> ReportAnalysis:
    """Used only if the LLM call fails even after a retry -- a plain,
    numbers-only fallback so a report can still be generated (with far less
    insight, but not blocked entirely) rather than failing outright."""
    note = " (Automatic analysis unavailable -- the Report Agent's LLM call failed; the PDF still " \
           "includes every chart, table, and number below.)"
    return ReportAnalysis(
        system_analysis=f"{context['system_name']} has {context['n_states']} states and "
                         f"{context['n_inputs']} inputs.{note}",
        controller_analysis=f"Final configuration: Np={context['best_np']}, Nc={context['best_nc']}, "
                             f"dt={context['best_dt']}s, Q={context['best_q']}, R={context['best_r']}.{note}",
        search_process_analysis=f"{context['n_iterations']} iterations were run "
                                 f"({context['n_ok']} successful, {context['n_unstable']} unstable, "
                                 f"{context['n_failed']} failed).{note}",
        results_analysis=f"Best MSE achieved: {context['best_mse']} (started at {context['first_mse']}).{note}",
        theoretical_context=f"See the charts and table below for the full quantitative picture.{note}",
        recommendations=f"Review the Data & Export table and Convergence chart directly.{note}",
        conclusion=f"Automatic conclusion unavailable.{note}",
    )


def generate_report_analysis(
    system_name: str,
    state_names: List[str],
    input_names: List[str],
    results_data: List[Dict[str, Any]],
    best_row: Optional[Dict[str, Any]],
    stopped_by_user: bool,
    tracker: Optional[Any] = None,
) -> ReportAnalysis:
    """Pure function: takes the same results_data/best_row app.py already
    has in session_state, returns the analysis sections. No LangGraph state
    involved -- this runs once, on demand, outside the tuning graph.

    ``tracker``: optional TokenUsageTracker (see agents/llm_base.py) -- if
    given, this call's usage is accumulated into the same run-wide tracker
    as the main tuning loop.
    """
    ok_rows = [r for r in results_data if r.get("ok")]
    n_unstable = sum(1 for r in results_data if r.get("unstable"))
    n_failed = len(results_data) - len(ok_rows)
    mse_history = [r["mse"] for r in ok_rows if r.get("mse") is not None]
    first_mse = mse_history[0] if mse_history else None
    strategy_history = [r.get("strategy", "?") for r in results_data]
    dt_history = [r["dt_mpc"] for r in results_data if r.get("dt_mpc") is not None]

    best_row = best_row or {}
    context = dict(
        system_name=system_name,
        n_states=len(state_names) or "?",
        n_inputs=len(input_names) or "?",
        state_names=", ".join(state_names) or "unknown",
        input_names=", ".join(input_names) or "unknown",
        n_iterations=len(results_data),
        n_ok=len(ok_rows),
        n_unstable=n_unstable,
        n_failed=n_failed,
        stopped_by_user=stopped_by_user,
        first_mse=round_floats(first_mse) if first_mse is not None else "n/a",
        best_mse=round_floats(best_row.get("mse")) if best_row.get("mse") is not None else "n/a",
        mse_history=round_floats(mse_history),
        strategy_history=strategy_history,
        dt_history=round_floats(dt_history) if dt_history else "not tracked / never changed",
        best_np=best_row.get("np", "n/a"),
        best_nc=best_row.get("nc", "n/a"),
        best_q=best_row.get("Q_formatted", "n/a"),
        best_r=best_row.get("R_formatted", "n/a"),
        best_dt=round_floats(best_row.get("dt_mpc")) if best_row.get("dt_mpc") is not None else "n/a",
        best_overshoot=round_floats(best_row.get("overshoot")) if best_row.get("overshoot") is not None
                       and best_row.get("overshoot_meaningful", True) else "N/A (tracking a moving reference, or zero initial error)",
        best_settling=round_floats(best_row.get("settling")) if best_row.get("settling") not in (None, float("inf")) else "not settled within simulation window",
        best_effort=round_floats(best_row.get("effort")) if best_row.get("effort") is not None else "n/a",
        best_is_stable=best_row.get("is_stable", "n/a"),
        best_oscillations=best_row.get("oscillation_count", "n/a"),
        best_per_state_mse=round_floats(best_row.get("per_state_mse", {})) or "not available",
    )

    try:
        llm = get_llm().with_structured_output(ReportAnalysis)
        prompt_text = report_prompt.format(**context)
        return invoke_with_retry(llm, prompt_text, max_retries=1, node_name="ReportAgent", tracker=tracker)
    except Exception as e:  # noqa: BLE001
        log.error("[ReportAgent] LLM call failed after retry, using numbers-only fallback: %s", e)
        return _fallback_analysis(context)
