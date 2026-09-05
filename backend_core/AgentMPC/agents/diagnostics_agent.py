"""
================================================================================
agents/diagnostics_agent.py
================================================================================
Diagnostics Agent: its ONLY job is noticing when something went wrong during
a run and saying so clearly -- not tuning analysis (that's report_agent.py),
just: what broke, how much, and what to do about it.

Two-stage design, deliberately:

  1. scan_for_issues() is a pure, deterministic pattern scan over what's
     already logged (st.session_state.logs, results_data, last_outputs) --
     no LLM call. Known failure modes (LLM rate limits, token/context
     limits, auth errors, QP solver struggles, dynamics crashes,
     instability) have fairly predictable text signatures, and matching
     them with plain keyword/regex checks is more RELIABLE than asking an
     LLM to correctly classify a raw error string -- and it's free/instant,
     so it can run automatically after every single run without needing to
     ask "should I spend an API call checking for problems?"

  2. generate_diagnostics_report() only runs (and only makes an LLM call)
     when the scan above actually found something -- it asks for grounded,
     specific recommendations and a rough "how much did this contribute"
     estimate for each detected category, using the real counts/examples
     the scan collected. If nothing was found, callers should skip this
     stage entirely (there's nothing to explain).

This directly addresses the reported problem of hitting an API limit with
no visible error: the existing per-agent fallback behavior (log the error,
keep going with a safe default) is good for run *resilience*, but means
these events were previously easy to miss, buried in the log panel. This
module's job is to surface them.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from ..utils.logging_utils import get_logger
from .llm_base import get_llm, invoke_with_retry
from .prompt_library import get_prompt

log = get_logger(__name__)


# Each category: keywords to match (case-insensitive substring), a title,
# and a fallback (non-LLM) recommendation used if the LLM stage is skipped
# or fails -- so a useful message is always available even without an
# extra API call succeeding.
_ERROR_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "rate_limit": {
        "keywords": ["rate limit", "429", "too many requests", "quota exceeded", "requests per minute"],
        "title": "LLM API rate limit reached",
        "fallback_recommendation": "Wait a bit before running again, or check your Groq account's rate "
                                    "limits/plan. Each agent already retries once automatically, but "
                                    "repeated rate-limiting across many iterations can still visibly "
                                    "degrade a run (affected iterations just keep their previous "
                                    "parameters unchanged instead of actually being tuned).",
    },
    "token_limit": {
        "keywords": ["context_length", "context length", "maximum context", "token limit", "tokens per minute",
                     "reduce the length", "too many tokens", "context window"],
        "title": "LLM context/token limit reached",
        "fallback_recommendation": "The prompt sent to the model (often the accumulated user guidance text, "
                                    "or a very long dynamics file) got too long for it to accept. Try "
                                    "shortening any free-text guidance you've entered, or check whether "
                                    "your Groq plan's tokens-per-minute limit is being hit across many "
                                    "iterations in a short time.",
    },
    "auth_error": {
        "keywords": ["401", "invalid api key", "unauthorized", "authentication", "incorrect api key"],
        "title": "LLM API authentication error",
        "fallback_recommendation": "Check that GROQ_API_KEY is set correctly (and hasn't expired/been "
                                    "revoked) in the .env file next to app.py, then restart the app.",
    },
    "solver_struggles": {
        "keywords": [],  # detected numerically below, not by text match
        "title": "MPC solver frequently not converging cleanly",
        "fallback_recommendation": "This often means the constraint bounds are too tight for the horizon/"
                                    "weights being tried (an infeasible or barely-feasible QP), or the "
                                    "problem is poorly conditioned. Try loosening state/input bounds, or "
                                    "review whether Q/R weights differ from each other by many orders of "
                                    "magnitude (which can hurt numerical conditioning).",
    },
    "dynamics_crash": {
        "keywords": [],  # detected via eval_error presence, not text match
        "title": "Dynamics simulation crashed on some iterations",
        "fallback_recommendation": "Check the Debug expander in the Live Run tab for the exact traceback. "
                                    "Common causes: a division by a state that reached exactly zero, "
                                    "np.linalg.solve on a singular mass matrix at an extreme state, or a "
                                    "state/input shape mismatch from an edited dynamics file.",
    },
    "frequent_instability": {
        "keywords": [],  # detected via unstable flag ratio, not text match
        "title": "Many iterations were flagged unstable",
        "fallback_recommendation": "If this keeps happening across most iterations regardless of the "
                                    "parameters tried, the search space bounds (Np/Nc/Q/R ranges) may be "
                                    "poorly matched to this system's actual timescale -- check the Initial "
                                    "Setup Analysis panel's suggested dt and Q/R for a sanity check.",
    },
}


ERROR_CATEGORY_TITLES: Dict[str, str] = {k: v["title"] for k, v in _ERROR_CATEGORIES.items()}


class _CategoryRecommendation(BaseModel):
    category: str = Field(description="The category key this recommendation is for, copied exactly from the input.")
    explanation: str = Field(description="2-3 sentences on what specifically happened, grounded in the actual "
                              "counts/examples given -- not generic.")
    recommendation: str = Field(description="1-3 concrete, specific next steps.")
    contribution_estimate: str = Field(description="A short (1 sentence) rough estimate of how much this "
                                        "specific issue contributed to the run's problems -- e.g. 'affected "
                                        "3 of 12 iterations, likely the main reason MSE plateaued early' or "
                                        "'only 1 isolated occurrence, unlikely to have meaningfully affected "
                                        "the final result'.")


class DiagnosticsReport(BaseModel):
    recommendations: List[_CategoryRecommendation]


_DIAGNOSTICS_PROMPT_TEMPLATE = get_prompt("diagnostics_agent")

_diagnostics_prompt = PromptTemplate(
    input_variables=["issues_block", "n_total", "run_context"],
    template=_DIAGNOSTICS_PROMPT_TEMPLATE,
)

_DIAGNOSTICS_CHAT_PROMPT_TEMPLATE = get_prompt("diagnostics_chat")

_diagnostics_chat_prompt = PromptTemplate(
    input_variables=["issues_block", "report_block", "run_error_block", "history_block", "user_message", "run_context"],
    template=_DIAGNOSTICS_CHAT_PROMPT_TEMPLATE,
)


def scan_for_issues(
    logs: List[Dict[str, str]],
    results_data: List[Dict[str, Any]],
    last_outputs: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Pure, deterministic scan -- no LLM call. Returns
    {category_key: {"count": int, "examples": [str, ...], "iterations": [int, ...]}}
    for every category with at least one hit. Empty dict if nothing found.
    """
    findings: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "examples": [], "iterations": []})

    # ---- text-pattern categories: scan logs + last_outputs ----
    text_sources: List[str] = [entry.get("message", "") for entry in logs]
    if last_outputs:
        text_sources.extend(last_outputs.values())

    for text in text_sources:
        if not text:
            continue
        text_lower = text.lower()
        for cat_key, cat in _ERROR_CATEGORIES.items():
            if not cat["keywords"]:
                continue
            if any(kw in text_lower for kw in cat["keywords"]):
                findings[cat_key]["count"] += 1
                if len(findings[cat_key]["examples"]) < 3:
                    findings[cat_key]["examples"].append(text[:300])

    # ---- numeric categories: scan results_data ----
    # NOTE: a failed iteration's row (see agent_mpc_app.py's row-building
    # code) stores its message under the row-local key "error" -- the
    # dynamics/evaluator layer's OWN key for the same string is "eval_error"
    # (agents/evaluator.py's evaluator_node), but that's the graph state key,
    # not what survives into the UI's results_data row. This used to read
    # "eval_error" here, which no row has ever had, so failed iterations were
    # silently invisible to the scan -- "No issues detected" even with every
    # iteration failing.
    n_total = len(results_data)
    n_crashed = sum(1 for r in results_data if not r.get("ok") and r.get("error"))
    if n_crashed > 0:
        findings["dynamics_crash"]["count"] = n_crashed
        findings["dynamics_crash"]["iterations"] = [
            r.get("iteration") for r in results_data if not r.get("ok") and r.get("error")
        ][:10]
        for r in results_data:
            if not r.get("ok") and r.get("error") and len(findings["dynamics_crash"]["examples"]) < 3:
                findings["dynamics_crash"]["examples"].append(str(r["error"])[:300])

    n_unstable = sum(1 for r in results_data if r.get("unstable"))
    if n_total > 0 and n_unstable / n_total > 0.3 and n_unstable >= 2:
        findings["frequent_instability"]["count"] = n_unstable
        findings["frequent_instability"]["iterations"] = [
            r.get("iteration") for r in results_data if r.get("unstable")
        ][:10]

    n_solver_bad = 0
    for r in results_data:
        diag = r.get("solver_diagnostics") or {}
        n_solver_bad += diag.get("solved_inaccurate", 0) + diag.get("other", 0)
    if n_solver_bad > 0 and n_total > 0 and n_solver_bad / max(n_total, 1) > 0.2:
        findings["solver_struggles"]["count"] = n_solver_bad

    return dict(findings)


def _format_issues_block(findings: Dict[str, Dict[str, Any]]) -> str:
    """Shared by generate_diagnostics_report and chat_about_issues so the
    LLM sees the exact same grounding text for the detected issues in both
    the one-shot report and any follow-up chat about it."""
    if not findings:
        return "(none detected by the deterministic scan)"
    lines = []
    for cat_key, info in findings.items():
        cat = _ERROR_CATEGORIES.get(cat_key, {"title": cat_key})
        lines.append(
            f"- [{cat_key}] {cat['title']}: {info['count']} occurrence(s)"
            + (f", iterations {info['iterations']}" if info.get("iterations") else "")
            + (f"\n  Example message(s): {info['examples']}" if info.get("examples") else "")
        )
    return "\n".join(lines)


def _fallback_report(findings: Dict[str, Dict[str, Any]]) -> DiagnosticsReport:
    """Used if the LLM call fails -- the deterministic fallback_recommendation
    text for each category, so something useful is still shown."""
    recs = []
    for cat_key, info in findings.items():
        cat = _ERROR_CATEGORIES.get(cat_key, {})
        recs.append(_CategoryRecommendation(
            category=cat_key,
            explanation=f"Detected {info['count']} occurrence(s) of: {cat.get('title', cat_key)}.",
            recommendation=cat.get("fallback_recommendation", "Check the logs for more detail."),
            contribution_estimate="(Automatic estimate unavailable -- LLM call failed.)",
        ))
    return DiagnosticsReport(recommendations=recs)


def generate_diagnostics_report(
    findings: Dict[str, Dict[str, Any]],
    n_total_iterations: int,
    run_context: str = "",
    tracker: Optional[Any] = None,
) -> DiagnosticsReport:
    """Only call this if scan_for_issues() actually found something --
    makes one LLM call to turn the raw findings into grounded explanations
    and recommendations. Falls back to the deterministic
    fallback_recommendation text (still useful, just less specific) if the
    LLM call fails.

    ``run_context``: the full picture beyond the matched keywords -- the
    system's declared state/input bounds, the scenario/trajectory actually
    used, the per-iteration parameter/metric history, and every other
    agent's own reasoning log (see app.py's _build_diagnostics_context).
    Without this, a category like "solver_struggles" can only ever repeat
    its generic fallback_recommendation text ("try loosening bounds"); WITH
    it, the model can look at the actual declared bounds and the actual
    per-iteration Q/R/Np/Nc history and say something specific to this run.
    """
    if not findings:
        return DiagnosticsReport(recommendations=[])

    issues_block = _format_issues_block(findings)

    try:
        llm = get_llm().with_structured_output(DiagnosticsReport)
        prompt_text = _diagnostics_prompt.format(
            issues_block=issues_block, n_total=n_total_iterations,
            run_context=run_context or "(not provided)",
        )
        return invoke_with_retry(llm, prompt_text, max_retries=1, node_name="DiagnosticsAgent", tracker=tracker)
    except Exception as e:  # noqa: BLE001
        log.error("[DiagnosticsAgent] LLM call failed, using fallback recommendations: %s", e)
        return _fallback_report(findings)


def _format_report_block(report: Optional[DiagnosticsReport]) -> str:
    if report is None or not report.recommendations:
        return "(no report generated yet)"
    lines = [
        f"- [{rec.category}] {rec.explanation} Recommendation: {rec.recommendation} "
        f"(Contribution: {rec.contribution_estimate})"
        for rec in report.recommendations
    ]
    return "\n".join(lines)


def _format_history_block(conversation_history: List[Dict[str, str]]) -> str:
    if not conversation_history:
        return "(this is the first message)"
    return "\n".join(f"{turn['role'].upper()}: {turn['content']}" for turn in conversation_history)


class _ChatReply(BaseModel):
    reply: str = Field(description="The reply to the user's message, in plain conversational text "
                        "(no markdown code fences, no JSON) -- this is shown to them as-is.")


def chat_about_issues(
    user_message: str,
    findings: Dict[str, Dict[str, Any]],
    report: Optional[DiagnosticsReport],
    conversation_history: List[Dict[str, str]],
    run_error: Optional[str] = None,
    run_context: str = "",
    tracker: Optional[Any] = None,
) -> str:
    """Free-form follow-up chat about a run's detected issues -- lets the
    user ask clarifying questions, describe what they've already tried, or
    ask about the run's raw error directly, instead of only ever reading the
    one-shot report generate_diagnostics_report() produces.

    Uses .with_structured_output(_ChatReply), same as every other call in
    this codebase, even though the content itself is free text -- NOT a
    style choice: get_llm()'s underlying client is constructed with OpenAI's
    response_format=json_object forced on for any "gpt-" model (see
    labcd_agents.providers._build_openai's json_mode default), which makes a
    genuinely PLAIN .invoke(str) call 400 on OpenAI (or silently return
    something useless on other providers) -- structured output is the only
    call shape that's guaranteed compatible with every model this app can be
    pointed at.

    Raises on failure -- unlike the report/scan functions above, there is no
    sensible non-LLM fallback for an open-ended chat reply; the caller (the
    Streamlit UI) is expected to catch this and show the error inline in the
    chat thread rather than losing the user's message.

    ``conversation_history`` holds only the turns BEFORE this one -- it must
    not already include ``user_message``.

    ``run_context``: the full picture -- declared state/input bounds, the
    scenario/trajectory used, the per-iteration parameter/metric history,
    and every other agent's own reasoning log (see app.py's
    _build_diagnostics_context). This is what lets a question like "why did
    Q3 increase" get answered by quoting the Actor's own stated reason for
    that change, instead of a generic non-answer.
    """
    llm = get_llm().with_structured_output(_ChatReply)
    prompt_text = _diagnostics_chat_prompt.format(
        issues_block=_format_issues_block(findings),
        report_block=_format_report_block(report),
        run_error_block=(run_error[:2000] if run_error else "(none -- the run did not fail outright)"),
        history_block=_format_history_block(conversation_history),
        user_message=user_message,
        run_context=run_context or "(not provided)",
    )
    response = invoke_with_retry(llm, prompt_text, max_retries=1, node_name="DiagnosticsChat", tracker=tracker)
    return response.reply
