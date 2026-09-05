"""
Run with: pytest backend_core/AgentMPC/tests/test_diagnostics_chat.py -v

Covers the new chat_about_issues() follow-up chat added to
agents/diagnostics_agent.py, plus the shared formatting helpers it and
generate_diagnostics_report() both now use (_format_issues_block was pulled
out of generate_diagnostics_report to avoid duplicating that formatting
logic between the one-shot report and the chat).

chat_about_issues uses .with_structured_output(_ChatReply) -- NOT a plain
get_llm().invoke(str) -- because get_llm()'s underlying client forces
OpenAI's response_format=json_object on for "gpt-" models (see
labcd_agents.providers._build_openai), which makes a genuinely free-text
.invoke(str) call 400 on OpenAI. This was caught live: the first real-browser
test of the chat feature returned the raw request echoed back as JSON
instead of an answer, exactly matching that failure mode.
"""

from unittest.mock import MagicMock

import pytest

from backend_core.AgentMPC.agents import diagnostics_agent as da


def _stub_llm(monkeypatch, reply_text):
    """Mirrors the .with_structured_output(...).invoke(...) pattern every
    other agent in this codebase uses (see test_agent_reasoning_history.py)."""
    structured = MagicMock()
    structured.invoke.return_value = da._ChatReply(reply=reply_text)
    client = MagicMock()
    client.with_structured_output.return_value = structured
    monkeypatch.setattr(da, "get_llm", lambda: client)
    return structured


FINDINGS = {
    "rate_limit": {"count": 3, "examples": ["429 too many requests"], "iterations": [2, 4, 6]},
}


def test_format_issues_block_empty():
    assert "none detected" in da._format_issues_block({})


def test_format_issues_block_lists_each_category():
    block = da._format_issues_block(FINDINGS)
    assert "rate_limit" in block
    assert "3 occurrence(s)" in block
    assert "429 too many requests" in block


def test_format_report_block_none():
    assert "no report generated yet" in da._format_report_block(None)


def test_format_report_block_lists_recommendations():
    report = da.DiagnosticsReport(recommendations=[
        da._CategoryRecommendation(
            category="rate_limit", explanation="Hit the API limit 3 times.",
            recommendation="Wait before retrying.", contribution_estimate="Affected 3 of 10 iterations.",
        ),
    ])
    block = da._format_report_block(report)
    assert "rate_limit" in block and "Wait before retrying." in block


def test_format_history_block_empty():
    assert "first message" in da._format_history_block([])


def test_format_history_block_renders_each_turn():
    history = [{"role": "user", "content": "why did this fail?"}, {"role": "assistant", "content": "rate limit."}]
    block = da._format_history_block(history)
    assert "USER: why did this fail?" in block
    assert "ASSISTANT: rate limit." in block


def test_chat_about_issues_returns_the_llms_reply(monkeypatch):
    _stub_llm(monkeypatch, "Try waiting a minute between runs.")
    reply = da.chat_about_issues(
        "why does this keep happening?", FINDINGS, report=None, conversation_history=[],
    )
    assert reply == "Try waiting a minute between runs."


def test_chat_about_issues_grounds_the_prompt_in_all_available_context(monkeypatch):
    """The actual prompt text sent to the LLM must include the findings, the
    prior report, the raw run error, AND the conversation history -- not just
    the new message -- otherwise the chat can't ground its answer in what
    the user is actually looking at."""
    structured = _stub_llm(monkeypatch, "...")
    report = da.DiagnosticsReport(recommendations=[
        da._CategoryRecommendation(
            category="rate_limit", explanation="explained here", recommendation="recommended here",
            contribution_estimate="estimated here",
        ),
    ])
    history = [{"role": "user", "content": "first question"}, {"role": "assistant", "content": "first answer"}]
    da.chat_about_issues(
        "second question", FINDINGS, report=report, conversation_history=history,
        run_error="Traceback: ValueError boom",
    )
    prompt_sent = structured.invoke.call_args[0][0]
    assert "rate_limit" in prompt_sent
    assert "recommended here" in prompt_sent
    assert "Traceback: ValueError boom" in prompt_sent
    assert "first question" in prompt_sent and "first answer" in prompt_sent
    assert "second question" in prompt_sent


def test_chat_about_issues_handles_no_run_error(monkeypatch):
    structured = _stub_llm(monkeypatch, "...")
    da.chat_about_issues("a question", {}, report=None, conversation_history=[], run_error=None)
    prompt_sent = structured.invoke.call_args[0][0]
    assert "did not fail outright" in prompt_sent


def test_chat_about_issues_includes_the_run_context(monkeypatch):
    """The actual feature this was built for: "why did Q3 increase" can only
    be answered from the Actor's own reasoning text -- which lives in
    run_context (app.py's _build_diagnostics_context), not in findings/report/
    run_error. Regression for the report that this info wasn't reaching the
    LLM at all before run_context was added."""
    structured = _stub_llm(monkeypatch, "...")
    run_context = (
        "Per-iteration history:\n  Iter 3 [explore] SETTLED: Q=[10.0, 2.0, 90.0]\n"
        "Agent reasoning log:\n  [10:00:01] [Actor] strategy=explore\nQ=[10.0, 2.0, 90.0]\n\n"
        "State 3 had a much higher per-state MSE than the others, so Q for that state was raised."
    )
    da.chat_about_issues("why did Q3 increase?", {}, report=None, conversation_history=[], run_context=run_context)
    prompt_sent = structured.invoke.call_args[0][0]
    assert "State 3 had a much higher per-state MSE" in prompt_sent


def test_chat_about_issues_defaults_run_context_when_not_given(monkeypatch):
    structured = _stub_llm(monkeypatch, "...")
    da.chat_about_issues("a question", {}, report=None, conversation_history=[])
    prompt_sent = structured.invoke.call_args[0][0]
    assert "(not provided)" in prompt_sent


def test_chat_about_issues_uses_structured_output_not_plain_invoke(monkeypatch):
    """Regression for the exact bug this was caught by: a plain get_llm()
    (no .with_structured_output) hits response_format=json_object forced on
    for "gpt-" models, which either 400s or returns the request echoed back
    as JSON instead of a real answer."""
    client = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = da._ChatReply(reply="a real answer")
    client.with_structured_output.return_value = structured
    monkeypatch.setattr(da, "get_llm", lambda: client)

    da.chat_about_issues("a question", {}, report=None, conversation_history=[])
    client.with_structured_output.assert_called_once_with(da._ChatReply)
    client.invoke.assert_not_called()


def test_chat_about_issues_raises_on_llm_failure(monkeypatch):
    """No non-LLM fallback for open-ended chat (unlike the report) -- the
    caller (Streamlit UI) is expected to catch this itself."""
    structured = MagicMock()
    structured.invoke.side_effect = RuntimeError("provider down")
    client = MagicMock()
    client.with_structured_output.return_value = structured
    monkeypatch.setattr(da, "get_llm", lambda: client)
    with pytest.raises(RuntimeError):
        da.chat_about_issues("a question", {}, report=None, conversation_history=[])


def test_generate_diagnostics_report_still_uses_the_shared_issues_block(monkeypatch):
    """Regression: generate_diagnostics_report was refactored to call
    _format_issues_block() instead of building its own inline copy -- must
    still produce the same grounded prompt content as before."""
    structured = MagicMock()
    structured.invoke.return_value = da.DiagnosticsReport(recommendations=[])
    client = MagicMock()
    client.with_structured_output.return_value = structured
    monkeypatch.setattr(da, "get_llm", lambda: client)

    da.generate_diagnostics_report(FINDINGS, n_total_iterations=10)
    prompt_sent = structured.invoke.call_args[0][0]
    assert "rate_limit" in prompt_sent
    assert "3 occurrence(s)" in prompt_sent


def test_generate_diagnostics_report_includes_the_run_context(monkeypatch):
    """Same reasoning as the chat test above: a "solver_struggles" or
    "frequent_instability" recommendation can only name the actual declared
    bounds / actual scenario if run_context reaches the prompt."""
    structured = MagicMock()
    structured.invoke.return_value = da.DiagnosticsReport(recommendations=[])
    client = MagicMock()
    client.with_structured_output.return_value = structured
    monkeypatch.setattr(da, "get_llm", lambda: client)

    da.generate_diagnostics_report(
        FINDINGS, n_total_iterations=10,
        run_context="Declared input bounds: F: [-5, 5]\nStop reason: hit the Max Iterations cap",
    )
    prompt_sent = structured.invoke.call_args[0][0]
    assert "Declared input bounds: F: [-5, 5]" in prompt_sent
    assert "Max Iterations cap" in prompt_sent


def test_generate_diagnostics_report_defaults_run_context_when_not_given(monkeypatch):
    structured = MagicMock()
    structured.invoke.return_value = da.DiagnosticsReport(recommendations=[])
    client = MagicMock()
    client.with_structured_output.return_value = structured
    monkeypatch.setattr(da, "get_llm", lambda: client)

    da.generate_diagnostics_report(FINDINGS, n_total_iterations=10)
    prompt_sent = structured.invoke.call_args[0][0]
    assert "(not provided)" in prompt_sent
