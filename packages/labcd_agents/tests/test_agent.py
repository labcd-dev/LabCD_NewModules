from types import SimpleNamespace

import pytest

from labcd_agents.agent import BaseAgent
from labcd_agents.exceptions import LLMInvocationError


class FakeClient:
    """Minimal stand-in for a LangChain chat model."""

    def __init__(self, responses=None, model="fake-model"):
        self.model = model
        self._responses = list(responses) if responses is not None else None
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self._responses is not None:
            item = self._responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return SimpleNamespace(
            content="hello world",
            usage_metadata={"input_tokens": 10, "output_tokens": 5},
        )

    def bind(self, **kwargs):
        return self


def make_agent(**kwargs):
    client = kwargs.pop("client", FakeClient())
    return BaseAgent(model="fake-model", client=client, **kwargs), client


def test_invoke_llm_success_tracks_usage_and_cost():
    agent, client = make_agent()
    agent.cost_calculator.register("fake-model", input_per_million=1_000_000, output_per_million=1_000_000)

    text, usage = agent.invoke_llm("system prompt", "user prompt")

    assert text == "hello world"
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 5
    assert "llm_time" in usage

    assert agent.total_usage.input_tokens == 10
    assert agent.total_usage.output_tokens == 5
    assert agent.total_cost == pytest.approx(10 * 1.0 + 5 * 1.0)

    # Messages sent as plain role/content dicts, matching legacy invoke_llm.
    sent = client.calls[0]
    assert sent[0] == {"role": "system", "content": "system prompt"}
    assert sent[1] == {"role": "user", "content": "user prompt"}


def test_invoke_llm_retries_then_succeeds():
    client = FakeClient(responses=[
        RuntimeError("transient failure"),
        SimpleNamespace(content="ok now", usage_metadata={"input_tokens": 1, "output_tokens": 1}),
    ])
    agent, _ = make_agent(client=client, max_retries=3)

    text, usage = agent.invoke_llm("sys", "usr")
    assert text == "ok now"
    assert len(client.calls) == 2


def test_invoke_llm_exhausts_retries_raises():
    client = FakeClient(responses=[RuntimeError("boom")] * 3)
    agent, _ = make_agent(client=client, max_retries=3)

    with pytest.raises(LLMInvocationError) as exc_info:
        agent.invoke_llm("sys", "usr")

    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_error, RuntimeError)
    assert len(client.calls) == 3


def test_monitor_hook_called_on_success():
    calls = []

    class FakeMonitor:
        def add_llm_response(self, agent_name, prompt, response):
            calls.append((agent_name, prompt, response))

    agent, _ = make_agent(monitor=FakeMonitor())
    agent.invoke_llm("system", "user")

    assert len(calls) == 1
    agent_name, prompt, response = calls[0]
    assert agent_name == "BaseAgent"
    assert response == "hello world"


def test_call_low_level_returns_text():
    agent, client = make_agent()
    result = agent.call("do the thing", system=True)
    assert result == "hello world"
    assert len(client.calls) == 1


def test_get_metrics_shape():
    agent, _ = make_agent()
    agent.invoke_llm("sys", "usr")
    metrics = agent.get_metrics()
    assert set(metrics.keys()) == {"token_usage", "total_cost"}
    assert metrics["token_usage"]["input_tokens"] == 10
