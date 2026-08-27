from types import SimpleNamespace

from labcd_agents.tokens import TokenTracker, TokenUsage, extract_usage


def test_token_usage_addition_and_totals():
    a = TokenUsage(input_tokens=10, output_tokens=5)
    b = TokenUsage(input_tokens=3, output_tokens=2)
    total = a + b
    assert total.input_tokens == 13
    assert total.output_tokens == 7
    assert total.total_tokens == 20


def test_extract_usage_from_usage_metadata():
    response = SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 50})
    usage = extract_usage(response)
    assert usage == TokenUsage(100, 50)


def test_extract_usage_from_response_metadata_token_usage():
    response = SimpleNamespace(
        usage_metadata=None,
        response_metadata={"token_usage": {"prompt_tokens": 20, "completion_tokens": 10}},
    )
    usage = extract_usage(response)
    assert usage == TokenUsage(20, 10)


def test_extract_usage_from_response_metadata_usage():
    response = SimpleNamespace(
        usage_metadata=None,
        response_metadata={"usage": {"prompt_tokens": 7, "completion_tokens": 3}},
    )
    usage = extract_usage(response)
    assert usage == TokenUsage(7, 3)


def test_extract_usage_from_openai_responses_api():
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=42, output_tokens=8, total_tokens=50))
    usage = extract_usage(response)
    assert usage == TokenUsage(42, 8)


def test_extract_usage_missing_returns_zero():
    response = SimpleNamespace()
    usage = extract_usage(response)
    assert usage == TokenUsage(0, 0)


def test_extract_usage_from_dict():
    assert extract_usage({"input_tokens": 5, "output_tokens": 1}) == TokenUsage(5, 1)


def test_token_tracker_records_and_accumulates():
    tracker = TokenTracker()
    tracker.record(TokenUsage(10, 5), model="gpt-4o-mini", cost=0.01)
    tracker.record(TokenUsage(20, 10), model="gpt-4o-mini", cost=0.02)

    assert tracker.totals == TokenUsage(30, 15)
    assert tracker.total_cost == 0.03
    assert tracker.as_dict() == {"input_tokens": 30, "output_tokens": 15, "total_tokens": 45}


def test_token_tracker_as_state_update():
    tracker = TokenTracker()
    tracker.record(TokenUsage(10, 5), cost=0.5)
    update = tracker.as_state_update(existing_cost=1.0)
    assert update["token_usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert update["total_cost"] == 1.5


def test_token_tracker_reset():
    tracker = TokenTracker()
    tracker.record(TokenUsage(1, 1))
    tracker.reset()
    assert tracker.totals == TokenUsage(0, 0)
    assert tracker.history == []
