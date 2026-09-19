"""Random seeds and synthetic usage shared only by the mock Adaptive agents."""

import copy
import json
import random
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace


MOCK_COST_USD_PER_TOKEN = 0.001  # Synthetic test price, never an API charge.
_SESSION = ContextVar("adaptive_mock_session", default=None)
_ROLE_OFFSETS = {"design": 101, "tuner": 211, "reporter": 307}


def current_session():
    return _SESSION.get()


@contextmanager
def mock_session(seed=42):
    session = MockSession(seed)
    token = _SESSION.set(session)
    try:
        yield session
    finally:
        _SESSION.reset(token)


class MockSession:
    def __init__(self, seed):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("mock_seed must be an integer")
        self.seed = seed
        self.messages = []
        self.rngs = {role: random.Random(seed + offset)
                     for role, offset in _ROLE_OFFSETS.items()}

    def model_name(self, role):
        return "mock-adaptive-" + role

class _MockAgent:
    def __init__(self, session, role, context):
        self.session, self.role, self.context = session, role, context
        self.rng = session.rngs[role]

    def invoke(self, messages):
        payload = self.payload()
        # These are test data, not token counts measured by a tokenizer.
        input_tokens = self.rng.randint(400, 1000)
        output_tokens = self.rng.randint(80, 220)
        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                 "total_tokens": input_tokens + output_tokens,
                 "input_token_details": {"cache_read": 0}}
        self.session.messages.append({
            "agent": self.role, "call": 1 + sum(
                m["agent"] == self.role for m in self.session.messages),
            "model": self.session.model_name(self.role),
            "response": copy.deepcopy(payload), "usage": copy.deepcopy(usage),
        })
        return SimpleNamespace(content=json.dumps(payload), usage_metadata=usage)


def resolved_models():
    return {role: "mock-adaptive-" + name for role, name in (
        ("agent", "design"), ("tuner", "tuner"), ("reporter", "reporter"),
        ("clarifier", "clarifier"),
    )}
