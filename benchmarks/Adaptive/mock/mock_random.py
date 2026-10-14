"""Seeded randomness and synthetic usage shared by the mock agents."""

from __future__ import annotations

import copy
import json
import random
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, Iterator


MOCK_COST_USD_PER_TOKEN = 0.001
_SESSION: ContextVar[MockSession | None] = ContextVar(
    "adaptive_benchmark_mock_session", default=None
)
_ROLE_OFFSETS = {"design": 101, "tuner": 211, "reporter": 307}


def current_session() -> MockSession | None:
    return _SESSION.get()


@contextmanager
def mock_session(seed: int = 42) -> Iterator[MockSession]:
    session = MockSession(seed)
    token = _SESSION.set(session)
    try:
        yield session
    finally:
        _SESSION.reset(token)


class MockSession:
    def __init__(self, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("mock_seed must be an integer")
        self.seed = seed
        self.messages: list[dict[str, Any]] = []
        self.rngs = {
            role: random.Random(seed + offset)
            for role, offset in _ROLE_OFFSETS.items()
        }

    @staticmethod
    def model_name(role: str) -> str:
        return f"mock-adaptive-{role}"


class MockAgent:
    def __init__(self, session: MockSession, role: str, context: dict[str, Any]):
        self.session = session
        self.role = role
        self.context = context
        self.rng = session.rngs[role]

    def payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def invoke(self, messages: list[Any]) -> SimpleNamespace:
        del messages
        payload = self.payload()
        input_tokens = self.rng.randint(400, 1000)
        output_tokens = self.rng.randint(80, 220)
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": 0},
        }
        self.session.messages.append(
            {
                "agent": self.role,
                "call": 1
                + sum(message["agent"] == self.role for message in self.session.messages),
                "model": self.session.model_name(self.role),
                "response": copy.deepcopy(payload),
                "usage": copy.deepcopy(usage),
            }
        )
        return SimpleNamespace(content=json.dumps(payload), usage_metadata=usage)


def resolved_models() -> dict[str, str]:
    return {
        "agent": "mock-adaptive-design",
        "tuner": "mock-adaptive-tuner",
        "clarifier": "mock-adaptive-clarifier",
        "reporter": "mock-adaptive-reporter",
    }
