"""Seeded mock Reporter for benchmark-only runs."""

from __future__ import annotations

from backend_core.AgentAdaptive.agents.agent_io import _extract_usage

from .mock_random import MockAgent, MockSession, current_session


class MockReporter(MockAgent):
    def payload(self) -> dict[str, str]:
        return {
            "abstract": (
                f"MOCK Report Writer. {self.context['system_name']} uses "
                f"{self.context['method_label']}. {self.context['outcome_text']}. "
                "The outcome comes from numerical simulation; agent responses "
                "and token usage are simulated."
            )
        }


def write_abstract(system_name, method_label, agents_used, outcome_text, why=""):
    del agents_used, why
    session = current_session() or MockSession(42)
    response = MockReporter(
        session,
        "reporter",
        {
            "system_name": system_name,
            "method_label": method_label,
            "outcome_text": outcome_text,
        },
    ).invoke([])
    return session.messages[-1]["response"]["abstract"], _extract_usage(response)
