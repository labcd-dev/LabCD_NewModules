"""Mock Report Writer. The real agent is report_writer_main.py."""

from .agent_io import _extract_usage
from .mock_random import _MockAgent, MockSession, current_session


class MockReporter(_MockAgent):
    def payload(self):
        return {"abstract": "MOCK Report Writer. %s uses %s. %s. "
                            "The reported outcome comes from the numerical simulation; "
                            "agent responses and token usage are simulated."
                            % (self.context["system_name"], self.context["method_label"],
                               self.context["outcome_text"])}


def write_abstract(system_name, method_label, agents_used, outcome_text, why=""):
    session = current_session() or MockSession(42)
    response = MockReporter(session, "reporter", {
        "system_name": system_name, "method_label": method_label,
        "outcome_text": outcome_text,
    }).invoke([])
    return session.messages[-1]["response"]["abstract"], _extract_usage(response)
