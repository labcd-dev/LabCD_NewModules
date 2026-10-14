"""Seeded mock Designer that delegates design and simulation to Python."""

from __future__ import annotations

import sympy as sp

from backend_core.AgentAdaptive.agents.agent_io import (
    _SyntheticMessage,
    _extract_usage,
    _per_turn_usage,
)
from backend_core.AgentAdaptive.controller.structure_build import (
    _build_structure_from_spec,
    _validate_method,
    _verify_structure,
)
from backend_core.AgentAdaptive.tools.progress import _emit
from backend_core.AgentAdaptive.tools.reporter import _render_clarification_section

from .mock_random import MockAgent, MockSession, current_session


class MockDesigner(MockAgent):
    def payload(self) -> dict[str, str]:
        dynamics = self.context["spec"]["dynamics"]
        names = (
            dynamics["states"]
            + dynamics["inputs"]
            + list(dynamics.get("parameters", {}))
        )
        symbols = {name: sp.Symbol(name) for name in names}
        states = [symbols[name] for name in dynamics["states"]]
        inputs = {symbols[name] for name in dynamics["inputs"]}
        equations = [
            sp.sympify(equation, locals=symbols)
            for equation in dynamics["state_equations"]
        ]

        is_chain = (
            len(inputs) == 1
            and dynamics["outputs"] == [dynamics["states"][0]]
        )
        for index, equation in enumerate(equations[:-1]):
            forbidden = set(states[index + 2 :]) | inputs
            gain = sp.diff(equation, states[index + 1])
            is_chain = (
                is_chain
                and not (equation.free_symbols & forbidden)
                and gain != 0
                and sp.diff(gain, states[index + 1]) == 0
            )

        if is_chain:
            method = "backstepping"
        elif len(inputs) == len(dynamics["outputs"]):
            method = "smc"
        else:
            raise ValueError(
                "Mock Designer requires a strict-feedback SISO or square system."
            )
        return {
            "method": method,
            "reasoning": (
                f"MOCK Designer selected {method} from the plant structure. "
                "Python performs the real controller design and simulation."
            ),
            "notes_limitations": (
                "Mock agent output for benchmark testing, not an LLM recommendation."
            ),
        }


def run_extraction(spec, on_event=None, clarification_record=None):
    _emit(on_event, kind="stage_start", stage="design", deferred=True)
    session = current_session() or MockSession(42)
    response = MockDesigner(session, "design", {"spec": spec}).invoke([])
    payload = session.messages[-1]["response"]
    structure, limitations = _build_structure_from_spec(spec)
    structure = _verify_structure(structure)
    _validate_method(payload["method"], structure)
    arguments = dict(structure)
    arguments.update(payload)
    if limitations:
        arguments["notes_limitations"] += "\n" + "\n".join(limitations)

    last = {
        "ok": True,
        "report": "(mock method selected; numerical simulation pending)",
        "args": arguments,
        "why": payload["reasoning"],
    }
    _emit(
        on_event,
        kind="stage_done",
        stage="design",
        ok=True,
        deferred=True,
        args=dict(arguments),
    )
    tokens = _extract_usage(response)
    usage = {
        "agent": tokens,
        "total": tokens,
        "agent_turns": list(_per_turn_usage([response])),
        "timeline": [
            {
                "actor": "design",
                "round": 0,
                "label": "Mock method selection",
                "tokens": tokens["total_tokens"],
                "input_tokens": tokens["input_tokens"],
                "output_tokens": tokens["output_tokens"],
                "detail": "ok",
            }
        ],
    }
    result = {
        "messages": [
            _SyntheticMessage(
                last["report"]
                + _render_clarification_section(clarification_record)
            )
        ]
    }
    return result, usage, last
