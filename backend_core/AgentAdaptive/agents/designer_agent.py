"""Mock Designer for Adaptive tests. The real agent is designer_agent_main.py."""

from backend_core.AgentAdaptive.controller.structure_build import (
    _build_structure_from_spec, _verify_structure, _validate_method,
)
from backend_core.AgentAdaptive.tools.progress import _emit
from backend_core.AgentAdaptive.tools.reporter import _render_clarification_section
from .agent_io import _SyntheticMessage, _extract_usage, _per_turn_usage
from .mock_random import _MockAgent, MockSession, current_session


class MockDesigner(_MockAgent):
    def payload(self):
        import sympy as sp

        dyn = self.context["spec"]["dynamics"]
        names = dyn["states"] + dyn["inputs"] + list(dyn.get("parameters", {}))
        symbols = {name: sp.Symbol(name) for name in names}
        states = [symbols[name] for name in dyn["states"]]
        inputs = {symbols[name] for name in dyn["inputs"]}
        equations = [sp.sympify(eq, locals=symbols) for eq in dyn["state_equations"]]
        chain = (len(inputs) == 1 and dyn["outputs"] == [dyn["states"][0]])
        for i, eq in enumerate(equations[:-1]):
            forbidden = set(states[i + 2:]) | inputs
            gain = sp.diff(eq, states[i + 1])
            chain = chain and not (eq.free_symbols & forbidden) and gain != 0
            chain = chain and sp.diff(gain, states[i + 1]) == 0
        if chain:
            method = "backstepping"
        elif len(inputs) == len(dyn["outputs"]):
            method = "smc"
        else:
            raise ValueError("Mock Designer requires a strict-feedback SISO or square system.")
        return {
            "method": method,
            "reasoning": "MOCK Designer: selected %s from the plant structure. "
                         "Python performs the real controller design and simulation." % method,
            "notes_limitations": "Mock agent output for testing, not an LLM recommendation.",
        }


def run_extraction(spec, on_event=None, clarification_record=None):
    _emit(on_event, kind="stage_start", stage="design", deferred=True)
    session = current_session() or MockSession(42)
    agent = MockDesigner(session, "design", {"spec": spec})
    response = agent.invoke([])
    payload = session.messages[-1]["response"]
    structure, limitations = _build_structure_from_spec(spec)
    structure = _verify_structure(structure)
    _validate_method(payload["method"], structure)
    args = dict(structure)
    args.update(payload)
    if limitations:
        args["notes_limitations"] += "\n" + "\n".join(limitations)
    last = {"ok": True, "report": "(mock method decided; real simulation pending)",
            "args": args, "why": payload["reasoning"]}
    _emit(on_event, kind="stage_done", stage="design", ok=True, deferred=True,
          args=dict(args))
    tokens = _extract_usage(response)
    usage = {
        "agent": tokens, "total": tokens,
        "agent_turns": list(_per_turn_usage([response])),
        "timeline": [{"actor": "design", "round": 0, "label": "Mock method selection",
                      "tokens": tokens["total_tokens"], "input_tokens": tokens["input_tokens"],
                      "output_tokens": tokens["output_tokens"], "detail": "ok"}],
    }
    result = {"messages": [_SyntheticMessage(
        last["report"] + _render_clarification_section(clarification_record))]}
    return result, usage, last
