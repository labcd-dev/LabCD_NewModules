import matplotlib.pyplot as plt

from . import llm_factory
from backend_core.AgentAdaptive.tools import system_spec as system_spec_mod
from .prompt_loader import load_prompt

from backend_core.AgentAdaptive.controller.structure_build import (
    _build_structure_from_spec, _verify_structure, _validate_method)
from backend_core.AgentAdaptive.tools.reporter import _render_clarification_section
from backend_core.AgentAdaptive.tools.progress import _emit
from .agent_io import (
    _extract_json_payload, _sum_usage_from_messages, _per_turn_usage,
    _SyntheticMessage,
)

EXTRACTOR_SYSTEM_PROMPT = load_prompt("designer_agent_prompt.yaml")


def _coerce_method_payload(payload):
    if not isinstance(payload, dict):
        raise TypeError("the reply was not a JSON object")
    if "method" not in payload:
        raise KeyError("missing required field: method")
    method = str(payload["method"])
    if method not in ("smc", "backstepping"):
        raise ValueError("method must be \"smc\" or \"backstepping\", got %r" % method)
    return {
        "method": method,
        "reasoning": str(payload.get("reasoning") or ""),
        "notes_limitations": (str(payload["notes_limitations"])
                               if payload.get("notes_limitations") else None),
    }


class _JSONExtractorAgent:
    def __init__(self, session, spec, on_event=None):
        self.session = session
        self.spec = spec
        self.on_event = on_event
        self.llm = llm_factory.build_llm("design")

    def _record_failure(self, report, replies):
        self.session["last"] = {"ok": False, "report": report, "args": {}, "why": ""}
        return {"messages": replies}

    def _record_success(self, args_for_report, replies):
        method_label = "SMC" if args_for_report["method"] == "smc" else "Backstepping"
        print("\n=== method decided ===")
        print("method: %s (%s)" % (method_label,
              "square system" if args_for_report["method"] == "smc" else "strict-feedback chain"))
        print("reasoning: %s" % args_for_report["reasoning"])
        print("=======================\n")
        self.session["last"] = {
            "ok": True,
            "report": "(method decided; design and simulation pending build)",
            "args": args_for_report, "why": args_for_report.get("reasoning", ""),
        }
        return {"messages": replies}

    def _announce(self, method):
        _emit(self.on_event, kind="note", stage="design",
              text="System type detected: %s (%s)"
              % ("SMC" if method == "smc" else "Backstepping",
                 "square system" if method == "smc" else "strict-feedback chain"))

    @staticmethod
    def _merge(method_payload, structure, limitations):
        notes = method_payload.get("notes_limitations")
        if limitations:
            extra = "\n".join("- " + line for line in limitations)
            notes = (notes + "\n" + extra) if notes else extra
        out = dict(structure)
        out["method"] = method_payload["method"]
        out["reasoning"] = method_payload["reasoning"]
        out["notes_limitations"] = notes
        return out

    def invoke(self, input_dict):
        user_content = input_dict["messages"][-1]["content"]
        messages = [{"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}]
        replies = []

        # built and checked before the model's even called: it doesn't depend on the reply.
        # A failure here is a bad spec upstream, not something a repair turn could fix.
        try:
            structure, limitations = _build_structure_from_spec(self.spec)
            structure = _verify_structure(structure)
        except Exception as e:
            plt.close("all")  # avoid leaving a half-drawn figure hanging around
            return self._record_failure(
                "EXTRACTION FAILED: the confirmed system could not be "
                "parsed (%s: %s). This points at a problem in the plant "
                "spec itself, upstream of the Designer Agent."
                % (type(e).__name__, e), replies)

        resp = self.llm.invoke(messages)
        replies.append(resp)
        payload, err = self._coerce(resp.content)
        if err is None:
            try:
                _validate_method(payload["method"], structure)
            except ValueError as e:
                err = e

        if err is None:
            self._announce(payload["method"])
            return self._record_success(self._merge(payload, structure, limitations), replies)

        # one repair turn, same conversation
        repair_prompt = (
            "Your previous reply could not be used: %s: %s\n\n"
            "Reply again with either the complete corrected JSON object "
            "(every required field), or exactly "
            "{\"status\": \"failed\", \"explanation\": \"<3-5 sentence "
            "plain-language explanation, no raw errors or tracebacks>\"} "
            "if this cannot be fixed."
            % (type(err).__name__, err)
        )
        messages = messages + [
            {"role": "assistant", "content": resp.content},
            {"role": "user", "content": repair_prompt},
        ]
        resp2 = self.llm.invoke(messages)
        replies.append(resp2)
        payload2, parse_err2 = _extract_json_payload(resp2.content)

        if payload2 is not None and isinstance(payload2, dict) and payload2.get("status") == "failed":
            explanation = str(payload2.get("explanation") or "").strip()
            return self._record_failure(
                explanation or "The method could not be decided, and no "
                "explanation was given.", replies)

        if payload2 is None:
            return self._record_failure(
                "EXTRACTION FAILED. The model's reply could not be read as "
                "JSON even after one retry (%s: %s)."
                % (type(parse_err2).__name__, parse_err2), replies)

        args2, err2 = self._coerce(resp2.content, payload=payload2)
        if err2 is None:
            try:
                _validate_method(args2["method"], structure)
            except ValueError as e:
                err2 = e
        if err2 is not None:
            return self._record_failure(
                "EXTRACTION FAILED; the retried reply still did not carry "
                "a valid method (%s: %s)." % (type(err2).__name__, err2), replies)

        self._announce(args2["method"])
        return self._record_success(self._merge(args2, structure, limitations), replies)

    @staticmethod
    def _coerce(raw_text, payload=None):
        if payload is None:
            payload, err = _extract_json_payload(raw_text)
            if payload is None:
                return {}, err
        try:
            return _coerce_method_payload(payload), None
        except Exception as e:
            return {}, e


def build_extractor_agent(spec, on_event=None):
    session = {"last": None}
    return _JSONExtractorAgent(session, spec, on_event=on_event), session


def run_extraction(spec, on_event=None, clarification_record=None):
    _emit(on_event, kind="stage_start", stage="design", deferred=True)
    agent, session = build_extractor_agent(spec, on_event=on_event)
    description = system_spec_mod.designer_view_to_text(spec)
    result = agent.invoke({"messages": [{"role": "user", "content": description}]})
    # copied instead of aliased, since an event consumer must not be able to reach
    # into the live extraction session by mutating what it's handed
    _emit(on_event, kind="stage_done", stage="design",
          ok=bool(session["last"] and session["last"]["ok"]), deferred=True,
          args=dict((session["last"] or {}).get("args") or {}))

    agent_usage = _sum_usage_from_messages(result["messages"])
    agent_turns = list(_per_turn_usage(result["messages"]))
    timeline = [{
        "actor": "design", "round": 0, "label": "Method selection",
        "tokens": agent_usage["total_tokens"], "input_tokens": agent_usage["input_tokens"],
        "output_tokens": agent_usage["output_tokens"],
        "detail": "ok" if (session["last"] and session["last"]["ok"]) else "failed",
    }]

    last = session["last"]
    clarification_section_text = _render_clarification_section(clarification_record)
    if last is not None:
        final_text = last["report"] + clarification_section_text
        result = dict(result)
        messages = list(result["messages"])
        if messages:
            messages[-1] = _SyntheticMessage(final_text)
        else:
            # early structure-build failure means the LLM was never even called,
            # so there's nothing to overwrite yet -- just add the message.
            messages.append(_SyntheticMessage(final_text))
        result["messages"] = messages

    usage = {"agent": agent_usage, "total": agent_usage,
              "agent_turns": agent_turns, "timeline": timeline}
    return result, usage, last
