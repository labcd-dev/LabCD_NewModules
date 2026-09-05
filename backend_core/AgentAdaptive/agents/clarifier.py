import os
import re
import json
import time
import traceback

from backend_core.AgentAdaptive.tools import system_spec
from . import llm_factory

from .prompt_loader import load_prompt
CLARIFIER_SYSTEM_PROMPT = load_prompt("clarifier_agent_prompt.yaml")

DEFAULT_OPENAI_MODEL = llm_factory.DEFAULT_OPENAI_MODEL

_TERM_LIST_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"state": {"type": "string"}, "expr": {"type": "string"}},
        "required": ["state", "expr"],
        "additionalProperties": False,
    },
}

# grammar-constrained: the server can't emit an unterminated string or any
# other malformed JSON, unlike plain response_format=json_object.
_CLARIFIER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "clarifier_reply",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["continue", "complete"]},
                "reply": {"type": "string"},
                "dynamics": {
                    "type": ["object", "null"],
                    "properties": {
                        "uncertainty": _TERM_LIST_SCHEMA,
                        "disturbance": _TERM_LIST_SCHEMA,
                        "references": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"output": {"type": "string"}, "expr": {"type": "string"}},
                                "required": ["output", "expr"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["uncertainty", "disturbance", "references"],
                    "additionalProperties": False,
                },
            },
            "required": ["status", "reply", "dynamics"],
            "additionalProperties": False,
        },
    },
}

DEFAULT_T_END = system_spec.DEFAULT_SIM_TIME
DEFAULT_DT = system_spec.DEFAULT_SOLVER_STEP


def _empty_usage():
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "cached_input_tokens": 0}


def _extract_usage(message_or_response):
    # cached_input_tokens is a SUBSET of input_tokens (the part OpenAI served from
    # cache), not extra on top of it.
    usage = getattr(message_or_response, "usage_metadata", None) or {}
    details = usage.get("input_token_details") or {}
    return {
        "input_tokens": usage.get("input_tokens", 0) or 0,
        "output_tokens": usage.get("output_tokens", 0) or 0,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "cached_input_tokens": details.get("cache_read", 0) or 0,
    }


def _sum_usage(*usages):
    total = _empty_usage()
    for u in usages:
        for k in total:
            total[k] += u.get(k, 0) or 0
    return total


def _emit(on_event, **fields):
    if on_event is not None:
        fields.setdefault("ts", time.time())
        on_event(fields)


def _as_text(value, fallback=""):
    if value is None:
        return fallback
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    return text if text else fallback


# backslash that isn't part of a legal JSON escape. doubling just these lets a
# LaTeX-flavoured payload parse without touching real escapes
_STRAY_BACKSLASH_RE = re.compile(r'\\(?!["\\/bfnrtu])')

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _outermost_object(text):
    # scanning instead of regex, since braces can nest or sit inside strings:
    # greedy would eat trailing prose, lazy would stop too early
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


def parse_reply(text):
    raw = _as_text(text)
    if not raw:
        return None
    candidates = [raw]
    unfenced = _FENCE_RE.sub("", raw).strip()
    if unfenced and unfenced != raw:
        candidates.append(unfenced)
    inner = _outermost_object(unfenced or raw)
    if inner and inner not in candidates:
        candidates.append(inner)

    for candidate in candidates:
        for attempt in (candidate, _STRAY_BACKSLASH_RE.sub(r"\\\\", candidate)):
            try:
                out = json.loads(attempt)
            except ValueError:
                continue
            if isinstance(out, dict):
                return out
    return None


def _describe_bad_reply(text, response):
    meta = getattr(response, "response_metadata", None) or {}
    finish = meta.get("finish_reason") or meta.get("stop_reason") or "unknown"
    if finish == "length":
        return ("the reply was cut off before the JSON was complete. Raise "
                "OPENAI_MAX_TOKENS_CLARIFIER (currently %s)."
                % os.environ.get("OPENAI_MAX_TOKENS_CLARIFIER", "8000"))
    preview = " ".join(_as_text(text).split())[:300]
    if not preview:
        return "the model returned an empty reply (finish_reason=%s)." % finish
    return 'the reply was not JSON (finish_reason=%s): "%s"' % (finish, preview)


def build_clarifier_llm(json_mode=True):
    if json_mode:
        return llm_factory.build_llm(
            "clarifier", json_mode=True,
            model_kwargs={"response_format": _CLARIFIER_RESPONSE_FORMAT})
    return llm_factory.build_llm("clarifier", json_mode=False)


# matched on wording, not "retry on any exception": a rate limit or bad key would
# just get reproduced (and re-billed) by a blind retry
_JSON_MODE_REJECTION = ("response_format", "json_object", "json mode")


def length_limit_hit(exc):
    return (type(exc).__name__ == "LengthFinishReasonError"
            or "length limit was reached" in str(exc).lower())


def _invoke(messages):
    try:
        return build_clarifier_llm(json_mode=True).invoke(messages)
    except Exception as e:
        if length_limit_hit(e):
            raise
        if not any(word in str(e).lower() for word in _JSON_MODE_REJECTION):
            raise
        return build_clarifier_llm(json_mode=False).invoke(messages)


# TEMPORARY DEBUG DUMP, delete once the clarifier's behaving. logs every exchange
# (model, prompt, reply, traceback) to clarify_debug.log in the project root.
# Climb agents -> AgentAdaptive -> backend_core -> repo root for the debug log.
DEBUG_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "clarify_debug.log")


def _debug(title, **fields):
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as fh:
            fh.write("\n%s\n=== %s  %s ===\n"
                     % ("-" * 70, title, time.strftime("%Y-%m-%d %H:%M:%S")))
            for key, value in fields.items():
                text = value if isinstance(value, str) else repr(value)
                fh.write("--- %s ---\n%s\n" % (key, text))
    except Exception:
        pass


MAX_CLARIFY_TURNS = 12


def _plant_context(spec):
    view = system_spec.uncertainty_clarifier_view(spec)
    return ("Plant (fixed and read-only, except for normalizing "
            "`references`):\n%s"
            % json.dumps(view, indent=2, ensure_ascii=False))


def start_conversation(spec):
    return [{"role": "user", "content": _plant_context(spec)}]


def run_clarifier_turn(messages, on_event=None, round_num=1, force_finish=False,
                       _nudged=False):
    usage = _empty_usage()
    _emit(on_event, kind="stage_start", stage="clarify", round=round_num)

    call_messages = [{"role": "system", "content": CLARIFIER_SYSTEM_PROMPT}] + list(messages)
    if force_finish:
        call_messages = call_messages + [{"role": "user", "content":
            "Reply now with status \"complete\", using your best understanding "
            "of everything said so far. If existence itself is still genuinely "
            "unclear, treat it as no uncertainty/disturbance."}]

    _debug("REQUEST round %d" % round_num,
           model=(os.environ.get("OPENAI_MODEL_CLARIFIER")
                  or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)),
           messages=json.dumps(call_messages, indent=2, ensure_ascii=False))
    try:
        response = _invoke(call_messages)
        usage = _extract_usage(response)
        raw_content = _as_text(getattr(response, "content", ""))
        payload = parse_reply(raw_content)
        _debug("REPLY round %d" % round_num, content=raw_content,
               usage_metadata=getattr(response, "usage_metadata", None),
               parsed_ok=payload is not None)
    except Exception as e:
        _debug("EXCEPTION round %d" % round_num, traceback=traceback.format_exc())
        if length_limit_hit(e):
            detail = ("the model used its entire %s-token budget without "
                      "producing a reply. Raise OPENAI_MAX_TOKENS_CLARIFIER."
                      % os.environ.get("OPENAI_MAX_TOKENS_CLARIFIER", "8000"))
        else:
            detail = "%s: %s" % (type(e).__name__, e)
        _emit(on_event, kind="note", stage="clarify", round=round_num,
              text="Round %d failed: %s" % (round_num, detail))
        _emit(on_event, kind="stage_done", stage="clarify", round=round_num, n_questions=0)
        return "error", detail, None, usage, detail, messages

    if payload is None:
        detail = _describe_bad_reply(raw_content, response)
        _emit(on_event, kind="note", stage="clarify", round=round_num,
              text="Round %d: %s" % (round_num, detail))
        _emit(on_event, kind="stage_done", stage="clarify", round=round_num, n_questions=0)
        return "error", detail, None, usage, detail, messages

    status = _as_text(payload.get("status")).lower()
    reply = _as_text(payload.get("reply"), "(no reply)")
    updated_messages = list(messages) + [{"role": "assistant", "content": raw_content}]

    if status == "complete" and not force_finish and not _nudged and not any(
            m.get("role") == "assistant" for m in messages):
        # "complete" on the very first reply is always premature, since nothing's
        # been confirmed yet. one silent nudge-retry, capped by _nudged so it can't loop.
        _emit(on_event, kind="note", stage="clarify", round=round_num,
              text="Round %d answered complete before asking anything. Asking it to "
                   "check with the user first." % round_num)
        nudged_messages = updated_messages + [{"role": "user", "content":
            "You answered complete without asking me anything. The plant JSON not "
            "mentioning uncertainty is not the same as me confirming there is none, "
            "so ask me directly first."}]
        (retry_status, retry_reply, retry_dynamics, retry_usage, retry_error,
         retry_messages) = run_clarifier_turn(
            nudged_messages, on_event=on_event, round_num=round_num,
            force_finish=False, _nudged=True)
        # both calls happened for this one logical turn, so both costs get summed in
        return (retry_status, retry_reply, retry_dynamics,
               _sum_usage(usage, retry_usage), retry_error, retry_messages)

    if status == "complete":
        body = payload.get("dynamics") if isinstance(payload.get("dynamics"), dict) else {}
        normalized = system_spec.normalize_spec({"dynamics": body})["dynamics"]
        dynamics = {"uncertainty": normalized["uncertainty"],
                    "disturbance": normalized["disturbance"],
                    "references": normalized["references"]}
        _emit(on_event, kind="note", stage="clarify", round=round_num, text=reply)
        _emit(on_event, kind="stage_done", stage="clarify", round=round_num, n_questions=0)
        return "complete", reply, dynamics, usage, "", updated_messages

    _emit(on_event, kind="note", stage="clarify", round=round_num, text=reply)
    _emit(on_event, kind="stage_done", stage="clarify", round=round_num, n_questions=1)
    return "continue", reply, None, usage, "", updated_messages


def _clamp(value, low, high):
    return low if value < low else (high if value > high else value)


def sim_overrides_from_spec(spec):
    # clamps aren't cosmetic: bad t_end/dt hangs the UI, diverges the integrator, or
    # blows up memory. dt <= t_end/100 also catches combos legal alone but useless together.
    raw = system_spec.sim_overrides_from_spec(spec)
    t_end = _clamp(float(raw["t_end"]), 0.1, 600.0)
    dt = _clamp(float(raw["dt"]), 1e-6, 0.05)
    if dt > t_end / 100.0:
        dt = t_end / 100.0
    return {"t_end": t_end, "dt": dt}
