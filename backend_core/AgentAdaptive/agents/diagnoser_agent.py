# the two LLM calls for the Diagnoser -- the initial report and the
# follow-up chat. evidence gathering lives in diagnostics_evidence.py, not
# here. if a call fails we just say so, no faking a diagnosis

import json

from . import llm_factory
from .prompt_loader import load_prompt
from .agent_io import _extract_json_payload, _extract_usage, _empty_usage

DIAGNOSER_SYSTEM_PROMPT = load_prompt("diagnoser_agent_prompt.yaml")
DIAGNOSER_CHAT_SYSTEM_PROMPT = load_prompt("diagnoser_chat_prompt.yaml")

# the only four things a user can actually go change -- the UI uses this
# to draw a jump-to-field button next to each suggestion
_SUGGESTION_LEVERS = ("system", "reference", "initial_condition", "step_time")


def _error_report(detail):
    # just one message here, not a list of bullets -- frontend handles both fine
    return {
        "headline": "Diagnosis unavailable",
        "explanation": "The diagnoser could not produce an explanation for this run: %s" % detail,
        "suggestions": [],
        "chart_series": [],
        "error": detail,
    }


def _coerce_explanation(value):
    # model's supposed to send a list of bullets, but if it sends a plain
    # string just wrap it -- no reason to toss a good report over that
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()][:5]
    return []


def _coerce_report(payload):
    if not isinstance(payload, dict):
        raise TypeError("the reply was not a JSON object")
    headline = str(payload.get("headline") or "").strip()
    explanation = _coerce_explanation(payload.get("explanation"))
    if not headline or not explanation:
        raise KeyError("missing required field(s): headline/explanation")
    suggestions = []
    for item in (payload.get("suggestions") or [])[:3]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        lever = item.get("lever")
        # bad or missing lever just means this one suggestion won't get a
        # button -- doesn't mean we throw out the whole report
        lever = lever if lever in _SUGGESTION_LEVERS else None
        target = item.get("target")
        target = target.strip() if isinstance(target, str) and target.strip() else None
        options = []
        for opt in (item.get("options") or [])[:3]:
            if not isinstance(opt, dict):
                continue
            value = opt.get("value")
            if value is None or not str(value).strip():
                continue
            value = str(value).strip()
            options.append({"label": str(opt.get("label") or value).strip(), "value": value})
        suggestions.append({"title": title, "detail": str(item.get("detail") or "").strip(),
                            "lever": lever, "target": target, "options": options})
    chart_series = [str(k) for k in (payload.get("chart_series") or []) if isinstance(k, str)][:2]
    return {"headline": headline, "explanation": explanation,
            "suggestions": suggestions, "chart_series": chart_series}


def diagnose(evidence):
    # the one call that actually diagnoses a run. always returns
    # (report, usage) in the same shape whether it worked or not -- a
    # failed call just gets an "error" key, nothing special to handle
    user_content = json.dumps(evidence, indent=2, ensure_ascii=False)
    try:
        llm = llm_factory.build_llm("diagnoser")
        resp = llm.invoke([
            {"role": "system", "content": DIAGNOSER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ])
        usage = _extract_usage(resp)
        payload, err = _extract_json_payload(resp.content)
        if payload is None:
            return (_error_report("the reply could not be read as JSON (%s: %s)"
                                  % (type(err).__name__, err)), usage)
        try:
            return _coerce_report(payload), usage
        except Exception as e:
            return _error_report("%s: %s" % (type(e).__name__, e)), usage
    except Exception as e:
        return _error_report("%s: %s" % (type(e).__name__, e)), _empty_usage()


def answer_followup(evidence, report, question, history=None):
    # follow-up chat, only fires when someone actually types a question.
    # same grounding rule as diagnose() -- stick to the given evidence.
    # history is just this chat's prior turns, not the diagnosis itself
    history_lines = "\n".join(
        "%s: %s" % ("You" if h.get("role") == "assistant" else "User", h.get("text", ""))
        for h in (history or [])
    )
    user_content = (
        "Evidence for this run:\n%s\n\n"
        "Your diagnosis report so far:\n%s\n\n"
        "%s"
        "New question: %s"
        % (json.dumps(evidence, indent=2, ensure_ascii=False),
           json.dumps(report, indent=2, ensure_ascii=False),
           ("Conversation so far:\n%s\n\n" % history_lines) if history_lines else "",
           question)
    )
    try:
        llm = llm_factory.build_llm("diagnoser")
        resp = llm.invoke([
            {"role": "system", "content": DIAGNOSER_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ])
        usage = _extract_usage(resp)
        payload, err = _extract_json_payload(resp.content)
        if not isinstance(payload, dict) or not str(payload.get("reply") or "").strip():
            detail = ("%s: %s" % (type(err).__name__, err)) if err else "empty reply"
            return "I couldn't produce an answer to that (%s)." % detail, usage
        return str(payload["reply"]).strip(), usage
    except Exception as e:
        return "I couldn't produce an answer to that (%s: %s)." % (type(e).__name__, e), _empty_usage()
