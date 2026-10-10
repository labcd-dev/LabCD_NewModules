# the two LLM calls for the Diagnoser -- the initial report and the
# follow-up chat. evidence gathering lives in diagnostics_evidence.py, not
# here. if a call fails we just say so, no faking a diagnosis

import difflib
import json
import re

from . import llm_factory
from .prompt_loader import load_prompt
from .agent_io import _extract_json_payload, _extract_usage, _empty_usage
from backend_core.AgentAdaptive.tools import diagnosis_chain, failure_mode

DIAGNOSER_SYSTEM_PROMPT = load_prompt("diagnoser_agent_prompt.yaml")
DIAGNOSER_CHAT_SYSTEM_PROMPT = load_prompt("diagnoser_chat_prompt.yaml")

# the only four things a user can actually go change -- the UI uses this
# to draw a jump-to-field button next to each suggestion
_SUGGESTION_LEVERS = ("system", "reference", "initial_condition", "step_time")

_NOTE_CHARS = diagnosis_chain.NOTE_CHARS


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
    # the explanation may be empty: python adds the computed facts as bullets afterwards
    if not headline:
        raise KeyError("missing required field: headline")
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
    # short memo to its future self; capped here so a wordy reply can't bloat later evidence
    note = str(payload.get("note") or "").strip()[:_NOTE_CHARS] or None
    return {"headline": headline, "explanation": explanation,
            "suggestions": suggestions, "chart_series": chart_series, "note": note}


def _x0_at_a_spike(evidence):
    # True when the starting state is itself where a control spike happened (all states close)
    plant = evidence.get("plant") or {}
    x0, names = plant.get("x0"), plant.get("states") or []
    events = ((evidence.get("control_spikes") or {}).get("largest")) or []
    if not x0 or len(x0) != len(names):
        return False
    for ev in events:
        then = ev.get("states_then") or {}
        if all(isinstance(then.get(n), (int, float)) and isinstance(v, (int, float))
               and abs(v - then[n]) <= 0.1 * max(1.0, abs(then[n])) for n, v in zip(names, x0)):
            return True
    return False


_LIMIT_WORDS = ("limit", "actuator", "saturat")
_MAX_SUGGESTIONS = 3
# a model bullet about earlier runs: the python-written first bullet already covers history
_HISTORY_CLAIM = re.compile(
    r"\b(previous|earlier|prior|last)\s+(runs?|attempts?)\b|\bsince\s+(the\s+)?(last|previous)\b|\bpreviously\b"
    r"|\bcompared\s+(with|to)\s+(the\s+)?(last|previous|earlier)\b", re.I)


def _trouble_starts_at_once(evidence):
    # the first sign of trouble (instability, clipping, spike) in the first 1/20 of the run
    first = (evidence or {}).get("first_trouble_s")
    t_end = ((evidence or {}).get("simulation") or {}).get("t_end")
    return first is not None and bool(t_end) and first <= t_end / 20.0


def _numerical_trouble(evidence):
    # only a numerical failure (NaN/blow-up with no other explanation) can be cured by a smaller dt
    if (evidence.get("metrics") or {}).get("numerically_healthy") is False:
        return True
    explained = (evidence.get("control_spikes") or evidence.get("pre_run_warnings")
                 or (evidence.get("failure_mode") or {}).get("mode") in (
                     "limit_too_small", "limit_adds_to_failure", "bound_violated_by_limit",
                     "output_bound_violated"))
    return bool(evidence.get("failure_window")) and not explained


def _already_run(memory, lever, target):
    # every value of this lever that was run and failed (any target when the card names none)
    out = set()
    for (lv, tg), values in (memory or {}).get("tried", {}).items():
        if lv == lever and (target is None or tg is None or tg == target):
            out |= values
    return out


def _apply_memory(suggestions, memory):
    # drop values the user already ran, and cards offered twice and never followed
    kept = []
    for s in suggestions:
        if not s.get("lever"):
            kept.append(s)
            continue
        # a computed suggestion states a fact that still holds, so it is only dropped after being
        # shown three times without the user ever taking its value
        if s.get("computed") and (s["lever"], s.get("target")) in (memory or {}).get("tired", ()):
            continue
        if not s.get("computed") and any(
                lv == s["lever"] and (tg == s.get("target") or tg is None or s.get("target") is None)
                for lv, tg in (memory or {}).get("ignored", ())):
            continue
        if s.get("options"):
            seen = _already_run(memory, s["lever"], s.get("target"))
            options = [o for o in s["options"] if diagnosis_chain._norm_value(s["lever"], o["value"]) not in seen]
            if not options:
                continue
            s = dict(s, options=options)
        kept.append(s)
    return kept


_STOP = frozenset("the a an of to in is was were are be and or for by it its that this with as at on "
                  "from which than then so but not".split())


def _words(text):
    # content words cut to 5 letters ("limit" == "limited") and numbers
    return [w[:5] for w in re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower()) if w not in _STOP]


def _adds_nothing(sentence, facts, names):
    # a model sentence is dropped when it names nothing (no number, no state or input) or when
    # most of its words are already in the computed sentences
    has_number = bool(re.search(r"\d", sentence))
    has_name = any(re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(n), sentence) for n in names)
    if not (has_number or has_name):
        return True
    words = _words(sentence)
    known = {w for f in facts for w in _words(f)}
    return bool(words) and sum(w in known for w in words) / len(words) >= 0.6


def _reference_span(expr, t_end):
    # (min, max) of a reference expression over the run, or None if it cannot be evaluated
    try:
        import numpy as np
        import sympy as sp
        from backend_core.AgentAdaptive.controller import structure_build as sb
        f = sp.lambdify(sb._REF_T, sb._ref_from_expr(expr), "numpy")
        vals = np.asarray(f(np.linspace(0.0, float(t_end), 2000)), dtype=float)
        vals = np.broadcast_to(vals, (2000,))
        return float(vals.min()), float(vals.max())
    except Exception:  # noqa: BLE001 -- an option we cannot check is kept, not guessed at
        return None


def _inside_known_window(report, evidence):
    # a reference option that leaves the window the plant analysis computed is a known bad
    # option: the same fact that was warned about before the run
    windows = (evidence or {}).get("reference_windows") or {}
    t_end = ((evidence or {}).get("simulation") or {}).get("t_end")
    if not windows or not t_end:
        return report
    for s in report["suggestions"]:
        if s["lever"] != "reference" or not s.get("options"):
            continue
        window = windows.get(s.get("target")) or (list(windows.values())[0] if len(windows) == 1 else None)
        if not window:
            continue
        keep = []
        for o in s["options"]:
            span = _reference_span(o["value"], t_end)
            if span is None or (span[0] >= window[0] - 1e-6 and span[1] <= window[1] + 1e-6):
                keep.append(o)
        s["options"] = keep
    report["suggestions"] = [s for s in report["suggestions"]
                             if not (s["lever"] == "reference" and s.get("options") == [])]
    return report


def _prune_for_evidence(report, evidence, memory=None):
    # rules the prompt states but a small model does not keep, so they are enforced here
    evidence = evidence or {}
    # a different start only matters if the trouble begins at once or x0 is the bad point;
    # a failure that develops later (or is only steady-state error) is not about x0
    if not (_x0_at_a_spike(evidence) or _trouble_starts_at_once(evidence)):
        report["suggestions"] = [s for s in report["suggestions"] if s["lever"] != "initial_condition"]
    # a smaller solver step only for a numerical failure, and never a larger one
    dt = ((evidence.get("simulation") or {}).get("dt"))
    if not _numerical_trouble(evidence):
        report["suggestions"] = [s for s in report["suggestions"] if s["lever"] != "step_time"]
    elif dt:
        for s in report["suggestions"]:
            if s["lever"] == "step_time":
                s["options"] = [o for o in s["options"] if float(o["value"]) < float(dt)]
        report["suggestions"] = [s for s in report["suggestions"]
                                 if not (s["lever"] == "step_time" and not s["options"])]
    # the model never proposes the limits (no button, no number to stand behind), with or without
    # a limit in this run: the one such card is built from computed facts. reference cards stay
    report["suggestions"] = [s for s in report["suggestions"]
                             if not (s["lever"] in (None, "system", "input_limits")
                                     and any(w in s["title"].lower() for w in _LIMIT_WORDS))]
    card = failure_mode.limit_card(evidence) if evidence.get("input_limits") else None
    if card:
        report["suggestions"].insert(0, card)
    widen = failure_mode.bounds_card(evidence)
    if widen:
        report["suggestions"].insert(0, widen)
    computed = failure_mode.headline(evidence)
    if computed:
        report["headline"] = computed
    report = _inside_known_window(report, evidence)
    ref_card = failure_mode.reference_card(evidence)
    if ref_card:
        # the computed card replaces the model's own reference cards for that output
        report["suggestions"] = [s for s in report["suggestions"]
                                 if not (s["lever"] == "reference"
                                         and s.get("target") in (None, ref_card["target"]))]
        report["suggestions"].insert(0, ref_card)
    report["suggestions"] = _apply_memory(report["suggestions"], memory)[:_MAX_SUGGESTIONS]
    if isinstance(report.get("explanation"), list):
        # earlier runs are described by the computed bullet, not by the model
        own = [e for e in report["explanation"] if not _HISTORY_CLAIM.search(e)]
        mode = evidence.get("failure_mode") or {}
        facts = list(mode.get("chain") or [])
        if mode.get("ruled_out"):
            facts.append("Checked and not the cause: %s." % "; ".join(mode["ruled_out"]))
        bullet = diagnosis_chain.describe_changes(
            evidence.get("previous_runs"), evidence.get("changes_since_last_run"), evidence.get("metrics"))
        # a computed pointer for an error that has no computed cause: the tuning stage is the tool
        # (also when the same design misses the target without the limits: another cause is there)
        free = ((evidence.get("input_limits") or {}).get("without_limits")) or {}
        no_cause = (mode.get("mode") in ("tracking_error_only", "not_limit_related")
                    or (mode.get("mode") == "limit_adds_to_failure" and free.get("failed_check") == "mse_target"))
        if (no_cause and evidence.get("tuning_ran") is False
                and str((evidence.get("metrics") or {}).get("success_reason", "")).startswith("mse_target")):
            facts.append("Tuning was off for this run. For a tracking error with no computed cause, "
                         "turning it on may give better results.")
        # the model often restates the computed facts or says nothing: keep only what adds something
        plant = evidence.get("plant") or {}
        names = list(plant.get("states") or []) + list(plant.get("inputs") or [])
        own = [e for e in own if not _adds_nothing(
            e, facts + ([bullet] if bullet else []) + [report.get("headline") or ""], names)]
        # computed bullets first (history, then the causal chain), the model's own after them
        report["explanation"] = ([bullet] if bullet else []) + facts + own
    return report


def diagnose(evidence, memory=None):
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
            return _prune_for_evidence(_coerce_report(payload), evidence, memory), usage
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
