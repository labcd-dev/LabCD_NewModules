"""Plant-compatibility step: Python finds what is wrong, an LLM proposes conversion plans,
Python verifies and labels each one, a second LLM may veto. If the LLM fails, simple rules
take over. Plants that already fit never reach a model.
"""

import json

from . import llm_factory
from .agent_io import _empty_usage, _extract_json_payload, _extract_usage, _sum_usage
from .prompt_loader import load_prompt
from backend_core.AgentAdaptive.controller import plant_conversion
from backend_core.AgentAdaptive.controller.plant_analysis import analyze_spec
from backend_core.AgentAdaptive.controller.plant_conversion import ConversionError
from backend_core.AgentAdaptive.tools import prose_check, system_spec
from backend_core.AgentAdaptive.tools.progress import _emit

STRATEGIST_PROMPT = load_prompt("compat_strategist_prompt.yaml")
CRITIC_PROMPT = load_prompt("compat_critic_prompt.yaml")
EXPLAINER_PROMPT = load_prompt("compat_explainer_prompt.yaml")
CHAT_PROMPT = load_prompt("compat_chat_prompt.yaml")

MAX_REPAIR_ROUNDS = 2
MAX_PLANS = 4
SUPPORTED_TYPES = ("T0", "T1", "T2", "T3", "T4", "T5", "T6")
_MAX_MENU_CHOICES = 8
_RECORD_CHARS = 3000


def _plant_view(spec):
    dyn = system_spec.substitute_parameters(spec)["dynamics"]
    return {k: dyn[k] for k in ("states", "state_meanings", "state_equations", "inputs",
                                "outputs", "references", "uncertainty")}


def _analysis_view(a):
    return {
        "summary": a["summary"], "dims": a["dims"],
        "relative_degrees": a["relative_degrees"], "hidden_dimension": a["hidden_dimension"],
        "reasons": {"smc": a["methods"]["smc"]["reasons"],
                    "backstepping": a["methods"]["backstepping"]["reasons"]},
        "findings": [f["message"] for f in a["findings"]],
    }


def _record_text(record):
    if not record:
        return ""
    text = record if isinstance(record, str) else json.dumps(record, ensure_ascii=False)
    return text[:_RECORD_CHARS]


def _build_menu(spec, analysis, want_backstepping=False):
    # a conversion is only offered if its search found something that fits
    hints = {h["id"] for h in analysis["conversion_hints"]}
    menu = []
    chain = analysis["chain"]
    if chain.get("relabel_order") and not chain.get("given_order_ok"):
        menu.append({"type": "T0", "does": "reorder the states so the plant is a strict-feedback chain",
                     "params": {}})
    if "T1" in hints:
        menu.append({"type": "T1", "does": "drive the inputs with one new command per output",
                     "params": {"weights": "one positive number per input, in the order of "
                                           "`inputs` (higher = used less)"},
                     "influence": plant_conversion.decoupling_at(spec)})
    if "T2" in hints:
        try:
            choices = plant_conversion.rank_output_selections(spec)
        except ConversionError:
            choices = []
        if choices:
            n_in = analysis["dims"]["n_inputs"]
            menu.append({"type": "T2", "does": "track only some of the outputs",
                         "params": {"outputs": "exactly %d output name(s)" % n_in},
                         "choices": choices})
    n_in, n_out = analysis["dims"]["n_inputs"], analysis["dims"]["n_outputs"]
    if "T3" in hints and n_in == n_out:
        try:
            choices = plant_conversion.rank_output_reselections(spec)[:_MAX_MENU_CHOICES]
        except ConversionError:
            choices = []
        if any(c["fits"] for c in choices):
            menu.append({"type": "T3", "does": "track different state variables as the outputs",
                         "params": {"outputs": "exactly %d state name(s)" % n_in},
                         "note": "the user must supply a reference for any new output; never invent one",
                         "choices": choices})
    if "T4" in hints:
        try:
            candidates = plant_conversion.propose_extensions(spec)
        except ConversionError:
            candidates = []
        fitting = [c for c in candidates if c["fits"]]
        if fitting:
            menu.append({"type": "T4",
                         "does": "add integrators in front of some inputs; the controller then commands the highest derivative",
                         "params": {"inputs": "names of the inputs to extend (optional, default: the first fitting set)",
                                    "orders": "{input name: number of integrators 1-3} (optional; default: the fitting choice in menu.choices)",
                                    "initial_inputs": "{input name: starting value} (optional; unknown values are assumed to be 1)"},
                         "choices": fitting[:3]})
    if "T5" in hints and want_backstepping:
        try:
            plant_conversion.apply_lie_coordinates(spec)
            menu.append({"type": "T5", "does": "change to chain coordinates so backstepping can be used",
                         "params": {}})
        except ConversionError:
            pass
    if "T6" in hints and want_backstepping:
        try:
            plant_conversion.apply_term_shift(spec)
            menu.append({"type": "T6", "experimental": True,
                         "does": "design backstepping for a simplified chain model and treat the terms that break "
                                 "the chain as uncertainty; the plant itself is unchanged",
                         "params": {}})
        except ConversionError:
            pass
    optional = {"T5", "T6"}
    not_offered = sorted(h for h in hints if h not in {m["type"] for m in menu}
                         and (want_backstepping or h not in optional))
    return menu, not_offered


class _CallFailed(Exception):
    pass


def _call(role, system_prompt, messages):
    # returns (reply_text, payload, usage, parse_error); raises _CallFailed if the model
    # could not be reached at all
    try:
        llm = llm_factory.build_llm(role)
        resp = llm.invoke([{"role": "system", "content": system_prompt}] + messages)
    except Exception as e:  # noqa: BLE001
        raise _CallFailed("%s: %s" % (type(e).__name__, e)) from e
    usage = _extract_usage(resp)
    text = getattr(resp, "content", "") or ""
    payload, err = _extract_json_payload(text)
    if payload is None:
        return text, None, usage, "the reply could not be read as JSON (%s: %s)" % (type(err).__name__, err)
    return text, payload, usage, None


def _coerce_plans(payload):
    # keeps what is usable; everything dropped is explained in `problems`
    problems, plans = [], []
    if not isinstance(payload, dict) or not isinstance(payload.get("plans"), list):
        return [], None, ['the reply needs a "plans" list.']
    seen = set()
    for i, raw in enumerate(payload["plans"][:MAX_PLANS], start=1):
        if not isinstance(raw, dict):
            problems.append("plan %d is not an object." % i)
            continue
        pid = str(raw.get("id") or "p%d" % i)
        if pid in seen:
            pid = "p%d" % i
        seen.add(pid)
        ptype = str(raw.get("type") or "")
        if ptype not in SUPPORTED_TYPES:
            problems.append("plan %s has type %r; allowed types are %s." % (pid, ptype, ", ".join(SUPPORTED_TYPES)))
            continue
        params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
        plans.append({"id": pid, "type": ptype, "params": params,
                      "rationale": str(raw.get("rationale") or "").strip(),
                      "risks": str(raw.get("risks") or "").strip()})
    rec = payload.get("recommended")
    return plans, (str(rec) if rec else None), problems


def run_conversion(spec, ptype, params, on_event=None, references=None):
    # one place that maps a plan/option to the matching apply_* call
    params = params or {}
    if ptype == "T0":
        return plant_conversion.apply_relabel(spec, on_event=on_event)
    if ptype == "T1":
        return plant_conversion.apply_input_squaring(
            spec, weights=params.get("weights"), virtual_names=params.get("virtual_names"),
            on_event=on_event)
    if ptype == "T2":
        return plant_conversion.apply_output_selection(spec, params.get("outputs") or [], on_event=on_event)
    if ptype == "T3":
        return plant_conversion.apply_output_reselection(
            spec, params.get("outputs") or [], references=references or params.get("references"),
            on_event=on_event)
    if ptype == "T5":
        return plant_conversion.apply_lie_coordinates(spec, on_event=on_event)
    if ptype == "T6":
        return plant_conversion.apply_term_shift(spec, on_event=on_event)
    if ptype == "T4":
        return plant_conversion.apply_dynamic_extension(
            spec, inputs=params.get("inputs"), initial_inputs=params.get("initial_inputs"),
            orders=params.get("orders"), on_event=on_event)
    raise ConversionError("unknown conversion type %r." % ptype)


def _verify(spec, plans):
    verified, failures = [], []
    for plan in plans:
        try:
            result = run_conversion(spec, plan["type"], plan["params"])
        except (ConversionError, TypeError, ValueError, KeyError) as e:
            failures.append({"id": plan["id"], "type": plan["type"], "params": plan["params"],
                             "error": str(e)})
            continue
        verified.append((plan, result))
    return verified, failures


def _normalized_params(ptype, result):
    rec, dyn = result["record"], result["spec"]["dynamics"]
    if ptype == "T1":
        return {"weights": rec["details"]["weights"], "virtual_names": rec["details"]["virtual_inputs"]}
    if ptype in ("T2", "T3"):
        return {"outputs": list(dyn["outputs"])}
    if ptype == "T4":
        out = {"inputs": rec["details"]["extended_inputs"], "initial_inputs": rec["details"]["initial_inputs"]}
        if any(k > 1 for k in rec["details"]["orders"].values()):
            out["orders"] = rec["details"]["orders"]
        return out
    return {}


def _option(plan, result):
    rec = result["record"]
    warnings = [f["message"] for f in result["analysis"]["findings"] if f["severity"] == "warning"]
    return {
        "id": plan["id"], "type": plan["type"], "title": rec["title"],
        "params": _normalized_params(plan["type"], result),
        "label": rec["label"], "label_reasons": rec["label_reasons"],
        "task_changed": rec["task_changed"], "changes": rec["changes"],
        "untracked_outputs": rec.get("untracked_outputs", []),
        "input_map": rec["details"].get("input_map"),
        "input_map_display": rec["details"].get("input_map_display"),
        "effort_share": rec["details"].get("effort_share"),
        "limits": _limits(rec),
        "needs_references": rec["details"].get("missing_references", []),
        "assumed_initial_inputs": rec["details"].get("assumed_initial_inputs", []),
        "rationale": plan["rationale"], "risks": plan["risks"],
        "warnings": warnings, "vetoed": False, "veto_reason": "",
    }


# fixed wording per conversion type: what the no-model plans say, and what replaces a
# model sentence that fails the fact check
_RULE_TEXT = {
    "T0": ("Default rule: reorder the states into a chain.", "None beyond the new state order."),
    "T1": ("Default rule: use all inputs equally.", "Actuator limits are not considered."),
    "T2": ("Default rule: best-ranked output set (fits, no unstable-zero hint, fewest hidden states).",
           "The other outputs are no longer controlled."),
    "T3": ("Default rule: the best-ranked output set (fits, no unstable-zero hint, keeps the most original outputs).",
           "The task changes and the new outputs need a reference from you."),
    "T4": ("Default rule: the smallest set of inputs that restores an invertible matrix.",
           "The starting value of the extended input is assumed; check it."),
    "T5": ("Default rule: the plant is not a chain, and Lie coordinates make it one.",
           "The inverse and the gain are only checked near the initial state."),
    "T6": ("Default rule: the terms that break the chain are small enough to treat as uncertainty.",
           "Experimental: the stability proof does not cover the gap between the design model and the plant."),
}


def _limits(rec):
    # what the user should keep in mind, written by python from the verified record
    details, out = rec["details"], []
    if rec["task_changed"]:
        out.append("The task changes: %s no longer tracked, and nothing is guaranteed for %s."
                   % (", ".join(rec.get("untracked_outputs") or []) or "no original output is",
                      "it" if len(rec.get("untracked_outputs") or []) == 1 else "them"))
    if details.get("missing_references"):
        out.append("A reference is still needed for %s." % ", ".join(details["missing_references"]))
    if details.get("assumed_initial_inputs"):
        out.append("The starting value of %s is assumed; check it." % ", ".join(details["assumed_initial_inputs"]))
    if details.get("rate_inputs"):
        out.append("The real actuator has to be driven through the added integrator(s).")
    if details.get("caveat"):
        out.append(details["caveat"][:1].upper() + details["caveat"][1:])
    if rec.get("experimental"):
        out.append("Experimental: the stability proof does not cover the gap between the design model and the plant.")
    return out


def _rule_based_plans(menu):
    # no-model fallback: equal weights (T1), the best-ranked output set (T2, T3), the smallest extension (T4)
    plans = []
    for item in menu:
        kind = item["type"]
        params = {}
        if kind in ("T2", "T3"):
            fitting = [c for c in item["choices"] if c["fits"]]
            if not fitting:
                continue
            params = {"outputs": fitting[0]["outputs"]}
        elif kind == "T4":
            params = {"inputs": item["choices"][0]["inputs"]}
            if any(k > 1 for k in item["choices"][0]["orders"].values()):
                params["orders"] = item["choices"][0]["orders"]
        plans.append({"id": "r" + kind[1:], "type": kind, "params": params,
                      "rationale": _RULE_TEXT[kind][0], "risks": _RULE_TEXT[kind][1]})
    return plans[:MAX_PLANS]


def _check_plan_text(plan, evidence_text, dims, notes):
    # a plan's rationale/risks that contradict the facts are swapped for the fixed wording
    for idx, field in enumerate(("rationale", "risks")):
        problems = prose_check.check(plan[field], evidence_text, dims)
        if problems:
            fixed = _RULE_TEXT[plan["type"]][idx].replace("Default rule: ", "")
            plan[field] = fixed[:1].upper() + fixed[1:]
            notes.append("plan %s: the model's %s was replaced by fixed text (%s)"
                         % (plan["id"], field, problems[0]))


def _run_critic(evidence, options, dims=None):
    view = {"plant": evidence["plant"], "user_conversation": evidence["user_conversation"],
            "options": [_for_model(o, ("id", "type", "params", "label", "label_reasons", "task_changed",
                                       "changes", "input_map", "effort_share", "limits")) for o in options]}
    try:
        _, payload, usage, err = _call("compat_critic", CRITIC_PROMPT,
                                       [{"role": "user", "content": json.dumps(view, ensure_ascii=False, indent=2)}])
    except _CallFailed as e:
        return {}, {"status": "unavailable", "error": str(e)}, _empty_usage()
    if err or not isinstance(payload, dict) or not isinstance(payload.get("verdicts"), list):
        return {}, {"status": "unavailable", "error": err or 'the reply needs a "verdicts" list.'}, usage
    verdicts, dropped = {}, []
    facts = json.dumps(view, ensure_ascii=False)
    for v in payload["verdicts"]:
        if isinstance(v, dict) and v.get("verdict") == "veto" and str(v.get("reason") or "").strip():
            reason = str(v["reason"]).strip()
            # an objection that names things or numbers the facts do not contain is not trusted
            problems = prose_check.check(reason, facts + "\n" + evidence["user_conversation"], dims)
            if problems:
                dropped.append({"id": str(v.get("id")), "problem": problems[0]})
            else:
                verdicts[str(v.get("id"))] = reason
    return verdicts, {"status": "ok", "vetoes": len(verdicts), "dropped": dropped}, usage


def _order(options, recommended_id):
    index = {o["id"]: i for i, o in enumerate(options)}
    ranked = sorted(options, key=lambda o: (o["vetoed"], o["label"] != "A", o["task_changed"], index[o["id"]]))
    pick = next((o for o in options if o["id"] == recommended_id and not o["vetoed"]), None)
    safer = [o for o in options if not o["vetoed"] and not o["task_changed"]]
    if pick is not None and pick["task_changed"] and safer:
        pick = None
    if pick is not None:
        ranked.remove(pick)
        ranked.insert(0, pick)
    best = ranked[0] if ranked and not ranked[0]["vetoed"] else None
    return ranked, (best["id"] if best else None)


def _message(analysis, options, recommended, source):
    lines = [analysis["summary"]]
    if not options:
        return "\n".join(lines)
    if analysis["status"] == "fits":
        lines.append("The plant already fits as it is; the options below address its warnings.")
    lines.append("%d conversion option(s) can be applied%s:" % (
        len(options), "" if source == "llm" else " (chosen by fixed rules, no model was used)"))
    for o in options:
        flags = ["label %s" % o["label"]]
        if o["task_changed"]:
            flags.append("changes the task")
        if o["vetoed"]:
            flags.append("vetoed by the reviewer: %s" % o["veto_reason"])
        lines.append("- %s%s: %s (%s)" % (o["id"], " [recommended]" if o["id"] == recommended else "",
                                          o["changes"][0], "; ".join(flags)))
    return "\n".join(lines)


def _result(status, analysis, **extra):
    out = {"status": status, "plant_fits": analysis["status"] == "fits", "analysis": analysis,
           "options": [], "recommended": None,
           "source": None, "message": analysis["summary"], "llm_calls": 0,
           "usage": _empty_usage(), "usage_by_role": {}, "notes": [], "critic": None,
           "llm_error": None}
    out.update(extra)
    return out


def optional_conversions(spec):
    # python only, no model: for a plant that already fits, what could still be offered.
    # each item: {"kind": "backstepping" | "hidden_dynamics", "text", "experimental"}
    analysis = analyze_spec(spec)
    if analysis["status"] != "fits":
        return []
    ids = {h["id"] for h in analysis["conversion_hints"]}
    out = []
    if "backstepping" not in analysis["candidates"] and ids & {"T5", "T6"}:
        for kind, apply, text, experimental in (
                ("T5", plant_conversion.apply_lie_coordinates,
                 "Backstepping is also possible: an exact change of coordinates makes the plant a "
                 "strict-feedback chain. SMC works without it.", False),
                ("T6", plant_conversion.apply_term_shift,
                 "Backstepping is only possible through an experimental simplification (some terms are "
                 "treated as uncertainty). SMC works without it.", True)):
            if kind not in ids:
                continue
            try:
                apply(spec)
            except ConversionError:
                continue
            out.append({"kind": "backstepping", "text": text, "experimental": experimental})
            break
    if "T3" in ids:
        try:
            choices = plant_conversion.rank_output_reselections(spec)
        except ConversionError:
            choices = []
        if any(c["fits"] for c in choices):
            warning = next((f["message"] for f in analysis["findings"] if f["check"] == "C6"), "")
            out.append({"kind": "hidden_dynamics", "experimental": False,
                        "text": (warning + " " if warning else "")
                                + "Tracking other outputs may avoid this, but it changes the task."})
    return out


def propose_conversions(spec, clarification_record=None, on_event=None,
                        max_repair_rounds=MAX_REPAIR_ROUNDS, offer_for_warnings=False,
                        want_backstepping=False):
    # offer_for_warnings: also look at plants that fit but whose hidden dynamics look unstable
    # want_backstepping: also offer chain coordinates (T5) when SMC already accepts the plant
    analysis = analyze_spec(spec)
    hint_ids = {h["id"] for h in analysis["conversion_hints"]}
    fits_without_bs = analysis["status"] == "fits" and "backstepping" not in analysis["candidates"]
    warned = (analysis["status"] == "fits"
              and ((offer_for_warnings and "T3" in hint_ids)
                   or (want_backstepping and fits_without_bs and hint_ids & {"T5", "T6"})))
    if analysis["status"] != "does_not_fit" and not warned:
        return _result("fits" if analysis["status"] == "fits" else "cannot_analyze", analysis)

    menu, not_offered = _build_menu(spec, analysis, want_backstepping=want_backstepping)
    if not menu and analysis["status"] == "fits":
        return _result("fits", analysis, notes=["an optional conversion was looked for but none can be applied"])
    if not menu:
        extra = ("" if not not_offered else " Looked for a fix (%s) but none gives a plant the module accepts."
                 % ", ".join(not_offered))
        msg = analysis["summary"] + extra
        _emit(on_event, kind="note", stage="design", text="No conversion can be applied. " + msg)
        return _result("rejected", analysis, message=msg, notes=["no executable conversion"])

    evidence = {"plant": _plant_view(spec), "analysis": _analysis_view(analysis), "menu": menu,
                "user_conversation": _record_text(clarification_record)}
    _emit(on_event, kind="note", stage="design",
          text="%s Asking for conversion plans (%s)." % (
              "The plant fits but has warnings." if warned else "The plant does not fit.",
              ", ".join(m["type"] for m in menu)))

    usage = {"compat": _empty_usage(), "compat_critic": _empty_usage()}
    calls, notes, llm_error = 0, [], None
    messages = [{"role": "user", "content": json.dumps(evidence, ensure_ascii=False, indent=2)}]
    verified, source, recommended_id = [], "llm", None
    for round_no in range(max_repair_rounds + 1):
        calls += 1
        try:
            text, payload, u, err = _call("compat", STRATEGIST_PROMPT, messages)
        except _CallFailed as e:
            llm_error = str(e)
            break
        usage["compat"] = _sum_usage(usage["compat"], u)
        plans, recommended_id, problems = ([], None, [err]) if err else _coerce_plans(payload)
        verified, failures = _verify(spec, plans)
        failures += [{"id": None, "error": p} for p in problems]
        _emit(on_event, kind="note", stage="design",
              text="Round %d: %d plan(s) verified, %d rejected." % (round_no + 1, len(verified), len(failures)))
        if verified:
            notes += ["plan %s dropped: %s" % (f["id"], f["error"]) for f in failures]
            break
        if round_no < max_repair_rounds:
            messages += [{"role": "assistant", "content": text or "{}"},
                         {"role": "user", "content": json.dumps({"failed_plans": failures}, ensure_ascii=False)}]

    if not verified:
        source = "rules"
        notes.append("the model gave no usable plan (%s); fixed rules were used instead."
                     % (llm_error or "all plans failed verification"))
        verified, _ = _verify(spec, _rule_based_plans(menu))
        recommended_id = None
    if not verified:
        msg = analysis["summary"] + " No conversion could be verified."
        _emit(on_event, kind="note", stage="design", text=msg)
        return _result("rejected", analysis, message=msg, notes=notes, llm_calls=calls,
                       usage=usage["compat"], usage_by_role=usage, llm_error=llm_error)

    if source == "llm":
        evidence_text = json.dumps(evidence, ensure_ascii=False)
        for plan, _ in verified:
            _check_plan_text(plan, evidence_text, analysis["dims"], notes)
    options = [_option(plan, result) for plan, result in verified]
    critic = None
    if source == "llm":
        vetoes, critic, u = _run_critic(evidence, options, analysis["dims"])
        calls += 1
        usage["compat_critic"] = u
        for o in options:
            if o["id"] in vetoes:
                o["vetoed"], o["veto_reason"] = True, vetoes[o["id"]]
                _emit(on_event, kind="note", stage="design",
                      text="Reviewer vetoed %s: %s" % (o["id"], vetoes[o["id"]]))
    if source == "llm" and options and all(o["vetoed"] for o in options):
        # dead end: every plan was vetoed, so the strategist gets one revision round with the objections
        revised = _revise_after_veto(spec, evidence, analysis, messages, text, options, usage, notes, on_event)
        calls += revised["calls"]
        if revised["options"]:
            options, recommended_id = revised["options"], revised["recommended"]
    options, recommended = _order(options, recommended_id)
    _emit(on_event, kind="note", stage="design",
          text="%d conversion option(s) ready; recommended: %s." % (len(options), recommended or "none"))
    return _result(
        "needs_choice", analysis, options=options, recommended=recommended, source=source,
        message=_message(analysis, options, recommended, source), llm_calls=calls, notes=notes,
        usage=_sum_usage(usage["compat"], usage["compat_critic"]), usage_by_role=usage,
        critic=critic, llm_error=llm_error)


def _revise_after_veto(spec, evidence, analysis, messages, last_reply, vetoed, usage, notes, on_event):
    # one extra strategist call that sees the reviewer's reasons, then verify and review again.
    # Returns the new options only if at least one of them survives the reviewer.
    objections = [{"id": o["id"], "reason": o["veto_reason"]} for o in vetoed]
    messages = messages + [
        {"role": "assistant", "content": last_reply or "{}"},
        {"role": "user", "content": json.dumps(
            {"reviewer_objections": objections,
             "instruction": "Every plan was vetoed. Revise the parameters (or choose another conversion from the "
                            "menu) so the objection no longer applies, and return the complete object."},
            ensure_ascii=False)}]
    out = {"calls": 1, "options": [], "recommended": None}
    _emit(on_event, kind="note", stage="design", text="Every plan was vetoed; asking for a revision.")
    try:
        text, payload, u, err = _call("compat", STRATEGIST_PROMPT, messages)
    except _CallFailed as e:
        notes.append("the revision after the reviewer's objection failed (%s)" % e)
        return out
    usage["compat"] = _sum_usage(usage["compat"], u)
    plans, recommended_id, _ = ([], None, [err]) if err else _coerce_plans(payload)
    verified, _ = _verify(spec, plans)
    if not verified:
        notes.append("the revision after the reviewer's objection gave no usable plan")
        return out
    evidence_text = json.dumps(evidence, ensure_ascii=False)
    for plan, _ in verified:
        _check_plan_text(plan, evidence_text, analysis["dims"], notes)
    options = [_option(plan, result) for plan, result in verified]
    vetoes, _, u = _run_critic(evidence, options, analysis["dims"])
    out["calls"] += 1
    usage["compat_critic"] = _sum_usage(usage["compat_critic"], u)
    for o in options:
        if o["id"] in vetoes:
            o["vetoed"], o["veto_reason"] = True, vetoes[o["id"]]
    if all(o["vetoed"] for o in options):
        notes.append("a revision was tried after the reviewer's objection; the reviewer vetoed it too")
        return out
    notes.append("revised after the reviewer's objection (%s)" % "; ".join(o["reason"] for o in objections))
    out["options"], out["recommended"] = options, recommended_id
    return out


def apply_option(spec, option, on_event=None, references=None):
    # references: {output: expression} for outputs the user is adding (T3)
    return run_conversion(spec, option.get("type"), option.get("params"), on_event=on_event,
                          references=references)


def _for_model(option, keys):
    # same facts the user sees; the model gets the rounded split, not the 15-digit one
    view = {k: option[k] for k in keys}
    if option.get("input_map_display"):
        view["input_map"] = option["input_map_display"]
    return view


_MAX_FACTS = 4
_CHAT_FALLBACK = ("I could not check that answer against the verified facts, so I am not showing it. "
                  "Please ask again about a specific option, or use the details above.")


def _facts(analysis):
    # the reasons Python found, in a fixed order: these are what the user is told, in the model's words
    reasons = []
    for key in ("smc", "backstepping"):
        for r in analysis["methods"][key]["reasons"]:
            if r not in reasons:
                reasons.append(r)
    if not reasons:
        reasons = [f["message"] for f in analysis["findings"] if f["severity"] == "warning"]
    return reasons[:_MAX_FACTS]


def _explain_evidence(proposal, spec):
    keys = ("id", "type", "title", "changes", "label", "label_reasons", "task_changed",
            "untracked_outputs", "input_map", "effort_share", "limits", "needs_references",
            "assumed_initial_inputs", "warnings", "vetoed", "veto_reason")
    return {"plant": _plant_view(spec), "analysis": _analysis_view(proposal["analysis"]),
            "facts": [{"id": "f%d" % (i + 1), "text": t} for i, t in enumerate(_facts(proposal["analysis"]))],
            "options": [_for_model(o, keys) for o in proposal["options"]],
            "recommended": proposal["recommended"], "notes": proposal["notes"]}


def _template_explanation(proposal):
    # plain fallback built from the same verified facts; also what a model reply is merged over
    a, options = proposal["analysis"], proposal["options"]
    per_option = {}
    for o in options:
        text = o["changes"][0]
        if o["vetoed"]:
            text += " The reviewer vetoed it: %s" % o["veto_reason"]
        per_option[o["id"]] = text
    best = next((o for o in options if o["id"] == proposal["recommended"]), None)
    if best:
        rec = "Recommended: %s. %s" % (best["id"], best["changes"][0])
    elif options:
        rec = "No option is recommended because the reviewer vetoed every one."
    else:
        rec = "No conversion could be applied to this plant."
    headline = ("This plant needs a conversion before it can be designed" if options
                else "This plant cannot be designed for with the available conversions")
    return {"headline": headline, "why_not_fit": _facts(a), "options": per_option, "recommendation": rec}


def _read_explanation(payload, proposal, evidence_text, dims):
    # keeps only the model sentences that pass the fact check; `problems` lists the rest.
    # why_not_fit always has one bullet per fact: the model's rewrite if it checks out, else the fact itself
    facts = _facts(proposal["analysis"])
    accepted, problems = {}, []

    def take(field, text, number_source=None):
        text = str(text or "").strip()
        if not text:
            return None
        found = prose_check.check(text, evidence_text, dims, number_source=number_source)
        if found:
            problems.append({"field": field, "text": text, "problem": found[0]})
            return None
        return text

    for field in ("headline", "recommendation"):
        text = take(field, payload.get(field))
        if text:
            accepted[field] = text
    raw = payload.get("why_not_fit") if isinstance(payload.get("why_not_fit"), list) else []
    by_id = {str(item.get("id")): item.get("text") for item in raw if isinstance(item, dict)}
    by_pos = [item for item in raw if isinstance(item, str)]
    bullets = []
    for i, fact in enumerate(facts):
        rewrite = by_id.get("f%d" % (i + 1)) or (by_pos[i] if i < len(by_pos) else None)
        bullets.append(take("why_not_fit f%d" % (i + 1), rewrite, number_source=fact) or fact)
    accepted["why_not_fit"] = bullets
    known = {o["id"] for o in proposal["options"]}
    summaries = {}
    for item in payload.get("options") or []:
        if isinstance(item, dict) and str(item.get("id")) in known:
            text = take("option %s" % item["id"], item.get("summary"))
            if text:
                summaries[str(item["id"])] = text
    accepted["options"] = summaries
    return accepted, problems


def explain(proposal, spec):
    # plain-language explanation of a propose_conversions result; never raises. The model rewrites
    # Python's facts; every sentence is fact-checked, and one repair round is allowed
    out = dict(_template_explanation(proposal), source="template", error=None, usage=_empty_usage(), replaced=[])
    if proposal["status"] != "needs_choice":
        return out
    evidence_text = json.dumps(_explain_evidence(proposal, spec), ensure_ascii=False, indent=2)
    dims = proposal["analysis"]["dims"]
    messages = [{"role": "user", "content": evidence_text}]
    parsed = None
    for attempt in range(2):
        try:
            text, payload, usage, err = _call("compat_explainer", EXPLAINER_PROMPT, messages)
        except _CallFailed as e:
            out["error"] = out["error"] or str(e)
            break
        out["usage"] = _sum_usage(out["usage"], usage)
        if err or not isinstance(payload, dict):
            out["error"] = out["error"] or err or "the reply was not a JSON object."
            break
        parsed = _read_explanation(payload, proposal, evidence_text, dims)
        out["error"] = None
        if not parsed[1] or attempt == 1:
            break
        feedback = [{"field": p["field"], "problem": p["problem"]} for p in parsed[1]]
        messages += [{"role": "assistant", "content": text or "{}"},
                     {"role": "user", "content": json.dumps(
                         {"rejected_sentences": feedback,
                          "instruction": "Fix exactly these sentences using only the facts, and return the complete object."},
                         ensure_ascii=False)}]
    if parsed is None:
        return out
    accepted, problems = parsed
    for field in ("headline", "recommendation", "why_not_fit"):
        if accepted.get(field):
            out[field] = accepted[field]
    out["options"].update(accepted["options"])
    out["replaced"] = [{"field": p["field"], "problem": p["problem"]} for p in problems]
    out["source"] = "llm"
    return out


def answer_followup(proposal, spec, question, history=None, explanation=None):
    # chat about the options; returns (reply, usage) and never raises. The reply is fact-checked
    # (one repair round) and replaced by a plain notice if it still contradicts the facts
    turns = "\n".join("%s: %s" % ("You" if h.get("role") == "assistant" else "User", h.get("text", ""))
                      for h in (history or []))
    evidence_text = json.dumps(_explain_evidence(proposal, spec), ensure_ascii=False, indent=2)
    earlier = json.dumps(explanation or _template_explanation(proposal), ensure_ascii=False, indent=2)
    parts = ["Evidence:\n" + evidence_text, "Your earlier explanation:\n" + earlier]
    if turns:
        parts.append("Conversation so far:\n" + turns)
    parts.append("New question: " + question)
    messages = [{"role": "user", "content": "\n\n".join(parts)}]
    allowed = "\n".join([evidence_text, earlier, turns, question])
    dims = proposal["analysis"]["dims"]
    total = _empty_usage()
    for attempt in range(2):
        try:
            text, payload, usage, err = _call("compat_explainer", CHAT_PROMPT, messages)
        except _CallFailed as e:
            return "I couldn't answer that (%s)." % e, total
        total = _sum_usage(total, usage)
        reply = str(payload.get("reply") or "").strip() if isinstance(payload, dict) else ""
        if not reply:
            return "I couldn't answer that (%s)." % (err or "empty reply"), total
        problems = prose_check.check(reply, allowed, dims, numbers=False)
        if not problems:
            return reply, total
        messages += [{"role": "assistant", "content": text or "{}"},
                     {"role": "user", "content": json.dumps(
                         {"rejected": problems,
                          "instruction": "Answer again using only the facts, as the same JSON object."},
                         ensure_ascii=False)}]
    return _CHAT_FALLBACK, total


def replan(spec, proposal, feedback, clarification_record=None, on_event=None):
    # the user rejected what was offered: run the whole proposal again with their feedback up front
    shown = "; ".join("%s (%s)" % (o["id"], o["changes"][0]) for o in proposal["options"])
    text = "\n".join(["User feedback on the previous options: %s" % feedback,
                      "Previous options: %s" % shown,
                      _record_text(clarification_record)[:2000]])
    return propose_conversions(spec, clarification_record=text, on_event=on_event)


def build_conversion_record(proposal, option, applied, chosen_by="user", explanation=None):
    # what the pipeline and the report need to know about the conversion, JSON-safe
    rec = dict(applied["record"])
    rec["why"] = proposal["analysis"]["summary"]
    rec["verification"] = {
        "summary": applied["analysis"]["summary"],
        "warnings": [f["message"] for f in applied["analysis"]["findings"] if f["severity"] == "warning"],
        "relative_degrees": applied["analysis"]["relative_degrees"],
        "hidden_dimension": applied["analysis"]["hidden_dimension"]}
    rec["selection"] = {
        "option_id": option["id"], "source": proposal["source"], "chosen_by": chosen_by,
        "rationale": option["rationale"], "risks": option["risks"],
        "vetoed": option["vetoed"], "veto_reason": option["veto_reason"]}
    exp = explanation or {}
    rec["explanation"] = ({"headline": exp.get("headline"), "why_not_fit": exp.get("why_not_fit"),
                           "summary": (exp.get("options") or {}).get(option["id"]),
                           "recommendation": exp.get("recommendation")} if exp else None)
    rec["simulation_check"] = None
    return json.loads(json.dumps(rec))
