def render_final_report(method_label, args, components, why, notes_limitations,
                         tuning_values=None, metrics_report_text=None):
    lines = [
        "## Method: %s" % method_label,
        "**Why:** %s" % why,
        "",
        "## System (state-space form)",
        components["system"],
        "",
        "## Setup",
        "| Parameter | Value |",
        "|---|---|",
        "| has_delta | %s |" % args.get("has_delta"),
        "| has_disturbance | %s |" % args.get("has_disturbance"),
        "| delta_exprs | %s |" % args.get("delta_exprs"),
        "| dist_exprs | %s |" % args.get("dist_exprs"),
        "| states | %s |" % args.get("states"),
        "| inputs | %s |" % args.get("inputs"),
        "| outputs | %s |" % args.get("outputs"),
        "| refs | %s |" % str(args.get("refs")).replace("*", "\\*"),   # markdown table: keep the asterisks
        "| x0 | %s |" % args.get("x0"),
    ]
    if args.get("input_limits"):
        # one [min, max] per input (None = free): only worth a row when something is limited
        lines.append("| input_limits | %s |" % ", ".join(
            "%s in %s" % (name, "free" if lim is None else
                          "[%s, %s]" % ("-inf" if lim[0] is None else "%g" % lim[0],
                                        "+inf" if lim[1] is None else "%g" % lim[1]))
            for name, lim in zip(args.get("inputs") or [], args["input_limits"])))
    if args.get("output_bounds"):
        lines.append("| output_bounds | %s |" % ", ".join(
            "%s in %s" % (name, "free" if b is None else
                          "[%s, %s]" % ("-inf" if b[0] is None else "%g" % b[0],
                                        "+inf" if b[1] is None else "%g" % b[1]))
            for name, b in zip(args.get("outputs") or [], args["output_bounds"])))
    lines.append("")
    if tuning_values:
        lines.append("## Final Tuning Parameters")
        lines.append("| Parameter | Value |")
        lines.append("|---|---|")
        for _k, _v in tuning_values.items():
            lines.append("| %s | %s |" % (_k, _v))
        lines.append("")
    lines += [
        "## Control Law",
        components["control_law"],
        "",
        "## Stability Guarantee",
        components["stability"],
        "",
        "## Performance",
    ]
    if metrics_report_text and metrics_report_text.strip():
        lines.append("```")
        lines.append(metrics_report_text)
        lines.append("```")
    elif components.get("rms_str"):
        lines.append("- Last-window tracking RMS per output = [%s]" % components["rms_str"])
    else:
        lines.append("- (no uncertainty estimation active for this design)")
    lines.append("")
    lines += ["## Notes & Limitations",
              notes_limitations or "Nothing to report: the request was fully represented."]
    return "\n".join(lines)


_LABEL_TEXT = {
    "A": "A: exact rewrite of the nominal model; the controller's stability argument applies to the converted plant.",
    "C": "C: evidence only; the stability argument does not fully cover this plant (see the reasons).",
}


def _fmt(value):
    return "n/a" if value is None else "%.4g" % value


def _render_conversion_section(conversion):
    if not conversion:
        return ""
    lines = ["", "## Plant Conversion", "",
             "**Why the plant was converted:** %s" % conversion.get("why", "")]
    lines += ["", "**What was done:** %s." % conversion.get("title", "")]
    lines += ["- " + c for c in conversion.get("changes") or []]
    label = conversion.get("label")
    lines += ["", "**Safety label:** %s" % _LABEL_TEXT.get(label, label)]
    lines += ["- " + r for r in conversion.get("label_reasons") or []]
    if conversion.get("task_changed"):
        lines += ["", "**This conversion changes the task.** Outputs no longer tracked: %s. "
                      "Nothing is guaranteed for them." % ", ".join(conversion.get("untracked_outputs") or [])]
    original, details = conversion.get("original") or {}, conversion.get("details") or {}
    shown_map = details.get("input_map_display") or details.get("input_map")
    if shown_map:
        lines += ["", "**Original inputs in terms of the new commands:**"]
        lines += ["- %s = %s" % (k, v) for k, v in shown_map.items()]
    if details.get("missing_references"):
        lines += ["", "**A reference is still needed for:** %s." % ", ".join(details["missing_references"])]
    if details.get("rate_inputs"):
        orders = details.get("orders") or {}
        ordinal = {1: "rate of change", 2: "second derivative", 3: "third derivative"}
        what = ["%s (%s of %s)" % (new, ordinal[orders.get(old, 1)], old)
                for old, new in zip(details["extended_inputs"], details["rate_inputs"])]
        lines += ["", "**New control inputs:** %s. The real actuator has to be driven through %s."
                  % (", ".join(what), "integrators" if any(k > 1 for k in orders.values()) else "an integrator")]
        lines += ["- %s starts at %g%s" % (k, v, " (assumed)" if k in details.get("assumed_initial_inputs", []) else "")
                  for k, v in (details.get("initial_inputs") or {}).items()]
        if details.get("derivative_states"):
            lines.append("- the extra derivative states (%s) start at 0"
                         % ", ".join(n for names in details["derivative_states"].values() for n in names))
    if details.get("order"):
        lines += ["", "**New state order:** %s (was %s)." % (details["order"], original.get("states"))]
    if details.get("removed_terms"):
        lines += ["", "**Experimental: terms moved out of the design model into the uncertainty:**"]
        for state, terms in details["removed_terms"].items():
            lines.append("- %s: %s (%s%% of its right-hand side)"
                         % (state, " + ".join(terms), details["removed_size_percent"].get(state)))
        lines += ["", "The controller was built from: " + "; ".join(
            "%s' = %s" % kv for kv in (details.get("kept_equations") or {}).items())]
    if details.get("transformation"):
        lines += ["", "**New states in terms of the old ones:**"]
        lines += ["- %s = %s" % (k, v) for k, v in details["transformation"].items()]
        lines += ["", "**Old states in terms of the new ones (to read results back):**"]
        lines += ["- %s = %s" % (k, v) for k, v in (details.get("inverse") or {}).items()]
        lines += ["", "**Last equation:** %s' = %s" % (details["order"][-1], details.get("last_equation"))]
    ver = conversion.get("verification") or {}
    lines += ["", "**Checked after the conversion:** %s" % ver.get("summary", "")]
    if ver.get("relative_degrees"):
        hidden = ver.get("hidden_dimension")
        lines.append("- Relative degrees of the outputs: %s%s." % (
            ver["relative_degrees"],
            "; no hidden states" if hidden == 0 else ("; %d hidden state(s)" % hidden if hidden else "")))
    lines += ["- " + w for w in ver.get("warnings") or []]
    sim = conversion.get("simulation_check")
    if sim and sim.get("available"):
        verdict = ("all states stayed bounded" if sim["ok"]
                   else "state(s) %s grew or were not finite" % ", ".join(sim["diverging_states"]))
        lines += ["", "**Simulation check (empirical, not a proof):** %s." % verdict]
        if sim.get("untracked_outputs"):
            lines += ["", "| Untracked output | min | max | final |", "|---|---|---|---|"]
            lines += ["| %s | %s | %s | %s |" % (k, _fmt(v["min"]), _fmt(v["max"]), _fmt(v["final"]))
                      for k, v in sim["untracked_outputs"].items()]
    sel = conversion.get("selection") or {}
    if sel:
        how = "you chose it" if sel.get("chosen_by") == "user" else "it was selected automatically"
        lines += ["", "**How it was chosen:** %s; proposed by %s." % (
            how, "the model" if sel.get("source") == "llm" else "fixed rules (no model was used)")]
        if sel.get("vetoed"):
            lines.append("- The automatic reviewer raised a concern (an opinion, not a check): %s" % sel.get("veto_reason"))
    exp = conversion.get("explanation") or {}
    if exp.get("summary"):
        lines += ["", "**In plain words:** %s" % exp["summary"]]
    return "\n".join(lines)


def _render_clarification_section(clarification_record):
    if not clarification_record:
        return ""
    lines = ["", "## Clarifications Applied"]
    for it in clarification_record:
        question = it.get("question") or it.get("id") or "(question not recorded)"
        answer = it.get("answer_label") or it.get("answer_value") or "(no answer recorded)"
        source = it.get("source") or ("user" if it.get("answered") else "default")
        if source == "description":
            # never actually asked, so show the sentence it was read from
            # instead of pretending the user answered a question
            evidence = (it.get("evidence") or "").strip()
            if evidence:
                lines.append("- %s -> %s  *(not asked, your description already says: \"%s\")*"
                             % (question, answer, evidence))
            else:
                lines.append("- %s -> %s  *(not asked: already stated in your description)*"
                             % (question, answer))
        elif source == "user":
            lines.append("- %s -> %s" % (question, answer))
        else:
            lines.append("- %s -> %s  *(assumed default, not specified by the user)*"
                          % (question, answer))
    return "\n".join(lines)
