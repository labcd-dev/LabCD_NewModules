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
        "| refs | %s |" % args.get("refs"),
        "| x0 | %s |" % args.get("x0"),
        "",
    ]
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
