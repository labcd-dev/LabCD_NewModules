# when the output bound cannot work with the setup (nothing was simulated), python writes the
# whole diagnosis: the problems are facts, and each fix is computed and checked here, no model call
from backend_core.AgentAdaptive.tools import failure_mode, system_spec


def _num(v):
    return failure_mode._fmt(v)


def cards_for(spec, problems, t_end):
    dyn = system_spec.substitute_parameters(spec)["dynamics"]
    states = list(dyn.get("states") or [])
    x0 = [float(v) for v in dyn.get("x0") or []]
    refs = {r.get("output"): r.get("expr", "") for r in dyn.get("references") or []}
    cards = []
    for p in problems:
        name, kind = p["output"], p["kind"]
        if kind == "reference_outside":
            lo, hi = p["bounds"]
            if lo is None or hi is None or not t_end:
                continue
            options = failure_mode._fit(refs.get(name, ""), (lo, hi), t_end)
            if options:
                cards.append({
                    "title": "Keep the reference of %s inside [%s, %s]" % (name, _num(lo), _num(hi)),
                    "detail": "The reference goes from %s to %s." % (_num(p["reference_span"][0]),
                                                                    _num(p["reference_span"][1])),
                    "lever": "reference", "target": name, "computed": True,
                    "options": [{"label": "%d%% of the window" % round(100 * f), "value": v}
                                for f, v in zip(failure_mode._FILL, options)]})
        elif kind in ("start_outside", "start_too_far"):
            if name not in states or len(x0) != len(states):
                continue
            kb = p.get("kb")
            new = p["r0"] if kind == "start_outside" else p["r0"] + (0.5 * kb if p["e0"] >= 0 else -0.5 * kb)
            moved = list(x0)
            moved[states.index(name)] = float("%.4g" % new)
            cards.append({
                "title": "Start %s at %s" % (name, _num(moved[states.index(name)])),
                "detail": ("It now starts at %s, outside its bound." % _num(p["y0"]) if kind == "start_outside"
                           else "It starts %s from its reference; the error must stay below %s."
                                % (_num(abs(p["e0"])), _num(kb))),
                "lever": "initial_condition", "computed": True,
                "options": [{"label": "initial state", "value": ",".join(_num(v) for v in moved)}]})
    return cards[:3]


def diagnosis_for(spec, problems, t_end=None):
    # same shape as a normal diagnosis, so the dialog shows it unchanged
    report = {
        "headline": "The run was not started: the output bound cannot work with this setup.",
        "explanation": [p["message"] for p in problems][:3],
        "suggestions": cards_for(spec, problems, t_end),
        "chart_series": [], "note": None}
    evidence = {"blocked_before_run": True, "output_bound_problems": problems}
    return {"report": report, "evidence": evidence, "chart_data": {}}
