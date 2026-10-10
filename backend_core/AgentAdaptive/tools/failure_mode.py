# python decides WHY a run failed (from computed facts) and the model only explains it.
# classify() returns a mode plus a short chain of factual sentences; limit_card() is the one
# suggestion about input limits that is ever offered, built here with a computed number
import math


def _fmt(v):
    # plain digits for big numbers (627100, not 6.271e+05)
    return "%.0f" % v if abs(v) >= 1000 else "%.4g" % v


def _bound(limit):
    bounds = [abs(v) for v in (limit or []) if v is not None]
    return max(bounds) if bounds else None


def _chain_clipped(il, name_limits, sat):
    clip = "The input is limited to %s; the plant clipped the command %.0f%% of the run" % (
        name_limits, 100.0 * sat["any_input_fraction"])
    if sat.get("first_saturated_at_s") is not None:
        clip += ", first at t=%.3g s" % sat["first_saturated_at_s"]
    return clip + "."


def classify(evidence):
    il = evidence.get("input_limits") or {}
    sat = il.get("saturation") or {}
    clipped = (sat.get("any_input_fraction") or 0.0) >= 0.01
    free = il.get("without_limits")
    spikes = evidence.get("control_spikes")
    warnings = evidence.get("pre_run_warnings") or []
    limits_text = ", ".join("%s in [%s, %s]" % (n, "-inf" if lo is None else _fmt(lo),
                                                 "+inf" if hi is None else _fmt(hi))
                            for n, (lo, hi) in (il.get("limits") or {}).items())
    chain = []
    ob = evidence.get("output_bounds") or {}

    if ob.get("violated"):
        name, st = next(((n, v) for n, v in (ob.get("seen") or {}).items() if v.get("violated")), (None, None))
        if st is not None:
            lo, hi = (ob.get("bounds") or {}).get(name) or (None, None)
            chain.append("The output %s left its allowed range [%s, %s]: it went from %s to %s%s." % (
                name, "-inf" if lo is None else _fmt(lo), "+inf" if hi is None else _fmt(hi),
                _fmt(st["min"]), _fmt(st["max"]),
                (", first outside at t=%s s" % _fmt(ob["first_violation_s"]))
                if ob.get("first_violation_s") is not None else ""))
        if il and clipped:
            chain.append(_chain_clipped(il, limits_text, sat))
        if il and clipped and free and free.get("passes"):
            mode = "bound_violated_by_limit"
            need = ", ".join("%s up to %s (limit %s)" % (n, _fmt(p), _fmt(_bound(il["limits"][n])))
                             for n, p in free["peak_by_input"].items() if n in il["limits"])
            chain.append("Near the edge the barrier asks for a much larger command than the limit allows. "
                         "The same design WITHOUT the input limit stays inside the bound and needs %s." % need)
        else:
            mode = "output_bound_violated"
            if not (il and clipped):
                kb = ", ".join("%s=%s" % (n, _fmt(k)) for n, k in (ob.get("error_limit_kb") or {}).items())
                chain.append("The input was not clipped. The barrier works on an error limit of %s." % kb)
    elif il and clipped:
        chain.append(_chain_clipped(il, limits_text, sat))
        if free and free.get("passes"):
            mode = "limit_too_small"
            need = ", ".join("%s up to %s (limit %s)" % (n, _fmt(p), _fmt(_bound(il["limits"][n])))
                             for n, p in free["peak_by_input"].items() if n in il["limits"])
            chain.append("The same design WITHOUT the limits passes (tracking %.1f%%) and needs %s."
                         % (free.get("tracking_pct") or 0.0, need))
            first_clip = sat.get("first_saturated_at_s")
            if spikes and first_clip is not None and (spikes.get("first_t") or 0) >= first_clip:
                state = ((spikes.get("largest") or [{}])[0]).get("states_then")
                chain.append("After the clipping began the plant left its normal range%s and the "
                             "control law's request exploded: that is a consequence of the "
                             "clipping, not its cause." % (
                                 (" (states at the largest request: %s)" % ", ".join(
                                     "%s=%s" % (k, _fmt(v)) for k, v in state.items())) if state else ""))
        elif free:
            actual = ((evidence.get("metrics") or {}).get("tracking_pct_headline")) or 0.0
            # the tracking figure of a run that failed on spikes ("bounded") is not comparable
            if free.get("failed_check") == "mse_target" and (free.get("tracking_pct") or 0.0) - actual >= 20.0:
                mode = "limit_adds_to_failure"
                chain.append("The same design WITHOUT the limits fails too (tracking %.1f%%, %s), so "
                             "it has another cause as well, but the limits make it much worse "
                             "(tracking %.1f%% with them)." % (
                                 free.get("tracking_pct") or 0.0,
                                 free.get("failed_check") or "failed check", actual))
            else:
                mode = "fails_even_without_limit"
                chain.append("The same design WITHOUT the limits fails too (%s), so the limits are "
                             "not the root cause." % (free.get("failed_check") or "failed check"))
            chain += warnings[:1]
        else:
            mode = "clipped"
    elif il:
        mode = "not_limit_related"
        chain.append("The input limits (%s) were never reached (clipped %.1f%% of the run): the "
                     "failure is not caused by them." % (limits_text, 100.0 * (sat.get("any_input_fraction") or 0.0)))
    elif warnings or spikes:
        mode = "singularity_or_spikes"
        chain += warnings[:1]
    else:
        mode = "tracking_error_only"
    out = {"mode": mode, "chain": chain}
    if mode in ("not_limit_related", "tracking_error_only"):
        # no computed cause here: say what was checked and found absent, so nothing is invented
        metrics = evidence.get("metrics") or {}
        ruled = []
        if il:
            ruled.append("input limits (never reached)")
        if not spikes:
            ruled.append("control-signal spikes (none)")
        if not warnings:
            ruled.append("a predicted singularity crossing (no pre-run warning)")
        if metrics.get("numerically_healthy", True) and not evidence.get("failure_window"):
            ruled.append("numerical blow-up (none)")
        out["ruled_out"] = ruled
    return out


def _round_up_2sf(v):
    if v <= 0:
        return v
    exp = math.floor(math.log10(v)) - 1
    return math.ceil(v / 10 ** exp) * 10 ** exp


def limit_card(evidence):
    # only when the limit itself is the cause (the same design passes without it). None otherwise
    il = evidence.get("input_limits") or {}
    free = il.get("without_limits")
    if (evidence.get("failure_mode") or {}).get("mode") not in ("limit_too_small",
                                                                 "bound_violated_by_limit") or not free:
        return None
    best = None
    for name, peak in (free.get("peak_by_input") or {}).items():
        bound = _bound((il.get("limits") or {}).get(name))
        if bound and peak > bound and (best is None or peak / bound > best[1]):
            best = (name, peak / bound, peak)
    if best is None:
        return None
    name, ratio, peak = best
    new = _round_up_2sf(1.25 * peak)
    lo, hi = il["limits"][name]
    value = ("" if lo is None else _fmt(-new)) + "," + ("" if hi is None else _fmt(new))
    return {"title": "Raise the %s limit to about %s" % (name, _fmt(new)),
            "detail": "Without any limit this design asks for up to %s; the limit is %s.%s"
                      % (_fmt(peak), _fmt(_bound((il["limits"] or {})[name])),
                         " That is far above your limit, so a gentler reference or lower gains are the "
                         "realistic fix, not a bigger limit." if ratio > 5 else ""),
            "lever": "input_limits", "target": name, "computed": True,
            "options": [{"label": "limit for %s" % name, "value": value}]}


_FILL = (0.8, 0.5)


def _fit(expr_text, window, t_end):
    # the reference with its swing scaled about the window's middle so the whole run stays inside
    # it, to 4 figures. None if that cannot be done or checked
    try:
        import numpy as np
        import sympy as sp
        from backend_core.AgentAdaptive.controller import structure_build as sb
        expr = sb._ref_from_expr(expr_text)
        f = sp.lambdify(sb._REF_T, expr, "numpy")
        vals = np.broadcast_to(np.asarray(f(np.linspace(0.0, float(t_end), 2000)), dtype=float), (2000,))
        width = float(vals.max() - vals.min())
        lo, hi = window
        if width <= 0:
            return []
        mid = (lo + hi) / 2.0
        out = []
        for fill in _FILL:
            k = fill * (hi - lo) / width
            new = sp.expand(mid + k * (expr - mid))
            new = new.xreplace({a: sp.Float(float("%.4g" % float(a))) for a in new.atoms(sp.Float)})
            check = np.broadcast_to(np.asarray(sp.lambdify(sb._REF_T, new, "numpy")(
                np.linspace(0.0, float(t_end), 2000)), dtype=float), (2000,))
            if check.min() >= lo - 1e-6 and check.max() <= hi + 1e-6:
                out.append(str(new))
        return out
    except Exception:  # noqa: BLE001
        return []


def reference_card(evidence):
    # when the reference leaves the window the plant analysis computed, the fix is known: the
    # same reference scaled to fit. built and checked here, never guessed by the model
    windows = evidence.get("reference_windows") or {}
    spans = evidence.get("reference_span") or {}
    refs = {r["output"]: r["expr"] for r in evidence.get("reference") or []}
    t_end = (evidence.get("simulation") or {}).get("t_end")
    for out, (lo, hi) in windows.items():
        span, expr = spans.get(out), refs.get(out)
        if not span or not expr or not t_end:
            continue
        if span["min"] >= lo - 1e-6 and span["max"] <= hi + 1e-6:
            continue
        options = _fit(expr, (lo, hi), t_end)
        if options:
            return {"title": "Keep %s inside (%s, %s)" % (out, _fmt(lo), _fmt(hi)),
                    "detail": "The reference goes from %s to %s; the control law is singular outside "
                              "this window." % (_fmt(span["min"]), _fmt(span["max"])),
                    "lever": "reference", "target": out, "computed": True,
                    "options": [{"label": "%d%% of the window" % round(100 * f), "value": v}
                                for f, v in zip(_FILL, options)]}
    return None


def bounds_card(evidence):
    # the run kept the input free but still left its bound: the one computed fix is a wider bound,
    # built from where the output really went. offered only for that mode
    if (evidence.get("failure_mode") or {}).get("mode") != "output_bound_violated":
        return None
    ob = evidence.get("output_bounds") or {}
    for name, st in (ob.get("seen") or {}).items():
        if not st.get("violated"):
            continue
        lo, hi = (ob.get("bounds") or {}).get(name) or (None, None)
        pad = 0.1 * max(st["max"] - st["min"], 1e-9)
        new_lo = None if lo is None else min(lo, st["min"] - pad)
        new_hi = None if hi is None else max(hi, st["max"] + pad)
        value = ("" if new_lo is None else _fmt(new_lo)) + "," + ("" if new_hi is None else _fmt(new_hi))
        return {"title": "Allow %s to go to [%s, %s]" % (name, "-inf" if new_lo is None else _fmt(new_lo),
                                                         "+inf" if new_hi is None else _fmt(new_hi)),
                "detail": "In this run %s reached [%s, %s]." % (name, _fmt(st["min"]), _fmt(st["max"])),
                "lever": "output_bounds", "target": name, "computed": True,
                "options": [{"label": "bound for %s" % name, "value": value}]}
    return None


def headline(evidence):
    # the one-sentence cause, written from the computed mode and numbers so it cannot carry a
    # number that is not in the evidence. None when the evidence has no mode (the model's stays)
    mode = (evidence.get("failure_mode") or {}).get("mode")
    il = evidence.get("input_limits") or {}
    sat = il.get("saturation") or {}
    free = il.get("without_limits") or {}
    metrics = evidence.get("metrics") or {}
    spikes = evidence.get("control_spikes")
    if mode == "bound_violated_by_limit":
        return ("The output left its allowed range because the input limit stopped the controller from "
                "pushing back (the command was clipped %.0f%% of the run); without the limit it stays "
                "inside." % (100.0 * (sat.get("any_input_fraction") or 0.0)))
    if mode == "output_bound_violated":
        ob = evidence.get("output_bounds") or {}
        name = next((n for n, v in (ob.get("seen") or {}).items() if v.get("violated")), None)
        if name:
            st = ob["seen"][name]
            return "The output %s left its allowed range: it reached %s." % (
                name, _fmt(st["max"] if (ob.get("bounds") or {}).get(name, [None, None])[1] is not None
                           and st["max"] > ob["bounds"][name][1] else st["min"]))
    if mode == "limit_too_small":
        return ("The input limit is too small: the command was clipped %.0f%% of the run, and the "
                "same design passes without the limit." % (100.0 * (sat.get("any_input_fraction") or 0.0)))
    if mode == "limit_adds_to_failure":
        return ("The design fails even without the limit, and the limit makes it worse (tracking "
                "%.0f%% with it, %.0f%% without)." % (metrics.get("tracking_pct_headline") or 0.0,
                                                      free.get("tracking_pct") or 0.0))
    if mode == "clipped":
        return "The command was clipped %.0f%% of the run." % (100.0 * (sat.get("any_input_fraction") or 0.0))
    if mode in ("fails_even_without_limit", "singularity_or_spikes"):
        if evidence.get("pre_run_warnings") or spikes:
            peak = ((spikes or {}).get("largest") or [{}])[0].get("u")
            return ("The reference takes the plant through a point where the control law is singular, "
                    "and the command spikes%s." % ((" (largest request %s)" % _fmt(abs(peak))) if peak else ""))
        return "The design fails even without the input limit (%s)." % (free.get("failed_check") or "failed check")
    if mode in ("not_limit_related", "tracking_error_only"):
        frac, target = metrics.get("steady_rms_frac"), metrics.get("success_target_frac")
        if frac and target:
            return ("The worst output follows its reference only %.0f%% in steady state (error %.1f%% "
                    "of the task scale, target %.1f%%)." % (metrics.get("tracking_pct_headline") or 0.0,
                                                           100.0 * max(frac), 100.0 * target))
    return None
