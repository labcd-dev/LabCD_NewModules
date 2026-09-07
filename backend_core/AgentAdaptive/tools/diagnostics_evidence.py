# builds evidence for the Diagnoser -- not a classifier. everything here
# just transcribes or computes plain facts, never decides what went wrong.
# that reasoning belongs in diagnoser_agent.py, not an if/else in here

import numpy as np

from backend_core.AgentAdaptive.tools.series_export import build_series
from backend_core.AgentAdaptive.tools import system_spec

# whatever estimator arrays got logged during the sim, turned into a
# magnitude-over-time series each -- no judgment on what the numbers mean
_ESTIMATOR_LOG_KEYS = ("Delta_hat", "Delta_true", "D_hat", "D_true", "Xi_hat", "Xi_true")

# points to keep on either side of a blowup so the model can actually see
# what happened at full resolution instead of some downsampled blur
_FAILURE_WINDOW_HALF_WIDTH = 60


def _round(value, sig=4):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(f):
        return None if np.isnan(f) else ("inf" if f > 0 else "-inf")
    if f == 0:
        return 0.0
    return float(("%.*g") % (sig, f))


def _round_any(value, sig=4):
    if isinstance(value, dict):
        return {k: _round_any(v, sig) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_any(v, sig) for v in value]
    if isinstance(value, bool):
        # bool sneaks in as an int in python -- catch it here or True/False
        # quietly turn into 1.0/0.0 in the evidence
        return value
    if isinstance(value, (int, float, np.floating, np.integer)):
        return _round(value, sig)
    return value


def _safe_list(arr):
    # NaN/Inf can't go in JSON, so None it is -- better to read as a gap
    # than to lie and call it zero
    out = []
    for v in np.asarray(arr, dtype=float).reshape(-1):
        out.append(None if not np.isfinite(v) else _round(v))
    return out


def first_instability_index(x, x0=None):
    # same trick AgentMPC uses for its own divergence check -- just finds
    # where things went non-finite or blew way past their starting size
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None
    norms = np.linalg.norm(np.nan_to_num(x, nan=np.inf, posinf=np.inf, neginf=np.inf), axis=1)
    x0_norm = float(np.linalg.norm(x0)) if x0 is not None else float(norms[0])
    bound = max(1e6, 1000.0 * (x0_norm + 1.0))
    bad = ~np.isfinite(x).all(axis=1) | (norms > bound)
    hits = np.flatnonzero(bad)
    return int(hits[0]) if hits.size else None


def _failure_window(t, states, x, idx, half_width=_FAILURE_WINDOW_HALF_WIDTH):
    if idx is None:
        return None
    lo = max(0, idx - half_width)
    hi = min(len(t), idx + half_width + 1)
    x = np.asarray(x, dtype=float)
    return {
        "note": ("Full time resolution around the point things first looked wrong "
                 "(index %d of %d total steps). The row at 'flagged_index' is the "
                 "first one that was non-finite or far outside a sane range for "
                 "this run -- the row just before it is usually the last "
                 "trustworthy one." % (idx, len(t))),
        "flagged_index": idx - lo,
        "t": _safe_list(t[lo:hi]),
        "states": {name: _safe_list(x[lo:hi, i]) for i, name in enumerate(states)},
    }


def _estimator_vitals(alog, t):
    # collapses whatever's there into one magnitude series per key, same
    # way for SMC or backstepping -- no guessing what a spike means
    vitals = {}
    if not alog:
        return vitals
    for key in _ESTIMATOR_LOG_KEYS:
        arr = alog.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] != len(t):
            continue
        norm = np.linalg.norm(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
        vitals[key] = norm
    return vitals


_ESTIMATOR_LABELS = {
    "Delta_hat": "estimated model-mismatch magnitude (NN estimator output)",
    "Delta_true": "actual injected model-mismatch magnitude (ground truth, controller never sees this)",
    "D_hat": "estimated disturbance magnitude (disturbance observer output)",
    "D_true": "actual injected disturbance magnitude (ground truth, controller never sees this)",
    "Xi_hat": "estimated lumped uncertainty magnitude (SMC surface-level compensation)",
    "Xi_true": "actual lumped uncertainty magnitude (ground truth, controller never sees this)",
}


def build_evidence(context, metrics, state_meanings=None, assumptions=None, parameters=None):
    # turns the raw context + metrics into what actually gets sent to the
    # LLM. assumptions/parameters need to come from before substitution
    # wiped the names, otherwise a "1.0" in the equations means nothing
    method = context["method"]
    states, inputs, outputs = context["states"], context["inputs"], context["outputs"]
    t, y, ref, u, x = context["t"], context["y"], context["ref"], context["u"], context["x"]
    alog = context.get("alog")

    meanings = state_meanings or {}
    variable_meanings = {name: meanings.get(name, name) for name in states}

    series = build_series(t, y, ref, u, x, dt=context.get("dt"), t_end=context.get("t_end"),
                          output_names=outputs, input_names=inputs, state_names=states)

    blowup_idx = first_instability_index(x, x0=context.get("x0"))
    failure_window = _failure_window(t, states, x, blowup_idx)

    vitals = _estimator_vitals(alog, t)

    chart_data = {}
    available_charts = {}

    def _column(rows, i):
        # build_series stores this time-major (one row per timestep, all
        # channels in it) -- this just pulls one channel's series back out
        return [(row[i] if row is not None and i < len(row) else None) for row in rows]

    x_data = series["channels"]["x"]["data"]
    u_data = series["channels"]["u"]["data"]
    y_data = series["channels"]["y"]["data"]
    ref_data = series["channels"]["ref"]["data"]

    for i, name in enumerate(states):
        key = "state:%s" % name
        chart_data[key] = {"t": series["channels"]["t"]["data"], "lines": {name: _column(x_data, i)}}
        available_charts[key] = "state %r (%s) over time" % (name, variable_meanings.get(name, name))

    for i, name in enumerate(inputs):
        key = "input:%s" % name
        chart_data[key] = {"t": series["channels"]["t"]["data"], "lines": {name: _column(u_data, i)}}
        available_charts[key] = "control input %r over time" % name

    for i, name in enumerate(outputs):
        key = "output:%s" % name
        y_line = _column(y_data, i)
        ref_line = _column(ref_data, i)
        chart_data[key] = {"t": series["channels"]["t"]["data"],
                           "lines": {"%s (actual)" % name: y_line, "%s (reference)" % name: ref_line}}
        available_charts[key] = "output %r vs its reference, over time" % name

        err_key = "tracking_error:%s" % name
        err = [None if (a is None or b is None) else (a - b) for a, b in zip(y_line, ref_line)]
        chart_data[err_key] = {"t": series["channels"]["t"]["data"], "lines": {"error": err}}
        available_charts[err_key] = "tracking error (output %r minus its reference) over time" % name

    for key, arr in vitals.items():
        chart_key = "estimator:%s" % key
        chart_data[chart_key] = {"t": _safe_list(t), "lines": {key: _safe_list(arr)}}
        available_charts[chart_key] = _ESTIMATOR_LABELS.get(key, key)

    if failure_window is not None:
        chart_data["failure_window"] = {"t": failure_window["t"], "lines": failure_window["states"]}
        available_charts["failure_window"] = (
            "every state, at full time resolution, zoomed in on the moments "
            "right around where the run first looked unhealthy")

    evidence = {
        "method": "SMC" if method == "smc" else "Backstepping",
        "plant": {
            "state_equations": context["system_text"],
            "states": list(states),
            "variable_meanings": variable_meanings,
            "inputs": list(inputs),
            "x0": _round_any(context.get("x0")),
            # real parameter values and the assumptions behind them, grabbed
            # before they got baked into bare numbers in the equations
            "parameters": _round_any(dict(parameters or {})),
            "assumptions": list(assumptions or []),
        },
        "controller": {
            "control_law": context["control_law_text"],
            "tuning_parameters_used": _round_any(context.get("tuning")),
        },
        "reference": [{"output": r["output"], "expr": r["expr"]} for r in context.get("refs", [])],
        "simulation": {
            "dt": context.get("dt"),
            "t_end": context.get("t_end"),
            "n_steps": len(t),
        },
        "metrics": _round_any(metrics),
        # what the UI can actually fill in for a suggestion's concrete options --
        # step_time in particular is a fixed dropdown, not free text, so a value
        # outside this list can't be applied even if it'd be a good idea
        "editable_options": {
            "outputs": list(outputs),
            "step_time_choices": list(system_spec.SOLVER_STEP_PRESETS),
        },
        "estimator_vitals_available": sorted(vitals.keys()),
        "failure_window": (
            {"note": failure_window["note"], "flagged_index": failure_window["flagged_index"],
             "t": _round_any(failure_window["t"]),
             "states": _round_any(failure_window["states"])}
            if failure_window is not None else None
        ),
        "available_charts": available_charts,
    }
    return evidence, chart_data
