"""Closed-loop look at a converted plant: do all states stay bounded, and what did the
outputs we stopped tracking do. Works on the series the pipeline already records.
"""

import copy

import numpy as np

_BLOWUP = 1e6
_GROWTH_RATIO = 3.0
_LATE_FRACTION = 0.25


def _stats(column):
    finite = np.isfinite(column)
    good = column[finite]
    if good.size == 0:
        return {"finite": False, "min": None, "max": None, "final": None, "max_abs": None}
    return {"finite": bool(finite.all()), "min": float(good.min()), "max": float(good.max()),
            "final": float(good[-1]), "max_abs": float(np.abs(good).max())}


def simulation_check(series, untracked_outputs=()):
    # empirical only: a state that grows late in the run is flagged, a bounded run proves nothing
    x = ((series or {}).get("channels") or {}).get("x") or {}
    names, data = x.get("names") or [], x.get("data") or []
    if not names or not data:
        return {"available": False}
    matrix = np.array(data, dtype=float)
    cut = int(len(matrix) * (1.0 - _LATE_FRACTION))
    states, diverging = {}, []
    for i, name in enumerate(names):
        col = matrix[:, i]
        info = _stats(col)
        late = np.abs(col[cut:][np.isfinite(col[cut:])])
        early = np.abs(col[:cut][np.isfinite(col[:cut])])
        late_max = float(late.max()) if late.size else 0.0
        early_max = float(early.max()) if early.size else 0.0
        growing = late_max > _GROWTH_RATIO * max(early_max, 1e-9) and late_max > 1.0
        if not info["finite"] or (info["max_abs"] or 0.0) > _BLOWUP or growing:
            diverging.append(name)
        states[name] = info
    return {"available": True, "empirical": True, "n_points": int(len(matrix)),
            "ok": not diverging, "diverging_states": diverging, "states": states,
            "untracked_outputs": {o: states[o] for o in untracked_outputs if o in states}}


def apply_simulation_check(conversion, series):
    # adds the check to the record and drops label A to C if a state looks like it is blowing up
    rec = copy.deepcopy(conversion)
    check = simulation_check(series, rec.get("untracked_outputs") or [])
    rec["simulation_check"] = check
    if check.get("available") and not check["ok"]:
        reason = ("the simulation shows state(s) %s growing or not finite after the conversion."
                  % ", ".join(check["diverging_states"]))
        if reason not in rec["label_reasons"]:
            rec["label_reasons"] = list(rec["label_reasons"]) + [reason]
        rec["label"] = "C"
    return rec
