"""
Output bounds (barrier Lyapunov function) helpers. Pure numbers, no simulation.

The barrier lives on the tracking error e = y - r with ONE constant half-width kb per output:
kb = the smallest distance from the reference to a given bound, over the whole run.
Keeping |e| < kb then keeps y inside [lo, hi] (conservative: the error is limited on both sides).
"""
import numpy as np
import sympy as sp

# the weight is evaluated at most this far into the barrier, so a numerical overshoot can't divide by zero
BARRIER_CLIP = 0.98


def weight(value, kb):
    # barrier gain w = kb^2 / (kb^2 - v^2) >= 1, equal to 1 when there is no bound (kb = inf)
    if not np.isfinite(kb):
        return 1.0
    v = min(max(float(value), -BARRIER_CLIP * kb), BARRIER_CLIP * kb)
    return kb * kb / (kb * kb - v * v)


def half_widths(bounds, ref_values):
    # bounds: [[lo, hi] or None per output]; ref_values: (N, p) samples of the reference over the run.
    # returns one kb per output (inf = not bounded). kb <= 0 means the reference touches or leaves its window
    ref = np.asarray(ref_values, dtype=float).reshape(len(ref_values), -1)
    out = np.full(ref.shape[1], np.inf)
    for i, b in enumerate(bounds or []):
        if b is None or i >= ref.shape[1]:
            continue
        lo, hi = b
        gaps = []
        if lo is not None:
            gaps.append(float(np.min(ref[:, i] - lo)))
        if hi is not None:
            gaps.append(float(np.min(hi - ref[:, i])))
        if gaps:
            out[i] = min(gaps)
    return out


def reference_grid(ref_exprs, t_end, n=4001):
    # samples of every reference expression (strings in t) on a uniform grid
    from .structure_build import _ref_from_expr, _REF_T
    t = np.linspace(0.0, float(t_end), n)
    cols = []
    for expr in ref_exprs:
        f = sp.lambdify(_REF_T, _ref_from_expr(expr), "numpy")
        cols.append(np.broadcast_to(np.asarray(f(t), dtype=float), t.shape))
    return t, np.stack(cols, axis=1)


def start_problems(names, bounds, kb, y0, r0):
    # the barrier needs the run to start strictly inside it: |e(0)| < kb. one message per broken output
    out = []
    for i, name in enumerate(names):
        b = (bounds or [None] * len(names))[i]
        if b is None or not np.isfinite(kb[i]):
            continue
        e0 = float(y0[i]) - float(r0[i])
        lo, hi = b
        if (lo is not None and y0[i] <= lo) or (hi is not None and y0[i] >= hi):
            out.append({"output": name, "kind": "start_outside", "y0": float(y0[i]),
                        "r0": float(r0[i]), "bounds": list(b),
                        "message": ("%s starts at %.4g, which is not inside its bound %s."
                                    % (name, y0[i], _fmt_bound(b)))})
        elif abs(e0) >= BARRIER_CLIP * kb[i]:
            out.append({"output": name, "kind": "start_too_far", "y0": float(y0[i]), "r0": float(r0[i]),
                        "e0": e0, "kb": float(kb[i]), "bounds": list(b),
                        "message": ("%s starts %.4g away from its reference, but the error must stay below "
                                    "%.4g (the smallest distance from the reference to the bound %s)."
                                    % (name, abs(e0), kb[i], _fmt_bound(b)))})
    return out


def _fmt_bound(b):
    return "[%s, %s]" % ("-inf" if b[0] is None else "%g" % b[0], "inf" if b[1] is None else "%g" % b[1])


def find_problems(names, bounds, ref, y0):
    # (kb, problems) from sampled references (N, p) and the output values at t = 0
    kb = half_widths(bounds, ref)
    problems = []
    for i, name in enumerate(names):
        if bounds[i] is not None and kb[i] <= 0.0:
            problems.append({
                "output": name, "kind": "reference_outside", "bounds": list(bounds[i]),
                "reference_span": [float(ref[:, i].min()), float(ref[:, i].max())],
                "message": ("The reference for %s goes from %.4g to %.4g, which is not strictly inside "
                            "its bound %s. The barrier needs the whole reference inside."
                            % (name, ref[:, i].min(), ref[:, i].max(), _fmt_bound(bounds[i])))})
    if not problems:
        problems = start_problems(names, bounds, kb, y0, ref[0])
    return kb, problems


class OutputBoundError(ValueError):
    def __init__(self, message, problems=None):
        super().__init__(message)
        self.problems = problems or []


def run_half_widths(bounds, ref_exprs, names, t_end, y0):
    # kb per output for one run, or None when nothing is bounded. raises when the barrier can't start
    if not bounds or not any(b is not None for b in bounds):
        return None
    _, ref = reference_grid(ref_exprs, t_end)
    kb, problems = find_problems(names, list(bounds), ref, y0)
    if problems:
        raise OutputBoundError(" ".join(p["message"] for p in problems), problems)
    return kb


def check_output_bounds(spec, t_end=None, n=4001):
    # python-only check of a confirmed spec, before any design or simulation. [] = the barrier can run.
    # problems: the reference leaves its window, or the start is outside / too far from the reference
    from backend_core.AgentAdaptive.tools import system_spec as ss
    from .structure_build import _parse_system
    try:
        dyn = ss.substitute_parameters(spec)["dynamics"]
        outputs = list(dyn.get("outputs") or [])
        bounds = ss.bounds_per_output(dyn)
        if not outputs or not any(b is not None for b in bounds):
            return []
        by_ref = {r.get("output"): r.get("expr", "") for r in (dyn.get("references") or [])}
        horizon = float(t_end or dyn.get("sim_time") or 8.0)
        _, ref = reference_grid([by_ref.get(o, "") for o in outputs], horizon, n)
        states, _, out_syms, _, _ = _parse_system(
            dyn["states"], dyn["state_equations"], dyn["inputs"], outputs)
        x0 = [float(v) for v in dyn["x0"]]
        y0 = [float(sp.lambdify(list(states), h, "numpy")(*x0)) for h in out_syms]
        return find_problems(outputs, bounds, ref, y0)[1]
    except Exception:  # noqa: BLE001 -- advisory pre-check, never break a run over it
        return []


def error_limits(spec, t_end=None, n=4001):
    # {output: kb} for the bounded outputs of a confirmed spec; {} when none or when it cannot be computed
    from backend_core.AgentAdaptive.tools import system_spec as ss
    try:
        dyn = ss.substitute_parameters(spec)["dynamics"]
        outputs = list(dyn.get("outputs") or [])
        bounds = ss.bounds_per_output(dyn)
        if not outputs or not any(b is not None for b in bounds):
            return {}
        by_ref = {r.get("output"): r.get("expr", "") for r in (dyn.get("references") or [])}
        _, ref = reference_grid([by_ref.get(o, "") for o in outputs], float(t_end or dyn.get("sim_time") or 8.0), n)
        kb = half_widths(bounds, ref)
        return {o: float(k) for o, k in zip(outputs, kb) if np.isfinite(k)}
    except Exception:  # noqa: BLE001 -- only used to word a note
        return {}
