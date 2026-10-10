import numpy as np
import sympy as sp

from .estimators import eval_uncertainty
from .structure_build import _ref_symbolic_derivatives, _REF_T


def _ref_derivative_funcs(refs, ref_orders, structure_cache=None):
    funcs = []
    for i, degree in enumerate(ref_orders):
        cache_key = ("ref_funcs", refs[i]["expr"], degree)
        cached = structure_cache.get(cache_key) if structure_cache is not None else None
        if cached is None:
            derivs = _ref_symbolic_derivatives(refs[i]["expr"], degree, _REF_T)
            cached = [sp.lambdify(_REF_T, d, "numpy") for d in derivs]
            if structure_cache is not None:
                structure_cache[cache_key] = cached
        funcs.append(cached)
    return funcs


def _pad_diverged(logs, steps, dt, t, time_log):
    # after an early stop, fill every log up to the planned length: time keeps counting, the rest is NaN
    # (so the verdict's "finite" check fails exactly as if the run had gone on)
    missing = steps - len(time_log)
    if missing <= 0:
        return
    for log in logs:
        if not log or log is time_log:
            continue
        last = log[-1]
        filler = np.full(np.shape(last), np.nan)
        for _ in range(missing):
            log.append(filler.tolist() if isinstance(last, list) else filler.copy())
    for _ in range(missing):
        time_log.append(t)
        t = t + dt


def limit_arrays(input_limits, m):
    # per-input [lo, hi] (None = free) -> (lo, hi) arrays, or None when nothing is limited
    lo, hi = np.full(m, -np.inf), np.full(m, np.inf)
    for j, lim in enumerate(input_limits or []):
        if lim is None or j >= m:
            continue
        if lim[0] is not None:
            lo[j] = float(lim[0])
        if lim[1] is not None:
            hi[j] = float(lim[1])
    return (lo, hi) if (np.isfinite(lo).any() or np.isfinite(hi).any()) else None


def simulate(system, controller, ref_orders, refs, x0, dt, t_end,
             adaptive=None, true_delta_func=None, true_dist_func=None,
             structure_cache=None, input_limits=None, extras=None):
    # true_delta_func/true_dist_func inject REAL uncertainty into the plant only,
    # the controller never sees it. adaptive=None gives a plain closed loop, otherwise the estimator steps too each tick.
    # input_limits: the plant gets clip(u_cmd); input_log holds what the plant got, and the raw
    # command goes into extras["u_cmd"] (if a dict is passed) so saturation can be measured

    states = system["states"]
    inputs = system["inputs"]
    outputs = system["outputs"]
    f = system["f"]

    n = len(states)
    p = len(outputs)

    if structure_cache is None:
        f_func = sp.lambdify(list(states) + list(inputs), sp.Matrix(f), "numpy")
        out_funcs = [sp.lambdify(list(states), h, "numpy") for h in outputs]
    else:
        cache_key = ("plant",)
        plant_funcs = structure_cache.get(cache_key)
        if plant_funcs is None:
            f_func = sp.lambdify(list(states) + list(inputs), sp.Matrix(f), "numpy")
            out_funcs = [sp.lambdify(list(states), h, "numpy") for h in outputs]
            structure_cache[cache_key] = (f_func, out_funcs)
        else:
            f_func, out_funcs = plant_funcs

    ref_funcs = _ref_derivative_funcs(refs, ref_orders, structure_cache)

    time_log = []
    output_log = []
    ref_log = []
    input_log = []
    state_log = []

    Delta_hat_log = []
    D_hat_log = []
    Delta_true_log = []
    D_true_log = []
    xc_log = []
    xdf_log = []
    Xi_hat_log = []
    Xi_true_log = []
    D_hat_smc_log = []
    D_true_smc_log = []
    g_hat_log = []
    g_true_log = []

    x = np.array(x0, dtype=float)
    t = 0.0
    steps = int(t_end / dt)

    limits = limit_arrays(input_limits, len(inputs))
    u_cmd_log = []

    if adaptive is not None:
        adaptive.reset(x)

    for step in range(steps):
        ref_tables = [[float(f(t)) for f in ref_funcs[i]] for i in range(p)]

        if adaptive is not None:
            u = adaptive.compute(list(x), ref_tables)
        else:
            u = controller(list(x), ref_tables)

        u_cmd_log.append(list(u))
        if limits is not None:
            u_cmd = np.asarray(u, dtype=float)
            u = [float(v) for v in np.clip(u_cmd, limits[0], limits[1])]
            if adaptive is not None:
                adaptive.apply_input(u_cmd, np.asarray(u, dtype=float))

        time_log.append(t)
        output_log.append([float(out_funcs[i](*x)) for i in range(p)])
        ref_log.append([ref_tables[i][0] for i in range(p)])
        input_log.append(list(u))
        state_log.append(list(x))

        # this is the real plant: uncertainty and disturbance included,
        # none of which the controller ever gets to see
        xdot = np.array(f_func(*x, *u), dtype=float).reshape(n)
        delta_true = eval_uncertainty(true_delta_func, x, u, t, n)
        dist_true = eval_uncertainty(true_dist_func, x, u, t, n)
        xdot = xdot + delta_true + dist_true

        if adaptive is not None:
            adaptive.compute_derivs(xdot)
            est = adaptive.log()
            if "Xi_hat" in est:
                # SMC path: lumped Xi plus whichever sub-estimates are present
                Xi_hat_log.append(np.asarray(est["Xi_hat"], dtype=float).copy())
                Xi_true_log.append(np.asarray(est["Xi_true"], dtype=float).copy())
                if "Delta_hat" in est:
                    Delta_hat_log.append(np.asarray(est["Delta_hat"], dtype=float).copy())
                    Delta_true_log.append(np.asarray(est["Delta_true"], dtype=float).copy())
                if "D_hat" in est:
                    D_hat_smc_log.append(np.asarray(est["D_hat"], dtype=float).copy())
                    D_true_smc_log.append((est["Js"] @ dist_true).copy())
                if "Delta_hat" in est or "D_hat" in est:
                    # lift D_hat into state space via Js pseudo-inverse.
                    # that makes it comparable to Delta_hat, but diagnostic only
                    delta_part = (np.asarray(est["Delta_hat"], dtype=float)
                                  if "Delta_hat" in est else np.zeros(n))
                    if "D_hat" in est and np.all(np.isfinite(est["Js"])):
                        # (skipped for a diverged run: pinv of NaN would crash instead of failing the verdict)
                        Js_pinv = np.linalg.pinv(est["Js"])
                        d_part = Js_pinv @ est["D_hat"]
                    else:
                        d_part = np.zeros(n)
                    g_hat_log.append((delta_part + d_part).copy())
                    g_true_log.append((delta_true + dist_true).copy())
            else:
                # backstepping path: plain per-state Delta/D estimates
                Delta_hat_log.append(np.asarray(est["Delta_hat"], dtype=float).copy())
                D_hat_log.append(np.asarray(est["D_hat"], dtype=float).copy())
                Delta_true_log.append(delta_true.copy())
                D_true_log.append(dist_true.copy())
                if "xc" in est:
                    xc_log.append(np.asarray(est["xc"], dtype=float).copy())
                    xdf_log.append(np.asarray(est["xdf"], dtype=float).copy())

        x = x + dt * xdot
        if adaptive is not None:
            adaptive.step(dt)
        t = t + dt
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e60:
            break                        # diverged: nothing after this can pass, so stop spending time on it

    _pad_diverged([time_log, output_log, ref_log, input_log, state_log, u_cmd_log, Delta_hat_log, D_hat_log,
                   Delta_true_log, D_true_log, xc_log, xdf_log, Xi_hat_log, Xi_true_log, D_hat_smc_log,
                   D_true_smc_log, g_hat_log, g_true_log], steps, dt, t, time_log)

    if extras is not None:
        extras["u_cmd"] = np.array(u_cmd_log)
        extras["input_limits"] = limits

    adaptive_log = None
    if adaptive is not None:
        if Xi_hat_log:
            adaptive_log = {
                "Xi_hat": np.array(Xi_hat_log),
                "Xi_true": np.array(Xi_true_log),
            }
            if Delta_hat_log:
                adaptive_log["Delta_hat"] = np.array(Delta_hat_log)
                adaptive_log["Delta_true"] = np.array(Delta_true_log)
            if D_hat_smc_log:
                adaptive_log["D_hat"] = np.array(D_hat_smc_log)
                adaptive_log["D_true"] = np.array(D_true_smc_log)
            if g_hat_log:
                adaptive_log["g_hat"] = np.array(g_hat_log)
                adaptive_log["g_true"] = np.array(g_true_log)
        else:
            adaptive_log = {
                "Delta_hat": np.array(Delta_hat_log),
                "D_hat": np.array(D_hat_log),
                "Delta_true": np.array(Delta_true_log),
                "D_true": np.array(D_true_log),
            }
            if xc_log:
                adaptive_log["xc"] = np.array(xc_log)
                adaptive_log["xdf"] = np.array(xdf_log)

    return (np.array(time_log), np.array(output_log),
            np.array(ref_log), np.array(input_log), np.array(state_log),
            adaptive_log)
