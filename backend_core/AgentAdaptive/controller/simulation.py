from math import sin, pi
import numpy as np
import sympy as sp

from .estimators import eval_uncertainty


def desired_output_table(kind, t, amplitude, omega, degree, offset=0.0):
    # offset only shifts the 0th derivative (bias on desired position).
    # a constant has zero derivative so it never shows up in yd1, yd2, etc.
    table = []
    if kind == "step":
        table.append(amplitude + offset)
        for k in range(degree):
            table.append(0.0)
    elif kind == "sine":
        for k in range(degree + 1):
            value = amplitude * (omega**k) * sin(omega*t + k*pi/2)
            table.append(value + offset if k == 0 else value)
    else:
        for k in range(degree + 1):
            value = amplitude * (omega**k) * sin(omega*t + pi/2 + k*pi/2)
            table.append(value + offset if k == 0 else value)
    return table


def simulate(system, controller, ref_orders, refs, x0, dt, t_end,
             adaptive=None, true_delta_func=None, true_dist_func=None,
             structure_cache=None):
    # true_delta_func/true_dist_func inject REAL uncertainty into the plant only,
    # the controller never sees it. adaptive=None gives a plain closed loop, otherwise the estimator steps too each tick.

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

    if adaptive is not None:
        adaptive.reset(x)

    for step in range(steps):
        ref_tables = []
        for i in range(p):
            spec = refs[i]
            ref_tables.append(desired_output_table(spec["kind"], t, spec["amp"], spec["omega"], ref_orders[i], spec.get("offset", 0.0)))

        if adaptive is not None:
            u = adaptive.compute(list(x), ref_tables)
        else:
            u = controller(list(x), ref_tables)

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
                    if "D_hat" in est:
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
