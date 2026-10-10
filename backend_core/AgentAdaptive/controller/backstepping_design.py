import numpy as np
import sympy as sp

from .estimators import make_rbf, round_floats, AdaptiveBackstepping
from .output_bounds import BARRIER_CLIP


def _build_backstepping_structure(states, u, f, c_gains, barrier=False):
    n = len(states)

    yd = []
    for k in range(n + 1):
        yd.append(sp.Symbol("yd%d" % k))

    g_list = []
    f_part = []
    for i in range(n):
        if i < n - 1:
            next_state_input = states[i + 1]
        else:
            next_state_input = u
        g = sp.diff(f[i], next_state_input)
        g_list.append(g)
        f_part.append(sp.expand(f[i] - g * next_state_input))

    f_no_u = [fi.subs(u, 0) for fi in f]

    # chain rule through states + ref derivatives (alpha depends on both),
    # so we get d(alpha)/dt without ever symbolically touching u
    def time_derivative(alpha):
        derivative = 0
        for i in range(n):
            derivative = derivative + sp.diff(alpha, states[i]) * f_no_u[i]
        for k in range(n):
            derivative = derivative + sp.diff(alpha, yd[k]) * yd[k + 1]
        return sp.expand(derivative)

    if c_gains is None:
        c = [2 + 2 * i for i in range(n)]
    else:
        c = list(c_gains)
        if len(c) != n:
            raise ValueError("c_gains must have one gain per state (length %d)" % n)

    # standard backstepping recursion: z_i is tracking error at step i.
    # alpha_i is the virtual control that step wants next; last alpha is u_law.
    z = [None] * n
    alpha = [None] * n

    # output bound: the first step's gain and the coupling into step 2 get the barrier weight
    # w = kb^2/(kb^2 - z0^2). w = 1 without a bound, which gives back the plain law
    kb_sym = sp.Symbol("kb_barrier")
    z[0] = states[0] - yd[0]
    w0 = kb_sym**2 / (kb_sym**2 - z[0]**2) if barrier else 1
    alpha[0] = (-c[0] * w0 * z[0] - f_part[0] + yd[1]) / g_list[0]
    alpha[0] = sp.expand(alpha[0])

    for i in range(1, n):
        z[i] = states[i] - alpha[i - 1]
        d_alpha = time_derivative(alpha[i - 1])
        coupling = g_list[i - 1] * z[i - 1] * (w0 if i == 1 else 1)
        alpha[i] = (-c[i] * z[i] - coupling
                    - f_part[i] + d_alpha) / g_list[i]
        alpha[i] = sp.expand(alpha[i])

    u_law = alpha[n - 1]

    # cse: with an output bound the law has 10^5 operations; shared pieces are computed once per call
    u_func = sp.lambdify(list(states) + yd + ([kb_sym] if barrier else []), u_law, "numpy", cse=True)

    return {
        "g_list": g_list,
        "f_part": f_part,
        "u_law": u_law,
        "u_func": u_func,
        "c": c,
        "yd": yd,
        "barrier": barrier,
    }


def design_backstepping(system, use_uncertainty_estimation=False,
                        c_gains=None,
                        Gamma=25, kappa=5, k2=1, k3=1, k4=1, sigma_W=0.1,
                        tau=0.05, N=25, width=1.5, seed=0,
                        rbf_spread=1.0, rbf_normalize="meanstd",
                        delta_u_dependent=True,
                        use_filtered_error=False,
                        lambda_I=0.5,
                        estimate_delta=True, estimate_disturbance=True,
                        structure_cache=None, output_bound=None):
    # output_bound: half-width kb of the error barrier on y = x1 (None / inf = no bound)
    states = system["states"]
    u = system["inputs"][0]
    f = system["f"]
    n = len(states)

    barrier = output_bound is not None and bool(np.isfinite(output_bound))
    kb = float(output_bound) if barrier else None

    if structure_cache is None:
        struct = _build_backstepping_structure(states, u, f, c_gains, barrier)
    else:
        cache_key = ("backstepping",
                     tuple(c_gains) if c_gains is not None else None, barrier)
        struct = structure_cache.get(cache_key)
        if struct is None:
            struct = _build_backstepping_structure(states, u, f, c_gains, barrier)
            structure_cache[cache_key] = struct

    g_list = struct["g_list"]
    f_part = struct["f_part"]
    u_law = struct["u_law"]
    u_func = struct["u_func"]
    c = struct["c"]
    yd = struct["yd"]

    def controller(x_values, ref_tables):
        yd_now = ref_tables[0]
        if barrier:
            # stretch kb just enough that w stays finite if a step overshoots the barrier
            kb_now = max(kb, abs(x_values[0] - yd_now[0]) / BARRIER_CLIP)
            return [float(u_func(*x_values, *yd_now, kb_now))]
        return [float(u_func(*x_values, *yd_now))]

    print("\n--- Backstepping Controller ---")
    print("output is y = x1, system order n =", n)
    if barrier:
        # the law is far too long to print (printing it alone took seconds)
        print("control law with an output bound: barrier on the error z1 = x1 - yd, kb_barrier = %.6g" % kb)
    else:
        print("control law:  u =", round_floats(u_law))

    # symbolic diff of every virtual control blows up as n grows, so this is
    # command-filtered instead: each step gets its own NN+observer and a filter.
    adaptive = None
    if use_uncertainty_estimation:
        f_part_funcs = [sp.lambdify(list(states), fp, "numpy") for fp in f_part]
        g_funcs = [sp.lambdify(list(states), g, "numpy") for g in g_list]
        if delta_u_dependent:
            rbf_list = [make_rbf((i + 1) + 1, N, width, seed=seed + i,
                                 spread=rbf_spread, normalize=rbf_normalize)
                        for i in range(n)]
        else:
            rbf_list = [make_rbf(i + 1, N, width, seed=seed + i,
                                 spread=rbf_spread, normalize=rbf_normalize)
                        for i in range(n)]
        adaptive = AdaptiveBackstepping(n, f_part_funcs, g_funcs, c, rbf_list, tau,
                                        Gamma, kappa, k2, k3, k4, sigma_W, N,
                                        delta_u_dependent=delta_u_dependent,
                                        use_filtered_error=use_filtered_error,
                                        lambda_I=lambda_I,
                                        estimate_delta=estimate_delta,
                                        estimate_disturbance=estimate_disturbance,
                                        barrier_kb=kb if barrier else None)
        mode = "mu(x,u_k) one-step-delayed/explicit" if delta_u_dependent \
            else "mu(x) explicit/algebraic"
        print("adaptive NN+DO enabled  (command-filtered, %s | tau=%g Gamma=%g "
              "kappa=%g k2=%g k3=%g k4=%g sigma_W=%g N=%d width=%g spread=%g "
              "normalize=%s | use_filtered_error=%s lambda_I=%g)"
              % (mode, tau, Gamma, kappa, k2, k3, k4, sigma_W, N, width,
                 rbf_spread, rbf_normalize, use_filtered_error, lambda_I))
        print("  predicting:  Delta (NN) = %s    disturbance d (observer) = %s"
              % (estimate_delta, estimate_disturbance))

    return controller, [n], adaptive, u_law, yd
