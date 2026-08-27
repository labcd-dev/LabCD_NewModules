from math import comb
import numpy as np
import sympy as sp

from .estimators import saturate, make_rbf, round_floats, AdaptiveSMC


def _build_smc_structure(states, inputs, outputs, f, surface_lambda):
    # pulled out into its own function to be memoizable. it only depends on
    # system + surface_lambda; K, Gamma, kappa, etc. are cheap runtime params applied later

    zero_inputs = {u: 0 for u in inputs}

    rel_degrees = []
    surface_list = []
    drift_list = []
    reference_part_list = []
    decoupling_rows = []
    yd_symbols = []
    compensation_rows = []

    for out_index in range(len(outputs)):
        output_expr = outputs[out_index]

        # differentiate the output until an input finally shows up.
        # however many times that takes is the relative degree
        derivatives = [output_expr]
        relative_degree = 0
        decoupling_row = None
        drift_term = None
        while True:
            current = derivatives[-1]
            derivative = 0
            for i in range(len(states)):
                derivative = derivative + sp.diff(current, states[i]) * f[i]
            derivative = sp.expand(derivative)

            input_present = any(derivative.has(u) for u in inputs)
            relative_degree += 1
            if input_present:
                decoupling_row = [sp.diff(derivative, u) for u in inputs]
                drift_term = derivative.subs(zero_inputs)
                break
            derivatives.append(derivative)

        rel_degrees.append(relative_degree)
        decoupling_rows.append(decoupling_row)

        yd = [sp.Symbol("yd%d_%d" % (out_index, k))
              for k in range(relative_degree + 1)]
        yd_symbols.append(yd)

        # binomial coefficients of (d/dt + lambda)^(r-1), with the leading
        # (highest derivative) coefficient forced to 1: the standard sliding surface
        surface_coeffs = [comb(relative_degree - 1, k)
                          * surface_lambda**(relative_degree - 1 - k)
                          for k in range(relative_degree)]
        surface_coeffs[relative_degree - 1] = 1.0

        surface_expr = 0
        for k in range(relative_degree):
            surface_expr = surface_expr + surface_coeffs[k] * (derivatives[k] - yd[k])
        surface_list.append(sp.expand(surface_expr))

        drift_expr = drift_term
        for k in range(relative_degree - 1):
            drift_expr = drift_expr + surface_coeffs[k] * derivatives[k + 1]
        drift_list.append(sp.expand(drift_expr))

        reference_part = yd[relative_degree]
        for k in range(relative_degree - 1):
            reference_part = reference_part + surface_coeffs[k] * yd[k + 1]
        reference_part_list.append(sp.expand(reference_part))

        # C_i(x) lets Delta_hat reach into surface levels below the top one,
        # so unmatched uncertainty (degree > 1) gets compensated too. degree-1 outputs just get a zero row
        compensation_row = [sp.Integer(0)] * len(states)
        for k in range(1, relative_degree):
            gradient = [sp.diff(derivatives[k - 1], states[j])
                        for j in range(len(states))]
            for j in range(len(states)):
                compensation_row[j] = compensation_row[j] + surface_coeffs[k] * gradient[j]
        compensation_rows.append([sp.expand(e) for e in compensation_row])

    decoupling_matrix = sp.Matrix(decoupling_rows)
    decoupling_matrix_inv = decoupling_matrix.inv()

    yd_flat = [symbol for row in yd_symbols for symbol in row]

    surface_func = sp.lambdify(list(states) + yd_flat, sp.Matrix(surface_list), "numpy")
    drift_func = sp.lambdify(list(states), sp.Matrix(drift_list), "numpy")
    reference_func = sp.lambdify(yd_flat, sp.Matrix(reference_part_list), "numpy")
    decoupling_inv_func = sp.lambdify(list(states), decoupling_matrix_inv, "numpy")
    compensation_func = sp.lambdify(list(states), sp.Matrix(compensation_rows), "numpy")

    return {
        "rel_degrees": rel_degrees,
        "surface_list": surface_list,
        "drift_list": drift_list,
        "reference_part_list": reference_part_list,
        "decoupling_matrix_inv": decoupling_matrix_inv,
        "yd_symbols": yd_symbols,
        "yd_flat": yd_flat,
        "compensation_rows": compensation_rows,
        "surface_func": surface_func,
        "drift_func": drift_func,
        "reference_func": reference_func,
        "decoupling_inv_func": decoupling_inv_func,
        "compensation_func": compensation_func,
    }


def design_smc(system, use_uncertainty_estimation=False,
               surface_lambda=2, K=4, Lam=5, phi_layer=0.05,
               Gamma=25, kappa=5, kappa_s=None, k2=1, k3=1, k4=1, sigma_W=0.1,
               N=25, width=1.5, seed=0,
               rbf_spread=1.0, rbf_normalize="meanstd",
               delta_u_dependent=True,
               use_filtered_error=False,
               lambda_I=0.5,
               estimate_delta=True, estimate_disturbance=True,
               structure_cache=None):

    states = system["states"]
    inputs = system["inputs"]
    outputs = system["outputs"]
    f = system["f"]

    if structure_cache is None:
        struct = _build_smc_structure(states, inputs, outputs, f, surface_lambda)
    else:
        cache_key = ("smc", float(surface_lambda))
        struct = structure_cache.get(cache_key)
        if struct is None:
            struct = _build_smc_structure(states, inputs, outputs, f, surface_lambda)
            structure_cache[cache_key] = struct

    rel_degrees = struct["rel_degrees"]
    surface_list = struct["surface_list"]
    drift_list = struct["drift_list"]
    reference_part_list = struct["reference_part_list"]
    decoupling_matrix_inv = struct["decoupling_matrix_inv"]
    yd_symbols = struct["yd_symbols"]
    yd_flat = struct["yd_flat"]
    compensation_rows = struct["compensation_rows"]
    surface_func = struct["surface_func"]
    drift_func = struct["drift_func"]
    reference_func = struct["reference_func"]
    decoupling_inv_func = struct["decoupling_inv_func"]
    compensation_func = struct["compensation_func"]

    p = len(outputs)

    def controller(x_values, ref_tables):
        yd_now = [value for i in range(p) for value in ref_tables[i]]

        s = np.array(surface_func(*x_values, *yd_now), dtype=float).reshape(p)
        beta = np.array(drift_func(*x_values), dtype=float).reshape(p)
        v = np.array(reference_func(*yd_now), dtype=float).reshape(p)
        Dinv = np.array(decoupling_inv_func(*x_values), dtype=float).reshape(p, p)

        reaching_term = np.array(
            [K * saturate(s[i] / phi_layer) + Lam * s[i] for i in range(p)])
        u = Dinv.dot(v - beta - reaching_term)
        return list(u)

    sat_symbol = sp.Function("sat")
    reaching_symbolic = [K * sat_symbol(surface_list[i] / phi_layer) + Lam * surface_list[i]
                         for i in range(p)]
    u_symbolic = decoupling_matrix_inv * (sp.Matrix(reference_part_list)
                                          - sp.Matrix(drift_list)
                                          - sp.Matrix(reaching_symbolic))

    print("\n--- Sliding Mode Controller ---")
    print("relative degrees of the outputs:", rel_degrees)
    for i in range(p):
        print("sliding surface s%d = %s" % (i + 1, surface_list[i]))
    print("control law form:  u = D(x)^-1 * ( v - beta(x) - K*sat(s/phi) - Lam*s )")
    for i in range(p):
        print("u%d = %s" % (i + 1, round_floats(u_symbolic[i])))

    adaptive = None
    if use_uncertainty_estimation:
        n = len(states)
        m = len(inputs)
        kappa_s_value = kappa if kappa_s is None else kappa_s

        nominal_dynamics_func = sp.lambdify(list(states) + list(inputs),
                                            sp.Matrix(f), "numpy")
        surface_jacobian_expr = sp.Matrix(surface_list).jacobian(sp.Matrix(states))
        surface_jacobian_func = sp.lambdify(list(states) + yd_flat,
                                            surface_jacobian_expr, "numpy")

        basis_dim = (n + m) if delta_u_dependent else n
        basis_func = make_rbf(basis_dim, N, width, seed=seed,
                              spread=rbf_spread, normalize=rbf_normalize)

        adaptive = AdaptiveSMC(n, p, surface_func, drift_func, reference_func,
                               decoupling_inv_func, surface_jacobian_func,
                               nominal_dynamics_func, compensation_func, basis_func,
                               K, Lam, phi_layer, Gamma, sigma_W, N,
                               kappa=kappa, kappa_s=kappa_s_value, k2=k2, k3=k3, k4=k4,
                               delta_u_dependent=delta_u_dependent,
                               estimate_delta=estimate_delta,
                               estimate_disturbance=estimate_disturbance)

        basis_mode = "phi(x,u_k) one-step-delayed" if delta_u_dependent else "phi(x) algebraic"
        print("(1) state-space NN identifier  (%s | Gamma=%g kappa=%g sigma_W=%g "
              "N=%d width=%g spread=%g normalize=%s)"
              % (basis_mode, Gamma, kappa, sigma_W, N, width, rbf_spread, rbf_normalize))
        print("    Delta_hat = W_Delta^T phi(z)  (W_Delta in R^{%d x %d})" % (N, n))
        print("    predictor : x_hat_dot = f(x,u) + Delta_hat + kappa (x - x_hat)")
        print("    update    : W_Delta_dot = Gamma( phi x_tilde^T - sigma_W W_Delta )")
        print("    surface   : sigma = s + C(x) Delta_hat   (unmatched compensation)")
        print("    active = %s" % estimate_delta)
        print("(2) surface-level Disturbance Observer  (kappa_s=%g k2=%g k3=%g k4=%g)"
              % (kappa_s_value, k2, k3, k4))
        print("    D_hat in R^%d  (= dim(u) = dim(y), square-system condition)" % p)
        print("    predictor : s_hat_dot = [drift-v] + J_s Delta_hat + D_hat + kappa_s(s-s_hat)")
        print("    drive     : k2*s - k3*(s_hat-s)")
        print("    update    : D_hat_dot = drive - k4( s_hat_dot - s_dot_real + kappa_s e_D )")
        print("    active = %s" % estimate_disturbance)
        print("control   : u = D^-1( v - beta - (J_s Delta_hat + D_hat)"
              " - K sat(sigma/phi) - Lam sigma )")

    return controller, rel_degrees, adaptive, u_symbolic, yd_symbols
