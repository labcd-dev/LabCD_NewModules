from math import comb
import numpy as np
import sympy as sp

from .estimators import saturate, make_rbf, round_floats, AdaptiveSMC
from .output_bounds import BARRIER_CLIP
from .plant_analysis import lie_chain


def _build_smc_structure(states, inputs, outputs, f, surface_lambda, kb=None):
    # pulled out into its own function to be memoizable. it only depends on
    # system + surface_lambda (+ the error limits kb); K, Gamma, kappa, etc. are cheap runtime params
    if kb is not None and np.isfinite(np.asarray(kb, dtype=float)).any():
        return _build_barrier_structure(states, inputs, outputs, f, surface_lambda, kb)

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
        # however many times that takes is the relative degree (capped: a plant
        # whose output no input ever reaches raises instead of looping forever)
        chain = lie_chain(output_expr, states, inputs, f, output_name=output_expr)
        derivatives = chain["derivatives"]
        relative_degree = chain["relative_degree"]
        decoupling_row = chain["decoupling_row"]
        drift_term = chain["drift_term"]

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

    surface_func = sp.lambdify(list(states) + yd_flat, sp.Matrix(surface_list), "numpy", cse=True)
    # every function takes (x, yd, kb...) so the plain and the barrier law are called the same way
    drift_func = sp.lambdify(list(states) + yd_flat, sp.Matrix(drift_list), "numpy", cse=True)
    reference_func = sp.lambdify(list(states) + yd_flat, sp.Matrix(reference_part_list), "numpy", cse=True)
    decoupling_inv_func = sp.lambdify(list(states) + yd_flat, decoupling_matrix_inv, "numpy", cse=True)
    decoupling_func = sp.lambdify(list(states) + yd_flat, decoupling_matrix, "numpy", cse=True)
    compensation_func = sp.lambdify(list(states) + yd_flat, sp.Matrix(compensation_rows), "numpy", cse=True)

    return {
        "kb_symbols": [], "err_func": None,
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
        "decoupling_func": decoupling_func,
        "compensation_func": compensation_func,
    }


def _build_barrier_structure(states, inputs, outputs, f, lam, kb):
    # output bound: the surface is built from the transformed error zeta = kb * atanh(e / kb)
    # (e = kb * tanh(zeta / kb) never leaves (-kb, kb) for any finite zeta). an output with no bound
    # keeps zeta = e. the chain rule is done symbolically on E_k = k-th derivative of e, so the
    # surface, its drift and decoupling all come out of sympy, nothing is derived by hand here
    p = len(outputs)
    kb = np.asarray(kb, dtype=float).reshape(p)
    kb_symbols, rel_degrees, surface_list, comp_rows, yd_symbols, err_list = [], [], [], [], [], []
    for i in range(p):
        chain = lie_chain(outputs[i], states, inputs, f, output_name=outputs[i])
        derivs, rho = chain["derivatives"], chain["relative_degree"]
        rel_degrees.append(rho)
        yd = [sp.Symbol("yd%d_%d" % (i, k)) for k in range(rho + 1)]
        yd_symbols.append(yd)
        E = [sp.Symbol("E%d_%d" % (i, k)) for k in range(rho + 1)]
        if np.isfinite(kb[i]):
            ks = sp.Symbol("kb%d" % i)
            kb_symbols.append(ks)
            zeta = ks * sp.atanh(E[0] / ks)
        else:
            zeta = E[0]

        def dt(expr):
            return sum(sp.diff(expr, E[k]) * E[k + 1] for k in range(rho))

        z = [zeta]
        for _ in range(1, rho):
            z.append(dt(z[-1]))
        S = sum(comb(rho - 1, k) * lam ** (rho - 1 - k) * z[k] for k in range(rho))
        sub = {E[j]: derivs[j] - yd[j] for j in range(rho)}
        surface_list.append(S.subs(sub))
        err_list.append(derivs[0] - yd[0])
        # unmatched compensation: how a state-space correction moves the lower levels of s
        row = [0] * len(states)
        for k in range(1, rho):
            coef = sp.diff(S, E[k]).subs(sub)
            for j in range(len(states)):
                row[j] = row[j] + coef * sp.diff(derivs[k - 1], states[j])
        comp_rows.append(row)

    s_vec = sp.Matrix(surface_list)
    Js = s_vec.jacobian(sp.Matrix(states))
    Jf = Js * sp.Matrix(f)
    zero_u = {u: 0 for u in inputs}
    beta = Jf.subs(zero_u)
    D = Jf.jacobian(sp.Matrix(inputs))
    if D.free_symbols & set(inputs):
        raise ValueError("the output bound needs the input to enter the plant affinely")
    s_t = sp.Matrix([sum(sp.diff(surface_list[i], yd_symbols[i][j]) * yd_symbols[i][j + 1]
                         for j in range(rel_degrees[i])) for i in range(p)])
    v_vec = -s_t
    D_inv = D.inv()

    yd_flat = [x for row in yd_symbols for x in row]
    args_xyd = list(states) + yd_flat + list(kb_symbols)
    lam_ = lambda a, e: sp.lambdify(a, e, "numpy", cse=True)  # noqa: E731
    return {
        "kb_symbols": kb_symbols,
        "err_func": sp.lambdify(list(states) + yd_flat, sp.Matrix(err_list), "numpy"),
        "rel_degrees": rel_degrees,
        "surface_list": surface_list,
        "drift_list": list(beta),
        "reference_part_list": list(v_vec),
        "decoupling_matrix_inv": D_inv,
        "yd_symbols": yd_symbols,
        "yd_flat": yd_flat,
        "compensation_rows": comp_rows,
        "surface_func": lam_(args_xyd, s_vec),
        "drift_func": lam_(args_xyd, beta),
        "reference_func": lam_(args_xyd, v_vec),
        "decoupling_inv_func": lam_(args_xyd, D_inv),
        "decoupling_func": lam_(args_xyd, D),
        "compensation_func": lam_(args_xyd, sp.Matrix(comp_rows)),
    }


class BarrierAux:
    # per step: the error limit actually used (stretched just enough that the weight stays finite if
    # a step overshoots the barrier) and the gain w = kb^2/(kb^2 - e^2) that scales the switching term
    def __init__(self, err_func, kb):
        self.err_func = err_func
        self.kb = np.asarray(kb, dtype=float)

    def args(self, x, yd_flat):
        e = np.array(self.err_func(*x, *yd_flat), dtype=float).reshape(len(self.kb))
        extra, gain = [], np.ones(len(self.kb))
        for i, k in enumerate(self.kb):
            if np.isfinite(k):
                k_now = max(k, abs(e[i]) / BARRIER_CLIP)
                extra.append(k_now)
                gain[i] = k_now ** 2 / (k_now ** 2 - e[i] ** 2)
        return extra, gain


def design_smc(system, use_uncertainty_estimation=False,
               surface_lambda=2, K=4, Lam=5, phi_layer=0.05,
               Gamma=25, kappa=5, kappa_s=None, k2=1, k3=1, k4=1, sigma_W=0.1,
               N=25, width=1.5, seed=0,
               rbf_spread=1.0, rbf_normalize="meanstd",
               delta_u_dependent=True,
               use_filtered_error=False,
               lambda_I=0.5,
               estimate_delta=True, estimate_disturbance=True,
               structure_cache=None, output_bounds_kb=None):
    # output_bounds_kb: half-width kb of the error barrier per output (inf / None = no bound)

    states = system["states"]
    inputs = system["inputs"]
    outputs = system["outputs"]
    f = system["f"]

    kb = None
    if output_bounds_kb is not None and np.isfinite(np.asarray(output_bounds_kb, dtype=float)).any():
        kb = np.asarray(output_bounds_kb, dtype=float)

    if structure_cache is None:
        struct = _build_smc_structure(states, inputs, outputs, f, surface_lambda, kb)
    else:
        cache_key = ("smc", float(surface_lambda), None if kb is None else tuple(float(k) for k in kb))
        struct = structure_cache.get(cache_key)
        if struct is None:
            struct = _build_smc_structure(states, inputs, outputs, f, surface_lambda, kb)
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

    # output bound: the surface is built on the transformed error, and the switching term is scaled
    # by the weight w = kb^2/(kb^2 - e^2). no bound: aux is None and everything below is the plain law
    aux = BarrierAux(struct["err_func"], kb) if kb is not None else None

    def controller(x_values, ref_tables):
        yd_now = [value for i in range(p) for value in ref_tables[i]]
        extra, gain = aux.args(x_values, yd_now) if aux is not None else ([], np.ones(p))

        a = (*x_values, *yd_now, *extra)
        s = np.array(surface_func(*a), dtype=float).reshape(p)
        beta = np.array(drift_func(*a), dtype=float).reshape(p)
        v = np.array(reference_func(*a), dtype=float).reshape(p)
        Dinv = np.array(decoupling_inv_func(*a), dtype=float).reshape(p, p)

        reaching_term = np.array(
            [K * gain[i] * saturate(s[i] / phi_layer) + Lam * s[i] for i in range(p)])
        u = Dinv.dot(v - beta - reaching_term)
        return list(u)

    sat_symbol = sp.Function("sat")
    if aux is None:
        reaching_symbolic = [K * sat_symbol(surface_list[i] / phi_layer) + Lam * surface_list[i]
                             for i in range(p)]
        u_symbolic = decoupling_matrix_inv * (sp.Matrix(reference_part_list)
                                              - sp.Matrix(drift_list)
                                              - sp.Matrix(reaching_symbolic))
    else:
        # the barrier law is too long to print; show its structure with named pieces instead
        s_sym = [sp.Symbol("s_%d" % (i + 1)) for i in range(p)]
        w_sym = [sp.Symbol("w_%d" % (i + 1)) for i in range(p)]
        v_sym = sp.Matrix([sp.Symbol("v_%d" % (i + 1)) for i in range(p)])
        b_sym = sp.Matrix([sp.Symbol("beta_%d" % (i + 1)) for i in range(p)])
        d_sym = sp.Matrix(p, p, lambda i, j: sp.Symbol("Dinv_%d%d" % (i + 1, j + 1)))
        u_symbolic = d_sym * (v_sym - b_sym - sp.Matrix(
            [K * w_sym[i] * sat_symbol(s_sym[i] / phi_layer) + Lam * s_sym[i] for i in range(p)]))

    print("\n--- Sliding Mode Controller ---")
    print("relative degrees of the outputs:", rel_degrees)
    if aux is None:
        for i in range(p):
            print("sliding surface s%d = %s" % (i + 1, surface_list[i]))
        print("control law form:  u = D(x)^-1 * ( v - beta(x) - K*sat(s/phi) - Lam*s )")
        for i in range(p):
            print("u%d = %s" % (i + 1, round_floats(u_symbolic[i])))
    else:
        print("output bound: s = (d/dt + lambda)^(rho-1) zeta, zeta = kb*atanh(e/kb), kb = %s"
              % ", ".join("%.6g" % k for k in kb))
        print("control law form:  u = D(x)^-1 * ( v - beta(x) - K*w*sat(s/phi) - Lam*s ),  "
              "w = kb^2/(kb^2 - e^2)  (D, beta, v, s all from the transformed surface)")

    adaptive = None
    if use_uncertainty_estimation:
        n = len(states)
        m = len(inputs)
        kappa_s_value = kappa if kappa_s is None else kappa_s

        nominal_dynamics_func = sp.lambdify(list(states) + list(inputs),
                                            sp.Matrix(f), "numpy")
        surface_jacobian_expr = sp.Matrix(surface_list).jacobian(sp.Matrix(states))
        surface_jacobian_func = sp.lambdify(list(states) + yd_flat + list(struct["kb_symbols"]),
                                            surface_jacobian_expr, "numpy", cse=True)

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
                               estimate_disturbance=estimate_disturbance,
                               decoupling_func=struct["decoupling_func"],
                               barrier_aux=aux)

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
