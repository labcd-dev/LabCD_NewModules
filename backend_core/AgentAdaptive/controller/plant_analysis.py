"""Static checks on a plant before design: relative degrees, decoupling matrix, chain shape.
Anything sampled around x0 is evidence, not proof. lie_chain is shared with the SMC builder.
"""

import numpy as np
import sympy as sp

from .structure_build import _parse_system
from backend_core.AgentAdaptive.tools.progress import _emit
from backend_core.AgentAdaptive.tools import system_spec as _system_spec

MAX_RELATIVE_DEGREE = 6

# a singular value below this (relative to the largest one) counts as singular
_SINGULAR_REL_TOL = 1e-8
# transmission zeros beyond this magnitude are numerical artifacts of the pencil
_ZERO_MAGNITUDE_CAP = 1e6
_EQUILIBRIUM_TOL = 1e-6


class RelativeDegreeError(ValueError):
    pass


def lie_chain(output_expr, states, inputs, f, max_rd=MAX_RELATIVE_DEGREE,
              output_name=None):
    # derivatives excludes the one that finally contains u; the drift is that
    # one with u set to 0. Gives up with RelativeDegreeError after max_rd tries.
    zero_inputs = {u: 0 for u in inputs}
    derivatives = [output_expr]
    relative_degree = 0
    while relative_degree < max_rd:
        current = derivatives[-1]
        derivative = 0
        for i in range(len(states)):
            derivative = derivative + sp.diff(current, states[i]) * f[i]
        derivative = sp.expand(derivative)
        relative_degree += 1
        if any(derivative.has(u) for u in inputs):
            return {
                "derivatives": derivatives,
                "relative_degree": relative_degree,
                "decoupling_row": [sp.diff(derivative, u) for u in inputs],
                "drift_term": derivative.subs(zero_inputs),
            }
        derivatives.append(derivative)
    name = str(output_name if output_name is not None else output_expr)
    raise RelativeDegreeError(
        "output %s is not influenced by any input after %d successive "
        "derivatives: the plant is not controllable from this output, or its "
        "relative degree is undefined or larger than %d."
        % (name, max_rd, max_rd))


def _finite_array(value):
    try:
        arr = np.array(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return arr if np.all(np.isfinite(arr)) else None


def _sample_points(x0, n_samples, halfwidth, seed):
    rng = np.random.default_rng(seed)
    x0 = np.array(x0, dtype=float)
    points = [x0, np.zeros_like(x0)]
    for _ in range(int(n_samples)):
        points.append(x0 + rng.uniform(-halfwidth, halfwidth, size=x0.shape))
    return points


_SIMPLIFY_OPS_LIMIT = 300


def _is_zero(expr):
    expr = sp.sympify(expr)
    if expr == 0:
        return True
    if sp.count_ops(expr) > _SIMPLIFY_OPS_LIMIT:
        # simplify() can take very long on big expressions: probe numerically instead
        symbols = sorted(expr.free_symbols, key=str)
        rng = np.random.default_rng(0)
        fun = sp.lambdify(symbols, expr, "numpy")
        for _ in range(5):
            val = _finite_array(fun(*rng.uniform(-1.0, 1.0, size=len(symbols))))
            if val is None or np.any(np.abs(val) > 1e-12):
                return False
        return True
    try:
        return sp.simplify(expr) == 0
    except Exception:
        return False


def _chain_violations(order, f, states, inputs, out_sym):
    # row k may only look at states up to k+1, be linear in state k+1, and u
    # may only show up in the last row. Returns the broken rules as sentences.
    n = len(order)
    pos = {states[idx]: k for k, idx in enumerate(order)}
    input_set = set(inputs)
    violations = []
    if states[order[0]] != out_sym:
        violations.append("the output %s is not the first state in the chain." % out_sym)
    for k, idx in enumerate(order):
        name = str(states[idx])
        row = f[idx]
        free = row.free_symbols
        ahead = sorted(str(s) for s in free if s in pos and pos[s] > k + 1)
        if ahead:
            violations.append(
                "the equation of %s depends on %s, which is more than one step "
                "ahead in the chain." % (name, ", ".join(ahead)))
        if k < n - 1:
            used_inputs = sorted(str(s) for s in free if s in input_set)
            if used_inputs:
                violations.append(
                    "the input %s appears in the equation of %s; backstepping "
                    "needs the input only in the last equation."
                    % (", ".join(used_inputs), name))
            nxt = states[order[k + 1]]
            g = sp.diff(row, nxt)
            if _is_zero(g):
                violations.append(
                    "the equation of %s does not depend on the next state %s, "
                    "so the chain is broken." % (name, nxt))
            elif not _is_zero(sp.diff(g, nxt)):
                violations.append(
                    "the equation of %s is not linear in the next state %s."
                    % (name, nxt))
        else:
            used_inputs = [s for s in free if s in input_set]
            if not used_inputs:
                violations.append("the last equation (%s) does not contain the input." % name)
            else:
                g = sp.diff(row, used_inputs[0])
                if g.free_symbols & input_set:
                    violations.append(
                        "the last equation (%s) is not linear in the input." % name)
                elif _is_zero(g):
                    violations.append(
                        "the last equation (%s) does not depend on the input." % name)
    return violations


def _find_chain_order(f, states, out_idx):
    # each row must bring in exactly one new state, so the order is forced
    # once you fix the output; no search needed
    order, used = [out_idx], {out_idx}
    while len(order) < len(states):
        free = f[order[-1]].free_symbols
        deps = [i for i, s in enumerate(states) if s in free and i not in used]
        if len(deps) != 1:
            return None
        order.append(deps[0])
        used.add(deps[0])
    return order


def _chain_g_exprs(order, f, states, inputs):
    g = []
    for k, idx in enumerate(order):
        target = states[order[k + 1]] if k < len(order) - 1 else inputs[0]
        g.append(sp.diff(f[idx], target))
    return g


def _frozen_zeros(f, states, inputs, out_idx, point):
    # zeros of the linearization frozen at `point` with u=0: the finite
    # eigenvalues of [[A,B],[C,0]] - lambda*[[I,0],[0,0]]. Only a local hint.
    try:
        from scipy import linalg as sla
    except ImportError:
        return None, None
    n, m = len(states), len(inputs)
    fvec = sp.Matrix(f)
    args = list(states) + list(inputs)
    a_fun = sp.lambdify(args, fvec.jacobian(states), "numpy")
    b_fun = sp.lambdify(args, fvec.jacobian(inputs), "numpy")
    f_fun = sp.lambdify(args, fvec, "numpy")
    values = list(point) + [0.0] * m
    A = _finite_array(a_fun(*values))
    B = _finite_array(b_fun(*values))
    f0 = _finite_array(f_fun(*values))
    if A is None or B is None or f0 is None:
        return None, None
    p = len(out_idx)
    C = np.zeros((p, n))
    for row, idx in enumerate(out_idx):
        C[row, idx] = 1.0
    top = np.hstack([A, B])
    bottom = np.hstack([C, np.zeros((p, m))])
    M = np.vstack([top, bottom])
    N = np.zeros_like(M)
    N[:n, :n] = np.eye(n)
    try:
        values_ = sla.eigvals(M, N)
    except Exception:
        return None, None
    finite = [complex(v) for v in values_
              if np.isfinite(v) and abs(v) < _ZERO_MAGNITUDE_CAP]
    return finite, bool(np.linalg.norm(f0) < _EQUILIBRIUM_TOL)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _finding(check, severity, message):
    return {"check": check, "severity": severity, "message": message}


def _cannot_analyze(message, dims=None):
    return _jsonable({
        "status": "cannot_analyze",
        "summary": "The plant could not be analyzed: %s" % message,
        "dims": dims or {},
        "candidates": [],
        "designer_default": None,
        "findings": [_finding("parse", "blocking", message)],
        "conversion_hints": [],
    })


def analyze_plant(dyn, *, n_samples=30, halfwidth=1.0, seed=0, on_event=None):
    # takes the `dynamics` block with parameters already baked in (analyze_spec
    # does that). status is "fits", "does_not_fit" or "cannot_analyze".
    states_n = list(dyn.get("states") or [])
    inputs_n = list(dyn.get("inputs") or [])
    outputs_n = list(dyn.get("outputs") or [])
    equations = list(dyn.get("state_equations") or [])
    n, m, p = len(states_n), len(inputs_n), len(outputs_n)
    dims = {"n_states": n, "n_inputs": m, "n_outputs": p}

    def done(report):
        _emit(on_event, kind="note", stage="design",
              text="Plant compatibility check: %s" % report["summary"])
        return report

    if n == 0 or len(equations) != n:
        return done(_cannot_analyze(
            "the plant needs one state equation per state (got %d state(s), %d equation(s))."
            % (n, len(equations)), dims))
    if m == 0 or p == 0:
        return done(_cannot_analyze(
            "the plant has %d input(s) and %d output(s); both are needed." % (m, p), dims))

    unknown_outputs = [o for o in outputs_n if o not in states_n]
    if unknown_outputs:
        return done(_cannot_analyze(
            "output %s is not one of the states; the module accepts only state "
            "names as outputs." % ", ".join(unknown_outputs), dims))

    try:
        states, inputs, outputs, f, symbol_map = _parse_system(
            states_n, equations, inputs_n, outputs_n)
    except Exception as e:  # noqa: BLE001
        return done(_cannot_analyze("%s: %s" % (type(e).__name__, e), dims))

    known = set(states) | set(inputs)
    extra = sorted({str(s) for expr in f for s in expr.free_symbols if s not in known})
    if extra:
        return done(_cannot_analyze(
            "the equations contain undefined symbol(s): %s (parameters must be "
            "substituted first)." % ", ".join(extra), dims))

    out_idx = []
    for o in outputs:
        if o not in states:
            return done(_cannot_analyze(
                "output %s is not one of the states; the module accepts only "
                "state names as outputs." % o, dims))
        out_idx.append(states.index(o))

    x0 = dyn.get("x0") or []
    x0 = [float(v) for v in x0] if len(x0) == n else [0.0] * n
    points = _sample_points(x0, n_samples, halfwidth, seed)
    n_points = len(points)

    findings = []
    smc_reasons, bs_reasons = [], []
    hints = []

    # u inside a nonlinear term (sin(u), u1*u2) breaks the decoupling matrix
    nonaffine = sorted({
        "%s (in the equation of %s)" % (u, states[i])
        for i in range(n) for u in inputs
        if sp.diff(f[i], u).free_symbols & set(inputs)})
    affine = not nonaffine
    if not affine:
        msg = ("the equations are not linear in the input: %s. SMC and "
               "backstepping both need the input to enter linearly."
               % "; ".join(nonaffine))
        findings.append(_finding("C1", "blocking", msg))
        smc_reasons.append(msg)

    square = (m == p)
    if not square:
        msg = ("the plant has %d input(s) and %d output(s). SMC needs equal "
               "numbers of inputs and outputs." % (m, p))
        findings.append(_finding("C2", "blocking", msg))
        smc_reasons.append(msg)
    if not (m == 1 and p == 1):
        bs_reasons.append(
            "backstepping needs exactly one input and one output (this plant has "
            "%d input(s) and %d output(s))." % (m, p))

    rel_degrees, decoupling_rows, rd_errors = [], [], []
    for out_sym in outputs:
        try:
            chain = lie_chain(out_sym, states, inputs, f, output_name=out_sym)
            rel_degrees.append(chain["relative_degree"])
            decoupling_rows.append(chain["decoupling_row"])
        except RelativeDegreeError as e:
            rel_degrees.append(None)
            decoupling_rows.append(None)
            rd_errors.append(str(e))
    rho_finite = all(r is not None for r in rel_degrees)
    for msg in rd_errors:
        findings.append(_finding("C3", "blocking", msg))
        smc_reasons.append(msg)
        hints.append({"id": "T3", "reason": "re-select the outputs: " + msg})

    # sum(rho) < n means the leftover states are internal dynamics the surface never sees
    sum_rho = sum(rel_degrees) if rho_finite else None
    # leftover states only make sense when each input drives one output
    hidden_dim = (n - sum_rho) if (rho_finite and square) else None
    if hidden_dim and hidden_dim > 0:
        findings.append(_finding(
            "C5", "warning",
            "the output dynamics use %d of %d states (relative degrees %s), so %d "
            "state(s) are hidden internal dynamics. The controller inverts the "
            "output dynamics and does not control these states directly; the "
            "module has no stability check for them." % (sum_rho, n, rel_degrees, hidden_dim)))
    elif hidden_dim is not None and hidden_dim < 0:
        findings.append(_finding(
            "C5", "info",
            "the relative degrees %s add up to %d, more than the %d states."
            % (rel_degrees, sum_rho, n)))

    # D(x) is only tested at x0, the origin and random points around x0
    decoupling = {"computed": False, "empirical": True, "n_points": n_points}
    if rho_finite and affine:
        d_expr = sp.Matrix(decoupling_rows)
        d_fun = sp.lambdify(list(states), d_expr, "numpy")
        min_sigma, bad, ranks, x0_bad = None, 0, [], False
        for k, pt in enumerate(points):
            arr = _finite_array(d_fun(*pt))
            if arr is None:
                bad += 1
                x0_bad = x0_bad or k == 0
                ranks.append(0)
                continue
            s = np.linalg.svd(arr, compute_uv=False)
            smax = float(s.max()) if s.size else 0.0
            ranks.append(int(np.sum(s > _SINGULAR_REL_TOL * max(1.0, smax))))
            smin = float(s.min()) if s.size else 0.0
            if square:
                min_sigma = smin if min_sigma is None else min(min_sigma, smin)
                if smin <= _SINGULAR_REL_TOL * max(1.0, smax):
                    bad += 1
                    x0_bad = x0_bad or k == 0
        decoupling.update({
            "computed": True,
            "shape": [p, m],
            "min_sigma": min_sigma,
            "min_rank": min(ranks) if ranks else None,
            "singular_points": bad if square else None,
        })
        if square and bad:
            # singular at x0 or on most of the region is fatal; a few isolated points are a hazard to avoid
            fatal = x0_bad or bad * 2 >= n_points
            msg = ("the decoupling matrix is singular at %d of %d sampled operating "
                   "points (smallest singular value %.2e). SMC inverts this matrix, "
                   "so the control would blow up there. (Empirical: sampled points "
                   "around the initial state.)" % (bad, n_points, min_sigma))
            if fatal:
                findings.append(_finding("C4", "blocking", msg))
                smc_reasons.append(msg)
                hints.append({"id": "T4", "reason": "dynamic extension may restore an invertible decoupling matrix."})
            else:
                findings.append(_finding(
                    "C4", "warning", msg + " The initial state is fine; keep the trajectory away from those points."))
        if not square and m > p and decoupling["min_rank"] == p:
            hints.append({"id": "T1", "reason": "more inputs than outputs: split the commands across the inputs."})
        elif not square and m > p:
            findings.append(_finding(
                "C4", "warning",
                "the decoupling matrix loses rank at some sampled points, so "
                "splitting commands across inputs may not work everywhere."))
    if not square and m < p:
        hints.append({"id": "T2", "reason": "fewer inputs than outputs: choose which outputs to track."})

    # zeros only matter when there are hidden states
    zeros_report = {"computed": False, "empirical": True}
    if square and affine and rho_finite and hidden_dim and hidden_dim > 0:
        checked = []
        for pt in {tuple(points[0]), tuple(points[1])}:
            zs, is_eq = _frozen_zeros(f, states, inputs, out_idx, pt)
            if zs is None:
                continue
            checked.append({"point": list(pt), "is_equilibrium": is_eq,
                            "zeros": [[z.real, z.imag] for z in zs]})
            unstable = [z for z in zs if z.real > 1e-9]
            if unstable:
                msg = ("the linearization frozen at %s has a transmission zero in the "
                       "right half-plane (%s): the hidden dynamics may diverge while "
                       "the outputs look fine. Local, indicative estimate%s."
                       % ([float(v) for v in np.round(pt, 4)],
                          ", ".join("%.3g%+.3gj" % (z.real, z.imag) for z in unstable),
                          "" if is_eq else "; the point is not an equilibrium"))
                findings.append(_finding("C6", "warning", msg))
                hints.append({"id": "T3", "reason": "unstable hidden dynamics: re-select the outputs."})
        zeros_report = {"computed": bool(checked), "empirical": True, "points": checked}

    # backstepping is SISO only; if the given order fails, check whether
    # just reordering the states would give a chain
    chain = {"applicable": bool(m == 1 and p == 1)}
    if chain["applicable"]:
        out_sym = outputs[0]
        given = list(range(n))
        violations = _chain_violations(given, f, states, inputs, out_sym)
        chain["given_order_ok"] = not violations
        chain["violations"] = violations
        relabel = None
        if violations:
            order = _find_chain_order(f, states, out_idx[0])
            if order is not None and not _chain_violations(order, f, states, inputs, out_sym):
                relabel = [str(states[i]) for i in order]
        chain["relabel_order"] = relabel
        if violations:
            bs_reasons.extend(violations)
            if relabel:
                bs_reasons.append(
                    "reordering the states as %s would make it a strict-feedback chain."
                    % relabel)
                hints.append({"id": "T0", "reason": "relabel the states as %s." % relabel})
        else:
            g_exprs = _chain_g_exprs(given, f, states, inputs)
            g_fun = sp.lambdify(list(states), sp.Matrix(g_exprs), "numpy")
            zero_points = 0
            for pt in points:
                arr = _finite_array(g_fun(*pt))
                if arr is None or np.any(np.abs(arr) < _SINGULAR_REL_TOL):
                    zero_points += 1
            if zero_points:
                findings.append(_finding(
                    "C7", "warning",
                    "a chain gain g_i vanishes (or is not finite) at %d of %d sampled "
                    "operating points; backstepping divides by it. (Empirical.)"
                    % (zero_points, n_points)))
            chain["g_vanishing_points"] = zero_points
    else:
        chain["given_order_ok"] = False
        chain["violations"] = []
        chain["relabel_order"] = None

    # optional: only useful when someone wants backstepping on a plant SMC already accepts
    if (chain["applicable"] and affine and not chain.get("given_order_ok") and not chain.get("relabel_order")
            and rho_finite and sum_rho == n):
        hints.append({"id": "T5", "reason": "Lie-derivative coordinates would make it a strict-feedback chain."})

    if (chain["applicable"] and affine and not chain.get("given_order_ok") and not chain.get("relabel_order")):
        hints.append({"id": "T6", "reason": "experimental: terms that break the chain could be treated as uncertainty."})

    smc_viable = not smc_reasons
    bs_viable = chain["applicable"] and chain.get("given_order_ok", False) and affine
    if chain["applicable"] and not affine and not bs_reasons:
        bs_reasons.append("the equations are not linear in the input.")

    candidates = [name for name, ok in (("smc", smc_viable), ("backstepping", bs_viable)) if ok]
    # a SISO chain goes to backstepping even though SMC would work too (designer's tie-break)
    designer_default = ("backstepping" if bs_viable else ("smc" if smc_viable else None))
    status = "fits" if candidates else "does_not_fit"

    if status == "fits":
        summary = ("The plant fits the Adaptive module (available: %s; default: %s)."
                   % (", ".join(candidates), designer_default))
        if any(f_["severity"] == "warning" for f_ in findings):
            summary += " There are warnings to review."
    else:
        first = (smc_reasons or [""])[0]
        summary = ("The plant does not fit the Adaptive module. SMC: %s Backstepping: %s"
                   % (first or "not applicable.", (bs_reasons or ["not applicable."])[0]))

    report = {
        "status": status,
        "summary": summary,
        "dims": dims,
        "affine": affine,
        "relative_degrees": rel_degrees,
        "sum_relative_degrees": sum_rho,
        "hidden_dimension": hidden_dim,
        "decoupling": decoupling,
        "internal_dynamics": zeros_report,
        "chain": chain,
        "methods": {
            "smc": {"viable": smc_viable, "reasons": smc_reasons},
            "backstepping": {"viable": bs_viable, "reasons": bs_reasons},
        },
        "candidates": candidates,
        "designer_default": designer_default,
        "findings": findings,
        "conversion_hints": hints,
        "assumptions": [
            "the full state is measured.",
            "the references and their derivatives up to the relative degree are known exactly.",
            "numbers marked empirical come from %d sampled operating points around the "
            "initial state (half-width %g); they are evidence, not proof." % (n_points, halfwidth),
        ],
        "meta": {"n_samples": int(n_samples), "halfwidth": float(halfwidth),
                 "seed": int(seed), "max_relative_degree": MAX_RELATIVE_DEGREE},
    }
    return done(_jsonable(report))


def analyze_spec(spec, **kwargs):
    substituted = _system_spec.substitute_parameters(spec)
    return analyze_plant(substituted["dynamics"], **kwargs)


analyze = analyze_spec
