"""Exact plant rewrites for plants the module rejects: reorder states, split inputs, pick outputs.
apply_* returns the converted spec, a record of what changed and a fresh analysis of the result;
it raises ConversionError instead if the converted plant would not fit.
"""

import copy
import itertools

import numpy as np
import sympy as sp

from .plant_analysis import (
    analyze_plant, analyze_spec, lie_chain, _finite_array, _sample_points,
)
from .structure_build import _parse_system, _sympify
from backend_core.AgentAdaptive.tools import system_spec as _system_spec
from backend_core.AgentAdaptive.tools.progress import _emit

# the symbolic inverse of D*W^-1*D^T gets huge quickly
_MAX_SQUARING_OUTPUTS = 4
_MAX_SELECTION_COMBINATIONS = 50
_SIMPLIFY_OPS_LIMIT = 150
_EXACTNESS_TOL = 1e-8


class ConversionError(ValueError):
    pass


def _prepare(spec):
    base = copy.deepcopy(_system_spec.substitute_parameters(spec))
    return base, base["dynamics"]


def _expr_str(expr):
    if sp.count_ops(expr) <= _SIMPLIFY_OPS_LIMIT:
        expr = sp.simplify(expr)
    return str(expr)


def _uncertainty_uses_inputs(dyn):
    symbols = {name: sp.Symbol(name, real=True)
               for name in list(dyn["states"]) + list(dyn["inputs"])}
    inputs = {symbols[name] for name in dyn["inputs"]}
    for entry in dyn.get("uncertainty") or []:
        text = (entry.get("expr") or "").strip()
        if not text:
            continue
        try:
            if _sympify(text, symbols).free_symbols & inputs:
                return True
        except Exception:
            return True
    return False


def _label(dyn, analysis):
    reasons = []
    if _uncertainty_uses_inputs(dyn):
        reasons.append("the uncertainty depends on the input, which the stability proofs do not cover.")
    hidden = analysis.get("hidden_dimension")
    if hidden:
        # the SMC proofs cover the output surfaces only, not the leftover states
        reasons.append("%d hidden state(s) are only checked locally; the stability proofs do not cover them." % hidden)
    if any(f["check"] == "C6" and f["severity"] == "warning" for f in analysis["findings"]):
        reasons.append("a local check suggests the hidden dynamics may be unstable.")
    if any(f["check"] == "C4" and f["severity"] == "warning" for f in analysis["findings"]):
        reasons.append("the decoupling matrix is singular at some sampled points; the proofs assume it is invertible everywhere.")
    if any(f["check"] == "C7" and f["severity"] == "warning" for f in analysis["findings"]):
        reasons.append("a chain gain vanishes at some sampled points; backstepping divides by it.")
    return ("C" if reasons else "A"), reasons


def _system_type(dyn):
    return "SISO" if len(dyn["inputs"]) == 1 and len(dyn["outputs"]) == 1 else "MIMO"


def _finish(base, dyn, record, problem, on_event, extra_reasons=None):
    # problem() looks at the fresh analysis and returns why the result is
    # unusable, or None. Nothing is handed back if it is. extra_reasons are
    # label-C reasons the caller already knows about.
    dyn["system_type"] = _system_type(dyn)
    converted = _system_spec.normalize_spec(
        {"status": base["status"], "system_name": base["system_name"], "dynamics": dyn})
    analysis = analyze_spec(converted)
    if analysis["status"] != "fits":
        raise ConversionError("the converted plant does not fit: %s" % analysis["summary"])
    why = problem(analysis)
    if why:
        raise ConversionError("the converted plant is not usable: %s" % why)
    label, reasons = _label(converted["dynamics"], analysis)
    reasons = list(extra_reasons or []) + reasons
    label = "C" if reasons else "A"
    record["label"], record["label_reasons"] = label, reasons
    _emit(on_event, kind="note", stage="design",
          text="Plant conversion %s (%s, label %s): %s"
               % (record["id"], record["title"], label, " ".join(record["changes"])))
    return {"spec": converted, "record": record, "analysis": analysis}


def _original(dyn):
    return {"states": list(dyn["states"]), "inputs": list(dyn["inputs"]),
            "outputs": list(dyn["outputs"])}


def apply_relabel(spec, on_event=None):
    base, dyn = _prepare(spec)
    first = analyze_plant(dyn)
    if first["status"] == "cannot_analyze":
        raise ConversionError(first["summary"])
    chain = first["chain"]
    if not chain["applicable"]:
        raise ConversionError("reordering only applies to plants with one input and one output.")
    if chain["given_order_ok"]:
        raise ConversionError("the states already form a chain; there is nothing to reorder.")
    if not chain["relabel_order"]:
        raise ConversionError("no ordering of the states makes this plant a strict-feedback chain.")

    record = {"id": "T0", "title": "reorder the states", "task_changed": False,
              "exact_on_nominal": True, "original": _original(dyn)}
    idx = [dyn["states"].index(name) for name in chain["relabel_order"]]
    n = len(idx)
    for key in ("states", "state_equations", "state_meanings"):
        if len(dyn[key]) == n:
            dyn[key] = [dyn[key][i] for i in idx]
    if len(dyn["x0"]) == n:
        dyn["x0"] = [dyn["x0"][i] for i in idx]
    record["changes"] = ["The states were reordered as %s so the plant is a strict-feedback chain."
                         % chain["relabel_order"]]
    record["details"] = {"order": list(chain["relabel_order"])}
    return _finish(base, dyn, record,
                   lambda a: None if a["chain"]["given_order_ok"] else "the chain test still fails.",
                   on_event)


def _virtual_names(dyn, count, requested):
    if requested:
        if len(requested) != count or len(set(requested)) != count:
            raise ConversionError("virtual_names needs %d distinct names." % count)
        return list(requested)
    taken = set(dyn["states"]) | set(dyn["inputs"]) | set(dyn.get("parameters") or {})
    names, k = [], 1
    while len(names) < count:
        if "v%d" % k not in taken:
            names.append("v%d" % k)
        k += 1
    return names


def decoupling_at(spec, point=None):
    # D(x) at x0 (or `point`): rows are outputs, columns inputs, so you can see which
    # inputs reach which output. None if it can't be evaluated.
    base, dyn = _prepare(spec)
    try:
        states, inputs, outputs, f, _ = _parse_system(
            dyn["states"], dyn["state_equations"], dyn["inputs"], dyn["outputs"])
        rows = [lie_chain(o, states, inputs, f, output_name=o)["decoupling_row"] for o in outputs]
        x = point if point is not None else (
            dyn["x0"] if len(dyn["x0"]) == len(states) else [0.0] * len(states))
        arr = _finite_array(sp.lambdify(list(states), sp.Matrix(rows), "numpy")(*x))
    except Exception:  # noqa: BLE001
        return None
    if arr is None:
        return None
    return {"outputs": list(dyn["outputs"]), "inputs": list(dyn["inputs"]),
            "matrix": np.round(arr, 4).tolist(), "evaluated_at": "x0" if point is None else "given point"}


def apply_input_squaring(spec, weights=None, virtual_names=None, on_event=None):
    base, dyn = _prepare(spec)
    first = analyze_plant(dyn)
    if first["status"] == "cannot_analyze":
        raise ConversionError(first["summary"])
    m, p = first["dims"]["n_inputs"], first["dims"]["n_outputs"]
    if m <= p:
        raise ConversionError(
            "splitting commands needs more inputs than outputs (this plant has %d input(s) and %d output(s))." % (m, p))
    if p > _MAX_SQUARING_OUTPUTS:
        raise ConversionError("input splitting is limited to %d outputs." % _MAX_SQUARING_OUTPUTS)
    if not first["affine"]:
        raise ConversionError("the equations are not linear in the input.")
    if any(r is None for r in first["relative_degrees"]):
        raise ConversionError(first["methods"]["smc"]["reasons"][0])
    rank = first["decoupling"]["min_rank"]
    if rank != p:
        raise ConversionError(
            "the decoupling matrix drops to rank %s (of %d) at some sampled points, so the commands cannot be split everywhere." % (rank, p))

    weights = [1.0] * m if weights is None else [float(w) for w in weights]
    if len(weights) != m or any(w <= 0 for w in weights):
        raise ConversionError("weights needs %d positive numbers, one per input." % m)

    states, inputs, outputs, f, symbol_map = _parse_system(
        dyn["states"], dyn["state_equations"], dyn["inputs"], dyn["outputs"])
    rows = [lie_chain(o, states, inputs, f, output_name=o)["decoupling_row"] for o in outputs]
    D = sp.Matrix(rows)
    w_inv = sp.diag(*[sp.Float(1.0 / w) for w in weights])
    try:
        M = w_inv * D.T * (D * w_inv * D.T).inv()
    except Exception as e:  # noqa: BLE001
        raise ConversionError("D*W^-1*D^T could not be inverted symbolically: %s" % e)
    M = M.applyfunc(lambda e: sp.simplify(e) if sp.count_ops(e) <= _SIMPLIFY_OPS_LIMIT else e)

    names = _virtual_names(dyn, p, virtual_names)
    v_syms = [sp.Symbol(nm, real=True) for nm in names]
    u_of_v = {inputs[j]: sum(M[j, k] * v_syms[k] for k in range(p)) for j in range(m)}
    new_f = [fi.subs(u_of_v, simultaneous=True) for fi in f]

    _check_exactness(states, inputs, f, v_syms, new_f, M, D, dyn["x0"])

    for entry in dyn.get("uncertainty") or []:
        text = (entry.get("expr") or "").strip()
        if text:
            entry["expr"] = _expr_str(_sympify(text, symbol_map).subs(u_of_v, simultaneous=True))

    shares = _effort_shares(states, inputs, D, M, names, dyn["x0"])
    original = _original(dyn)
    dyn["inputs"] = names
    dyn["state_equations"] = [_expr_str(e) for e in new_f]
    record = {
        "id": "T1", "title": "split the commands across the inputs", "task_changed": False,
        "exact_on_nominal": True, "original": original,
        "changes": ["The %d inputs %s are driven by %d new commands %s, split with weights %s so each "
                    "command acts on one output. The plant behaves exactly as before.%s"
                    % (m, original["inputs"], p, names, weights, _share_text(shares))],
        "details": {"weights": weights, "virtual_inputs": names,
                    "input_map": {str(inputs[j]): str(u_of_v[inputs[j]]) for j in range(m)},
                    # 4 significant digits for people and models; input_map stays exact
                    "input_map_display": {str(inputs[j]): str(sp.N(u_of_v[inputs[j]], 4)) for j in range(m)},
                    "effort_share": shares,
                    "caveat": "actuator limits are not handled."},
    }
    return _finish(base, dyn, record,
                   lambda a: None if a["methods"]["smc"]["viable"] else "SMC still rejects it.",
                   on_event)


def _effort_shares(states, inputs, D, M, names, x0):
    # share of each command's effect carried by each input, in percent, at x0: D[k,j]*M[j,k] adds up
    # to 1 over the inputs j of one command k. The coefficients alone hide this when D is not 1.
    try:
        x = x0 if len(x0) == len(states) else [0.0] * len(states)
        fun = sp.lambdify(list(states), sp.Matrix([[D[k, j] * M[j, k] for j in range(len(inputs))]
                                                    for k in range(len(names))]), "numpy")
        arr = _finite_array(fun(*x))
    except Exception:  # noqa: BLE001
        return {}
    if arr is None:
        return {}
    return {names[k]: {str(inputs[j]): round(100.0 * float(arr[k, j]), 1)
                       for j in range(len(inputs)) if abs(arr[k, j]) > 5e-4}
            for k in range(len(names))}


def _share_text(shares):
    if not shares:
        return ""
    parts = ["%s: %s" % (k, ", ".join("%s %g%%" % (u, v) for u, v in per.items())) for k, per in shares.items()]
    return " Share of each command carried by each input at the initial state: %s." % "; ".join(parts)


def _check_exactness(states, inputs, f, v_syms, new_f, M, D, x0):
    # same plant, two ways: f(x, M(x)v) against the substituted equations, and D*M against I
    rng = np.random.default_rng(1)
    args_old, args_new = list(states) + list(inputs), list(states) + list(v_syms)
    f_old = sp.lambdify(args_old, sp.Matrix(f), "numpy")
    f_new = sp.lambdify(args_new, sp.Matrix(new_f), "numpy")
    m_fun = sp.lambdify(list(states), M, "numpy")
    d_fun = sp.lambdify(list(states), D, "numpy")
    worst_f, worst_dm = 0.0, 0.0
    for x in _sample_points(x0 if len(x0) == len(states) else [0.0] * len(states), 8, 1.0, 1):
        v = rng.uniform(-1.0, 1.0, size=len(v_syms))
        m_val, d_val = _finite_array(m_fun(*x)), _finite_array(d_fun(*x))
        if m_val is None or d_val is None:
            raise ConversionError("the command split is not finite at a sampled point.")
        old = _finite_array(f_old(*x, *(m_val @ v)))
        new = _finite_array(f_new(*x, *v))
        if old is None or new is None:
            raise ConversionError("the converted equations are not finite at a sampled point.")
        worst_f = max(worst_f, float(np.max(np.abs(old - new))))
        worst_dm = max(worst_dm, float(np.max(np.abs(d_val @ m_val - np.eye(len(v_syms))))))
    if worst_f > _EXACTNESS_TOL or worst_dm > _EXACTNESS_TOL:
        raise ConversionError(
            "internal check failed (dynamics error %.2e, D*M-I error %.2e)." % (worst_f, worst_dm))


def rank_output_selections(spec):
    base, dyn = _prepare(spec)
    m, p = len(dyn["inputs"]), len(dyn["outputs"])
    if m >= p:
        raise ConversionError("output selection needs fewer inputs than outputs (this plant has %d input(s) and %d output(s))." % (m, p))
    combos = list(itertools.combinations(dyn["outputs"], m))
    if len(combos) > _MAX_SELECTION_COMBINATIONS:
        raise ConversionError("too many output combinations (%d)." % len(combos))
    entries = []
    for combo in combos:
        report = analyze_plant(dict(dyn, outputs=list(combo)))
        unstable = any(f["check"] == "C6" for f in report["findings"])
        hidden = report["hidden_dimension"]
        sigma = report["decoupling"].get("min_sigma") or 0.0
        entries.append({
            "outputs": list(combo),
            "fits": report["status"] == "fits",
            "relative_degrees": report["relative_degrees"],
            "hidden_dimension": hidden,
            "min_sigma": sigma,
            "unstable_hidden_dynamics_hint": unstable,
            "summary": report["summary"],
            "_key": (report["status"] != "fits", unstable,
                     hidden if hidden is not None else 99, -sigma),
        })
    entries.sort(key=lambda e: e["_key"])
    for e in entries:
        del e["_key"]
    return entries


def apply_output_selection(spec, outputs, on_event=None):
    base, dyn = _prepare(spec)
    m, p = len(dyn["inputs"]), len(dyn["outputs"])
    outputs = list(outputs)
    if m >= p:
        raise ConversionError("output selection needs fewer inputs than outputs (this plant has %d input(s) and %d output(s))." % (m, p))
    unknown = [o for o in outputs if o not in dyn["outputs"]]
    if unknown or len(set(outputs)) != len(outputs):
        raise ConversionError("outputs must be distinct members of %s (got %s)." % (dyn["outputs"], outputs))
    if len(outputs) != m:
        raise ConversionError("pick exactly %d output(s), one per input (got %d)." % (m, len(outputs)))

    original = _original(dyn)
    untracked = [o for o in dyn["outputs"] if o not in outputs]
    given = {r.get("output") for r in dyn.get("references") or [] if (r.get("expr") or "").strip()}
    dyn["outputs"] = outputs
    dyn["references"] = [r for r in dyn.get("references") or [] if r.get("output") in outputs]
    record = {
        "id": "T2", "title": "track a subset of the outputs", "task_changed": True,
        "exact_on_nominal": True, "original": original, "untracked_outputs": untracked,
        "changes": ["The plant now tracks %s only. %s %s no longer controlled, so nothing is "
                    "guaranteed for %s."
                    % (", ".join(outputs), ", ".join(untracked),
                       "is" if len(untracked) == 1 else "are",
                       "it" if len(untracked) == 1 else "them")],
        "details": {"missing_references": [o for o in outputs if o not in given]},
    }
    return _finish(base, dyn, record,
                   lambda a: None if a["methods"]["smc"]["viable"] else "SMC rejects it.",
                   on_event)


def _outputs_entry(dyn, combo):
    report = analyze_plant(dict(dyn, outputs=list(combo)))
    hidden = report["hidden_dimension"]
    sigma = report["decoupling"].get("min_sigma") or 0.0
    unstable = any(f["check"] == "C6" for f in report["findings"])
    kept = [o for o in combo if o in dyn["outputs"]]
    return {
        "outputs": list(combo), "keeps_original": kept,
        "drops_original": [o for o in dyn["outputs"] if o not in combo],
        "fits": report["status"] == "fits", "relative_degrees": report["relative_degrees"],
        "hidden_dimension": hidden, "min_sigma": sigma,
        "unstable_hidden_dynamics_hint": unstable, "summary": report["summary"],
        "_key": (report["status"] != "fits", unstable, -len(kept),
                 hidden if hidden is not None else 99, -sigma),
    }


def rank_output_reselections(spec):
    # same number of outputs, but any states: lets a plant whose outputs can't be controlled
    # (or hide unstable dynamics) be tracked through other variables
    base, dyn = _prepare(spec)
    m, p = len(dyn["inputs"]), len(dyn["outputs"])
    if m != p:
        raise ConversionError("output re-selection needs as many inputs as outputs (this plant has %d input(s) and %d output(s))." % (m, p))
    combos = [c for c in itertools.combinations(dyn["states"], m) if set(c) != set(dyn["outputs"])]
    if len(combos) > _MAX_SELECTION_COMBINATIONS:
        raise ConversionError("too many output combinations (%d)." % len(combos))
    entries = sorted((_outputs_entry(dyn, c) for c in combos), key=lambda e: e["_key"])
    for e in entries:
        del e["_key"]
    return entries


def apply_output_reselection(spec, outputs, references=None, on_event=None):
    base, dyn = _prepare(spec)
    m, p = len(dyn["inputs"]), len(dyn["outputs"])
    outputs = list(outputs)
    if m != p:
        raise ConversionError("output re-selection needs as many inputs as outputs (this plant has %d input(s) and %d output(s))." % (m, p))
    if any(o not in dyn["states"] for o in outputs) or len(set(outputs)) != len(outputs):
        raise ConversionError("outputs must be distinct states of the plant %s (got %s)." % (dyn["states"], outputs))
    if len(outputs) != m:
        raise ConversionError("pick exactly %d output(s), one per input (got %d)." % (m, len(outputs)))
    if set(outputs) == set(dyn["outputs"]):
        raise ConversionError("these are already the plant's outputs.")

    original = _original(dyn)
    refs = {r["output"]: r.get("expr") for r in dyn.get("references") or [] if (r.get("expr") or "").strip()}
    refs.update({k: v for k, v in (references or {}).items() if str(v).strip()})
    dyn["outputs"] = outputs
    dyn["references"] = [{"output": o, "expr": refs[o]} for o in outputs if o in refs]
    dropped = [o for o in original["outputs"] if o not in outputs]
    missing = [o for o in outputs if o not in refs]
    record = {
        "id": "T3", "title": "track different outputs", "task_changed": True,
        "exact_on_nominal": True, "original": original, "untracked_outputs": dropped,
        "changes": ["The plant now tracks %s instead of %s. %s no longer tracked, so nothing is guaranteed for %s.%s"
                    % (", ".join(outputs), ", ".join(original["outputs"]),
                       ", ".join(dropped) or "No original output is",
                       "it" if len(dropped) == 1 else "them",
                       " A reference is still needed for %s." % ", ".join(missing) if missing else "")],
        "details": {"missing_references": missing,
                    "kept_outputs": [o for o in outputs if o in original["outputs"]]},
    }
    return _finish(base, dyn, record,
                   lambda a: None if a["methods"]["smc"]["viable"] else "SMC rejects it.", on_event)


_MAX_EXTENDABLE_INPUTS = 4
_DEFAULT_INITIAL_INPUT = 1.0
_MAX_ORDER_PER_INPUT = 3
_MAX_TOTAL_INTEGRATORS = 3
# state names for an input u that is integrated k times: u, u_dot, u_ddot; the new input is the next one
_LEVELS = ("", "_dot", "_ddot", "_dddot")
_ORDINAL = {2: "second", 3: "third"}


def _unique_name(base, taken):
    name = base
    while name in taken:
        name += "_"
    taken.add(name)
    return name


def _extended_dyn(dyn, orders, initial):
    # input u integrated k times: u becomes a state (same name, so the equations stay as they are),
    # followed by k-1 derivative states, and the plant gets a new input for the k-th derivative.
    # orders: {input name: k}. Returns the new dynamics, {input: new input name} and {input: its states}
    new = copy.deepcopy(dyn)
    taken = set(dyn["states"]) | set(dyn["inputs"])
    n = len(dyn["states"])
    x0 = dyn["x0"] if len(dyn["x0"]) == n else [0.0] * n
    meanings = list(dyn["state_meanings"]) + [""] * (n - len(dyn["state_meanings"]))
    states, equations, start, texts = list(dyn["states"]), list(dyn["state_equations"]), list(x0), list(meanings)
    tops, levels = {}, {}
    for name in dyn["inputs"]:
        k = orders.get(name)
        if not k:
            continue
        chain = [name] + [_unique_name(name + _LEVELS[j], taken) for j in range(1, k)]
        tops[name] = _unique_name(name + _LEVELS[k], taken)
        levels[name] = chain
        states += chain
        equations += chain[1:] + [tops[name]]
        start += [float(initial[name])] + [0.0] * (k - 1)
        texts += ["%s, now a state" % name] + ["derivative %d of %s" % (j, name) for j in range(1, k)]
    new["states"], new["state_equations"], new["x0"], new["state_meanings"] = states, equations, start, texts
    new["inputs"] = [tops.get(name, name) for name in dyn["inputs"]]
    return new, tops, levels


def _check_extension(dyn, new, orders, levels):
    # the original plant fed u = z must equal the extended one, and the new rows must be shifts
    states, inputs, _, f, _ = _parse_system(dyn["states"], dyn["state_equations"], dyn["inputs"], dyn["outputs"])
    n_states, n_inputs, _, g, _ = _parse_system(new["states"], new["state_equations"], new["inputs"], new["outputs"])
    f_old = sp.lambdify(list(states) + list(inputs), sp.Matrix(f), "numpy")
    f_new = sp.lambdify(list(n_states) + list(n_inputs), sp.Matrix(g), "numpy")
    rng = np.random.default_rng(2)
    n, worst = len(states), 0.0
    for _ in range(8):
        x = rng.uniform(-1, 1, n)
        value = {name: rng.uniform(-1, 1, len(chain)) for name, chain in levels.items()}
        other = {name: rng.uniform(-1, 1) for name in dyn["inputs"]}
        u_old = [value[name][0] if name in value else other[name] for name in dyn["inputs"]]
        extra = np.concatenate([value[name] for name in dyn["inputs"] if name in value]) if value else np.array([])
        u_new = [other[name] for name in dyn["inputs"]]
        got = _finite_array(f_new(*x, *extra, *u_new))
        old = _finite_array(f_old(*x, *u_old))
        if old is None or got is None:
            raise ConversionError("the extended equations are not finite at a sampled point.")
        old, got = old.ravel(), got.ravel()
        worst = max(worst, float(np.max(np.abs(old - got[:n]))))
        row = n
        for name in dyn["inputs"]:
            if name not in value:
                continue
            v = value[name]
            for j in range(len(v)):
                want = v[j + 1] if j + 1 < len(v) else other[name]
                worst = max(worst, abs(float(got[row + j]) - float(want)))
            row += len(v)
    if worst > _EXACTNESS_TOL:
        raise ConversionError("internal check failed (extension error %.2e)." % worst)


def _extension_precheck(dyn):
    first = analyze_plant(dyn)
    if first["status"] == "cannot_analyze":
        raise ConversionError(first["summary"])
    m, p = first["dims"]["n_inputs"], first["dims"]["n_outputs"]
    if m != p:
        raise ConversionError("dynamic extension needs as many inputs as outputs (this plant has %d input(s) and %d output(s))." % (m, p))
    if m > _MAX_EXTENDABLE_INPUTS:
        raise ConversionError("dynamic extension is limited to %d inputs." % _MAX_EXTENDABLE_INPUTS)
    if not first["affine"]:
        raise ConversionError("the equations are not linear in the input.")
    if any(r is None for r in first["relative_degrees"]):
        raise ConversionError(first["methods"]["smc"]["reasons"][0])
    if not any(h["id"] == "T4" for h in first["conversion_hints"]):
        raise ConversionError("the decoupling matrix is already invertible; there is nothing to fix.")


def _order_vectors(names, total):
    # every {input: k >= 1} with the k adding up to `total`, fewest extended inputs first
    out = []
    for size in range(1, min(len(names), total) + 1):
        for chosen in itertools.combinations(names, size):
            for ks in itertools.product(range(1, _MAX_ORDER_PER_INPUT + 1), repeat=size):
                if sum(ks) == total:
                    out.append(dict(zip(chosen, ks)))
    return out


def propose_extensions(spec, initial_inputs=None):
    # tries integrators on the inputs, fewest integrators in total first, and stops at the first
    # total that gives a plant the module accepts. Every entry tried is reported.
    base, dyn = _prepare(spec)
    _extension_precheck(dyn)
    initial = {name: _DEFAULT_INITIAL_INPUT for name in dyn["inputs"]}
    initial.update(initial_inputs or {})
    entries = []
    for total in range(1, _MAX_TOTAL_INTEGRATORS + 1):
        found = False
        for orders in _order_vectors(dyn["inputs"], total):
            new, _, _ = _extended_dyn(dyn, orders, initial)
            report = analyze_plant(new)
            fits = report["status"] == "fits"
            found = found or fits
            entries.append({
                "inputs": list(orders), "orders": dict(orders), "fits": fits,
                "relative_degrees": report["relative_degrees"],
                "hidden_dimension": report["hidden_dimension"],
                "min_sigma": report["decoupling"].get("min_sigma") or 0.0,
                "summary": report["summary"]})
        if found:
            break
    return entries


def _extension_text(name, k, chain, top):
    if k == 1:
        return "%s became a state and the controller now commands %s, its rate of change" % (name, top)
    return ("%s and its derivative(s) %s became states and the controller now commands %s, its %s derivative"
            % (name, ", ".join(chain[1:]), top, _ORDINAL[k]))


def apply_dynamic_extension(spec, inputs=None, initial_inputs=None, orders=None, on_event=None):
    # orders: {input: number of integrators}; without it every listed input gets one. Without
    # inputs too, the cheapest working choice is taken.
    base, dyn = _prepare(spec)
    _extension_precheck(dyn)
    initial_inputs = dict(initial_inputs or {})
    unknown = [k for k in initial_inputs if k not in dyn["inputs"]]
    if unknown:
        raise ConversionError("initial_inputs names %s are not inputs of the plant." % unknown)
    if inputs is None and orders:
        inputs = list(orders)
    if inputs is None:
        fitting = [e for e in propose_extensions(spec, initial_inputs) if e["fits"]]
        if not fitting:
            raise ConversionError("integrators on the inputs (up to %d in total) do not give an invertible "
                                  "decoupling matrix." % _MAX_TOTAL_INTEGRATORS)
        inputs, orders = fitting[0]["inputs"], fitting[0]["orders"]
    chosen = list(inputs)
    if not chosen or len(set(chosen)) != len(chosen) or any(c not in dyn["inputs"] for c in chosen):
        raise ConversionError("inputs must be distinct inputs of the plant %s (got %s)." % (dyn["inputs"], chosen))
    orders = {c: int((orders or {}).get(c, 1)) for c in chosen}
    if any(not 1 <= k <= _MAX_ORDER_PER_INPUT for k in orders.values()) or sum(orders.values()) > _MAX_TOTAL_INTEGRATORS:
        raise ConversionError("each input takes 1 to %d integrators, and at most %d in total (got %s)."
                              % (_MAX_ORDER_PER_INPUT, _MAX_TOTAL_INTEGRATORS, orders))
    initial = {c: float(initial_inputs.get(c, _DEFAULT_INITIAL_INPUT)) for c in chosen}
    assumed = [c for c in chosen if c not in initial_inputs]

    original = _original(dyn)
    new, tops, levels = _extended_dyn(dyn, orders, initial)
    _check_extension(dyn, new, orders, levels)
    dyn.update({k: new[k] for k in ("states", "state_meanings", "state_equations", "inputs", "x0")})
    parts = "; ".join(_extension_text(c, orders[c], levels[c], tops[c]) for c in chosen)
    deeper = [c for c in chosen if orders[c] > 1]
    record = {
        "id": "T4", "title": "add an integrator in front of the input" if not deeper
                              else "add integrators in front of the input", "task_changed": False,
        "exact_on_nominal": True, "original": original,
        "changes": ["%s. The plant behaves exactly as before. Starting value(s): %s%s.%s"
                    % (parts, ", ".join("%s = %g" % (c, initial[c]) for c in chosen),
                       " (assumed; change them if the real starting command differs)" if assumed else "",
                       " The extra derivatives start at 0." if deeper else "")],
        "details": {"extended_inputs": chosen, "rate_inputs": [tops[c] for c in chosen], "orders": orders,
                    "derivative_states": {c: levels[c][1:] for c in deeper},
                    "initial_inputs": initial, "assumed_initial_inputs": assumed},
    }
    return _finish(base, dyn, record,
                   lambda a: None if a["methods"]["smc"]["viable"] else "SMC still rejects it.", on_event)


_MAX_LIE_STATES = 5
_INVERSE_OPS_LIMIT = 200
_ROUNDTRIP_TOL = 1e-6


def _lie_names(out_name, n, taken):
    names = [out_name]
    for k in range(2, n + 1):
        candidate = "z%d" % k
        while candidate in taken or candidate in names:
            candidate += "_"
        names.append(candidate)
    return names


def _lie_precheck(dyn):
    first = analyze_plant(dyn)
    if first["status"] == "cannot_analyze":
        raise ConversionError(first["summary"])
    if first["dims"]["n_inputs"] != 1 or first["dims"]["n_outputs"] != 1:
        raise ConversionError("Lie coordinates are only built for one input and one output.")
    if not first["affine"]:
        raise ConversionError("the equations are not linear in the input.")
    chain = first["chain"]
    if chain["given_order_ok"]:
        raise ConversionError("the plant is already a strict-feedback chain; there is nothing to convert.")
    if chain["relabel_order"]:
        raise ConversionError("reordering the states (T0) already gives a chain; that is simpler and exact.")
    n = first["dims"]["n_states"]
    if n > _MAX_LIE_STATES:
        raise ConversionError("Lie coordinates are limited to %d states (this plant has %d)." % (_MAX_LIE_STATES, n))
    rho = first["relative_degrees"][0]
    if rho is None:
        raise ConversionError(first["methods"]["smc"]["reasons"][0])
    if rho != n:
        raise ConversionError(
            "the relative degree is %d but the plant has %d states; the other states are hidden "
            "dynamics that Lie coordinates cannot describe." % (rho, n))


def _invert_transform(T, states, tmp, x0):
    # solves T(x) = z for x in closed form and keeps the branch that goes through x0
    if sp.count_ops(T) > _INVERSE_OPS_LIMIT:
        raise ConversionError("the coordinate change is too large to invert in closed form.")
    try:
        sols = sp.solve([T[k] - tmp[k] for k in range(len(tmp))], list(states), dict=True)
    except Exception as e:  # noqa: BLE001
        raise ConversionError("no closed-form inverse of the coordinate change (%s)." % type(e).__name__) from e
    at_x0 = dict(zip(states, x0))
    z0 = {tmp[k]: T[k].subs(at_x0) for k in range(len(tmp))}
    for sol in sols:
        if any(s not in sol for s in states):
            continue
        try:
            back = [complex(sp.N(sol[s].subs(z0))) for s in states]
        except (TypeError, ValueError):
            continue
        if all(abs(b.imag) < 1e-9 and abs(b.real - x) < 1e-6 for b, x in zip(back, x0)):
            return [sol[s] for s in states]
    raise ConversionError("no closed-form inverse of the coordinate change that passes through the initial state.")


def _lie_numeric_checks(states, inputs, f, T, inv, tmp, a_expr, b_expr, x0):
    # round trip, invertible Jacobian and the new equations against the old ones at sample points
    jac = T.jacobian(states)
    t_fun = sp.lambdify(list(states), T, "numpy")
    jac_fun = sp.lambdify(list(states), jac, "numpy")
    inv_fun = sp.lambdify(list(tmp), sp.Matrix(inv), "numpy")
    f_fun = sp.lambdify(list(states) + list(inputs), sp.Matrix(f), "numpy")
    last_fun = sp.lambdify(list(tmp) + list(inputs), sp.Matrix([a_expr + b_expr * inputs[0]]), "numpy")
    rng = np.random.default_rng(3)
    points = _sample_points(x0, 20, 0.5, 1)
    bad, worst = [], 0.0
    for k, pt in enumerate(points):
        z = _finite_array(t_fun(*pt))
        back = None if z is None else _finite_array(inv_fun(*z.ravel()))
        jac_val = None if z is None else _finite_array(jac_fun(*pt))
        ok = (back is not None and jac_val is not None
              and float(np.max(np.abs(back.ravel() - pt))) < _ROUNDTRIP_TOL * (1 + np.max(np.abs(pt)))
              and abs(np.linalg.det(jac_val)) > 1e-9)
        if not ok:
            bad.append(k)
            continue
        u = rng.uniform(-1, 1)
        old = _finite_array(f_fun(*pt, u))
        new_last = _finite_array(last_fun(*z.ravel(), u))
        if old is None or new_last is None:
            bad.append(k)
            continue
        want = np.append(z.ravel()[1:], float(new_last.ravel()[0]))
        worst = max(worst, float(np.max(np.abs(jac_val @ old.ravel() - want))))
    if 0 in bad or len(bad) * 2 >= len(points):
        raise ConversionError("the coordinate change or its inverse is not valid at the initial state or over "
                              "most of the sampled region.")
    if worst > _EXACTNESS_TOL * 1e3:
        raise ConversionError("internal check failed (coordinate error %.2e)." % worst)
    return len(bad), len(points)


def _transform_terms(entries, states, inputs, tmp, zs, inv, jac, time_only):
    # maps uncertainty/disturbance entries (given per state) to the new coordinates
    symbol_map = {"t": sp.Symbol("t")} if time_only else {str(s): s for s in list(states) + list(inputs)}
    vec = [0] * len(states)
    names = [str(s) for s in states]
    for entry in entries or []:
        text = (entry.get("expr") or "").strip()
        if not text:
            continue
        if entry.get("state") not in names:
            raise ConversionError("an uncertainty or disturbance entry names %r, which is not a state." % entry.get("state"))
        try:
            vec[names.index(entry["state"])] += _sympify(text, symbol_map)
        except Exception as e:  # noqa: BLE001
            raise ConversionError("could not read the term %r (%s)." % (text, type(e).__name__)) from e
    rows = jac * sp.Matrix(vec)
    sub, back = dict(zip(states, inv)), dict(zip(tmp, zs))
    out = []
    for k in range(len(states)):
        expr = rows[k].xreplace(sub).xreplace(back)
        out.append(sp.simplify(expr) if sp.count_ops(expr) <= _SIMPLIFY_OPS_LIMIT else expr)
    return out


def _is_zero_expr(expr):
    return sp.count_ops(expr) <= _SIMPLIFY_OPS_LIMIT and sp.simplify(expr) == 0


def apply_lie_coordinates(spec, on_event=None):
    base, dyn = _prepare(spec)
    _lie_precheck(dyn)
    states, inputs, outputs, f, _ = _parse_system(dyn["states"], dyn["state_equations"], dyn["inputs"], dyn["outputs"])
    n = len(states)
    chain = lie_chain(outputs[0], states, inputs, f, output_name=dyn["outputs"][0])
    T = sp.Matrix(chain["derivatives"])
    drift, gain = chain["drift_term"], chain["decoupling_row"][0]
    x0 = [float(v) for v in dyn["x0"]] if len(dyn["x0"]) == n else [0.0] * n

    names = _lie_names(dyn["outputs"][0], n, set(dyn["states"]) | set(dyn["inputs"]))
    tmp = [sp.Symbol("_zt%d" % k, real=True) for k in range(n)]
    zs = [sp.Symbol(name, real=True) for name in names]
    inv = _invert_transform(T, states, tmp, x0)
    sub, to_z = dict(zip(states, inv)), dict(zip(tmp, zs))
    a_expr, b_expr = drift.xreplace(sub), gain.xreplace(sub)
    bad, total = _lie_numeric_checks(states, inputs, f, T, inv, tmp, a_expr, b_expr, x0)

    jac = T.jacobian(states)
    delta = _transform_terms(dyn.get("uncertainty"), states, inputs, tmp, zs, inv, jac, False)
    dist = _transform_terms(dyn.get("disturbance"), states, inputs, tmp, zs, inv, jac, True)

    reasons = []
    if bad:
        reasons.append("the coordinate change and its inverse were only valid at %d of %d sampled points; "
                       "treat them as local to the initial state." % (total - bad, total))
    new_uncertainty = []
    for k, expr in enumerate(delta):
        if _is_zero_expr(expr):
            continue
        new_uncertainty.append({"state": names[k], "expr": str(expr)})
        later = sorted(str(s) for s in expr.free_symbols if str(s) in names and names.index(str(s)) > k)
        if later:
            reasons.append("the uncertainty in the equation of %s depends on %s after the change; "
                           "the stability proof needs it to depend on %s only."
                           % (names[k], ", ".join(later), ", ".join(names[:k + 1])))
    new_dist = []
    for k, expr in enumerate(dist):
        if _is_zero_expr(expr):
            continue
        if {str(s) for s in expr.free_symbols} - {"t"}:
            raise ConversionError("after the change the disturbance in %s would depend on the state, "
                                  "but the module expects a function of time only." % names[k])
        new_dist.append({"state": names[k], "expr": str(expr)})

    original = _original(dyn)
    meanings = list(dyn["state_meanings"]) + [""] * (n - len(dyn["state_meanings"]))
    out_meaning = meanings[dyn["states"].index(dyn["outputs"][0])]
    z_x0 = [float(T[k].subs(dict(zip(states, x0)))) for k in range(n)]
    drift_text = _expr_str(a_expr.xreplace(to_z))
    last = "(%s)*%s" % (_expr_str(b_expr.xreplace(to_z)), inputs[0])
    if drift_text != "0":
        last = "%s + %s" % (drift_text, last)
    dyn["states"] = names
    dyn["state_meanings"] = [out_meaning] + ["derivative %d of %s along the plant" % (k, names[0]) for k in range(1, n)]
    dyn["state_equations"] = names[1:] + [last]
    dyn["x0"] = z_x0
    dyn["outputs"] = [names[0]]
    dyn["uncertainty"], dyn["disturbance"] = new_uncertainty, new_dist
    record = {
        "id": "T5", "title": "change to chain coordinates", "task_changed": False,
        "exact_on_nominal": True, "original": original,
        "changes": ["The states became %s, where each one is the next derivative of %s, so the plant is a "
                    "strict-feedback chain. The tracked output and its reference are unchanged. SMC could have "
                    "handled the original plant too; this is only for backstepping." % (", ".join(names), names[0]),
                    "The new equations only describe the original plant where the inverse exists, so a warning "
                    "that appears far from the initial state may fall outside that region."],
        "details": {"new_states": names, "order": names,
                    "transformation": {names[k]: _expr_str(T[k]) for k in range(n)},
                    "inverse": {str(s): _expr_str(e.xreplace(to_z)) for s, e in zip(states, inv)},
                    "last_equation": last, "sampled_valid": [total - bad, total]},
    }
    return _finish(base, dyn, record,
                   lambda a: None if a["methods"]["backstepping"]["viable"] else "it is still not a chain.",
                   on_event, extra_reasons=reasons)


_MAX_REMOVED_RATIO = 0.5


def _term_split(f, states, inputs):
    # per row, terms that break the chain (look further than one state ahead, are nonlinear in the
    # next state, or hold the input in a row before the last) are separated from the rest
    n = len(states)
    kept, removed = [], []
    for i in range(n):
        terms = sp.Add.make_args(sp.expand(f[i]))
        if i == n - 1:
            kept.append(sp.Add(*terms))
            removed.append(sp.Integer(0))
            continue
        nxt, allowed = states[i + 1], set(states[:i + 2])
        keep, move = [], []
        for t in terms:
            free = t.free_symbols
            ok = (free <= allowed and not (free & set(inputs))
                  and (nxt not in free or sp.diff(t, nxt, 2) == 0))
            (keep if ok else move).append(t)
        kept.append(sp.Add(*keep))
        removed.append(sp.Add(*move))
    return kept, removed


def _removed_sizes(states, inputs, f, removed, x0):
    # rms of each removed term against the whole right-hand side at sample points
    fun_r = sp.lambdify(list(states) + list(inputs), sp.Matrix(removed), "numpy")
    fun_f = sp.lambdify(list(states) + list(inputs), sp.Matrix(f), "numpy")
    rng = np.random.default_rng(4)
    r_rows, f_rows = [], []
    for pt in _sample_points(x0, 20, 0.5, 2):
        u = rng.uniform(-1, 1, len(inputs))
        r, full = _finite_array(fun_r(*pt, *u)), _finite_array(fun_f(*pt, *u))
        if r is not None and full is not None:
            r_rows.append(r.ravel())
            f_rows.append(full.ravel())
    if not r_rows:
        raise ConversionError("the removed terms are not finite at the sampled points.")
    r_rms = np.sqrt(np.mean(np.square(r_rows), axis=0))
    f_rms = np.sqrt(np.mean(np.square(f_rows), axis=0))
    return [float(a / b) if b > 1e-12 else (0.0 if a < 1e-12 else float("inf")) for a, b in zip(r_rms, f_rms)]


def apply_term_shift(spec, on_event=None):
    # experimental: the controller is designed for a chain, the removed terms stay in the plant as
    # uncertainty, so the simulation still runs the full plant
    base, dyn = _prepare(spec)
    first = analyze_plant(dyn)
    if first["status"] == "cannot_analyze":
        raise ConversionError(first["summary"])
    if first["dims"]["n_inputs"] != 1 or first["dims"]["n_outputs"] != 1:
        raise ConversionError("moving terms to the uncertainty is only built for one input and one output.")
    if not first["affine"]:
        raise ConversionError("the equations are not linear in the input.")
    if first["chain"]["given_order_ok"]:
        raise ConversionError("the plant is already a strict-feedback chain; there is nothing to move.")
    if first["chain"]["relabel_order"]:
        raise ConversionError("reordering the states (T0) already gives a chain; that is simpler and exact.")
    states, inputs, outputs, f, _ = _parse_system(dyn["states"], dyn["state_equations"], dyn["inputs"], dyn["outputs"])
    n = len(states)
    x0 = [float(v) for v in dyn["x0"]] if len(dyn["x0"]) == n else [0.0] * n
    kept, removed = _term_split(f, states, inputs)
    for i in range(n - 1):
        if sp.simplify(sp.diff(kept[i], states[i + 1])) == 0:
            raise ConversionError(
                "removing the terms that break the chain also removes the link from %s to %s in its equation, "
                "so no chain is left." % (states[i], states[i + 1]))
    if not any(r != 0 for r in removed):
        raise ConversionError("no term can be moved to make this plant a chain.")
    ratios = _removed_sizes(states, inputs, f, removed, x0)
    too_big = [(str(states[i]), r) for i, r in enumerate(ratios) if r > _MAX_REMOVED_RATIO]
    if too_big:
        raise ConversionError(
            "the terms to remove are too large to treat as uncertainty (%s; limit %.0f%% of the equation)."
            % (", ".join("%s: %.0f%%" % (s, 100 * r) for s, r in too_big), 100 * _MAX_REMOVED_RATIO))

    original = _original(dyn)
    names = [str(s) for s in states]
    entries = {e["state"]: dict(e) for e in dyn.get("uncertainty") or [] if e.get("state") in names}
    reasons = ["experimental: the controller is designed for a simpler model than the plant, and the stability "
               "proof does not cover that gap."]
    removed_text, changes = {}, []
    for i, r in enumerate(removed):
        if r == 0:
            continue
        removed_text[names[i]] = [str(t) for t in sp.Add.make_args(r)]
        old = (entries.get(names[i], {}).get("expr") or "").strip()
        entries[names[i]] = {"state": names[i], "expr": ("%s + %s" % (old, r)) if old else str(r)}
        unseen = sorted(str(s) for s in r.free_symbols if str(s) in names and names.index(str(s)) > i)
        if unseen:
            reasons.append("the moved term in the equation of %s depends on %s, which the estimator for that "
                           "equation does not see." % (names[i], ", ".join(unseen)))
        changes.append("%s' lost %s (about %.1f%% of its right-hand side at the sampled points); it is now part "
                       "of the uncertainty." % (names[i], r, 100 * ratios[i]))
    dyn["state_equations"] = [str(k) for k in kept]
    dyn["uncertainty"] = [entries[k] for k in names if k in entries] + [
        e for e in dyn.get("uncertainty") or [] if e.get("state") not in names]
    record = {
        "id": "T6", "title": "move terms that break the chain into the uncertainty", "task_changed": False,
        "exact_on_nominal": False, "experimental": True, "original": original,
        "changes": changes + ["The simulation still runs the full plant; only the model the controller is built "
                              "from is a chain."],
        "details": {"removed_terms": removed_text,
                    "removed_size_percent": {names[i]: round(100 * ratios[i], 1) for i in range(n) if removed[i] != 0},
                    "kept_equations": dict(zip(names, dyn["state_equations"]))},
    }
    return _finish(base, dyn, record,
                   lambda a: None if a["methods"]["backstepping"]["viable"] else "it is still not a chain.",
                   on_event, extra_reasons=reasons)
