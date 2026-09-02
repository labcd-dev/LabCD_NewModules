import re
import sympy as sp
from pydantic import BaseModel, Field

from .estimators import round_floats

ALLOWED_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "sqrt": sp.sqrt, "tanh": sp.tanh, "Abs": sp.Abs,
    "Heaviside": sp.Heaviside,
}

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def coerce_float_list(value, expected_len, field_name="value"):
    if isinstance(value, (list, tuple)) and all(isinstance(v, (int, float)) for v in value):
        nums = [float(v) for v in value]
    else:
        joined = " ".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)
        nums = [float(m) for m in _NUMBER_RE.findall(joined)]

    if len(nums) != expected_len:
        print("warning: %s had %d numbers, expected %d (raw: %r); "
              "padding/truncating with zeros" % (field_name, len(nums), expected_len, value))
        nums = (nums + [0.0] * expected_len)[:expected_len]
    return nums


_ASSIGNMENT_PREFIX_RE = re.compile(r"^[^=<>!]*=(?!=)\s*")


def _sympify(expr_str, symbol_map):
    # model keeps writing full assignments ("x2_dot = x2") instead of bare RHS,
    # and sympy can't parse "=", so just strip a leading "anything =" prefix.
    stripped = _ASSIGNMENT_PREFIX_RE.sub("", expr_str.strip())
    if stripped != expr_str.strip():
        print("note: entry %r looked like an assignment; using %r instead"
              % (expr_str, stripped))
        expr_str = stripped
    local_dict = dict(symbol_map)
    local_dict.update(ALLOWED_FUNCS)
    return sp.sympify(expr_str, locals=local_dict)


def _parse_system(states, dynamics, inputs, outputs):
    # real=True actually matters here. skip it and differentiating x*Abs(x)
    # (a common drag term) gives re(x)/im(x) junk instead of the clean sign(x) you'd expect
    state_syms = list(sp.symbols(states, real=True)) if len(states) > 1 else [sp.Symbol(states[0], real=True)]
    input_syms = list(sp.symbols(inputs, real=True)) if len(inputs) > 1 else [sp.Symbol(inputs[0], real=True)]
    symbol_map = {str(s): s for s in state_syms + input_syms}
    output_syms = [symbol_map[name] for name in outputs]
    dyn_exprs = [_sympify(e, symbol_map) for e in dynamics]
    return state_syms, input_syms, output_syms, dyn_exprs, symbol_map


def _build_delta_func(delta_exprs, symbol_map, state_syms, input_syms):
    if not delta_exprs:
        return None
    exprs = [_sympify(e, symbol_map) for e in delta_exprs]
    fn = sp.lambdify([state_syms, input_syms], exprs, "numpy")
    return lambda x, u: fn(list(x), list(u))


def _build_dist_func(dist_exprs):
    if not dist_exprs:
        return None
    t_sym = sp.symbols("t")
    exprs = [_sympify(e, {"t": t_sym}) for e in dist_exprs]
    fn = sp.lambdify([t_sym], exprs, "numpy")
    return lambda t: fn(t)


def _latex_dot(state_name):
    m = re.match(r"^([A-Za-z]+)(\d+)$", state_name)
    if m:
        return r"\dot{%s}_{%s}" % (m.group(1), m.group(2))
    return r"\dot{%s}" % state_name


def _print_system_echo(states, dynamics, inputs, outputs, system_type,
                        has_delta, has_disturbance, reasoning, state_syms=None, dyn_exprs=None,
                        delta_exprs=None, dist_exprs=None):
    print("\n=== agent's decision ===")
    print("system_type:     %d (1=SMC/square, 2=Backstepping/strict-feedback)" % system_type)
    print("has_delta:       %s" % has_delta)
    print("has_disturbance: %s" % has_disturbance)
    print("reasoning:       %s" % reasoning)
    print("states:  ", states)
    print("inputs:  ", inputs)
    print("outputs: ", outputs)
    print("delta_exprs: ", delta_exprs)
    print("dist_exprs:  ", dist_exprs)
    state_space_lines = []
    for i, d in enumerate(dynamics):
        line = "%s_dot = %s" % (states[i], d)
        print("  " + line)
        if state_syms is not None and dyn_exprs is not None:
            latex_line = "$$%s = %s$$" % (_latex_dot(states[i]),
                                           sp.latex(round_floats(dyn_exprs[i])))
        else:
            latex_line = "$$%s$$" % line
        state_space_lines.append(latex_line)
    print("==========================================================================\n")
    return "\n\n".join(state_space_lines)


def _ref_symbolic_derivatives(expr_text, degree, t_sym):
    # any differentiable function of t -- a DiracDelta derivative (from a
    # Heaviside jump) is treated as 0, same as a plain step's derivatives.
    expr = _ref_from_expr(expr_text)
    table = [expr]
    for _ in range(degree):
        d = sp.diff(table[-1], t_sym)
        if d.has(sp.DiracDelta):
            d = d.replace(sp.DiracDelta, lambda *a: sp.Integer(0))
        table.append(d)
    return table


def _control_law_display(lhs_latex, expr):
    # keep this plain $$...$$: streamlit renders it live through katex, which
    # can't parse a raw latex env. dmath* wrapping happens in the pdf backend.
    return "$$%s = %s$$" % (lhs_latex, sp.latex(round_floats(expr)))


_REF_ACCENTS = ("y", r"\dot{y}", r"\ddot{y}", r"\dddot{y}")


def _ref_symbol_latex(order, out_index=None):
    # just renames the internal yd0_2-style symbols to something readable
    # (y_d, y_d-dot, ...) for display. the actual math is untouched
    base = _REF_ACCENTS[order] if order < len(_REF_ACCENTS) else "y^{(%d)}" % order
    sub = "d" if out_index is None else "d,%d" % (out_index + 1)
    return "%s_{%s}" % (base, sub)


def _pretty_ref_symbols(yd_rows):
    multi = len(yd_rows) > 1
    subs_map = {}
    for i, row in enumerate(yd_rows):
        for k, symbol in enumerate(row):
            subs_map[symbol] = sp.Symbol(_ref_symbol_latex(k, i if multi else None))
    return subs_map


def _ref_symbol_legend(multi_output):
    return ("Symbolic control law, in the states plus the reference and its "
            "time derivatives ($y_d$ is the reference itself, $\\dot{y}_d$ its "
            "first time derivative, $\\ddot{y}_d$ its second, and so on"
            + (", with the second subscript naming the output):"
               if multi_output else "):"))


def _format_u_with_reference_smc(u_symbolic, yd_symbols, refs):
    # the agent's summary only sees this string, not the console prints.
    # so any failure here should fall back to a short note, never a crash
    try:
        t_sym = sp.Symbol("t")
        subs_map = {}
        for i, yd_row in enumerate(yd_symbols):
            degree = len(yd_row) - 1
            yd_vals = _ref_symbolic_derivatives(refs[i]["expr"], degree, t_sym)
            subs_map.update(zip(yd_row, yd_vals))

        pretty_map = _pretty_ref_symbols(yd_symbols)
        symbolic_lines = [_control_law_display("u_{%d}" % (i + 1),
                                                 u_symbolic[i].subs(pretty_map))
                           for i in range(len(u_symbolic))]
        ref_lines = []
        for i in range(len(u_symbolic)):
            u_ref = sp.trigsimp(sp.expand(u_symbolic[i].subs(subs_map)))
            ref_lines.append(_control_law_display("u_{%d}(x,t)" % (i + 1), u_ref))

        report = (
            _ref_symbol_legend(len(yd_symbols) > 1) + "\n\n  "
            + "\n\n  ".join(symbolic_lines)
            + "\n\nSame control law with THIS system's actual reference substituted in "
              "(reference derivatives expanded, u(x,t)):\n\n  "
            + "\n\n  ".join(ref_lines)
        )
    except Exception as e:
        report = ("(could not substitute the reference into the symbolic control "
                   "law for display: %s: %s)" % (type(e).__name__, e))
    print("\n" + report)
    return report


def _format_u_with_reference_backstepping(u_law, yd, ref):
    try:
        t_sym = sp.Symbol("t")
        degree = len(yd) - 1
        yd_vals = _ref_symbolic_derivatives(ref["expr"], degree, t_sym)
        subs_map = dict(zip(yd, yd_vals))
        u_ref = sp.trigsimp(sp.expand(u_law.subs(subs_map)))
        pretty_map = _pretty_ref_symbols([yd])
        report = (
            _ref_symbol_legend(False) + "\n  "
            + _control_law_display("u", u_law.subs(pretty_map))
            + "\n\nSame control law with THIS system's actual reference substituted in "
              "(reference derivatives expanded):\n  "
            + _control_law_display("u(x,t)", u_ref)
        )
    except Exception as e:
        report = ("(could not substitute the reference into the symbolic control "
                   "law for display: %s: %s)" % (type(e).__name__, e))
    print("\n" + report)
    return report


class RefSpec(BaseModel):
    expr: str = Field(description="the reference y_d(t), any function of t (e.g. "
                                   "'0.5*sin(0.666*t)+0.333', 'exp(-t)', 'Heaviside(t)')")


_REF_T = sp.symbols("t")
_REF_LOCALS = dict(ALLOWED_FUNCS)
_REF_LOCALS["t"] = _REF_T
_REF_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _ref_from_expr(expr_text):
    # y_d(t) can be anything sympy can parse and differentiate -- not just a
    # sin/cos/step menu. Anything this can't make sense of raises instead of
    # quietly falling back to a zero reference.
    text = (expr_text or "").strip()
    if not text:
        raise ValueError("no reference expression was given")
    unknown = sorted({tok for tok in _REF_IDENT_RE.findall(text) if tok not in _REF_LOCALS})
    if unknown:
        raise ValueError(
            "reference %r uses unrecognized name(s) %s -- only 't' and %s "
            "are allowed" % (text, ", ".join(unknown), ", ".join(sorted(ALLOWED_FUNCS))))
    try:
        return sp.expand(sp.sympify(text, locals=_REF_LOCALS))
    except (sp.SympifyError, TypeError, ValueError, AttributeError) as e:
        raise ValueError("reference %r could not be parsed as a math expression: %s"
                          % (text, e))


def _build_structure_from_spec(spec):
    # builds everything the Designer's schema needs except method/reasoning/
    # notes_limitations (pure transcription off the confirmed spec, no LLM involved)
    dyn = spec["dynamics"]
    states = list(dyn["states"])
    limitations = []

    by_output = {r["output"]: r.get("expr", "") for r in dyn["references"]}
    refs = []
    for out in dyn["outputs"]:
        raw_expr = by_output.get(out, "")
        try:
            parsed = _ref_from_expr(raw_expr)
        except ValueError as e:
            raise ValueError("reference for output %r: %s" % (out, e)) from e
        refs.append(RefSpec.model_validate({"expr": str(parsed)}).model_dump())

    def _split(entries):
        # entries with no KNOWN formula anywhere stay None (has_delta/
        # has_disturbance still True): exprs only get built once something actually names a formula
        known = [e for e in entries if (e.get("expr") or "").strip()]
        if not known:
            return None
        exprs = ["0"] * len(states)
        for e in known:
            if e.get("state") in states:
                exprs[states.index(e["state"])] = e["expr"]
        return exprs

    structure = {
        "states": states,
        "dynamics": list(dyn["state_equations"]),
        "inputs": list(dyn["inputs"]),
        "outputs": list(dyn["outputs"]),
        "x0": list(dyn["x0"]),
        "refs": refs,
        "has_delta": bool(dyn["uncertainty"]),
        "has_disturbance": bool(dyn["disturbance"]),
        "delta_exprs": _split(dyn["uncertainty"]),
        "dist_exprs": _split(dyn["disturbance"]),
    }
    return structure, limitations


def _verify_structure(structure):
    # sanity-parses the structure Python just built. no repair turn on
    # failure since a blowup here means a bad spec, not a bad model reply.
    states, dynamics = structure["states"], structure["dynamics"]
    inputs, outputs = structure["inputs"], structure["outputs"]
    delta_exprs, dist_exprs = structure["delta_exprs"], structure["dist_exprs"]
    state_syms, input_syms, output_syms, dyn_exprs, symbol_map = _parse_system(
        states, dynamics, inputs, outputs)
    x0_vals = coerce_float_list(structure["x0"], len(state_syms), "x0")
    if delta_exprs:
        for e in delta_exprs:
            _sympify(e, symbol_map)
    if dist_exprs:
        t_sym = sp.symbols("t")
        for e in dist_exprs:
            _sympify(e, {"t": t_sym})
    out = dict(structure)
    out["x0"] = x0_vals
    return out


def _validate_method(method, structure):
    # unlike _verify_structure, THIS is the model's own call, it picked method.
    # here's why the caller gives it one more turn to reconsider on failure
    inputs, outputs, states = structure["inputs"], structure["outputs"], structure["states"]
    if method == "smc":
        if len(inputs) != len(outputs):
            raise ValueError(
                "method='smc' requires a square system (#inputs == #outputs); "
                "got %d input(s) and %d output(s)." % (len(inputs), len(outputs)))
    elif outputs != [states[0]]:
        raise ValueError(
            "method='backstepping' requires outputs == [states[0]]; got "
            "outputs=%s with states[0]=%r." % (outputs, states[0]))
