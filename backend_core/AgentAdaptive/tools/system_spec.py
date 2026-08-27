import json
import re

import sympy as sp

_KNOWN_NAMES = frozenset({
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "exp", "log", "sqrt", "tanh", "sinh", "cosh", "Abs", "sign",
    "t", "pi", "E", "I",
})

_NAME_RE = re.compile(r"(?<![A-Za-z_0-9])([A-Za-z_][A-Za-z_0-9]*)")

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

# (dynamics field, clarifier category slug) pairs the Clarifier works through.
# system_type is deliberately left out here. See the tail of missing_items for why.
CHECKLIST = (
    ("outputs",         "output"),
    ("references",      "reference"),
    ("x0",              "initial_condition"),
    ("sim_time",        "sim_time"),
    ("solver_step",     "solver_step"),
    ("uncertainty",     "uncertainty_split"),
    ("inputs",          "states_inputs"),
    ("state_equations", "dynamics"),
)

DEFAULT_SIM_TIME = 8.0
DEFAULT_SOLVER_STEP = 0.001

SIM_TIME_PRESETS = (5.0, 10.0, 30.0, 100.0)
SOLVER_STEP_PRESETS = (0.0001, 0.001, 0.01)

# fields a pasted plant-agent JSON blob is trusted for. Everything else
# comes from the UI form or the Clarifier instead, see normalize_plant_spec
PLANT_FIELDS = ("states", "state_meanings", "inputs", "outputs",
                "state_equations", "parameters", "system_type", "assumptions")

STATUS_DRAFT = "draft"
STATUS_COMPLETE = "complete"
STATUS_INCOMPLETE = "incomplete"


def _text(value, fallback=""):
    if value is None:
        return fallback
    out = value if isinstance(value, str) else str(value)
    out = out.strip()
    return out if out else fallback


def _text_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        one = _text(value)
        return [one] if one else []
    out = []
    try:
        for item in value:
            one = _text(item)
            if one:
                out.append(one)
    except TypeError:
        return []
    return out


_LEADING_NUMBER_RE = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _float_or_none(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            match = _LEADING_NUMBER_RE.match(value)
            if match:
                try:
                    out = float(match.group(1))
                except ValueError:
                    return None
            else:
                return None
        else:
            return None
    if out != out or out in (float("inf"), float("-inf")):   # NaN / inf
        return None
    return out


def _float_list(value):
    if value is None:
        return []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        one = _float_or_none(value)
        return [] if one is None else [one]
    out = []
    try:
        for item in value:
            one = _float_or_none(item)
            if one is not None:
                out.append(one)
    except TypeError:
        return []
    return out


# matches "Delta = -0.9*x2" style entries so the "Delta =" part can be
# stripped off
_ASSIGNMENT_RE = re.compile(
    r"^\s*[A-Za-z_][A-Za-z_0-9]*\s*(?:\([^)]*\))?\s*=(?!=)\s*(.+)$", re.DOTALL)


def _expression_of(value):
    # strips a "Delta = -0.9*x2" LHS down to the expression: left in,
    # "Delta" survives as an undefined symbol (bit us once for real)
    text = _text(value)
    match = _ASSIGNMENT_RE.match(text)
    return _text(match.group(1)) if match else text


def _pair_list(value, key_name):
    # empty expr entries are kept, since "this term exists, no formula yet" is a
    # real answer; only an entry naming nothing at all gets dropped
    out = []
    if not value:
        return out
    if isinstance(value, dict):
        return [{key_name: _text(k), "expr": _expression_of(v)}
                for k, v in value.items() if _text(k) or _text(v)]
    try:
        items = list(value)
    except TypeError:
        return out
    for item in items:
        if isinstance(item, dict):
            name = _text(item.get(key_name) or item.get("name") or item.get("target"))
            expr = _expression_of(item.get("expr") or item.get("expression"))
        else:
            name, expr = "", _expression_of(item)
        if not name and not expr:
            continue
        out.append({key_name: name, "expr": expr})
    return out


def _parameter_map(value):
    # dropping non-numeric values here is not an accident: a symbol with no number
    # is exactly what the completeness check needs to catch, not hide
    out = {}
    if not value:
        return out
    items = value.items() if isinstance(value, dict) else None
    if items is None:
        try:
            pairs = []
            for item in value:
                if isinstance(item, dict):
                    pairs.append((item.get("name") or item.get("symbol"),
                                  item.get("value")))
            items = pairs
        except TypeError:
            return out
    for name, raw in items:
        key = _text(name)
        number = _float_or_none(raw)
        if key and number is not None and _IDENT_RE.fullmatch(key):
            out[key] = number
    return out


def empty_dynamics():
    return {
        "states": [],
        "state_meanings": [],
        "inputs": [],
        "outputs": [],
        "state_equations": [],
        "x0": [],
        "references": [],
        "parameters": {},
        "uncertainty": [],
        "disturbance": [],
        "system_type": "",
        "sim_time": None,
        "solver_step": None,
        "assumptions": [],
    }


def empty_spec():
    return {"status": STATUS_DRAFT, "system_name": "",
            "dynamics": empty_dynamics()}


def normalize_spec(raw):
    if not isinstance(raw, dict):
        return empty_spec()

    # nested "dynamics" is the contract, but accept fields spread on the top
    # level too: an empty nested block counts as "not really there".
    body = raw.get("dynamics")
    if not isinstance(body, dict) or not body:
        alt = raw.get("system")
        body = alt if isinstance(alt, dict) and alt else raw

    dyn = empty_dynamics()
    dyn["states"] = _text_list(body.get("states"))
    dyn["state_meanings"] = _text_list(body.get("state_meanings"))
    dyn["inputs"] = _text_list(body.get("inputs"))
    dyn["outputs"] = _text_list(body.get("outputs"))
    dyn["state_equations"] = _text_list(
        body.get("state_equations") or body.get("equations"))
    dyn["x0"] = _float_list(body.get("x0") or body.get("initial_conditions"))
    dyn["references"] = _pair_list(body.get("references"), "output")
    dyn["parameters"] = _parameter_map(body.get("parameters"))
    dyn["uncertainty"] = _pair_list(body.get("uncertainty"), "state")
    dyn["disturbance"] = _pair_list(body.get("disturbance"), "state")
    dyn["system_type"] = _text(body.get("system_type")).upper()
    if dyn["system_type"] not in ("SISO", "MIMO"):
        dyn["system_type"] = ""
    dyn["sim_time"] = _float_or_none(body.get("sim_time"))
    dyn["solver_step"] = _float_or_none(body.get("solver_step"))
    dyn["assumptions"] = _text_list(body.get("assumptions"))

    meanings = dyn["state_meanings"][:len(dyn["states"])]
    meanings += [""] * (len(dyn["states"]) - len(meanings))
    dyn["state_meanings"] = meanings

    _strip_placeholders(dyn)

    status = _text(raw.get("status")).lower()
    if status != STATUS_COMPLETE:
        status = STATUS_DRAFT
    return {
        "status": status,
        "system_name": _text(raw.get("system_name") or raw.get("name")),
        "dynamics": dyn,
    }


def normalize_plant_spec(raw):
    # only plant-agent fields survive here. sim_time/x0/refs/uncertainty are
    # stripped so a plant blob can't short-circuit the form or Clarifier
    spec = normalize_spec(raw)
    dyn = empty_dynamics()
    for field in PLANT_FIELDS:
        dyn[field] = spec["dynamics"][field]
    spec["dynamics"] = dyn
    spec["status"] = STATUS_DRAFT
    return spec


def merge_sim_knobs(spec, sim_time=None, solver_step=None, x0=None,
                    reference_exprs=None):
    # doesn't normalize reference_exprs text here - that's the Clarifier's job later.
    spec = normalize_spec(spec)
    dyn = dict(spec["dynamics"])
    if sim_time is not None:
        dyn["sim_time"] = _float_or_none(sim_time)
    if solver_step is not None:
        dyn["solver_step"] = _float_or_none(solver_step)
    if x0 is not None:
        dyn["x0"] = _float_list(x0)
    if reference_exprs is not None:
        dyn["references"] = _pair_list(
            [{"output": name, "expr": text} for name, text in reference_exprs.items()],
            "output")
    return {"status": spec["status"], "system_name": spec["system_name"], "dynamics": dyn}


def uncertainty_clarifier_view(spec):
    # wider than designer_view (includes equations/params) so the Clarifier
    # can reason, not just repeat the user. Safe, since its reply can't write back.
    dyn = normalize_spec(spec)["dynamics"]
    return {
        "states": dyn["states"],
        "inputs": dyn["inputs"],
        "outputs": dyn["outputs"],
        "state_equations": dyn["state_equations"],
        "parameters": dyn["parameters"],
        "references": dyn["references"],
    }


# wording is named in prompts/designer_agent_prompt.yaml, keep in sync
DESIGNER_HEADER = ("=== PLANT STRUCTURE (JSON, confirmed) ===")


def designer_view(spec):
    # this view is narrowed by design. x0/refs/uncertainty/disturbance/parameters are
    # withheld entirely since none affect method choice; Python builds the rest
    dyn = normalize_spec(spec)["dynamics"]
    return {
        "states": dyn["states"],
        "state_equations": dyn["state_equations"],
        "inputs": dyn["inputs"],
        "outputs": dyn["outputs"],
    }


def designer_view_to_text(spec):
    return "%s\n%s" % (DESIGNER_HEADER,
                        json.dumps(designer_view(spec), indent=2, ensure_ascii=False))


def _strip_placeholders(dyn):
    # strips a leftover uncertainty/disturbance placeholder (e.g. "Delta")
    # from its state equation: left in, sympy hits a NameError parsing it.
    states = dyn["states"]
    known = set(_KNOWN_NAMES) | set(states) | set(dyn["inputs"]) | set(dyn["parameters"])
    carries_term = {_text(e.get("state"))
                    for kind in ("uncertainty", "disturbance") for e in dyn[kind]}
    for i, name in enumerate(states):
        if name not in carries_term or i >= len(dyn["state_equations"]):
            continue
        equation = dyn["state_equations"][i]
        for symbol in set(_NAME_RE.findall(equation or "")) - known:
            equation = re.sub(r"\s*[-+]\s*(?<![A-Za-z_0-9])" + re.escape(symbol)
                              + r"(?![A-Za-z_0-9])", "", equation)
            equation = re.sub(r"(?<![A-Za-z_0-9])" + re.escape(symbol)
                              + r"(?![A-Za-z_0-9])", "0", equation)
        dyn["state_equations"][i] = equation.strip()
    return dyn


def undefined_symbols(spec):
    spec = normalize_spec(spec)
    dyn = spec["dynamics"]
    known = set(_KNOWN_NAMES)
    known.update(dyn["states"])
    known.update(dyn["inputs"])
    known.update(dyn["parameters"].keys())

    # a bare-name expr like "Delta" is a placeholder, not a constant awaiting
    # a value, so treat it as known instead of asking for a number
    expressions = list(dyn["state_equations"])
    for kind in ("uncertainty", "disturbance"):
        for entry in dyn[kind]:
            expr = (entry.get("expr") or "").strip()
            if _IDENT_RE.fullmatch(expr):
                known.add(expr)
            elif expr:
                expressions.append(expr)

    found = set()
    for expr in expressions:
        for name in _NAME_RE.findall(expr or ""):
            if name not in known:
                found.add(name)
    return sorted(found)


def _uncertainty_gaps(dyn):
    states = dyn["states"]
    gaps = []
    for kind in ("uncertainty", "disturbance"):
        for entry in dyn[kind]:
            if not entry.get("state"):
                gaps.append({"item": kind, "category": "uncertainty_split",
                             "detail": "A %s term is given (%s) without saying which "
                             "state equation it enters."
                             % (kind, entry.get("expr") or "unnamed")})
                break
            if states and entry["state"] not in states:
                gaps.append({"item": kind, "category": "uncertainty_split",
                             "detail": "A %s term is attached to %r, which is not one "
                             "of the states." % (kind, entry["state"])})
                break
    return gaps


def missing_items(spec):
    spec = normalize_spec(spec)
    dyn = spec["dynamics"]
    states = dyn["states"]
    outputs = dyn["outputs"]
    gaps = []

    def gap(item, category, detail):
        gaps.append({"item": item, "category": category, "detail": detail})

    if not states:
        gap("state_equations", "dynamics",
            "The state vector is not established: no states are named.")
    if not dyn["state_equations"]:
        gap("state_equations", "dynamics", "No state equations are given.")
    elif len(dyn["state_equations"]) != len(states):
        gap("state_equations", "dynamics",
            "There are %d state(s) but %d state equation(s); one equation per "
            "state is required." % (len(states), len(dyn["state_equations"])))

    undefined = undefined_symbols(spec)
    if undefined:
        gap("parameters", "dynamics",
            "The state equations use %s, which %s given a numeric value."
            % (", ".join(undefined), "is not" if len(undefined) == 1 else "are not"))

    if not dyn["inputs"]:
        gap("inputs", "states_inputs", "The control input(s) are not named.")

    if not outputs:
        gap("outputs", "output", "The tracked output(s) are not named.")
    else:
        not_states = [o for o in outputs if states and o not in states]
        if not_states:
            gap("outputs", "output",
                "Output(s) %s are not among the states %s. Every output must be "
                "one of the states." % (", ".join(not_states), ", ".join(states)))
        if len(set(outputs)) != len(outputs):
            gap("outputs", "output",
                "The same state is listed as more than one output, but outputs "
                "must be distinct.")

    referenced = set(r["output"] for r in dyn["references"] if r.get("expr"))
    unreferenced = [o for o in outputs if o not in referenced]
    if outputs and unreferenced:
        gap("references", "reference",
            "No reference signal is given for output(s) %s." % ", ".join(unreferenced))

    if states and len(dyn["x0"]) != len(states):
        gap("x0", "initial_condition",
            "The initial condition covers %d value(s) for %d state(s); every "
            "state needs a numeric value at t=0." % (len(dyn["x0"]), len(states)))

    if not dyn["sim_time"] or dyn["sim_time"] <= 0:
        gap("sim_time", "sim_time", "The simulation time is not stated.")

    # solver_step has no gap check (it's a judgment call): omitting it just
    # takes the default via normalize_defaults instead of asking

    gaps.extend(_uncertainty_gaps(dyn))

    # system_type is never gated. It just restates counts checked above,
    # and the Designer reads the method off the structure, not this label

    return gaps


def normalize_defaults(spec):
    spec = normalize_spec(spec)
    if not spec["dynamics"]["sim_time"] or spec["dynamics"]["sim_time"] <= 0:
        spec["dynamics"]["sim_time"] = DEFAULT_SIM_TIME
    if not spec["dynamics"]["solver_step"] or spec["dynamics"]["solver_step"] <= 0:
        spec["dynamics"]["solver_step"] = DEFAULT_SOLVER_STEP
    return spec


def sim_overrides_from_spec(spec):
    spec = normalize_defaults(spec)
    return {"t_end": float(spec["dynamics"]["sim_time"]),
            "dt": float(spec["dynamics"]["solver_step"])}


def substitute_parameters(spec):
    # bakes params into the equations numerically here in Python, not by the
    # LLM, since it's just arithmetic - no judgment call needed.
    spec = normalize_spec(spec)
    dyn = dict(spec["dynamics"])
    params = dyn["parameters"]
    if not params:
        return spec
    subs = {sp.Symbol(name): sp.Float(value) for name, value in params.items()}

    def _sub(expr_text):
        if not expr_text:
            return expr_text
        try:
            expr = sp.sympify(expr_text)
        except (sp.SympifyError, TypeError, ValueError, AttributeError):
            return expr_text
        try:
            return str(expr.subs(subs))
        except Exception:
            return expr_text

    dyn["state_equations"] = [_sub(e) for e in dyn["state_equations"]]
    dyn["uncertainty"] = [dict(e, expr=_sub(e.get("expr"))) for e in dyn["uncertainty"]]
    dyn["disturbance"] = [dict(e, expr=_sub(e.get("expr"))) for e in dyn["disturbance"]]
    dyn["parameters"] = {}
    return {"status": spec["status"], "system_name": spec["system_name"], "dynamics": dyn}


def spec_to_json(spec, indent=2):
    try:
        return json.dumps(normalize_defaults(spec), indent=indent, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(empty_spec(), indent=indent)
