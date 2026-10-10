# plain-language notes about what the output bound and the input limit can and cannot promise.
# python only, built from the confirmed spec and the run's own numbers, shown to the user
# in the wizard, before the results, and in the report's notes
from backend_core.AgentAdaptive.tools import system_spec as ss


def _dyn(spec):
    return ss.normalize_spec(spec)["dynamics"]


def setup_notes(spec, t_end=None):
    # known as soon as the setup is confirmed (no method chosen yet)
    from backend_core.AgentAdaptive.controller import output_bounds as ob
    dyn = _dyn(spec)
    notes = []
    if dyn["output_bounds"]:
        kb = ob.error_limits(spec, t_end)
        if kb:
            notes.append("Output bound: %s. The error limit is the smallest distance from the reference to a "
                         "bound, applied on both sides. Near the edge the controller can ask for much more "
                         "control effort." % ", ".join("%s gets an error limit of %.4g" % (o, k)
                                                       for o, k in kb.items()))
    if dyn["output_bounds"] and dyn["input_limits"]:
        notes.append("Output bound together with an input limit: the bound is only guaranteed while the "
                     "actuator is not clipping. If the input limit is reached, the output can leave its "
                     "bound, and the run then fails the output-bound check.")
    return notes


def _rel_degrees(spec):
    from backend_core.AgentAdaptive.controller.plant_analysis import lie_chain
    from backend_core.AgentAdaptive.controller.structure_build import _parse_system
    dyn = ss.substitute_parameters(spec)["dynamics"]
    states, inputs, outputs, f, _ = _parse_system(dyn["states"], dyn["state_equations"],
                                                  dyn["inputs"], dyn["outputs"])
    return [lie_chain(o, states, inputs, f, output_name=str(o))["relative_degree"] for o in outputs]


def method_notes(spec, method):
    # once the design method is known
    dyn = _dyn(spec)
    notes = []
    if not dyn["output_bounds"]:
        return notes
    try:
        if method == "smc" and (dyn["uncertainty"] or dyn["disturbance"]):
            degrees = _rel_degrees(spec)
            if max(degrees) >= 3:
                notes.append("Sliding mode with relative degree %d and uncertainty: the output-bound guarantee is "
                             "proven for relative degree 1 or 2, or when nothing acts on the lower levels. Here "
                             "it is not proven; the run is checked and fails if the output leaves its bound."
                             % max(degrees))
        if method == "backstepping" and len(dyn["states"]) >= 4:
            notes.append("Output bound with a backstepping chain of %d states: building the control law can take "
                         "one to two minutes." % len(dyn["states"]))
    except Exception:  # noqa: BLE001 -- a note is never worth failing a run
        pass
    return notes


def run_notes(metrics):
    # what actually happened: the limit was reached while a bound was active
    metrics = metrics or {}
    sat, bounds = metrics.get("input_saturation"), metrics.get("output_bounds")
    if sat and bounds and (sat.get("any_input_fraction") or 0.0) >= 0.01:
        return ["The input limit was reached in %.0f%% of the run, so the output bound was not guaranteed during "
                "that time. The output %s its bound."
                % (100.0 * sat["any_input_fraction"],
                   "left" if bounds.get("any_violation") else "stayed inside")]
    return []


def limit_feasibility_notes(inputs, limits, free_metrics):
    # before the real run: what the same design asks for with no limit at all, set against the limit
    if not limits or not any(lim is not None for lim in limits):
        return []
    free_metrics = free_metrics or {}
    if free_metrics.get("numerically_healthy") is False or free_metrics.get("control_max") is None:
        return ["Without the input limit this design already diverges, so the limit cannot be judged."]
    notes = []
    for name, lim, peak in zip(inputs, limits, free_metrics["control_max"]):
        sides = [abs(v) for v in (lim or []) if v is not None]
        if sides and peak > max(sides):
            notes.append("Without limits this design asks for up to %.4g on %s, but your limit is %.4g. "
                         "Expect the command to be clipped%s." % (
                             peak, name, max(sides),
                             "; the design also fails without any limit" if not free_metrics.get("success", True)
                             else ", and the output may not track"))
    return notes


def all_notes(spec, method=None, metrics=None, t_end=None):
    return setup_notes(spec, t_end) + (method_notes(spec, method) if method else []) + run_notes(metrics)
