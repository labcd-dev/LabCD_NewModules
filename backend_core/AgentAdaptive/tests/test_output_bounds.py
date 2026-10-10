"""
Run from the repo root with:
    python -m pytest backend_core/AgentAdaptive/tests/test_output_bounds.py -v
"""

import contextlib
import io
import json
import os

import numpy as np
import pytest

from backend_core.AgentAdaptive.agents import tuner_agent as ta
from backend_core.AgentAdaptive.controller import backstepping_design, smc_design
from backend_core.AgentAdaptive.controller import output_bounds as ob
from backend_core.AgentAdaptive.controller.structure_build import (
    _build_structure_from_spec, _parse_system, _verify_structure)
from backend_core.AgentAdaptive.tools import bound_advice, diagnosis_chain as dc
from backend_core.AgentAdaptive.tools import failure_mode as fm
from backend_core.AgentAdaptive.tools import scoring, system_spec

CP_EQS = ["x2", "(-(u + 0.06*x2**2*sin(x1))*cos(x1) + 11.2815*sin(x1))/(0.06*sin(x1)**2 + 0.4)"]


def _spec(eqs=CP_EQS, x0=(0.2, 0.0), ref="0", bounds=None, limits=None, t_end=3.0):
    dyn = {"states": ["x1", "x2"], "inputs": ["u"], "outputs": ["x1"], "state_equations": eqs,
           "x0": list(x0), "references": [{"output": "x1", "expr": ref}], "solver_step": 0.001,
           "sim_time": t_end}
    if bounds:
        dyn["output_bounds"] = [{"output": "x1", "min": bounds[0], "max": bounds[1]}]
    if limits:
        dyn["input_limits"] = [{"input": "u", "min": limits[0], "max": limits[1]}]
    return system_spec.normalize_defaults({"status": "complete", "system_name": "P", "dynamics": dyn})


def _quiet_run(args):
    with contextlib.redirect_stdout(io.StringIO()):
        return ta._rerun_design_directly(args, for_tuning=True)


def _args(method, spec, **kw):
    st = _verify_structure(_build_structure_from_spec(spec)[0])
    args = dict(st, method=method, reasoning="t", notes_limitations=None, dt=0.001,
                t_end=float(spec["dynamics"]["sim_time"]), fail_tol=0.05)
    args.update(kw)
    return args


# ---------- the numbers ----------

def test_the_barrier_weight():
    assert ob.weight(0.0, 1.0) == pytest.approx(1.0)
    assert ob.weight(0.5, 1.0) == pytest.approx(1.0 / 0.75)
    assert ob.weight(-1.0, 2.0) == pytest.approx(4.0 / 3.0)
    assert ob.weight(5.0, 1.0) == pytest.approx(1.0 / (1.0 - 0.98 ** 2))     # clipped, finite
    assert ob.weight(5.0, np.inf) == 1.0                                       # no bound, no change


def test_half_width_is_the_smallest_distance_to_a_given_bound():
    ref = np.array([[0.0], [1.0], [-0.5]])
    assert ob.half_widths([[-3.0, 2.0]], ref)[0] == pytest.approx(1.0)
    assert ob.half_widths([[None, 2.0]], ref)[0] == pytest.approx(1.0)
    assert ob.half_widths([[-3.0, None]], ref)[0] == pytest.approx(2.5)
    assert np.isinf(ob.half_widths([None], ref)[0])
    assert ob.half_widths([[-1.57, 1.57]], np.zeros((5, 1)))[0] == pytest.approx(1.57)
    assert ob.half_widths([[-0.5, 0.5]], ref)[0] <= 0.0                         # reference touches the edge


def test_start_checks():
    assert ob.start_problems(["y"], [[-1, 1]], np.array([0.8]), [1.2], [0.0])[0]["kind"] == "start_outside"
    far = ob.start_problems(["y"], [[-1, 1]], np.array([0.8]), [0.9], [0.0])[0]
    assert far["kind"] == "start_too_far" and far["e0"] == pytest.approx(0.9)
    assert ob.start_problems(["y"], [[-1, 1]], np.array([0.8]), [0.5], [0.0]) == []


# ---------- the control laws do what the Lyapunov algebra says ----------

def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def test_backstepping_law_makes_the_lyapunov_derivative_negative_as_derived():
    # V = kb^2/2 ln(kb^2/(kb^2-z0^2)) + z1^2/2 ; Vdot must be -c0 w^2 z0^2 - c1 z1^2, computed here
    # by hand from the plant (not from the code under test) and compared with a finite difference
    st, ins, outs, f, _ = _parse_system(["x1", "x2"], ["0.3*sin(x1) + x2", "0.2*cos(x1) + u"], ["u"], ["x1"])
    system = {"states": st, "inputs": ins, "outputs": outs, "f": f}
    kb, c0, c1 = 1.0, 2.0, 4.0
    controller = _quiet(backstepping_design.design_backstepping, system, False, c_gains=[c0, c1],
                        output_bound=kb)[0]

    def ref(t):
        return 0.3 * np.sin(t), 0.3 * np.cos(t), -0.3 * np.sin(t)

    def V(x, t):
        r, rd, _ = ref(t)
        z0 = x[0] - r
        w = kb ** 2 / (kb ** 2 - z0 ** 2)
        a0 = (-c0 * w * z0 - 0.3 * np.sin(x[0]) + rd)
        return kb ** 2 / 2 * np.log(kb ** 2 / (kb ** 2 - z0 ** 2)) + 0.5 * (x[1] - a0) ** 2

    for x, t in (([0.6, 1.0], 0.0), ([-0.4, 0.3], 1.3), ([0.2, -2.0], 2.1)):
        x = np.array(x)
        r, rd, rdd = ref(t)
        u = controller(list(x), [[r, rd, rdd]])[0]
        xdot = np.array([0.3 * np.sin(x[0]) + x[1], 0.2 * np.cos(x[0]) + u])
        h = 1e-7
        numeric = (V(x + h * xdot, t + h) - V(x, t)) / h
        z0 = x[0] - r
        w = kb ** 2 / (kb ** 2 - z0 ** 2)
        z1 = x[1] - (-c0 * w * z0 - 0.3 * np.sin(x[0]) + rd)
        assert numeric == pytest.approx(-c0 * w ** 2 * z0 ** 2 - c1 * z1 ** 2, rel=1e-4, abs=1e-3)


def test_command_filtered_backstepping_uses_the_same_weight():
    # first call: filter state = command, xi = gamma = 0, estimates 0, so the command is the plain law
    st, ins, outs, f, _ = _parse_system(["x1", "x2"], ["0.3*sin(x1) + x2", "0.2*cos(x1) + u"], ["u"], ["x1"])
    system = {"states": st, "inputs": ins, "outputs": outs, "f": f}
    kb, c0, c1 = 1.0, 2.0, 4.0
    adaptive = _quiet(backstepping_design.design_backstepping, system, True, c_gains=[c0, c1],
                      output_bound=kb)[2]
    x, r, rd = np.array([0.6, 1.0]), 0.1, 0.2
    adaptive.reset(x)
    u = adaptive.compute(list(x), [[r, rd, 0.0]])[0]
    z0 = x[0] - r
    w = kb ** 2 / (kb ** 2 - z0 ** 2)
    xc1 = -c0 * w * z0 - 0.3 * np.sin(x[0]) + rd          # g0 = 1
    z1 = x[1] - xc1                                         # filter state starts at the command
    assert u == pytest.approx(-c1 * z1 - 1.0 * w * z0 - 0.2 * np.cos(x[0]), rel=1e-6)


def _smc_sdot_check(rho, lam=2.0, kb=1.0, K=4.0, Lam=5.0, phi=0.05):
    # chain of integrators x1^(rho) = u. s = (d/dt + lam)^(rho-1) zeta, zeta = kb atanh(e/kb), written
    # here by hand from the chain rule. the law must give sdot = -K w sat(s/phi) - Lam s exactly
    names = ["x%d" % (k + 1) for k in range(rho)]
    eqs = names[1:] + ["u"]
    st, ins, outs, f, _ = _parse_system(names, eqs, ["u"], ["x1"])
    system = {"states": st, "inputs": ins, "outputs": outs, "f": f}
    controller = _quiet(smc_design.design_smc, system, False, surface_lambda=lam, K=K, Lam=Lam,
                        phi_layer=phi, output_bounds_kb=np.array([kb]))[0]

    def surface(x, t):
        r = [0.3 * np.sin(t), 0.3 * np.cos(t), -0.3 * np.sin(t), -0.3 * np.cos(t)]
        E = [x[k] - r[k] for k in range(rho)]
        w = kb ** 2 / (kb ** 2 - E[0] ** 2)
        z = kb * np.arctanh(E[0] / kb)
        if rho == 1:
            return z
        zd = w * E[1]
        if rho == 2:
            return zd + lam * z
        zdd = w * E[2] + (2 * E[0] * w ** 2 / kb ** 2) * E[1] ** 2
        return zdd + 2 * lam * zd + lam ** 2 * z

    r_all = lambda t: [0.3 * np.sin(t), 0.3 * np.cos(t), -0.3 * np.sin(t), -0.3 * np.cos(t)]  # noqa: E731
    for x0, t in (([0.5, 0.7, -0.4][:rho], 0.0), ([-0.4, 0.3, 0.9][:rho], 1.3), ([0.2, -1.0, 0.5][:rho], 2.1)):
        x = np.array(x0, dtype=float)
        u = controller(list(x), [r_all(t)[:rho + 1]])[0]
        xdot = np.array(list(x[1:]) + [u])
        h = 1e-7
        numeric = (surface(x + h * xdot, t + h) - surface(x, t)) / h
        s0 = surface(x, t)
        w = kb ** 2 / (kb ** 2 - (x[0] - r_all(t)[0]) ** 2)
        expected = -K * w * np.clip(s0 / phi, -1, 1) - Lam * s0
        assert numeric == pytest.approx(expected, rel=2e-4, abs=2e-3)


@pytest.mark.parametrize("rho", [1, 2, 3])
def test_smc_surface_on_the_transformed_error_gives_the_derived_dynamics(rho):
    _smc_sdot_check(rho)


def test_smc_needs_no_start_condition_beyond_the_error_being_inside():
    # the old surface barrier refused x0 = (1, 6) (its surface started at 8 against a cap of 3.1).
    # on the transformed error only |e(0)| < kb matters
    m = _quiet_run(_args("smc", _spec(x0=(1.0, 6.0), bounds=(-1.57, 1.57), t_end=4.0)))[1]
    assert m["success"] is True and m["output_bounds"]["any_violation"] is False


def test_smc_two_outputs_with_a_bound_on_one_only():
    eqs = ["x2", "u1 + 0.5*sin(x3)", "x4", "u2"]
    spec = system_spec.normalize_defaults({"status": "complete", "system_name": "M", "dynamics": {
        "states": ["x1", "x2", "x3", "x4"], "inputs": ["u1", "u2"], "outputs": ["x1", "x3"],
        "state_equations": eqs, "x0": [0.8, 0.5, 0.0, 0.0],
        "references": [{"output": "x1", "expr": "0"}, {"output": "x3", "expr": "0.5*sin(t)"}],
        "solver_step": 0.001, "sim_time": 6.0,
        "output_bounds": [{"output": "x1", "min": -1.0, "max": 1.0}]}})
    m = _quiet_run(_args("smc", spec))[1]
    assert m["success"] is True
    assert m["output_bounds"]["per_output"][1] is None and m["output_bounds"]["per_output"][0]["margin"] > 0


def test_without_a_bound_nothing_changes():
    st, ins, outs, f, _ = _parse_system(["x1", "x2"], ["x2", "u"], ["u"], ["x1"])
    system = {"states": st, "inputs": ins, "outputs": outs, "f": f}
    plain = _quiet(smc_design.design_smc, system, False)[0]
    huge = _quiet(smc_design.design_smc, system, False, output_bounds_kb=np.array([1e9]))[0]
    inf = _quiet(smc_design.design_smc, system, False, output_bounds_kb=np.array([np.inf]))[0]
    x, ref = [0.5, 0.7], [[0.1, 0.2, -0.05]]
    assert inf(x, ref) == plain(x, ref)
    assert huge(x, ref)[0] == pytest.approx(plain(x, ref)[0], rel=1e-9)
    bs_plain = _quiet(backstepping_design.design_backstepping, system, False)[0]
    bs_inf = _quiet(backstepping_design.design_backstepping, system, False, output_bound=np.inf)[0]
    assert bs_inf(x, ref) == bs_plain(x, ref)


# ---------- closed loop ----------

def test_backstepping_keeps_the_cart_pole_inside_where_the_free_design_leaves():
    # starting at 1.0 with velocity 7 the free design swings past 1.57 (the singularity) and explodes
    free = _quiet_run(_args("backstepping", _spec(x0=(1.0, 7.0), t_end=6.0)))[1]
    bound = _quiet_run(_args("backstepping", _spec(x0=(1.0, 7.0), bounds=(-1.57, 1.57), t_end=6.0)))[1]
    assert free["success"] is False
    assert bound["success"] is True
    assert bound["output_bounds"]["any_violation"] is False
    assert bound["output_bounds"]["per_output"][0]["margin"] > 0.2        # about 0.31 measured


def test_smc_with_a_bound_runs_and_stays_inside():
    m = _quiet_run(_args("smc", _spec(x0=(1.0, 1.0), bounds=(-1.57, 1.57), t_end=4.0)))[1]
    assert m["success"] is True and m["output_bounds"]["any_violation"] is False


def test_a_bound_violation_fails_the_run_and_the_input_limit_is_named_as_the_cause():
    lin = ["x2", "-x1 - 0.5*x2 + u"]
    spec = _spec(eqs=lin, x0=(0.0, 0.0), ref="1.5*sin(t)", bounds=(-1.7, 1.7), limits=(-3.0, 3.0), t_end=15.0)
    m = _quiet_run(_args("backstepping", spec))[1]
    assert m["success"] is False and m["success_reason"].startswith("output_bounds")
    assert m["success_checks"]["output_bounds"] is False
    assert m["output_bounds"]["per_output"][0]["margin"] < 0
    free = _quiet_run(_args("backstepping", _spec(eqs=lin, x0=(0.0, 0.0), ref="1.5*sin(t)",
                                                  bounds=(-1.7, 1.7), t_end=15.0)))[1]
    assert free["success"] is True                                          # same design, no actuator limit


# ---------- scoring ----------

def test_verdict_has_the_bound_check_only_when_a_bound_is_given():
    t = np.linspace(0, 1, 100)
    y = np.linspace(0.0, 2.0, 100).reshape(-1, 1)
    ref = y.copy()
    u = np.zeros((100, 1))
    plain = scoring.compute_success_verdict(t, y, ref, u, y, 0.01)
    assert "output_bounds" not in plain["checks"] and plain["success"]
    bounds = scoring.bound_arrays([[-1.0, 1.5]], 1)
    verdict = scoring.compute_success_verdict(t, y, ref, u, y, 0.01, output_bounds=bounds)
    assert verdict["success"] is False and verdict["reason"].startswith("output_bounds")
    stats = scoring.output_bound_stats(t, y, bounds)
    assert stats["first_violation_s"] == pytest.approx(t[np.flatnonzero(y[:, 0] > 1.5)[0]])
    assert stats["per_output"][0]["margin"] == pytest.approx(-0.5)
    inside = scoring.compute_success_verdict(t, y, ref, u, y, 0.01,
                                             output_bounds=scoring.bound_arrays([[-1.0, 2.5]], 1))
    assert inside["success"] and inside["checks"]["output_bounds"] is True


# ---------- the spec and the wizard data ----------

def test_the_spec_keeps_output_bounds_and_flags_bad_ones():
    spec = system_spec.merge_sim_knobs(
        _spec(), output_bounds=[{"output": "x1", "min": -1.0, "max": 1.0}, {"output": "zz", "min": 0, "max": 1},
                                {"output": "x1", "min": None, "max": None}])
    assert [b["output"] for b in spec["dynamics"]["output_bounds"]] == ["x1", "zz"]
    assert any("zz" in p for p in system_spec.output_bound_problems(spec))
    bad = system_spec.merge_sim_knobs(_spec(), output_bounds=[{"output": "x1", "min": 2.0, "max": 1.0}])
    assert "needs min < max" in system_spec.output_bound_problems(bad)[0]
    assert _build_structure_from_spec(_spec(bounds=(-1.0, 1.0)))[0]["output_bounds"] == [[-1.0, 1.0]]
    assert _build_structure_from_spec(_spec())[0]["output_bounds"] is None


def test_pre_run_check_names_the_problem_and_is_quiet_when_fine():
    assert ob.check_output_bounds(_spec(bounds=(-1.57, 1.57))) == []
    assert ob.check_output_bounds(_spec()) == []
    out = ob.check_output_bounds(_spec(ref="0.5*sin(5*t)", bounds=(-0.3, 0.3)))
    assert out[0]["kind"] == "reference_outside"
    assert out[0]["reference_span"][1] == pytest.approx(0.5, abs=1e-3)
    assert ob.check_output_bounds(_spec(x0=(0.3, 0.0), bounds=(-0.25, 0.25)))[0]["kind"] == "start_outside"
    assert ob.check_output_bounds(_spec(x0=(0.248, 0.0), bounds=(-0.25, 0.25)))[0]["kind"] == "start_too_far"
    assert ob.check_output_bounds(_spec(x0=(0.2, 0.0), bounds=(-0.25, 0.25))) == []


def test_the_advice_cards_are_computed_and_checked():
    spec = _spec(ref="0.5*sin(5*t)", bounds=(-0.3, 0.3))
    problems = ob.check_output_bounds(spec)
    card = bound_advice.diagnosis_for(spec, problems, 3.0)["report"]["suggestions"][0]
    assert card["lever"] == "reference" and card["computed"] and card["target"] == "x1"
    from backend_core.AgentAdaptive.agents.diagnoser_agent import _reference_span
    for opt in card["options"]:
        lo, hi = _reference_span(opt["value"], 3.0)
        assert -0.3 < lo and hi < 0.3
    # a start outside: move that state to the reference
    spec2 = _spec(x0=(0.3, 0.0), bounds=(-0.25, 0.25))
    start = bound_advice.diagnosis_for(spec2, ob.check_output_bounds(spec2), 3.0)["report"]["suggestions"][0]
    assert start["lever"] == "initial_condition" and start["options"][0]["value"] == "0,0"
    # too far: half of the error limit away from the reference, same side
    spec3 = _spec(x0=(0.248, 0.0), bounds=(-0.25, 0.25))
    far = bound_advice.diagnosis_for(spec3, ob.check_output_bounds(spec3), 3.0)["report"]["suggestions"][0]
    assert far["options"][0]["value"] == "0.125,0"


def test_a_blocked_run_never_calls_the_designer(monkeypatch):
    os.environ["LABCD_ADAPTIVE_SHOW_PLOTS"] = "0"

    def boom(*a, **k):
        raise AssertionError("the designer must not run")
    monkeypatch.setattr(ta, "run_extraction", boom)
    spec = _spec(ref="0.5*sin(5*t)", bounds=(-0.3, 0.3))
    result, usage, log, best = ta.run_full_pipeline("", system_spec=spec)
    assert result["messages"][-1].content.startswith("OUTPUT BOUND NOT USABLE")
    assert result["final_metrics"] is None and best is None and log == []
    diag = result["diagnosis"]
    assert diag["evidence"]["blocked_before_run"] is True
    assert diag["report"]["suggestions"][0]["lever"] == "reference"


# ---------- the diagnoser ----------

def _viol_ev(clipped=True, free_pass=True):
    ev = {"output_bounds": {"bounds": {"x1": [-1.7, 1.7]}, "error_limit_kb": {"x1": 0.2},
                            "seen": {"x1": {"min": -1.76, "max": 1.76, "margin": -0.06, "violated": True}},
                            "violated": True, "first_violation_s": 1.6},
          "metrics": {"tracking_pct_headline": 100.0}}
    if clipped:
        ev["input_limits"] = {"limits": {"u": [-3.0, 3.0]},
                              "saturation": {"any_input_fraction": 0.3, "first_saturated_at_s": 1.5},
                              "without_limits": {"passes": free_pass, "tracking_pct": 99.0, "failed_check": None,
                                                 "peak_by_input": {"u": 9.0}}}
    return ev


def test_failure_modes_for_a_bound_violation():
    ev = _viol_ev()
    ev["failure_mode"] = fm.classify(ev)
    assert ev["failure_mode"]["mode"] == "bound_violated_by_limit"
    text = " ".join(ev["failure_mode"]["chain"])
    assert "left its allowed range [-1.7, 1.7]" in text and "first outside at t=1.6 s" in text
    assert "WITHOUT the input limit stays inside" in text
    assert "clipped" in fm.headline(ev) and fm.limit_card(ev)["options"][0]["value"] == "-12,12"
    assert fm.bounds_card(ev) is None                                       # the limit is the fix, not the bound
    plain = _viol_ev(clipped=False)
    plain["failure_mode"] = fm.classify(plain)
    assert plain["failure_mode"]["mode"] == "output_bound_violated"
    assert "The input was not clipped. The barrier works on an error limit of x1=0.2." in plain["failure_mode"]["chain"]
    card = fm.bounds_card(plain)
    assert card["lever"] == "output_bounds" and card["target"] == "x1" and card["computed"]
    assert card["options"][0]["value"] == "-2.112,2.112"            # seen range plus 10% of its width
    assert fm.headline(plain).startswith("The output x1 left its allowed range")
    still = _viol_ev(free_pass=False)
    still["failure_mode"] = fm.classify(still)
    assert still["failure_mode"]["mode"] == "output_bound_violated"


def test_the_chain_remembers_output_bound_edits():
    a = dc.snapshot(_spec(bounds=(-1.0, 1.0)))
    b = dc.snapshot(_spec(bounds=(-2.0, 2.0)))
    assert dc.diff(a, b) == [{"lever": "output_bounds", "what": "output bound for x1", "target": "x1",
                              "from": [-1.0, 1.0], "to": [-2.0, 2.0]}]
    assert dc._same_value("output_bounds", [-2.0, 2.0], "-2,2")
    assert not dc._same_value("output_bounds", [-2.0, 2.0], "-3,3")
    removed = dc.diff(b, dc.snapshot(_spec()))
    assert "output bound for x1 was removed (it was [-2, 2])" in dc._change_words(
        dict(removed[0], followed=None))


def test_pipeline_end_to_end_a_violation_gets_a_computed_diagnosis(monkeypatch):
    os.environ["LABCD_ADAPTIVE_SHOW_PLOTS"] = "0"
    from backend_core.AgentAdaptive.agents import llm_factory
    from backend_core.AgentAdaptive.agents.agent_io import _SyntheticMessage, _empty_usage

    def fake_extraction(spec, on_event=None, clarification_record=None):
        st = _verify_structure(_build_structure_from_spec(spec)[0])
        args = dict(st, method="backstepping", reasoning="t", notes_limitations=None)
        use = {"agent": _empty_usage(), "total": _empty_usage(), "agent_turns": [], "timeline": []}
        return {"messages": [_SyntheticMessage("d")]}, use, {"ok": True, "report": "", "args": args, "why": "t"}

    class Reply:
        content = json.dumps({"headline": "h", "explanation": ["e"], "suggestions": []})
        usage_metadata = {}

    class LLM:
        def invoke(self, msgs):
            return Reply()

    monkeypatch.setattr(ta, "run_extraction", fake_extraction)
    monkeypatch.setattr(ta, "write_abstract", lambda *a, **k: ("abs", ta._empty_usage()))
    monkeypatch.setattr(llm_factory, "build_llm", lambda role, **k: LLM())
    spec = _spec(eqs=["x2", "-x1 - 0.5*x2 + u"], x0=(0.0, 0.0), ref="1.5*sin(t)",
                 bounds=(-1.7, 1.7), limits=(-3.0, 3.0), t_end=15.0)
    result, *_ = ta.run_full_pipeline("", system_spec=spec)
    diag = result["diagnosis"]
    assert result["final_metrics"]["success_reason"].startswith("output_bounds")
    assert diag["evidence"]["failure_mode"]["mode"] == "bound_violated_by_limit"
    assert diag["report"]["headline"].startswith("The output left its allowed range because the input limit")
    assert diag["report"]["suggestions"][0]["lever"] == "input_limits"
    notes = result["constraint_notes"]
    assert any("only guaranteed while the actuator is not clipping" in n for n in notes)
    assert any("The input limit was reached in" in n and "left its bound" in n for n in notes)
    assert "Notes & Limitations" in result["messages"][-1].content
    assert "The input limit was reached in" in result["messages"][-1].content


# ---------- the notes the user is told ----------

def test_setup_notes_say_what_the_bound_and_the_limit_can_promise():
    from backend_core.AgentAdaptive.tools import constraint_notes as cn
    assert cn.setup_notes(_spec()) == []
    only = cn.setup_notes(_spec(ref="0.15*sin(1.2*t)", bounds=(-0.4, 0.4), t_end=8.0))
    assert len(only) == 1 and "x1 gets an error limit of 0.25" in only[0]
    both = cn.setup_notes(_spec(ref="0.15*sin(1.2*t)", bounds=(-0.4, 0.4), limits=(-5, 5), t_end=8.0))
    assert len(both) == 2 and "only guaranteed while the actuator is not clipping" in both[1]
    assert len(cn.setup_notes(_spec(limits=(-5, 5)))) == 0                      # a limit alone promises nothing


def test_method_notes_only_where_the_claim_is_weaker_or_slow():
    from backend_core.AgentAdaptive.tools import constraint_notes as cn
    chain4 = system_spec.normalize_defaults({"status": "complete", "system_name": "C", "dynamics": {
        "states": ["x1", "x2", "x3", "x4"], "inputs": ["u"], "outputs": ["x1"],
        "state_equations": ["x2", "x3", "x4", "u"], "x0": [0, 0, 0, 0],
        "references": [{"output": "x1", "expr": "0"}], "solver_step": 0.001, "sim_time": 3.0,
        "output_bounds": [{"output": "x1", "min": -1, "max": 1}],
        "uncertainty": [{"state": "x2", "expr": "0.1*sin(x1)"}]}})
    smc = cn.method_notes(chain4, "smc")
    assert len(smc) == 1 and "relative degree 4" in smc[0] and "not proven" in smc[0]
    bs = cn.method_notes(chain4, "backstepping")
    assert len(bs) == 1 and "4 states" in bs[0]
    assert cn.method_notes(_spec(bounds=(-1, 1)), "smc") == []                  # degree 2, nothing to add
    assert cn.method_notes(_spec(), "smc") == []


def test_run_notes_report_what_happened():
    from backend_core.AgentAdaptive.tools import constraint_notes as cn
    def m(frac, viol):
        return {"input_saturation": {"any_input_fraction": frac}, "output_bounds": {"any_violation": viol}}
    assert "reached in 30% of the run" in cn.run_notes(m(0.3, True))[0] and "left its bound" in cn.run_notes(m(0.3, True))[0]
    assert "stayed inside its bound" in cn.run_notes(m(0.3, False))[0]
    assert cn.run_notes(m(0.0, False)) == [] and cn.run_notes({}) == []


@pytest.mark.parametrize("method", ["smc", "backstepping"])
def test_the_report_carries_the_output_bound_proof_only_when_a_bound_is_set(method):
    def stability(spec):
        args = _args(method, spec, t_end=1.0)
        with contextlib.redirect_stdout(io.StringIO()):
            comps, _ = ta._rerun_design_directly(args, for_tuning=False)
        return comps["stability"]
    assert "Output bound --" in stability(_spec(bounds=(-1.57, 1.57)))
    assert "Output bound --" not in stability(_spec())


# ---------- the input limit is judged before the real run ----------

def test_limit_feasibility_note_compares_the_free_peak_with_the_limit():
    from backend_core.AgentAdaptive.tools import constraint_notes as cn
    free = {"control_max": [3478.98], "success": True}
    note = cn.limit_feasibility_notes(["u"], [[None, 20.0]], free)[0]
    assert "asks for up to 3479 on u, but your limit is 20" in note and "may not track" in note
    failing = cn.limit_feasibility_notes(["u"], [[-10.0, 10.0]], {"control_max": [50.0], "success": False})[0]
    assert "the design also fails without any limit" in failing
    assert cn.limit_feasibility_notes(["u"], [[-10.0, 10.0]], {"control_max": [9.9], "success": True}) == []
    assert cn.limit_feasibility_notes(["u"], [None], free) == []
    assert cn.limit_feasibility_notes(["u"], None, free) == []
    assert "already diverges" in cn.limit_feasibility_notes(
        ["u"], [[-1.0, 1.0]], {"numerically_healthy": False})[0]


def test_pipeline_warns_before_the_real_run_when_the_limit_is_below_the_free_peak(monkeypatch):
    # cart-pole 0.5 sin(5t) needs about 13.9; the limit is 10
    os.environ["LABCD_ADAPTIVE_SHOW_PLOTS"] = "0"
    from backend_core.AgentAdaptive.agents import llm_factory
    from backend_core.AgentAdaptive.agents.agent_io import _SyntheticMessage, _empty_usage

    def fake_extraction(spec, on_event=None, clarification_record=None):
        st = _verify_structure(_build_structure_from_spec(spec)[0])
        args = dict(st, method="backstepping", reasoning="t", notes_limitations=None)
        use = {"agent": _empty_usage(), "total": _empty_usage(), "agent_turns": [], "timeline": []}
        return {"messages": [_SyntheticMessage("d")]}, use, {"ok": True, "report": "", "args": args, "why": "t"}

    class Reply:
        content = json.dumps({"headline": "h", "explanation": ["e"], "suggestions": []})
        usage_metadata = {}

    class LLM:
        def invoke(self, msgs):
            return Reply()

    monkeypatch.setattr(ta, "run_extraction", fake_extraction)
    monkeypatch.setattr(ta, "write_abstract", lambda *a, **k: ("abs", ta._empty_usage()))
    monkeypatch.setattr(llm_factory, "build_llm", lambda role, **k: LLM())
    events = []
    spec = _spec(x0=(0.2, 0.0), ref="0.5*sin(5*t)", limits=(-10.0, 10.0), t_end=10.0)
    result, *_ = ta.run_full_pipeline("", system_spec=spec, on_event=events.append)
    warnings = [e["text"] for e in events if e.get("kind") == "note" and "Warning before the run" in e.get("text", "")]
    assert len(warnings) == 1 and "asks for up to 13.9 on u, but your limit is 10" in warnings[0]
    assert any("asks for up to 13.9" in n for n in result["constraint_notes"])
    # no limit, no extra run, no note
    clean, *_ = ta.run_full_pipeline("", system_spec=_spec(t_end=2.0))
    assert not any("asks for up to" in n for n in clean["constraint_notes"])


# ---------- speed: same numbers, fewer simulations ----------

def test_common_subexpression_elimination_does_not_change_the_laws():
    import sympy as sp
    st, ins, outs, f, _ = _parse_system(["x1", "x2"], CP_EQS, ["u"], ["x1"])
    pts = [([0.3, -0.2], [0.1, 0.2, -0.3]), ([-0.5, 1.1], [0.0, 0.0, 0.0]), ([0.9, 0.4], [0.4, -0.1, 0.2])]
    # backstepping with a bound: compare the cse law against the same law lambdified without cse
    struct = backstepping_design._build_backstepping_structure(st, ins[0], f, None, True)
    plain = sp.lambdify(list(st) + struct["yd"] + [sp.Symbol("kb_barrier")], struct["u_law"], "numpy")
    for x, yd in pts:
        assert struct["u_func"](*x, *yd, 1.6) == pytest.approx(plain(*x, *yd, 1.6), rel=1e-9)
    # sliding mode with a bound: the controller against a hand-free evaluation of the same struct
    smc = smc_design._build_smc_structure(st, ins, outs, f, 2.0, np.array([1.6]))
    for x, yd in pts:
        args = (*x, *yd, 1.6)
        raw = sp.lambdify(list(st) + smc["yd_flat"] + smc["kb_symbols"], sp.Matrix(smc["surface_list"]), "numpy")
        assert np.ravel(smc["surface_func"](*args))[0] == pytest.approx(np.ravel(raw(*args))[0], rel=1e-9)
        raw_d = sp.lambdify(list(st) + smc["yd_flat"] + smc["kb_symbols"], sp.Matrix(smc["drift_list"]), "numpy")
        assert np.ravel(smc["drift_func"](*args))[0] == pytest.approx(np.ravel(raw_d(*args))[0], rel=1e-9)


def test_the_free_run_made_before_the_run_is_reused_for_the_diagnosis(monkeypatch):
    os.environ["LABCD_ADAPTIVE_SHOW_PLOTS"] = "0"
    from backend_core.AgentAdaptive.agents import llm_factory
    from backend_core.AgentAdaptive.agents.agent_io import _SyntheticMessage, _empty_usage
    from backend_core.AgentAdaptive.controller import simulation

    def fake_extraction(spec, on_event=None, clarification_record=None):
        st = _verify_structure(_build_structure_from_spec(spec)[0])
        args = dict(st, method="backstepping", reasoning="t", notes_limitations=None)
        use = {"agent": _empty_usage(), "total": _empty_usage(), "agent_turns": [], "timeline": []}
        return {"messages": [_SyntheticMessage("d")]}, use, {"ok": True, "report": "", "args": args, "why": "t"}

    class Reply:
        content = json.dumps({"headline": "h", "explanation": ["e"], "suggestions": []})
        usage_metadata = {}

    class LLM:
        def invoke(self, msgs):
            return Reply()

    calls = []
    real = simulation.simulate
    monkeypatch.setattr(simulation, "simulate", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    monkeypatch.setattr(ta, "run_extraction", fake_extraction)
    monkeypatch.setattr(ta, "write_abstract", lambda *a, **k: ("abs", ta._empty_usage()))
    monkeypatch.setattr(llm_factory, "build_llm", lambda role, **k: LLM())
    spec = _spec(x0=(0.2, 0.0), ref="0.5*sin(5*t)", limits=(-10.0, 10.0), t_end=10.0)
    result, *_ = ta.run_full_pipeline("", system_spec=spec)
    assert len(calls) == 2                                    # one free run, one real run (was three)
    free = result["diagnosis"]["evidence"]["input_limits"]["without_limits"]
    assert free["passes"] is True and free["peak_by_input"]["u"] == pytest.approx(13.9, abs=0.2)
    assert result["diagnosis"]["evidence"]["failure_mode"]["mode"] == "limit_too_small"


# ---------- speed: nothing is computed that nobody reads ----------

def test_tuning_rounds_do_not_build_the_displayed_law_but_the_final_run_does(monkeypatch):
    from backend_core.AgentAdaptive.controller import runs
    calls = []
    real = runs._format_u_with_reference_smc
    monkeypatch.setattr(runs, "_format_u_with_reference_smc", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    spec = _spec(t_end=1.0)
    tuning, _ = _quiet_run_for(_args("smc", spec), for_tuning=True)
    assert calls == [] and tuning["control_law"] == ""
    final, _ = _quiet_run_for(_args("smc", spec), for_tuning=False)
    assert len(calls) == 1 and "u" in final["control_law"] and "$$" in final["control_law"]


def _quiet_run_for(args, for_tuning):
    with contextlib.redirect_stdout(io.StringIO()):
        return ta._rerun_design_directly(args, for_tuning=for_tuning)


def test_uncertainty_functions_are_built_once_per_cache():
    from backend_core.AgentAdaptive.controller import runs
    cache = {}
    args = _args("smc", system_spec.normalize_defaults({"status": "complete", "system_name": "U", "dynamics": {
        "states": ["x1", "x2"], "inputs": ["u"], "outputs": ["x1"], "state_equations": ["x2", "u"],
        "x0": [0, 0], "references": [{"output": "x1", "expr": "0"}], "solver_step": 0.001, "sim_time": 0.5,
        "uncertainty": [{"state": "x2", "expr": "0.1*sin(x1)"}]}}), has_delta=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ta._rerun_design_directly(args, for_tuning=True, structure_cache=cache)
        keys = [k for k in cache if k[0] == "uncertainty"]
        first = cache[keys[0]]
        ta._rerun_design_directly(args, for_tuning=True, structure_cache=cache)
    assert len(keys) == 1 and cache[keys[0]] is first and callable(first[0])


def test_the_argument_count_of_a_function_is_read_once():
    from backend_core.AgentAdaptive.controller import estimators
    estimators._arg_count_cached.cache_clear()
    f = lambda x, u: np.zeros(2)  # noqa: E731
    for _ in range(50):
        estimators.eval_uncertainty(f, np.zeros(2), np.zeros(1), 0.0, 2)
    info = estimators._arg_count_cached.cache_info()
    assert info.misses == 1 and info.hits == 49


def test_a_diverging_run_stops_early_and_still_fails_finite():
    from backend_core.AgentAdaptive.controller import simulation
    from backend_core.AgentAdaptive.tools import scoring
    st, ins, outs, f, _ = _parse_system(["x1", "x2"], ["x2", "x1**3 + u"], ["u"], ["x1"])
    system = {"states": st, "inputs": ins, "outputs": outs, "f": f}
    steps_calls = []
    controller = lambda x, ref: (steps_calls.append(1), [0.0])[1]  # noqa: E731  (open loop: finite-time blow-up)
    t, y, ref, u, x, _ = simulation.simulate(system, controller, [2], [{"expr": "0"}], [1.0, 1.0], dt=0.001, t_end=5.0)
    assert len(t) == len(y) == len(u) == len(x) == 5000                  # same shape as a full run
    assert len(steps_calls) < 2000                                        # but it stopped computing early
    assert np.isnan(y[-1]).all() and np.isfinite(y[0]).all() and t[-1] == pytest.approx(4.999, abs=1e-6)
    assert t[1] - t[0] == pytest.approx(0.001) and np.all(np.diff(t) > 0)
    verdict = scoring.compute_success_verdict(t, y, ref, u, x, 0.001)
    assert verdict["success"] is False and verdict["checks"]["finite"] is False


# ---------- the simplify button ----------

@pytest.mark.parametrize("method", ["smc", "backstepping"])
def test_the_law_is_simplified_only_on_request(method, monkeypatch):
    import sympy as sp
    from backend_core.AgentAdaptive.controller import runs
    calls = []
    real = sp.trigsimp
    monkeypatch.setattr(sp, "trigsimp", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    spec = _spec(ref="0.15*sin(1.2*t)", t_end=1.0)
    args = _args(method, spec)
    with contextlib.redirect_stdout(io.StringIO()):
        comps, _ = ta._rerun_design_directly(args, for_tuning=False)
    assert calls == [] and "$$" in comps["control_law"]                  # the run itself never simplifies
    job = dict({k: args[k] for k in ("states", "dynamics", "inputs", "outputs", "refs")}, method=method,
               **({"surface_lambda": 2, "K": 1.5, "Lam": 5, "phi_layer": 0.05} if method == "smc"
                  else {"c_gains": None}))
    text = runs.simplified_law_report(job)
    assert len(calls) >= 1 and "$$" in text and "THIS system's actual reference" in text


def test_the_pipeline_hands_over_the_job_unless_there_is_a_bound(monkeypatch):
    os.environ["LABCD_ADAPTIVE_SHOW_PLOTS"] = "0"
    from backend_core.AgentAdaptive.agents import llm_factory
    from backend_core.AgentAdaptive.agents.agent_io import _SyntheticMessage, _empty_usage

    def fake_extraction(spec, on_event=None, clarification_record=None):
        st = _verify_structure(_build_structure_from_spec(spec)[0])
        args = dict(st, method="smc", reasoning="t", notes_limitations=None)
        use = {"agent": _empty_usage(), "total": _empty_usage(), "agent_turns": [], "timeline": []}
        return {"messages": [_SyntheticMessage("d")]}, use, {"ok": True, "report": "", "args": args, "why": "t"}

    monkeypatch.setattr(ta, "run_extraction", fake_extraction)
    monkeypatch.setattr(ta, "write_abstract", lambda *a, **k: ("abs", ta._empty_usage()))
    plain, *_ = ta.run_full_pipeline("", system_spec=_spec(t_end=1.0))
    job = plain["law_job"]
    assert job["method"] == "smc" and job["surface_lambda"] == 2 and set(job) >= {"states", "dynamics", "refs", "K"}
    from backend_core.AgentAdaptive.controller import runs
    assert "$$" in runs.simplified_law_report(job)
    bounded, *_ = ta.run_full_pipeline("", system_spec=_spec(bounds=(-1.0, 1.0), t_end=1.0))
    assert bounded["law_job"] is None


# ---------- found by the real-model pass: a run that diverged under a limit ----------

def test_a_diverged_run_still_reports_how_much_the_limit_clipped():
    t = np.linspace(0, 1, 100)
    y = np.zeros((100, 1)); y[50:] = np.nan
    u_cmd = np.full((100, 1), 100.0); u_cmd[50:] = np.nan
    from backend_core.AgentAdaptive.controller import simulation
    limits = simulation.limit_arrays([[-20.0, 20.0]], 1)
    m = scoring.compute_simulation_metrics(t, y, y, u_cmd, y, None, 0.01, u_cmd=u_cmd, limits=limits)
    assert m["success"] is False and m["numerically_healthy"] is False
    sat = m["input_saturation"]
    assert sat["any_input_fraction"] == pytest.approx(0.5) and sat["peak_command"] == [100.0]
    assert sat["first_saturated_at_s"] == pytest.approx(0.0)


def test_the_limit_card_says_when_the_need_is_far_above_the_limit():
    ev = {"input_limits": {"limits": {"u": [None, 20.0]},
                           "saturation": {"any_input_fraction": 0.4, "first_saturated_at_s": 0.1},
                           "without_limits": {"passes": True, "tracking_pct": 99.0, "failed_check": None,
                                              "peak_by_input": {"u": 3479.0}}}}
    ev["failure_mode"] = fm.classify(ev)
    assert ev["failure_mode"]["mode"] == "limit_too_small"
    card = fm.limit_card(ev)
    assert "far above your limit" in card["detail"]
    ev["input_limits"]["without_limits"]["peak_by_input"]["u"] = 30.0
    assert "far above" not in fm.limit_card(ev)["detail"]


def test_the_warning_before_the_run_keeps_the_bound_so_it_matches_the_diagnosis(monkeypatch):
    # found in the live page: the warning said 6.07 (no bound) while the diagnosis said 18 (with the bound)
    os.environ["LABCD_ADAPTIVE_SHOW_PLOTS"] = "0"
    from backend_core.AgentAdaptive.agents import llm_factory
    from backend_core.AgentAdaptive.agents.agent_io import _SyntheticMessage, _empty_usage
    from backend_core.AgentAdaptive.controller import simulation

    def fake_extraction(spec, on_event=None, clarification_record=None):
        st = _verify_structure(_build_structure_from_spec(spec)[0])
        args = dict(st, method="backstepping", reasoning="t", notes_limitations=None)
        use = {"agent": _empty_usage(), "total": _empty_usage(), "agent_turns": [], "timeline": []}
        return {"messages": [_SyntheticMessage("d")]}, use, {"ok": True, "report": "", "args": args, "why": "t"}

    class Reply:
        content = json.dumps({"headline": "h", "explanation": ["e"], "suggestions": []})
        usage_metadata = {}

    class LLM:
        def invoke(self, msgs):
            return Reply()

    calls = []
    real = simulation.simulate
    monkeypatch.setattr(simulation, "simulate", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    monkeypatch.setattr(ta, "run_extraction", fake_extraction)
    monkeypatch.setattr(ta, "write_abstract", lambda *a, **k: ("abs", ta._empty_usage()))
    monkeypatch.setattr(llm_factory, "build_llm", lambda role, **k: LLM())
    events = []
    spec = _spec(x0=(0.0, 0.0), ref="0.24*sin(5*t)", bounds=(-0.3, 0.3), limits=(-4.0, 4.0), t_end=10.0)
    result, *_ = ta.run_full_pipeline("", system_spec=spec, on_event=events.append)
    note = [e["text"] for e in events if "Warning before the run" in e.get("text", "")][0]
    peak_in_note = float(note.split("up to ")[1].split(" on u")[0])
    peak_in_diag = result["diagnosis"]["evidence"]["input_limits"]["without_limits"]["peak_by_input"]["u"]
    assert peak_in_note == pytest.approx(peak_in_diag, rel=1e-3) and peak_in_note > 10.0   # the bound raises the need
    assert len(calls) == 2
