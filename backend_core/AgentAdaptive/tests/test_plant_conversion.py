"""
Run from the repo root with:
    python -m pytest backend_core/AgentAdaptive/tests/test_plant_conversion.py -v
"""

import copy
import json

import numpy as np
import pytest
import sympy as sp

from test_plant_analysis import (
    CART_POLE, DOUBLE_PENDULUM, MAGLEV, NON_AFFINE, NON_CHAIN, ONE_IN_TWO_OUT, SCRAMBLED_CHAIN, SINGULAR_D,
    THREE_IN_TWO_OUT, UNCONTROLLABLE, UNICYCLE, UNSTABLE_ZERO, dyn,
)
from backend_core.AgentAdaptive.controller.plant_analysis import analyze_spec
from backend_core.AgentAdaptive.controller.plant_conversion import (
    ConversionError, apply_dynamic_extension, apply_input_squaring, apply_lie_coordinates, apply_output_reselection,
    apply_output_selection, apply_relabel, apply_term_shift, propose_extensions, rank_output_reselections,
    rank_output_selections,
)
from backend_core.AgentAdaptive.tools import system_spec

# x1 is driven by u, x2 is a hidden state with an unstable zero seen from x1
UNSTABLE_PICK = dyn(["x1", "x2"], ["u", "x1 + x2"], ["u"], ["x1", "x2"])

# the commands u1 and u2 always act together, so D has rank 1 for 2 outputs
RANK_DEFICIENT = dyn(["x1", "x2"], ["u1 + u2", "u1 + u2"], ["u1", "u2", "u3"], ["x1", "x2"])


def spec_of(plant, references=None, uncertainty=None, parameters=None):
    d = dict(plant)
    d["references"] = references or []
    d["uncertainty"] = uncertainty or []
    d["parameters"] = parameters or {}
    return {"status": "complete", "system_name": "t", "dynamics": d}


def _fn(equations, args):
    syms = {a: sp.Symbol(a, real=True) for a in args}
    exprs = [sp.sympify(e, locals=syms) for e in equations]
    return sp.lambdify([syms[a] for a in args], sp.Matrix(exprs), "numpy")


def test_relabel_reorders_states_equations_and_x0():
    spec = spec_of(dict(SCRAMBLED_CHAIN, x0=[1.0, 2.0, 3.0]))
    before = copy.deepcopy(spec)
    out = apply_relabel(spec)
    d = out["spec"]["dynamics"]
    assert d["states"] == ["x1", "x3", "x2"]
    assert d["state_equations"] == ["x3", "x2", "u"]
    assert d["x0"] == [1.0, 3.0, 2.0]
    assert out["analysis"]["chain"]["given_order_ok"] is True
    assert out["analysis"]["designer_default"] == "backstepping"
    rec = out["record"]
    assert (rec["id"], rec["label"], rec["task_changed"]) == ("T0", "A", False)
    assert rec["original"]["states"] == ["x1", "x2", "x3"]
    assert spec == before


def test_relabel_refuses_when_nothing_to_do():
    with pytest.raises(ConversionError, match="already form a chain"):
        apply_relabel(spec_of(CART_POLE))
    with pytest.raises(ConversionError, match="no ordering"):
        apply_relabel(spec_of(NON_CHAIN))
    with pytest.raises(ConversionError, match="one input and one output"):
        apply_relabel(spec_of(DOUBLE_PENDULUM))


def test_input_squaring_gives_a_square_exact_plant():
    out = apply_input_squaring(spec_of(THREE_IN_TWO_OUT))
    d = out["spec"]["dynamics"]
    assert d["inputs"] == ["v1", "v2"] and d["system_type"] == "MIMO"
    a = out["analysis"]
    assert a["candidates"] == ["smc"] and a["relative_degrees"] == [1, 1]
    assert a["decoupling"]["min_sigma"] == pytest.approx(1.0)
    assert out["record"]["task_changed"] is False
    assert out["record"]["label"] == "C"
    assert "2 hidden state(s)" in out["record"]["label_reasons"][0]

    # independent exactness check: u from the recorded map, then f_old(x,u) == f_new(x,v)
    amap = out["record"]["details"]["input_map"]
    rng = np.random.default_rng(3)
    old = _fn(THREE_IN_TWO_OUT["state_equations"], ["x1", "x2", "x3", "x4", "u1", "u2", "u3"])
    new = _fn(d["state_equations"], ["x1", "x2", "x3", "x4", "v1", "v2"])
    u_of = _fn([amap["u1"], amap["u2"], amap["u3"]], ["v1", "v2"])
    for _ in range(5):
        x, v = rng.uniform(-1, 1, 4), rng.uniform(-1, 1, 2)
        u = np.array(u_of(*v), dtype=float).ravel()
        assert np.allclose(np.array(old(*x, *u), dtype=float), np.array(new(*x, *v), dtype=float))


def test_input_squaring_without_hidden_states_keeps_label_a():
    plant = dyn(["x1", "x2"], ["x2", "u1 + u2"], ["u1", "u2"], ["x1"])
    out = apply_input_squaring(spec_of(plant))
    assert out["record"]["label"] == "A"
    assert out["analysis"]["hidden_dimension"] == 0


def test_weights_move_effort_away_from_expensive_inputs():
    def u3_gain(weights):
        amap = apply_input_squaring(spec_of(THREE_IN_TWO_OUT), weights=weights)["record"]["details"]["input_map"]
        return abs(float(sp.sympify(amap["u3"]).coeff(sp.Symbol("v2"))))
    assert u3_gain([1, 1, 100]) < 0.1 * u3_gain([1, 1, 1])


def test_input_squaring_refusals():
    with pytest.raises(ConversionError, match="more inputs than outputs"):
        apply_input_squaring(spec_of(DOUBLE_PENDULUM))
    with pytest.raises(ConversionError, match="drops to rank"):
        apply_input_squaring(spec_of(RANK_DEFICIENT))
    with pytest.raises(ConversionError, match="weights"):
        apply_input_squaring(spec_of(THREE_IN_TWO_OUT), weights=[1, 1])
    with pytest.raises(ConversionError, match="weights"):
        apply_input_squaring(spec_of(THREE_IN_TWO_OUT), weights=[1, 0, 1])


def test_input_squaring_substitutes_parameters():
    plant = dyn(["x1", "x2", "x3"], ["a*x3 + u1", "u2", "-x3 + u2"], ["u1", "u2"], ["x1"])
    out = apply_input_squaring(spec_of(plant, parameters={"a": 2.0}))
    d = out["spec"]["dynamics"]
    assert d["parameters"] == {} and "a*" not in d["state_equations"][0]


def test_input_dependent_uncertainty_follows_the_new_commands_and_blocks_label_a():
    spec = spec_of(THREE_IN_TWO_OUT, uncertainty=[{"state": "x3", "expr": "0.1*u3"}])
    out = apply_input_squaring(spec)
    expr = out["spec"]["dynamics"]["uncertainty"][0]["expr"]
    assert "u3" not in expr and "v2" in expr
    assert out["record"]["label"] == "C"


def test_virtual_names_avoid_clashes_and_can_be_chosen():
    plant = dyn(["v1", "x2"], ["x2 + u1", "u2"], ["u1", "u2"], ["v1"])
    assert apply_input_squaring(spec_of(plant))["spec"]["dynamics"]["inputs"] == ["v2"]
    custom = apply_input_squaring(spec_of(THREE_IN_TWO_OUT), virtual_names=["a", "b"])
    assert custom["spec"]["dynamics"]["inputs"] == ["a", "b"]
    with pytest.raises(ConversionError, match="distinct"):
        apply_input_squaring(spec_of(THREE_IN_TWO_OUT), virtual_names=["a", "a"])


def test_converted_spec_is_normalized_and_record_is_json():
    out = apply_input_squaring(spec_of(THREE_IN_TWO_OUT))
    assert system_spec.normalize_spec(out["spec"]) == out["spec"]
    json.dumps(out["record"])
    json.dumps(out["analysis"])
    assert analyze_spec(out["spec"])["status"] == "fits"


def test_conversion_announces_itself():
    events = []
    apply_input_squaring(spec_of(THREE_IN_TWO_OUT), on_event=events.append)
    assert len(events) == 1 and "T1" in events[0]["text"] and "label C" in events[0]["text"]


def test_ranking_prefers_outputs_without_unstable_hidden_dynamics():
    ranking = rank_output_selections(spec_of(UNSTABLE_PICK))
    assert [r["outputs"] for r in ranking] == [["x2"], ["x1"]]
    assert ranking[0]["hidden_dimension"] == 0
    assert ranking[1]["unstable_hidden_dynamics_hint"] is True
    json.dumps(ranking)


def test_output_selection_changes_the_task_and_says_so():
    refs = [{"output": "x1", "expr": "sin(t)"}, {"output": "x2", "expr": "cos(t)"}]
    out = apply_output_selection(spec_of(UNSTABLE_PICK, references=refs), ["x2"])
    d = out["spec"]["dynamics"]
    assert d["outputs"] == ["x2"]
    assert d["references"] == [{"output": "x2", "expr": "cos(t)"}]
    rec = out["record"]
    assert rec["task_changed"] is True and rec["untracked_outputs"] == ["x1"]
    assert "x1 is no longer controlled" in rec["changes"][0]
    assert rec["label"] == "A" and out["analysis"]["status"] == "fits"


def test_selection_with_unstable_hidden_dynamics_gets_label_c():
    out = apply_output_selection(spec_of(UNSTABLE_PICK), ["x1"])
    assert out["record"]["label"] == "C"
    assert out["record"]["details"]["missing_references"] == ["x1"]


def test_output_selection_refusals():
    with pytest.raises(ConversionError, match="exactly 1"):
        apply_output_selection(spec_of(UNSTABLE_PICK), ["x1", "x2"])
    with pytest.raises(ConversionError, match="distinct members"):
        apply_output_selection(spec_of(UNSTABLE_PICK), ["y"])
    with pytest.raises(ConversionError, match="fewer inputs than outputs"):
        apply_output_selection(spec_of(DOUBLE_PENDULUM), ["x1", "x3"])
    with pytest.raises(ConversionError, match="fewer inputs than outputs"):
        rank_output_selections(spec_of(THREE_IN_TWO_OUT))


def test_one_input_two_outputs_plant_can_be_converted():
    ranking = rank_output_selections(spec_of(ONE_IN_TWO_OUT))
    assert all(r["fits"] for r in ranking)


def test_decoupling_at_shows_which_input_reaches_which_output():
    from backend_core.AgentAdaptive.controller.plant_conversion import decoupling_at
    d = decoupling_at(spec_of(THREE_IN_TWO_OUT))
    assert d["outputs"] == ["x1", "x2"] and d["inputs"] == ["u1", "u2", "u3"]
    assert d["matrix"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]
    json.dumps(d)
    assert decoupling_at(spec_of(dyn(["x1"], ["x1/0"], ["u"], ["x1"]))) is None


def test_extension_search_finds_the_input_that_unlocks_the_matrix():
    # the search stops at the first number of integrators that works, so nothing with two is tried
    found = {tuple(e["inputs"]): e["fits"] for e in propose_extensions(spec_of(UNICYCLE))}
    assert found == {("v",): True, ("om",): False}


def test_dynamic_extension_turns_the_input_into_a_state():
    out = apply_dynamic_extension(spec_of(UNICYCLE))
    d = out["spec"]["dynamics"]
    assert d["states"] == ["x", "y", "th", "v"] and d["inputs"] == ["v_dot", "om"]
    assert d["state_equations"] == ["v*cos(th)", "v*sin(th)", "om", "v_dot"]
    assert d["x0"] == [0.0, 0.0, 0.3, 1.0] and d["outputs"] == ["x", "y"]
    assert out["analysis"]["status"] == "fits" and out["analysis"]["relative_degrees"] == [2, 2]
    rec = out["record"]
    assert rec["id"] == "T4" and rec["task_changed"] is False
    assert rec["details"]["extended_inputs"] == ["v"] and rec["details"]["rate_inputs"] == ["v_dot"]
    assert rec["details"]["assumed_initial_inputs"] == ["v"] and "assumed" in rec["changes"][0]
    assert rec["label"] == "C" and "singular at some sampled points" in rec["label_reasons"][0]
    json.dumps(rec)


def test_dynamic_extension_uses_the_starting_value_it_is_given():
    out = apply_dynamic_extension(spec_of(UNICYCLE), initial_inputs={"v": 2.0})
    assert out["spec"]["dynamics"]["x0"][-1] == 2.0
    assert out["record"]["details"]["assumed_initial_inputs"] == []


def test_dynamic_extension_refusals():
    with pytest.raises(ConversionError, match="already invertible"):
        apply_dynamic_extension(spec_of(CART_POLE))
    with pytest.raises(ConversionError, match="as many inputs as outputs"):
        apply_dynamic_extension(spec_of(THREE_IN_TWO_OUT))
    with pytest.raises(ConversionError, match="not linear in the input"):
        apply_dynamic_extension(spec_of(NON_AFFINE))
    with pytest.raises(ConversionError, match="do not give an invertible"):
        apply_dynamic_extension(spec_of(SINGULAR_D))
    with pytest.raises(ConversionError, match="distinct inputs"):
        apply_dynamic_extension(spec_of(UNICYCLE), inputs=["nope"])
    with pytest.raises(ConversionError, match="not inputs"):
        apply_dynamic_extension(spec_of(UNICYCLE), initial_inputs={"zzz": 1})
    with pytest.raises(ConversionError, match="still rejects|does not fit"):
        apply_dynamic_extension(spec_of(UNICYCLE), inputs=["om"])


def test_reselection_finds_the_controllable_output():
    ranking = rank_output_reselections(spec_of(UNCONTROLLABLE))
    assert [r["outputs"] for r in ranking] == [["i_L"], ["v_in"]]
    assert ranking[0]["fits"] and not ranking[1]["fits"]
    json.dumps(ranking)


def test_reselection_changes_the_task_and_asks_for_the_missing_reference():
    out = apply_output_reselection(spec_of(UNCONTROLLABLE), ["i_L"])
    rec = out["record"]
    assert rec["id"] == "T3" and rec["task_changed"] is True
    assert rec["untracked_outputs"] == ["v_out"] and rec["details"]["missing_references"] == ["i_L"]
    assert "reference is still needed for i_L" in rec["changes"][0]
    assert out["spec"]["dynamics"]["outputs"] == ["i_L"] and out["spec"]["dynamics"]["references"] == []
    assert rec["label"] == "C"

    given = apply_output_reselection(spec_of(UNCONTROLLABLE), ["i_L"], references={"i_L": "0.5"})
    assert given["record"]["details"]["missing_references"] == []
    assert given["spec"]["dynamics"]["references"] == [{"output": "i_L", "expr": "0.5"}]


def test_reselection_away_from_unstable_hidden_dynamics_can_earn_label_a():
    ranking = rank_output_reselections(spec_of(UNSTABLE_ZERO))
    assert ranking[0]["outputs"] == ["x2"] and ranking[0]["hidden_dimension"] == 0
    out = apply_output_reselection(spec_of(UNSTABLE_ZERO), ["x2"], references={"x2": "sin(t)"})
    assert out["record"]["label"] == "A" and out["record"]["task_changed"] is True


def test_reselection_refusals():
    with pytest.raises(ConversionError, match="as many inputs as outputs"):
        apply_output_reselection(spec_of(THREE_IN_TWO_OUT), ["x1", "x2"])
    with pytest.raises(ConversionError, match="already the plant's outputs"):
        apply_output_reselection(spec_of(UNCONTROLLABLE), ["v_out"])
    with pytest.raises(ConversionError, match="distinct states"):
        apply_output_reselection(spec_of(UNCONTROLLABLE), ["nope"])
    with pytest.raises(ConversionError, match="exactly 1"):
        apply_output_reselection(spec_of(UNCONTROLLABLE), ["i_L", "v_in"])
    with pytest.raises(ConversionError, match="does not fit"):
        apply_output_reselection(spec_of(UNCONTROLLABLE), ["v_in"])


# the input reaches x1 only through exp(x2): relative degree 2 = n, but not a chain
EXP2 = dyn(["x1", "x2"], ["-x1 + exp(x2)", "u"], ["u"], ["x1"], [0.0, 0.0])
EXP3 = dyn(["x1", "x2", "x3"], ["x2", "exp(x3)", "u"], ["u"], ["x1"], [0.0, 0.0, 0.0])


def test_lie_coordinates_turn_a_non_chain_into_a_chain():
    assert analyze_spec(spec_of(EXP2))["conversion_hints"][0]["id"] == "T5"
    out = apply_lie_coordinates(spec_of(EXP2))
    d, rec = out["spec"]["dynamics"], out["record"]
    assert d["states"] == ["x1", "z2"] and d["outputs"] == ["x1"] and d["x0"] == [0.0, 1.0]
    assert rec["details"]["transformation"]["z2"] == "-x1 + exp(x2)"
    assert rec["details"]["inverse"]["x2"] == "log(x1 + z2)"
    assert rec["task_changed"] is False and out["analysis"]["methods"]["backstepping"]["viable"]


def test_lie_coordinates_give_the_same_trajectory_as_the_original():
    from scipy.integrate import solve_ivp
    out = apply_lie_coordinates(spec_of(EXP2))
    u = lambda t: 0.3 * np.sin(t)
    old = solve_ivp(lambda t, x: [-x[0] + np.exp(x[1]), u(t)], (0, 1), [0.0, 0.0], rtol=1e-10, atol=1e-12).y[:, -1]
    f = _fn(out["spec"]["dynamics"]["state_equations"], ["x1", "z2", "u"])
    new = solve_ivp(lambda t, z: f(z[0], z[1], u(t)).ravel(), (0, 1), [0.0, 1.0], rtol=1e-10, atol=1e-12).y[:, -1]
    assert new[0] == pytest.approx(old[0], abs=1e-6)
    assert new[1] == pytest.approx(-old[0] + np.exp(old[1]), abs=1e-6)


def test_lie_coordinates_refuse_plants_that_do_not_need_them():
    with pytest.raises(ConversionError, match="already a strict-feedback chain"):
        apply_lie_coordinates(spec_of(MAGLEV))
    with pytest.raises(ConversionError, match="T0"):
        apply_lie_coordinates(spec_of(SCRAMBLED_CHAIN))
    with pytest.raises(ConversionError, match="hidden"):
        apply_lie_coordinates(spec_of(NON_CHAIN))
    with pytest.raises(ConversionError, match="one input and one output"):
        apply_lie_coordinates(spec_of(UNICYCLE))


def test_matched_uncertainty_stays_in_the_last_row_and_unmatched_is_flagged():
    matched = apply_lie_coordinates(spec_of(EXP3, uncertainty=[{"state": "x3", "expr": "0.2*x1"}]))
    assert matched["spec"]["dynamics"]["uncertainty"] == [{"state": "z3", "expr": "0.2*x1*z3"}]
    assert not any("depends on" in r for r in matched["record"]["label_reasons"])

    unmatched = apply_lie_coordinates(spec_of(EXP3, uncertainty=[{"state": "x1", "expr": "0.1*sin(x3)"}]))
    assert unmatched["record"]["label"] == "C"
    assert any("uncertainty in the equation of x1 depends on z3" in r for r in unmatched["record"]["label_reasons"])


def test_disturbance_must_stay_a_function_of_time():
    plain = spec_of(EXP3)
    plain["dynamics"]["disturbance"] = [{"state": "x1", "expr": "sin(t)"}]
    out = apply_lie_coordinates(plain)
    assert out["spec"]["dynamics"]["disturbance"] == [{"state": "x1", "expr": "sin(t)"}]
    plain["dynamics"]["disturbance"] = [{"state": "x3", "expr": "sin(t)"}]
    with pytest.raises(ConversionError, match="function of time"):
        apply_lie_coordinates(plain)


# x3 is two steps ahead of x1, so the plant is not a chain; the term is small.
# x1 starts at 1 so the decoupling gain 0.1*x1 is not zero (SMC would reject it otherwise)
SMALL_JUMP = dyn(["x1", "x2", "x3"], ["x2 + 0.1*x1*x3", "x3", "u"], ["u"], ["x1"], [1.0, 0.0, 0.0])


def test_term_shift_makes_the_design_model_a_chain_and_keeps_the_plant():
    spec = spec_of(SMALL_JUMP, uncertainty=[{"state": "x2", "expr": "0.05*x1"}])
    assert any(h["id"] == "T6" for h in analyze_spec(spec)["conversion_hints"])
    out = apply_term_shift(spec)
    d, rec = out["spec"]["dynamics"], out["record"]
    assert d["state_equations"] == ["x2", "x3", "u"]
    assert {e["state"]: e["expr"] for e in d["uncertainty"]} == {"x1": "0.1*x1*x3", "x2": "0.05*x1"}
    assert out["analysis"]["methods"]["backstepping"]["viable"]
    assert rec["label"] == "C" and rec["experimental"] and rec["exact_on_nominal"] is False
    assert any("x3" in r and "does not see" in r for r in rec["label_reasons"])
    assert rec["details"]["removed_size_percent"]["x1"] < 50


def test_term_shift_refuses_when_nothing_is_left_or_the_term_is_big():
    with pytest.raises(ConversionError, match="no chain is left"):
        apply_term_shift(spec_of(NON_CHAIN))
    with pytest.raises(ConversionError, match="no chain is left"):
        apply_term_shift(spec_of(dyn(["x1", "x2"], ["-x1 + exp(x2)", "u"], ["u"], ["x1"], [0.0, 0.0])))
    big = dyn(["x1", "x2", "x3"], ["x2 + 5*x1*x3", "x3", "u"], ["u"], ["x1"], [0.0, 0.0, 0.0])
    with pytest.raises(ConversionError, match="too large"):
        apply_term_shift(spec_of(big))
    with pytest.raises(ConversionError, match="already a strict-feedback chain"):
        apply_term_shift(spec_of(MAGLEV))


# planar quadrotor (PVTOL), outputs x and z: the torque only shows up after the thrust is
# integrated twice, so one integrator is not enough
PVTOL = dyn(["x", "vx", "z", "vz", "th", "w"],
            ["vx", "-T*sin(th)", "vz", "T*cos(th) - 9.81", "w", "tau"], ["T", "tau"], ["x", "z"],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.0])


def test_two_integrators_on_the_thrust_are_found_when_one_is_not_enough():
    entries = propose_extensions(spec_of(PVTOL), {"T": 9.81})
    assert not any(e["fits"] for e in entries if sum(e["orders"].values()) == 1)
    fitting = [e for e in entries if e["fits"]]
    assert fitting[0]["orders"] == {"T": 2} and fitting[0]["relative_degrees"] == [4, 4]
    assert max(sum(e["orders"].values()) for e in entries) == 2


def test_extension_with_two_integrators_has_the_right_states_and_equations():
    out = apply_dynamic_extension(spec_of(PVTOL), initial_inputs={"T": 9.81})
    d, rec = out["spec"]["dynamics"], out["record"]
    assert d["states"] == ["x", "vx", "z", "vz", "th", "w", "T", "T_dot"] and d["inputs"] == ["T_ddot", "tau"]
    assert d["state_equations"][-2:] == ["T_dot", "T_ddot"] and d["x0"][-2:] == [9.81, 0.0]
    assert rec["details"]["orders"] == {"T": 2} and rec["details"]["derivative_states"] == {"T": ["T_dot"]}
    assert "second derivative" in rec["changes"][0] and "start at 0" in rec["changes"][0]
    assert out["analysis"]["relative_degrees"] == [4, 4] and out["analysis"]["hidden_dimension"] == 0
    assert rec["task_changed"] is False
    json.dumps(rec)


def test_extension_orders_are_validated():
    with pytest.raises(ConversionError, match="1 to 3"):
        apply_dynamic_extension(spec_of(UNICYCLE), inputs=["v"], orders={"v": 4})
    with pytest.raises(ConversionError, match="in total"):
        apply_dynamic_extension(spec_of(UNICYCLE), inputs=["v", "om"], orders={"v": 2, "om": 2})


def test_two_integrators_reproduce_the_original_plant():
    # fed T = z1, the extended plant moves exactly like the original one driven by that thrust
    from scipy.integrate import solve_ivp
    out = apply_dynamic_extension(spec_of(PVTOL), initial_inputs={"T": 9.81})
    d = out["spec"]["dynamics"]
    f_new = _fn(d["state_equations"], d["states"] + d["inputs"])
    f_old = _fn(PVTOL["state_equations"], PVTOL["states"] + PVTOL["inputs"])
    top = lambda t: 0.5 * np.cos(t)
    tau = lambda t: 0.2
    new = solve_ivp(lambda t, z: f_new(*z, top(t), tau(t)).ravel(), (0, 1), d["x0"], rtol=1e-10, atol=1e-12).y[:, -1]
    thrust = lambda t: 9.81 + 0.5 * (1 - np.cos(t))  # integral twice of 0.5*cos(t), derivative starts at 0
    old = solve_ivp(lambda t, z: f_old(*z, thrust(t), tau(t)).ravel(), (0, 1), PVTOL["x0"], rtol=1e-10, atol=1e-12).y[:, -1]
    assert new[:6] == pytest.approx(old, abs=1e-6)
    assert new[6] == pytest.approx(thrust(1.0), abs=1e-6)
