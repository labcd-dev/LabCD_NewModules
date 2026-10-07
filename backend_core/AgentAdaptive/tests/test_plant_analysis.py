"""
Run from the repo root with:
    python -m pytest backend_core/AgentAdaptive/tests/test_plant_analysis.py -v
"""

import json
import time

import pytest
import sympy as sp

from backend_core.AgentAdaptive.controller.plant_analysis import (
    MAX_RELATIVE_DEGREE, RelativeDegreeError, analyze_plant, analyze_spec, lie_chain,
)
from backend_core.AgentAdaptive.controller.smc_design import _build_smc_structure
from backend_core.AgentAdaptive.controller.structure_build import _parse_system


def dyn(states, equations, inputs, outputs, x0=None):
    return {"states": states, "state_equations": equations, "inputs": inputs,
            "outputs": outputs, "x0": x0 if x0 is not None else [0.0] * len(states)}


# demo plants from data/system.json

CART_POLE = dyn(
    ["x1", "x2"],
    ["x2",
     "(-(u + 0.06*x2**2*sin(x1))*cos(x1) + 11.2815*sin(x1))/(0.06*sin(x1)**2 + 0.4)"],
    ["u"], ["x1"], [0.2, 0.0])

MAGLEV = dyn(["x1", "x2", "x3"], ["x2", "-20*x3", "-50*x3 + 50*u"],
             ["u"], ["x1"], [0.002, 0.0, 0.0])

REACTION_WHEEL = dyn(["x1", "x2", "x3"], ["x2", "-0.0016*x3", "-20*x3 + 10*u"],
                     ["u"], ["x1"], [0.05, 0.0, 0.0])

DOUBLE_PENDULUM = dyn(
    ["x1", "x2", "x3", "x4"],
    ["x2",
     "(-0.25*u1 + 0.0625*x4**2*sin(x1 - x3) + 0.25*(u2 + 0.25*x2**2*sin(x1 - x3) - 4.905*sin(x3))*cos(x1 - x3) + 2.4525*sin(x1))/(0.0625*cos(x1 - x3)**2 - 0.125)",
     "x4",
     "(-0.5*u2 - 0.125*x2**2*sin(x1 - x3) - 0.25*(-u1 + 0.25*x4**2*sin(x1 - x3) + 9.81*sin(x1))*cos(x1 - x3) + 2.4525*sin(x3))/(0.0625*cos(x1 - x3)**2 - 0.125)"],
    ["u1", "u2"], ["x1", "x3"], [0.3, 0.0, -0.2, 0.0])

# one failure mode per plant

THREE_IN_TWO_OUT = dyn(
    ["x1", "x2", "x3", "x4"],
    ["x3 + u1", "x4 + u2 + u3", "-x3 + u1", "-x4 + u3"],
    ["u1", "u2", "u3"], ["x1", "x2"])

ONE_IN_TWO_OUT = dyn(["x1", "x2"], ["u", "-x2 + u"], ["u"], ["x1", "x2"])

# v_in is frozen at 0, so d never shows up in any derivative of v_out
UNCONTROLLABLE = dyn(["v_in", "v_out", "i_L"],
                     ["0", "(v_in - v_out)/2", "(v_in*d - v_out)/0.5"],
                     ["d"], ["v_out"], [1.0, 0.0, 0.0])

SINGULAR_D = dyn(["x1", "x2"], ["u1 + u2", "u1 + u2"], ["u1", "u2"], ["x1", "x2"])

NON_AFFINE = dyn(["x1", "x2"], ["x2", "sin(u)"], ["u"], ["x1"])

UNSTABLE_ZERO = dyn(["x1", "x2"], ["u", "x1 + x2"], ["u"], ["x1"])
STABLE_ZERO = dyn(["x1", "x2"], ["u", "x1 - x2"], ["u"], ["x1"])

# x1 -> x3 -> x2 is a chain, just listed in the wrong order
SCRAMBLED_CHAIN = dyn(["x1", "x2", "x3"], ["x3", "u", "x2"], ["u"], ["x1"])

# not a chain, and no reordering fixes it: x1 depends on both x2 and x3
NON_CHAIN = dyn(["x1", "x2", "x3"], ["x2 + x3", "-x2", "u"], ["u"], ["x1"])

# position is driven by v, heading by om; om reaches (x, y) only through v, so D is singular
UNICYCLE = dyn(["x", "y", "th"], ["v*cos(th)", "v*sin(th)", "om"], ["v", "om"], ["x", "y"], [0.0, 0.0, 0.3])

# no input reaches any state, so no choice of output helps
UNFIXABLE = dyn(["x1", "x2"], ["0", "-x2"], ["u"], ["x1"])

# D = x2: singular only at x2 = 0, which is away from x0 in one case and at x0 in the other
PARTIAL_SINGULAR = dyn(["x1", "x2"], ["x2*u", "-x2"], ["u"], ["x1"], [0.0, 1.0])
SINGULAR_AT_X0 = dyn(["x1", "x2"], ["x2*u", "-x2"], ["u"], ["x1"], [0.0, 0.0])

EARLY_INPUT = dyn(["x1", "x2"], ["x2 + u", "u"], ["u"], ["x1"])


@pytest.mark.parametrize("plant", [CART_POLE, MAGLEV, REACTION_WHEEL])
def test_siso_chain_demos_fit_and_default_to_backstepping(plant):
    r = analyze_plant(plant)
    assert r["status"] == "fits"
    assert r["candidates"] == ["smc", "backstepping"]
    # SISO chain -> backstepping, like the designer's tie-break
    assert r["designer_default"] == "backstepping"
    assert r["chain"]["given_order_ok"] is True


def test_double_pendulum_fits_smc_only():
    r = analyze_plant(DOUBLE_PENDULUM)
    assert r["status"] == "fits"
    assert r["candidates"] == ["smc"]
    assert r["designer_default"] == "smc"
    assert r["relative_degrees"] == [2, 2]
    assert r["hidden_dimension"] == 0
    assert r["decoupling"]["min_sigma"] > 0
    assert r["decoupling"]["singular_points"] == 0
    assert any("one input and one output" in x for x in r["methods"]["backstepping"]["reasons"])


def test_inverted_pendulum_artifact_with_parameters_is_unchanged():
    # a parameter called "I" must stay a plain symbol, not sympy's imaginary unit
    spec = {"dynamics": {
        "states": ["theta", "omega"], "inputs": ["tau"], "outputs": ["theta"],
        "state_equations": ["omega", "(tau - m * g * l * sin(theta)) / I"],
        "parameters": {"g": 9.81, "m": 0.5, "l": 1.0, "I": 0.16666666666666666},
        "x0": [0.1, 0.0]}}
    r = analyze_spec(spec)
    assert r["status"] == "fits"
    assert r["designer_default"] == "backstepping"


def test_three_inputs_two_outputs_is_rejected_with_numbers_and_hint():
    r = analyze_plant(THREE_IN_TWO_OUT)
    assert r["status"] == "does_not_fit"
    assert r["candidates"] == []
    assert "3 input(s) and 2 output(s)" in r["methods"]["smc"]["reasons"][0]
    assert "3 input(s) and 2 output(s)" in r["summary"]
    assert [h["id"] for h in r["conversion_hints"]] == ["T1"]
    assert r["decoupling"]["min_rank"] == 2
    # hidden states are only defined for square plants
    assert r["hidden_dimension"] is None
    assert not [f for f in r["findings"] if f["check"] == "C5"]


def test_fewer_inputs_than_outputs_hints_output_selection():
    r = analyze_plant(ONE_IN_TWO_OUT)
    assert r["status"] == "does_not_fit"
    assert "T2" in [h["id"] for h in r["conversion_hints"]]


def test_uncontrollable_plant_fails_fast_with_named_output():
    start = time.time()
    r = analyze_plant(UNCONTROLLABLE)
    assert time.time() - start < 10
    assert r["status"] == "does_not_fit"
    assert r["relative_degrees"] == [None]
    reason = r["methods"]["smc"]["reasons"][0]
    assert "v_out" in reason and str(MAX_RELATIVE_DEGREE) in reason
    assert "T3" in [h["id"] for h in r["conversion_hints"]]


def test_smc_builder_raises_instead_of_hanging_on_uncontrollable_plant():
    states, inputs, outputs, f, _ = _parse_system(
        UNCONTROLLABLE["states"], UNCONTROLLABLE["state_equations"],
        UNCONTROLLABLE["inputs"], UNCONTROLLABLE["outputs"])
    start = time.time()
    with pytest.raises(RelativeDegreeError, match="v_out"):
        _build_smc_structure(states, inputs, outputs, f, 2)
    assert time.time() - start < 10


def test_relative_degree_error_is_a_value_error():
    # the build stage's except clauses already catch ValueError
    assert issubclass(RelativeDegreeError, ValueError)


def test_singular_decoupling_matrix_is_rejected():
    r = analyze_plant(SINGULAR_D)
    assert r["status"] == "does_not_fit"
    assert r["methods"]["smc"]["viable"] is False
    assert "singular" in r["methods"]["smc"]["reasons"][0]
    assert "T4" in [h["id"] for h in r["conversion_hints"]]
    assert r["decoupling"]["empirical"] is True


def test_non_affine_input_is_rejected():
    r = analyze_plant(NON_AFFINE)
    assert r["status"] == "does_not_fit"
    assert r["affine"] is False
    assert "not linear in the input" in r["methods"]["smc"]["reasons"][0]


def test_unstable_zero_is_a_warning_not_a_rejection():
    r = analyze_plant(UNSTABLE_ZERO)
    assert r["status"] == "fits"           # SMC still applies, the risk is just flagged
    assert r["hidden_dimension"] == 1
    warnings = [x for x in r["findings"] if x["check"] == "C6"]
    assert warnings and "right half-plane" in warnings[0]["message"]
    assert "np.float64" not in warnings[0]["message"]
    assert "T3" in [h["id"] for h in r["conversion_hints"]]


def test_stable_zero_gives_no_c6_warning():
    r = analyze_plant(STABLE_ZERO)
    assert r["status"] == "fits"
    assert [x for x in r["findings"] if x["check"] == "C6"] == []
    zeros = r["internal_dynamics"]["points"][0]["zeros"]
    assert any(abs(re + 1.0) < 1e-6 for re, _ in zeros)


def test_hidden_states_are_reported_with_numbers():
    r = analyze_plant(NON_CHAIN)
    assert r["relative_degrees"] == [2]
    assert r["sum_relative_degrees"] == 2
    assert r["hidden_dimension"] == 1
    c5 = [x for x in r["findings"] if x["check"] == "C5"]
    assert c5 and "1 state(s) are hidden" in c5[0]["message"]


def test_scrambled_chain_is_detected_with_relabel_order():
    r = analyze_plant(SCRAMBLED_CHAIN)
    assert r["chain"]["given_order_ok"] is False
    assert r["chain"]["relabel_order"] == ["x1", "x3", "x2"]
    assert r["methods"]["backstepping"]["viable"] is False
    assert "T0" in [h["id"] for h in r["conversion_hints"]]
    assert r["candidates"] == ["smc"]       # SMC does not care about the state order


def test_non_chain_has_no_relabel_and_names_the_violation():
    r = analyze_plant(NON_CHAIN)
    assert r["chain"]["relabel_order"] is None
    assert any("more than one step ahead" in v for v in r["chain"]["violations"])


def test_input_before_last_equation_is_a_violation():
    r = analyze_plant(EARLY_INPUT)
    assert any("only in the last equation" in v for v in r["chain"]["violations"])
    assert r["methods"]["backstepping"]["viable"] is False


def test_undefined_symbol_cannot_be_analyzed():
    r = analyze_plant(dyn(["x1"], ["k*x1 + u"], ["u"], ["x1"]))
    assert r["status"] == "cannot_analyze"
    assert "undefined symbol" in r["summary"] and "k" in r["summary"]


def test_output_must_be_a_state():
    r = analyze_plant(dyn(["x1"], ["u"], ["u"], ["y"]))
    assert r["status"] == "cannot_analyze"
    assert "not one of the states" in r["summary"]


def test_equation_count_mismatch():
    r = analyze_plant(dyn(["x1", "x2"], ["u"], ["u"], ["x1"]))
    assert r["status"] == "cannot_analyze"


@pytest.mark.parametrize("plant", [CART_POLE, DOUBLE_PENDULUM, THREE_IN_TWO_OUT,
                                   UNCONTROLLABLE, UNSTABLE_ZERO, SCRAMBLED_CHAIN])
def test_report_is_json_serializable_with_a_status(plant):
    r = analyze_plant(plant)
    assert r["status"] in ("fits", "does_not_fit", "cannot_analyze")
    json.dumps(r)


def test_assumptions_and_empirical_labels_are_present():
    r = analyze_plant(CART_POLE)
    text = " ".join(r["assumptions"])
    assert "empirical" in text and "full state" in text


def test_analysis_emits_one_visible_note():
    events = []
    analyze_plant(THREE_IN_TWO_OUT, on_event=events.append)
    assert len(events) == 1
    assert events[0]["kind"] == "note" and "does not fit" in events[0]["text"]


def test_large_expressions_do_not_hang_the_zero_check():
    from backend_core.AgentAdaptive.controller.plant_analysis import _is_zero
    x = sp.Symbol("x")
    big_nonzero = sum(sp.sin(k * x) * sp.cos(k * x) for k in range(1, 120))
    big_zero = sum(sp.sin(k * x) ** 2 + sp.cos(k * x) ** 2 - 1 for k in range(1, 120))
    start = time.time()
    assert _is_zero(big_nonzero) is False
    assert _is_zero(big_zero) is True
    assert time.time() - start < 10


def test_analyze_alias_matches_the_proposal_name():
    from backend_core.AgentAdaptive.controller.plant_analysis import analyze
    assert analyze is analyze_spec


def _reference_chain(output_expr, states, inputs, f, limit=12):
    # the old uncapped loop, bounded here only so a bug can't hang the test
    zero_inputs = {u: 0 for u in inputs}
    derivatives = [output_expr]
    relative_degree = 0
    for _ in range(limit):
        current = derivatives[-1]
        derivative = 0
        for i in range(len(states)):
            derivative = derivative + sp.diff(current, states[i]) * f[i]
        derivative = sp.expand(derivative)
        relative_degree += 1
        if any(derivative.has(u) for u in inputs):
            return (derivatives, relative_degree,
                    [sp.diff(derivative, u) for u in inputs],
                    derivative.subs(zero_inputs))
        derivatives.append(derivative)
    raise AssertionError("reference loop did not terminate")


@pytest.mark.parametrize("plant", [CART_POLE, MAGLEV, REACTION_WHEEL, DOUBLE_PENDULUM])
def test_lie_chain_matches_the_original_loop(plant):
    states, inputs, outputs, f, _ = _parse_system(
        plant["states"], plant["state_equations"], plant["inputs"], plant["outputs"])
    for out in outputs:
        new = lie_chain(out, states, inputs, f)
        derivatives, rd, row, drift = _reference_chain(out, states, inputs, f)
        assert new["relative_degree"] == rd
        assert new["derivatives"] == derivatives
        assert new["decoupling_row"] == row
        assert new["drift_term"] == drift


def test_smc_builder_still_builds_the_double_pendulum():
    states, inputs, outputs, f, _ = _parse_system(
        DOUBLE_PENDULUM["states"], DOUBLE_PENDULUM["state_equations"],
        DOUBLE_PENDULUM["inputs"], DOUBLE_PENDULUM["outputs"])
    struct = _build_smc_structure(states, inputs, outputs, f, 2)
    assert struct["rel_degrees"] == [2, 2]


def test_isolated_singular_points_are_a_warning_when_x0_is_fine():
    r = analyze_plant(PARTIAL_SINGULAR)
    assert r["status"] == "fits"
    c4 = [f for f in r["findings"] if f["check"] == "C4"]
    assert c4 and c4[0]["severity"] == "warning" and "keep the trajectory away" in c4[0]["message"]
    assert r["decoupling"]["singular_points"] >= 1
    assert "T4" not in [h["id"] for h in r["conversion_hints"]]


def test_singular_at_x0_is_still_blocking():
    r = analyze_plant(SINGULAR_AT_X0)
    assert r["status"] == "does_not_fit"
    assert [f for f in r["findings"] if f["check"] == "C4"][0]["severity"] == "blocking"
    assert "T4" in [h["id"] for h in r["conversion_hints"]]


# reference clearance (C9): a reference that crosses a singular state of the control law

def _clearance(plant, ref, sim_time=10.0, name="x1"):
    from backend_core.AgentAdaptive.controller.plant_analysis import reference_clearance
    spec = {"dynamics": dict(plant, references=[{"output": name, "expr": ref}], sim_time=sim_time)}
    return reference_clearance(spec)


def test_a_reference_through_the_cart_pole_singularity_is_flagged():
    found = _clearance(CART_POLE, "10*sin(5*t)")
    assert len(found) == 1 and found[0]["check"] == "C9"
    lo, hi = found[0]["window"]
    assert lo == pytest.approx(-1.5708, abs=1e-3) and hi == pytest.approx(1.5708, abs=1e-3)
    assert found[0]["reference_span"][1] > 9.9


def test_a_small_cart_pole_reference_is_not_flagged():
    assert _clearance(CART_POLE, "0.15*sin(1.2*t)") == []
    assert _clearance(CART_POLE, "1.2*sin(2*t)") == []


def test_a_reference_just_over_the_edge_is_flagged():
    assert len(_clearance(CART_POLE, "1.7*sin(t)")) == 1


def test_clearance_never_raises_on_odd_input():
    from backend_core.AgentAdaptive.controller.plant_analysis import reference_clearance
    assert reference_clearance({}) == []
    assert _clearance(CART_POLE, "not a formula(") == []
