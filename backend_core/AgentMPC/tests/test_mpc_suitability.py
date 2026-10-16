"""
Run with: pytest backend_core/AgentMPC/tests/test_mpc_suitability.py -v

The Diagnostics assistant's deterministic checks -- the engine behind its
pop-ups. Two questions, answered from the plant itself with no LLM:

  * is this plant suitable for THIS MPC at all?
  * can this plant actually follow the reference the user picked?

Each test builds a plant or a reference where the right answer is not in
doubt, and checks the assistant reaches it. The calibration matters as much as
the detection: a check that raises an error on every sensible setup trains the
user to ignore the assistant, so the "sensible" cases are pinned too.
"""

import json

import numpy as np
import pytest

from backend_core.AgentMPC.agents.dynamics_validator import detect_derivative_pairs
from backend_core.AgentMPC.agents.mpc_suitability import (
    assess_mpc_suitability, assess_reference,
)
from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig


def _codes(findings, severity=None):
    return {f["code"] for f in findings if severity is None or f["severity"] == severity}


class _MassSpring(BaseDynamics):
    """Position/velocity, one force input. Damping and stiffness are
    parameters, so the same class can be well-behaved, undamped, or stiff."""

    def __init__(self, k=1.0, c=0.8, target=(0.0, 0.0), input_bounds=(-10.0, 10.0)):
        super().__init__(SystemConfig(
            n_states=2, n_inputs=1, state_names=["position", "velocity"], input_names=["force"],
            default_initial_state=np.array([0.5, 0.0]),
            default_target=np.asarray(target, dtype=float),
            state_bounds=(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
            input_bounds=(np.array([input_bounds[0]]), np.array([input_bounds[1]])),
            params={"k": k, "c": c},
        ))

    def dynamics(self, x, u):
        force = float(np.asarray(u).reshape(-1)[0])
        return np.array([x[1], force - self.params["k"] * x[0] - self.params["c"] * x[1]])


class _UnreachableUnstable(BaseDynamics):
    """An unstable mode no input touches -- not stabilisable by anything."""

    def __init__(self):
        super().__init__(SystemConfig(
            n_states=2, n_inputs=1, state_names=["driven", "runaway"], input_names=["u"],
            default_initial_state=np.array([0.0, 0.1]), default_target=np.array([0.0, 0.0]),
            params={}))

    def dynamics(self, x, u):
        return np.array([-x[0] + float(np.asarray(u).reshape(-1)[0]), 0.7 * x[1]])


class _NaNAtTarget(BaseDynamics):
    def __init__(self):
        super().__init__(SystemConfig(
            n_states=1, n_inputs=1, state_names=["x"], input_names=["u"],
            default_initial_state=np.array([1.0]), default_target=np.array([0.0]), params={}))

    def dynamics(self, x, u):
        return np.array([1.0 / x[0] + float(np.asarray(u).reshape(-1)[0])])


# ------------------------------------------------------------ suitability --
def test_a_well_behaved_plant_at_a_sensible_dt_is_suitable():
    """Calibration: the ordinary case must not raise anything."""
    result = assess_mpc_suitability(_MassSpring(), dt=0.02)

    assert result["verdict"] == "suitable"
    assert not _codes(result["findings"], "error")
    assert not _codes(result["findings"], "warning")


def test_an_unreachable_unstable_mode_makes_the_plant_unsuitable():
    result = assess_mpc_suitability(_UnreachableUnstable(), dt=0.02)

    assert result["verdict"] == "not_suitable"
    assert "not_stabilisable" in _codes(result["findings"], "error")


def test_dynamics_that_are_not_finite_at_the_target_are_caught_first():
    result = assess_mpc_suitability(_NaNAtTarget(), dt=0.02)

    assert result["verdict"] == "not_suitable"
    assert "dynamics_not_finite" in _codes(result["findings"], "error")


def test_a_dt_that_makes_the_euler_prediction_unstable_is_an_error():
    """A stable mode at rate 50/s: forward Euler goes unstable above
    dt = 2/50 = 0.04 s, so the controller would plan against a system that
    does not exist."""
    stiff = _MassSpring(k=0.0, c=50.0)

    assert "euler_unstable_at_dt" not in _codes(assess_mpc_suitability(stiff, dt=0.02)["findings"])
    assert "euler_unstable_at_dt" in _codes(assess_mpc_suitability(stiff, dt=0.1)["findings"], "error")


def test_an_undamped_oscillator_is_flagged_only_when_dt_inflates_it_materially():
    """Forward Euler inflates an undamped mode at EVERY dt; what matters is
    how much over a horizon. Fine at a small dt, an error at a coarse one."""
    oscillator = _MassSpring(k=20.0, c=0.0)      # w = sqrt(20) ~ 4.5 rad/s

    fine = assess_mpc_suitability(oscillator, dt=0.005)
    coarse = assess_mpc_suitability(oscillator, dt=0.5)

    assert "euler_inflates_oscillation" not in _codes(fine["findings"])
    assert "euler_inflates_oscillation" in _codes(coarse["findings"], "error")
    assert coarse["verdict"] == "not_suitable"


def test_an_equilibrium_input_outside_the_bounds_is_an_error():
    """Holding position 1 against a spring of stiffness 5 needs 5 N; the
    actuator only has 2."""
    plant = _MassSpring(k=5.0, target=(1.0, 0.0), input_bounds=(-2.0, 2.0))
    plant.get_equilibrium_input = lambda: np.array([5.0])

    result = assess_mpc_suitability(plant, dt=0.02)

    assert "equilibrium_input_out_of_bounds" in _codes(result["findings"], "error")


def test_the_suitability_result_is_json_friendly():
    json.dumps(assess_mpc_suitability(_MassSpring(k=20.0, c=0.0), dt=0.5))


# ------------------------------------------------------------- reference --
DT, T = 0.02, 300
TIME = np.arange(T) * DT
ZERO = np.zeros(T)


@pytest.fixture
def plant():
    p = _MassSpring()
    p.config.derivative_pairs = detect_derivative_pairs(p)
    return p


def test_a_reachable_constant_target_raises_nothing(plant):
    assert assess_reference(plant, np.column_stack([ZERO, ZERO]), DT) == []


def test_a_consistent_moving_reference_raises_nothing(plant):
    """Calibration again: position sin(t) with velocity cos(t) is exactly
    followable by a fully controllable mass-spring."""
    ref = np.column_stack([0.3 * np.sin(TIME), 0.3 * np.cos(TIME)])

    assert _codes(assess_reference(plant, ref, DT), "error") == set()
    assert "derivative_pair_inconsistent" not in _codes(assess_reference(plant, ref, DT))


def test_a_velocity_that_is_not_the_derivative_of_its_position_is_flagged(plant):
    """The classic wrong-trajectory-type mistake: a sinusoid on the position
    while the velocity is held at zero."""
    findings = assess_reference(plant, np.column_stack([0.3 * np.sin(TIME), ZERO]), DT)

    assert "derivative_pair_inconsistent" in _codes(findings, "warning")


def test_a_constant_target_the_plant_cannot_rest_at_is_an_error(plant):
    """A nonzero constant velocity is never a resting point."""
    findings = assess_reference(plant, np.column_stack([ZERO, ZERO + 0.5]), DT)

    assert "reference_infeasible" in _codes(findings, "error")


def test_a_moving_reference_that_cannot_be_followed_exactly_is_a_warning_not_an_error():
    """An under-actuated plant asked to move one state while holding another
    still is a trade-off, not a mistake -- flagged, but not as a blocker."""
    from backend_core.AgentMPC.dynamics.loader import DynamicLoader

    cart_pole = DynamicLoader.load_from_path(
        "backend_core/AgentMPC/dynamics/plugins/example_pendulum.py").create_dynamics()
    cart_pole.config.derivative_pairs = detect_derivative_pairs(cart_pole)
    ref = np.column_stack([0.3 * np.sin(TIME), 0.3 * np.cos(TIME), ZERO, ZERO])

    findings = assess_reference(cart_pole, ref, DT)

    assert "reference_infeasible" in _codes(findings, "warning")
    assert "reference_infeasible" not in _codes(findings, "error")


def test_a_non_finite_reference_is_an_error(plant):
    ref = np.column_stack([ZERO, ZERO])
    ref[10, 0] = np.nan

    assert "reference_not_finite" in _codes(assess_reference(plant, ref, DT), "error")


def test_a_reference_needing_more_actuation_than_available_is_flagged():
    weak = _MassSpring(input_bounds=(-0.05, 0.05))
    weak.config.derivative_pairs = detect_derivative_pairs(weak)
    ref = np.column_stack([2.0 * np.sin(3 * TIME), 6.0 * np.cos(3 * TIME)])

    assert "reference_exceeds_actuators" in _codes(assess_reference(weak, ref, DT), "warning")


def test_reference_findings_are_json_friendly(plant):
    json.dumps(assess_reference(plant, np.column_stack([0.3 * np.sin(TIME), ZERO]), DT))
