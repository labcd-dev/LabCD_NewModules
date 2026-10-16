"""
Run with: pytest backend_core/AgentMPC/tests/test_actuation_analysis.py -v

Catching a target the plant cannot physically be held at.

This is the failure that looks like bad tuning and is not. If a user asks an
under-actuated plant to hold every state at a chosen value, nothing errors:
the controller converges on the closest thing it can reach, and the leftover
steady-state error is reported by every metric as though the weights were
wrong. The tuning loop then spends its whole budget chasing an error that no
choice of Q, R, Np or Nc can remove.

Counting inputs against states is the obvious test and the wrong one -- a
double integrator has two states and one input and holds both at a setpoint
perfectly well. What decides it is how many state values can be commanded
INDEPENDENTLY at steady state: at equilibrium f(x*, u*) = 0, so A x* must lie
in the range of B, and the reachable equilibria form the subspace
{x : N A x = 0} with the rows of N spanning the left null space of B.
"""

import numpy as np
import pytest

from backend_core.AgentMPC.agents.dynamics_validator import (
    analyze_actuation, detect_derivative_pairs,
)
from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig


class _DoubleIntegrator(BaseDynamics):
    """Two states, one input -- under-actuated by the naive count, yet it holds
    a position setpoint perfectly well. The distinction this module exists to
    make."""

    def __init__(self, target=None):
        super().__init__(SystemConfig(
            n_states=2, n_inputs=1,
            state_names=["position", "velocity"], input_names=["force"],
            default_initial_state=np.array([0.5, 0.0]),
            default_target=np.asarray(target if target is not None else [0.0, 0.0], dtype=float),
            input_bounds=(np.array([-10.0]), np.array([10.0])),
            params={"mass": 1.0},
        ))

    def dynamics(self, x, u):
        return np.array([x[1], float(np.asarray(u).reshape(-1)[0]) / self.params["mass"]])


class _TwoIndependentMasses(BaseDynamics):
    """Two states, two inputs, one actuator each -- genuinely fully actuated."""

    def __init__(self, target=None):
        super().__init__(SystemConfig(
            n_states=2, n_inputs=2,
            state_names=["a", "b"], input_names=["ua", "ub"],
            default_initial_state=np.array([0.0, 0.0]),
            default_target=np.asarray(target if target is not None else [0.0, 0.0], dtype=float),
            params={},
        ))

    def dynamics(self, x, u):
        u = np.asarray(u).reshape(-1)
        return np.array([-x[0] + u[0], -x[1] + u[1]])


class _RedundantActuators(BaseDynamics):
    """One state, two inputs that do the same thing -- over-actuated."""

    def __init__(self):
        super().__init__(SystemConfig(
            n_states=1, n_inputs=2,
            state_names=["level"], input_names=["pump_a", "pump_b"],
            default_initial_state=np.array([0.0]),
            default_target=np.array([0.0]),
            params={},
        ))

    def dynamics(self, x, u):
        u = np.asarray(u).reshape(-1)
        return np.array([-x[0] + u[0] + u[1]])


# --------------------------------------------------------- classification --
def test_a_double_integrator_is_under_actuated_but_controllable():
    result = analyze_actuation(_DoubleIntegrator())

    assert result["classification"] == "under-actuated"
    assert result["fully_controllable"] is True, \
        "under-actuated is not the same as uncontrollable"
    assert result["independently_holdable"] == 1


def test_independent_actuators_count_as_fully_actuated():
    result = analyze_actuation(_TwoIndependentMasses())

    assert result["classification"] == "fully actuated"
    assert result["independently_holdable"] == 2


def test_duplicate_actuators_are_reported_as_redundant():
    result = analyze_actuation(_RedundantActuators())

    assert "over-actuated" in result["classification"]
    assert result["n_inputs"] == 2
    assert result["rank_B"] == 1, "the two pumps are not independent"


# ------------------------------------------------------ target feasibility --
def test_a_reachable_target_is_accepted():
    plant = _DoubleIntegrator(target=[1.0, 0.0])

    result = analyze_actuation(plant, target=plant.config.default_target)

    assert result["target_feasible"] is True
    assert result["unreachable_states"] == []


def test_a_nonzero_velocity_target_is_rejected():
    """Holding a velocity at a nonzero constant is not a steady state: if the
    velocity holds, the position never stops moving."""
    plant = _DoubleIntegrator(target=[0.0, 0.5])

    result = analyze_actuation(plant, target=plant.config.default_target)

    assert result["target_feasible"] is False
    assert any(s["state"] == "position" for s in result["unreachable_states"])


def test_a_fully_actuated_plant_accepts_a_target_on_every_state():
    plant = _TwoIndependentMasses(target=[0.3, -0.7])

    result = analyze_actuation(plant, target=plant.config.default_target)

    assert result["target_feasible"] is True
    assert result["independently_holdable"] == 2


def test_the_residual_names_the_states_that_cannot_be_held():
    plant = _DoubleIntegrator(target=[0.0, 2.0])

    result = analyze_actuation(plant, target=plant.config.default_target)

    offenders = {s["state"]: s["residual"] for s in result["unreachable_states"]}
    assert "position" in offenders
    assert offenders["position"] == pytest.approx(2.0, rel=1e-6), \
        "the position drifts at exactly the velocity it was told to hold"


# ------------------------------------------------------- derivative pairs --
def test_a_derivative_pair_violation_is_reported_by_name():
    plant = _DoubleIntegrator(target=[1.0, 0.4])
    plant.config.derivative_pairs = detect_derivative_pairs(plant)

    result = analyze_actuation(plant, target=plant.config.default_target)

    assert result["derivative_pair_violations"], "velocity is d(position)/dt and was given 0.4"
    violation = result["derivative_pair_violations"][0]
    assert violation["position"] == "position"
    assert violation["velocity"] == "velocity"
    assert violation["requested"] == pytest.approx(0.4)


def test_a_zero_velocity_target_is_not_a_violation():
    plant = _DoubleIntegrator(target=[1.0, 0.0])
    plant.config.derivative_pairs = detect_derivative_pairs(plant)

    result = analyze_actuation(plant, target=plant.config.default_target)

    assert result["derivative_pair_violations"] == []


# -------------------------------------------------------------- plumbing --
def test_the_analysis_runs_without_a_target():
    """The classification is a property of the plant and must not require a
    setpoint to be available yet."""
    result = analyze_actuation(_DoubleIntegrator())

    assert result["target_feasible"] is None
    assert result["classification"]


def test_the_result_is_json_friendly():
    """It travels into an LLM prompt, a report and potentially an API
    response, so no numpy scalars may survive."""
    import json

    plant = _DoubleIntegrator(target=[0.0, 0.5])
    result = analyze_actuation(plant, target=plant.config.default_target)

    json.dumps(result)   # raises TypeError on a numpy scalar


def test_an_unreachable_target_becomes_a_diagnostics_finding():
    from backend_core.AgentMPC.agents.diagnostics_agent import scan_for_issues

    plant = _DoubleIntegrator(target=[0.0, 0.5])
    actuation = analyze_actuation(plant, target=plant.config.default_target)
    rows = [{"iteration": 1, "ok": True, "unstable": False, "mse": 0.4,
             "solver_diagnostics": {"solved": 10, "solved_inaccurate": 0,
                                     "max_iter_reached": 0, "other": 0}}]

    findings = scan_for_issues([], rows, {}, actuation=actuation)

    assert "target_not_reachable" in findings
    assert findings["target_not_reachable"]["count"] >= 1


def test_a_reachable_target_produces_no_finding():
    from backend_core.AgentMPC.agents.diagnostics_agent import scan_for_issues

    plant = _DoubleIntegrator(target=[1.0, 0.0])
    actuation = analyze_actuation(plant, target=plant.config.default_target)
    rows = [{"iteration": 1, "ok": True, "unstable": False, "mse": 0.01,
             "solver_diagnostics": {"solved": 10, "solved_inaccurate": 0,
                                     "max_iter_reached": 0, "other": 0}}]

    findings = scan_for_issues([], rows, {}, actuation=actuation)

    assert "target_not_reachable" not in findings
