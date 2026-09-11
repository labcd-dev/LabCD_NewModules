"""
Run with: pytest backend_core/AgentMPC/tests/test_nonfinite_guards.py -v

Regression guard for "the evaluator returns no simulation at all, every
iteration fails with ``QP solve failed: maximum iterations reached``".

That message was a misattribution, and an expensive one: it points squarely
at the tuning, so the obvious response is to go fiddle with Q, R, Np and Nc
-- none of which can possibly help. The real trigger is a NaN/Inf reaching
the QP. Which message OSQP produces depends only on *where* the
contamination lands:

  * a NaN in the cost's linear term ``q`` alone (from the reference
    trajectory, the initial state, or the affine offset) leaves ``P``
    perfectly factorizable, so OSQP runs all the way to its iteration cap and
    reports "maximum iterations reached" while handing back a NaN iterate;
  * a NaN in ``P`` (from the weights or the linearized A/B) kills the LDL
    factorization instead, and surfaces as a bare ``OSQPException: 4``.

Neither names the actual culprit. Worse, the first one is indistinguishable
from a genuine hard-to-converge problem, and because the reference and the
initial state do not depend on Q/R/Np/Nc, *every* Actor proposal fails
identically at step 0 -- a whole tuning run, and all of its LLM calls, spent
chasing a phantom.

These tests pin the diagnosis to the source. Each one asserts both that the
right thing is named and that the misleading solver message is gone.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from backend_core.AgentMPC.agents.evaluator import run_closed_loop
from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig
from backend_core.AgentMPC.mpc.config import Config
from backend_core.AgentMPC.mpc.solver import QPSolver

MISLEADING = "maximum iterations reached"


class _LinearPlant(BaseDynamics):
    """Minimal 2-state (position, velocity) double integrator."""

    def __init__(self):
        super().__init__(SystemConfig(
            n_states=2,
            n_inputs=1,
            state_names=["position", "velocity"],
            input_names=["force"],
            default_initial_state=np.array([0.5, 0.0]),
            default_target=np.array([0.0, 0.0]),
            state_bounds=(np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
            input_bounds=(np.array([-5.0]), np.array([5.0])),
            params={"mass": 1.0},
        ))

    def dynamics(self, x, u):
        force = float(np.asarray(u).reshape(-1)[0])
        return np.array([x[1], force / self.params["mass"]])


def _cfg():
    cfg = Config()
    cfg.mpc.prediction_horizon = 8
    cfg.mpc.control_horizon = 4
    cfg.data.dt_mpc = 0.05
    cfg.data.simulation_time = 0.6
    cfg.data.trajectory_mode = "reg"
    return cfg


def _params(**over):
    p = {"Np": 8, "Nc": 4, "Q": [10.0, 1.0], "R": [0.1], "P": [10.0, 1.0]}
    p.update(over)
    return p


def test_a_clean_run_still_simulates():
    """The guards must not reject a perfectly ordinary run."""
    result = run_closed_loop(_LinearPlant(), _cfg(), _params())
    assert "error" not in result, result.get("error")
    assert result["states"].shape[0] > 1


def test_nan_in_the_reference_names_the_reference():
    """The exact failure the user hit: a NaN setpoint reported as a solver
    convergence problem."""
    plant = _LinearPlant()
    original = plant.config.desired_trajectory

    def poisoned(*args, **kwargs):
        ref = np.asarray(original(*args, **kwargs), dtype=float).copy()
        ref[3, 0] = np.nan
        return ref

    plant.config.desired_trajectory = poisoned
    error = run_closed_loop(plant, _cfg(), _params())["error"]

    assert "Reference trajectory contains NaN/Inf" in error
    assert "position" in error          # names the offending state
    assert "3" in error                 # names the offending sample
    assert MISLEADING not in error


def test_nan_in_the_initial_state_names_the_initial_state():
    plant = _LinearPlant()
    plant.config.default_initial_state = np.array([0.5, np.nan])
    error = run_closed_loop(plant, _cfg(), _params())["error"]

    assert "Initial state contains NaN/Inf" in error
    assert "velocity" in error
    assert MISLEADING not in error


def test_nan_from_the_plant_names_the_dynamics():
    """A division by zero (or singular inverse) inside a user-written plugin
    poisons the Jacobian, and used to surface as a bare OSQPException."""
    plant = _LinearPlant()
    plant.dynamics = lambda x, u: np.array([x[1], np.nan])
    error = run_closed_loop(plant, _cfg(), _params())["error"]

    assert "linearized model is not finite" in error
    assert "dynamics function" in error
    assert MISLEADING not in error


def test_nan_weight_names_the_weight_matrix():
    """A malformed Actor proposal is rejected as a weight problem, not blamed
    on the solver."""
    error = run_closed_loop(_LinearPlant(), _cfg(), _params(Q=[10.0, np.nan]))["error"]

    assert "state weights Q matrix contains NaN/Inf" in error
    assert MISLEADING not in error


def test_solver_rejects_a_contaminated_q_by_name():
    """The last line of defence, checked directly: a NaN in q must raise a
    data error rather than being laundered into a convergence complaint."""
    n = 3
    identity = sp.csc_matrix(np.eye(n))
    with pytest.raises(ValueError) as excinfo:
        QPSolver().solve(identity, np.full(n, np.nan), identity,
                         np.full(n, -1.0), np.full(n, 1.0))

    message = str(excinfo.value)
    assert "QP data is not finite" in message
    assert "q (cost linear term" in message
    assert MISLEADING not in message


def test_infinite_bounds_are_still_accepted():
    """+/-inf in l/u means "unbounded" and is how every plugin without
    declared state bounds arrives -- the guard must not reject it."""
    n = 3
    identity = sp.csc_matrix(np.eye(n))
    solution = QPSolver().solve(identity, np.ones(n), identity,
                                np.full(n, -np.inf), np.full(n, np.inf))

    assert np.all(np.isfinite(solution))
