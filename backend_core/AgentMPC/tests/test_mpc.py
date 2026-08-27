"""
Run with: pytest backend_core/AgentMPC/tests/test_mpc.py -v

These tests specifically regression-guard the bugs found in the original
notebook's MPC controller:
  * terminal weight P having zero effect on the control action,
  * state bounds never being enforced,
  * the solver silently producing NaN/garbage on infeasible or malformed
    problems.
"""

import numpy as np
import pytest

from backend_core.AgentMPC.dynamics.loader import DynamicLoader
from backend_core.AgentMPC.mpc.config import Config
from backend_core.AgentMPC.mpc.controller import GenericMPC

PLUGIN_PATH = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "dynamics" / "plugins" / "example_pendulum.py")


@pytest.fixture
def dynamics():
    plugin = DynamicLoader.load_from_path(PLUGIN_PATH)
    return plugin.create_dynamics()


@pytest.fixture
def base_config():
    cfg = Config()
    cfg.mpc.prediction_horizon = 10
    cfg.mpc.control_horizon = 4
    cfg.data.dt_mpc = 0.02
    return cfg


def test_terminal_weight_changes_control_action(dynamics, base_config):
    """Regression test for bug #1: P must actually affect the QP solution."""
    x0 = dynamics.config.default_initial_state.copy()
    u0 = np.zeros(dynamics.n_inputs)
    ref = np.tile(dynamics.config.default_target, (base_config.mpc.prediction_horizon, 1))

    base_config.mpc.terminal_weights = np.diag([1.0] * 4)
    mpc_low = GenericMPC(dynamics, base_config)
    u_low = mpc_low.control(x0, u0, ref)

    base_config.mpc.terminal_weights = np.diag([500.0] * 4)
    mpc_high = GenericMPC(dynamics, base_config)
    u_high = mpc_high.control(x0, u0, ref)

    assert not np.allclose(u_low, u_high), "terminal weight P has no measurable effect (bug not fixed)"


def test_state_bounds_are_enforced_over_a_closed_loop_run(dynamics, base_config):
    """Regression test for bug #2: predicted/realized states must respect
    config.x_bounds (here: the plugin's own state_bounds)."""
    x = dynamics.config.default_initial_state.copy()
    u = np.zeros(dynamics.n_inputs)
    mpc = GenericMPC(dynamics, base_config)
    ref = np.tile(dynamics.config.default_target, (base_config.mpc.prediction_horizon, 1))

    for _ in range(80):
        u = mpc.control(x, u, ref)
        x = dynamics.rk4_step(x, u, base_config.data.dt_mpc)
        assert np.all(x >= mpc.xmin - 1e-6) and np.all(x <= mpc.xmax + 1e-6), (
            f"state bound violated: x={x}, bounds=({mpc.xmin}, {mpc.xmax})"
        )


def test_update_parameters_reflected_in_next_control_call(dynamics, base_config):
    mpc = GenericMPC(dynamics, base_config)
    assert mpc.Np == 10
    mpc.update_parameters({"Np": 6, "Nc": 3, "Q": [1, 1, 1, 1], "R": [1.0]})
    assert mpc.Np == 6 and mpc.Nc == 3

    x0 = dynamics.config.default_initial_state.copy()
    u0 = np.zeros(dynamics.n_inputs)
    ref = np.tile(dynamics.config.default_target, (mpc.Np, 1))
    u = mpc.control(x0, u0, ref)
    assert u.shape == (dynamics.n_inputs,)
    assert np.all(np.isfinite(u))


def test_control_output_is_finite_from_a_disturbed_state(dynamics, base_config):
    mpc = GenericMPC(dynamics, base_config)
    x = np.array([0.3, -0.4, 0.5, 0.2])  # away from equilibrium
    u_prev = np.zeros(dynamics.n_inputs)
    ref = np.tile(dynamics.config.default_target, (base_config.mpc.prediction_horizon, 1))
    u = mpc.control(x, u_prev, ref)
    assert np.all(np.isfinite(u))


def test_solver_reuses_pattern_key_across_calls_with_same_horizon(dynamics, base_config):
    """Not a numerical-result test (that depends on OSQP being installed) --
    just checks the warm-start bookkeeping doesn't rebuild the solver
    unnecessarily when Np/Nc are unchanged between calls."""
    mpc = GenericMPC(dynamics, base_config)
    x = dynamics.config.default_initial_state.copy()
    u = np.zeros(dynamics.n_inputs)
    ref = np.tile(dynamics.config.default_target, (base_config.mpc.prediction_horizon, 1))

    mpc.control(x, u, ref)
    key_after_first = mpc.solver._pattern_key
    mpc.control(x, u, ref)
    key_after_second = mpc.solver._pattern_key

    assert key_after_first == key_after_second, "sparsity pattern key should be stable when Np/Nc don't change"
