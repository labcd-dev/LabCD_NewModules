"""Bounds typed into the UI's tables must not poison the QP.

st.data_editor shows +/-inf as an empty cell and returns a cell the user
clears as NaN. Before the fix those NaNs went straight into the QP's l/u and
every run died on its first step with "l (lower bounds) has 32 NaN of 32
entries" -- on an 8-input plant with Nc = 4.
"""
import numpy as np
import pytest

from backend_core.AgentMPC.mpc.config import MPCConfig


def test_blank_cells_mean_no_limit():
    cfg = MPCConfig()
    cfg.u_bounds = (np.array([-1.0, np.nan, np.nan]), np.array([np.nan, 2.0, np.nan]))
    lo, hi = cfg.get_u_bounds(3)
    assert np.array_equal(lo, [-1.0, -np.inf, -np.inf])
    assert np.array_equal(hi, [np.inf, 2.0, np.inf])


def test_inverted_bounds_are_named():
    cfg = MPCConfig()
    cfg.x_bounds = (np.array([0.0, 5.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="#1: min 5 > max 2"):
        cfg.get_x_bounds(2)


def test_run_survives_a_fully_cleared_bounds_table():
    """End to end: a fully cleared input-bounds table still yields a run."""
    from backend_core.AgentMPC.agents.evaluator import run_closed_loop
    from backend_core.AgentMPC.dynamics.loader import DynamicLoader
    from backend_core.AgentMPC.mpc.config import Config
    from pathlib import Path

    plugin = Path(__file__).resolve().parents[1] / "dynamics" / "plugins" / "example_quadrotor_vertical.py"
    dyn = DynamicLoader.load_from_path(str(plugin)).create_dynamics()
    cfg = Config()
    cfg.mpc.prediction_horizon, cfg.mpc.control_horizon = 10, 2
    cfg.data.dt_mpc, cfg.data.simulation_time = 0.05, 1.5
    cfg.mpc.u_bounds = (np.full(dyn.n_inputs, np.nan), np.full(dyn.n_inputs, np.nan))
    res = run_closed_loop(dyn, cfg, {"Np": 10, "Nc": 2, "Q": [1.0] * dyn.n_states,
                                     "R": [0.1] * dyn.n_inputs})
    assert "error" not in res, res.get("error")
    assert not res.get("unstable"), res.get("unstable_reason")
