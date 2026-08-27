"""
Run with: pytest backend_core/AgentMPC/tests/test_dynamics.py -v
"""

import numpy as np
import pytest

from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig, SystemSimulator
from backend_core.AgentMPC.dynamics.loader import DynamicLoader, DynamicsPluginError

PLUGIN_PATH = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "dynamics" / "plugins" / "example_pendulum.py")


def test_load_valid_plugin():
    plugin = DynamicLoader.load_from_path(PLUGIN_PATH)
    assert plugin.config.n_states == 4
    assert plugin.config.n_inputs == 1
    dyn = plugin.create_dynamics()
    assert isinstance(dyn, BaseDynamics)


def test_dynamics_output_shape_and_finiteness():
    plugin = DynamicLoader.load_from_path(PLUGIN_PATH)
    dyn = plugin.create_dynamics()
    x0 = plugin.config.default_initial_state
    u0 = np.zeros(plugin.config.n_inputs)
    dx = dyn.dynamics(x0, u0)
    assert dx.shape == (4,)
    assert np.all(np.isfinite(dx))


def test_reject_plugin_without_dynamics_class(tmp_path):
    bad_plugin = tmp_path / "bad.py"
    bad_plugin.write_text(
        "def create_config():\n"
        "    return SystemConfig(n_states=1, n_inputs=1, params={}, "
        "state_names=['x'], input_names=['u'], "
        "default_initial_state=np.zeros(1), default_target=np.zeros(1))\n"
    )
    with pytest.raises(DynamicsPluginError):
        DynamicLoader.load_from_path(str(bad_plugin))


def test_reject_plugin_with_wrong_output_shape(tmp_path):
    bad_plugin = tmp_path / "bad_shape.py"
    bad_plugin.write_text(
        "def create_config():\n"
        "    return SystemConfig(n_states=2, n_inputs=1, params={}, "
        "state_names=['a','b'], input_names=['u'], "
        "default_initial_state=np.zeros(2), default_target=np.zeros(2))\n"
        "\n"
        "class Bad(BaseDynamics):\n"
        "    def dynamics(self, x, u):\n"
        "        return np.zeros(3)\n"  # wrong shape on purpose
    )
    with pytest.raises(DynamicsPluginError):
        DynamicLoader.load_from_path(str(bad_plugin))


def test_simulate_open_loop_runs_and_respects_state_bounds_termination():
    plugin = DynamicLoader.load_from_path(PLUGIN_PATH)
    dyn = plugin.create_dynamics()
    sim = SystemSimulator(dyn, dt=0.01)

    # Large constant force should eventually push the pole outside its
    # configured bounds -> simulation should terminate early rather than
    # silently continuing with an out-of-bounds state.
    U = np.full((2000, 1), 50.0)
    X, dX, U_used = sim.simulate(plugin.config.default_initial_state, U)
    assert len(X) < 2000, "simulator should stop early once state bounds are violated"


def test_check_termination_false_when_no_bounds_set():
    cfg = SystemConfig(
        n_states=1, n_inputs=1, params={}, state_names=["x"], input_names=["u"],
        default_initial_state=np.zeros(1), default_target=np.zeros(1),
    )

    class Trivial(BaseDynamics):
        def dynamics(self, x, u):
            return u.copy()

    dyn = Trivial(cfg)
    assert dyn.check_termination(np.array([1e9])) is False
