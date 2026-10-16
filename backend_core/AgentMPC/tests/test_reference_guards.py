"""The three guards around what a run may be asked to do: the reference gate
for under-actuated plants, the backstepping-like feasible reference, and the
per-plant disturbance limit."""

import numpy as np
import pytest

from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig


def _cart_pole(force_bound=20.0):
    """Linear cart with an inverted pendulum: x, v, theta, omega; one force."""

    class CartPole(BaseDynamics):
        def dynamics(self, x, u):
            return np.array([x[1], u[0] - 0.5 * x[2], x[3], 15.0 * x[2] - u[0]])

    return CartPole(SystemConfig(
        n_states=4, n_inputs=1, params={}, state_names=["x", "v", "theta", "omega"], input_names=["F"],
        default_initial_state=np.zeros(4), default_target=np.zeros(4),
        state_bounds=(np.array([-2.0, -3.0, -0.6, -5.0]), np.array([2.0, 3.0, 0.6, 5.0])),
        input_bounds=(np.array([-force_bound]), np.array([force_bound]))))


def _fully_actuated():
    """Two independent double integrators, one force each."""

    class Two(BaseDynamics):
        def dynamics(self, x, u):
            return np.array([x[1], u[0], x[3], u[1]])

    return Two(SystemConfig(
        n_states=4, n_inputs=2, params={}, state_names=["a", "a_dot", "b", "b_dot"], input_names=["Fa", "Fb"],
        default_initial_state=np.zeros(4), default_target=np.zeros(4),
        state_bounds=(np.full(4, -2.0), np.full(4, 2.0)),
        input_bounds=(np.full(2, -5.0), np.full(2, 5.0))))


def _hover(min_thrust=0.0, max_thrust=20.0, w_bound=2.0, start=(0.0, 0.0), z_room=2.0):
    """Vertical hover against gravity: z, w; thrust must be ~9.81 to hold."""

    class Hover(BaseDynamics):
        def dynamics(self, x, u):
            return np.array([x[1], u[0] - 9.81 - 0.2 * x[1]])

    return Hover(SystemConfig(
        n_states=2, n_inputs=1, params={}, state_names=["z", "w"], input_names=["T"],
        default_initial_state=np.array(start, dtype=float), default_target=np.array([1.0, 0.0]),
        state_bounds=(np.array([1.0 - z_room, -w_bound]), np.array([1.0 + z_room, w_bound])),
        input_bounds=(np.array([min_thrust]), np.array([max_thrust]))))


# ---- 1. the gate --------------------------------------------------------------
def test_an_underactuated_plant_may_not_move_two_positions_at_once():
    from backend_core.AgentMPC.agents.reference_authority import reference_authority

    auth = reference_authority(_cart_pole(), moving=[0, 1, 2, 3])
    assert not auth["allowed"]
    assert auth["requested"] == ["x", "theta"]          # each velocity folded into its position
    assert auth["rank"] == 1 and len(auth["keep_options"]) == 1
    assert "G(s)" in auth["message"]


def test_one_position_with_its_own_velocity_is_allowed():
    from backend_core.AgentMPC.agents.reference_authority import reference_authority

    assert reference_authority(_cart_pole(), moving=[0, 1])["allowed"]


def test_a_fully_actuated_plant_may_move_every_position():
    from backend_core.AgentMPC.agents.reference_authority import reference_authority

    assert reference_authority(_fully_actuated(), moving=[0, 1, 2, 3])["allowed"]


def test_the_gate_reads_the_moving_states_off_the_reference_itself():
    from backend_core.AgentMPC.agents.reference_authority import reference_authority

    dyn = _cart_pole()
    dyn.config.derivative_pairs = [(0, 1), (2, 3)]
    t = np.arange(400) * 0.02
    ref = np.zeros((len(t), 4))
    ref[:, 0] = 0.3 * np.sin(t)
    ref[:, 2] = 0.1 * np.sin(2 * t)
    assert not reference_authority(dyn, reference=ref)["allowed"]
    ref[:, 2] = 0.0
    assert reference_authority(dyn, reference=ref)["allowed"]


# ---- 2. the feasible reference -----------------------------------------------
def test_feasible_reference_brings_the_primary_state_to_its_goal_and_the_rest_to_rest():
    from backend_core.AgentMPC.agents.reference_authority import FeasibleReference

    dyn = _cart_pole()
    fr = FeasibleReference(dyn, primary=0, goal=1.0, approach_time=2.0)
    res = fr.generate(0.01, 8.0)
    assert res["ok"] and res["settled"], res["message"]
    ref = res["reference"]
    assert ref[-1, 0] == pytest.approx(1.0, abs=0.02)
    assert np.all(np.abs(ref[-1, 1:]) < 0.02)
    # the pendulum has to lean to move the cart: its reference is not zero
    assert np.max(np.abs(ref[:, 2])) > 1e-3


def test_feasible_reference_is_a_trajectory_of_the_model_itself():
    from backend_core.AgentMPC.agents.mpc_suitability import assess_reference
    from backend_core.AgentMPC.agents.reference_authority import FeasibleReference

    dyn = _cart_pole()
    dyn.config.derivative_pairs = [(0, 1), (2, 3)]
    ref = FeasibleReference(dyn, primary=0, shape="sin", amplitude=0.3, frequency=0.2)(0.01, 10.0)
    findings = {f["code"]: f for f in assess_reference(dyn, ref, 0.01)}
    assert "derivative_pair_inconsistent" not in findings
    # at most the very first sample, where the tracker's input steps on
    if "reference_infeasible" in findings:
        assert findings["reference_infeasible"]["data"]["infeasible_fraction"] <= 0.01


def test_feasible_reference_refuses_a_value_the_plant_cannot_rest_at():
    from backend_core.AgentMPC.agents.reference_authority import FeasibleReference

    res = FeasibleReference(_cart_pole(), primary=2, goal=0.3).generate(0.01, 5.0)
    assert not res["ok"] and res["reference"] is None
    assert "no steady state" in res["message"]
    with pytest.raises(ValueError):
        FeasibleReference(_cart_pole(), primary=2, goal=0.3)(0.01, 5.0)


def test_feasible_reference_plugs_into_the_closed_loop_as_a_custom_trajectory():
    from backend_core.AgentMPC.agents.evaluator import run_closed_loop
    from backend_core.AgentMPC.agents.reference_authority import FeasibleReference
    from backend_core.AgentMPC.mpc.config import Config

    dyn = _cart_pole()
    cfg = Config()
    cfg.data.dt_mpc, cfg.data.simulation_time = 0.02, 8.0
    cfg.data.trajectory_mode = "custom"
    cfg.data.custom_trajectory_fn = FeasibleReference(dyn, primary=0, goal=0.5)
    res = run_closed_loop(dyn, cfg, {"Np": 40, "Nc": 8, "Q": [10, 1, 1, 0.1], "R": [0.01]})
    assert "error" not in res and not res["unstable"], res.get("error") or res.get("unstable_reason")
    assert res["states"][-1][0] == pytest.approx(0.5, abs=0.05)


# ---- 3. the disturbance limit --------------------------------------------------
def test_disturbance_limit_is_where_the_actuator_runs_out():
    """Hover: thrust 0..20 against a 9.81 trim. A downward push of 10*s m/s^2
    needs T = 9.81 + 10*s, and the steady state may use 80% of the headroom
    above the trim, so s_max = 0.8 * (20 - 9.81) / 10."""
    from backend_core.AgentMPC.agents.disturbance_limits import disturbance_limits

    # wide state bounds and a start at the target: only the actuator binds
    lim = disturbance_limits(_hover(w_bound=50.0, z_room=50.0, start=(1.0, 0.0)), direction=np.array([0.0, -10.0]))
    assert lim["ok"]
    assert lim["max_fraction"] == pytest.approx(0.8 * (20 - 9.81) / 10, rel=0.02)
    assert any("T at upper limit" in b for b in lim["binding"])


def test_a_push_that_throws_the_plant_at_its_bounds_is_refused_even_if_it_could_be_held():
    """Tight velocity bounds: the actuator could hold the push at steady state,
    but on the way the climb rate swings most of the way to its bound, where
    an MPC with hard state constraints runs out of feasible moves."""
    from backend_core.AgentMPC.agents.disturbance_limits import disturbance_limits

    roomy = disturbance_limits(_hover(w_bound=50.0, z_room=50.0, start=(1.0, 0.0)), direction=np.array([0.0, -10.0]))
    tight = disturbance_limits(_hover(w_bound=0.3, z_room=50.0, start=(1.0, 0.0)), direction=np.array([0.0, -10.0]))
    assert tight["max_fraction"] < 0.8 * roomy["max_fraction"]
    assert any("transient" in b for b in tight["binding"]), tight["binding"]


def test_a_weak_actuator_has_a_minimum_effectiveness():
    from backend_core.AgentMPC.agents.disturbance_limits import disturbance_limits

    lim = disturbance_limits(_hover(), effectiveness=0.4)     # 0.4 * 20 = 8 < 9.81
    assert not lim["ok"] and lim["max_fraction"] == 0.0
    # lambda * T_steady_max >= 9.81 with T_steady_max = 9.81 + 0.8 * (20 - 9.81)
    assert lim["min_effectiveness"] == pytest.approx(9.81 / (9.81 + 0.8 * (20 - 9.81)), abs=0.01)
    assert "lambda" in lim["message"]


def test_the_ui_direction_is_capped_by_the_plants_own_limit():
    from backend_core.AgentMPC.agents.disturbance_limits import disturbance_limits

    down = np.array([0.0, -10.0])
    weak = disturbance_limits(_hover(max_thrust=11.0, w_bound=50.0, z_room=50.0, start=(1.0, 0.0)), direction=down)
    strong = disturbance_limits(_hover(max_thrust=40.0, w_bound=50.0, z_room=50.0, start=(1.0, 0.0)), direction=down)
    assert 0.0 < weak["max_fraction"] < strong["max_fraction"]
