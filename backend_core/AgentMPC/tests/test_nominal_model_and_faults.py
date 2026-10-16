"""
Run with: pytest backend_core/AgentMPC/tests/test_nominal_model_and_faults.py -v

The controller must not know what the scenario did to the plant.

This is the property every robustness claim rests on, and it did not hold.
``run_closed_loop`` built the controller and the simulator from the SAME
dynamics object, and ``perturb_physical_parameters`` mutates that object's
params in place -- so the MPC's Jacobian moved with the plant. It predicted
with the perturbed mass and gravity, knew exactly what had changed, and had
nothing left to be robust against. Level 3 was measuring "tune against a
slightly different plant", not parametric uncertainty.

The plant is now the real, perturbed system while the controller is built from
a nominal copy frozen before any perturbation. Level 4 extends the same idea to
things a model cannot express at all: an actuator that delivers only part of
what it is asked for, and an external disturbance acting on the plant.
"""

import numpy as np
import pytest

from backend_core.AgentMPC.agents.evaluator import run_closed_loop
from backend_core.AgentMPC.agents.scenario_presets import (
    SCENARIO_LEVEL_NAMES, apply_scenario_level,
)
from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig
from backend_core.AgentMPC.mpc.config import Config
from backend_core.AgentMPC.mpc.controller import GenericMPC

PARAMS = {"Np": 12, "Nc": 2, "Q": [10.0, 1.0], "R": [0.1], "P": [10.0, 1.0]}


class _Plant(BaseDynamics):
    """Mass-damper whose behaviour depends on a declared physical parameter, so
    a perturbation to that parameter is visible in the Jacobian."""

    def __init__(self):
        super().__init__(SystemConfig(
            n_states=2, n_inputs=1,
            state_names=["position", "velocity"], input_names=["force"],
            default_initial_state=np.array([0.4, 0.0]),
            default_target=np.array([0.0, 0.0]),
            state_bounds=(np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
            input_bounds=(np.array([-8.0]), np.array([8.0])),
            params={"mass": 1.0, "damping": 0.4},
        ))

    def dynamics(self, x, u):
        force = float(np.asarray(u).reshape(-1)[0])
        mass, damping = self.params["mass"], self.params["damping"]
        return np.array([x[1], (force - damping * x[1]) / mass])


def _cfg():
    c = Config()
    c.mpc.prediction_horizon, c.mpc.control_horizon = 12, 2
    c.data.dt_mpc = 0.02
    c.data.simulation_time = 3.0
    c.data.trajectory_mode = "reg"
    c.random_seed = 11
    return c


@pytest.fixture
def cfg():
    return _cfg()


def _linearised(dynamics, cfg):
    cfg.mpc.initialize_weights(dynamics.n_states, dynamics.n_inputs)
    A, B, _, _ = GenericMPC(dynamics, cfg)._linearize(
        np.array([0.2, 0.0]), dynamics.get_equilibrium_input())
    return A, B


# -------------------------------------------------- the nominal/real split --
def test_level_3_leaves_the_controller_with_the_nominal_model(cfg):
    plant = _Plant()
    nominal_params = dict(plant.params)

    _, _, perturbed = apply_scenario_level(plant, cfg, 3)

    assert perturbed, "the plant was supposed to be perturbed"
    assert dict(plant.params) != nominal_params, "the plant should have moved"
    assert dict(cfg.data.nominal_dynamics.params) == nominal_params, \
        "the controller's model must keep the parameters it was designed for"


def test_the_controllers_jacobian_does_not_follow_the_plant(cfg):
    """The regression that matters: the two models must differ numerically,
    not merely be separate objects."""
    plant = _Plant()
    apply_scenario_level(plant, cfg, 3)

    A_model, B_model = _linearised(cfg.data.nominal_dynamics, cfg)
    A_plant, B_plant = _linearised(plant, cfg)

    assert not np.allclose(A_model, A_plant) or not np.allclose(B_model, B_plant)


@pytest.mark.parametrize("level", [1, 2])
def test_unperturbed_levels_are_unchanged(level, cfg):
    """Nothing about Levels 1 and 2 moves: no parameter is perturbed, so the
    nominal copy is identical and no fault is armed."""
    plant = _Plant()
    before = dict(plant.params)

    apply_scenario_level(plant, cfg, level)

    assert dict(plant.params) == before
    assert dict(cfg.data.nominal_dynamics.params) == before
    assert cfg.data.actuator_effectiveness is None
    assert cfg.data.state_disturbance is None
    assert cfg.data.input_disturbance is None


def test_a_level_change_clears_a_previous_faults_settings(cfg):
    """Re-running at a lower level inside one session must not leave the
    previous run's fault armed."""
    plant = _Plant()
    apply_scenario_level(plant, cfg, 4, actuator_effectiveness=0.5)
    assert cfg.data.actuator_effectiveness is not None

    apply_scenario_level(_Plant(), cfg, 1)

    assert cfg.data.actuator_effectiveness is None
    assert cfg.data.state_disturbance is None


# ---------------------------------------------------------------- level 4 --
def test_level_4_exists_and_is_named():
    assert SCENARIO_LEVEL_NAMES[4] == "Fault & Disturbance"


def test_a_faulted_actuator_delivers_only_lambda_of_what_was_commanded(cfg):
    plant = _Plant()
    apply_scenario_level(plant, cfg, 4, actuator_effectiveness=0.6,
                          perturb_physical_params=False,
                          state_disturbance=np.zeros(2))

    result = run_closed_loop(plant, cfg, dict(PARAMS))

    assert "error" not in result, result.get("error")
    commanded, applied = result["inputs"], result["applied_inputs"]
    assert commanded.shape == applied.shape
    moving = commanded != 0
    assert moving.any(), "the controller never actuated, so there is nothing to check"
    assert np.allclose(applied[moving] / commanded[moving], 0.6)


def test_the_controller_is_not_told_about_the_fault(cfg):
    """A faulted actuator is not part of the model: the controller's own
    linearisation is identical with and without the fault."""
    plant = _Plant()
    apply_scenario_level(plant, cfg, 4, actuator_effectiveness=0.3,
                          perturb_physical_params=False)
    A_faulted, B_faulted = _linearised(cfg.data.nominal_dynamics, cfg)

    A_healthy, B_healthy = _linearised(_Plant(), _cfg())

    assert np.allclose(A_faulted, A_healthy)
    assert np.allclose(B_faulted, B_healthy)


def test_an_external_disturbance_actually_moves_the_plant():
    """A disturbance that changes nothing would be worse than none at all."""
    clean_plant = _Plant()
    clean_cfg = _cfg()
    apply_scenario_level(clean_plant, clean_cfg, 1)
    undisturbed = run_closed_loop(clean_plant, clean_cfg, dict(PARAMS))

    disturbed_plant = _Plant()
    disturbed_cfg = _cfg()
    apply_scenario_level(disturbed_plant, disturbed_cfg, 4, actuator_effectiveness=1.0,
                          perturb_physical_params=False,
                          state_disturbance=np.array([0.0, 0.5]))
    disturbed = run_closed_loop(disturbed_plant, disturbed_cfg, dict(PARAMS))

    assert "error" not in undisturbed and "error" not in disturbed
    n = min(len(undisturbed["states"]), len(disturbed["states"]))
    assert not np.allclose(undisturbed["states"][:n], disturbed["states"][:n])


def test_applied_inputs_match_commanded_when_nothing_is_faulted(cfg):
    plant = _Plant()
    apply_scenario_level(plant, cfg, 1)

    result = run_closed_loop(plant, cfg, dict(PARAMS))

    assert np.allclose(result["inputs"], result["applied_inputs"])


def test_the_integrator_applies_the_disturbance_at_every_stage():
    """Held constant across the interval, but added inside RK4 rather than
    tacked onto the endpoint.

    The distinction is measurable. Starting at rest with no input, a constant
    disturbance d on the acceleration drives v' = d - (b/m) v, whose exact
    solution after dt is (d m / b)(1 - exp(-b dt / m)) -- strictly less than
    d*dt, because the damping starts opposing the motion as soon as the
    velocity leaves zero. An implementation that added d only to the final
    result would land on exactly d*dt and miss that interaction entirely.
    """
    plant = _Plant()
    dt = 0.1
    d = np.array([0.0, 1.0])           # constant acceleration of 1
    rate = plant.params["damping"] / plant.params["mass"]

    stepped = plant.rk4_step(np.array([0.0, 0.0]), np.zeros(1), dt, d_state=d)

    exact_velocity = (d[1] / rate) * (1.0 - np.exp(-rate * dt))
    exact_position = (d[1] / rate) * (dt - (1.0 - np.exp(-rate * dt)) / rate)

    assert stepped[1] == pytest.approx(exact_velocity, rel=1e-6)
    assert stepped[0] == pytest.approx(exact_position, rel=1e-4)
    # ...and demonstrably NOT the naive endpoint answer.
    assert stepped[1] != pytest.approx(d[1] * dt, rel=1e-4)
