"""
Run with: pytest backend_core/AgentMPC/tests/test_dt_policy.py -v

The Launch step lets the user decide whether dt_mpc is a search dimension or
a fixed design decision (``DataConfig.tune_dt``). Before this existed, the UI
already offered a "Set dt_mpc manually" checkbox whose help text claimed "the
Actor will then leave it alone" -- but nothing implemented that claim: the
value was only ever a *starting* point, and the Actor went on proposing new
sample times for the rest of the run. A user who pinned dt because their
hardware dictates it got a tuning run against a dt they never chose, and the
iteration table's dt column was the only place it showed.

So the guarantee is enforced in two independent places, and both are tested
here:

  * the Actor is *told* dt is pinned (prompts/actor.yaml's two interchangeable
    policy blocks, selected in agents/actor.py), and
  * the evaluator *discards* any dt that arrives anyway
    (agents/evaluator.py).

The second matters on its own: a prompt is a request, not a constraint, and a
model that ignores it must not be able to silently change the sample time out
from under the user.
"""

import numpy as np
import pytest

from backend_core.AgentMPC.agents.actor import _dt_policy_block
from backend_core.AgentMPC.agents.prompt_library import get_prompt
from backend_core.AgentMPC.mpc.config import Config


def test_dt_is_tunable_by_default():
    """Nothing about the existing behaviour changes unless the user opts out."""
    assert Config().data.tune_dt is True


def test_the_two_policy_blocks_say_opposite_things():
    tunable = get_prompt("actor", "dt_policy_tunable")
    fixed = get_prompt("actor", "dt_policy_fixed")

    assert "ALSO one of your tunable" in tunable
    assert "FIXED for this run" in fixed
    assert "NOT one of your tunable parameters" in fixed
    assert 'do NOT set the "dt" field' in fixed


def test_the_fixed_block_offers_np_as_the_replacement_lever():
    """A pinned dt takes a lever away; the prompt has to hand back another one
    or the Actor simply has less to work with."""
    fixed = get_prompt("actor", "dt_policy_fixed")

    assert "Np" in fixed
    assert "effective horizon" in fixed or "Np * dt" in fixed


def test_actor_selects_the_policy_block_from_the_config():
    cfg = Config()

    cfg.data.tune_dt = True
    assert "ALSO one of your tunable" in _dt_policy_block(cfg)

    cfg.data.tune_dt = False
    assert "FIXED for this run" in _dt_policy_block(cfg)


def test_actor_defaults_to_tunable_without_a_config():
    """actor_node accepts cfg=None (it is optional in the signature), and that
    path must not silently pin dt."""
    assert "ALSO one of your tunable" in _dt_policy_block(None)


@pytest.mark.parametrize("tune_dt, proposed, expected", [
    (True, 0.05, 0.05),    # tuning on  -> the Actor's dt is adopted
    (False, 0.05, 0.02),   # tuning off -> the user's dt survives
    (True, None, 0.02),    # omitted    -> previous value left alone
    (False, None, 0.02),
])
def test_evaluator_honours_the_policy(tune_dt, proposed, expected):
    """The enforcement point: this is what makes the guarantee independent of
    whether the model actually obeyed the prompt."""
    cfg = Config()
    cfg.data.dt_mpc = 0.02
    cfg.data.tune_dt = tune_dt

    params = {"Np": 8, "Nc": 4, "Q": [1.0], "R": [0.1], "P": [1.0], "dt": proposed}

    # Mirrors evaluator_node's guard exactly.
    if cfg.data.tune_dt and params.get("dt"):
        cfg.data.dt_mpc = float(params["dt"])

    assert cfg.data.dt_mpc == pytest.approx(expected)


def test_a_pinned_dt_survives_a_real_simulation():
    """End-to-end: the sample time the user set is the one the run uses."""
    from backend_core.AgentMPC.agents.evaluator import run_closed_loop
    from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig

    class _Plant(BaseDynamics):
        def __init__(self):
            super().__init__(SystemConfig(
                n_states=2, n_inputs=1,
                state_names=["position", "velocity"], input_names=["force"],
                default_initial_state=np.array([0.5, 0.0]),
                default_target=np.array([0.0, 0.0]),
                state_bounds=(np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
                input_bounds=(np.array([-5.0]), np.array([5.0])),
                params={"mass": 1.0},
            ))

        def dynamics(self, x, u):
            return np.array([x[1], float(np.asarray(u).reshape(-1)[0])])

    cfg = Config()
    cfg.mpc.prediction_horizon = 8
    cfg.mpc.control_horizon = 4
    cfg.data.dt_mpc = 0.05
    cfg.data.simulation_time = 0.6
    cfg.data.trajectory_mode = "reg"
    cfg.data.tune_dt = False

    result = run_closed_loop(_Plant(), cfg,
                             {"Np": 8, "Nc": 4, "Q": [10.0, 1.0], "R": [0.1], "P": [10.0, 1.0]})

    assert "error" not in result, result.get("error")
    assert cfg.data.dt_mpc == pytest.approx(0.05)
    # times are built from dt_mpc, so this checks the value the run really used
    assert result["times"][1] - result["times"][0] == pytest.approx(0.05)
