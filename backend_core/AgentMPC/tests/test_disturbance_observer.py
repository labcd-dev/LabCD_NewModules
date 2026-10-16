"""
Run with: pytest backend_core/AgentMPC/tests/test_disturbance_observer.py -v

The optional disturbance observer (offset-free MPC).

Without it, the incremental (delta-u) formulation finds the input that cancels
a constant disturbance -- but can only HOLD that input while an error persists
to generate it. So a constant disturbance leaves a permanent state offset that
grows with the disturbance and that no choice of Q, R, Np or Nc removes.

With it, the controller estimates an additive per-step disturbance from its own
one-step prediction error,

    eps_k   = x_k - x_hat_{k|k-1}
    d_hat_k = d_hat_{k-1} + alpha * eps_k

and folds it into the affine offset, c_tilde = c + d_hat. Once the estimate
converges the model is exact, and delta-u = 0 then requires x = x_ref.

These tests pin both halves: that it removes the offset, and that it leaves
everything else exactly as it was -- it is off by default so that earlier
results stay comparable.
"""

import numpy as np
import pytest

from backend_core.AgentMPC.agents.evaluator import run_closed_loop
from backend_core.AgentMPC.agents.mpc_equations import mpc_algorithm_latex, observer_summary
from backend_core.AgentMPC.agents.scenario_presets import apply_scenario_level
from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig
from backend_core.AgentMPC.mpc.config import Config

MASS = 1.0
PARAMS = {"Np": 15, "Nc": 3, "Q": [20.0, 1.0], "R": [0.1], "P": [20.0, 1.0]}


class _Hover(BaseDynamics):
    """A mass held at a height, with gravity trimmed out of the input -- so
    the origin is an equilibrium and a constant push is a clean disturbance."""

    def __init__(self):
        super().__init__(SystemConfig(
            n_states=2, n_inputs=1,
            state_names=["z", "w"], input_names=["thrust"],
            default_initial_state=np.array([0.0, 0.0]),
            default_target=np.array([1.0, 0.0]),
            state_bounds=(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
            input_bounds=(np.array([-20.0]), np.array([20.0])),
            params={"mass": MASS, "drag": 0.35},
        ))

    def dynamics(self, x, u):
        thrust = float(np.asarray(u).reshape(-1)[0])
        return np.array([x[1], (thrust - self.params["drag"] * x[1]) / self.params["mass"]])

    def get_equilibrium_input(self):
        return np.zeros(1)


def _run(observer, d_u=0.0, alpha=0.3, sim_time=12.0):
    plant = _Hover()
    cfg = Config()
    cfg.mpc.prediction_horizon, cfg.mpc.control_horizon = 15, 3
    cfg.data.dt_mpc, cfg.data.simulation_time = 0.02, sim_time
    cfg.data.trajectory_mode, cfg.random_seed = "reg", 7
    cfg.mpc.use_disturbance_observer = observer
    cfg.mpc.observer_gain = alpha
    if d_u:
        apply_scenario_level(plant, cfg, 4, noise_std_value=0.0, actuator_effectiveness=1.0,
                              perturb_physical_params=False, state_disturbance=np.zeros(2),
                              input_disturbance=np.array([d_u]))
    else:
        apply_scenario_level(plant, cfg, 1)
    result = run_closed_loop(plant, cfg, dict(PARAMS))
    assert "error" not in result, result.get("error")
    return result


def _settled_z(result, last_seconds=5.0):
    t = result["times"]
    return float(result["states"][:, 0][t >= t[-1] - last_seconds].mean())


# ------------------------------------------------------------- default ----
def test_the_observer_is_off_by_default():
    """Off by default, so every result produced before it existed stays
    comparable."""
    assert Config().mpc.use_disturbance_observer is False


# ---------------------------------------------------------- the offset ----
@pytest.mark.parametrize("d_u", [2.0, 5.0])
def test_without_the_observer_a_constant_disturbance_leaves_an_offset(d_u):
    """The problem this exists to solve -- pinned, so it is visible if the
    baseline controller ever changes underneath it."""
    offset = _settled_z(_run(observer=False, d_u=d_u)) - 1.0

    assert abs(offset) > 0.05, "the baseline should NOT be offset-free"


@pytest.mark.parametrize("d_u", [2.0, 5.0])
def test_with_the_observer_the_offset_goes_to_zero(d_u):
    offset = _settled_z(_run(observer=True, d_u=d_u)) - 1.0

    assert abs(offset) < 5e-3


def test_both_find_the_cancelling_input():
    """The difference is not WHICH input they find -- both find -d_u -- but
    whether they can hold it without a standing error."""
    for observer in (False, True):
        result = _run(observer=observer, d_u=2.0)
        t = result["times"][:-1]
        thrust = float(result["inputs"][:, 0][t >= t[-1] - 5.0].mean())
        assert thrust == pytest.approx(-2.0, abs=0.05)


# ------------------------------------------------------ nothing else moves --
def test_with_no_disturbance_the_observer_changes_nothing():
    off, on = _run(observer=False), _run(observer=True)

    assert _settled_z(on) == pytest.approx(_settled_z(off), abs=1e-4)


def test_off_records_no_estimates_and_on_records_one_per_step():
    off, on = _run(observer=False, d_u=2.0), _run(observer=True, d_u=2.0)

    assert len(off["disturbance_estimates"]) == 0
    assert len(on["disturbance_estimates"]) == len(on["inputs"])


# ----------------------------------------------------- the estimate itself --
def test_the_estimate_converges_to_the_injected_disturbance():
    """d_hat is a per-step state offset. A thrust disturbance d_u acts on the
    velocity state as dt * d_u / m per step -- the estimate should land there,
    not merely somewhere that happens to close the loop."""
    result = _run(observer=True, d_u=2.0)
    d_hat_w = result["disturbance_estimates"][-1][1]

    assert d_hat_w == pytest.approx(0.02 * 2.0 / MASS, rel=0.02)


def test_the_estimate_is_saturated():
    """Anti-windup: bounded at dt x each state's declared range, so an
    impossible target cannot drive it without limit."""
    result = _run(observer=True, d_u=2.0)
    limit = 0.02 * 10.0          # dt x span of [-5, 5]

    assert np.all(np.abs(result["disturbance_estimates"]) <= limit + 1e-12)


# ------------------------------------------------------------ equations ----
def test_the_equations_gain_an_observer_step_only_when_it_is_on():
    off = mpc_algorithm_latex(observer=False)
    on = mpc_algorithm_latex(observer=True, alpha=0.3)

    assert len(on) == len(off) + 1
    assert not any("Disturbance observer" in s["title"] for s in off)
    assert any("Disturbance observer" in s["title"] for s in on)


def test_the_equations_show_the_corrected_offset_and_the_gain():
    joined = " ".join(e for s in mpc_algorithm_latex(observer=True, alpha=0.25) for e in s["latex"])

    assert r"\tilde{c}_k" in joined
    assert r"\tilde{S}_c" in joined
    assert "0.25" in joined


def test_H_is_the_same_with_and_without_the_observer():
    """The claim the equations make -- and the reason the solver's warm start
    survives -- is that only q and the state rows change."""
    def h_line(steps):
        return next(e for s in steps for e in s["latex"] if e.startswith("H ="))

    off = h_line(mpc_algorithm_latex(observer=False))
    on = h_line(mpc_algorithm_latex(observer=True))
    assert off.split(r"\qquad")[0] == on.split(r"\qquad")[0]


def test_the_summary_says_what_the_toggle_does():
    assert "Offset-free" in observer_summary(True)
    assert "permanent state offset" in observer_summary(False)


def test_no_latex_spacing_command_runs_into_the_next_symbol():
    """Adjacent raw strings concatenate silently, so a '\qquad' at the end of
    one and a 'q' at the start of the next became '\qquadq' -- an unknown
    command KaTeX printed in red in the live equations panel. Any spacing
    command must be followed by a space, a brace or a backslash."""
    import re

    for observer in (False, True):
        for step in mpc_algorithm_latex(observer=observer):
            for expr in step["latex"]:
                assert not re.search(r"\\q?quad[A-Za-z]", expr), expr
