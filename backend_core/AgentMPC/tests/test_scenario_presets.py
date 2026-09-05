"""
Run with: pytest backend_core/AgentMPC/tests/test_scenario_presets.py -v

Regression guard for the "Overshoot shows N/A whenever the target is
non-zero" report. The actual trigger has nothing to do with the target's
sign or magnitude: it's the common plugin convention of declaring
``default_initial_state == default_target`` ("start already at the operating
point" -- more likely for a plugin with a genuinely nonzero equilibrium,
e.g. "hold this RPM", than for a zero-target one). With zero initial error,
there is nothing for Overshoot to measure, so ``overshoot_meaningful`` comes
back False -- correct math, but ``apply_scenario_level``'s Level 1/2 already
guarded against exactly this by nudging the initial state away from the
target. Manual Simulation had no such guard (it doesn't go through
``apply_scenario_level`` at all), so a degenerate plugin always showed
Overshoot N/A there regardless of Scenario Level.

The fix pulls the nudge out into a public, shared
``nudge_if_starts_at_target`` (used by Level 1, Level 2, AND Manual
Simulation now) instead of leaving it duplicated/only-inline where Manual
Simulation's code path couldn't reach it.
"""

import numpy as np
import pytest

from backend_core.AgentMPC.agents.scenario_presets import (
    apply_scenario_level, nudge_if_starts_at_target,
)
from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig
from backend_core.AgentMPC.agents.evaluator import run_closed_loop
from backend_core.AgentMPC.mpc.config import Config


class _LinearPlant(BaseDynamics):
    """Minimal 2-state (position, velocity) linear plant, generic enough to
    drive a real closed-loop MPC simulation end-to-end."""

    def __init__(self, target, initial_state=None, state_bounds=None):
        target = np.asarray(target, dtype=float)
        x0 = np.asarray(initial_state, dtype=float) if initial_state is not None else target.copy()
        config = SystemConfig(
            n_states=2, n_inputs=1,
            state_names=["pos", "vel"], input_names=["F"],
            default_initial_state=x0,
            default_target=target.copy(),
            state_bounds=state_bounds or (np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
            input_bounds=(np.array([-5.0]), np.array([5.0])),
            params={"mass": 1.0, "damping": 0.5},
        )
        super().__init__(config)

    def dynamics(self, x, u):
        pos, vel = x
        F = float(np.asarray(u).reshape(-1)[0])
        mass, damping = self.params["mass"], self.params["damping"]
        return np.array([vel, (F - damping * vel) / mass])


def _run(dynamics, dt=0.05, sim_time=5.0):
    cfg = Config()
    cfg.mpc.prediction_horizon = 12
    cfg.mpc.control_horizon = 5
    cfg.data.dt_mpc = dt
    cfg.data.simulation_time = sim_time
    cfg.data.trajectory_mode = "reg"
    return run_closed_loop(dynamics, cfg, {"Np": 12, "Nc": 5, "Q": [10.0, 1.0], "R": [0.1], "P": [10.0, 1.0]})


# --------------------------------------------------------------------------
# nudge_if_starts_at_target itself
# --------------------------------------------------------------------------

def test_nudge_leaves_a_distinct_initial_state_untouched():
    dyn = _LinearPlant(target=[0.0, 0.0], initial_state=[0.2, 0.0])
    x0 = dyn.config.default_initial_state.copy()
    result = nudge_if_starts_at_target(dyn, x0, dyn.config.default_target)
    assert np.array_equal(result, x0)


def test_nudge_moves_away_from_a_zero_target_when_degenerate():
    dyn = _LinearPlant(target=[0.0, 0.0])  # initial_state defaults to == target
    result = nudge_if_starts_at_target(dyn, dyn.config.default_initial_state, dyn.config.default_target)
    assert not np.allclose(result, dyn.config.default_target)


def test_nudge_moves_away_from_a_nonzero_target_when_degenerate():
    """The actual reported scenario: a nonzero setpoint plugin whose default
    initial state equals that setpoint."""
    dyn = _LinearPlant(target=[5.0, 0.0])  # initial_state defaults to == target
    result = nudge_if_starts_at_target(dyn, dyn.config.default_initial_state, dyn.config.default_target)
    assert not np.allclose(result, dyn.config.default_target)


# --------------------------------------------------------------------------
# apply_scenario_level (Level 1 / 2) -- still nudges, now via the shared helper
# --------------------------------------------------------------------------

@pytest.mark.parametrize("level", [1, 2])
def test_apply_scenario_level_nudges_a_degenerate_nonzero_target(level):
    dyn = _LinearPlant(target=[5.0, 0.0])
    cfg = Config()
    x0_after, target_after, _ = apply_scenario_level(dyn, cfg, level=level)
    assert not np.allclose(x0_after, target_after)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_apply_scenario_level_leaves_a_distinct_default_untouched(level):
    dyn = _LinearPlant(target=[5.0, 0.0], initial_state=[2.0, 0.0])
    cfg = Config()
    x0_after, target_after, _ = apply_scenario_level(dyn, cfg, level=level)
    assert np.array_equal(x0_after, np.array([2.0, 0.0]))


# --------------------------------------------------------------------------
# Level 3: parametric mismatch, NOT a harder initial state
# --------------------------------------------------------------------------

def test_level_3_no_longer_pushes_the_initial_state():
    """Level 3 used to shove the initial state toward the edge of its
    declared bounds; it now starts exactly where every other level does, so
    the only thing that differs is the plant itself. Comparing runs across
    levels is meaningless if the starting point moves too."""
    dyn = _LinearPlant(target=[0.0, 0.0], initial_state=[2.0, 0.0])
    cfg = Config()
    x0_after, _, _ = apply_scenario_level(dyn, cfg, level=3)
    assert np.array_equal(x0_after, np.array([2.0, 0.0]))


def test_level_3_perturbs_parameters_within_the_requested_bound():
    dyn = _LinearPlant(target=[0.0, 0.0], initial_state=[2.0, 0.0])
    cfg = Config()
    _, _, perturbed = apply_scenario_level(dyn, cfg, level=3, max_param_uncertainty=0.5)
    assert perturbed, "Level 3 should perturb at least one numeric parameter"
    for _name, (old, new) in perturbed.items():
        assert old <= new <= old * 1.5 + 1e-12


def test_level_3_uncertainty_bound_is_actually_honored():
    """A tighter bound must produce a strictly smaller perturbation for the
    same seed -- otherwise the new percentage slider would be decorative."""
    def _run(bound):
        dyn = _LinearPlant(target=[0.0, 0.0], initial_state=[2.0, 0.0])
        cfg = Config()
        _, _, perturbed = apply_scenario_level(dyn, cfg, level=3, max_param_uncertainty=bound)
        return {k: (new / old - 1) for k, (old, new) in perturbed.items()}

    small, large = _run(0.05), _run(0.5)
    assert small and large
    for key in small:
        assert small[key] <= large[key] + 1e-12
        assert small[key] <= 0.05 + 1e-12


def test_level_3_can_skip_parameter_perturbation_entirely():
    dyn = _LinearPlant(target=[0.0, 0.0], initial_state=[2.0, 0.0])
    cfg = Config()
    _, _, perturbed = apply_scenario_level(dyn, cfg, level=3, perturb_physical_params=False)
    assert perturbed == {}


# --------------------------------------------------------------------------
# End-to-end: Overshoot is actually computable now for a degenerate plugin
# --------------------------------------------------------------------------

def test_overshoot_was_not_a_number_before_the_nudge():
    """Pins the ACTUAL bug: without any nudge, a degenerate nonzero-target
    plugin gives overshoot_meaningful=False -- confirms the failure mode
    this fix addresses is real, not hypothetical."""
    dyn = _LinearPlant(target=[5.0, 0.0])  # default_initial_state == default_target
    result = _run(dyn)
    assert result["metrics"].overshoot_meaningful is False


def test_overshoot_is_computable_after_manual_simulations_nudge():
    """Exactly what render_manual_simulation_tab() now does before calling
    run_closed_loop() when the user is NOT using a custom initial state."""
    dyn = _LinearPlant(target=[5.0, 0.0])
    dyn.config.default_initial_state = nudge_if_starts_at_target(
        dyn, dyn.config.default_initial_state, dyn.config.default_target,
    )
    result = _run(dyn)
    assert result["metrics"].overshoot_meaningful is True


def test_a_genuinely_distinct_manual_initial_state_is_never_touched():
    """The nudge must never fire when the user (or Manual Simulation's own
    "custom initial state" override) already provided a distinct value --
    only the degenerate "identical to target" case is special-cased."""
    dyn = _LinearPlant(target=[5.0, 0.0], initial_state=[8.0, 0.0])
    original = dyn.config.default_initial_state.copy()
    nudged = nudge_if_starts_at_target(dyn, dyn.config.default_initial_state, dyn.config.default_target)
    assert np.array_equal(nudged, original)
