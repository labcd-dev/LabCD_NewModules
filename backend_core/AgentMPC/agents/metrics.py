"""
================================================================================
agents/metrics.py
================================================================================
Performance metrics computed from a closed-loop simulation, used by both the
Evaluator agent (for the LLM-facing feedback loop) and the numeric tuner
(for a fast, LLM-free search -- see numeric_tuner.py).

Kept as pure functions (arrays in, floats out) so they're trivial to unit
test and reuse outside the agent graph. Everything here is dimension-generic
(works for any n_states/n_inputs) -- nothing is specific to any one dynamics
plugin.

v3 changes:

  5. INTEGRAL ERROR (IAE / ISE), per state. MSE/per-state-MSE is already a
     time-averaged quantity, but it's still just one aggregate number per
     state; the classical control-engineering "integral of error over the
     run" indices (IAE = integral of |e(t)| dt, ISE = integral of e(t)^2 dt)
     make the same accumulated-deviation information explicit and in the
     units/vocabulary a controls engineer -- and, when described this way in
     the prompt, the LLM -- reasons about directly: "this state accumulated
     this much total tracking error over the whole run", not just "this
     state's error was, on average, this big at a random instant".

  6. REGULATION vs. TRACKING awareness. Overshoot / settling time /
     oscillation-count are classical *step-response* metrics: they're only
     meaningful when there's a single fixed target the system moves toward
     once and (ideally) settles at. For a moving reference (trajectory_mode
     "sin"/"pulse" in mpc/config.py), the tracking error's sign flips
     constantly as the state crosses back and forth around the reference
     *by design*, even during perfectly good tracking -- which the old
     always-on overshoot/oscillation logic misread as huge overshoot and
     dozens of "oscillations". Those three metrics are now computed only
     when ``is_regulation=True`` is passed in; otherwise they're reported as
     not-meaningful (and scalar_cost() drops them from the ranking formula
     entirely for tracking runs, relying on MSE/IAE/ISE and control effort
     instead, which stay meaningful for any reference type).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PerformanceMetrics:
    mse: float
    rmse: float
    overshoot: float           # fraction, e.g. 0.15 == swung 15% of the initial error past the target --
                                  # aggregated ONLY over states where overshoot is actually computable; see
                                  # overshoot_meaningful below
    overshoot_meaningful: bool = True   # False only if NO state had a computable overshoot (all moving/
                                          # zero-initial-error) -- distinct from "overshoot happens to be 0.0"
    settling_time: float = 0.0        # seconds until the error enters tolerance AND stays there
    settled: bool = False                # True only if it actually entered and *held* the tight tolerance band
    is_stable: bool = False               # True if the error is bounded/converging (NOT diverging or growing) --
                                    # a much more lenient bar than `settled`. See compute_metrics' docstring.
    control_effort: float = 0.0         # mean squared control input (proxy for energy/actuator wear)
    oscillation_count: int = 0     # zero-crossings of the worst-off state's error signal
    per_state_mse: List[float] = field(default_factory=list)
    per_state_overshoot: List[Optional[float]] = field(default_factory=list)   # None = not computable for
                                                                                  # that state (moving reference
                                                                                  # or zero initial error)
    # integral error -- meaningful for ANY trajectory type (regulation or tracking)
    integral_abs_error: float = 0.0        # sum over states of per-state IAE
    integral_sq_error: float = 0.0          # sum over states of per-state ISE
    per_state_iae: List[float] = field(default_factory=list)   # integral |error| dt, per state
    per_state_ise: List[float] = field(default_factory=list)    # integral error^2 dt, per state
    # whether the FULL reference is constant (every state) -- used to choose
    # the settling-tolerance basis (see compute_metrics). NOT what gates
    # overshoot anymore -- overshoot is now computed per-state regardless of
    # this flag; see overshoot_meaningful above.
    is_regulation: bool = True

    def as_dict(self) -> dict:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "overshoot": self.overshoot,
            "overshoot_meaningful": self.overshoot_meaningful,
            "settling_time": self.settling_time,
            "settled": self.settled,
            "is_stable": self.is_stable,
            "control_effort": self.control_effort,
            "oscillation_count": self.oscillation_count,
            "per_state_mse": self.per_state_mse,
            "per_state_overshoot": self.per_state_overshoot,
            "integral_abs_error": self.integral_abs_error,
            "integral_sq_error": self.integral_sq_error,
            "per_state_iae": self.per_state_iae,
            "per_state_ise": self.per_state_ise,
            "is_regulation": self.is_regulation,
        }


def compute_metrics(
    states: np.ndarray,
    inputs: np.ndarray,
    reference: np.ndarray,
    dt: float,
    settling_tolerance: float = 0.05,
    min_settle_fraction: float = 0.15,
    is_regulation: bool = True,
) -> PerformanceMetrics:
    """
    Args:
        states: (T, n_states)
        inputs: (T, n_inputs)
        reference: (T, n_states) -- same length as states
        dt: timestep between samples
        settling_tolerance: fraction of the initial error norm considered "close enough"
        min_settle_fraction: the tolerance band must hold for at least this
            fraction of the recorded trajectory (floored at 3 samples) before
            it counts as "settled" -- guards against a trajectory that's
            still oscillating but happens to end near zero error.
        is_regulation: True for a fixed target (trajectory_mode="reg"),
            False for a moving reference ("sin"/"pulse"). Gates whether
            overshoot/settling_time/oscillation_count are computed at all --
            see the v3 changelog note in this module's docstring for why.
    """
    T = min(len(states), len(reference))
    states, reference = states[:T], reference[:T]
    n_states = states.shape[1] if states.ndim == 2 else 1

    error = states - reference
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    per_state_mse = np.mean(error**2, axis=0).tolist()

    # ---- integral error: meaningful regardless of trajectory type ----
    per_state_iae = (np.sum(np.abs(error), axis=0) * dt).tolist()
    per_state_ise = (np.sum(error**2, axis=0) * dt).tolist()
    integral_abs_error = float(np.sum(per_state_iae))
    integral_sq_error = float(np.sum(per_state_ise))

    initial_error = error[0]
    abs_initial = np.abs(initial_error)
    moving_mask = abs_initial > 1e-6

    control_effort = float(np.mean(inputs**2)) if inputs.size else 0.0
    err_norm = np.linalg.norm(error, axis=1)
    min_hold_samples = max(int(min_settle_fraction * T), 3)
    hold_fraction = 0.95

    # ---- settled / settling_time: computed the SAME way (a tolerance band on
    # the error norm, held for a meaningful minimum duration) for BOTH
    # regulation and tracking runs -- only the tolerance's *baseline* differs,
    # because "how close counts as close enough" needs a different reference
    # point in each case:
    #   regulation: tolerance scales with how far the initial state was from
    #     the (fixed) target -- "did it close X% of the original gap".
    #   tracking: there's no single "initial gap" that means anything for a
    #     continuously moving target, so tolerance instead scales with the
    #     reference signal's own average magnitude -- "is the tracking error
    #     small relative to the size of the thing being tracked". A tracking
    #     run that settles into a small bounded error around the moving
    #     target (the best any real controller can do -- it can't have zero
    #     phase lag) correctly reports settled=True here; a previous version
    #     of this function hardcoded settled=False for every tracking run
    #     regardless of how well it was actually tracking, which was wrong.
    if is_regulation:
        tol = max(settling_tolerance * float(np.linalg.norm(initial_error)), 1e-6)
    else:
        ref_scale = float(np.mean(np.linalg.norm(reference, axis=1)))
        tol = max(settling_tolerance * ref_scale, 1e-6)

    within_tol = err_norm <= tol
    settling_time = float(T * dt)
    settled = False
    for idx in np.where(within_tol)[0]:
        tail = within_tol[idx:]
        if len(tail) >= min_hold_samples and np.mean(tail) >= hold_fraction:
            settling_time = float(idx * dt)
            settled = True
            break

    # ---- is_stable: a much more lenient "is the response bounded and
    # converging (or already converged), not diverging or growing" check --
    # distinct from `settled` above, which demands the error has already
    # entered AND HELD a tight tolerance band for a meaningful duration. A
    # response can be perfectly well-behaved (steadily decaying error, no
    # sign of divergence) while simply not having crossed that tight
    # threshold yet within the recorded simulation window -- `settled` used
    # to be the only thing shown in the UI's "Stable" column, which
    # regularly looked wrong to someone watching an obviously-converging
    # plot that still said "No". This compares the error's RMS level in an
    # early reference window against a late window: if it hasn't gotten
    # meaningfully worse, the response counts as stable, regardless of
    # whether it's fully settled to within a tight band yet. (Already being
    # `settled` obviously implies stable too.)
    quarter = max(T // 4, 1)
    early_rms = float(np.sqrt(np.mean(err_norm[:quarter] ** 2)))
    late_rms = float(np.sqrt(np.mean(err_norm[-quarter:] ** 2)))
    is_stable = settled or (late_rms <= early_rms * 1.05 + 1e-9)

    # ---- overshoot: standard definition -- how far past the target (i.e. with
    # the error's sign *flipped* relative to where it started) does the
    # response swing, relative to how far away it started. Computed PER
    # STATE, independently, based on whether THAT state's own reference is
    # constant -- not gated by a single global is_regulation flag. A run
    # can perfectly well have some states at a fixed target (e.g. altitude
    # hold) and others tracking a moving reference (e.g. a sinusoidal
    # position) at the same time (see agents/scenario_presets.py /
    # dynamics/base.py's per_state_modes); zeroing overshoot for EVERY
    # state just because ONE of them is tracking was the actual bug behind
    # "Overshoot always shows N/A now". States with a moving reference
    # contribute NaN (genuinely not computed, not 0.0), so the aggregate
    # below only reflects states where the notion applies. ----
    state_is_constant = np.array([bool(np.allclose(reference[:, i], reference[0, i])) for i in range(n_states)])
    per_state_overshoot = np.full(n_states, np.nan)
    for i in range(n_states):
        if not (state_is_constant[i] and moving_mask[i]):
            continue
        e0 = initial_error[i]
        col = error[:, i]
        crossed = col[np.sign(col) == -np.sign(e0)]
        per_state_overshoot[i] = float(np.max(np.abs(crossed))) / abs(e0) if crossed.size > 0 else 0.0

    overshoot_meaningful = bool(np.any(~np.isnan(per_state_overshoot)))
    overshoot = float(np.nanmax(per_state_overshoot)) if overshoot_meaningful else 0.0

    # ---- oscillation: zero-crossings of the state that started furthest off
    # target (the one most likely to reveal ringing/underdamped behavior).
    # Well-defined for any reference type (constant or moving), so this
    # doesn't need to be gated by is_regulation either. ----
    oscillation_count = 0
    if np.any(moving_mask):
        dominant = int(np.argmax(abs_initial))
        signs = np.sign(error[:, dominant])
        signs = signs[signs != 0]
        if signs.size > 1:
            oscillation_count = int(np.sum(np.diff(signs) != 0))

    return PerformanceMetrics(
        mse=mse, rmse=rmse, overshoot=overshoot, settling_time=settling_time, settled=settled,
        is_stable=is_stable, control_effort=control_effort, oscillation_count=oscillation_count,
        per_state_mse=per_state_mse,
        per_state_overshoot=[(None if np.isnan(v) else v) for v in per_state_overshoot],
        integral_abs_error=integral_abs_error, integral_sq_error=integral_sq_error,
        per_state_iae=per_state_iae, per_state_ise=per_state_ise, is_regulation=is_regulation,
        overshoot_meaningful=overshoot_meaningful,
    )


def scalar_cost(metrics: PerformanceMetrics, weights: Optional[dict] = None) -> float:
    """Single scalar objective for the numeric tuner AND for "best result"
    ranking in the UI (lower is better). The Actor/Critic LLM loop can reason
    over the individual metrics, but a numeric optimizer -- and "which
    iteration was actually best" -- needs one number.

    Deliberately does not rank purely on MSE: a response that dips to a low
    instantaneous error while still oscillating, or that never actually
    settles, is penalized here even if its raw MSE looks good, so "best"
    reflects genuine convergence quality, not a lucky low-error sample.

    For a tracking run (``metrics.is_regulation is False``), the
    overshoot/settling_time/oscillation terms are dropped from the formula
    entirely rather than contributing their (meaningless, always-zero)
    values -- otherwise a tracking run would look artificially cheap, as if
    it had zero overshoot by genuine merit rather than because that metric
    simply isn't computed for a moving reference.
    """
    w = weights or OPTIMIZATION_FOCUS_PRESETS["balanced"]

    if not metrics.is_regulation:
        return (
            w.get("mse", 1.0) * metrics.mse
            + w.get("control_effort", 0.0) * metrics.control_effort
            + 0.1 * metrics.integral_sq_error
        )

    cost = (
        w.get("mse", 1.0) * metrics.mse
        + w.get("overshoot", 0.0) * metrics.overshoot
        + w.get("settling_time", 0.0) * metrics.settling_time
        + w.get("control_effort", 0.0) * metrics.control_effort
        + w.get("oscillation", 0.0) * metrics.oscillation_count
    )
    if not metrics.settled:
        cost += w.get("unsettled_penalty", 1.0)
    return cost


# Named weight profiles for scalar_cost(). Used by:
#   * app.py's "Optimization Focus" selector, which threads the chosen
#     profile through graph state (cost_weights) into evaluator_node, so
#     the SAME weights drive both the engine's own best-so-far tracking
#     (used in Critic/Actor prompts) and the UI's "Best Result" tab -- not
#     just a display-side re-ranking that would disagree with what the
#     agents themselves think is "best".
#   * agents/numeric_tuner.py, optionally, via make_objective(cost_weights=...).
#
# "balanced" (the default used when nothing is selected/provided) matches
# the original scalar_cost() weights.
OPTIMIZATION_FOCUS_PRESETS: dict = {
    "balanced": {
        "mse": 1.0, "overshoot": 2.0, "settling_time": 0.5, "control_effort": 0.05,
        "oscillation": 0.05, "unsettled_penalty": 1.0,
    },
    "mse": {
        "mse": 1.0, "overshoot": 0.1, "settling_time": 0.05, "control_effort": 0.01,
        "oscillation": 0.02, "unsettled_penalty": 0.5,
    },
    "overshoot": {
        "mse": 0.3, "overshoot": 5.0, "settling_time": 0.1, "control_effort": 0.02,
        "oscillation": 0.1, "unsettled_penalty": 0.5,
    },
    "settling_time": {
        "mse": 0.3, "overshoot": 0.5, "settling_time": 3.0, "control_effort": 0.02,
        "oscillation": 0.05, "unsettled_penalty": 1.5,
    },
    "control_effort": {
        "mse": 0.3, "overshoot": 0.5, "settling_time": 0.1, "control_effort": 2.0,
        "oscillation": 0.02, "unsettled_penalty": 0.3,
    },
}

OPTIMIZATION_FOCUS_LABELS: dict = {
    "balanced": "Balanced (default) -- all metrics reasonably good",
    "mse": "Minimize MSE",
    "overshoot": "Minimize Overshoot",
    "settling_time": "Minimize Settling Time",
    "control_effort": "Minimize Control Effort",
}
