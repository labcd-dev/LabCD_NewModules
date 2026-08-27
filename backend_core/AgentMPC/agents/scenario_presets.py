"""
================================================================================
agents/scenario_presets.py
================================================================================
Deterministic scenario presets for the Streamlit UI's "Scenario Level"
selector (Level 1/2/3 = Nominal/Noise/Robust). This mirrors what the old UI
exposed, but is *not* the same thing as the LLM-driven Scenarist node
(agents/scenarist.py): the UI lets the user pick the scenario explicitly and
deterministically, rather than having an LLM design one, so
``build_ui_tuning_graph`` (graph/workflow.py) skips the Scenarist node
entirely and calls ``apply_scenario_level`` once before the run instead.

Level definitions:
    1 (Nominal): use the plugin's own default initial state, no noise.
    2 (Noise):    same initial state, but with additive Gaussian measurement
                  noise on the realized state at every step. The noise
                  magnitude and WHICH states receive it are now editable
                  from the UI (see the noise_std_value / noise_state_mask
                  parameters below) rather than a fixed, invisible default.
    3 (Robust):   initial state pushed out toward the edge of the plugin's
                  declared state bounds (or, if it declares none, scaled
                  further from the target than the default), plus moderate
                  noise -- a harder starting point to stabilize from. The
                  push aggressiveness, WHICH states get pushed, and the
                  noise settings are all now editable too.

Both levels' noise is applied as a genuinely PER-STATE array (cfg.data.noise_std
is now an array, not a single scalar) -- numpy's rng.normal(scale=...) already
broadcasts an array scale correctly (a state with scale=0 gets exactly zero
noise), so agents/evaluator.py and dynamics/base.py needed no changes at all
to support this; only what gets stored in cfg.data.noise_std changed.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..dynamics.base import BaseDynamics
from ..mpc.config import Config

SCENARIO_LEVEL_NAMES: Dict[int, str] = {1: "Nominal", 2: "Noise", 3: "Robust"}


def _noise_scale(dynamics: BaseDynamics) -> float:
    """The default per-state noise standard deviation (~1% of that state's
    own declared bound span, or a flat 0.01 if no bounds are declared) --
    used as the UI's suggested starting value, always overridable."""
    bounds = dynamics.get_state_bounds()
    if bounds is not None:
        lo, hi = bounds
        span = hi - lo
        span = np.where(np.isfinite(span), span, 1.0)
        return float(np.mean(span) * 0.01)
    return 0.01


def _default_nudge_magnitude(dynamics: BaseDynamics) -> float:
    """Same state-bound-derived scale as _noise_scale, but ~20x larger --
    meant to be a genuinely testable initial displacement (not a tiny
    sensor-noise-like perturbation). Used only for Level 1's degenerate
    "default_initial_state already equals default_target" case below."""
    return _noise_scale(dynamics) * 20.0


def suggested_noise_std(dynamics: BaseDynamics) -> float:
    """Public wrapper around _noise_scale, for the UI to show as a starting
    point/default value for its editable noise controls (see app.py)."""
    return _noise_scale(dynamics)


def perturb_physical_parameters(
    dynamics: BaseDynamics, max_boost_fraction: float = 0.2, rng: Optional[np.random.Generator] = None,
) -> Dict[str, Tuple[float, float]]:
    """For Level 3 (Robust): selects SOME of the plugin's own declared
    physical parameters (``dynamics.params`` -- mass, length, damping,
    friction, gravity, etc; every plugin in this codebase declares its
    parameters this same way, via ``SystemConfig(params={...})``) and
    increases EACH selected one by its own RANDOM fraction between 0% and
    ``max_boost_fraction`` (e.g. one parameter might end up +5%, another
    +15%, another +19% -- not a uniform +20% across the board), mutating
    ``dynamics.params`` in place. This is genuine PARAMETRIC UNCERTAINTY /
    plant-model mismatch -- a harder and more realistic robustness test
    than only perturbing the initial condition, since it means the
    parameters tuned during this run are being validated against a system
    that's subtly different from the nominal model, the same way a real
    physical plant always differs somewhat from its textbook parameters
    (and rarely by the exact same fraction on every single parameter).

    Parameter SELECTION is still deterministic (every OTHER parameter, in
    sorted-name order -- "some, not all" per the original design intent,
    without needing domain knowledge about which parameters "matter most").
    Only the PER-PARAMETER MAGNITUDE is now randomized. ``rng`` should
    normally be seeded from the run's own ``cfg.random_seed`` (see
    apply_scenario_level below) so the perturbation is reproducible given
    the same seed, rather than genuinely unseeded randomness that would
    make a run impossible to reproduce even with everything else fixed.

    Returns {param_name: (old_value, new_value)} for exactly the parameters
    that were changed -- empty if the plugin declares no numeric params.
    """
    if not dynamics.params:
        return {}
    numeric_keys = sorted(
        k for k, v in dynamics.params.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    )
    if not numeric_keys:
        return {}

    rng = rng if rng is not None else np.random.default_rng()
    chosen = numeric_keys[::2]  # every other one, deterministic "some, not all"
    changed: Dict[str, Tuple[float, float]] = {}
    for key in chosen:
        old_value = dynamics.params[key]
        boost = float(rng.uniform(0.0, max_boost_fraction))
        new_value = old_value * (1.0 + boost)
        dynamics.params[key] = new_value
        changed[key] = (old_value, new_value)
    return changed


def apply_scenario_level(
    dynamics: BaseDynamics,
    cfg: Config,
    level: int,
    noise_std_value: Optional[float] = None,
    noise_state_mask: Optional[np.ndarray] = None,
    robust_push_scale: Optional[float] = None,
    robust_state_mask: Optional[np.ndarray] = None,
    robust_noise_fraction: Optional[float] = None,
    perturb_physical_params: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[float, float]]]:
    """Mutates ``dynamics.config.default_initial_state`` and
    ``cfg.data.noise_std`` (now a per-state array) in place for the given
    level, and returns the (initial_state, target_state) pair that will be
    used.

    All the ``Optional`` parameters default to reproducing the exact
    previous fixed behavior when left as ``None`` -- they only change
    anything when the UI explicitly passes a user-edited value:

    noise_std_value:     the per-state noise standard deviation to use
                          wherever ``noise_state_mask`` is True (Levels 2
                          and 3). Defaults to ``_noise_scale(dynamics)``.
    noise_state_mask:    boolean array, one per state -- True means that
                          state receives noise, False means it stays exact.
                          Defaults to all-True (every state gets noise, the
                          original behavior).
    robust_push_scale:   Level 3 only. Scales how far the initial state is
                          pushed from the target, relative to the original
                          fixed aggressiveness (1.0 = exactly the original
                          default; 0.5 = half as aggressive; 2.0 = twice as
                          aggressive). Defaults to 1.0.
    robust_state_mask:   Level 3 only. boolean array, one per state -- True
                          means that state participates in the push toward
                          a harder starting point; False means it's left at
                          its Level-1 (nominal) value instead. Defaults to
                          all-True (every state pushed, the original
                          behavior).
    robust_noise_fraction: Level 3 only. Level 3's noise is
                          ``noise_std_value * robust_noise_fraction`` --
                          defaults to 0.5 (half of Level 2's magnitude),
                          matching the original fixed behavior.
    perturb_physical_params: Level 3 only. Whether to also boost some of
                          the plugin's own physical parameters (mass,
                          length, damping, etc -- see
                          perturb_physical_parameters above) by 20%,
                          simulating plant-model mismatch. Defaults to True
                          -- this is now the core of what makes Level 3
                          "Robust" rather than just a harder Level 1.

    This is called once by the UI before building/running the graph -- not
    per-iteration -- so the same scenario is used consistently across the
    whole tuning run.

    Returns (initial_state, target_state, perturbed_params) where
    perturbed_params is {name: (old_value, new_value)} for whichever
    physical parameters were changed (always empty outside Level 3, or
    when perturb_physical_params=False).
    """
    if level not in SCENARIO_LEVEL_NAMES:
        raise ValueError(f"Unknown scenario level: {level!r}. Expected one of {list(SCENARIO_LEVEL_NAMES)}.")

    base_x0 = dynamics.config.default_initial_state.copy()
    target = dynamics.config.default_target.copy()
    n_states = len(base_x0)

    if level == 1:  # Nominal
        cfg.data.noise_std = np.zeros(n_states)
        if np.allclose(base_x0, target):
            # The plugin didn't declare a meaningful default displacement
            # (common convention: default_initial_state == default_target,
            # i.e. "start at equilibrium"). Left as-is, EVERY state would
            # have exactly zero initial error, and step-response metrics
            # like Overshoot have nothing to compute -- not a bug, but a
            # confusing default for a "Nominal" test run. Nudge each state
            # by a small, deterministic, alternating-sign amount (scaled to
            # that state's own declared bounds when available) so Level 1
            # is a genuinely exercised scenario by default. A plugin that
            # DOES declare its own distinct default_initial_state is left
            # completely untouched -- this only fires for the degenerate
            # "identical to target" case.
            nudge = _default_nudge_magnitude(dynamics)
            pattern = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_states)])
            new_x0 = target + pattern * nudge
        else:
            new_x0 = base_x0

    elif level == 2:  # Noise
        base_noise = noise_std_value if noise_std_value is not None else _noise_scale(dynamics)
        mask = noise_state_mask if noise_state_mask is not None else np.ones(n_states, dtype=bool)
        cfg.data.noise_std = np.where(mask, base_noise, 0.0)
        if np.allclose(base_x0, target):
            # Same degenerate case Level 1 already guards against (see its
            # comment above) -- a plugin whose default_initial_state equals
            # default_target (even when that target is a nonzero setpoint,
            # e.g. "start already at the operating altitude") would
            # otherwise have exactly zero initial error at this level too,
            # silently making Overshoot show N/A with no way to tell why.
            nudge = _default_nudge_magnitude(dynamics)
            pattern = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_states)])
            new_x0 = target + pattern * nudge
        else:
            new_x0 = base_x0

    else:  # level == 3, Robust
        base_noise = noise_std_value if noise_std_value is not None else _noise_scale(dynamics)
        noise_frac = robust_noise_fraction if robust_noise_fraction is not None else 0.5
        n_mask = noise_state_mask if noise_state_mask is not None else np.ones(n_states, dtype=bool)
        cfg.data.noise_std = np.where(n_mask, base_noise * noise_frac, 0.0)

        push_scale = robust_push_scale if robust_push_scale is not None else 1.0
        p_mask = robust_state_mask if robust_state_mask is not None else np.ones(n_states, dtype=bool)

        direction = base_x0 - target
        if np.allclose(direction, 0):
            direction = np.ones(n_states)
        direction = direction / (np.linalg.norm(direction) + 1e-9)

        bounds = dynamics.get_state_bounds()
        if bounds is not None:
            lo, hi = bounds
            half_span = (hi - lo) / 2
            half_span = np.where(np.isfinite(half_span), half_span, np.abs(target) + 1.0)
            pushed_x0 = target + direction * 0.7 * push_scale * half_span
        else:
            pushed_x0 = target + (1.0 + push_scale) * (base_x0 - target)

        # states NOT selected for the push keep their normal Level-1 value
        # instead of the aggressive one.
        new_x0 = np.where(p_mask, pushed_x0, base_x0)

        perturbed_params = (
            perturb_physical_parameters(dynamics, rng=np.random.default_rng(cfg.random_seed))
            if perturb_physical_params else {}
        )

    dynamics.config.default_initial_state = new_x0
    return new_x0, target, perturbed_params if level == 3 else {}
