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

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..dynamics.base import BaseDynamics
from ..mpc.config import Config

SCENARIO_LEVEL_NAMES: Dict[int, str] = {
    1: "Nominal", 2: "Noise", 3: "Robust", 4: "Fault & Disturbance",
}

# ---------------------------------------------------------------------------
# The magnitudes that define what each scenario level MEANS.
#
# These were literals buried at their use sites (a 0.01 here, a * 20.0 there,
# a 0.5 in one branch and a 0.2 in another), which made the disturbance model
# impossible to read off or to state in a report: you had to go find each
# number to know how hard "Robust" actually was. They are defaults, not
# policy -- every one is overridable per run from the UI -- but a default
# that decides how severe a robustness test is deserves a name and a reason.
# ---------------------------------------------------------------------------

#: Default per-state sensor-noise sigma, as a fraction of that state's own
#: declared bound span. 1% of full scale is the order of magnitude of a decent
#: industrial sensor, and scaling to the span keeps it meaningful whether the
#: state is radians or RPM.
NOISE_FRACTION_OF_STATE_SPAN: float = 0.01

#: Used instead when the plugin declares no state bounds, so there is no span
#: to take a fraction of. Absolute, in the state's own units.
NOISE_STD_WITHOUT_BOUNDS: float = 0.01

#: Level 4 default: how much of what it is asked for a faulted actuator
#: actually delivers. 0.7 is a degraded-but-alive actuator -- enough to matter,
#: not so little that every controller fails regardless of tuning.
DEFAULT_ACTUATOR_EFFECTIVENESS: float = 0.7

#: Level 4 default: the constant external disturbance on the state derivative,
#: as a fraction of each state's own scale per second (its bound span, else
#: its initial gap or target, else UNBOUNDED_STATE_SCALE -- see
#: _state_span_scale). Expressed
#: relative to the state's own scale so it means the same thing on a plant
#: measured in radians as on one measured in RPM.
DEFAULT_DISTURBANCE_FRACTION: float = 0.05

#: How much larger the Level-1 initial-state nudge is than the noise scale.
#: The nudge has to be a genuinely testable displacement rather than a
#: sensor-noise-sized one, or the metrics it exists to make computable
#: (overshoot, settling) are measuring rounding.
NUDGE_MULTIPLE_OF_NOISE: float = 20.0

#: Level 3 runs LESS sensor noise than Level 2, deliberately: its difficulty
#: comes from plant-model mismatch, and stacking full noise on top would
#: confound the two effects rather than test the harder one.
ROBUST_NOISE_FRACTION: float = 0.5

#: Bound of the per-parameter change drawn for Level 3. Each selected
#: parameter gets its own draw in [-this, +this], so 20% is the worst case for
#: any single parameter, not a uniform shift of all of them.
MAX_PARAM_UNCERTAINTY: float = 0.2

#: Parameter names that are constants of nature, not uncertain properties of
#: the plant, so Level 3 never perturbs them. The quadcopter campaign's Level 3
#: raised gravity by 19.5% (bug 10). Matched case-insensitively.
PHYSICAL_CONSTANT_NAMES = frozenset({"g", "g0", "grav", "gravity", "pi", "mu0", "eps0", "epsilon0"})

#: A state with neither declared bounds nor an initial gap/target to measure
#: its size by is given this scale (in its own units) for noise and
#: disturbances.
UNBOUNDED_STATE_SCALE: float = 1.0

#: Stride for "some, but not all" parameter selection: every Nth parameter in
#: sorted-name order. Deterministic on purpose -- which parameters are
#: mismatched stays fixed across a run so results remain comparable, while
#: only the magnitudes are random.
PARAM_SELECTION_STRIDE: int = 2


def _noise_scale(dynamics: BaseDynamics) -> float:
    """The default per-state noise standard deviation (~1% of that state's
    own declared bound span, or a flat 0.01 if no bounds are declared) --
    used as the UI's suggested starting value, always overridable."""
    bounds = dynamics.get_state_bounds()
    if bounds is not None:
        lo, hi = bounds
        span = hi - lo
        span = np.where(np.isfinite(span), span, 1.0)
        return float(np.mean(span) * NOISE_FRACTION_OF_STATE_SPAN)
    return NOISE_STD_WITHOUT_BOUNDS


def _default_nudge_magnitude(dynamics: BaseDynamics) -> float:
    """Same state-bound-derived scale as _noise_scale, but ~20x larger --
    meant to be a genuinely testable initial displacement (not a tiny
    sensor-noise-like perturbation). Used only for Level 1's degenerate
    "default_initial_state already equals default_target" case below."""
    return _noise_scale(dynamics) * NUDGE_MULTIPLE_OF_NOISE


def _state_span_scale(dynamics: BaseDynamics) -> np.ndarray:
    """A per-state magnitude to express a disturbance (or default noise)
    against, so one "strength" number means the same thing on a state in
    radians and one in RPM.

    In order of preference: the state's declared bound span; else how far it
    starts from its target, or the target's own size; else
    UNBOUNDED_STATE_SCALE. The last fallback used to be NOISE_STD_WITHOUT_BOUNDS
    (0.01) -- a noise sigma standing in for a span -- which made Level 4's
    default disturbance 0.0005 per second on every state of a plant without
    declared bounds: no disturbance at all (quadcopter campaign, bug 10).
    """
    n = dynamics.n_states
    scale = np.zeros(n)
    bounds = dynamics.get_state_bounds()
    if bounds is not None:
        span = np.abs(np.asarray(bounds[1], dtype=float) - np.asarray(bounds[0], dtype=float))
        scale = np.where(np.isfinite(span) & (span > 0), span, 0.0)
    x0 = np.asarray(dynamics.config.default_initial_state, dtype=float)
    target = np.asarray(dynamics.config.default_target, dtype=float)
    own = np.maximum(np.abs(x0 - target), np.abs(target))
    scale = np.where(scale > 0, scale, own)
    return np.where(scale > 1e-12, scale, UNBOUNDED_STATE_SCALE)


def _default_noise(dynamics: BaseDynamics) -> np.ndarray:
    """Per-state noise sigma when the caller gives none: 1% of each state's
    own scale (see _state_span_scale), not one absolute number shared by
    metres, m/s and radians. The UI passes its own single, labelled value."""
    return NOISE_FRACTION_OF_STATE_SPAN * _state_span_scale(dynamics)


def suggested_noise_std(dynamics: BaseDynamics) -> float:
    """Public wrapper around _noise_scale, for the UI to show as a starting
    point/default value for its editable noise controls (see app.py)."""
    return _noise_scale(dynamics)


def nudge_if_starts_at_target(dynamics: BaseDynamics, x0: np.ndarray, target: np.ndarray) -> np.ndarray:
    """If ``x0`` equals ``target``, nudge each state by a small, deterministic,
    alternating-sign amount (scaled to that state's own declared bounds when
    available) so there is a genuine initial error to measure a step response
    from. Returns ``x0`` completely untouched otherwise.

    ``default_initial_state == default_target`` (i.e. "start already at the
    operating point") is a common, reasonable plugin convention -- but left
    as-is, EVERY state would have exactly zero initial error, and
    step-response metrics like Overshoot have nothing to compute. Public (not
    just inlined in apply_scenario_level below) because Manual Simulation
    hits this exact same degenerate case -- most often with a plugin whose
    equilibrium/target is a nonzero setpoint (e.g. "hold this RPM"), where
    "start at equilibrium" is an even more natural default than for a
    zero-target plant -- but has no scenario-level machinery of its own to
    route through.
    """
    if not np.allclose(x0, target):
        return x0
    nudge = _default_nudge_magnitude(dynamics)
    pattern = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(len(x0))])
    return target + pattern * nudge


def perturbable_parameter_names(dynamics: BaseDynamics) -> List[str]:
    """Exactly which parameters Level 3 perturbs -- public so the UI can name
    them before a run from the same rule rather than a copy of it.

    Every other numeric parameter in sorted-name order, among the genuinely
    uncertain ones: not constants of nature (gravity -- the quadcopter
    campaign's Level 3 raised g by 19.5%), and not parameters that are exactly
    zero (a product of inertia of 0 stays 0 when scaled, so selecting it just
    wastes the slot).
    """
    params = getattr(dynamics, "params", None) or {}
    numeric = sorted(
        k for k, v in params.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
        and k.lower() not in PHYSICAL_CONSTANT_NAMES and v != 0
    )
    return numeric[::PARAM_SELECTION_STRIDE]


def perturb_physical_parameters(
    dynamics: BaseDynamics, max_boost_fraction: float = MAX_PARAM_UNCERTAINTY,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Tuple[float, float]]:
    """For Level 3 (Robust): selects SOME of the plugin's own declared
    physical parameters (``dynamics.params`` -- mass, length, damping,
    friction, etc; never constants of nature such as gravity, see
    perturbable_parameter_names) and moves EACH selected one by its own
    RANDOM fraction between -``max_boost_fraction`` and +``max_boost_fraction``
    (e.g. one parameter might end up +5%, another -15% -- not a uniform shift
    across the board, and not only upward), mutating
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
    chosen = perturbable_parameter_names(dynamics)
    if not chosen:
        return {}
    rng = rng if rng is not None else np.random.default_rng()
    changed: Dict[str, Tuple[float, float]] = {}
    for key in chosen:
        old_value = dynamics.params[key]
        # Either way: a real plant is as likely to be lighter or less damped
        # than its datasheet as heavier.
        boost = float(rng.uniform(-max_boost_fraction, max_boost_fraction))
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
    robust_noise_fraction: Optional[float] = None,
    perturb_physical_params: bool = True,
    max_param_uncertainty: Optional[float] = None,
    actuator_effectiveness: Optional[Any] = None,
    disturbance_fraction: Optional[float] = None,
    state_disturbance: Optional[Any] = None,
    input_disturbance: Optional[Any] = None,
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
    robust_noise_fraction: Level 3 only. Level 3's noise is
                          ``noise_std_value * robust_noise_fraction`` --
                          defaults to 0.5 (half of Level 2's magnitude).
    perturb_physical_params: Level 3 only. Whether to boost some of the
                          plugin's own physical parameters (mass, length,
                          damping, etc -- see perturb_physical_parameters
                          above), simulating plant-model mismatch. Defaults
                          to True -- this is what makes Level 3 "Robust".
    max_param_uncertainty: Level 3 only. The upper bound of the per-parameter
                          random boost, as a FRACTION (0.2 = "up to +20%").
                          Each selected parameter draws its own value in
                          [0, this], so they don't all move by the same
                          amount. Defaults to 0.2.

    Level 3 no longer pushes the initial state toward its bounds: what makes
    it "Robust" is the parametric mismatch above (tuning against a plant
    whose parameters aren't what the model assumes), not a harder starting
    point, which only made it a noisier Level 1 with a different initial
    condition and muddied comparisons between levels. All three levels now
    use the same initial state -- the plugin's own, nudged only in the
    degenerate "already at the target" case.

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

    # Freeze what the controller is allowed to believe, BEFORE anything below
    # perturbs the plant. Everything after this point changes reality; the MPC
    # goes on predicting with this copy and is told nothing, which is what
    # makes a robustness result mean anything at all. Taken for every level so
    # the two objects are always distinct -- at Level 1 and 2 nothing is
    # perturbed, so the copy is identical and behaviour is unchanged.
    cfg.data.nominal_dynamics = copy.deepcopy(dynamics)

    # Faults and disturbances are Level 4 only; clear them so a level change
    # within one session cannot leave a previous run's fault in place.
    cfg.data.actuator_effectiveness = None
    cfg.data.input_disturbance = None
    cfg.data.state_disturbance = None

    base_x0 = dynamics.config.default_initial_state.copy()
    target = dynamics.config.default_target.copy()
    n_states = len(base_x0)

    if level == 1:  # Nominal
        cfg.data.noise_std = np.zeros(n_states)
        # See nudge_if_starts_at_target's docstring for why this is needed:
        # a plugin that DOES declare its own distinct default_initial_state
        # is left completely untouched -- this only fires for the degenerate
        # "identical to target" case, so Level 1 is a genuinely exercised
        # scenario by default.
        new_x0 = nudge_if_starts_at_target(dynamics, base_x0, target)

    elif level == 2:  # Noise
        base_noise = noise_std_value if noise_std_value is not None else _default_noise(dynamics)
        mask = noise_state_mask if noise_state_mask is not None else np.ones(n_states, dtype=bool)
        cfg.data.noise_std = np.where(mask, base_noise, 0.0)
        # Same degenerate case Level 1 already guards against.
        new_x0 = nudge_if_starts_at_target(dynamics, base_x0, target)

    elif level == 4:  # Fault & Disturbance
        # Everything Level 3 does to the plant, plus two things the controller
        # cannot see at all: a degraded actuator and an external disturbance.
        base_noise = noise_std_value if noise_std_value is not None else _default_noise(dynamics)
        noise_frac = robust_noise_fraction if robust_noise_fraction is not None else ROBUST_NOISE_FRACTION
        n_mask = noise_state_mask if noise_state_mask is not None else np.ones(n_states, dtype=bool)
        cfg.data.noise_std = np.where(n_mask, base_noise * noise_frac, 0.0)
        new_x0 = nudge_if_starts_at_target(dynamics, base_x0, target)

        perturbed_params = (
            perturb_physical_parameters(
                dynamics,
                max_boost_fraction=(max_param_uncertainty if max_param_uncertainty is not None
                                     else MAX_PARAM_UNCERTAINTY),
                rng=np.random.default_rng(cfg.random_seed),
            )
            if perturb_physical_params else {}
        )

        # Actuator effectiveness: the plant receives lambda * u. The MPC still
        # computes u against a healthy model, so it must recover the shortfall
        # through feedback alone -- it never learns that the actuator is weak.
        n_inputs = dynamics.n_inputs
        if actuator_effectiveness is None:
            lam = np.full(n_inputs, DEFAULT_ACTUATOR_EFFECTIVENESS)
        else:
            lam = np.broadcast_to(
                np.asarray(actuator_effectiveness, dtype=float).ravel(), (n_inputs,)).copy()
        cfg.data.actuator_effectiveness = np.clip(lam, 0.0, 1.0)

        # External disturbance on the state derivative -- a force, load or gust
        # acting on the plant itself, not an input anyone chose. Scaled to each
        # state's own magnitude so one number means the same thing across
        # plants with different units.
        if state_disturbance is None:
            scale = _state_span_scale(dynamics)
            cfg.data.state_disturbance = scale * (
                disturbance_fraction if disturbance_fraction is not None
                else DEFAULT_DISTURBANCE_FRACTION)
        else:
            cfg.data.state_disturbance = np.broadcast_to(
                np.asarray(state_disturbance, dtype=float).ravel(), (n_states,)).copy()

        if input_disturbance is not None:
            cfg.data.input_disturbance = np.broadcast_to(
                np.asarray(input_disturbance, dtype=float).ravel(), (dynamics.n_inputs,)).copy()

    else:  # level == 3, Robust
        base_noise = noise_std_value if noise_std_value is not None else _default_noise(dynamics)
        noise_frac = robust_noise_fraction if robust_noise_fraction is not None else ROBUST_NOISE_FRACTION
        n_mask = noise_state_mask if noise_state_mask is not None else np.ones(n_states, dtype=bool)
        cfg.data.noise_std = np.where(n_mask, base_noise * noise_frac, 0.0)

        # Same initial state as every other level (see the docstring: the
        # harder starting point is gone; parametric mismatch below is what
        # makes this level Robust).
        new_x0 = nudge_if_starts_at_target(dynamics, base_x0, target)

        perturbed_params = (
            perturb_physical_parameters(
                dynamics,
                max_boost_fraction=(max_param_uncertainty if max_param_uncertainty is not None
                                     else MAX_PARAM_UNCERTAINTY),
                rng=np.random.default_rng(cfg.random_seed),
            )
            if perturb_physical_params else {}
        )

    dynamics.config.default_initial_state = new_x0
    return new_x0, target, perturbed_params if level in (3, 4) else {}
