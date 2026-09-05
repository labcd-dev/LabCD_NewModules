"""
================================================================================
agents/dynamics_validator.py
================================================================================
A "Dynamics File Validator" agent: checks an uploaded dynamics plugin file
against the package's standard (see DYNAMICS_STANDARD below) and, if it
doesn't conform, asks the LLM to repair it -- then re-checks the repair
deterministically before trusting it.

Design principle: the LLM is only ever used to *propose* a fix. Whether a
file is valid, and whether a proposed fix actually works, is always decided
by the exact same deterministic mechanism the rest of the package already
uses for this (``DynamicLoader.load_from_path`` -- see dynamics/loader.py).
The LLM cannot talk its way into a "valid" verdict; a fix is only accepted
if it *actually* loads successfully afterward. This mirrors the same
principle as the rest of AgentMPC: the LLM makes proposals, deterministic
code verifies them (see mpc/controller.py's QP solve, or agents/evaluator.py's
closed-loop simulation -- the LLM never gets to just *assert* a result).

Flow:
    1. Try to load the file exactly as uploaded (fast, free, no LLM call).
       If it already conforms, we're done -- most files should hit this path.
    2. If it fails, send the standard + the exact error + the original source
       to the LLM, asking for a complete corrected file.
    3. Re-validate the LLM's fix the same deterministic way. If it now loads,
       accept it (and the caller -- app.py -- offers it for download so the
       user can just use the corrected file directly next time, skipping
       this whole step).
    4. If it still doesn't load, retry once more with the new error message,
       up to `max_attempts`. If it never succeeds, report clearly rather than
       silently proceeding with a broken file.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..dynamics.loader import DynamicLoader, DynamicsPluginError
from ..utils.logging_utils import get_logger
from .prompt_library import get_prompt

log = get_logger(__name__)


# Re-exported at module level (not just used as a prompt): the Streamlit UI
# imports this name and renders it as the user-facing plugin contract.
DYNAMICS_STANDARD = get_prompt("dynamics_validator", "standard")


@dataclass
class ValidationOutcome:
    valid: bool
    error: Optional[str] = None
    summary: Optional[dict] = None


@dataclass
class FixOutcome:
    valid: bool
    used_llm_fix: bool
    final_code: str
    original_error: Optional[str] = None
    explanation: Optional[str] = None
    still_broken_error: Optional[str] = None
    attempts: int = 0
    history: list = field(default_factory=list)   # list of {"attempt": int, "error": str} for failed tries


@dataclass
class SetupResult:
    """Everything the (renamed-in-spirit) "Initial Setup Agent" produces once
    a dynamics file is confirmed valid: not just pass/fail, but the concrete
    starting point for a tuning run -- which states are derivative pairs
    (for physically-consistent sin/cos reference generation), a Bryson's-rule
    initial Q/R guess, and a suggested MPC sample time dt. All three are
    computed deterministically from the dynamics itself (short simulations /
    linearization), never guessed by an LLM -- see each function's docstring
    below for exactly how."""
    fix: FixOutcome
    derivative_pairs: list = field(default_factory=list)      # list of (position_idx, velocity_idx) tuples
    suggested_Q: Optional[list] = None
    suggested_R: Optional[list] = None
    suggested_dt: Optional[float] = None
    suggested_feedforward: Optional[list] = None                # see estimate_feedforward_trim -- opt-in via the
                                                                    # Configure section's feedforward toggle, unlike
                                                                    # dt/Q/R which are used automatically
    setup_notes: list = field(default_factory=list)             # human-readable notes on how each value was derived
    qr_diagnostics: Optional[dict] = None                          # probe trajectory + formula components, for the
                                                                      # setup-agent visualization -- see estimate_initial_qr


def detect_derivative_pairs(dynamics, n_samples: int = 12, rtol: float = 1e-4, atol: float = 1e-6) -> list:
    """Deterministically detect which states are the time-derivative of
    which other state -- e.g. state 1 ("cart_vel") is the derivative of
    state 0 ("cart_pos") -- WITHOUT relying on state names or an LLM guess.

    The test is purely mathematical: for a state-space model, if state j IS
    the velocity whose integral gives position state i, then by definition
    dx_i/dt = x_j identically, for ANY (x, u) -- i.e. dynamics(x, u)[i] ==
    x[j] at every point, not just near equilibrium. This is checked by
    evaluating dynamics() at several random (x, u) samples (within the
    plugin's declared bounds when available, otherwise a modest default
    range) and requiring the identity to hold, within tolerance, at EVERY
    sample -- a coincidental match at a single point is not enough evidence,
    but an identity that holds across a dozen unrelated random points is
    essentially never a coincidence.

    Used so that when the user picks "sin" for a position state (either via
    the global Trajectory Type or the per-state editor), the framework can
    automatically generate the mathematically-consistent "cos" reference for
    its paired velocity state instead of requiring the user to set it by
    hand every time -- see dynamics/base.py:SystemConfig.desired_trajectory's
    ``mode`` path, which already does this via a hardcoded (2i, 2i+1)
    convention; this detection makes that pairing exact (and correct even
    when a plugin's state ordering does NOT follow the (2i, 2i+1) 
    convention) rather than assumed.
    """
    n_states, n_inputs = dynamics.n_states, dynamics.n_inputs
    rng = np.random.default_rng(0)

    x_lo, x_hi = dynamics.get_state_bounds() or (None, None)
    u_lo, u_hi = dynamics.get_input_bounds() or (None, None)

    def sample_x():
        base = dynamics.config.default_initial_state
        if x_lo is not None and np.all(np.isfinite(x_lo)) and np.all(np.isfinite(x_hi)):
            return rng.uniform(x_lo, x_hi)
        return base + rng.uniform(-1.0, 1.0, size=n_states)

    def sample_u():
        base = dynamics.get_equilibrium_input()
        if u_lo is not None and np.all(np.isfinite(u_lo)) and np.all(np.isfinite(u_hi)):
            return rng.uniform(u_lo, u_hi)
        return base + rng.uniform(-1.0, 1.0, size=n_inputs)

    samples = []
    for _ in range(n_samples):
        x, u = sample_x(), sample_u()
        try:
            dx = np.asarray(dynamics.dynamics(x, u), dtype=float)
        except Exception:  # noqa: BLE001
            continue
        if dx.shape != (n_states,) or not np.all(np.isfinite(dx)):
            continue
        samples.append((x, dx))

    if len(samples) < max(3, n_samples // 2):
        return []  # too few usable samples (e.g. a plugin that's finicky far from equilibrium) -- skip silently

    pairs = []
    used_as_velocity = set()
    for i in range(n_states):
        best_j, best_err = None, None
        for j in range(n_states):
            if j == i or j in used_as_velocity:
                continue
            errs = [abs(dx[i] - x[j]) / max(abs(x[j]), 1.0) for x, dx in samples]
            max_err = max(errs)
            if max_err < rtol + atol and (best_err is None or max_err < best_err):
                best_j, best_err = j, max_err
        if best_j is not None:
            pairs.append((i, best_j))
            used_as_velocity.add(best_j)

    return pairs


def estimate_initial_qr(dynamics, step_fraction: float = 0.25, probe_time_horizon: float = 2.0, probe_steps: int = 100):
    """Bryson's rule (a standard LQR/MPC weight-selection heuristic): weight
    each state/input by the inverse SQUARE of its own "characteristic range",
    so that a unit of cost means roughly the same thing (a comparable
    fraction of that variable's natural scale) for every state and input,
    regardless of their different physical units. Concretely:

        Q_ii = 1 / range_i^2          R_jj = 1 / step_j^2

    ``range_i`` is measured directly: apply a modest constant step input
    (a fraction of the input's declared bounds, or of its equilibrium value
    if no bounds are declared) on top of the equilibrium input, integrate
    the TRUE nonlinear dynamics open-loop for a short probe window, and
    measure how far each state actually moved (max-min over the probe). This
    is exactly "look at the system's own step response" rather than
    guessing -- and because it's the real nonlinear dynamics.rk4_step, it
    works the same way regardless of how nonlinear or exotic a given plugin
    is. The probe window is deliberately short so that even an open-loop-
    unstable system (e.g. an inverted pendulum, a quadcopter) gives a
    finite, meaningful range before diverging numerically.

    The probe duration is a fixed ~probe_time_horizon (default 2 simulated
    seconds), independent of whatever dt estimate_dt() suggests -- a
    position-like state (a double integral of force/acceleration) needs a
    much longer window to show a comparable range to a fast rotation-rate-
    like state; tying this probe to a very fine dt (appropriate for
    sampling the fastest mode) would give it far too short a total window
    to move meaningfully, skewing the whole Q vector toward the fast states.

    The raw Bryson values can span many orders of magnitude across
    different systems; the returned Q/R are rescaled together (preserving
    their ratio, which is what actually matters for the optimum) so the
    largest Q entry lands at a friendly reference value the Actor/Critic
    prompts and exploration-intensity multipliers already assume.
    """
    n_states, n_inputs = dynamics.n_states, dynamics.n_inputs
    x0 = dynamics.config.default_initial_state.copy()
    u_eq = dynamics.get_equilibrium_input()
    probe_dt = probe_time_horizon / probe_steps

    u_lo, u_hi = dynamics.get_input_bounds() or (None, None)
    step_mag = np.empty(n_inputs)
    for j in range(n_inputs):
        if u_lo is not None and np.isfinite(u_lo[j]) and np.isfinite(u_hi[j]) and (u_hi[j] - u_lo[j]) > 1e-9:
            step_mag[j] = step_fraction * (u_hi[j] - u_lo[j])
        else:
            step_mag[j] = step_fraction * max(abs(u_eq[j]), 1.0)

    def run_probe(u_probe):
        traj = [x0.copy()]
        x = x0.copy()
        diverged = False
        for _ in range(probe_steps):
            try:
                x = dynamics.rk4_step(x, u_probe, probe_dt)
            except Exception:  # noqa: BLE001
                diverged = True
                break
            if not np.all(np.isfinite(x)):
                diverged = True
                break
            traj.append(x.copy())
            if np.max(np.abs(x)) > 1e8:  # blew up -- stop early, use whatever range was seen before this
                diverged = True
                break
        traj = np.array(traj)
        ranges = np.ptp(traj, axis=0) if len(traj) > 1 else np.zeros(n_states)
        return ranges, diverged, len(traj) - 1, traj

    # A single uniform step (same sign on every input) can fail to excite
    # some states entirely -- e.g. an identical thrust correction on all four
    # rotors of a quadcopter doesn't excite yaw at all, by symmetry. Probe
    # with several randomized per-input sign patterns and take the largest
    # range seen per state across all of them, so a state only ends up with
    # a small range if it's genuinely insensitive to every direction tried,
    # not just one unlucky one.
    rng = np.random.default_rng(0)
    sign_patterns = [np.ones(n_inputs)] + [rng.choice([-1.0, 1.0], size=n_inputs) for _ in range(4)]

    best_ranges = np.zeros(n_states)
    any_diverged = False
    steps_used = probe_steps
    first_probe_traj = None
    for k, signs in enumerate(sign_patterns):
        ranges_trial, diverged, n_steps_trial, traj = run_probe(u_eq + signs * step_mag)
        if k == 0:
            first_probe_traj = traj  # the baseline (all-+1) probe -- used only for the setup-agent visualization
        best_ranges = np.maximum(best_ranges, ranges_trial)
        any_diverged = any_diverged or diverged
        steps_used = min(steps_used, n_steps_trial) if diverged else steps_used

    ranges = np.maximum(best_ranges, 1e-6)
    step_safe = np.maximum(np.abs(step_mag), 1e-6)

    Q_raw = 1.0 / ranges ** 2
    # A state whose open-loop range is enormous (or hit the divergence cap)
    # isn't showing a meaningful "natural operating range" -- for an
    # inherently unstable system (e.g. a quadcopter's attitude with no
    # stabilizing input), it's just unbounded drift. Naively trusting that
    # inflated range gives Bryson's rule exactly the wrong signal (a tiny
    # weight, as if the state barely matters) for precisely the state that
    # most needs to be tightly controlled. Treat any state within ~1 decade
    # of the numerical divergence cap as "diverged" and give it the SAME
    # raw weight as the most range-sensitive well-behaved state instead of
    # trusting the inflated range.
    diverged_mask = ranges >= 1e7
    if np.any(diverged_mask) and not np.all(diverged_mask):
        Q_raw[diverged_mask] = np.max(Q_raw[~diverged_mask])
    elif np.all(diverged_mask):
        Q_raw[:] = 1.0  # every state diverged -- can't distinguish relative importance here, fall back to uniform
    R_raw = 1.0 / step_safe ** 2

    # Bryson's inverse-square rule is only meaningful for comparing
    # magnitudes WITHIN one group (which state most needs a higher Q, which
    # input most needs a higher R) -- both Q_raw and R_raw are internally
    # consistent in that sense. But Q_raw is derived from OBSERVED state
    # ranges in each state's own physical units, while R_raw is derived from
    # each plugin's DECLARED input range, which can live in an entirely
    # different, arbitrarily-normalized numeric scale (e.g. the overactuated
    # quadcopter plugin's inputs are normalized to roughly [-1, 1] purely for
    # QP conditioning -- see its U_SCALE comment -- even though that "1 unit"
    # corresponds to a huge physical thrust change). Scaling Q and R by ONE
    # SHARED factor -- as if their raw magnitudes were directly comparable --
    # let that arbitrary unit mismatch leak into the Q-vs-R balance: a
    # numerically "small" declared input range inflates 1/step^2 for reasons
    # that have nothing to do with how much control effort should actually
    # cost relative to tracking error, which is how a quadcopter ended up
    # with R noticeably LARGER than Q -- backwards from the standard
    # convention (and from what actually controls the system well) that
    # STATE tracking should dominate the cost, with R acting as a secondary
    # regularizer against excessive/jerky control.
    #
    # Fix: scale Q and R independently, each to its own friendly reference
    # magnitude, preserving the WITHIN-group relative weighting Bryson's
    # rule correctly captures, while fixing the ACROSS-group Q-vs-R balance
    # to the conventional expectation instead of an artifact of unit choice.
    Q_TARGET_MAX = 10.0
    R_TARGET_MAX = 1.0   # deliberately a decade below Q_TARGET_MAX -- R is a secondary regularizer, not the primary objective
    Q = (Q_raw * (Q_TARGET_MAX / np.max(Q_raw))).tolist()
    R = (R_raw * (R_TARGET_MAX / np.max(R_raw))).tolist()

    note = (
        f"Bryson's-rule estimate from {len(sign_patterns)} randomized-direction open-loop probes "
        f"({steps_used * probe_dt:.2g}s simulated each, dt={probe_dt:.4g}s, "
        f"input step = {step_fraction:.0%} of {'declared bounds' if u_lo is not None else 'equilibrium input'})"
    )
    if any_diverged:
        note += " -- at least one probe direction diverged partway through (expected for an open-loop-unstable system); estimate uses the largest range observed across all directions before any divergence."

    diagnostics = {
        "trajectory": first_probe_traj,          # (T, n_states) -- the baseline probe, for plotting
        "probe_dt": probe_dt,
        "ranges": ranges.tolist(),                 # the (max-across-directions) range used for Q, per state
        "step_mag": step_mag.tolist(),               # the input step used for R, per input
        "Q_TARGET_MAX": Q_TARGET_MAX,
        "R_TARGET_MAX": R_TARGET_MAX,
    }
    return Q, R, note, diagnostics


def estimate_dt(dynamics) -> float:
    """Suggests an MPC sample time from the system's own fastest natural
    dynamics, NOT a fixed default -- a slow system (e.g. a large thermal or
    chemical process) can use a much coarser dt than a fast one (e.g. a
    quadcopter) without losing anything, and using the same fixed dt for
    both either wastes computation on the slow system or under-samples the
    fast one.

    Combines two independent estimates and takes the more conservative
    (smaller) one:

    1. Linearized eigenvalues: reuse the same linearize() the MPC controller
       itself uses (mpc/jacobian.py), evaluated at (default_initial_state,
       equilibrium_input); take the fastest eigenvalue's magnitude (shortest
       natural time constant) and sample ~15x within it. Cheap and exact
       when it works -- but degenerates for a system that's a pure
       integrator chain at its own equilibrium (drift-only eigenvalues near
       zero even though the system responds quickly once a control input is
       actually applied -- true of e.g. a hovering quadcopter, whose
       position/attitude states have no self-restoring drift at u=0).

    2. Step-response rise time: apply a modest input step on top of
       equilibrium and integrate the TRUE nonlinear dynamics.rk4_step at a
       fine trial resolution; find how long the fastest-moving state takes
       to cover ~10% of its own eventual range, and sample ~8x within that.
       This captures actuator-driven response speed directly, sidestepping
       the drift-eigenvalue degeneracy above. The trial resolution is
       widened geometrically (up to a few attempts) if nothing moves
       meaningfully in the initial window, so a genuinely slow system
       doesn't get mistaken for a fast one just because the first probe
       window was too short to see any motion.

    This is a one-time suggestion, not something the tuning agents
    optimize -- dt is fixed for the duration of a run, same as any other
    simulation setting.
    """
    from ..mpc.jacobian import linearize

    x0 = dynamics.config.default_initial_state.copy()
    u_eq = dynamics.get_equilibrium_input()
    n_states, n_inputs = dynamics.n_states, dynamics.n_inputs

    # ---- estimate 1: linearized eigenvalues ----
    eig_dt = None
    try:
        A, _ = linearize(dynamics.dynamics, x0, u_eq, torch_dynamics_fn=getattr(dynamics, "dynamics_torch", None))
        fastest = float(np.max(np.abs(np.linalg.eigvals(A))))
        if fastest > 1e-4:  # treat anything slower than this as "no reliable fast mode" rather than trust a huge dt
            eig_dt = 1.0 / fastest / 15.0
    except Exception:  # noqa: BLE001
        pass

    # ---- estimate 2: step-response rise time ----
    u_lo, u_hi = dynamics.get_input_bounds() or (None, None)
    step = np.empty(n_inputs)
    for j in range(n_inputs):
        if u_lo is not None and np.isfinite(u_lo[j]) and np.isfinite(u_hi[j]) and (u_hi[j] - u_lo[j]) > 1e-9:
            step[j] = 0.25 * (u_hi[j] - u_lo[j])
        else:
            step[j] = 0.25 * max(abs(u_eq[j]), 1.0)
    u_probe = u_eq + step

    rise_dt = None
    for trial_dt in (0.001, 0.02, 0.4):
        x = x0.copy()
        traj = [x0.copy()]
        ok = True
        for _ in range(300):
            try:
                x = dynamics.rk4_step(x, u_probe, trial_dt)
            except Exception:  # noqa: BLE001
                ok = False
                break
            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e8:
                break
            traj.append(x.copy())
        if not ok or len(traj) < 5:
            continue
        traj = np.array(traj)
        ranges = np.ptp(traj, axis=0)
        rise_times = []
        for i in range(n_states):
            target = 0.1 * ranges[i]
            if target < 1e-9:
                continue
            deviation = np.abs(traj[:, i] - traj[0, i])
            hit = np.where(deviation >= target)[0]
            if hit.size > 0 and hit[0] > 0:
                rise_times.append(hit[0] * trial_dt)
        if rise_times:
            rise_dt = min(rise_times) / 8.0
            break  # found meaningful motion at this trial resolution -- no need to widen further

    candidates = [d for d in (eig_dt, rise_dt) if d is not None and d > 0]
    dt = min(candidates) if candidates else 0.05  # neither signal available -- sane fallback
    return float(np.clip(dt, 0.001, 0.5))


def validate_dynamics_source(source_code: str) -> ValidationOutcome:
    """Deterministic check: does this source code conform to the standard?
    No LLM involved -- this is the same mechanism DynamicLoader always uses."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(source_code)
            temp_path = f.name
        plugin = DynamicLoader.load_from_path(temp_path)
        return ValidationOutcome(valid=True, summary=plugin.summary())
    except DynamicsPluginError as e:
        return ValidationOutcome(valid=False, error=str(e))
    except Exception as e:  # noqa: BLE001
        return ValidationOutcome(valid=False, error=f"{type(e).__name__}: {e}")
    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()


def _fix_prompt(source_code: str, error_message: str) -> str:
    return get_prompt("dynamics_validator", "fix_prompt").format(
        standard=DYNAMICS_STANDARD,
        error_message=error_message,
        source_code=source_code,
    )


def fix_dynamics_with_llm(source_code: str, error_message: str):
    """One LLM call proposing a fix. Returns the raw structured output
    (see FixProposal) -- validate_and_fix_dynamics is what actually decides
    whether to trust it (by re-running validate_dynamics_source on the
    result)."""
    from pydantic import BaseModel, Field

    from .llm_base import get_llm

    class FixProposal(BaseModel):
        explanation: str = Field(description="Plain-language summary of what was wrong and what was changed.")
        fixed_code: str = Field(description="The complete, corrected .py file content.")

    llm = get_llm().with_structured_output(FixProposal)
    return llm.invoke(_fix_prompt(source_code, error_message))


def validate_and_fix_dynamics(source_code: str, max_attempts: int = 2) -> FixOutcome:
    """Full pipeline: validate as-is; if that fails, ask the LLM to fix it,
    re-validate the fix, and retry (with the new error) up to
    ``max_attempts`` times. Every "valid" verdict -- including for an
    LLM-produced fix -- comes from the same deterministic DynamicLoader
    check, never from the LLM's own say-so.
    """
    outcome = validate_dynamics_source(source_code)
    if outcome.valid:
        return FixOutcome(valid=True, used_llm_fix=False, final_code=source_code, attempts=0)

    original_error = outcome.error
    current_code = source_code
    current_error = outcome.error
    history = []

    for attempt in range(1, max_attempts + 1):
        log.info("Dynamics validation failed (attempt %d/%d): %s", attempt, max_attempts, current_error)
        try:
            proposal = fix_dynamics_with_llm(current_code, current_error)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM fix attempt %d failed to even run: %s", attempt, e)
            history.append({"attempt": attempt, "error": f"LLM call failed: {e}"})
            break

        recheck = validate_dynamics_source(proposal.fixed_code)
        if recheck.valid:
            return FixOutcome(
                valid=True, used_llm_fix=True, final_code=proposal.fixed_code,
                original_error=original_error, explanation=proposal.explanation,
                attempts=attempt, history=history,
            )

        history.append({"attempt": attempt, "error": recheck.error})
        current_code = proposal.fixed_code
        current_error = recheck.error

    return FixOutcome(
        valid=False, used_llm_fix=True, final_code=current_code,
        original_error=original_error, still_broken_error=current_error,
        attempts=max_attempts, history=history,
    )


def estimate_feedforward_trim(dynamics):
    """Numerically solves for a steady-state ("trim"/feedforward) input at
    the plugin's own default_target: the u that makes dynamics(x_target, u)
    as close to zero as possible.

    This generalizes ``BaseDynamics.get_equilibrium_input()`` -- which
    defaults to all-zeros and is only actually correct for systems that
    genuinely need zero input to hold their target (many don't: gravity,
    steady drag, a required hover thrust, etc. all need a nonzero trim).
    Rather than relying on every plugin author to have manually derived and
    implemented that method correctly, this solves for it directly from
    the plugin's own ``dynamics()`` function via least-squares, seeded from
    whatever ``get_equilibrium_input()`` already returns (so a plugin that
    DOES implement it well just gets numerically confirmed/refined, not
    overridden by a worse guess).

    Returns (trim_input, note). Opt-in from the UI (the "Use computed
    feedforward trim input" toggle in the Configure section) -- unlike
    dt/Q/R, this is NOT applied automatically, since a wrong trim guess for
    a system with no real equilibrium at the target would actively hurt
    rather than help; see the residual-norm check in the note.
    """
    from scipy.optimize import least_squares

    x_target = dynamics.config.default_target.copy()
    u0 = np.asarray(dynamics.get_equilibrium_input(), dtype=float).copy()

    def residual(u):
        return np.asarray(dynamics.dynamics(x_target, u), dtype=float)

    bounds = (-np.inf, np.inf)
    input_bounds = dynamics.get_input_bounds()
    if input_bounds is not None:
        lo, hi = input_bounds
        lo = np.where(np.isfinite(lo), lo, -1e6)
        hi = np.where(np.isfinite(hi), hi, 1e6)
        bounds = (lo, hi)
        u0 = np.clip(u0, lo, hi)

    result = least_squares(residual, u0, bounds=bounds)
    residual_norm = float(np.linalg.norm(result.fun))
    trim = result.x.tolist()

    if residual_norm < 1e-3:
        quality = "a genuine equilibrium (residual is essentially zero)"
    elif residual_norm < 1.0:
        quality = f"an approximate equilibrium (residual norm {residual_norm:.3g} -- small but not exact)"
    else:
        quality = (f"NOT a good equilibrium (residual norm {residual_norm:.3g} is large) -- the target state "
                    f"may not be a true steady state for this system, or it may be outside the input bounds' reach")

    note = f"Computed feedforward trim input: {[round(v, 4) for v in trim]} -- {quality}."
    return trim, note


def analyze_and_setup(source_code: str, max_attempts: int = 2) -> SetupResult:
    """Full pipeline for a newly-uploaded dynamics file: validate/repair
    (validate_and_fix_dynamics, unchanged), then -- only if the result is a
    genuinely loadable dynamics -- run the three deterministic setup
    analyses (derivative-pair detection, initial Q/R, initial dt) on it.
    This is the single entry point app.py calls on every dynamics upload;
    everything downstream of "is this file valid" lives in one agent
    instead of being spread across separate ones, per design.
    """
    fix = validate_and_fix_dynamics(source_code, max_attempts=max_attempts)
    if not fix.valid:
        return SetupResult(fix=fix)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(fix.final_code)
            temp_path = f.name
        plugin = DynamicLoader.load_from_path(temp_path)
        dynamics = plugin.create_dynamics()
    except Exception as e:  # noqa: BLE001
        # Shouldn't happen (fix.valid already confirmed this loads), but
        # don't let a setup-analysis crash take down the whole upload flow.
        log.error("Unexpected error re-loading validated dynamics for setup analysis: %s", e)
        return SetupResult(fix=fix)
    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()

    notes = []
    try:
        pairs = detect_derivative_pairs(dynamics)
        if pairs:
            names = dynamics.state_names
            notes.append("Detected derivative pairs: " + ", ".join(f"{names[j]} = d({names[i]})/dt" for i, j in pairs))
        else:
            notes.append("No derivative pairs detected (or too few reliable samples) -- per-state sin/cos will need to be set manually if wanted.")
    except Exception as e:  # noqa: BLE001
        log.warning("Derivative-pair detection failed: %s", e)
        pairs = []
        notes.append(f"Derivative-pair detection skipped ({e}).")

    suggested_dt = None
    try:
        suggested_dt = estimate_dt(dynamics)
        notes.append(f"Suggested dt_mpc = {suggested_dt:.4g}s (from the fastest linearized mode at equilibrium).")
    except Exception as e:  # noqa: BLE001
        log.warning("dt estimation failed: %s", e)
        notes.append(f"dt estimation skipped ({e}); using the default.")

    suggested_Q, suggested_R = None, None
    qr_diagnostics = None
    try:
        suggested_Q, suggested_R, qr_note, qr_diagnostics = estimate_initial_qr(dynamics)
        notes.append(qr_note)
    except Exception as e:  # noqa: BLE001
        log.warning("Initial Q/R estimation failed: %s", e)
        notes.append(f"Initial Q/R estimation skipped ({e}); using flat defaults.")

    suggested_feedforward = None
    try:
        suggested_feedforward, ff_note = estimate_feedforward_trim(dynamics)
        notes.append(ff_note)
    except Exception as e:  # noqa: BLE001
        log.warning("Feedforward trim estimation failed: %s", e)
        notes.append(f"Feedforward trim estimation skipped ({e}).")

    return SetupResult(
        fix=fix, derivative_pairs=pairs, suggested_Q=suggested_Q, suggested_R=suggested_R,
        suggested_dt=suggested_dt, suggested_feedforward=suggested_feedforward,
        setup_notes=notes, qr_diagnostics=qr_diagnostics,
    )
