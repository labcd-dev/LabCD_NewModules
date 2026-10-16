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
from typing import Any, Dict, Optional

import numpy as np

from ..mpc.jacobian import linearize

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
    suggested_Np: Optional[int] = None                         # see estimate_horizons
    suggested_Nc: Optional[int] = None
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
    automatically generate the mathematically-consistent derivative
    reference (A*w*cos for A*sin) for its paired velocity state instead of
    requiring the user to set it by hand every time -- see
    dynamics/base.py:SystemConfig.desired_trajectory's ``mode`` path. That
    path used to fall back to a guessed (2i, 2i+1) pairing; it now refuses
    the whole-system modes when this detection finds no pairs.
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


#: Closed-loop time constant the initial weights aim for [s]. It sets the
#: acceptable size of a rate in Bryson's rule (a position allowed to be a off
#: may move at about a / SEED_TIME_CONSTANT) and the overall R scale.
SEED_TIME_CONSTANT = 0.3

#: Prediction horizon rule (Garriga & Soroush 2010 review: the horizon should
#: cover the closed-loop settling time): Np*dt ~ HORIZON_SETTLING_MULTIPLE x
#: the time constant -- but no longer than the time in which an UNSTABLE
#: open-loop mode grows by HORIZON_GROWTH_LIMIT. Past that the linearised
#: prediction of an unstable plant is no longer worth trusting: the double
#: pendulum (lambda = 5.7 1/s) settled with a 0.49 s horizon (growth 16x) and
#: diverged with 0.73 s (62x), with any weights.
HORIZON_SETTLING_MULTIPLE = 4.0
HORIZON_GROWTH_LIMIT = 20.0
NP_MIN, NP_MAX = 10, 60          # NP_MAX is the Actor's own upper bound on Np
NC_FRACTION = 0.2                # matches actor.MAX_NC_FRACTION_OF_NP

#: Time in which a rate may close its position's acceptable gap: the rate's
#: acceptable size in Bryson's rule is a_position / RATE_TIME. Half the target
#: time constant -- with the full one the rate weight itself capped the loop at
#: ~1.4 x SEED_TIME_CONSTANT, so no R could reach the target.
RATE_TIME = SEED_TIME_CONSTANT / 2

#: The rate/position weight ratio Bryson's rule implies (RATE_TIME**2 per
#: level of a derivative chain), for callers that still import it.
RATE_WEIGHT_RATIO = RATE_TIME ** 2


def _kinematic_rate_pairs(A: np.ndarray, B: np.ndarray, tol: float = 1e-9) -> Dict[int, int]:
    """{rate state j: the state i it is the rate of}, at the linearisation
    point: row i of A has a single entry, at column j, equal to +/-1, and no
    input acts on x_i directly. The +/-1 is what separates kinematics
    (x' = v) from physics that happens to have one term (a ball on a beam:
    r_dot' = (5/7) g theta -- theta is an angle to hold, not a velocity)."""
    n = A.shape[0]
    pairs: Dict[int, int] = {}
    a_scale = max(float(np.max(np.abs(A))), 1e-300)
    b_scale = max(float(np.max(np.abs(B))), 1e-300)
    for i in range(n):
        nz = np.flatnonzero(np.abs(A[i]) > tol * a_scale)
        if (len(nz) == 1 and nz[0] != i and abs(abs(A[i, nz[0]]) - 1.0) < 1e-6
                and np.max(np.abs(B[i])) <= tol * b_scale):
            pairs.setdefault(int(nz[0]), i)
    return pairs


def _kinematic_rate_states(A: np.ndarray, B: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Boolean mask of the kinematic rate states (see _kinematic_rate_pairs)."""
    rate = np.zeros(A.shape[0], dtype=bool)
    rate[list(_kinematic_rate_pairs(A, B, tol))] = True
    return rate


def _acceptable_from_bounds(bounds, size: int, fallback) -> np.ndarray:
    """Bryson's 'largest acceptable value' where nothing better is given: half
    the declared span (finite bounds), else ``fallback`` (an array), else 1."""
    out = np.zeros(size)
    if bounds is not None:
        span = np.abs(np.asarray(bounds[1], dtype=float) - np.asarray(bounds[0], dtype=float))
        out = np.where(np.isfinite(span) & (span > 0), 0.5 * span, 0.0)
    fb = np.abs(np.asarray(fallback, dtype=float)) if fallback is not None else np.zeros(size)
    out = np.where(out > 0, out, fb)
    return np.where(out > 1e-12, out, 1.0)


def _variable_scale(bounds, operating_point, start) -> np.ndarray:
    """Each variable's own magnitude: half its declared span, else its size at
    the operating point (target / equilibrium), else at the start, else 1."""
    op = np.abs(np.asarray(operating_point, dtype=float))
    st = np.abs(np.asarray(start, dtype=float))
    scale = np.zeros(op.shape)
    if bounds is not None:
        span = np.abs(np.asarray(bounds[1], dtype=float) - np.asarray(bounds[0], dtype=float))
        scale = np.where(np.isfinite(span) & (span > 0), 0.5 * span, 0.0)
    scale = np.where(scale > 0, scale, op)
    scale = np.where(scale > 1e-12, scale, st)
    return np.where(scale > 1e-12, scale, 1.0)


def suggest_horizon(A, dt: float, tau: float = None) -> Dict[str, Any]:
    """Np, Nc and the horizon in seconds for sample time ``dt`` (see
    HORIZON_SETTLING_MULTIPLE / HORIZON_GROWTH_LIMIT). ``A`` is the
    continuous-time linearisation; None skips the instability cap."""
    tau = SEED_TIME_CONSTANT if tau is None or not np.isfinite(tau) else max(tau, SEED_TIME_CONSTANT)
    T_h = HORIZON_SETTLING_MULTIPLE * tau
    lam_u = 0.0
    if A is not None and np.all(np.isfinite(A)):
        lam_u = float(max(0.0, np.max(np.real(np.linalg.eigvals(A)))))
        if lam_u > 1e-6:
            T_h = min(T_h, np.log(HORIZON_GROWTH_LIMIT) / lam_u)
    Np = int(min(NP_MAX, max(NP_MIN, int(np.ceil(T_h / dt)))))
    Nc = max(1, int(np.ceil(NC_FRACTION * Np)))
    return {"Np": Np, "Nc": Nc, "horizon_s": Np * dt, "target_horizon_s": T_h, "unstable_rate": lam_u}


def _mpc_time_constant(Ad, Bd, Q, Rdu, Np: int, Nc: int, dt: float, horizon: float = 10.0,
                       terminal: str = "dare") -> float:
    """How fast the controller's OWN closed loop, unconstrained, brings x back
    to zero: decision du over Nc moves (u = u_prev + cumulative du), cost
    sum x'Qx over Np steps (terminal P from the DARE, as the controller does
    by default) + du'R du, state augmented with
    u_prev exactly as the controller carries it.

    Measured on x itself -- each state displaced in turn, time until x stays
    below 5% of the displacement, divided by 3 (a time constant) -- not as a
    spectral radius: input directions in B's null space (an over-actuated
    plant) are modes with |lambda| = 1 that no cost term sees, and they made
    every R look infinitely slow. Returns inf for a loop that does not settle
    within ``horizon`` seconds or grows."""
    n, m = Bd.shape
    Az = np.block([[Ad, Bd], [np.zeros((m, n)), np.eye(m)]])
    Bz = np.vstack([Bd, np.eye(m)])
    Cx = np.hstack([np.eye(n), np.zeros((n, m))])
    powers = [np.eye(n + m)]
    for _ in range(Np):
        powers.append(Az @ powers[-1])
    Sx = np.vstack([Cx @ powers[k + 1] for k in range(Np)])
    Su = np.zeros((Np * n, Nc * m))
    for k in range(Np):
        for j in range(min(k + 1, Nc)):
            Su[k * n:(k + 1) * n, j * m:(j + 1) * m] = Cx @ powers[k - j] @ Bz
    Qbar = np.kron(np.eye(Np), Q)
    if terminal == "dare":                  # the controller's default terminal weight
        try:
            from scipy.linalg import solve_discrete_are
            Qbar[(Np - 1) * n:, (Np - 1) * n:] = solve_discrete_are(Ad, Bd, Q, Rdu)
        except (ValueError, np.linalg.LinAlgError):
            pass
    H = Su.T @ Qbar @ Su + np.kron(np.eye(Nc), Rdu)
    K = np.linalg.solve(H, Su.T @ Qbar @ Sx)[:m]
    Acl = Az - Bz @ K
    steps = int(round(horizon / dt))
    worst = 0.0
    for i in range(n):
        z = np.zeros(n + m)
        z[i] = 1.0
        xs = np.empty(steps + 1)
        xs[0] = 1.0
        for k in range(steps):
            z = Acl @ z
            xs[k + 1] = float(np.max(np.abs(z[:n] / np.maximum(np.abs(np.eye(n)[i]), 1.0))))
            if not np.isfinite(xs[k + 1]) or xs[k + 1] > 1e6:
                return float("inf")
        above = np.flatnonzero(xs > 0.05)
        if above.size and above[-1] >= steps:
            return float("inf")                  # still outside the band at the end
        settle = (above[-1] + 1) * dt if above.size else dt
        worst = max(worst, settle / 3.0)
    return worst


def _mpc_r_scale(A, B, Q, R_shape, dt: float, tau_target: float = None,
                 Np: int = None, Nc: int = None):
    """The largest multiplier r on R_shape for which the controller's own
    unconstrained closed loop (see _mpc_time_constant) settles with a
    slowest time constant <= tau_target; if none does, the r where the time
    constant stops improving (within 10% of the best seen). Returns (r, tau)
    or (None, None).

    An infinite-horizon LQR on u was used first, and it is not this
    controller: on an unstable double pendulum its "fast enough" R let the
    short-horizon, du-penalising MPC diverge."""
    tau_target = SEED_TIME_CONSTANT if tau_target is None else tau_target
    if Np is None or Nc is None:
        hz = suggest_horizon(A, dt)
        Np, Nc = hz["Np"], hz["Nc"]
    n, m = B.shape
    Ad, Bd = np.eye(n) + dt * A, dt * B
    Qd = np.diag(Q)
    seen = []
    for r in np.logspace(2, -8, 41):
        try:
            tau = _mpc_time_constant(Ad, Bd, Qd, np.diag(r * R_shape), Np, Nc, dt)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.isfinite(tau):
            continue
        if tau <= tau_target:
            return float(r), float(tau)
        seen.append((float(r), float(tau)))
    if not seen:
        return None, None
    best = min(t for _, t in seen)
    return next((r, t) for r, t in seen if t <= 1.1 * best)


def _lqr_r_scale(A, B, Q, R_shape, dt: float, tau_target: float = None):
    """The largest multiplier r on R_shape whose discrete LQR closed loop has
    a slowest time constant <= tau_target. If no r reaches it, the r where
    the time constant stops improving (within 10% of the best seen) --
    never the most extreme value just because the plant is slow.
    Returns (r, tau) or (None, None)."""
    from scipy.linalg import solve_discrete_are

    tau_target = SEED_TIME_CONSTANT if tau_target is None else tau_target
    n, m = B.shape
    Ad, Bd = np.eye(n) + dt * A, dt * B
    Qd = np.diag(Q)
    seen = []
    for r in np.logspace(2, -6, 33):
        Rd = np.diag(r * R_shape)
        try:
            P = solve_discrete_are(Ad, Bd, Qd, Rd)
            K = np.linalg.solve(Rd + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
            rho = float(np.max(np.abs(np.linalg.eigvals(Ad - Bd @ K))))
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.isfinite(rho) or rho >= 1.0:
            continue
        tau = -dt / np.log(rho) if rho > 0 else 0.0
        if tau <= tau_target:
            return float(r), float(tau)
        seen.append((float(r), float(tau)))
    if not seen:
        return None, None
    best = min(t for _, t in seen)
    return next((r, t) for r, t in seen if t <= 1.1 * best)


def _role_seed(dynamics, x0=None, u_eq=None, lqr_dt: float = 0.02, acceptable=None) -> Dict[str, Any]:
    """The initial Q/R and what they were built from, with no probing: the
    linearisation (A, B -- None when it fails), which states are kinematic
    rates, Q by role over units, and R from the LQR time-constant target.
    Shared by estimate_initial_qr and estimate_dt so both reason about the
    same closed loop."""
    from ..mpc.jacobian import linearize

    n_states = dynamics.n_states
    x0 = dynamics.config.default_initial_state.copy() if x0 is None else x0
    u_eq = dynamics.get_equilibrium_input() if u_eq is None else u_eq
    try:
        A_lin, B_lin = linearize(dynamics.dynamics, x0, u_eq,
                                 torch_dynamics_fn=getattr(dynamics, "dynamics_torch", None))
        if not (np.all(np.isfinite(A_lin)) and np.all(np.isfinite(B_lin))):
            A_lin, B_lin = None, None
    except Exception:  # noqa: BLE001
        A_lin, B_lin = None, None
    pairs = _kinematic_rate_pairs(A_lin, B_lin) if A_lin is not None else {}
    rate = np.zeros(n_states, dtype=bool)
    rate[list(pairs)] = True

    # Bryson's rule (Bryson & Ho 1975): Q_ii = 1 / (largest acceptable
    # deviation of x_i)^2, R_jj = 1 / (largest acceptable value of u_j)^2.
    #   * acceptable deviations: the user's (``acceptable`` or the plugin's
    #     config.acceptable_deviation) where given; else half the declared
    #     bound span; else the initial gap |x0 - target|; else 1;
    #   * a kinematic RATE with no value of its own may be as large as needed
    #     to close its position's acceptable gap in RATE_TIME seconds:
    #     a_rate = a_position / RATE_TIME -- never its own declared
    #     bounds, which are safety limits (narrow rate bounds put the double
    #     pendulum's angular rates above its angles);
    #   * inputs: half the declared span, else the equilibrium input, else 1.
    # The OLD version took the "acceptable deviation" from an open-loop probe
    # -- how far the plant drifts on its own -- which gave the quadcopter's
    # yaw a weight 1700x below pitch.
    target = np.asarray(dynamics.config.default_target, dtype=float)
    user = acceptable if acceptable is not None else getattr(dynamics.config, "acceptable_deviation", None)
    user = (np.full(n_states, np.nan) if user is None
            else np.asarray(user, dtype=float).ravel())
    user_ok = np.isfinite(user) & (user > 0)
    a = _acceptable_from_bounds(dynamics.get_state_bounds(), n_states, np.abs(np.asarray(x0, dtype=float) - target))
    a = np.where(user_ok, user, a)
    for _ in range(n_states):                     # chains: jerk -> accel -> vel -> pos
        for j, i in pairs.items():
            if not user_ok[j]:
                a[j] = a[i] / RATE_TIME
    Q_bryson = 1.0 / a ** 2
    Q = Q_bryson * (10.0 / np.max(Q_bryson))      # display scale only: MPC sees Q/R
    u_max = _acceptable_from_bounds(dynamics.get_input_bounds(), dynamics.n_inputs, u_eq)
    R_shape = 1.0 / u_max ** 2
    # The one free number Bryson leaves -- how much input to spend overall --
    # set from the closed loop: the cheapest R whose MPC loop (this
    # controller's own formulation, DARE terminal, suggested horizon) settles
    # with a SEED_TIME_CONSTANT time constant.
    r_scale, tau = (_mpc_r_scale(A_lin, B_lin, Q, R_shape, dt=lqr_dt)
                    if A_lin is not None else (None, None))
    if r_scale is None:
        r_scale = 1.0 / np.max(R_shape)          # no usable linearisation: R_max = 1
    return {"A": A_lin, "B": B_lin, "rate": rate, "Q": Q, "R": R_shape * r_scale, "tau": tau,
            "tau_met": tau is not None and tau <= SEED_TIME_CONSTANT + 1e-12, "acceptable": a,
            "u_max": u_max}


def estimate_initial_qr(dynamics, step_fraction: float = 0.25, probe_time_horizon: float = 2.0,
                        probe_steps: int = 100, lqr_dt: float = 0.02, acceptable=None):
    """Initial Q/R for a newly loaded plant, by Bryson's rule (Bryson & Ho
    1975): Q_ii = 1 / (largest acceptable deviation of x_i)^2 and
    R_jj = 1 / (largest acceptable value of u_j)^2, then one overall R scale
    chosen so the controller's own closed loop settles in SEED_TIME_CONSTANT.
    See _role_seed for where each acceptable value comes from; ``acceptable``
    (one value per state, NaN = automatic) overrides them.

    The open-loop step probe below no longer sets any weight -- it used to,
    as Q = 1/range^2, and an open-loop excursion is how far the plant drifts,
    not how much error is acceptable (quadcopter campaign, bug 12). It still
    runs for the Setup panel's plot.
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

    seed = _role_seed(dynamics, x0, u_eq, lqr_dt, acceptable=acceptable)
    rate, Q, R, tau = seed["rate"], seed["Q"], seed["R"], seed["tau"]
    Q_TARGET_MAX, R_TARGET_MAX = float(np.max(Q)), float(np.max(R))
    Q, R = Q.tolist(), R.tolist()

    rate_names = [dynamics.state_names[i] if i < len(dynamics.state_names) else f"x{i}"
                  for i in np.flatnonzero(rate)]
    note = (
        "Initial Q/R by Bryson's rule: Q_ii = 1/(acceptable deviation)^2, R_jj = 1/(acceptable input)^2"
        + (f"; rates ({', '.join(rate_names)}) may be as large as their position's acceptable deviation "
           f"per {RATE_TIME:g}s" if rate_names else "")
        + "; overall R scale "
        + (f"set so this MPC's closed loop settles with a {tau:.2g}s time constant." if tau is not None
           else "left at R_max = 1 (no usable linearisation).")
    )

    diagnostics = {
        "trajectory": first_probe_traj,          # (T, n_states) -- the baseline probe, for plotting
        "probe_dt": probe_dt,
        "ranges": ranges.tolist(),                 # the (max-across-directions) range used for Q, per state
        "step_mag": step_mag.tolist(),               # the input step used for R, per input
        "Q_TARGET_MAX": Q_TARGET_MAX,
        "R_TARGET_MAX": R_TARGET_MAX,
        "rate_states": [int(i) for i in np.flatnonzero(rate)],
        "lqr_time_constant": tau,
        "acceptable_deviation": np.asarray(seed["acceptable"]).tolist(),
        "acceptable_input": np.asarray(seed["u_max"]).tolist(),
    }
    return Q, R, note, diagnostics


def estimate_horizons(dynamics, dt: float, tau: Optional[float] = None) -> Dict[str, Any]:
    """Np and Nc for sample time ``dt``: a prediction horizon of about
    HORIZON_SETTLING_MULTIPLE closed-loop time constants (the horizon should
    cover the settling time), capped where an unstable open-loop mode would
    grow past HORIZON_GROWTH_LIMIT, with Nc = NC_FRACTION of Np. Np used to be
    a fixed 12-20 whatever dt was, which left a 0.01 s horizon on a plant
    sampled at 0.0005 s and made most of the test-suite plants fail."""
    from ..mpc.jacobian import linearize

    try:
        A, _ = linearize(dynamics.dynamics, dynamics.config.default_initial_state.copy(),
                         dynamics.get_equilibrium_input(),
                         torch_dynamics_fn=getattr(dynamics, "dynamics_torch", None))
    except Exception:  # noqa: BLE001
        A = None
    hz = suggest_horizon(A, dt, tau=tau)
    capped = hz["target_horizon_s"] < HORIZON_SETTLING_MULTIPLE * max(tau or 0.0, SEED_TIME_CONSTANT) - 1e-12
    hz["note"] = (
        f"Suggested Np = {hz['Np']}, Nc = {hz['Nc']} (prediction horizon {hz['horizon_s']:.3g}s"
        + (f", kept under ln({HORIZON_GROWTH_LIMIT:g})/{hz['unstable_rate']:.3g} so the unstable mode's "
           f"prediction grows at most {HORIZON_GROWTH_LIMIT:g}x" if capped else
           f" ~ {HORIZON_SETTLING_MULTIPLE:g} closed-loop time constants")
        + (f"; Np capped at {NP_MAX}" if hz["Np"] == NP_MAX and hz["target_horizon_s"] / dt > NP_MAX else "")
        + "). Terminal weight from the Riccati equation (DARE)."
    )
    return hz


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
    euler_cap = None
    try:
        A, _ = linearize(dynamics.dynamics, x0, u_eq, torch_dynamics_fn=getattr(dynamics, "dynamics_torch", None))
        eigs = np.linalg.eigvals(A)
        fastest = float(np.max(np.abs(eigs)))
        if fastest > 1e-4:  # treat anything slower than this as "no reliable fast mode" rather than trust a huge dt
            eig_dt = 1.0 / fastest / 15.0
        # The MPC predicts with forward Euler, which is only stable on a
        # stable mode while |1 + dt*lambda| < 1, i.e. dt < -2 Re(l) / |l|^2.
        # A lightly damped mode makes that much tighter than 1/(15 |l|), and
        # the suitability check rejects a dt past it -- so Setup must not
        # suggest one. Half the limit, for margin.
        limits = [-2.0 * z.real / abs(z) ** 2 for z in eigs if z.real < -1e-9]
        if limits:
            euler_cap = 0.5 * min(limits)
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

    # ---- estimate 3: the closed loop the seed weights produce ----
    # The rise-time estimate above measures how fast the plant reacts to its
    # inputs, which for any directly actuated state is near-instant -- on the
    # quadcopter it gave 0.00375 s where 0.01-0.02 s works perfectly
    # (campaign bug 13). What dt has to resolve is the CONTROLLED response:
    # ~15 samples per time constant of the dominant (slowest) mode of the
    # LQR built from the initial Q/R. Fast closed-loop poles do not need
    # sampling on their own: where the open-loop model is fast, estimate 1
    # already bounds dt; where it is an integrator, Euler is exact.
    cl_dt = None
    seed = None
    try:
        seed = _role_seed(dynamics, x0, u_eq)
        # Only a time constant the seed actually reaches is a basis for dt.
        # When no R gets there (planar VTOL: best ~2.8 s with a 20-step
        # horizon), tau/15 inflated dt to 0.19 s and the run diverged; the
        # conservative estimates above decide instead.
        if seed.get("tau_met") and seed["tau"] > 0:
            cl_dt = seed["tau"] / 15.0
    except Exception:  # noqa: BLE001
        cl_dt = None

    if cl_dt is not None:
        candidates = [d for d in (eig_dt, cl_dt) if d is not None and d > 0]
    else:
        candidates = [d for d in (eig_dt, rise_dt) if d is not None and d > 0]
    dt = min(candidates) if candidates else 0.05  # no signal available -- sane fallback

    # ---- the horizon has to fit in NP_MAX steps ----
    # A prediction horizon of HORIZON_SETTLING_MULTIPLE time constants (see
    # suggest_horizon) at a tiny dt needs more than NP_MAX steps; Np was then
    # capped and the horizon in SECONDS collapsed (0.03-0.2 s on half the test
    # suite, which then failed). Coarsen dt until the horizon fits -- but never
    # past what the open-loop model needs (estimate 1) or Euler stability.
    try:
        # A time constant the seed did not reach is no basis for the horizon
        # either: a slow, missed tau stretched it to 2.8 s and dt to 0.19 s.
        tau_h = seed["tau"] if seed and seed.get("tau_met") else None
        hz = suggest_horizon(seed["A"] if seed else None, 1.0, tau=tau_h)
        dt = max(dt, hz["target_horizon_s"] / NP_MAX)
        if eig_dt is not None:
            dt = min(dt, eig_dt)
    except Exception:  # noqa: BLE001
        pass
    if euler_cap is not None:
        dt = min(dt, euler_cap)
    # The floor used to be 0.001 s, which could push dt back past the Euler
    # cap above; a slow run is better than one the suitability check rejects.
    return float(np.clip(dt, 1e-4, 0.5))


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


def analyze_actuation(dynamics, target=None, rank_tol: float = 1e-8,
                       residual_tol: float = 1e-6) -> Dict[str, Any]:
    """Can this plant actually be held where the user asked it to be held?

    Counting inputs against states is the obvious test and it is the wrong
    one: a double integrator has two states and one input, and it holds both
    of them perfectly well at a setpoint. What decides the question is how
    many state values can be commanded INDEPENDENTLY at steady state, which is
    a rank condition, not a count.

    At an equilibrium, f(x*, u*) = 0. Linearising, that needs A x* to lie in
    the range of B, so the reachable equilibria form the subspace
    {x : N A x = 0} where the rows of N span the left null space of B. Its
    dimension -- reported here as ``independently_holdable`` -- is the number
    of state values that can be dialled in at once. Everything else follows
    from the plant, whatever the user types into the target field.

    Asking for a target outside that subspace is not a hard tuning problem; it
    is an impossible one. The controller will chase it forever, the error
    never settles, and every metric looks like bad tuning -- which is why this
    belongs in diagnostics rather than being left for the user to infer from a
    run that simply never converges.

    ``target`` is optional; without it the classification is still produced,
    just with no feasibility verdict.

    Returns a plain dict (JSON-friendly, no numpy scalars) so it can travel
    into a prompt, a report, or an API response unchanged.
    """
    import numpy as _np

    n_states, n_inputs = dynamics.n_states, dynamics.n_inputs
    names = list(getattr(dynamics, "state_names", []) or [f"x{i}" for i in range(n_states)])
    x_eq = _np.asarray(dynamics.config.default_target, dtype=float).copy()
    u_eq = _np.asarray(dynamics.get_equilibrium_input(), dtype=float).ravel()

    A, B = linearize(dynamics_fn=dynamics.dynamics, x=x_eq, u=u_eq,
                      torch_dynamics_fn=getattr(dynamics, "dynamics_torch", None))
    A = _np.atleast_2d(_np.asarray(A, dtype=float))
    B = _np.asarray(B, dtype=float).reshape(n_states, n_inputs)

    rank_B = int(_np.linalg.matrix_rank(B, tol=rank_tol))

    # Controllability, for context: a plant can be under-actuated and still
    # fully controllable (you steer the unactuated states THROUGH the actuated
    # ones). That is a different question from what can be HELD, and conflating
    # the two is the usual source of confusion here.
    blocks, term = [B], B
    for _ in range(n_states - 1):
        term = A @ term
        blocks.append(term)
    ctrb_rank = int(_np.linalg.matrix_rank(_np.hstack(blocks), tol=rank_tol))

    # The equilibrium subspace: rows of N span the left null space of B.
    u_svd, s_svd, _ = _np.linalg.svd(B)
    left_null = u_svd[:, rank_B:]                       # (n_states, n_states - rank_B)
    if left_null.size:
        holdable = n_states - int(_np.linalg.matrix_rank(left_null.T @ A, tol=rank_tol))
    else:
        holdable = n_states

    if rank_B >= n_states:
        classification = "fully actuated"
    elif n_inputs > rank_B:
        classification = "over-actuated (redundant actuators)"
    else:
        classification = "under-actuated"
    if n_inputs > n_states and rank_B >= n_states:
        classification = "over-actuated (redundant actuators)"

    out: Dict[str, Any] = {
        "n_states": int(n_states),
        "n_inputs": int(n_inputs),
        "rank_B": rank_B,
        "controllability_rank": ctrb_rank,
        "fully_controllable": bool(ctrb_rank >= n_states),
        "classification": classification,
        "independently_holdable": int(holdable),
        "state_names": names,
        "target_feasible": None,
        "target_residual": None,
        "unreachable_states": [],
        "derivative_pair_violations": [],
        "notes": [],
    }

    if holdable < n_states:
        out["notes"].append(
            f"At most {holdable} of the {n_states} states can be held at independently chosen "
            f"values; the remaining {n_states - holdable} follow from the plant."
        )
    if ctrb_rank < n_states:
        out["notes"].append(
            f"The linearised plant is NOT fully controllable at the target "
            f"(controllability rank {ctrb_rank} of {n_states}); some directions cannot be "
            f"influenced by the inputs at all."
        )

    # ---- a velocity-like state cannot sit at a nonzero constant ------------
    pairs = getattr(dynamics.config, "derivative_pairs", None) or []
    if target is not None:
        tgt = _np.asarray(target, dtype=float).ravel()
        for pos_i, vel_j in pairs:
            if vel_j < len(tgt) and abs(float(tgt[vel_j])) > residual_tol:
                out["derivative_pair_violations"].append({
                    "position": names[pos_i] if pos_i < len(names) else f"x{pos_i}",
                    "velocity": names[vel_j] if vel_j < len(names) else f"x{vel_j}",
                    "requested": float(tgt[vel_j]),
                })
        if out["derivative_pair_violations"]:
            out["notes"].append(
                "A state that is the derivative of another cannot hold a nonzero constant value: "
                "if the velocity stays nonzero the position keeps moving, so this is not a steady "
                "state at all. Use a moving reference if that is what you meant."
            )

        # ---- is the requested target an equilibrium at all? ----------------
        # Find the input that comes closest to holding it, then look at what is
        # left over. Whatever the residual lands on is a state the plant cannot
        # be parked at, no matter how the controller is tuned.
        try:
            drift = _np.asarray(dynamics.dynamics(tgt, u_eq), dtype=float).ravel()
            du, *_ = _np.linalg.lstsq(B, -drift, rcond=None)
            residual = drift + B @ du
            res_norm = float(_np.linalg.norm(residual))
            scale = max(float(_np.linalg.norm(drift)), 1.0)
            out["target_residual"] = res_norm
            out["target_feasible"] = bool(res_norm <= max(residual_tol, 1e-6 * scale))
            if not out["target_feasible"]:
                worst = _np.argsort(-_np.abs(residual))
                out["unreachable_states"] = [
                    {"state": names[i] if i < len(names) else f"x{i}",
                     "residual": float(residual[i])}
                    for i in worst if abs(float(residual[i])) > max(residual_tol, 1e-3 * res_norm)
                ][:5]
                out["notes"].append(
                    f"The requested target is not an equilibrium of this plant: with the best "
                    f"possible input the state derivative still has magnitude {res_norm:.4g}. "
                    f"The controller cannot park the system there."
                )
        except Exception as exc:  # noqa: BLE001 -- diagnosis must never crash the caller
            out["notes"].append(f"Target feasibility could not be evaluated: {exc}")

    return out


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

    suggested_Np, suggested_Nc = None, None
    try:
        if suggested_dt:
            tau_seed = (qr_diagnostics or {}).get("lqr_time_constant")
            hz = estimate_horizons(dynamics, suggested_dt,
                                   tau=tau_seed if tau_seed is not None and tau_seed <= SEED_TIME_CONSTANT + 1e-12
                                   else None)
            suggested_Np, suggested_Nc = hz["Np"], hz["Nc"]
            notes.append(hz["note"])
    except Exception as e:  # noqa: BLE001
        log.warning("Horizon suggestion failed: %s", e)
        notes.append(f"Horizon suggestion skipped ({e}).")

    suggested_feedforward = None
    try:
        suggested_feedforward, ff_note = estimate_feedforward_trim(dynamics)
        notes.append(ff_note)
    except Exception as e:  # noqa: BLE001
        log.warning("Feedforward trim estimation failed: %s", e)
        notes.append(f"Feedforward trim estimation skipped ({e}).")

    return SetupResult(
        fix=fix, derivative_pairs=pairs, suggested_Q=suggested_Q, suggested_R=suggested_R,
        suggested_dt=suggested_dt, suggested_Np=suggested_Np, suggested_Nc=suggested_Nc,
        suggested_feedforward=suggested_feedforward, setup_notes=notes, qr_diagnostics=qr_diagnostics,
    )
