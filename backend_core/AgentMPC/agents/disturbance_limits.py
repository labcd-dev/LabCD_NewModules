"""
How large a constant disturbance (and how weak an actuator) a plant can
reasonably be asked to survive.

A Level-4 disturbance d pushes on the state derivative,
x' = f(x, lambda * u) + d. Typing an arbitrary number there easily asks for
something no controller can do -- a push bigger than the actuators can ever
oppose -- and the run then "fails" for a reason that has nothing to do with
tuning. The limit used here is a steady-state one:

    d is admissible  <=>  some (x, u) satisfies  f(x, lambda * u) + d = 0
                          with |x - target| <= acceptable deviation  and
                          u within HOLD_RESERVE of each input's range,

i.e. the plant, with its (possibly weakened) actuators, can still be held
somewhere within the acceptable deviation of the target while keeping part
of every actuator's range in reserve for the transient. Solved as a bounded
least-squares problem.

Holding is necessary but not enough: the push arrives as a step, and before
any controller has caught it the states may already be outside their declared
bounds -- where an MPC with state constraints has no feasible QP left. So the
disturbance must also pass a TRANSIENT check: from the run's real initial
state, under a reference LQR (Bryson weights from the same acceptable
deviations, saturated inputs, nominal model -- it does not know lambda or d),
the plant must stay within TRANSIENT_MARGIN of the way from the target to
each declared state bound (within the acceptable deviation for a state with
no declared bound) for TRANSIENT_HORIZON -- an MPC predicting with a
linearisation needs room before its hard bounds, and a push that takes the
plant most of the way there leaves its QP infeasible (cart-pole: a 20% push
reached 1.1 of 1.57 rad and the QP failed, while the LQR itself survived).
The largest strength passing both is found by bisection along the
disturbance direction (by default the UI's: each state's own range, scaled
by one fraction).

The acceptable deviations are the user's own where given (Setup's Bryson
table), else half the declared bounds, else the initial gap. Unlike the
weight seed, a rate state is NOT widened to its position's deviation over
RATE_TIME: at rest a velocity should stay near zero, and letting it drift
by a/RATE_TIME would let any push on a kinematic row look "held".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

#: Share of each input's range (from its trim value to its bound) the
#: steady state may use; the rest is kept for the controller's transient.
HOLD_RESERVE = 0.8

#: A steady state counts as held when every derivative is below this
#: fraction of its state's scale per second.
HOLD_TOL = 1e-6

#: Weight of the pull towards the start point in the hold problem.
REGULARISATION = 1e-6

#: Bisection resolution on the disturbance fraction (relative).
FRACTION_RESOLUTION = 1e-2

#: The search never goes past this fraction (100% of the range per second).
FRACTION_CEILING = 1.0

#: The transient must stay within this fraction of the distance from the
#: target to each declared state bound (the initial state always counts as
#: inside).
TRANSIENT_MARGIN = 0.5

#: How long the transient check follows the disturbed plant (seconds).
TRANSIENT_HORIZON = 3.0

#: Upper limit on integration steps in one transient check.
TRANSIENT_MAX_STEPS = 6000

#: The transient check stops early once every rate is below this fraction
#: of its state's scale per second (the plant has settled).
SETTLED_RATE = 1e-4


def _names(dynamics, n, attr, prefix):
    names = list(getattr(dynamics, attr, None) or [])
    return names if len(names) == n else [f"{prefix}{i}" for i in range(n)]


def acceptable_deviation(dynamics) -> np.ndarray:
    """Acceptable steady-state deviation per state (see the module doc)."""
    from .dynamics_validator import _acceptable_from_bounds

    n = dynamics.n_states
    x0 = np.asarray(dynamics.config.default_initial_state, dtype=float)
    tg = np.asarray(dynamics.config.default_target, dtype=float)
    a = _acceptable_from_bounds(dynamics.get_state_bounds(), n, np.abs(x0 - tg))
    user = getattr(dynamics.config, "acceptable_deviation", None)
    user_ok = np.zeros(n, dtype=bool)
    if user is not None:
        user = np.asarray(user, dtype=float).ravel()
        if user.size == n:
            user_ok = np.isfinite(user) & (user > 0)
            a = np.where(user_ok, user, a)
    return a


class _HoldProblem:
    """Can the plant be held near the target against disturbance d?"""

    def __init__(self, dynamics, effectiveness=None, reserve: float = HOLD_RESERVE):
        from .reference_authority import _input_bounds, _linear_model, state_scale, trim_input

        self.dyn = dynamics
        n, m = dynamics.n_states, dynamics.n_inputs
        self.n, self.m = n, m
        self.target = np.asarray(dynamics.config.default_target, dtype=float)
        self.scale = state_scale(dynamics)
        lam = np.ones(m) if effectiveness is None else np.broadcast_to(
            np.asarray(effectiveness, dtype=float).ravel(), (m,)).copy()
        self.lam = np.clip(np.where(np.isfinite(lam), lam, 1.0), 0.0, 1.0)
        self.a = acceptable_deviation(dynamics)
        lo, hi = _input_bounds(dynamics, m)
        u_trim = trim_input(dynamics, self.target)
        u_trim = np.clip(u_trim, np.where(np.isfinite(lo), lo, -np.inf), np.where(np.isfinite(hi), hi, np.inf))
        self.u_lo = np.where(np.isfinite(lo), u_trim - reserve * (u_trim - lo), -np.inf)
        self.u_hi = np.where(np.isfinite(hi), u_trim + reserve * (hi - u_trim), np.inf)
        self.u_trim = u_trim
        from .dynamics_validator import _acceptable_from_bounds
        self.u_scale = _acceptable_from_bounds(dynamics.get_input_bounds(), m, np.abs(u_trim))
        x_lo, x_hi = self.target - self.a, self.target + self.a
        sb = dynamics.get_state_bounds()
        if sb is not None:
            x_lo = np.maximum(x_lo, np.where(np.isfinite(sb[0]), sb[0], -np.inf))
            x_hi = np.minimum(x_hi, np.where(np.isfinite(sb[1]), sb[1], np.inf))
        self.lb = np.concatenate([x_lo, self.u_lo])
        self.ub = np.concatenate([x_hi, self.u_hi])
        self._z = np.concatenate([self.target, np.clip(u_trim, self.u_lo, self.u_hi)])
        # The linearisation's Jacobian, held constant: one model call per
        # iteration instead of n+m+1 finite differences -- the solves stay
        # near the target, where it is accurate enough to converge.
        A, B, _ = _linear_model(dynamics, self.target, u_trim)
        self._A, self._B = A, B

        # ---- the transient check's reference controller --------------------
        self.x0 = np.asarray(dynamics.config.default_initial_state, dtype=float)
        self.lo_full, self.hi_full = lo, hi
        # A state without a declared bound is held to its acceptable deviation
        # instead (as if its bound were twice that away), so a plant that
        # declares no bounds at all is still checked against something.
        blo = self.target - 2.0 * self.a
        bhi = self.target + 2.0 * self.a
        if sb is not None:
            blo = np.where(np.isfinite(sb[0]), np.asarray(sb[0], dtype=float), blo)
            bhi = np.where(np.isfinite(sb[1]), np.asarray(sb[1], dtype=float), bhi)
        inner_lo = self.target - TRANSIENT_MARGIN * np.maximum(self.target - blo, 0.0)
        inner_hi = self.target + TRANSIENT_MARGIN * np.maximum(bhi - self.target, 0.0)
        self.x_lo = np.minimum(inner_lo, self.x0)
        self.x_hi = np.maximum(inner_hi, self.x0)
        self.K = None
        self.transient_note = ""
        try:
            from scipy.linalg import solve_continuous_are

            Rw = np.diag(1.0 / self.u_scale ** 2)
            P = solve_continuous_are(A, B, np.diag(1.0 / self.a ** 2), Rw)
            self.K = np.linalg.solve(Rw, B.T @ P)
            rate = max(float(np.max(np.abs(np.linalg.eigvals(A - B @ self.K)))),
                       float(np.max(np.abs(np.linalg.eigvals(A)))), 1e-6)
            self.h = min(0.01, 1.0 / rate)          # RK4 is stable up to h|lambda| ~ 2.8
            self.t_steps = min(int(np.ceil(TRANSIENT_HORIZON / self.h)), TRANSIENT_MAX_STEPS)
            if not self.transient(np.zeros(n))[0]:
                # the reference LQR cannot even recover from x0 inside the
                # bounds: it is no yardstick for this plant, so skip the check
                self.transient_note = ("A reference LQR leaves the state bounds' margin from this initial "
                                       "state even without a disturbance, so only the steady-state "
                                       "test was applied.")
                self.K = None
        except Exception:  # noqa: BLE001 -- not stabilisable: steady-state test only
            self.K = None
            self.transient_note = "No reference LQR exists for this plant; only the steady-state test was applied."

    def transient(self, d, lam=None):
        """(stays inside the margin of the declared bounds, first state to leave)
        for disturbance d applied as a step from t = 0."""
        if self.K is None:
            return True, None
        from .reference_authority import _rk4

        lam = self.lam if lam is None else lam
        d = np.asarray(d, dtype=float)
        x = self.x0.copy()
        f = lambda xx, uu: np.asarray(self.dyn.dynamics(xx, lam * uu), dtype=float) + d  # noqa: E731
        for k in range(self.t_steps):
            u = np.clip(self.u_trim - self.K @ (x - self.target), self.lo_full, self.hi_full)
            if k % 50 == 49 and np.max(np.abs(f(x, u)) / self.scale) < SETTLED_RATE:
                return True, None
            x = _rk4(f, x, u, self.h)
            if not np.all(np.isfinite(x)) or np.max(np.abs(x - self.target) / self.scale) > 1e3:
                return False, None
            out = np.flatnonzero((x < self.x_lo - 1e-9 * self.scale) | (x > self.x_hi + 1e-9 * self.scale))
            if out.size:
                return False, int(out[0])
        return True, None

    def admissible(self, d, lam=None) -> bool:
        return self.solve(d, lam)[0] and self.transient(d, lam)[0]

    def solve(self, d, lam=None):
        """(held, worst residual, x, u) for disturbance d."""
        from scipy.optimize import least_squares

        lam = self.lam if lam is None else lam
        d = np.asarray(d, dtype=float)

        def resid(z):
            x, u = z[: self.n], z[self.n:]
            return (np.asarray(self.dyn.dynamics(x, lam * u), dtype=float) + d) / self.scale

        # n equations in n + m unknowns: a tiny pull towards the start point
        # makes the problem well posed (the trust region otherwise stalls on
        # the flat directions) without moving the answer measurably.
        z0 = np.clip(self._z, self.lb, self.ub)
        zs = np.concatenate([self.a, self.u_scale])
        J = np.vstack([np.hstack([self._A, self._B * lam[None, :]]) / self.scale[:, None],
                       np.diag(REGULARISATION / zs)])
        try:
            sol = least_squares(lambda z: np.concatenate([resid(z), REGULARISATION * (z - z0) / zs]),
                                z0, jac=lambda z: J, bounds=(self.lb, self.ub), xtol=1e-15,
                                ftol=1e-15, gtol=1e-15, max_nfev=200)
        except Exception:  # noqa: BLE001
            return False, np.inf, None, None
        res = float(np.max(np.abs(resid(sol.x))))
        return bool(np.isfinite(res) and res <= HOLD_TOL), res, sol.x[: self.n], sol.x[self.n:]

    def max_fraction(self, direction, lam=None):
        """Largest s with s * direction admissible (bisection); 0 if none."""
        held, *_ = self.solve(np.zeros(self.n), lam)
        if not held:
            return 0.0
        lo, hi = 0.0, None
        s = 0.01
        while s <= FRACTION_CEILING:
            if self.admissible(s * direction, lam):
                lo, s = s, s * 2.0
            else:
                hi = s
                break
        if hi is None:
            if self.admissible(FRACTION_CEILING * direction, lam):
                return FRACTION_CEILING
            hi = FRACTION_CEILING
        while hi - lo > FRACTION_RESOLUTION * max(hi, 1e-3):
            mid = 0.5 * (lo + hi)
            if self.admissible(mid * direction, lam):
                lo = mid
            else:
                hi = mid
        return lo


def disturbance_limits(dynamics, effectiveness=None, direction=None, per_state: bool = False,
                       reserve: float = HOLD_RESERVE) -> Dict[str, Any]:
    """The largest admissible disturbance strength for this plant.

    ``direction`` is the per-state disturbance per unit fraction (default:
    each state's own range, the UI's convention); ``effectiveness`` is the
    actuator lambda (scalar or one per input). Returns a plain dict:

    ok                 -- False when the target cannot be held even with no
                          disturbance (then no limit applies; see message)
    max_fraction       -- largest fraction for the UI's all-states push
    per_state_fraction -- largest fraction when only that state is pushed
    min_effectiveness  -- smallest uniform lambda that still holds the target
    binding            -- what runs out first at the limit
    message            -- user-facing explanation (English)
    """
    from .scenario_presets import _state_span_scale

    prob = _HoldProblem(dynamics, effectiveness, reserve)
    n, m = prob.n, prob.m
    names = _names(dynamics, n, "state_names", "x")
    in_names = _names(dynamics, m, "input_names", "u")
    span = _state_span_scale(dynamics)
    direction = span if direction is None else np.asarray(direction, dtype=float)

    held0, res0, *_ = prob.solve(np.zeros(n))
    out: Dict[str, Any] = {"ok": held0, "max_fraction": None, "per_state_fraction": None,
                           "min_effectiveness": None, "binding": [], "reserve": reserve,
                           "acceptable_deviation": prob.a.tolist(), "message": ""}

    # the weakest uniform actuator that still holds the target, undisturbed
    if prob.solve(np.zeros(n), np.ones(m))[0]:
        lo, hi = 0.0, 1.0
        if prob.solve(np.zeros(n), np.zeros(m))[0]:
            hi = 0.0
        while hi - lo > 1e-2:
            mid = 0.5 * (lo + hi)
            if prob.solve(np.zeros(n), np.full(m, mid))[0]:
                hi = mid
            else:
                lo = mid
        out["min_effectiveness"] = float(hi)

    if not held0:
        weak = out["min_effectiveness"] is not None
        out["message"] = (
            ("With this actuator effectiveness the plant cannot be held near its target even "
             f"without any disturbance: it needs at least lambda = {out['min_effectiveness']:.2f} "
             "on every actuator. No disturbance is admissible until lambda is raised.")
            if weak else
            ("The target is not a steady state this plant can hold within its acceptable "
             "deviation, so no disturbance limit can be computed (fix the target first).")
        )
        out["max_fraction"] = 0.0
        return out

    s_max = prob.max_fraction(direction)
    out["max_fraction"] = float(s_max)
    if per_state:
        per = []
        for i in range(n):
            e = np.zeros(n)
            e[i] = direction[i]
            per.append(float(prob.max_fraction(e)))
        out["per_state_fraction"] = per

    # what binds at the limit: a state bound in the transient, or (steady
    # state) inputs at their reserve edge / states at their acceptable deviation
    beyond = min(s_max * 1.02 + 1e-6, FRACTION_CEILING) * direction
    held, _, x, u = prob.solve(beyond)
    binding: List[str] = []
    stays, first_out = prob.transient(beyond)
    if held and not stays:
        binding.append(f"{names[first_out]} gets too close to its bounds during the transient"
                       if first_out is not None else "the transient diverges")
        x = u = None
    if u is not None:
        for j in range(m):
            for edge, label in ((prob.u_lo[j], "lower"), (prob.u_hi[j], "upper")):
                if np.isfinite(edge) and abs(u[j] - edge) <= 1e-3 * max(abs(edge), 1.0):
                    binding.append(f"{in_names[j]} at {label} limit")
    if x is not None:
        for i in range(n):
            if abs(abs(x[i] - prob.target[i]) - prob.a[i]) <= 1e-3 * prob.a[i]:
                binding.append(f"{names[i]} at its acceptable deviation")
    out["binding"] = binding[:6]
    if s_max >= FRACTION_CEILING:
        out["message"] = (
            f"Even 100% of each state's range per second can be held within the acceptable "
            f"deviation of the target (keeping {100 * (1 - reserve):.0f}% of each actuator's range "
            f"in reserve) and stays well inside the state bounds, so the slider's own range is the only "
            f"limit." + (f" {prob.transient_note}" if prob.transient_note else ""))
        out["binding"] = []
        return out
    transient = bool(binding) and "transient" in binding[0]
    out["message"] = (
        (f"At most {100 * s_max:.3g}% of each state's range per second: beyond that the push, "
         f"arriving as a step, carries the plant more than {100 * TRANSIENT_MARGIN:.0f}% of the way to "
         f"its declared state bounds before a controller can catch it (checked from the initial "
         f"state under a reference LQR)"
         if transient else
         f"At most {100 * s_max:.3g}% of each state's range per second: beyond that no admissible "
         f"input holds the plant within the acceptable deviation of its target "
         f"(keeping {100 * (1 - reserve):.0f}% of each actuator's range in reserve)")
        + (f" -- what runs out first: {', '.join(binding[:3])}." if binding else ".")
        + (f" {prob.transient_note}" if prob.transient_note else "")
    )
    return out
