"""
Which states may be given their own trajectory -- and, for the ones that may
not, what their trajectory has to be.

1. ``reference_authority`` -- the gate. An under-actuated plant cannot make
   every state follow an independently chosen trajectory: a cart-pole has one
   input, so once the cart is told where to go, the pendulum's motion is
   decided by the physics. The test is right-invertibility of the linearised
   plant from the inputs to the states the user wants to move,

       G_S(s) = C_S (sI - A)^-1 B ,      rank_normal G_S = |S|  required,

   where C_S picks the moving states (a velocity whose position also moves
   is the same degree of freedom and is not counted twice). When the normal
   rank is short, almost no pair of trajectories can be produced by any
   input -- not with any tuning -- so the reference is refused rather than
   left to fail as "bad tuning".

2. ``FeasibleReference`` -- the opt-in way out, in the spirit of
   backstepping: the user names ONE primary state and where it should go;
   every other state gets the "virtual reference" it must follow for the
   primary to get there while the whole plant stays stable. Computed, not
   guessed: the steady state the primary asks for is solved from
   f(x, u) = 0 (with x_primary fixed), a stabilising LQ tracker is designed on the
   linearisation there, and the full nonlinear model is simulated under the
   LQ tracker with preview -- feedback plus the feedforward that anticipates
   the known command -- (inputs clipped to their bounds) from the actual
   initial state. The
   result is a trajectory of the model itself, so the MPC is asked to track
   something the plant can actually do, and it ends exactly at the requested
   value of the primary state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..mpc.jacobian import linearize

#: A state whose reference spans less than this (relative to its own size)
#: is treated as constant.
MOVING_TOL = 1e-9

#: Relative singular-value threshold for the normal rank of G_S(s).
RANK_TOL = 1e-7

#: Weight multipliers on the primary state the generator's LQ tracker tries,
#: gentle to aggressive. The first that stays inside the bounds and meets
#: SETTLE_FRACTION (step) / SIN_TRACK_TOL (sin) is kept; if none does, the
#: one that comes closest.
PRIMARY_WEIGHTS = (10.0, 100.0, 1e3, 1e4)

#: Default time for the primary state to reach its new value (~98%).
DEFAULT_APPROACH_TIME = 2.0

#: The generator retries with a slower approach (x2 each time) this often
#: when the simulated trajectory diverges or leaves the state bounds.
APPROACH_RETRIES = 3

#: A step is "reached" within this fraction of the requested change.
SETTLE_FRACTION = 0.02

#: A sinusoid is "tracked" when the primary's RMS error over the second half
#: is below this fraction of the amplitude.
SIN_TRACK_TOL = 0.02

#: A steady state is accepted when every derivative is below this fraction
#: of its state's own scale per second.
EQUILIBRIUM_TOL = 1e-5


# =============================================================================
# shared helpers
# =============================================================================
def _names(dynamics) -> List[str]:
    names = list(getattr(dynamics, "state_names", None) or [])
    return names if len(names) == dynamics.n_states else [f"x{i}" for i in range(dynamics.n_states)]


def _input_bounds(dynamics, m):
    b = dynamics.get_input_bounds()
    if b is None:
        return np.full(m, -np.inf), np.full(m, np.inf)
    lo = np.asarray(b[0], dtype=float).ravel()
    hi = np.asarray(b[1], dtype=float).ravel()
    return np.where(np.isfinite(lo), lo, -np.inf), np.where(np.isfinite(hi), hi, np.inf)


def state_scale(dynamics) -> np.ndarray:
    """Each state's own size: half its declared span, else its distance
    from the target / the target's size, else 1."""
    n = dynamics.n_states
    out = np.zeros(n)
    b = dynamics.get_state_bounds()
    if b is not None:
        span = np.abs(np.asarray(b[1], dtype=float) - np.asarray(b[0], dtype=float))
        out = np.where(np.isfinite(span) & (span > 0), 0.5 * span, 0.0)
    x0 = np.asarray(dynamics.config.default_initial_state, dtype=float)
    tg = np.asarray(dynamics.config.default_target, dtype=float)
    own = np.maximum(np.abs(x0 - tg), np.abs(tg))
    out = np.where(out > 0, out, own)
    return np.where(out > 1e-12, out, 1.0)


def trim_input(dynamics, x, u_guess=None) -> np.ndarray:
    """The input that comes closest to holding ``x`` at rest (bounded
    least squares on f(x, u)); the plugin's own equilibrium input seeds it."""
    from scipy.optimize import least_squares

    m = dynamics.n_inputs
    u0 = np.asarray(u_guess if u_guess is not None else dynamics.get_equilibrium_input(),
                    dtype=float).ravel()
    lo, hi = _input_bounds(dynamics, m)
    u0 = np.clip(u0, np.where(np.isfinite(lo), lo, -1e300), np.where(np.isfinite(hi), hi, 1e300))
    try:
        sol = least_squares(lambda u: np.asarray(dynamics.dynamics(np.asarray(x, float), u), float),
                            u0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)
        return np.asarray(sol.x, dtype=float)
    except Exception:  # noqa: BLE001
        return u0


def _linear_model(dynamics, x=None, u=None):
    x = np.asarray(dynamics.config.default_target if x is None else x, dtype=float)
    u = trim_input(dynamics, x) if u is None else np.asarray(u, dtype=float)
    A, B = linearize(dynamics_fn=dynamics.dynamics, x=x, u=u,
                     torch_dynamics_fn=getattr(dynamics, "dynamics_torch", None))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float).reshape(dynamics.n_states, dynamics.n_inputs)
    return A, B, u


def rate_to_position(dynamics, A, B) -> Dict[int, int]:
    """{velocity state: its position state}, from the Setup Agent's derivative
    pairs and from the linearisation's kinematic rows (x_i' = +/- x_j)."""
    from .dynamics_validator import _kinematic_rate_pairs

    out = dict(_kinematic_rate_pairs(A, B))
    for pos_i, vel_j in (getattr(dynamics.config, "derivative_pairs", None) or []):
        out.setdefault(int(vel_j), int(pos_i))
    return out


def moving_states(reference, scale=None) -> List[int]:
    """Indices of the states whose reference changes over time."""
    ref = np.asarray(reference, dtype=float)
    if ref.ndim != 2 or ref.shape[0] < 2:
        return []
    size = np.maximum(np.max(np.abs(ref), axis=0), 1.0) if scale is None else np.asarray(scale, float)
    return [int(i) for i in np.flatnonzero(np.ptp(ref, axis=0) > MOVING_TOL * size)]


def _normal_rank(A, B, rows: Sequence[int]) -> int:
    """Normal rank of C_S (sI - A)^-1 B, sampled at a few complex s scaled
    to the plant's own speed (an eigenvalue is hit with probability zero)."""
    rows = list(rows)
    if not rows:
        return 0
    n = A.shape[0]
    speed = max(1.0, float(np.max(np.abs(np.linalg.eigvals(A)))) if n else 1.0)
    best = 0
    for s in (0.37 + 1.13j, 1.71 + 0.41j, -0.63 + 2.29j, 0.11 + 0.07j):
        try:
            G = np.linalg.solve(s * speed * np.eye(n) - A, B)[rows, :]
        except np.linalg.LinAlgError:
            continue
        norms = np.linalg.norm(G, axis=1)
        if not np.any(norms > 0):
            continue
        G = G[norms > 0] / norms[norms > 0, None]
        sv = np.linalg.svd(G, compute_uv=False)
        best = max(best, int(np.sum(sv > RANK_TOL * sv[0])))
    return best


# =============================================================================
# 1. the gate
# =============================================================================
def reference_authority(dynamics, reference=None, moving: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """May these states follow their own trajectories at the same time?

    Give either the reference array (its time-varying columns are the
    request) or ``moving`` directly. Returns a plain dict:

    allowed           -- False when the request is physically impossible
    requested         -- names of the independent degrees of freedom asked
                         to move (a velocity moving with its position is
                         folded into the position)
    rank              -- how many of them can move independently
    max_independent   -- how many states this plant can move independently
                         at all (normal rank over every position)
    keep_options      -- a largest subset of the request that IS allowed
    title, message    -- the user-facing explanation (English)
    """
    names = _names(dynamics)
    n, m = dynamics.n_states, dynamics.n_inputs
    if moving is None:
        moving = moving_states(reference, None) if reference is not None else []
    moving = sorted(set(int(i) for i in moving))
    A, B, _ = _linear_model(dynamics)
    r2p = rate_to_position(dynamics, A, B)
    request = [i for i in moving if not (i in r2p and r2p[i] in moving)]
    positions = [i for i in range(n) if i not in r2p]
    rank_b = int(np.linalg.matrix_rank(B, tol=1e-8 * max(float(np.max(np.abs(B))), 1e-300)))
    rank = _normal_rank(A, B, request)
    max_ind = _normal_rank(A, B, positions) if positions else rank_b

    keep: List[int] = []
    for i in request:
        if _normal_rank(A, B, keep + [i]) > len(keep):
            keep.append(i)

    out: Dict[str, Any] = {
        "allowed": rank >= len(request),
        "requested": [names[i] for i in request],
        "requested_idx": request,
        "rank": int(rank),
        "max_independent": int(max_ind),
        "n_inputs": int(m),
        "rank_B": rank_b,
        "keep_options": [names[i] for i in keep],
        "title": "",
        "message": "",
    }
    if out["allowed"]:
        return out

    req_txt = ", ".join(out["requested"])
    blocked = [names[i] for i in request if i not in keep]
    out["title"] = "This reference cannot be followed by this plant"
    out["message"] = (
        f"You gave {len(request)} states their own trajectory: {req_txt}. This plant has {m} "
        f"input{'s' if m != 1 else ''} ({rank_b} independent), and from those inputs only {rank} of "
        f"these states can be steered independently -- the motion of the rest is decided by the "
        f"plant's own dynamics. For example, with a trajectory on {', '.join(out['keep_options'])}, "
        f"the plant itself dictates what {', '.join(blocked)} must do; a second, unrelated "
        f"trajectory for {'it' if len(blocked) == 1 else 'them'} has no input that can produce it, "
        f"whatever the tuning. In mathematical terms, the transfer matrix from the inputs to these "
        f"states, G(s) = C (sI - A)^-1 B, has rank {rank} < {len(request)}: the plant is not "
        f"right-invertible for this choice of outputs.\n\n"
        f"What you can do: keep a trajectory on at most {rank} of them (for instance "
        f"{', '.join(out['keep_options'])}) and leave the others on Regulation, or turn on "
        f"'Feasible reference (backstepping-like)' so AgentMPC computes the trajectory the other "
        f"states must follow."
    )
    return out


# =============================================================================
# 2. the backstepping-like feasible reference
# =============================================================================
def equilibrium_at(dynamics, primary: int, value: float, x_guess=None, u_guess=None) -> Dict[str, Any]:
    """The steady state with x[primary] = value closest to ``x_guess``
    (default: the plugin's target): f(x, u) = 0 solved for the other states
    and the inputs. ``ok`` is False when no such steady state exists -- the
    plant cannot rest with the primary state there."""
    from scipy.optimize import least_squares

    n, m = dynamics.n_states, dynamics.n_inputs
    xs = state_scale(dynamics)
    xg = np.asarray(dynamics.config.default_target if x_guess is None else x_guess, dtype=float).copy()
    xg[primary] = value
    ug = trim_input(dynamics, xg, u_guess)
    free = [i for i in range(n) if i != primary]
    lo_u, hi_u = _input_bounds(dynamics, m)

    def unpack(z):
        x = xg.copy()
        x[free] = z[: len(free)]
        return x, z[len(free):]

    def resid(z, w):
        x, u = unpack(z)
        f = np.asarray(dynamics.dynamics(x, u), dtype=float) / xs
        return np.concatenate([f, w * (x[free] - xg[free]) / xs[free]])

    z0 = np.concatenate([xg[free], ug])
    lb = np.concatenate([np.full(len(free), -np.inf), lo_u])
    ub = np.concatenate([np.full(len(free), np.inf), hi_u])
    z0 = np.clip(z0, np.where(np.isfinite(lb), lb, -1e300), np.where(np.isfinite(ub), ub, 1e300))
    try:
        z = least_squares(resid, z0, args=(1e-3,), bounds=(lb, ub), max_nfev=400).x
        # polish onto the manifold: the regulariser only picked the point
        z = least_squares(lambda zz: resid(zz, 0.0)[:n], z, bounds=(lb, ub),
                          xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=400).x
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "x": xg, "u": ug, "residual": np.inf, "error": str(exc)}
    x, u = unpack(z)
    res = float(np.max(np.abs(np.asarray(dynamics.dynamics(x, u), dtype=float) / xs)))
    return {"ok": bool(np.isfinite(res) and res <= EQUILIBRIUM_TOL), "x": x, "u": u, "residual": res}


def _lq_tracker(Ad, Bd, Qd, Rd):
    """Discrete LQ tracker: (K, G B', A - B K, Q) for
    u_k = -K x_k - G B' s_{k+1},  s_k = (A - B K)' s_{k+1} - Q r_k,
    the optimal input for sum (x - r)'Q(x - r) + u'Ru when the whole future
    r is known -- the feedforward part anticipates the command, so the
    trajectory does not lag it the way pure state feedback does."""
    from scipy.linalg import solve_discrete_are

    P = solve_discrete_are(Ad, Bd, Qd, Rd)
    G = np.linalg.inv(Rd + Bd.T @ P @ Bd)
    K = G @ Bd.T @ P @ Ad
    Acl = Ad - Bd @ K
    if np.max(np.abs(np.linalg.eigvals(Acl))) >= 1.0:
        raise np.linalg.LinAlgError("the LQ design does not stabilise the linearisation")
    return K, G @ Bd.T, Acl, Qd


def _rk4(f, x, u, h):
    k1 = f(x, u)
    k2 = f(x + 0.5 * h * k1, u)
    k3 = f(x + 0.5 * h * k2, u)
    k4 = f(x + h * k3, u)
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


class FeasibleReference:
    """A reference for an under-actuated plant built around ONE primary state.

    Callable with the custom-trajectory signature ``(dt, T, n_states,
    state_names) -> (n_steps, n_states)``, so it plugs into
    ``cfg.data.custom_trajectory_fn`` unchanged. ``model`` is the model the
    trajectory is computed on (the controller's nominal model);
    ``start_source`` is whatever object's ``config.default_initial_state``
    holds the run's real starting point at call time (default: ``model``).

    shape "step": the primary goes to ``goal`` within ``approach_time``.
    shape "sin":  the primary follows target + amplitude * sin(2 pi f t).
    """

    def __init__(self, model, primary: int, goal: Optional[float] = None, shape: str = "step",
                 amplitude: float = 0.5, frequency: float = 0.2,
                 approach_time: float = DEFAULT_APPROACH_TIME, start_source=None):
        if shape not in ("step", "sin"):
            raise ValueError(f"shape must be 'step' or 'sin', not {shape!r}")
        self.model = model
        self.primary = int(primary)
        tgt = np.asarray(model.config.default_target, dtype=float)
        self.goal = float(tgt[self.primary] if goal is None else goal)
        self.shape = shape
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.approach_time = float(approach_time)
        self.start_source = start_source if start_source is not None else model
        self.last: Dict[str, Any] = {}
        self._cache: Dict[Any, np.ndarray] = {}

    # ---- the steady states the primary asks for ---------------------------
    def _manifold(self):
        """(values, X_eq, U_eq): steady states along the primary's range."""
        tgt = np.asarray(self.model.config.default_target, dtype=float)
        if self.shape == "step":
            values = np.array([self.goal])
        else:
            c = float(tgt[self.primary])
            values = np.linspace(c - abs(self.amplitude), c + abs(self.amplitude), 21)
        X, U, worst = [], [], 0.0
        # continuation from the centre outwards keeps every solve near the last
        order = np.argsort(np.abs(values - (values[0] if len(values) == 1 else np.median(values))))
        sol = {}
        x_prev, u_prev = None, None
        for k in order:
            eq = equilibrium_at(self.model, self.primary, float(values[k]), x_prev, u_prev)
            if not eq["ok"]:
                return values, None, None, eq
            sol[k] = eq
            x_prev, u_prev = eq["x"], eq["u"]
            worst = max(worst, eq["residual"])
        X = np.array([sol[k]["x"] for k in range(len(values))])
        U = np.array([sol[k]["u"] for k in range(len(values))])
        return values, X, U, {"ok": True, "residual": worst}

    def generate(self, dt: float, T: float) -> Dict[str, Any]:
        model = self.model
        n, m = model.n_states, model.n_inputs
        names = _names(model)
        p = self.primary
        steps = max(int(round(T / dt)), 1)
        t = np.arange(steps) * dt
        x0 = np.asarray(self.start_source.config.default_initial_state, dtype=float).copy()

        values, X, U, eq = self._manifold()
        if X is None:
            return {"ok": False, "reference": None, "t": t,
                    "message": (f"The plant has no steady state with {names[p]} = {eq['x'][p]:.4g}: "
                                f"it cannot rest there with any admissible input (left-over rate "
                                f"{eq['residual']:.2g} of the state scale per second). Pick a value "
                                f"of {names[p]} the plant can actually hold, or another primary state.")}

        centre = X[len(values) // 2] if self.shape == "sin" else X[0]
        u_c = U[len(values) // 2] if self.shape == "sin" else U[0]
        A, B, _ = _linear_model(model, centre, u_c)
        from scipy.linalg import expm

        from .dynamics_validator import _acceptable_from_bounds

        # zero-order-hold discretisation of the linearisation at dt
        M = np.zeros((n + m, n + m))
        M[:n, :n], M[:n, n:] = A, B
        E = expm(M * dt)
        Ad, Bd = E[:n, :n], E[:n, n:]
        scale = state_scale(model)
        a = _acceptable_from_bounds(model.get_state_bounds(), n, np.abs(x0 - centre))
        u_max = _acceptable_from_bounds(model.get_input_bounds(), m, np.abs(U).max(axis=0))
        Rd = np.diag(1.0 / u_max ** 2) * dt
        designs, err = [], None
        for w in PRIMARY_WEIGHTS:
            q = 1.0 / a ** 2
            q[p] *= w
            try:
                designs.append((w, _lq_tracker(Ad, Bd, np.diag(q) * dt, Rd)))
            except Exception as exc:  # noqa: BLE001
                err = exc
        if not designs:
            return {"ok": False, "reference": None, "t": t,
                    "message": f"No stabilising feedback exists around this steady state ({err}): the "
                               f"linearised plant is not stabilisable there."}
        lo_u, hi_u = _input_bounds(model, m)
        sb = model.get_state_bounds()
        f = lambda x, u: np.asarray(model.dynamics(x, u), dtype=float)  # noqa: E731
        fastest = float(np.max(np.abs(np.linalg.eigvals(A)))) if n else 0.0
        n_prev = steps                     # preview beyond T, so the end is not rushed

        def target_path(r):
            """Full-state target along the command: the steady state each
            command value asks for (the primary set to the command itself)."""
            if len(values) == 1:
                xs = np.repeat(X[:1], len(r), axis=0)
            else:
                xs = np.column_stack([np.interp(r, values, X[:, i]) for i in range(n)])
            xs[:, p] = r
            return xs

        def simulate(design, r_full):
            K, G_Bt, Acl, Qd = design
            xr = target_path(r_full) - centre
            # backward pass of the LQ tracker: s_k = Acl' s_{k+1} - Qd xr_k,
            # ending in the steady value for a target held from there on
            s = -np.linalg.solve(np.eye(n) - Acl.T, Qd @ xr[-1])
            ff = np.empty((steps, m))
            for k in range(len(r_full) - 1, -1, -1):
                if k < steps:
                    ff[k] = -G_Bt @ s          # uses s_{k+1}
                s = Acl.T @ s - Qd @ xr[k]
            rate = max(fastest, float(np.max(np.abs(np.log(np.maximum(
                np.abs(np.linalg.eigvals(Acl)), 1e-12))))) / dt)
            sub = max(1, int(np.ceil(dt * rate / 0.5)))
            h = dt / sub
            ref = np.full((steps, n), np.nan)
            x = x0.copy()
            for k in range(steps):
                ref[k] = x
                u = np.clip(u_c - K @ (x - centre) + ff[k], lo_u, hi_u)
                for _ in range(sub):
                    x = _rk4(f, x, u, h)
                if not np.all(np.isfinite(x)) or np.max(np.abs(x - centre) / scale) > 1e3:
                    return ref, False
            return ref, True

        def in_bounds(ref):
            if sb is None:
                return True
            lo_x, hi_x = np.asarray(sb[0], float), np.asarray(sb[1], float)
            return bool(np.all((ref >= lo_x - 1e-9) | ~np.isfinite(lo_x))
                        and np.all((ref <= hi_x + 1e-9) | ~np.isfinite(hi_x)))

        def score(ref, r):
            if self.shape == "step":
                # how far the WHOLE plant still is from resting at the goal
                return float(np.max(np.abs(ref[-1] - np.where(np.arange(n) == p, self.goal, X[0])) / scale))
            half = slice(steps // 2, None)
            return float(np.sqrt(np.mean((ref[half, p] - r[half]) ** 2)) / max(abs(self.amplitude), 1e-12))

        approach = self.approach_time
        best: Dict[str, Any] = {}
        good_enough = (SIN_TRACK_TOL if self.shape == "sin" else
                       SETTLE_FRACTION * max(abs(self.goal - x0[p]) / scale[p], 0.05))
        for attempt in range(APPROACH_RETRIES + 1):
            tau_f = max(approach / 4.0, 1e-6)
            t_full = np.arange(steps + n_prev) * dt
            if self.shape == "step":
                # first-order prefilter: the primary command eases from x0 to the goal
                r_full = self.goal + (x0[p] - self.goal) * np.exp(-t_full / tau_f)
            else:
                c = float(np.asarray(model.config.default_target, dtype=float)[p])
                ramp = 1.0 - np.exp(-t_full / tau_f)
                r_full = c + (x0[p] - c) * np.exp(-t_full / tau_f) + ramp * self.amplitude * np.sin(
                    2 * np.pi * self.frequency * t_full)
            r = r_full[:steps]
            for w, design in designs:
                K = design[0]
                ref, finite = simulate(design, r_full)
                cand = {"ok": bool(finite and in_bounds(ref)), "reference": ref, "t": t,
                        "primary_command": r, "approach_time": approach, "attempts": attempt + 1,
                        "in_bounds": bool(finite and in_bounds(ref)), "finite": finite,
                        "score": score(ref, r) if finite else np.inf, "primary_weight": w,
                        "steady_state": centre, "gain": K,
                        "final_error": abs(float(ref[-1, p] - r[-1])) if finite else np.inf}
                rank = (not cand["ok"], not cand["finite"], cand["score"])
                if not best or rank < (not best["ok"], not best["finite"], best["score"]):
                    best = cand
                if cand["ok"] and cand["score"] <= good_enough:
                    # the designs go from gentle to aggressive: the first one
                    # that does the job asks the least of the actuators
                    best = cand
                    break
            if best["ok"]:
                break
            approach *= 2.0
        result = best
        result["settled"] = bool(result["ok"] and (
            self.shape == "sin" or result["score"] <= good_enough))

        if result["ok"]:
            result["message"] = (
                f"{names[p]} is the primary state; the other states follow the trajectory they must "
                f"take for it to get there, computed on the model under a stabilising LQ tracker with preview"
                + (f" (approach time raised to {result['approach_time']:.3g} s to stay stable and "
                   f"inside the bounds)" if result["attempts"] > 1 else "") + "."
                + ("" if result["settled"] else
                   " The plant is still moving at the end of the simulation -- it gets there "
                   "slowly; a longer simulation shows the rest."))
        elif result["finite"]:
            result["message"] = (f"The best trajectory found leaves the declared state bounds, even "
                                 f"with an approach time of {result['approach_time']:.3g} s. Try a "
                                 f"smaller change of {names[p]}, a longer simulation, or wider bounds.")
        else:
            result["message"] = (f"The plant could not be steered to {names[p]} = {self.goal:.4g} "
                                 f"from its initial state without diverging, even with an approach "
                                 f"time of {result['approach_time']:.3g} s.")
        self.last = result
        return result

    def __call__(self, dt, T, n_states=None, state_names=None):
        key = (round(float(dt), 12), round(float(T), 9),
               tuple(np.round(np.asarray(self.start_source.config.default_initial_state, float), 12)))
        if key not in self._cache:
            res = self.generate(float(dt), float(T))
            ref = res["reference"]
            if ref is None or not np.all(np.isfinite(ref)):
                raise ValueError(res["message"])
            self._cache = {key: ref}
        return self._cache[key].copy()
