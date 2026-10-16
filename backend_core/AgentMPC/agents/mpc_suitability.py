"""
================================================================================
agents/mpc_suitability.py
================================================================================
Deterministic checks the Diagnostics assistant raises BEFORE and AROUND a
tuning run -- the engine behind its pop-ups.

Two questions, each answered from the plant itself, with no LLM:

  assess_mpc_suitability  -- can this plant be controlled by THIS MPC at all?
                             (stabilisability, the forward-Euler prediction at
                             the chosen dt, input authority at the target,
                             nonlinearity, stiffness, scaling, actuation)
  assess_reference        -- can this plant actually FOLLOW the reference the
                             user picked? (dynamic feasibility sample by
                             sample, derivative-pair consistency, actuator
                             authority along the way)

Why deterministic: these run on every relevant widget change, so they must be
instant, free, and never wrong in a way an LLM can be. The LLM is brought in
only when the user asks the assistant about a specific finding, to explain it
and propose a fix in context.

Every check returns plain-dict FINDINGS of one shape -- severity, code, title,
message, suggestion, where -- so any screen can surface any of them the same
way, and the Diagnostics tab can collect them into one inbox. JSON-friendly by
construction (no numpy scalars), so they can also travel into a prompt or an
API response unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..mpc.jacobian import linearize

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}

#: Forward Euler is only stable for a mode lambda when |1 + dt*lambda| < 1.
#: For accuracy -- not merely stability -- the fastest mode should be sampled
#: much more finely than that; 0.1/|lambda| keeps the per-step Euler error of
#: the fastest mode to roughly a percent.
EULER_ACCURACY_FACTOR = 0.1

#: A mode slower than this many sample periods is irrelevant to the choice of
#: dt, so no dt bound is derived from it.
RELEVANT_MODE_HORIZON = 1000.0

#: A reference sample is "infeasible" when no input brings the plant's
#: derivative within this fraction of the reference's own rate of change.
REFERENCE_RESIDUAL_TOL = 0.05

# -- Every other threshold the checks below use, named. The Diagnostics tab's
# "How the assistant decides" panel (agents/diagnostics_rules.py) reads these
# same constants, so the rules a user is shown can never drift from the rules
# the code actually applies.

#: Steps over which forward Euler's inflation of an undamped mode is judged.
EULER_GROWTH_HORIZON = 20
#: |1 + dt*lambda|^horizon above which the inflation is a warning / an error.
EULER_GROWTH_WARN = 1.5
EULER_GROWTH_ERROR = 10.0
#: |lambda|_max * dt above (this x EULER_ACCURACY_FACTOR) is "coarse".
EULER_COARSE_MULTIPLE = 3.0
#: Fastest/slowest mode magnitude ratio above which the plant is "stiff".
STIFFNESS_RATIO = 1e3
#: Nonlinearity probe: how many points, how far out (fraction of each state's
#: range), and the median relative error that counts as mild / strong.
NONLINEAR_SAMPLES = 40
NONLINEAR_SPREAD = 0.25
NONLINEAR_INFO = 0.1
NONLINEAR_WARN = 0.5
#: Ratio between the widest and narrowest state range that counts as badly scaled.
SCALE_RATIO = 1e3
#: Relative disagreement between a velocity reference and the rate of change
#: of its position reference above which the pair is flagged.
DERIVATIVE_MISMATCH_TOL = 0.2


def _finding(severity: str, code: str, title: str, message: str, suggestion: str,
             where: str, **data: Any) -> Dict[str, Any]:
    return {"severity": severity, "code": code, "title": title, "message": message,
            "suggestion": suggestion, "where": where,
            "data": {k: _plain(v) for k, v in data.items()}}


def _plain(value):
    """numpy -> plain Python, recursively, so findings serialise as-is."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    return value


def _names(dynamics) -> List[str]:
    n = dynamics.n_states
    return list(getattr(dynamics, "state_names", None) or [f"x{i}" for i in range(n)])


def _input_names(dynamics) -> List[str]:
    m = dynamics.n_inputs
    return list(getattr(dynamics, "input_names", None) or [f"u{j}" for j in range(m)])


def _jacobians(dynamics, x, u):
    A, B = linearize(dynamics_fn=dynamics.dynamics, x=np.asarray(x, dtype=float),
                     u=np.asarray(u, dtype=float),
                     torch_dynamics_fn=getattr(dynamics, "dynamics_torch", None))
    return (np.atleast_2d(np.asarray(A, dtype=float)),
            np.asarray(B, dtype=float).reshape(dynamics.n_states, dynamics.n_inputs))


def sort_findings(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(findings, key=lambda f: SEVERITY_ORDER.get(f.get("severity"), 9))


# =============================================================================
# 1. Is this plant suitable for this MPC at all?
# =============================================================================
def assess_mpc_suitability(dynamics, dt: Optional[float] = None,
                            rank_tol: float = 1e-8,
                            horizon_steps: int = EULER_GROWTH_HORIZON) -> Dict[str, Any]:
    """Verdict on whether the loaded plant can be controlled by this MPC.

    Returns ``{"verdict": "suitable" | "caveats" | "not_suitable",
    "findings": [...], "facts": {...}}``. Any error makes it not_suitable, any
    warning makes it "suitable with caveats".

    Everything is evaluated at the plant's declared target and equilibrium
    input -- the operating point the controller will spend its time near.
    """
    where = "dynamics"
    findings: List[Dict[str, Any]] = []
    names, in_names = _names(dynamics), _input_names(dynamics)
    n, m = dynamics.n_states, dynamics.n_inputs
    x0 = np.asarray(dynamics.config.default_initial_state, dtype=float)
    target = np.asarray(dynamics.config.default_target, dtype=float)
    u_eq = np.asarray(dynamics.get_equilibrium_input(), dtype=float).ravel()
    facts: Dict[str, Any] = {"n_states": n, "n_inputs": m}

    # ---- 0. the dynamics must be evaluable where the run will start and end
    for label, x in (("initial state", x0), ("target", target)):
        try:
            fx = np.asarray(dynamics.dynamics(x, u_eq), dtype=float)
            ok = bool(np.all(np.isfinite(fx)))
        except Exception as exc:  # noqa: BLE001
            ok, fx = False, None
            label = f"{label} ({exc})"
        if not ok:
            findings.append(_finding(
                "error", "dynamics_not_finite",
                f"The dynamics cannot be evaluated at the {label}",
                "f(x, u) returned NaN/Inf or raised. Nothing downstream -- linearisation, "
                "prediction, the QP -- can work from here.",
                "Look for a division by zero, a singular matrix inverse, or a sqrt/log of an "
                "invalid argument at that state. A NaN here is also what later surfaces as a "
                "misleading 'QP solve failed'.",
                where))
            return {"verdict": "not_suitable", "findings": findings, "facts": facts}

    try:
        A_c, B_c = _jacobians(dynamics, target, u_eq)
    except Exception as exc:  # noqa: BLE001
        findings.append(_finding(
            "error", "linearisation_failed", "The plant could not be linearised at the target",
            f"Taking the Jacobian failed: {exc}.",
            "The MPC re-linearises the plant at every step, so this must work. Check that "
            "dynamics(x, u) accepts numpy arrays of the declared sizes.", where))
        return {"verdict": "not_suitable", "findings": findings, "facts": facts}

    if not (np.all(np.isfinite(A_c)) and np.all(np.isfinite(B_c))):
        findings.append(_finding(
            "error", "jacobian_not_finite", "The linearised model contains NaN/Inf",
            "The Jacobian of the dynamics at the target is not finite.",
            "Usually a singularity exactly at the operating point (e.g. dividing by a state "
            "that is zero there). Reformulate the model or move the target slightly.", where))
        return {"verdict": "not_suitable", "findings": findings, "facts": facts}

    eig = np.linalg.eigvals(A_c)
    facts["eigenvalues"] = [[float(z.real), float(z.imag)] for z in eig]

    # ---- 1. stabilisability (PBH test) ------------------------------------
    # A mode with Re(lambda) >= 0 that no input reaches cannot be stabilised by
    # MPC or by any other controller. This is the one genuine "not suitable".
    unreachable_unstable = []
    for lam in eig:
        if lam.real >= -1e-9:
            pbh = np.hstack([lam * np.eye(n) - A_c, B_c.astype(complex)])
            if np.linalg.matrix_rank(pbh, tol=rank_tol) < n:
                unreachable_unstable.append(lam)
    if unreachable_unstable:
        lam_txt = ", ".join(f"{z.real:+.4g}{z.imag:+.4g}j" for z in unreachable_unstable)
        findings.append(_finding(
            "error", "not_stabilisable", "The plant is not stabilisable",
            f"Mode(s) {lam_txt} are unstable or marginal and cannot be influenced by any input. "
            "No controller -- MPC or otherwise -- can bring them to rest.",
            "Check the input matrix: an actuator that should drive this mode may be missing from "
            "the model, or the model may declare the wrong inputs. If the mode is a pure "
            "integrator of something you do not care about, remove that state.",
            where, modes=[[float(z.real), float(z.imag)] for z in unreachable_unstable]))

    # ---- 2. controllability (weaker) ----------------------------------------
    blocks, term = [B_c], B_c
    for _ in range(n - 1):
        term = A_c @ term
        blocks.append(term)
    ctrb_rank = int(np.linalg.matrix_rank(np.hstack(blocks), tol=rank_tol))
    facts["controllability_rank"] = ctrb_rank
    if ctrb_rank < n and not unreachable_unstable:
        findings.append(_finding(
            "warning", "not_fully_controllable", "Some directions cannot be steered",
            f"Controllability rank {ctrb_rank} of {n}. The uncontrollable modes are stable, so "
            "they will settle on their own -- but they cannot be steered anywhere.",
            "Do not give those states a target the plant cannot reach on its own; the "
            "reference check will flag it if you do.", where, controllability_rank=ctrb_rank))

    # ---- 3. open-loop instability (fine, but worth knowing) ----------------
    unstable = [z for z in eig if z.real > 1e-9]
    if unstable:
        fastest = max(z.real for z in unstable)
        findings.append(_finding(
            "info", "open_loop_unstable", "The plant is open-loop unstable",
            f"{len(unstable)} unstable mode(s); the fastest doubles every "
            f"{np.log(2) / fastest:.3g}s. MPC can stabilise it, but only through feedback -- "
            "a solver failure or a saturated input lets it run away.",
            f"Keep dt well below {np.log(2) / fastest:.3g}s and prefer a prediction horizon "
            "long enough to see the instability coming (Np x dt a few times that doubling time).",
            where, fastest_unstable_rate=float(fastest)))

    # ---- 4. input authority at the target ---------------------------------
    bounds = dynamics.get_input_bounds()
    if bounds is not None:
        lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
        outside = [j for j in range(m) if u_eq[j] < lo[j] - 1e-12 or u_eq[j] > hi[j] + 1e-12]
        if outside:
            findings.append(_finding(
                "error", "equilibrium_input_out_of_bounds",
                "Holding the target needs more input than the actuators allow",
                "The equilibrium input is outside the declared input bounds on "
                + ", ".join(f"{in_names[j]} (needs {u_eq[j]:.4g}, allowed "
                            f"[{lo[j]:.4g}, {hi[j]:.4g}])" for j in outside) + ".",
                "Widen those input bounds, or choose a target that can be held within them.",
                where))

    # ---- 5. the controller's forward-Euler prediction at this dt -----------
    abs_eig = np.abs(eig)
    fastest_mag = float(abs_eig.max()) if abs_eig.size else 0.0
    facts["fastest_mode_rate"] = fastest_mag
    # Only for a mode that matters at the time scales being sampled. An
    # integrator chain has eigenvalues that are zero up to rounding (1.3e-5
    # on the quadcopter), and "Euler stays accurate up to 7600 s" is true but
    # no guidance at all (campaign bug 13).
    if fastest_mag > 0 and 1.0 / fastest_mag < RELEVANT_MODE_HORIZON * (dt or 0.02):
        facts["recommended_dt_max"] = EULER_ACCURACY_FACTOR / fastest_mag
    if dt and fastest_mag > 0:
        # Exact forward-Euler stability bound for a stable complex mode:
        # |1 + dt*lambda| < 1  <=>  dt < -2 Re(lambda) / |lambda|^2
        broken = [z for z in eig if z.real < -1e-9 and abs(1 + dt * z) >= 1.0]
        if broken:
            dt_limit = min(-2 * z.real / abs(z) ** 2 for z in broken)
            findings.append(_finding(
                "error", "euler_unstable_at_dt",
                "At this dt the controller's own prediction is unstable",
                f"The MPC discretises with forward Euler. At dt = {dt:.4g}s, {len(broken)} STABLE "
                f"mode(s) of the plant become UNSTABLE in the prediction model, so the controller "
                f"is planning against a system that does not exist.",
                f"Use dt below {dt_limit:.3g}s (for accuracy, about "
                f"{EULER_ACCURACY_FACTOR / fastest_mag:.3g}s).",
                where, dt=dt, dt_stability_limit=float(dt_limit)))
        # An undamped oscillatory mode (Re = 0, Im != 0) is the case that
        # check misses: forward Euler has |1 + j*dt*w| > 1 for EVERY dt, so it
        # always inflates such a mode -- the only question is how fast. Over a
        # typical horizon a small per-step growth is harmless; a large one
        # means the controller is predicting an oscillation that explodes.
        marginal = [z for z in eig if abs(z.real) <= 1e-9 and abs(z.imag) > 1e-9]
        growth = max((abs(1 + dt * z) ** horizon_steps for z in marginal), default=1.0)
        if growth > EULER_GROWTH_WARN:
            w = max(abs(z.imag) for z in marginal)
            findings.append(_finding(
                "error" if growth > EULER_GROWTH_ERROR else "warning", "euler_inflates_oscillation",
                "At this dt the controller predicts an oscillation that grows",
                f"The plant has an undamped oscillatory mode ({w:.3g} rad/s). Forward Euler "
                f"always inflates such a mode, and at dt = {dt:.4g}s it grows by a factor of "
                f"{growth:.3g} over {horizon_steps} steps -- the controller would be planning "
                "against an oscillation that does not exist.",
                f"Use dt around {EULER_ACCURACY_FACTOR / w:.3g}s or less, where the inflation "
                "over a horizon stays negligible.",
                where, dt=dt, growth_over_horizon=float(growth)))
        elif fastest_mag * dt > EULER_COARSE_MULTIPLE * EULER_ACCURACY_FACTOR:
            findings.append(_finding(
                "warning", "euler_inaccurate_at_dt",
                "dt is coarse for the fastest mode",
                f"The fastest mode has rate {fastest_mag:.3g}/s, so dt = {dt:.4g}s takes "
                f"{fastest_mag * dt:.2f} of its time constant per step -- the Euler prediction "
                "will be noticeably wrong over a long horizon.",
                f"dt around {EULER_ACCURACY_FACTOR / fastest_mag:.3g}s or less would be accurate.",
                where, dt=dt))

    # ---- 6. stiffness ------------------------------------------------------
    nonzero = abs_eig[abs_eig > 1e-9]
    if nonzero.size >= 2:
        ratio = float(nonzero.max() / nonzero.min())
        facts["stiffness_ratio"] = ratio
        if ratio > STIFFNESS_RATIO:
            findings.append(_finding(
                "warning", "stiff_plant", "The plant is stiff",
                f"Its fastest and slowest modes differ by a factor of {ratio:.3g}. A dt fine "
                "enough for the fast mode needs a very long horizon to see the slow one.",
                "Expect a long Np at a small dt. If the fast mode is a well-damped actuator, "
                "modelling it as instantaneous often removes the stiffness.", where,
                stiffness_ratio=ratio))

    # ---- 7. how nonlinear is it across the operating envelope? ------------
    sbounds = dynamics.get_state_bounds()
    span = None
    if sbounds is not None:
        span = np.abs(np.asarray(sbounds[1], dtype=float) - np.asarray(sbounds[0], dtype=float))
        span = np.where(np.isfinite(span) & (span > 0), span, 1.0)
    else:
        span = np.maximum(np.abs(x0 - target), 1.0)
    rng = np.random.default_rng(0)
    f_target = np.asarray(dynamics.dynamics(target, u_eq), dtype=float)
    rel_errors = []
    for _ in range(NONLINEAR_SAMPLES):
        x = target + span * rng.uniform(-NONLINEAR_SPREAD, NONLINEAR_SPREAD, size=n)
        try:
            f_true = np.asarray(dynamics.dynamics(x, u_eq), dtype=float)
        except Exception:  # noqa: BLE001
            continue
        f_lin = f_target + A_c @ (x - target)
        denom = np.linalg.norm(f_true - f_target) + 1e-9
        rel_errors.append(float(np.linalg.norm(f_true - f_lin) / denom))
    if rel_errors:
        nl = float(np.median(rel_errors))
        facts["nonlinearity"] = nl
        if nl > NONLINEAR_WARN:
            findings.append(_finding(
                "warning", "strongly_nonlinear", "The plant is strongly nonlinear",
                f"Across a quarter of the state range around the target, the linear model is off "
                f"by {nl * 100:.0f}% (median). The controller re-linearises every step, which "
                "helps, but a long prediction horizon extrapolates one linearisation a long way.",
                "Keep the prediction window Np x dt moderate, and start close to the target where "
                "possible.", where, nonlinearity=nl))
        elif nl > NONLINEAR_INFO:
            findings.append(_finding(
                "info", "mildly_nonlinear", "The plant is mildly nonlinear",
                f"The linear model is off by {nl * 100:.0f}% (median) across the operating "
                "envelope -- comfortably within what per-step re-linearisation handles.",
                "Nothing to do.", where, nonlinearity=nl))

    # ---- 8. scaling -------------------------------------------------------
    if sbounds is not None and n > 1:
        finite = span[span > 0]
        if finite.size >= 2 and finite.max() / finite.min() > SCALE_RATIO:
            findings.append(_finding(
                "info", "badly_scaled", "The states differ wildly in scale",
                f"Their ranges differ by a factor of {finite.max() / finite.min():.3g}, which "
                "makes a hand-picked Q easy to get wrong by orders of magnitude.",
                "Start from the Setup Agent's role-based weights rather than uniform ones -- "
                "they normalise each state by its own range.", where))

    # ---- 9. actuation ------------------------------------------------------
    try:
        from .dynamics_validator import analyze_actuation
        act = analyze_actuation(dynamics, target=target)
        facts["actuation"] = {k: act[k] for k in ("classification", "rank_B",
                                                   "independently_holdable", "fully_controllable")}
        findings.append(_finding(
            "info", "actuation", f"Actuation: {act['classification']}",
            f"{m} input(s) for {n} states; at most {act['independently_holdable']} state value(s) "
            "can be held at independently chosen values.",
            "Give constant targets only to states you actually control; the others settle "
            "wherever the physics puts them.", where, **facts["actuation"]))
        if act.get("target_feasible") is False:
            findings.append(_finding(
                "error", "target_not_reachable", "The declared target is not a steady state",
                f"With the best possible input the state derivative still has magnitude "
                f"{act['target_residual']:.4g} at the target.",
                f"Target only the {act['independently_holdable']} state(s) you actually control.",
                where, unreachable_states=act.get("unreachable_states")))
    except Exception:  # noqa: BLE001 -- actuation is advisory here
        pass

    findings = sort_findings(findings)
    sev = {f["severity"] for f in findings}
    verdict = "not_suitable" if "error" in sev else ("caveats" if "warning" in sev else "suitable")
    return {"verdict": verdict, "findings": findings, "facts": _plain(facts)}


# =============================================================================
# 2. Can the plant follow the reference the user picked?
# =============================================================================
def _input_rank(dynamics, x, u) -> Optional[int]:
    """Independent input directions at (x, u): rank of the linearised B."""
    try:
        _, B = linearize(dynamics.dynamics, np.asarray(x, dtype=float), np.asarray(u, dtype=float))
        if not np.all(np.isfinite(B)):
            return None
        return int(np.linalg.matrix_rank(B, tol=1e-8 * max(float(np.max(np.abs(B))), 1e-300)))
    except Exception:  # noqa: BLE001
        return None


def assess_reference(dynamics, reference, dt: float, max_samples: int = 120,
                     where: str = "reference") -> List[Dict[str, Any]]:
    """Dynamic feasibility of a reference, sample by sample.

    For each sampled instant the reference asks for a state r_k AND a rate of
    change (r_{k+1} - r_k)/dt. The plant can only move as f(r_k, u) allows, so
    the question is whether ANY input produces that rate. Solved per sample by
    a short Gauss-Newton on u; whatever cannot be matched is the part of the
    reference the plant physically cannot follow -- regardless of tuning.

    This is the general form of the static target check: a constant reference
    reduces to "is the target an equilibrium", and a moving one catches the
    classic wrong-trajectory-type mistakes -- a position asked to follow a
    sinusoid while its velocity is held at zero, or an under-actuated plant
    asked to move more states independently than it has inputs for.
    """
    ref = np.asarray(reference, dtype=float)
    findings: List[Dict[str, Any]] = []
    names, in_names = _names(dynamics), _input_names(dynamics)
    n, m = dynamics.n_states, dynamics.n_inputs
    if ref.ndim != 2 or ref.shape[1] != n or ref.shape[0] < 2:
        return findings
    if not np.all(np.isfinite(ref)):
        findings.append(_finding(
            "error", "reference_not_finite", "The reference contains NaN/Inf",
            "Some samples of the reference are not finite. Every Actor proposal would fail at "
            "step 0, because the reference does not depend on Q, R, Np or Nc.",
            "Fix whatever generates the reference -- usually a division or a log in a custom "
            "trajectory file.", where))
        return findings

    moving = not np.allclose(ref, ref[0])
    rate = np.diff(ref, axis=0) / dt
    idx = np.unique(np.linspace(0, len(rate) - 1, min(max_samples, len(rate))).astype(int))

    # ---- derivative-pair consistency --------------------------------------
    for pos_i, vel_j in (getattr(dynamics.config, "derivative_pairs", None) or []):
        if pos_i >= n or vel_j >= n:
            continue
        implied = rate[:, pos_i]
        declared = ref[:-1, vel_j]
        scale = max(float(np.max(np.abs(implied))), float(np.max(np.abs(declared))), 1e-9)
        mismatch = float(np.max(np.abs(implied - declared))) / scale
        if mismatch > DERIVATIVE_MISMATCH_TOL and scale > 1e-6:
            findings.append(_finding(
                "warning", "derivative_pair_inconsistent",
                f"{names[vel_j]} does not match the rate of change of {names[pos_i]}",
                f"{names[vel_j]} is the time derivative of {names[pos_i]}, but the reference "
                f"asks for values that disagree by up to {mismatch * 100:.0f}% of their scale. "
                f"The plant cannot satisfy both at once, so it will compromise between them.",
                f"Give {names[vel_j]} the derivative of {names[pos_i]}'s reference (per-state mode "
                f"d_sin for a sin position: A*w*cos, not plain cos), or "
                f"leave {names[vel_j]} at the regulation default and let the tracking follow.",
                where, position=names[pos_i], velocity=names[vel_j], mismatch=mismatch))

    # ---- sample-by-sample dynamic feasibility -----------------------------
    u_eq = np.asarray(dynamics.get_equilibrium_input(), dtype=float).ravel()
    bounds = dynamics.get_input_bounds()
    lo = hi = None
    if bounds is not None:
        lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)

    residuals, per_state, needs_more = [], np.zeros(n), 0
    scale_samples = []
    for k in idx:
        x_k, want = ref[k], rate[k]
        u = u_eq.copy()
        try:
            for _ in range(2):           # affine-in-u plants converge in one step
                res = np.asarray(dynamics.dynamics(x_k, u), dtype=float) - want
                _, B = _jacobians(dynamics, x_k, u)
                du, *_ = np.linalg.lstsq(B, -res, rcond=None)
                u = u + du
            res = np.asarray(dynamics.dynamics(x_k, u), dtype=float) - want
        except Exception:  # noqa: BLE001
            continue
        residuals.append(res)
        scale_samples.append(max(float(np.linalg.norm(want)),
                                 float(np.linalg.norm(np.asarray(dynamics.dynamics(x_k, u_eq)) - want))))
        if lo is not None and np.any((u < lo - 1e-9) | (u > hi + 1e-9)):
            needs_more += 1

    if residuals:
        R = np.array(residuals)
        scale = max(float(np.percentile(scale_samples, 95)), 1e-6)
        rel = np.linalg.norm(R, axis=1) / scale
        bad = rel > REFERENCE_RESIDUAL_TOL
        frac = float(bad.mean())
        if frac > 0:
            per_state = np.mean(np.abs(R[bad]), axis=0)
            worst = [names[i] for i in np.argsort(-per_state)[:3] if per_state[i] > 1e-9]
            if not moving:
                # A constant target the plant cannot rest at is a mistake in the
                # setup, and no tuning survives it.
                findings.append(_finding(
                    "error", "reference_infeasible",
                    "The plant cannot be held at this target",
                    f"No input makes the target a resting point; the mismatch lands on "
                    f"{', '.join(worst)}. The controller will settle at the closest point it can "
                    "reach, and the leftover error will read like bad tuning -- no Q/R/Np/Nc can "
                    "remove it.",
                    "Give constant targets only to the states you actually control, and leave the "
                    "others at the value the physics settles them to.",
                    where, infeasible_fraction=frac, worst_states=worst))
            else:
                # A moving reference that cannot be followed EXACTLY. Two very
                # different causes, and the message has to name the right one
                # -- it used to always blame under-actuation, which on the
                # over-actuated quadcopter pointed away from the real cause, a
                # velocity reference that was not its position's derivative
                # (campaign bug 14).
                n_moving = int(np.sum(np.ptp(ref, axis=0) > 1e-9))
                rank_b = _input_rank(dynamics, ref[0], u_eq)
                if rank_b is not None and n_moving > rank_b:
                    cause = (f" The reference moves {n_moving} states independently, but the inputs "
                             f"can drive only {rank_b} independent directions at once -- moving one "
                             f"state necessarily disturbs the ones coupled to it.")
                    fix = (f"Decide which states matter: weight them up in Q and weight "
                           f"{', '.join(worst)} down, or give a moving reference to at most "
                           f"{rank_b} independent states.")
                else:
                    cause = (" The inputs could drive every moving state"
                             + (f" ({n_moving} moving, {rank_b} independent input directions)"
                                if rank_b is not None else "")
                             + ", so the conflict is kinematic: a state whose reference is not the "
                               "derivative of the state it is the rate of -- e.g. a position that "
                               "moves while its velocity reference stays at 0.")
                    fix = ("Make each velocity reference the derivative of its position reference: "
                           "per-state modes d_sin / d_cos / d_pulse give exactly the derivative of "
                           "sin / cos / pulse, or upload a trajectory that sets both.")
                findings.append(_finding(
                    "warning", "reference_infeasible",
                    "The plant cannot follow this reference exactly",
                    f"For {frac * 100:.0f}% of the reference, no input makes every state move the "
                    f"way it asks at once. The compromise will land mostly on "
                    f"{', '.join(worst)}: expect tracking error there however it is tuned." + cause,
                    fix, where, infeasible_fraction=frac, worst_states=worst))

        if needs_more and lo is not None:
            findings.append(_finding(
                "warning", "reference_exceeds_actuators",
                "Following this reference needs more actuation than is available",
                f"At {needs_more} of {len(residuals)} sampled instants the input required to "
                "follow the reference is outside the declared input bounds, so the controller "
                "will saturate there and fall behind.",
                "Slow the reference down (lower frequency or amplitude), or widen the input "
                "bounds if the hardware allows it.",
                where, samples_needing_more=needs_more))

    return sort_findings(findings)
