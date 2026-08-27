import numpy as np


def _as_2d(a):
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _task_scale(y2, ref2):
    # RMS error a controller frozen at y(0) would get: beats dividing by
    # max|ref|, which flatters an offset reference (also works for ref=0).
    null_err = y2[0:1, :] - ref2
    scale = np.sqrt(np.mean(null_err ** 2, axis=0))
    scale = np.asarray(scale, dtype=float).reshape(-1)
    trivial = np.where(np.isfinite(scale), scale < 1e-9, False)
    return scale, trivial


def _pct_from_mse(mse, scale_i, trivial_i):
    if not np.isfinite(mse) or mse < 0.0:
        return 0.0
    rms = float(np.sqrt(mse))
    if trivial_i:
        # task was "just stay where you already are". Nothing to divide by,
        # so score against a flat 1e-6 tolerance instead
        if rms <= 1e-6:
            return 100.0
        return float(100.0 * max(0.0, 1.0 - rms / 1e-6))
    if not np.isfinite(scale_i) or scale_i <= 0.0:
        return 0.0
    return float(100.0 * max(0.0, 1.0 - rms / scale_i))


def compute_tracking_score(t, y, ref):
    y2 = _as_2d(y)
    ref2 = _as_2d(ref)
    e = y2 - ref2
    n = e.shape[0]
    p = e.shape[1]

    if n < 1 or p < 1:
        empty = {"full": [], "steady": [], "transient": []}
        return {"tracking_pct": dict(empty), "tracking_mse": dict(empty),
                "task_scale": [], "task_trivial": [],
                "tracking_pct_headline": 0.0, "tracking_pct_mean": 0.0}

    scale, trivial = _task_scale(y2, ref2)
    if scale.shape[0] != p:
        scale = np.resize(scale, p)
        trivial = np.resize(trivial, p)

    w = max(1, n // 5)
    windows = {"full": e, "steady": e[-w:], "transient": e[:w]}

    pct = {}
    mse = {}
    for name in ("full", "steady", "transient"):
        seg = windows[name]
        seg_mse = np.mean(seg ** 2, axis=0)
        seg_mse = np.asarray(seg_mse, dtype=float).reshape(-1)
        mse[name] = [float(v) for v in seg_mse]
        pct[name] = [_pct_from_mse(seg_mse[i], float(scale[i]), bool(trivial[i]))
                     for i in range(p)]

    steady_pct = pct["steady"]
    # headline = worst output, not the mean, since one good channel shouldn't
    # hide a broken one
    headline = float(min(steady_pct)) if steady_pct else 0.0
    mean_pct = float(np.mean(steady_pct)) if steady_pct else 0.0

    return {
        "tracking_pct": pct,
        "tracking_mse": mse,
        "task_scale": [float(v) for v in scale],
        "task_trivial": [bool(v) for v in trivial],
        "tracking_pct_headline": headline,
        "tracking_pct_mean": mean_pct,
    }


def _zero_tracking_score(y, ref):
    # this covers a run that already blew up. Real MSE is inf but gets
    # reported as 0.0, so it doesn't break every %-format/JSON caller downstream
    try:
        p = max(1, _as_2d(y).shape[1])
    except Exception:
        p = 1
    zeros = [0.0] * p
    return {
        "tracking_pct": {"full": list(zeros), "steady": list(zeros),
                          "transient": list(zeros)},
        "tracking_mse": {"full": list(zeros), "steady": list(zeros),
                          "transient": list(zeros)},
        "task_scale": list(zeros),
        "task_trivial": [False] * p,
        "tracking_pct_headline": 0.0,
        "tracking_pct_mean": 0.0,
    }


# hard checks, ANDed in order. success_reason names whichever fails first.
# advisory checks used to exist too, kept as empty tuple for compatibility.
_SUCCESS_HARD_CHECKS = ("finite", "bounded", "mse_target")
_SUCCESS_ADVISORY_CHECKS = ()
_SUCCESS_CHECK_ORDER = _SUCCESS_HARD_CHECKS + _SUCCESS_ADVISORY_CHECKS

_SUCCESS_REASONS = {
    "finite": "finite: the simulation produced NaN or Inf values (the closed loop diverged).",
    "bounded": "bounded: the output and/or the control signal grew far beyond any plausible "
               "range for this task (|y| above 1000x the task scale, or |u| above 1e12).",
    "mse_target": "mse_target: the steady-state RMS error is larger than your target (%.1f%% "
                  "of the task scale) for at least one output; the controller does not "
                  "track the reference closely enough.",
}


def compute_success_verdict(t, y, ref, u, x_states, dt, fail_tol=0.02):
    y2 = _as_2d(y)
    ref2 = _as_2d(ref)
    u2 = _as_2d(u)
    x2 = _as_2d(x_states)

    checks = {}

    finite = bool(np.all(np.isfinite(y2)) and np.all(np.isfinite(u2))
                  and np.all(np.isfinite(x2)))
    checks["finite"] = finite
    if not finite:
        for name in _SUCCESS_CHECK_ORDER[1:]:
            checks[name] = False
        return {"success": False, "checks": checks,
                "reason": _SUCCESS_REASONS["finite"], "target_frac": float(fail_tol),
                "steady_rms_frac": None}

    e = y2 - ref2
    n = e.shape[0]
    p = e.shape[1]

    scale, trivial = _task_scale(y2, ref2)
    if scale.shape[0] != p:
        scale = np.resize(scale, p)
        trivial = np.resize(trivial, p)
    # trivial task -> S_i carries no info, fall back to ref amplitude (floor 1.0)
    ref_amp = np.max(np.abs(ref2), axis=0).reshape(-1)
    if ref_amp.shape[0] != p:
        ref_amp = np.resize(ref_amp, p)
    eff_scale = np.where(trivial, np.maximum(1.0, ref_amp), scale)
    eff_scale = np.where(np.isfinite(eff_scale) & (eff_scale > 0.0), eff_scale, 1.0)

    y_max = np.max(np.abs(y2), axis=0).reshape(-1) if n else np.zeros(p)
    y_ok = bool(np.all(y_max <= 1e3 * eff_scale))
    if u2.size:
        u_max = float(np.max(np.abs(u2)))
    else:
        u_max = 0.0
    u_ok = bool(np.isfinite(u_max) and u_max < 1e12)
    checks["bounded"] = bool(y_ok and u_ok)

    # returned verbatim as metrics["steady_rms_frac"] everywhere downstream.
    # don't recompute against max|ref| elsewhere, that let Tuner claim "met" on a fail.
    w = max(1, n // 5)
    steady_rms = np.sqrt(np.mean(e[-w:] ** 2, axis=0)).reshape(-1)
    steady_frac = steady_rms / eff_scale
    checks["mse_target"] = bool(np.all(np.isfinite(steady_frac))
                                and np.all(steady_frac <= fail_tol))

    reason = ""
    for name in _SUCCESS_HARD_CHECKS:
        if not checks[name]:
            reason = _SUCCESS_REASONS[name]
            if name == "mse_target":
                reason = reason % (100.0 * fail_tol)
            break

    return {"success": all(checks[k] for k in _SUCCESS_HARD_CHECKS),
            "checks": checks, "reason": reason, "target_frac": float(fail_tol),
            "steady_rms_frac": [float(v) for v in steady_frac]}


def compute_simulation_metrics(t, y, ref, u, x_states, alog, dt, fail_tol=0.02):
    metrics = {}
    finite = (np.all(np.isfinite(y)) and np.all(np.isfinite(u))
              and np.all(np.isfinite(x_states)))
    metrics["numerically_healthy"] = bool(finite)

    verdict = compute_success_verdict(t, y, ref, u, x_states, dt, fail_tol=fail_tol)
    metrics["success"] = bool(verdict["success"])
    metrics["success_checks"] = verdict["checks"]
    metrics["success_reason"] = verdict["reason"]
    metrics["success_target_frac"] = verdict["target_frac"]
    tracking = compute_tracking_score(t, y, ref) if finite else _zero_tracking_score(y, ref)
    metrics["tracking_pct"] = tracking["tracking_pct"]
    metrics["tracking_mse"] = tracking["tracking_mse"]
    metrics["task_scale"] = tracking["task_scale"]
    metrics["task_trivial"] = tracking["task_trivial"]
    metrics["tracking_pct_headline"] = tracking["tracking_pct_headline"]
    metrics["tracking_pct_mean"] = tracking["tracking_pct_mean"]

    if not finite:
        return metrics

    e = y - ref
    n = y.shape[0]
    w = max(1, n // 5)
    ref_scale = np.maximum(np.max(np.abs(ref), axis=0), 1e-9)

    metrics["transient_rms"] = np.sqrt(np.mean(e[:w] ** 2, axis=0)).tolist()
    metrics["steady_rms"] = np.sqrt(np.mean(e[-w:] ** 2, axis=0)).tolist()
    # reused from the verdict, not recomputed against ref_scale (see the
    # comment in compute_success_verdict, same reasoning)
    metrics["steady_rms_frac"] = verdict["steady_rms_frac"]
    metrics["overshoot_frac"] = (np.max(np.abs(e), axis=0) / ref_scale).tolist()

    band = 0.02 * ref_scale
    within_band = np.all(np.abs(e) <= band, axis=1) if e.ndim > 1 else (np.abs(e) <= band)
    settle_idx = n
    for i in range(n):
        if np.all(within_band[i:]):
            settle_idx = i
            break
    metrics["settling_time"] = float(t[settle_idx]) if settle_idx < n else None
    metrics["settling_time_reached"] = settle_idx < n

    metrics["control_rms"] = np.sqrt(np.mean(u ** 2, axis=0)).tolist()
    metrics["control_max"] = np.max(np.abs(u), axis=0).tolist()
    du = np.diff(u, axis=0) / dt
    metrics["control_rate_max"] = (np.max(np.abs(du), axis=0) if du.size else np.zeros(u.shape[1])).tolist()
    total_variation = np.sum(np.abs(np.diff(u, axis=0)), axis=0)
    duration = max(t[-1] - t[0], 1e-9)
    metrics["chattering_tv_per_sec"] = (total_variation / duration).tolist()

    if alog:
        combined_hat = combined_true = None
        if "g_hat" in alog:
            combined_hat = alog["g_hat"]
            combined_true = alog.get("g_true")
        elif "Delta_hat" in alog and "D_hat" in alog:
            combined_hat = alog["Delta_hat"] + alog["D_hat"]
            if "Delta_true" in alog and "D_true" in alog:
                combined_true = alog["Delta_true"] + alog["D_true"]
        if combined_hat is not None and combined_true is not None:
            resid = combined_hat[-w:] - combined_true[-w:]
            metrics["estimator_residual_rms"] = np.sqrt(np.mean(resid ** 2, axis=0)).tolist()

    return metrics


def _fmt_list(values, fmt="%.4f"):
    if values is None:
        return "n/a"
    return "[" + ", ".join(fmt % v for v in values) + "]"


def _format_verdict_lines(metrics):
    if "success" not in metrics:
        return []

    lines = []
    if metrics.get("success"):
        lines.append("RUN VERDICT: PASS")
    else:
        lines.append("RUN VERDICT: FAIL: %s"
                      % (metrics.get("success_reason") or "unspecified check failed"))

    pct = metrics.get("tracking_pct") or {}
    steady = pct.get("steady") or []
    full = pct.get("full") or []
    transient = pct.get("transient") or []
    worst_full = min(full) if full else 0.0
    lines.append("REFERENCE TRACKING: %.1f%% (worst output, steady-state) | "
                  "%.1f%% mean | %.1f%% full-window"
                  % (metrics.get("tracking_pct_headline", 0.0),
                     metrics.get("tracking_pct_mean", 0.0), worst_full))

    if len(steady) > 1:
        parts = []
        for i in range(len(steady)):
            parts.append("y%d %.1f/%.1f/%.1f"
                          % (i + 1, full[i] if i < len(full) else 0.0, steady[i],
                             transient[i] if i < len(transient) else 0.0))
        lines.append("- per-output tracking %% (full/steady/transient): %s"
                      % " | ".join(parts))

    return lines


def format_metrics_report(metrics, method, has_delta, has_disturbance, explicit_uncertainty):
    verdict_lines = _format_verdict_lines(metrics)
    if not metrics.get("numerically_healthy", True):
        return "\n".join(verdict_lines + ([""] if verdict_lines else []) + [
            "NUMERICAL HEALTH: FAILED. The simulation produced NaN/Inf values "
            "(the closed loop diverged). This almost always means the gains are "
            "too aggressive for this system's dynamics and/or the fixed "
            "integration step `dt`: reduce gains and/or `dt` before trying "
            "anything else; do not interpret this as a tracking-accuracy problem."])

    lines = verdict_lines + ([""] if verdict_lines else [])
    lines += ["NUMERICAL HEALTH: OK", ""]
    lines.append("TRACKING")
    lines.append("- transient RMS (first 20%%): %s" % _fmt_list(metrics["transient_rms"]))
    lines.append("- steady-state RMS (last 20%%): %s" % _fmt_list(metrics["steady_rms"]))
    lines.append("- steady-state RMS as a fraction of the task scale (same basis "
                  "as your target and the mse_target check above): %s"
                  % _fmt_list(metrics["steady_rms_frac"], "%.4f"))
    lines.append("- overshoot (max deviation, as a fraction of reference amplitude): %s"
                  % _fmt_list(metrics["overshoot_frac"], "%.4f"))
    lines.append("- settling time (first time error enters and stays within a 2%% band): %s"
                  % (("%.3fs" % metrics["settling_time"]) if metrics["settling_time_reached"]
                     else "never settled within the simulated horizon"))
    lines.append("")
    lines.append("CONTROL EFFORT")
    lines.append("- RMS(u): %s" % _fmt_list(metrics["control_rms"]))
    lines.append("- max|u|: %s" % _fmt_list(metrics["control_max"]))
    lines.append("- max|du/dt|: %s" % _fmt_list(metrics["control_rate_max"], "%.2f"))
    if method == "SMC":
        lines.append("- chattering (total variation of u per second; SMC-specific, "
                      "high values mean the switching term is too aggressive relative "
                      "to the boundary layer): %s" % _fmt_list(metrics["chattering_tv_per_sec"], "%.2f"))

    if has_delta or has_disturbance:
        lines.append("")
        if explicit_uncertainty and "estimator_residual_rms" in metrics:
            lines.append("ESTIMATOR QUALITY (explicit ground truth was available)")
            lines.append("- steady-state |Delta_hat+D_hat - true| RMS, per state: %s"
                          % _fmt_list(metrics["estimator_residual_rms"]))
        else:
            lines.append("ESTIMATOR QUALITY: not measurable. No explicit "
                          "delta_exprs/dist_exprs formula was given for this system, so "
                          "there is no ground truth to compare the estimate against. "
                          "Judge this design by tracking/control-effort metrics only.")

    return "\n".join(lines)
