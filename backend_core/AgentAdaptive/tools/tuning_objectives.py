import collections
import math

DEFAULT_WEIGHT = 3
WEIGHT_MIN = 1
WEIGHT_MAX = 5

# "never settled" cost when horizon is unknown: big enough to swamp any
# real settling time but not inf, which would nan the weighted average
_UNSETTLED_SENTINEL = 1.0e6

# restated from tuner_agent (can't import it, that'd be circular). A safety
# gate always unioned into the user's scope so a diverging run can be pulled back
_DIVERGENCE_PARAMS = frozenset(
    {"K", "Lam", "c_gains", "Gamma", "kappa", "surface_lambda"})

_ALL_SYMPTOMS = frozenset({
    "numerical_divergence", "steady_state_error", "slow_transient",
    "high_overshoot", "control_effort_too_high", "chattering",
    "estimator_lag",
})

_P_STEADY = frozenset({"K", "Lam", "c_gains", "Gamma", "kappa",
                       "kappa_s", "k2", "k3", "k4"})
_P_GAINS = frozenset({"K", "Lam", "c_gains"})
_P_CHATTER = frozenset({"phi_layer", "K"})
_P_ESTIMATOR = frozenset({"Gamma", "sigma_W"})


# dict order = UI display order. every objective_value() is a BADNESS (lower
# is better), goodness metrics like tracking % get flipped once on entry.
OBJECTIVES = collections.OrderedDict()

OBJECTIVES["steady_state_error"] = {
    "label": "Steady-state error",
    "help": "how far off the output is once things settle down",
    "metric_keys": ("steady_rms_frac",),
    "symptoms": ("steady_state_error",),
    "params": _P_STEADY,
    "requires": None,
    "lower_is_better": True,
}

OBJECTIVES["tracking_mse"] = {
    "label": "Overall tracking accuracy",
    "help": "how well it tracks the reference the whole time, not just at the end",
    "metric_keys": ("tracking_pct_headline", "tracking_mse"),
    "symptoms": ("steady_state_error", "slow_transient"),
    "params": _P_STEADY | _P_GAINS,
    "requires": None,
    "lower_is_better": True,
}

OBJECTIVES["overshoot"] = {
    "label": "Overshoot",
    "help": "how far it overshoots the target before coming back",
    "metric_keys": ("overshoot_frac",),
    "symptoms": ("high_overshoot",),
    "params": _P_GAINS,
    "requires": None,
    "lower_is_better": True,
}

OBJECTIVES["settling_time"] = {
    "label": "Settling time",
    "help": "how long it takes to settle near the target",
    "metric_keys": ("settling_time", "settling_time_reached"),
    "symptoms": ("slow_transient",),
    "params": _P_GAINS,
    "requires": None,
    "lower_is_better": True,
}

OBJECTIVES["transient_error"] = {
    "label": "Transient error",
    "help": "how big the error is early on, before it settles",
    "metric_keys": ("transient_rms", "task_scale"),
    "symptoms": ("slow_transient",),
    "params": _P_GAINS,
    "requires": None,
    "lower_is_better": True,
}

OBJECTIVES["control_effort"] = {
    "label": "Control effort",
    "help": "how hard the actuator works on average",
    "metric_keys": ("control_rms",),
    "symptoms": ("control_effort_too_high",),
    "params": _P_GAINS,
    "requires": None,
    "lower_is_better": True,
}

OBJECTIVES["control_max"] = {
    "label": "Peak control effort",
    "help": "the biggest control push demanded, this is what maxes out a real actuator",
    "metric_keys": ("control_max",),
    "symptoms": ("control_effort_too_high",),
    "params": _P_GAINS,
    "requires": None,
    "lower_is_better": True,
}

OBJECTIVES["control_smoothness"] = {
    "label": "Control smoothness",
    "help": "how jumpy the control signal is, jumpy = more wear on the actuator",
    "metric_keys": ("control_rate_max",),
    "symptoms": ("control_effort_too_high", "chattering"),
    "params": _P_GAINS | _P_CHATTER,
    "requires": None,
    "lower_is_better": True,
}

OBJECTIVES["chattering"] = {
    "label": "Chattering (SMC)",
    "help": "the buzzy switching noise you get with sliding-mode designs",
    "metric_keys": ("chattering_tv_per_sec",),
    "symptoms": ("chattering",),
    "params": _P_CHATTER,
    "requires": "smc",
    "lower_is_better": True,
}

OBJECTIVES["estimator_accuracy"] = {
    "label": "Estimator accuracy",
    "help": "how close the estimator's guess is to the real unknown stuff",
    "metric_keys": ("estimator_residual_rms",),
    "symptoms": ("estimator_lag",),
    "params": _P_ESTIMATOR,
    "requires": "estimator",
    "lower_is_better": True,
}


def _finite(value):
    # rejecting bools here is intentional: True would otherwise sail through as 1.0
    # and turn a misplaced flag into a fake metric value
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _worst(value):
    if isinstance(value, dict):
        return None
    if isinstance(value, (list, tuple)):
        usable = [v for v in (_finite(x) for x in value) if v is not None]
        return max(usable) if usable else None
    return _finite(value)


def _usable_metrics(metrics):
    if not isinstance(metrics, dict):
        return None
    if not metrics.get("numerically_healthy", True):
        return None
    return metrics


def applicable_objectives(method=None, has_estimator=True):
    m = (method or "").strip().lower()
    out = []
    for key, spec in OBJECTIVES.items():
        requires = spec["requires"]
        if requires == "smc" and m == "backstepping":
            continue
        if requires == "estimator" and not has_estimator:
            continue
        out.append(key)
    return out


def normalize_selection(selection):
    if not selection:
        return {}

    raw = {}
    if isinstance(selection, dict):
        raw = selection
    elif isinstance(selection, (list, tuple, set, frozenset)):
        raw = dict.fromkeys(selection, DEFAULT_WEIGHT)
    else:
        return {}

    weights = {}
    for key, weight in raw.items():
        if key not in OBJECTIVES:
            continue
        value = _finite(weight)
        if value is None:
            value = DEFAULT_WEIGHT
        weights[key] = max(WEIGHT_MIN, min(WEIGHT_MAX, int(value)))

    return collections.OrderedDict(
        (key, weights[key]) for key in OBJECTIVES if key in weights)


def objective_value(key, metrics, fixed_args=None):
    metrics = _usable_metrics(metrics)
    if metrics is None or key not in OBJECTIVES:
        return None

    if key == "steady_state_error":
        return _worst(metrics.get("steady_rms_frac"))

    if key == "tracking_mse":
        headline = _finite(metrics.get("tracking_pct_headline"))
        if headline is not None:
            return max(0.0, 1.0 - headline / 100.0)
        mse = metrics.get("tracking_mse")
        if isinstance(mse, dict):
            for window in ("steady", "full", "transient"):
                worst = _worst(mse.get(window))
                if worst is not None:
                    return worst
            return None
        return _worst(mse)

    if key == "overshoot":
        return _worst(metrics.get("overshoot_frac"))

    if key == "settling_time":
        return _settling_value(metrics, fixed_args)

    if key == "transient_error":
        worst = _worst(metrics.get("transient_rms"))
        if worst is None:
            return None
        scale = _worst(metrics.get("task_scale"))
        if scale is not None and scale > 0.0:
            return worst / scale
        return worst

    if key == "control_effort":
        return _worst(metrics.get("control_rms"))

    if key == "control_max":
        return _worst(metrics.get("control_max"))

    if key == "control_smoothness":
        return _worst(metrics.get("control_rate_max"))

    if key == "chattering":
        return _worst(metrics.get("chattering_tv_per_sec"))

    if key == "estimator_accuracy":
        return _worst(metrics.get("estimator_residual_rms"))

    return None                                     # pragma: no cover


def _settling_value(metrics, fixed_args):
    has_time = "settling_time" in metrics
    has_flag = "settling_time_reached" in metrics
    if not has_time and not has_flag:
        return None

    if metrics.get("settling_time_reached"):
        value = _finite(metrics.get("settling_time"))
        if value is not None:
            return value

    # never settled: price as at-least-t_end, not None, so it doesn't tie
    # with a design that settled instantly
    if isinstance(fixed_args, dict):
        horizon = _finite(fixed_args.get("t_end"))
        if horizon is not None and horizon > 0.0:
            return horizon
    return _UNSETTLED_SENTINEL


def allowed_symptoms(selection):
    selection = normalize_selection(selection)
    if not selection:
        return frozenset(_ALL_SYMPTOMS)
    out = {"numerical_divergence"}
    for key in selection:
        out.update(OBJECTIVES[key]["symptoms"])
    return frozenset(out)


def allowed_params(selection):
    selection = normalize_selection(selection)
    if not selection:
        return None
    out = set(_DIVERGENCE_PARAMS)
    for key in selection:
        out.update(OBJECTIVES[key]["params"])
    return frozenset(out)


def weighted_score(metrics, selection, baseline_metrics=None, fixed_args=None):
    # this gates on "bounded", not full success, and that's the point: gating on
    # full success tied every pre-target round at inf, hiding real improvement behind round 0.
    if not metrics or not isinstance(metrics, dict):
        return float("inf")
    if not metrics.get("numerically_healthy", True):
        return float("inf")
    if not metrics.get("success_checks", {}).get("bounded", True):
        return float("inf")

    selection = normalize_selection(selection)
    if not selection:
        return None

    total_weight = 0.0
    total = 0.0
    for key, weight in selection.items():
        value = objective_value(key, metrics, fixed_args)
        if value is None:
            continue
        ratio = value
        if baseline_metrics:
            base = objective_value(key, baseline_metrics, fixed_args)
            if base is not None and base > 0.0:
                ratio = value / base
        total += weight * ratio
        total_weight += weight

    if total_weight <= 0.0:
        return None
    return total / total_weight


def format_priorities_block(selection):
    selection = normalize_selection(selection)
    if not selection:
        return ""

    lines = ["USER PRIORITIES (binding, weight 1-5, 5 = most important):"]
    for key, weight in _by_weight(selection):
        lines.append("- %s (weight %d): %s"
                     % (key, weight, OBJECTIVES[key]["label"]))
    lines.append("ALLOWED SYMPTOMS: %s"
                 % ", ".join(sorted(allowed_symptoms(selection))))
    lines.append("ALLOWED PARAMETERS: %s"
                 % ", ".join(sorted(allowed_params(selection))))
    lines.append("Everything else is out of scope this run.")
    return "\n".join(lines)


def selection_summary(selection):
    selection = normalize_selection(selection)
    if not selection:
        return "(no explicit priorities; tuner uses its default judgement)"
    return ", ".join("%s (%d)" % (OBJECTIVES[key]["label"].lower(), weight)
                     for key, weight in _by_weight(selection))


def _by_weight(selection):
    order = {key: i for i, key in enumerate(OBJECTIVES)}
    return sorted(selection.items(), key=lambda kv: (-kv[1], order[kv[0]]))


# _UNITS labels a column so numbers aren't misread as one scale: for example
# steady_state_error is vs task scale but overshoot is vs ref amplitude.
_UNITS = {
    "steady_state_error": "of task scale",
    "overshoot": "of ref amplitude",
    "transient_error": "RMS",
    "control_effort": "RMS(u)",
    "control_max": "peak |u|",
    "control_smoothness": "max |du/dt|",
    "chattering": "TV(u)/s",
    "estimator_accuracy": "RMS residual",
}


def objective_display(key, metrics, fixed_args=None):
    if key not in OBJECTIVES or not _usable_metrics(metrics):
        return None

    if key == "tracking_mse":
        # shown as the goodness % a user actually understands, not the
        # 1-p/100 badness the scorer works in internally
        pct = metrics.get("tracking_pct_headline")
        if _finite(pct):
            return "%.1f%% tracked (worst output)" % float(pct)
        return None

    if key == "settling_time":
        if not metrics.get("settling_time_reached", False):
            return "not reached within horizon"
        value = metrics.get("settling_time")
        return "%.3f s" % float(value) if _finite(value) else None

    metric_key = OBJECTIVES[key]["metric_keys"][0]
    worst = _worst(metrics.get(metric_key))
    if worst is None:
        return None
    unit = _UNITS.get(key, "")
    return ("%.4f %s" % (worst, unit)).strip()


def objective_rows(selection, metrics, fixed_args=None, baseline_metrics=None):
    selection = normalize_selection(selection)
    rows = []
    for key, _weight in _by_weight(selection):
        shown = objective_display(key, metrics, fixed_args)
        if shown is None:
            continue
        before = None
        if baseline_metrics is not None:
            before = objective_display(key, baseline_metrics, fixed_args)
        rows.append((OBJECTIVES[key]["label"], shown, before))
    return rows
