"""Success Rate 2: equally weighted pass fraction for configured fixed targets.

MSE uses declared controlled outputs only. Settling/overshoot/effort retain
their native verification definitions. Missing/undefined checks give a null
score, never an inflated mean over only available checks. Numerical failures
fail every configured check; this is partial-target credit, not full-task pass rate.
"""

from __future__ import annotations

import math
from typing import Any

from .success_rate_1 import _named_output_values, _nonnegative_number

SUPPORTED_TARGETS = {
    "mse_max", "settling_time_max_s", "overshoot_max_pct", "control_effort_max",
}


def validate_fixed_targets(fixed_targets: Any) -> dict[str, float]:
    """Omitted/null target values disable a check; malformed limits are errors."""
    if fixed_targets is None:
        return {}
    if not isinstance(fixed_targets, dict):
        raise ValueError("simulation_parameters.fixed_targets must be an object or null.")
    unknown = fixed_targets.keys() - SUPPORTED_TARGETS
    if unknown:
        raise ValueError(f"Unsupported fixed target(s): {sorted(unknown)}")
    limits = {}
    for key, value in fixed_targets.items():
        if value is None:
            continue
        number = _nonnegative_number(value)
        if number is None:
            raise ValueError(f"fixed_targets.{key} must be a finite nonnegative number or null.")
        limits[key] = number
    return limits


def evaluate_fixed_targets(verification: Any, fixed_targets: Any,
                           outputs: list[str]) -> dict[str, Any]:
    """Return Rate 2 and each target's actual value, limit and pass/fail reason."""
    limits = validate_fixed_targets(fixed_targets)
    verification = verification if isinstance(verification, dict) else {}
    raw = verification.get("raw_result")
    raw = raw if isinstance(raw, dict) else {}
    status = verification.get("status")
    status = status if isinstance(status, dict) else {}
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        metrics = verification.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    numerical_failure = (raw.get("unstable") is True or status.get("unstable") is True
                         or bool(raw.get("error")))
    checks = {}

    for key, limit in limits.items():
        measured = None
        reason = ""
        available = False
        extra = {}
        if key == "mse_max":
            tracking = verification.get("tracking_mse")
            values = _named_output_values(
                verification, tracking.get("full") if isinstance(tracking, dict) else None, outputs)
            numbers = [_nonnegative_number(value) for value in values] if values else []
            available = bool(outputs and numbers and all(value is not None for value in numbers))
            if available:
                measured = math.fsum(value / len(numbers) for value in numbers)
                extra["per_output_mse"] = dict(zip(outputs, numbers))
            reason = "" if available else "Controlled-output full-window MSE is missing or invalid."
            source = "verification.tracking_mse.full (mean over declared outputs)"
        elif key == "settling_time_max_s":
            measured = _nonnegative_number(metrics.get("settling_time"))
            settled = metrics.get("settled", status.get("settled"))
            extra["settled"] = settled if isinstance(settled, bool) else None
            # A finite fallback duration must never be treated as achieved settling.
            available = settled is False or (settled is True and measured is not None)
            reason = ("The response did not settle." if settled is False else
                      "" if available else "Settling time or settled flag is missing/invalid.")
            source = "verification.raw_result.metrics.settling_time, with settled=true"
        elif key == "overshoot_max_pct":
            fraction = _nonnegative_number(metrics.get("overshoot"))
            meaningful = metrics.get("overshoot_meaningful")
            available = fraction is not None and meaningful is True
            if available:
                measured = 100.0 * fraction
                if not math.isfinite(measured):
                    measured, available = None, False
            extra["overshoot_meaningful"] = meaningful if isinstance(meaningful, bool) else None
            reason = "" if available else "Overshoot is undefined, missing or invalid; not a zero overshoot."
            source = "100 * verification.raw_result.metrics.overshoot (largest meaningful state overshoot)"
        else:  # control_effort_max
            measured = _nonnegative_number(metrics.get("control_effort"))
            available = measured is not None
            reason = "" if available else "Mean squared applied-input effort is missing or invalid."
            source = "verification.raw_result.metrics.control_effort (mean squared applied inputs)"

        passed = None
        if numerical_failure:
            passed = False
            reason = "Verification reported numerical divergence, a bound violation, or an error."
        elif available:
            passed = measured is not None and measured <= limit
            if key == "settling_time_max_s" and extra["settled"] is False:
                passed = False
            if not reason and not passed:
                reason = "Measured value exceeds the configured limit."
        checks[key] = {
            "limit": limit, "measured_value": measured, "comparison": "<=",
            "passed": passed, "source": source, "reason": reason, **extra,
        }

    total = len(checks)
    passed = sum(check["passed"] is True for check in checks.values())
    failed = sum(check["passed"] is False for check in checks.values())
    unavailable = [key for key, check in checks.items() if check["passed"] is None]
    score = round(passed / total, 6) if total and not unavailable else None
    return {
        "success_rate_2": score,
        "passed_targets": passed, "total_targets": total, "failed_targets": failed,
        "unavailable_targets": unavailable,
        "all_targets_passed": False if failed else True if total and not unavailable else None,
        "mse_scope": "mean_full_window_mse_of_controlled_outputs",
        "outputs": list(outputs),
        "numerically_failed": numerical_failure,
        "checks": checks,
    }


def calculate_success_rate_2(verification: Any, fixed_targets: Any,
                             outputs: list[str]) -> float | None:
    """Fraction of configured targets passed, or None if checks are unavailable."""
    return evaluate_fixed_targets(verification, fixed_targets, outputs)["success_rate_2"]
