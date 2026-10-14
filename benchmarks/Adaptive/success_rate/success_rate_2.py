"""Success Rate 2: equally weighted fixed-target pass fraction.

Never interpret Adaptive's overshoot_frac (maximum absolute error/reference)
as true overshoot. Read final output/reference trajectories to measure target
crossing. Downsampled exports may prove a failure, but cannot prove a pass.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .success_rate_1 import _named_output_values, _nonnegative_number

SUPPORTED_TARGETS = {
    "mse_max", "settling_time_max_s", "overshoot_max_pct", "control_effort_max",
}


def validate_fixed_targets(fixed_targets: Any) -> dict[str, float]:
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
            raise ValueError(f"fixed_targets.{key} must be finite and nonnegative, or null.")
        limits[key] = number
    return limits


def true_overshoot(record: dict[str, Any] | None, outputs: list[str],
                   report_path: Path | None = None) -> dict[str, Any]:
    """Maximum eligible output's target crossing / its initial gap, in percent."""
    result = {"measured_value": None, "per_output_percent": {}, "exact": False,
              "downsampled": None, "source": "saved final output/reference trajectories",
              "reason": "No saved final trajectories available."}
    if not isinstance(record, dict) or not outputs:
        return result
    series = record.get("series")
    if not isinstance(series, dict):
        return result
    channels = series.get("channels") or {}
    if not isinstance(channels, dict):
        result["reason"] = "Saved trajectory channel metadata is invalid."
        return result
    applied = record.get("applied_system_spec")
    applied = applied if isinstance(applied, dict) else {}
    spec = applied.get("dynamics")
    spec = spec if isinstance(spec, dict) else {}
    if any(not isinstance(channels.get(key, {}), dict) for key in ("y", "ref")):
        result["reason"] = "Saved output/reference channel metadata is invalid."
        return result
    names = (channels.get("y") or {}).get("names") or spec.get("outputs")
    ref_names = (channels.get("ref") or {}).get("names") or spec.get("outputs")
    if (not isinstance(names, list) or not names or names != ref_names
            or any(not isinstance(name, str) for name in names)
            or len(names) != len(set(names)) or not set(outputs).issubset(names)):
        result["reason"] = "Saved output/reference names are missing, mismatched or ambiguous."
        return result
    try:
        link = series.get("trajectory_file")
        if link:
            if report_path is None:
                raise ValueError("A report path is needed to resolve the trajectory archive.")
            archive = Path(link)
            if archive.is_absolute():
                raise ValueError("trajectory_file must be relative to its result JSON.")
            archive = (report_path.parent / archive).resolve()
            if not archive.is_relative_to(report_path.parent.resolve()):
                raise ValueError("trajectory_file must stay inside its model/seed results folder.")
            with np.load(archive, allow_pickle=False) as arrays:
                times = arrays["times"]
                actual = arrays["outputs"]
                reference = arrays["reference"]
            result["source"] = f"series.trajectory_file: {link} (outputs, reference, times)"
        else:
            times = np.asarray(channels["t"]["data"], dtype=float)
            actual = np.asarray(channels["y"]["data"], dtype=float)
            reference = np.asarray(channels["ref"]["data"], dtype=float)
        if (any(array.dtype.kind not in "biuf" for array in (times, actual, reference))
                or times.ndim != 1 or not len(times) or not np.isfinite(times).all()
                or abs(float(times[0])) > 1e-9
                or np.any(np.diff(times) <= 0)
                or actual.ndim != 2 or actual.shape != reference.shape
                or actual.shape != (len(times), len(names))
                or not np.isfinite(actual).all() or not np.isfinite(reference).all()):
            raise ValueError("Final output/reference arrays or timestamps are invalid.")
        if series.get("n_points", len(times)) != len(times):
            raise ValueError("Trajectory sample count disagrees with series.n_points.")
        result["downsampled"] = series.get("downsampled")
        result["exact"] = series.get("downsampled") is False
        eligible = []
        for name in outputs:
            index = names.index(name)
            desired = reference[:, index]
            gap = float(actual[0, index] - desired[0])
            # Match MPC's eligibility convention; moving targets have no
            # single regulation overshoot and zero gaps cannot normalize it.
            if not np.allclose(desired, desired[0]) or abs(gap) <= 1e-6:
                result["per_output_percent"][name] = None
                continue
            error = actual[:, index] - desired
            crossed = error[np.sign(error) == -np.sign(gap)]
            value = 100.0 * float(np.max(np.abs(crossed))) / abs(gap) if crossed.size else 0.0
            if not math.isfinite(value):
                raise ValueError("Calculated overshoot is nonfinite.")
            result["per_output_percent"][name] = value
            eligible.append(value)
        if not eligible:
            result["reason"] = "No controlled output has a constant reference and nonzero initial gap."
            return result
        result["measured_value"] = max(eligible)
        result["reason"] = ("" if result["exact"] else
                            "Export is downsampled or its sampling coverage is unknown; peaks may be missing.")
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        result.update(measured_value=None, exact=False,
                      reason=f"Cannot calculate true overshoot: {error}")
    return result


def evaluate_fixed_targets(verification: Any, fixed_targets: Any, outputs: list[str],
                           *, record: dict[str, Any] | None = None,
                           report_path: Path | None = None) -> dict[str, Any]:
    """Return Rate 2 plus auditable limits, actual measurements and check results."""
    limits = validate_fixed_targets(fixed_targets)
    verification = verification if isinstance(verification, dict) else {}
    health = verification.get("success_checks")
    health = health if isinstance(health, dict) else {}
    numerical_failure = (verification.get("numerically_healthy") is False
                         or health.get("finite") is False or health.get("bounded") is False)
    checks = {}
    for key, limit in limits.items():
        measured, available, reason, extra = None, False, "", {}
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
            source = "verification.tracking_mse.full (mean over controlled outputs)"
        elif key == "settling_time_max_s":
            measured = _nonnegative_number(verification.get("settling_time"))
            reached = verification.get("settling_time_reached")
            extra["settling_time_reached"] = reached if isinstance(reached, bool) else None
            available = reached is False or (reached is True and measured is not None)
            reason = ("The response did not settle." if reached is False else
                      "" if available else "Settling time or reached flag is missing/invalid.")
            source = "verification.settling_time, with settling_time_reached=true"
        elif key == "control_effort_max":
            values = verification.get("control_rms")
            numbers = [_nonnegative_number(value) for value in values] if isinstance(values, list) else []
            available = bool(numbers and all(value is not None for value in numbers))
            squared = [value * value for value in numbers] if available else []
            available = available and all(math.isfinite(value) for value in squared)
            if available:
                measured = math.fsum(value / len(squared) for value in squared)
                extra["mean_square_per_input"] = squared
            reason = "" if available else "Per-input RMS values are missing or invalid."
            source = "mean_i(verification.control_rms[i] ** 2), not mean RMS squared"
        else:  # overshoot_max_pct
            derived = true_overshoot(record, outputs, report_path) if not numerical_failure else {}
            measured = derived.get("measured_value")
            # Full samples allow pass/fail. Even a downsampled observed excess
            # proves a failure, but a small observed peak cannot prove a pass.
            available = measured is not None and (derived.get("exact") is True or measured > limit)
            reason = derived.get("reason", "No valid true-overshoot measurement.")
            source = derived.get("source", "saved final output/reference trajectories")
            extra = {key: derived.get(key) for key in ("per_output_percent", "downsampled", "exact")}
        passed = None
        if numerical_failure:
            passed = False
            reason = "Verification reported failed numerical finite/bounded health checks."
        elif available:
            passed = measured is not None and measured <= limit
            if key == "settling_time_max_s" and extra["settling_time_reached"] is False:
                passed = False
            if not passed and not reason:
                reason = "Measured value exceeds the configured limit."
        checks[key] = {"limit": limit, "measured_value": measured, "comparison": "<=",
                       "passed": passed, "source": source, "reason": reason, **extra}
    total = len(checks)
    passed = sum(check["passed"] is True for check in checks.values())
    failed = sum(check["passed"] is False for check in checks.values())
    unavailable = [key for key, check in checks.items() if check["passed"] is None]
    return {
        "success_rate_2": round(passed / total, 6) if total and not unavailable else None,
        "passed_targets": passed, "total_targets": total, "failed_targets": failed,
        "unavailable_targets": unavailable,
        "all_targets_passed": False if failed else True if total and not unavailable else None,
        "mse_scope": "mean_full_window_mse_of_controlled_outputs",
        "overshoot_scope": "max_controlled_output_target_crossing_relative_to_initial_gap",
        "outputs": list(outputs), "numerically_failed": numerical_failure, "checks": checks,
    }


def calculate_success_rate_2(verification: Any, fixed_targets: Any, outputs: list[str],
                             **kwargs: Any) -> float | None:
    return evaluate_fixed_targets(verification, fixed_targets, outputs, **kwargs)["success_rate_2"]
