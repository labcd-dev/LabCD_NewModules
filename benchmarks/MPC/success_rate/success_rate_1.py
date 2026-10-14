"""Success Rate 1: existing normalized full-window controlled-output MSE score."""

from __future__ import annotations

import math
from typing import Any

SUCCESS_RATE_FACTOR = 10.0


def _nonnegative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _named_output_values(verification: dict[str, Any], values: Any,
                         outputs: list[str]) -> list[Any] | None:
    """Align a saved per-output array by name, never by an assumed order."""
    names = verification.get("outputs")
    if (not isinstance(names, list) or not isinstance(values, list)
            or len(names) != len(values)
            or any(not isinstance(name, str) for name in names)
            or len(names) != len(set(names)) or not set(outputs).issubset(names)):
        return None
    return [values[names.index(name)] for name in outputs]


def output_task_scales_mse(verification: Any, outputs: list[str]) -> list[float] | None:
    """Read saved scales only; interpret unmarked historical values as RMS."""
    if not isinstance(verification, dict):
        return None
    saved = _named_output_values(verification, verification.get("task_scale"), outputs)
    basis = verification.get("task_scale_basis", "rms")
    if saved is None or basis not in ("mse", "rms"):
        return None
    scales = []
    for value in saved:
        number = _nonnegative_number(value)
        if number is None:
            return None
        scale = number if basis == "mse" else number * number
        if not math.isfinite(scale):
            return None
        scales.append(scale)
    return scales


def calculate_success_rate_1(verification: Any, outputs: list[str]) -> float | None:
    """Return exp(-10 * mean_i(saved tracking MSE_i / saved task-scale MSE_i))."""
    if not isinstance(verification, dict):
        return None
    # MPC's divergence/bounds flag corresponds to Adaptive's hard health checks;
    # lack of settling alone is not a numerical failure and must not force zero.
    raw = verification.get("raw_result")
    raw = raw if isinstance(raw, dict) else {}
    status = verification.get("status") or {}
    if raw.get("unstable") is True or (isinstance(status, dict) and status.get("unstable") is True):
        return 0.0
    task_scale_mse = output_task_scales_mse(verification, outputs)
    if (not outputs or not isinstance(task_scale_mse, list)
            or len(task_scale_mse) != len(outputs)):
        return None
    tracking = verification.get("tracking_mse") or {}
    mse_values = _named_output_values(
        verification, tracking.get("full") if isinstance(tracking, dict) else None, outputs)
    if mse_values is None:
        return None
    ratios = []
    zero_scale_failure = False
    for mse_value, scale_value in zip(mse_values, task_scale_mse):
        mse, scale = _nonnegative_number(mse_value), _nonnegative_number(scale_value)
        if mse is None or scale is None:
            return None
        if scale == 0:
            zero_scale_failure |= mse != 0
            ratios.append(0.0)
        else:
            ratios.append(mse / scale)
    if zero_scale_failure:
        return 0.0
    # Divide before summing to avoid overflow for large finite normalized errors.
    return round(math.exp(-SUCCESS_RATE_FACTOR * math.fsum(ratio / len(ratios) for ratio in ratios)), 6)
