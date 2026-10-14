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
                         outputs: list[str] | None) -> list[Any] | None:
    """Core arrays follow applied_spec outputs; named arrays can be reordered."""
    if not isinstance(values, list) or not values:
        return None
    names = verification.get("outputs")
    if names is None:
        # Unmodified Adaptive core does not include output names in metrics.
        # The caller supplies the applied specification's output order.
        return values if outputs is None or len(values) == len(outputs) else None
    if (not isinstance(names, list) or len(names) != len(values)
            or any(not isinstance(name, str) for name in names)
            or len(names) != len(set(names))):
        return None
    selected = names if outputs is None else outputs
    if (not selected or len(selected) != len(set(selected))
            or not set(selected).issubset(names)):
        return None
    return [values[names.index(name)] for name in selected]


def output_task_scales_mse(verification: Any, outputs: list[str] | None = None) -> list[float] | None:
    """Square each saved RMS scale once; preserve core task_scale unchanged."""
    if not isinstance(verification, dict):
        return None
    has_mse = "task_scale_mse" in verification
    values = verification.get("task_scale_mse" if has_mse else "task_scale")
    basis = "mse" if has_mse else verification.get("task_scale_basis", "rms")
    saved = _named_output_values(verification, values, outputs)
    if saved is None or basis not in ("rms", "mse"):
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


def calculate_success_rate_1(verification: Any, outputs: list[str] | None = None) -> float | None:
    """Return exp(-10 * mean_i(output MSE_i / task-scale MSE_i))."""
    if not isinstance(verification, dict):
        return None
    if verification.get("numerically_healthy") is False:
        return 0.0
    checks = verification.get("success_checks") or {}
    if checks.get("finite") is False or checks.get("bounded") is False:
        return 0.0
    tracking = verification.get("tracking_mse") or {}
    full_mse = _named_output_values(
        verification, tracking.get("full") if isinstance(tracking, dict) else None, outputs)
    scale = output_task_scales_mse(verification, outputs)
    if full_mse is None or scale is None or len(full_mse) != len(scale):
        return None

    normalized_errors = []
    for mse_i, scale_i in zip(full_mse, scale):
        mse_value = _nonnegative_number(mse_i)
        scale_value = _nonnegative_number(scale_i)
        if mse_value is None or scale_value is None:
            return None
        if scale_value == 0:
            if mse_value != 0:
                return 0.0
            normalized_errors.append(0.0)
            continue
        normalized_errors.append(mse_value / scale_value)
    return round(math.exp(-SUCCESS_RATE_FACTOR * math.fsum(
        ratio / len(normalized_errors) for ratio in normalized_errors)), 6)
