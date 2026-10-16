"""
================================================================================
agents/formatting.py
================================================================================
Rounds floats to a fixed precision before they go into an LLM prompt.

IMPORTANT SCOPE: this is only ever applied at the "presentation boundary" --
right before a value is formatted into a prompt string (and, in app.py, right
before display in the UI). It is NEVER applied inside the actual simulation
math (RK4 integration, the QP solve, metric computation itself all stay full
double precision) -- rounding intermediate physics/optimization values would
change the actual simulated behavior, not just how it's displayed. Only the
number that gets read by a human or an LLM is rounded.

Why: a raw Python float's repr (e.g. inside an f-string or a dict interpolated
into a prompt) can run to 15+ significant digits ("0.043478260869565216"),
none of which carries meaningful information for a tuning decision and all of
which costs tokens on every single LLM call, every iteration. Two decimal
places, widened to three significant figures for small values, keeps what a
tuning decision needs: a fixed two decimals hid a 35% MSE improvement
(0.0112 -> 0.0073 both read "0.01") from the Report Agent.
"""

from __future__ import annotations

import math
from typing import Any


#: Never show fewer significant figures than this, however many decimals
#: that takes.
MIN_SIGNIFICANT_FIGURES = 3


def _decimals_for(value: float, ndigits: int) -> int:
    """``ndigits`` decimals, or more when a small value would otherwise keep
    fewer than MIN_SIGNIFICANT_FIGURES. Two fixed decimals turned an MSE that
    improved 35% (0.0112 -> 0.0073) into "0.01 -> 0.01", and the Report Agent
    duly wrote that MSE "remained at 0.01 throughout" (campaign bug 9)."""
    if value == 0.0:
        return ndigits
    magnitude = math.floor(math.log10(abs(value)))
    return max(ndigits, MIN_SIGNIFICANT_FIGURES - 1 - magnitude)


def smart_round(value: float, ndigits: int = 2) -> Any:
    """Round to ``ndigits`` decimal places -- or to MIN_SIGNIFICANT_FIGURES
    significant figures when that needs more decimals -- so a small value is
    never flattened into a few identical-looking digits. Large values keep
    their ``ndigits`` decimals (123.456 -> 123.46). Non-finite values and
    non-floats pass through unchanged.
    """
    if not isinstance(value, float) or not math.isfinite(value):
        return value
    return round(value, _decimals_for(value, ndigits))


def fmt_num(value: Any, ndigits: int = 2) -> str:
    """UI-display counterpart to smart_round: always returns a string, with
    at least ``ndigits`` decimals and at least MIN_SIGNIFICANT_FIGURES
    significant figures; tiny values switch to scientific notation so the
    string stays short. Non-numeric / non-finite values fall back to
    str(value)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return str(v)
    decimals = _decimals_for(v, ndigits)
    if decimals > 6:
        return f"{v:.{MIN_SIGNIFICANT_FIGURES - 1}e}"
    return f"{v:.{decimals}f}"


def round_floats(obj: Any, ndigits: int = 2) -> Any:
    """Recursively round every float inside a (possibly nested) dict/list/
    tuple structure. Non-float leaves (int, str, bool, None) pass through
    unchanged. Safe to call on already-rounded or non-numeric data."""
    if isinstance(obj, bool):
        return obj  # bool is a subclass of int -- don't let it fall into the float branch
    if isinstance(obj, float):
        return smart_round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    if isinstance(obj, tuple):
        return tuple(round_floats(v, ndigits) for v in obj)
    return obj


def short_stop_reason(reason: Any) -> str:
    """A few words for why a closed-loop run was stopped early, for a table
    cell. The evaluator's full ``unstable_reason`` stays in the logs; this is
    what lets a results table say WHICH kind of stop it was -- a diverging
    plant and a QP that could not be solved need opposite fixes, and a bare
    "UNSTABLE" gives no way to tell them apart."""
    text = str(reason or "").lower()
    if "controller failed" in text:
        return "solver failed"
    if "non-finite" in text or "nan" in text:
        return "NaN state"
    if "diverged" in text:
        return "diverged"
    if "state bounds" in text:
        return "hit state bound"
    return "stopped early"
