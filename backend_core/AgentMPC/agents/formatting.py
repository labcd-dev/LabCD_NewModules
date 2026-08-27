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
places is already tighter than the noise floor of most metrics computed here
anyway.
"""

from __future__ import annotations

import math
from typing import Any


def smart_round(value: float, ndigits: int = 2) -> Any:
    """Round to ``ndigits`` decimal places, EXCEPT when that would display as
    a misleading "0.00" for a value that is genuinely nonzero (e.g. 0.0006 at
    2 decimals) -- in that case, switch to a compact scientific-notation
    string ("6e-04") instead, which stays informative at the same ~ndigits+1
    significant-figure budget rather than silently rounding the value away.

    Returns a float for the normal case (so arithmetic/serialization on
    ordinary values is unaffected) and a str only for the small-value case
    (scientific notation isn't a meaningful float to round-trip through --
    it's already display text at that point, exactly like the UI's f"{x:.2f}").
    """
    if not isinstance(value, float) or not math.isfinite(value):
        return value
    rounded = round(value, ndigits)
    if value != 0.0 and rounded == 0.0:
        mantissa, _, exp = f"{value:.{ndigits}e}".partition("e")
        mantissa = mantissa.rstrip("0").rstrip(".") if "." in mantissa else mantissa
        return f"{mantissa}e{exp}"
    return rounded


def fmt_num(value: Any, ndigits: int = 2) -> str:
    """UI-display counterpart to smart_round: always returns a string (used
    in app.py wherever code currently does f"{x:.2f}"), switching to compact
    scientific notation for the same "would otherwise show as 0.00" case.
    Non-numeric / non-finite values fall back to str(value)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return str(v)
    result = smart_round(v, ndigits)
    return result if isinstance(result, str) else f"{result:.{ndigits}f}"


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
