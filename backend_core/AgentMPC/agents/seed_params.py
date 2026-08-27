"""
================================================================================
agents/seed_params.py
================================================================================
Parses and validates the optional "Initial Parameters" the user can type into
the Streamlit UI (Np, Nc, and comma-separated Q/R) into the same
{"Np":..., "Nc":..., "Q":[...], "R":[...], "P":[...]} dict shape the Actor
agent normally produces.

Deliberately has no pydantic/langchain dependency (unlike agents/schemas.py)
so it stays usable from any thin UI layer without pulling in the LLM stack
just to validate a few numbers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _parse_float_list(text: str, expected_len: int, field_name: str) -> list[float]:
    text = text.strip()
    if not text:
        raise ValueError(f"{field_name} is empty.")
    try:
        values = [float(v.strip()) for v in text.split(",") if v.strip()]
    except ValueError as e:
        raise ValueError(f"{field_name} must be comma-separated numbers, e.g. '10, 1, 20, 1'.") from e

    if len(values) != expected_len:
        raise ValueError(f"{field_name} needs exactly {expected_len} value(s), got {len(values)}.")
    if any(v <= 0 for v in values):
        raise ValueError(f"{field_name} values must all be positive (they're used as weights).")
    return values


def parse_seed_params(
    np_val: int,
    nc_val: int,
    q_text: str,
    r_text: str,
    n_states: int,
    n_inputs: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Returns (params_dict, None) on success, or (None, error_message) on
    validation failure -- never raises, so the UI can just check which one
    came back non-None."""
    try:
        if np_val < 1:
            raise ValueError("Np must be >= 1.")
        if nc_val < 1 or nc_val > np_val:
            raise ValueError("Nc must be >= 1 and <= Np.")

        q = _parse_float_list(q_text, n_states, "Q")
        r = _parse_float_list(r_text, n_inputs, "R")

        return {"Np": int(np_val), "Nc": int(nc_val), "Q": q, "R": r, "P": q}, None

    except ValueError as e:
        return None, str(e)
