"""JSON-friendly simulation time-series export for API / Recharts clients.

Metrics are always computed on full-resolution arrays. This module only
downsamples the **exported** payload.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

DEFAULT_MAX_POINTS = 2000
SERIES_KEY = "_series"  # reserved on components dict


def _env_max_points() -> int:
    raw = os.environ.get("LABCD_ADAPTIVE_SERIES_MAX_POINTS")
    if raw is None or not str(raw).strip():
        return DEFAULT_MAX_POINTS
    try:
        return max(10, int(raw))
    except ValueError:
        return DEFAULT_MAX_POINTS


def should_create_plots() -> bool:
    """Whether to build matplotlib figures during design/sim.

    Streamlit uses Agg and captures open figures for the UI and LaTeX PDF.
    Only skip figure creation when LABCD_ADAPTIVE_SHOW_PLOTS is explicitly off
    (API workers). Unset / true → create figures even under Agg.
    """
    val = os.environ.get("LABCD_ADAPTIVE_SHOW_PLOTS", "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    return True


def should_show_plots() -> bool:
    """Interactive GUI show (plt.show). Never required for Streamlit PDF capture.

    Off when LABCD_ADAPTIVE_SHOW_PLOTS is explicit false, or non-interactive
    backends (Agg). Streamlit still creates figures via should_create_plots().
    """
    val = os.environ.get("LABCD_ADAPTIVE_SHOW_PLOTS", "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    if val in ("1", "true", "yes", "on"):
        # Still respect Agg — interactive show would only warn
        backend = os.environ.get("MPLBACKEND", "").lower()
        if backend in ("agg", "pdf", "svg", "template"):
            return False
        return True
    backend = os.environ.get("MPLBACKEND", "").lower()
    if backend in ("agg", "pdf", "svg", "template"):
        return False
    return bool(os.environ.get("DISPLAY") or os.environ.get("MPLBACKEND"))


def maybe_show_plots() -> None:
    """Optional interactive show. Do not close figures — Streamlit needs them open."""
    if should_show_plots():
        import matplotlib.pyplot as plt

        plt.show()


def _as_2d(arr: Any) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def _json_float(v: float) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def _channel_matrix(
    arr: Any,
    indices: np.ndarray,
) -> List[List[Optional[float]]]:
    """Time-major list of channel vectors."""
    m = _as_2d(arr)
    if m.shape[0] == 0:
        return []
    m = m[indices]
    out: List[List[Optional[float]]] = []
    for row in m:
        out.append([_json_float(float(v)) for v in np.asarray(row).reshape(-1)])
    return out


def _name_list(names: Optional[Sequence[Any]], width: int, prefix: str) -> List[str]:
    if names is None:
        return [f"{prefix}{i}" for i in range(width)]
    cleaned = [str(n) for n in names]
    if len(cleaned) >= width:
        return cleaned[:width]
    cleaned = cleaned + [f"{prefix}{i}" for i in range(len(cleaned), width)]
    return cleaned


def build_series(
    t: Any,
    y: Any,
    ref: Any,
    u: Any,
    x_states: Any,
    *,
    dt: float,
    t_end: float,
    output_names: Optional[Sequence[Any]] = None,
    input_names: Optional[Sequence[Any]] = None,
    state_names: Optional[Sequence[Any]] = None,
    max_points: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the series contract described in ASSIGNMENT_SERIES_DATAPOINTS."""
    t_arr = np.asarray(t, dtype=float).reshape(-1)
    y2 = _as_2d(y)
    ref2 = _as_2d(ref)
    u2 = _as_2d(u)
    x2 = _as_2d(x_states)

    n = int(t_arr.shape[0])
    # Align lengths defensively
    n = min(n, y2.shape[0], ref2.shape[0], u2.shape[0], x2.shape[0]) if n else 0
    if n == 0:
        return {
            "version": 1,
            "dt": float(dt),
            "t_end": float(t_end),
            "n_points": 0,
            "downsampled": False,
            "max_points": int(max_points if max_points is not None else _env_max_points()),
            "channels": {
                "t": {"label": "time", "unit": "s", "data": []},
                "y": {"label": "outputs", "names": [], "data": []},
                "ref": {"label": "references", "names": [], "data": []},
                "u": {"label": "inputs", "names": [], "data": []},
                "x": {"label": "states", "names": [], "data": []},
            },
        }

    t_arr = t_arr[:n]
    y2, ref2, u2, x2 = y2[:n], ref2[:n], u2[:n], x2[:n]

    cap = int(max_points if max_points is not None else _env_max_points())
    cap = max(10, cap)
    if n > cap:
        # Uniform stride including last index
        indices = np.linspace(0, n - 1, num=cap, dtype=int)
        indices = np.unique(indices)
        downsampled = True
    else:
        indices = np.arange(n, dtype=int)
        downsampled = False

    t_data = [_json_float(float(v)) for v in t_arr[indices]]
    n_out = int(y2.shape[1]) if y2.ndim > 1 else 1
    n_in = int(u2.shape[1]) if u2.ndim > 1 else 1
    n_st = int(x2.shape[1]) if x2.ndim > 1 else 1

    return {
        "version": 1,
        "dt": float(dt),
        "t_end": float(t_end),
        "n_points": int(len(indices)),
        "downsampled": bool(downsampled),
        "max_points": cap,
        "channels": {
            "t": {"label": "time", "unit": "s", "data": t_data},
            "y": {
                "label": "outputs",
                "names": _name_list(output_names, n_out, "y"),
                "data": _channel_matrix(y2, indices),
            },
            "ref": {
                "label": "references",
                "names": _name_list(output_names, n_out, "ref"),
                "data": _channel_matrix(ref2, indices),
            },
            "u": {
                "label": "inputs",
                "names": _name_list(input_names, n_in, "u"),
                "data": _channel_matrix(u2, indices),
            },
            "x": {
                "label": "states",
                "names": _name_list(state_names, n_st, "x"),
                "data": _channel_matrix(x2, indices),
            },
        },
    }


def attach_series(
    components: Dict[str, Any],
    t: Any,
    y: Any,
    ref: Any,
    u: Any,
    x_states: Any,
    *,
    dt: float,
    t_end: float,
    outputs: Optional[Sequence[Any]] = None,
    inputs: Optional[Sequence[Any]] = None,
    states: Optional[Sequence[Any]] = None,
    max_points: Optional[int] = None,
    include: bool = True,
) -> Dict[str, Any]:
    """Store series on components under SERIES_KEY when include is true."""
    if not include:
        return components
    components[SERIES_KEY] = build_series(
        t,
        y,
        ref,
        u,
        x_states,
        dt=dt,
        t_end=t_end,
        output_names=outputs,
        input_names=inputs,
        state_names=states,
        max_points=max_points,
    )
    return components


def extract_series(components: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(components, dict):
        return None
    series = components.get(SERIES_KEY)
    return series if isinstance(series, dict) else None


def strip_series(components: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Copy components without the reserved series key (for report text paths)."""
    if not isinstance(components, dict):
        return {}
    return {k: v for k, v in components.items() if k != SERIES_KEY}
