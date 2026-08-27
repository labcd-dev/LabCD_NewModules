"""
================================================================================
mpc/config.py
================================================================================
Configuration objects for the MPC controller.

Changes vs. the original notebook:
  * ``DoMPCConfig`` (do-mpc / ipopt / qpoases settings) was removed. It was
    never actually consumed by GenericMPC (search the old code: do-mpc is
    never imported or called) -- it was dead configuration that made the
    controller look like it used a nonlinear NLP solver when it actually runs
    a linearized QP through OSQP. If do-mpc support is wanted later, it
    should be a separate ``NMPCConfig`` used by a separate controller class,
    not a silently-unused field on this one.
  * ``x_bounds`` is now actually threaded into the QP (see controller.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

ArrayLike = Union[List[float], np.ndarray]


@dataclass
class DataConfig:
    simulation_time: float = 10.0
    dt_simulation: float = 0.001
    dt_mpc: float = 0.01
    noise_std: Union[float, ArrayLike] = 0.0   # scalar (applied uniformly) or one value per state --
                                                  # see agents/scenario_presets.py's per-state noise controls
    feedforward_override: Optional[ArrayLike] = None   # when set, used INSTEAD of dynamics.get_equilibrium_input()
                                                           # to seed u_prev -- see agents/dynamics_validator.py's
                                                           # estimate_feedforward_trim and the Configure section's
                                                           # "Use computed feedforward trim input" toggle
    trajectory_mode: str = "reg"   # "reg" | "sin" | "pulse" | "custom" -- see dynamics/base.py: SystemConfig.desired_trajectory
    trajectory_amplitude: float = 0.5    # sin/pulse amplitude
    trajectory_frequency: float = 0.5     # sin frequency (Hz)
    trajectory_pulse_start: float = 0.2    # pulse rise time, as a fraction (0-1) of simulation_time
    trajectory_pulse_end: float = 0.7       # pulse fall time, as a fraction (0-1) of simulation_time
    trajectory_per_state_modes: Optional[List[str]] = None   # length n_states, each "reg"/"sin"/"cos"/"pulse" --
                                                                # overrides trajectory_mode entirely when set; see
                                                                # dynamics/base.py: SystemConfig.desired_trajectory
    custom_trajectory_fn: Optional[Callable] = None   # set by app.py when a validated custom trajectory file is loaded
    settling_tolerance: float = 0.05   # fraction of the initial error norm considered "settled" -- see agents/metrics.py


@dataclass
class MPCConfig:
    """MPC hyperparameters. This is the object the Actor agent proposes
    updates to (Np, Nc, Q, R, P)."""

    prediction_horizon: int = 15
    control_horizon: int = 5

    state_weights: Optional[np.ndarray] = None      # Q
    input_weights: Optional[np.ndarray] = None       # R
    terminal_weights: Optional[np.ndarray] = None    # P  (now actually used)

    u_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    x_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

    finite_diff_step: float = 1e-4
    use_analytic_jacobian: bool = True  # torch-autograd when available, else finite-diff

    def initialize_weights(self, n_states: int, n_inputs: int) -> None:
        self.state_weights = self._as_matrix(self.state_weights, n_states, default=1.0)
        self.input_weights = self._as_matrix(self.input_weights, n_inputs, default=0.1)
        self.terminal_weights = self._as_matrix(self.terminal_weights, n_states, default=1.0)

    @staticmethod
    def _as_matrix(weights: Optional[ArrayLike], dim: int, default: float) -> np.ndarray:
        if weights is None:
            return np.diag(np.full(dim, default))
        arr = np.asarray(weights, dtype=float)
        if arr.shape == (dim, dim):
            return arr
        if arr.shape == (dim,):
            return np.diag(arr)
        raise ValueError(f"Weight array must have shape ({dim},) or ({dim},{dim}), got {arr.shape}")

    def set_from_dict(self, params: Dict[str, Any]) -> None:
        """Apply a parameter dict as proposed by the Actor agent."""
        if "Np" in params:
            self.prediction_horizon = int(params["Np"])
        if "Nc" in params:
            self.control_horizon = min(int(params["Nc"]), self.prediction_horizon)
        if "Q" in params and params["Q"] is not None:
            self.state_weights = np.diag(np.asarray(params["Q"], dtype=float))
        if "R" in params and params["R"] is not None:
            self.input_weights = np.diag(np.asarray(params["R"], dtype=float))
        if "P" in params and params["P"] is not None:
            self.terminal_weights = np.diag(np.asarray(params["P"], dtype=float))

    @staticmethod
    def _resolve_bounds(bounds: Optional[Tuple[np.ndarray, np.ndarray]], dim: int):
        if bounds is None:
            return np.full(dim, -np.inf), np.full(dim, np.inf)
        lo, hi = (np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float))
        if lo.size == 1:
            lo = np.full(dim, lo.item())
        if hi.size == 1:
            hi = np.full(dim, hi.item())
        if lo.shape != (dim,) or hi.shape != (dim,):
            raise ValueError(f"bounds must broadcast to ({dim},), got {lo.shape}/{hi.shape}")
        return lo, hi

    def get_u_bounds(self, n_inputs: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._resolve_bounds(self.u_bounds, n_inputs)

    def get_x_bounds(self, n_states: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._resolve_bounds(self.x_bounds, n_states)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    system_name: str = "loaded_dynamics"
    random_seed: int = 42
