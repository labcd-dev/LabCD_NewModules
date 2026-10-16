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
    trajectory_per_state_modes: Optional[List[str]] = None   # length n_states, each "reg"/"sin"/"cos"/"pulse"/"d_sin"/"d_cos"/"d_pulse" --
                                                                # overrides trajectory_mode entirely when set; see
                                                                # dynamics/base.py: SystemConfig.desired_trajectory
    custom_trajectory_fn: Optional[Callable] = None   # set by app.py when a validated custom trajectory file is loaded
    settling_tolerance: float = 0.05   # fraction of the initial error norm considered "settled" -- see agents/metrics.py
    # -- what the CONTROLLER believes, vs what the plant actually is ---------
    # The MPC is designed against a NOMINAL model. When a scenario perturbs the
    # plant -- physical parameters, actuator faults, an external disturbance --
    # the controller must keep predicting with the model it was designed for,
    # or there is no mismatch to be robust against and the whole robustness
    # test measures nothing. Set to a copy of the dynamics taken BEFORE any
    # perturbation (see agents/scenario_presets.apply_scenario_level); left
    # None, the controller and the simulator share one object, which is the
    # right thing when nothing has been perturbed.
    nominal_dynamics: Optional[Any] = None

    # -- actuator health -----------------------------------------------------
    # Per-input effectiveness lambda in (0, 1]: the plant actually receives
    # lambda * u while the controller computes and reports u, and is told
    # nothing. lambda=1 is a healthy actuator, 0.6 is one delivering 60% of
    # what it is asked for, 0 is a dead actuator.
    actuator_effectiveness: Optional[ArrayLike] = None

    # -- exogenous disturbance ------------------------------------------------
    # Both are either a fixed vector or a callable t -> vector, held constant
    # across each sample interval (zero-order hold). Neither is visible to the
    # controller: they are injected between the controller's output and the
    # plant, which is what makes them a disturbance rather than an input.
    #   input_disturbance: added to the applied input, length n_inputs -- the
    #       natural home for a wind force/torque acting on the same channels as
    #       the actuators.
    #   state_disturbance: added to the state derivative, length n_states --
    #       the general case, for anything that does not enter through an
    #       actuator channel.
    input_disturbance: Optional[Any] = None
    state_disturbance: Optional[Any] = None

    tune_dt: bool = True   # is dt_mpc part of the search, or pinned to the value the user set?
                              # False => the Actor is told not to propose dt and any dt it
                              # proposes anyway is ignored (see agents/actor.py and
                              # agents/evaluator.py). Driven by the Launch step's
                              # "Sample time (dt_mpc)" choice.


@dataclass
class MPCConfig:
    """MPC hyperparameters. This is the object the Actor agent proposes
    updates to (Np, Nc, Q, R, P)."""

    prediction_horizon: int = 15
    control_horizon: int = 5

    state_weights: Optional[np.ndarray] = None      # Q
    input_weights: Optional[np.ndarray] = None       # R
    terminal_weights: Optional[np.ndarray] = None    # P  (now actually used)
    # How the terminal weight is chosen. "dare": P solves the discrete
    # algebraic Riccati equation of the linearised model with this Q and R at
    # every step -- the standard MPC terminal cost (Mayne et al. 2000): the
    # finite-horizon cost then approximates the infinite-horizon LQR cost, which
    # gives nominal stability even with a short horizon. "given": P as proposed
    # (the Actor's P, or Q when none is given). DARE falls back to "given" for a
    # step whose model has no stabilising solution.
    terminal_weight_mode: str = "dare"

    u_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    x_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

    finite_diff_step: float = 1e-4
    use_analytic_jacobian: bool = True  # torch-autograd when available, else finite-diff

    # -- disturbance observer (offset-free MPC) -------------------------------
    # Off by default so every result produced before it existed stays
    # comparable. When on, the controller estimates an additive per-step state
    # disturbance d_hat from the one-step prediction error and adds it to its
    # own prediction model:
    #     eps_k   = x_k - x_hat_{k|k-1}
    #     d_hat_k = d_hat_{k-1} + observer_gain * eps_k
    # The controller is still never TOLD the disturbance -- it infers it from
    # what it measures, which is exactly what a real controller does. See
    # mpc/controller.py for the full derivation.
    use_disturbance_observer: bool = False
    # alpha in (0, 1]. 1 corrects the whole residual in one step (deadbeat,
    # but passes measurement noise straight into the estimate); smaller values
    # filter noise at the cost of slower convergence.
    observer_gain: float = 0.3

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
        # A blank cell means "no limit on this side". The UI's bounds tables
        # are st.data_editor grids, which display +/-inf as an EMPTY cell and
        # hand a cell the user clears back as None -> NaN. Passed through, a
        # NaN bound reaches the QP as `l`/`u` and the solver refuses the whole
        # problem ("l (lower bounds) has 32 NaN of 32 entries") -- a run that
        # dies on its first step over what the user meant as "unbounded".
        lo = np.where(np.isnan(lo), -np.inf, lo)
        hi = np.where(np.isnan(hi), np.inf, hi)
        inverted = np.flatnonzero(lo > hi)
        if inverted.size:
            pairs = ", ".join(f"#{i}: min {lo[i]:g} > max {hi[i]:g}" for i in inverted)
            raise ValueError(f"bounds are inverted ({pairs}) -- no input or state can satisfy them, "
                             "so every QP would be infeasible. Swap or fix these limits.")
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
