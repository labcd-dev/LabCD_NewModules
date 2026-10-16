"""
================================================================================
dynamics/base.py
================================================================================
Generic building blocks that every dynamics plugin is built on top of.

Design contract (unchanged from the notebook, made explicit here):

    1. Every plugin defines a zero-argument ``create_config() -> SystemConfig``.
    2. Every plugin defines exactly one class that subclasses ``BaseDynamics``
       and implements ``dynamics(self, x, u) -> dx/dt``.

Because ``BaseDynamics.__init__`` always sets ``self.config``, downstream code
never needs to guess whether a dynamics instance "has" a config, dimensions,
state names, etc. -- it always does. The original notebook re-checked this with
long ``hasattr(...)`` chains in half a dozen places; that redundancy has been
removed on purpose. If a plugin is broken, ``DynamicLoader`` (loader.py) is the
single place that validates it, and it fails loudly at *load* time instead of
silently at *call* time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TrajectoryMode = str  # "sin" | "pulse" | "reg"


@dataclass
class SystemConfig:
    """Static description of a system. Knows nothing about MPC or agents."""

    n_states: int
    n_inputs: int
    params: Dict[str, Any]
    state_names: List[str]
    input_names: List[str]
    default_initial_state: np.ndarray
    default_target: np.ndarray
    state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    input_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    # (position_idx, velocity_idx) pairs -- NOT set by plugin authors. Left
    # as None by default and populated at runtime by
    # agents/dynamics_validator.py:detect_derivative_pairs() right after a
    # dynamics file is loaded (see app.py's load_dynamics_from_file), which
    # verifies the relationship mathematically (dx_i/dt == x_j at many
    # random points) rather than assuming it. When set, desired_trajectory's
    # global "sin"/"pulse" mode uses these exact pairs instead of the
    # (2i, 2i+1) convention fallback below.
    derivative_pairs: Optional[List[Tuple[int, int]]] = None
    # Optional: the largest acceptable deviation of each state from its
    # target, in the state's own units (NaN = let the Setup Agent choose). It
    # is what Bryson's rule weights by -- Q_ii = 1/acceptable_i^2 -- for the
    # initial Q (agents/dynamics_validator.py:estimate_initial_qr).
    acceptable_deviation: Optional[np.ndarray] = None
    # Physical units, one string per state / per input (e.g. ["rad", "rad/s"],
    # ["N.m"]). Purely descriptive -- nothing in the controller, the metrics or
    # the agents reads these, and leaving them None changes no behaviour. They
    # exist because a tuning result is not self-describing without them: "MSE =
    # 0.03" or a bound of "±2" means nothing to someone reading the exported
    # report later unless they know whether that is radians, metres or RPM.
    # Optional precisely so every existing plugin keeps working untouched.
    state_units: Optional[List[str]] = None
    input_units: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.default_initial_state = np.asarray(self.default_initial_state, dtype=float)
        self.default_target = np.asarray(self.default_target, dtype=float)
        if self.default_initial_state.shape != (self.n_states,):
            raise ValueError(
                f"default_initial_state must have shape ({self.n_states},), "
                f"got {self.default_initial_state.shape}"
            )
        if self.default_target.shape != (self.n_states,):
            raise ValueError(
                f"default_target must have shape ({self.n_states},), "
                f"got {self.default_target.shape}"
            )

    def desired_trajectory(
        self,
        dt_mpc: float,
        simulation_time: float,
        mode: TrajectoryMode = "reg",
        amplitude: float = 0.5,
        frequency: float = 0.5,
        pulse_start: float = 0.2,
        pulse_end: float = 0.7,
        per_state_modes: Optional[List[str]] = None,
        extend_steps: int = 0,
    ) -> np.ndarray:
        """Generate a reference trajectory. Kept small and explicit (no hidden
        module-level flag) -- callers pass the mode they want instead of
        relying on a global ``TRAJECTORY_MODE`` constant, which made the
        original code's behaviour depend on import order / cell execution
        order in the notebook.

        Two ways to control this, in increasing order of granularity:

        1. ``mode`` (default): one mode for the whole system. For "sin"/
           "pulse", every (position, velocity) pair in ``derivative_pairs``
           (found by the Setup Agent, agents/dynamics_validator.py:
           detect_derivative_pairs) moves together, and the velocity-slot
           reference is the ANALYTIC TIME-DERIVATIVE of the position-slot
           one (position ~ sin(t) implies velocity ~ omega*cos(t)). States in
           no pair hold at the target. With no pairs known on a multi-state
           plant this raises: guessing them from index order (2i, 2i+1)
           gave nonsense on any other state layout (campaign bug 5).

        2. ``per_state_modes`` (optional, overrides ``mode`` entirely when
           given): a list of length n_states, each one of "reg"/"sin"/
           "cos"/"pulse", or a derivative mode "d_sin"/"d_cos"/"d_pulse"
           -- choose the trajectory type independently per state. This is
           how you get the SAME physical consistency as (1) under your own
           control: set a position state to "sin" and its velocity state to
           "d_sin" (= A*w*cos(wt), exactly d/dt of A*sin(wt)). Note "cos" is
           A*cos(wt) -- a shape, e.g. y = cos for a circle with x = sin --
           NOT the derivative of "sin". Mix types freely for states that
           aren't a position/velocity pair at all. All states sharing a mode share the same amplitude/
           frequency/pulse timing -- there's deliberately no per-state
           amplitude/frequency to avoid a UI control explosion for systems
           with many states; independent per-state amplitude/frequency
           would be a natural follow-up if you ever need it.

        Args:
            pulse_start / pulse_end: as a FRACTION of simulation_time (0-1),
                when the pulse rises and falls -- e.g. 0.2/0.7 means the
                pulse is up from 20% to 70% of the way through the run.
            extend_steps: extra samples past simulation_time, so the MPC's
                look-ahead window exists on the last simulated steps too
                (the evaluator asks for Np of them). Signals simply continue;
                pulse timing stays a fraction of simulation_time.
        """
        n_steps = max(int(round(simulation_time / dt_mpc)), 1)
        # Sample k sits at exactly k*dt -- the simulation's own clock. This was
        # linspace(0, T, n), whose spacing T/(n-1) is not dt: the reference ran
        # 0.2% slow at T=5 s, dt=0.01 s (campaign bug 16).
        t = np.arange(n_steps + max(int(extend_steps), 0)) * dt_mpc
        n_steps = len(t)
        ref = np.tile(self.default_target, (n_steps, 1))

        if per_state_modes is not None:
            if len(per_state_modes) != self.n_states:
                raise ValueError(
                    f"per_state_modes must have length n_states ({self.n_states}), got {len(per_state_modes)}"
                )
            omega = 2 * np.pi * frequency
            total_time = simulation_time if simulation_time > 0 else 1.0
            t_norm = t / total_time
            step, d_step = _smooth_step_with_derivative(t_norm, pulse_start, pulse_end, dt_norm_dt=1.0 / total_time)

            for i, state_mode in enumerate(per_state_modes):
                if state_mode == "reg":
                    continue  # ref[:, i] already holds default_target[i] from the np.tile above
                elif state_mode == "sin":
                    ref[:, i] = amplitude * np.sin(omega * t)
                elif state_mode == "cos":
                    ref[:, i] = amplitude * np.cos(omega * t)
                elif state_mode == "pulse":
                    ref[:, i] = amplitude * step
                # Derivative modes, for a velocity state whose position state
                # runs one of the shapes above. "cos" is NOT one of these:
                # it is A*cos(wt), a position-like signal (a circle is x=sin,
                # y=cos), so a velocity set to "cos" disagreed with its
                # position by a factor w -- pi at 0.5 Hz (campaign bug 6).
                elif state_mode == "d_sin":
                    ref[:, i] = amplitude * omega * np.cos(omega * t)
                elif state_mode == "d_cos":
                    ref[:, i] = -amplitude * omega * np.sin(omega * t)
                elif state_mode == "d_pulse":
                    ref[:, i] = amplitude * d_step
                else:
                    raise ValueError(f"Unknown per-state trajectory mode: {state_mode!r} (state index {i})")
            return ref

        if mode == "reg":
            return ref

        if mode in ("sin", "pulse") and not self.derivative_pairs and self.n_states > 1:
            # This used to guess the pairs from index adjacency -- (0,1), (2,3),
            # ... -- which is only right for a [pos, vel, pos, vel] layout. On
            # the overactuated quadcopter ([x y z u v w phi theta psi p q r])
            # it paired x with y and phi with theta, gave all 12 states a
            # meaningless reference, and drove the vehicle into gimbal lock
            # (campaign bug 5). Without known pairs there is no consistent
            # whole-system wave to build: say so instead of guessing.
            raise ValueError(
                f"The {mode!r} trajectory moves every position state together with its velocity, "
                f"but no position/velocity pairs are known for this plant (the Setup Agent found "
                f"none -- e.g. velocities in the body frame are not the derivatives of positions). "
                f"Choose the moving states yourself with per-state trajectory modes instead."
            )

        if mode == "sin":
            omega = 2 * np.pi * frequency
            if self.derivative_pairs:
                for k, (pos_i, vel_i) in enumerate(self.derivative_pairs):
                    phase = (np.pi / 4) * k
                    ref[:, pos_i] = amplitude * np.sin(omega * t + phase)
                    ref[:, vel_i] = amplitude * omega * np.cos(omega * t + phase)
            else:  # a single-state plant: nothing to pair
                ref[:, 0] = amplitude * np.sin(omega * t)
            return ref

        if mode == "pulse":
            total_time = simulation_time if simulation_time > 0 else 1.0
            t_norm = t / total_time
            step, step_deriv_dt = _smooth_step_with_derivative(t_norm, pulse_start, pulse_end, dt_norm_dt=1.0 / total_time)
            if self.derivative_pairs:
                for pos_i, vel_i in self.derivative_pairs:
                    ref[:, pos_i] = amplitude * step
                    ref[:, vel_i] = amplitude * step_deriv_dt
            else:  # a single-state plant: nothing to pair
                ref[:, 0] = amplitude * step
            return ref

        raise ValueError(f"Unknown trajectory mode: {mode!r}")


def _smooth_step_with_derivative(
    t_norm: np.ndarray, start: float, end: float, dt_norm_dt: float, steepness: float = 50.0
):
    """Returns (step, d(step)/dt) for a smooth rise-then-fall pulse built from
    two sigmoids: step(t_norm) = sigmoid(k*(t_norm-start)) - sigmoid(k*(t_norm-end)).

    ``dt_norm_dt`` is d(t_norm)/dt = 1/total_time -- the chain-rule factor to
    convert the derivative from "per unit of normalized time" to "per second",
    since t_norm = t / total_time.
    """
    s1 = 1 / (1 + np.exp(-steepness * (t_norm - start)))
    s2 = 1 / (1 + np.exp(-steepness * (t_norm - end)))
    step = s1 - s2

    # d(sigmoid(k*(x-a)))/dx = k * sigmoid(k*(x-a)) * (1 - sigmoid(k*(x-a)))
    ds1_dtnorm = steepness * s1 * (1 - s1)
    ds2_dtnorm = steepness * s2 * (1 - s2)
    step_deriv_dt = (ds1_dtnorm - ds2_dtnorm) * dt_norm_dt

    return step, step_deriv_dt


class BaseDynamics(ABC):
    """Every dynamics plugin subclasses this."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.params = config.params
        self.n_states = config.n_states
        self.n_inputs = config.n_inputs
        self.state_names = config.state_names
        self.input_names = config.input_names

    @abstractmethod
    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Continuous-time dx/dt = f(x, u). Must return shape (n_states,)."""
        raise NotImplementedError

    # -- optional hooks -----------------------------------------------------
    def get_equilibrium_input(self) -> np.ndarray:
        """Override if the system has a non-zero trim/equilibrium input."""
        return np.zeros(self.n_inputs)

    def get_input_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Override, or set ``config.input_bounds``, to enforce input limits."""
        return self.config.input_bounds

    def get_state_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Override, or set ``config.state_bounds``, to enforce state limits."""
        return self.config.state_bounds

    def check_termination(self, x: np.ndarray) -> bool:
        bounds = self.get_state_bounds()
        if bounds is None:
            return False
        lower, upper = bounds
        return bool(np.any(x < lower) or np.any(x > upper))

    # -- integration ----------------------------------------------------
    def rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float,
                  d_state: Optional[np.ndarray] = None) -> np.ndarray:
        """One RK4 step of the true plant.

        ``d_state`` is an additive term on the state derivative -- an external
        disturbance acting on the plant, not an input the controller can
        choose. It is held constant across the interval (zero-order hold),
        which is the standard treatment for a signal sampled at dt, and is
        added at every stage so it genuinely perturbs the trajectory rather
        than only its endpoint.
        """
        if d_state is None:
            k1 = self.dynamics(x, u)
            k2 = self.dynamics(x + dt / 2 * k1, u)
            k3 = self.dynamics(x + dt / 2 * k2, u)
            k4 = self.dynamics(x + dt * k3, u)
        else:
            d = np.asarray(d_state, dtype=float)
            k1 = self.dynamics(x, u) + d
            k2 = self.dynamics(x + dt / 2 * k1, u) + d
            k3 = self.dynamics(x + dt / 2 * k2, u) + d
            k4 = self.dynamics(x + dt * k3, u) + d
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class SystemSimulator:
    """Thin, generic open/closed-loop simulator wrapper around a BaseDynamics."""

    def __init__(self, dynamics: BaseDynamics, dt: float):
        self.dynamics = dynamics
        self.dt = dt
        self.config = dynamics.config
        self.n_states = dynamics.n_states
        self.n_inputs = dynamics.n_inputs

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.dynamics.dynamics(x, u)

    def step(self, x: np.ndarray, u: np.ndarray,
             d_state: Optional[np.ndarray] = None) -> np.ndarray:
        return self.dynamics.rk4_step(x, u, self.dt, d_state=d_state)

    def simulate(
        self, x0: np.ndarray, U: np.ndarray, noise_std: float = 0.0, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng()
        n = len(U)
        X = np.zeros((n + 1, self.n_states))
        dX = np.zeros((n, self.n_states))
        X[0] = x0

        for k in range(n):
            dX[k] = self.f(X[k], U[k])
            X[k + 1] = self.step(X[k], U[k])
            if np.any(np.asarray(noise_std) > 0):
                X[k + 1] += rng.normal(scale=noise_std, size=x0.shape)
            if self.dynamics.check_termination(X[k + 1]):
                return X[: k + 2][:-1], dX[: k + 1], U[: k + 1]

        return X[:-1], dX, U
