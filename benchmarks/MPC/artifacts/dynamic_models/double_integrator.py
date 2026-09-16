"""Simple 1-DOF position model: acceleration is commanded by one input."""

import numpy as np

from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig


def create_config() -> SystemConfig:
    return SystemConfig(
        n_states=2,
        n_inputs=1,
        params={"mass": 1.0},
        state_names=["position", "velocity"],
        input_names=["force"],
        # Full-state targets are required. The velocity target 0 means
        # "reach the desired position and come to rest."
        default_initial_state=np.array([1.0, 0.0], dtype=float),
        default_target=np.array([0.0, 0.0], dtype=float),
        state_bounds=(
            np.array([-12.0, -8.0], dtype=float),
            np.array([12.0, 8.0], dtype=float),
        ),
        input_bounds=(
            np.array([-10.0], dtype=float),
            np.array([10.0], dtype=float),
        ),
        derivative_pairs=[(0, 1)],
    )


class DoubleIntegratorDynamics(BaseDynamics):
    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        velocity = float(x[1])
        force = float(u[0])
        mass = float(self.params["mass"])
        return np.array([velocity, force / mass], dtype=float)

    def get_equilibrium_input(self) -> np.ndarray:
        return np.array([0.0], dtype=float)

