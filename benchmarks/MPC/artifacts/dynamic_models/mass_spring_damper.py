"""Small dynamics plugin used by the MPC walkthrough notebook.

The plant is a controlled mass-spring-damper system:

    mass * position_ddot + damping * position_dot
        + spring_constant * position = force
"""

import numpy as np

from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig


def create_config() -> SystemConfig:
    """Return the complete static description required by DynamicLoader."""
    return SystemConfig(
        n_states=2,
        n_inputs=1,
        params={
            "mass": 1.0,
            "damping": 0.4,
            "spring_constant": 2.0,
        },
        state_names=["position", "velocity"],
        input_names=["force"],
        default_initial_state=np.array([1.0, 0.0], dtype=float),
        # Position is the commanded coordinate. A zero velocity target means
        # the mass should reach that position and remain there at rest.
        default_target=np.array([0.0, 0.0], dtype=float),
        state_bounds=(
            np.array([-8.0, -10.0], dtype=float),
            np.array([8.0, 10.0], dtype=float),
        ),
        input_bounds=(
            np.array([-20.0], dtype=float),
            np.array([20.0], dtype=float),
        ),
        derivative_pairs=[(0, 1)],
    )


class MassSpringDamperDynamics(BaseDynamics):
    """Continuous-time state equation dx/dt = f(x, u)."""

    def __init__(self, config: SystemConfig):
        super().__init__(config=config)

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        position = float(x[0])
        velocity = float(x[1])
        force = float(u[0])

        mass = float(self.params["mass"])
        damping = float(self.params["damping"])
        spring_constant = float(self.params["spring_constant"])

        position_dot = velocity
        velocity_dot = (
            force - damping * velocity - spring_constant * position
        ) / mass

        return np.array([position_dot, velocity_dot], dtype=float)

    def get_equilibrium_input(self) -> np.ndarray:
        return np.array([0.0], dtype=float)

    def get_input_bounds(self):
        return (
            np.array([-20.0], dtype=float),
            np.array([20.0], dtype=float),
        )

    def get_state_bounds(self):
        return (
            np.array([-8.0, -10.0], dtype=float),
            np.array([8.0, 10.0], dtype=float),
        )

    def check_termination(self, x: np.ndarray) -> bool:
        lower, upper = self.get_state_bounds()
        return bool(np.any(x < lower) or np.any(x > upper))