"""Intermediate 2-DOF model: two actuated masses coupled by a spring/damper."""

import numpy as np

from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig


def create_config() -> SystemConfig:
    return SystemConfig(
        n_states=4,
        n_inputs=2,
        params={
            "mass_1": 1.0,
            "mass_2": 1.5,
            "ground_spring_1": 1.0,
            "ground_spring_2": 1.2,
            "coupling_spring": 2.0,
            "ground_damping_1": 0.25,
            "ground_damping_2": 0.30,
            "coupling_damping": 0.20,
        },
        state_names=["position_1", "velocity_1", "position_2", "velocity_2"],
        input_names=["force_1", "force_2"],
        default_initial_state=np.array([1.0, 0.0, -0.5, 0.0], dtype=float),
        # Position targets are indices 0 and 2. Velocity targets at indices
        # 1 and 3 are zero so both masses settle rather than pass the target.
        default_target=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        state_bounds=(
            np.array([-8.0, -10.0, -8.0, -10.0], dtype=float),
            np.array([8.0, 10.0, 8.0, 10.0], dtype=float),
        ),
        input_bounds=(
            np.array([-20.0, -20.0], dtype=float),
            np.array([20.0, 20.0], dtype=float),
        ),
        derivative_pairs=[(0, 1), (2, 3)],
    )


class TwoMassSpringDynamics(BaseDynamics):
    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x1, v1, x2, v2 = map(float, x)
        f1, f2 = map(float, u)
        p = self.params

        coupling_force_on_1 = (
            -p["coupling_spring"] * (x1 - x2)
            - p["coupling_damping"] * (v1 - v2)
        )
        coupling_force_on_2 = -coupling_force_on_1

        a1 = (
            f1
            - p["ground_spring_1"] * x1
            - p["ground_damping_1"] * v1
            + coupling_force_on_1
        ) / p["mass_1"]
        a2 = (
            f2
            - p["ground_spring_2"] * x2
            - p["ground_damping_2"] * v2
            + coupling_force_on_2
        ) / p["mass_2"]

        return np.array([v1, a1, v2, a2], dtype=float)

    def get_equilibrium_input(self) -> np.ndarray:
        return np.array([0.0, 0.0], dtype=float)

