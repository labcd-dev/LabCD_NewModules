"""Complex nonlinear planar drone with horizontal, vertical and attitude DOFs."""

import numpy as np

from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig


def create_config() -> SystemConfig:
    mass = 1.2
    gravity = 9.81
    return SystemConfig(
        n_states=6,
        n_inputs=2,
        params={
            "mass": mass,
            "gravity": gravity,
            "inertia": 0.08,
            "linear_drag": 0.15,
            "angular_drag": 0.04,
        },
        state_names=["x", "vx", "z", "vz", "pitch", "pitch_rate"],
        input_names=["total_thrust", "pitch_torque"],
        default_initial_state=np.array([1.0, 0.0, 1.0, 0.0, 0.15, 0.0], dtype=float),
        # Controlled position/configuration states are x, z and pitch at
        # indices 0, 2 and 4. Their derivative targets are zero.
        default_target=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float),
        state_bounds=(
            np.array([-15.0, -10.0, 0.0, -10.0, -1.0, -6.0], dtype=float),
            np.array([15.0, 10.0, 20.0, 10.0, 1.0, 6.0], dtype=float),
        ),
        input_bounds=(
            np.array([0.0, -4.0], dtype=float),
            np.array([3.0 * mass * gravity, 4.0], dtype=float),
        ),
        derivative_pairs=[(0, 1), (2, 3), (4, 5)],
    )


class PlanarDroneDynamics(BaseDynamics):
    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        _, vx, _, vz, pitch, pitch_rate = map(float, x)
        total_thrust, pitch_torque = map(float, u)
        p = self.params

        ax = (
            -total_thrust * np.sin(pitch) - p["linear_drag"] * vx
        ) / p["mass"]
        az = (
            total_thrust * np.cos(pitch) - p["mass"] * p["gravity"]
            - p["linear_drag"] * vz
        ) / p["mass"]
        pitch_acceleration = (
            pitch_torque - p["angular_drag"] * pitch_rate
        ) / p["inertia"]

        return np.array([vx, ax, vz, az, pitch_rate, pitch_acceleration], dtype=float)

    def get_equilibrium_input(self) -> np.ndarray:
        hover_thrust = self.params["mass"] * self.params["gravity"]
        return np.array([hover_thrust, 0.0], dtype=float)

