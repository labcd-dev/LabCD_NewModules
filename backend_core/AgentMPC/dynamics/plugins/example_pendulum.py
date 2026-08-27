"""
Example dynamics plugin: single inverted pendulum on a cart.
Used as a reference implementation and in the test-suite -- it shows the
minimum a real plugin (e.g. Double_Inverted_Pendulum, Overactuated_Quadcopter)
needs to provide.

NOTE: do not import BaseDynamics/SystemConfig/np explicitly when this file is
loaded through DynamicLoader -- they are injected automatically. The imports
below are only needed so the file is also valid on its own (e.g. for editors,
static type checkers, or `python -m` usage).
"""

try:
    from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig
    import numpy as np
except ImportError:
    pass  # will be injected by DynamicLoader when loaded as a plugin


def create_config() -> "SystemConfig":
    return SystemConfig(
        n_states=4,
        n_inputs=1,
        params={"M": 1.0, "m": 0.1, "l": 0.5, "g": 9.81, "b": 0.05},
        state_names=["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"],
        input_names=["force"],
        default_initial_state=np.array([0.0, 0.0, 0.2, 0.0]),
        default_target=np.array([0.0, 0.0, 0.0, 0.0]),
        state_bounds=(
            np.array([-2.0, -5.0, -np.pi / 2, -10.0]),
            np.array([2.0, 5.0, np.pi / 2, 10.0]),
        ),
        input_bounds=(np.array([-10.0]), np.array([10.0])),
    )


class CartPolePendulum(BaseDynamics):
    def dynamics(self, x: "np.ndarray", u: "np.ndarray") -> "np.ndarray":
        M, m, l, g, b = (self.params[k] for k in ("M", "m", "l", "g", "b"))
        _, x_dot, theta, theta_dot = x
        F = u[0]

        sin_t, cos_t = np.sin(theta), np.cos(theta)
        den = M + m * sin_t**2

        x_ddot = (F + m * sin_t * (l * theta_dot**2 + g * cos_t) - b * x_dot) / den
        theta_ddot = (-F * cos_t - m * l * theta_dot**2 * cos_t * sin_t - (M + m) * g * sin_t) / (l * den)

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def get_equilibrium_input(self) -> "np.ndarray":
        return np.zeros(self.n_inputs)  # upright equilibrium requires zero force
