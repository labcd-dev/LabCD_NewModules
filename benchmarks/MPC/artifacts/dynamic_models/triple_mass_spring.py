import numpy as np
from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig

class TripleMassSpringDynamics(BaseDynamics):
    def __init__(self, config: SystemConfig):
        super().__init__(config)
        self.m1 = config.params.get("m1", 1.0)
        self.m2 = config.params.get("m2", 1.0)
        self.m3 = config.params.get("m3", 1.0)
        self.k1 = config.params.get("k1", 1.0)
        self.k2 = config.params.get("k2", 1.0)
        self.k3 = config.params.get("k3", 1.0)
        self.k4 = config.params.get("k4", 1.0)
        self.b1 = config.params.get("b1", 0.1)
        self.b2 = config.params.get("b2", 0.1)
        self.b3 = config.params.get("b3", 0.1)

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # states: x1, v1, x2, v2, x3, v3
        x1, v1, x2, v2, x3, v3 = x
        force = u[0] if u.size > 0 else 0.0

        # Spring forces: wall--k1--m1--k2--m2--k3--m3--k4--wall
        f_spring1 = -self.k1 * x1
        f_spring2 = -self.k2 * (x1 - x2)
        f_spring3 = -self.k3 * (x2 - x3)
        f_spring4 = -self.k4 * x3

        a1 = (f_spring1 + f_spring2 - self.b1 * v1 + force) / self.m1
        a2 = (f_spring3 - self.b2 * v2) / self.m2
        a3 = (f_spring4 - self.b3 * v3) / self.m3

        return np.array([v1, a1, v2, a2, v3, a3], dtype=float)

    def get_state_bounds(self):
        return (np.array([-8.0, -10.0, -8.0, -10.0, -8.0, -10.0]),
                np.array([8.0, 10.0, 8.0, 10.0, 8.0, 10.0]))

    def get_input_bounds(self):
        return (np.array([-20.0]), np.array([20.0]))

def create_config():
    return SystemConfig(
        n_states=6,
        n_inputs=1,
        state_names=["x1", "v1", "x2", "v2", "x3", "v3"],
        input_names=["force"],
        params={"m1":1.0, "m2":1.0, "m3":1.0, "k1":1.0, "k2":1.0, "k3":1.0, "k4":1.0, "b1":0.1, "b2":0.1, "b3":0.1},
        default_initial_state=[0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
        default_target=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        state_bounds=(np.array([-8.0, -10.0, -8.0, -10.0, -8.0, -10.0]),
                     np.array([8.0, 10.0, 8.0, 10.0, 8.0, 10.0])),
        input_bounds=(np.array([-20.0]), np.array([20.0])),
        derivative_pairs=[(0,1), (2,3), (4,5)]
    )
