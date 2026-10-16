"""
Reference plugin: how to separate manipulated inputs from disturbances.

A quadrotor reduced to its vertical and two tilt axes. Three actuators, three
commands -- and the wind is NOT one of them.

WHY THIS FILE EXISTS
--------------------------------------------------------------------------------
The natural-looking way to model wind is to add Fw_z, tau_w_phi and tau_w_theta
to the input vector, giving a six-dimensional u. It is also wrong, and wrong in
a way that quietly invalidates every number a benchmark produces:

  * the MPC's QP optimises over its whole input vector, so it would literally
    CHOOSE the wind -- and it would choose whatever wind makes tracking easy.
    There is nothing left to reject.
  * u_bounds would put constraint rows on the wind, so the solver would treat a
    gust as a bounded actuator.
  * R would carry a weight per wind channel, so the control-effort metric would
    charge the controller for "energy" it never spent, and the Actor would tune
    weights for things that are not actuators.
  * the QP would carry Nc x 6 decision variables instead of Nc x 3 -- twice the
    solve cost, for channels nobody commands.

So this plant declares only what an actuator can actually deliver. The wind is
injected from outside, through ``cfg.data.input_disturbance``, which the
simulator adds AFTER the controller has produced u and which appears nowhere in
the prediction model:

    u_applied = lambda * u + d_u        (lambda: actuator effectiveness)
    x_dot     = f(x, u_applied) + d_x

See agents/evaluator.py:run_closed_loop for where that happens, and Scenario
Level 4 in agents/scenario_presets.py for the UI that sets it.

NOTE: do not import BaseDynamics/SystemConfig/np explicitly when this file is
loaded through DynamicLoader -- they are injected automatically. The imports
below are only needed so the file is also valid on its own.

STATES   z (m), w (m/s, climb rate), phi (rad), theta (rad)
INPUTS   T_cmd (N, thrust above hover), tau_phi_cmd (N.m), tau_theta_cmd (N.m)
"""

try:
    from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig
    import numpy as np
except ImportError:
    pass  # injected by DynamicLoader when loaded as a plugin


def create_config() -> "SystemConfig":
    return SystemConfig(
        n_states=4,
        n_inputs=3,          # exactly the three things an actuator delivers
        params={
            "mass": 1.2,             # kg
            "drag": 0.35,            # N.s/m, vertical aerodynamic drag
            "inertia_phi": 0.015,    # kg.m^2
            "inertia_theta": 0.015,  # kg.m^2
            "tilt_damping": 0.9,     # N.m.s/rad
        },
        state_names=["z", "w", "phi", "theta"],
        input_names=["T_cmd", "tau_phi_cmd", "tau_theta_cmd"],
        default_initial_state=np.array([0.0, 0.0, 0.0, 0.0]),
        # Only z carries a nonzero target. w is its derivative and must be zero
        # at any resting point, and the two tilt angles settle where the physics
        # puts them. Asking for a nonzero w here would not be a hard tuning
        # problem but an impossible one -- see
        # agents/dynamics_validator.analyze_actuation, which now says so out
        # loud in Diagnostics rather than leaving it to be inferred from a run
        # that never converges.
        default_target=np.array([1.0, 0.0, 0.0, 0.0]),
        state_bounds=(
            np.array([-2.0, -5.0, -0.6, -0.6]),
            np.array([6.0, 5.0, 0.6, 0.6]),
        ),
        input_bounds=(
            np.array([-6.0, -0.8, -0.8]),
            np.array([12.0, 0.8, 0.8]),
        ),
    )


class QuadrotorVerticalDynamics(BaseDynamics):
    """Vertical + tilt dynamics of a small quadrotor, linear in its inputs.

    Deliberately simple: the point of this plugin is the input/disturbance
    boundary, not the aerodynamics. Tilt is a first-order response to commanded
    torque, which is enough to make the plant genuinely multi-input without
    obscuring what it is demonstrating.
    """

    def dynamics(self, x: "np.ndarray", u: "np.ndarray") -> "np.ndarray":
        """Continuous-time dx/dt.

        Note what is absent: no wind term, no gust, no actuator-health factor.
        Those are injected by the simulator rather than declared here, precisely
        so that the controller's model -- which is built by differentiating THIS
        function -- cannot contain them.
        """
        mass, drag = self.params["mass"], self.params["drag"]
        i_phi, i_theta = self.params["inertia_phi"], self.params["inertia_theta"]
        tilt_damping = self.params["tilt_damping"]

        _z, w, phi, theta = x
        thrust, tau_phi, tau_theta = u[0], u[1], u[2]

        # T_cmd is thrust ABOVE hover, so gravity is already trimmed out and the
        # origin is an equilibrium. That keeps the linearisation honest, and
        # means the Setup Agent's Bryson weights are measured about the
        # operating point the controller actually sits at.
        return np.array([
            w,
            (thrust - drag * w) / mass,
            (tau_phi - tilt_damping * phi) / i_phi,
            (tau_theta - tilt_damping * theta) / i_theta,
        ])

    def get_equilibrium_input(self) -> "np.ndarray":
        return np.zeros(self.n_inputs)  # hover is trimmed out of T_cmd
