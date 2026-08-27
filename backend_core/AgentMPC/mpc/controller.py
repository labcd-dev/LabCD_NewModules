"""
================================================================================
mpc/controller.py
================================================================================
Generic linear(ized) MPC controller, working with any BaseDynamics plugin.

Fixes applied relative to the original notebook implementation:

  1. TERMINAL WEIGHT (P) IS NOW ACTUALLY USED.
     Previously ``self.P`` was stored (and proposed by the Actor agent every
     iteration!) but the cost matrix ``Qbar`` used ``Q`` for every block,
     including the last one. The Actor was tuning a parameter with zero
     effect on the controller. Here ``Qbar``'s last (terminal) block is ``P``.

  2. STATE BOUNDS ARE NOW ENFORCED.
     ``config.x_bounds`` used to be parsed/validated and then never appear
     in the QP. Predicted states are affine in the decision variable
     (X = Sx @ x0 + Sc + Su @ dU), so the bound is added as extra linear
     rows in the QP -- no extra solver needed.

  3. INPUT BOUNDS ARE ENFORCED ON THE *CUMULATIVE* INPUT, NOT JUST THE FIRST
     MOVE. The original code bounded each ``du_i`` independently to
     ``[umin - u_prev, umax - u_prev]``, which only correctly constrains the
     first step; later steps in the horizon (u_prev + du_0 + ... + du_i) could
     silently leave the box. A block lower-triangular cumulative-sum matrix
     fixes this.

  4. THE QP SOLVER IS WARM-STARTED (see solver.py) INSTEAD OF REBUILT EVERY
     CALL.

  5. LINEARIZATION prefers analytic (torch autograd) Jacobians when the
     plugin provides a torch-compatible dynamics function, falling back to
     central finite differences otherwise (see jacobian.py).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from ..dynamics.base import BaseDynamics
from .config import Config, MPCConfig
from .jacobian import linearize
from .solver import QPSolver


class GenericMPC:
    def __init__(self, dynamics: BaseDynamics, cfg: Config):
        self.dynamics = dynamics
        self.cfg = cfg
        self.n_states = dynamics.n_states
        self.n_inputs = dynamics.n_inputs

        cfg.mpc.initialize_weights(self.n_states, self.n_inputs)
        self._sync_from_config()

        self.umin, self.umax = cfg.mpc.get_u_bounds(self.n_inputs)
        # input bounds default to the plugin's own bounds if the MPC config
        # didn't explicitly override them
        if cfg.mpc.u_bounds is None:
            plugin_bounds = dynamics.get_input_bounds()
            if plugin_bounds is not None:
                self.umin, self.umax = plugin_bounds

        self.xmin, self.xmax = cfg.mpc.get_x_bounds(self.n_states)
        if cfg.mpc.x_bounds is None:
            plugin_state_bounds = dynamics.get_state_bounds()
            if plugin_state_bounds is not None:
                self.xmin, self.xmax = plugin_state_bounds

        self.Ue = dynamics.get_equilibrium_input()
        self.solver = QPSolver()

        # torch-differentiable dynamics is optional; used only if present
        self._torch_dynamics = getattr(dynamics, "dynamics_torch", None)

    # ------------------------------------------------------------------
    def _sync_from_config(self) -> None:
        m = self.cfg.mpc
        self.Np = m.prediction_horizon
        self.Nc = m.control_horizon
        self.Q = m.state_weights
        self.R = m.input_weights
        self.P = m.terminal_weights
        self.dt = self.cfg.data.dt_mpc
        self.h = m.finite_diff_step

    def update_parameters(self, params: Dict[str, Any]) -> None:
        """Apply a parameter proposal (e.g. from the Actor agent) and
        re-sync the controller's cached fields."""
        self.cfg.mpc.set_from_dict(params)
        self.cfg.mpc.initialize_weights(self.n_states, self.n_inputs)
        self._sync_from_config()

    # ------------------------------------------------------------------
    def _linearize(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        A_cont, B_cont = linearize(
            dynamics_fn=self.dynamics.dynamics,
            x=x,
            u=u,
            torch_dynamics_fn=self._torch_dynamics,
            h=self.h,
            prefer_analytic=self.cfg.mpc.use_analytic_jacobian,
        )
        A = np.eye(self.n_states) + self.dt * A_cont
        B = self.dt * B_cont
        return A, B, A_cont, B_cont

    def _prediction_matrices(self, A: np.ndarray, B: np.ndarray):
        """Condensed prediction matrices Sx, Su such that
        X = Sx @ x0 + Su @ dU (+ Sc, added separately for the affine term)."""
        n, p = self.n_states, self.n_inputs
        A_powers = [np.eye(n)]
        for _ in range(self.Np):
            A_powers.append(A @ A_powers[-1])

        A_power_sums = [A_powers[0].copy()]
        for k in range(1, self.Np + 1):
            A_power_sums.append(A_power_sums[-1] + A_powers[k])

        Sx = np.vstack(A_powers[1 : self.Np + 1])

        Su = np.zeros((n * self.Np, p * self.Nc))
        for i in range(self.Np):
            for j in range(min(i + 1, self.Nc)):
                Su[i * n : (i + 1) * n, j * p : (j + 1) * p] = A_power_sums[i - j] @ B

        return Sx, Su

    def _affine_offset(self, A: np.ndarray, A_cont: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        c = self.dt * (self.dynamics.dynamics(x, u) - A_cont @ x)
        blocks = [c.copy()]
        acc = c.copy()
        for _ in range(1, self.Np):
            acc = A @ acc + c
            blocks.append(acc.copy())
        return np.concatenate(blocks)

    def _cumulative_input_matrix(self) -> np.ndarray:
        """Block lower-triangular ones -> maps dU to cumulative input change
        at each control-horizon step: L @ dU = [du0, du0+du1, du0+du1+du2, ...]."""
        p = self.n_inputs
        L = np.zeros((self.Nc * p, self.Nc * p))
        for i in range(self.Nc):
            for j in range(i + 1):
                L[i * p : (i + 1) * p, j * p : (j + 1) * p] = np.eye(p)
        return L

    # ------------------------------------------------------------------
    def control(self, x: np.ndarray, u_prev: np.ndarray, x_ref_seq: np.ndarray) -> np.ndarray:
        """Compute the next control input. Returns absolute input (shape (n_inputs,))."""
        x = np.asarray(x, dtype=float).flatten()
        u_prev = np.asarray(u_prev, dtype=float).flatten()

        A, B, A_cont, _ = self._linearize(x, u_prev)
        Sx, Su = self._prediction_matrices(A, B)
        Sc = self._affine_offset(A, A_cont, x, u_prev)
        x_ref_flat = np.asarray(x_ref_seq, dtype=float).reshape(-1)

        # ---- cost: fix #1, terminal weight P now used on the last block ----
        n = self.n_states
        Qbar = np.kron(np.eye(self.Np), self.Q)
        Qbar[(self.Np - 1) * n :, (self.Np - 1) * n :] = self.P
        Rbar = np.kron(np.eye(self.Nc), self.R)

        H = Su.T @ Qbar @ Su + Rbar
        q = Su.T @ Qbar @ (Sx @ x + Sc - x_ref_flat)
        H = (H + H.T) / 2

        # ---- constraints ----
        rows_A, rows_l, rows_u = [], [], []

        # fix #3: bound the *cumulative* absolute input, not each du in isolation
        L = self._cumulative_input_matrix()
        u_lo = np.tile(self.umin - u_prev, self.Nc)
        u_hi = np.tile(self.umax - u_prev, self.Nc)
        rows_A.append(L)
        rows_l.append(u_lo)
        rows_u.append(u_hi)

        # fix #2: enforce predicted state bounds, if finite
        if np.isfinite(self.xmin).any() or np.isfinite(self.xmax).any():
            x_lo_rep = np.tile(self.xmin, self.Np) - (Sx @ x + Sc)
            x_hi_rep = np.tile(self.xmax, self.Np) - (Sx @ x + Sc)
            rows_A.append(Su)
            rows_l.append(x_lo_rep)
            rows_u.append(x_hi_rep)

        A_ineq = sp.csc_matrix(np.vstack(rows_A))
        l_ineq = np.concatenate(rows_l)
        u_ineq = np.concatenate(rows_u)

        dU = self.solver.solve(sp.csc_matrix(H), q, A_ineq, l_ineq, u_ineq)
        du0 = dU[: self.n_inputs]

        # Linearization is done directly around the *absolute* (x, u_prev),
        # so the equilibrium offset is already implicit in the affine term
        # `Sc` above -- it does not need to be added again here. `self.Ue`
        # is kept available (e.g. to initialize u_prev on the first call, or
        # for logging) rather than silently folded into every step.
        return u_prev + du0

    # ------------------------------------------------------------------
    def get_predicted_trajectory(self, x: np.ndarray, u_prev: np.ndarray, dU: np.ndarray) -> np.ndarray:
        """Helper for diagnostics/plots: reconstruct the predicted state
        trajectory for a given dU (e.g. the one just solved)."""
        A, B, A_cont, _ = self._linearize(x, u_prev)
        Sx, Su = self._prediction_matrices(A, B)
        Sc = self._affine_offset(A, A_cont, x, u_prev)
        X = Sx @ x + Sc + Su @ dU
        return X.reshape(self.Np, self.n_states)
