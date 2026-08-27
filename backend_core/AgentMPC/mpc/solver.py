"""
================================================================================
mpc/solver.py
================================================================================
QP backend for the MPC controller, with actual warm-starting.

The original notebook did this on *every single control step*:

    def setup(self, P, q, A, l, u):
        self.solver = osqp.OSQP()
        self.solver.setup(P, q, A, l, u, verbose=False)

i.e. it threw away the solver and re-ran the full setup/factorization from
scratch every time, even though the sparsity pattern of P and A almost never
changes step to step (only the numeric values do, and even those change
smoothly). That's the single biggest performance bug in the controller for
any real-time use case.

Here, ``QPSolver.solve()``:
  * builds the problem once (``osqp.OSQP().setup(...)``) the first time it's
    called, or whenever the sparsity *pattern* of P/A changes (e.g. Np/Nc
    changed after an Actor proposal),
  * afterwards calls ``solver.update(Px=..., q=..., l=..., u=...)`` which
    reuses the existing KKT factorization and warm-starts from the previous
    solution -- this is the documented OSQP pattern for solving a sequence of
    related QPs (MPC) and is dramatically cheaper per step.

If OSQP isn't installed in the current environment, a small dense
box-constrained least-squares fallback is used instead so the rest of the
package stays importable and testable. It is *not* a full QP solver (it
ignores general linear inequality structure beyond simple box bounds) --
install osqp for real use.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp

try:
    import osqp

    _HAS_OSQP = True
except ImportError:  # pragma: no cover
    _HAS_OSQP = False


class QPSolver:
    """Solves min 0.5 x'Px + q'x  s.t.  l <= Ax <= u, with warm-starting
    across calls that share the same sparsity pattern."""

    def __init__(
        self,
        eps_abs: float = 1e-4,
        eps_rel: float = 1e-4,
        max_iter: int = 4000,
        polish: bool = False,
    ):
        self._solver: Optional["osqp.OSQP"] = None
        self._pattern_key: Optional[tuple] = None
        self._settings = dict(
            verbose=False,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
            # Polishing is an optional post-processing step that refines the
            # solution when some inequality constraints are active at the
            # optimum. It's off by default here: (a) OSQP's C library prints
            # "Polishing not needed - no active set detected at optimal
            # point" directly to stdout whenever it's on and no constraint
            # happens to be active -- which is common and harmless, but
            # ignores `verbose=False` and floods the log during a long
            # tuning run; (b) the default eps_abs/eps_rel above are already
            # tight enough for MPC control accuracy without it. Pass
            # `polish=True` if you need the extra accuracy for a
            # constraint-heavy problem and don't mind the noisier log.
            polish=polish,
            adaptive_rho=True,
            warm_start=True,
        )
        # Solve-quality diagnostics, accumulated across every solve() call on
        # this instance (i.e. across a whole closed-loop simulation, since a
        # GenericMPC/QPSolver is created fresh per run_closed_loop call).
        # Answers "did the numerical solver actually converge cleanly at
        # every step, or was it struggling (inaccurate solves) or outright
        # failing (infeasible/unsolved)?" -- separate from whether the
        # *simulation* succeeded, since an inaccurate-but-usable solve
        # doesn't raise.
        self.diagnostics: Dict[str, int] = {"solved": 0, "solved_inaccurate": 0, "other": 0}
        self.other_status_counts: Dict[str, int] = {}

    @staticmethod
    def _sparsity_key(P: sp.spmatrix, A: sp.spmatrix) -> tuple:
        Pc, Ac = P.tocsc(), A.tocsc()
        return (Pc.shape, tuple(Pc.indptr), tuple(Pc.indices), Ac.shape, tuple(Ac.indptr), tuple(Ac.indices))

    def solve(self, P: sp.spmatrix, q: np.ndarray, A: sp.spmatrix, l: np.ndarray, u: np.ndarray) -> np.ndarray:
        if not _HAS_OSQP:
            return self._dense_box_fallback(P, q, A, l, u)

        # OSQP requires P to be the UPPER TRIANGULAR part of the (symmetric)
        # cost matrix in CSC format -- it silently only stores/updates the
        # upper-triangular nonzeros internally. Passing the full symmetric
        # matrix (as this used to do) makes `setup()` retain some nnz count
        # N (whatever the upper triangle happens to have), while a later
        # `update()` call keeps trying to push the *full* matrix's nnz count
        # -- which is bigger and doesn't match, causing:
        #   "ERROR in osqp_update_data_mat: new number of elements (...)
        #    out of bounds for P (... max)"
        # every single call after the first. Using sp.triu(...) here makes
        # what we compute the sparsity key from, what we call setup() with,
        # and what we call update() with, all consistent.
        P = sp.triu(sp.csc_matrix(P), format="csc")
        A = sp.csc_matrix(A)
        key = self._sparsity_key(P, A)

        if self._solver is None or key != self._pattern_key:
            self._solver = osqp.OSQP()
            self._solver.setup(P, q, A, l, u, **self._settings)
            self._pattern_key = key
        else:
            # same structure as last time -> cheap warm-started update
            self._solver.update(Px=P.data, q=q, Ax=A.data, l=l, u=u)

        result = self._solver.solve()
        status = result.info.status

        if status == "solved":
            self.diagnostics["solved"] += 1
        elif status == "solved inaccurate":
            self.diagnostics["solved_inaccurate"] += 1
        else:
            self.diagnostics["other"] += 1
            self.other_status_counts[status] = self.other_status_counts.get(status, 0) + 1
            raise RuntimeError(f"QP solve failed: {status}")

        return result.x

    def _dense_box_fallback(self, P: sp.spmatrix, q: np.ndarray, A: sp.spmatrix, l: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Dense QP fallback used only when osqp isn't installed, via
        scipy.optimize.minimize (SLSQP). MPC condensed QPs are small
        (Nc*n_inputs variables), so this is adequate for development/tests;
        it is not a substitute for OSQP in production (no warm-start, slower,
        less robust on larger horizons)."""
        from scipy.optimize import minimize, LinearConstraint

        Pd = np.asarray(P.todense()) if sp.issparse(P) else np.asarray(P)
        Ad = np.asarray(A.todense()) if sp.issparse(A) else np.asarray(A)
        reg = 1e-9 * np.eye(Pd.shape[0])
        Pd = Pd + reg

        x0 = np.zeros(Pd.shape[0])

        def obj(v):
            return 0.5 * v @ Pd @ v + q @ v

        def grad(v):
            return Pd @ v + q

        # Rows that are unbounded on both sides (e.g. a plugin with no
        # declared input/state bounds -> l=-inf, u=+inf) carry no actual
        # constraint information. Passing an all-infinite LinearConstraint
        # to SciPy's SLSQP wrapper triggers an internal bug (IndexError:
        # list index out of range) in some SciPy versions when *every* row
        # is like this. Drop those rows -- they were never real constraints
        # -- and fall back to an unconstrained solve if nothing is left.
        finite_rows = ~(np.isneginf(l) & np.isposinf(u))
        if not np.any(finite_rows):
            res = minimize(obj, x0, jac=grad, method="BFGS", options={"maxiter": 200})
            (self.diagnostics.__setitem__("solved", self.diagnostics["solved"] + 1) if res.success
             else self.diagnostics.__setitem__("other", self.diagnostics["other"] + 1))
            return res.x

        constraint = LinearConstraint(Ad[finite_rows], l[finite_rows], u[finite_rows])
        res = minimize(obj, x0, jac=grad, constraints=[constraint], method="SLSQP",
                        options={"maxiter": 200, "ftol": 1e-9})
        if res.success:
            self.diagnostics["solved"] += 1
        else:
            self.diagnostics["other"] += 1
            self.other_status_counts[res.message] = self.other_status_counts.get(res.message, 0) + 1
        return res.x
