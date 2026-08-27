"""
================================================================================
mpc/jacobian.py
================================================================================
Linearization of the plugin's dynamics around (x, u).

The original notebook always used central finite differences, calling
``dynamics()`` 2*(n_states + n_inputs) times *every single control step*.
That's correct but slow and numerically noisy (error ~ O(h^2), and h is a
single global constant regardless of each state's scale).

Here we prefer automatic differentiation through torch when it's available
(exact Jacobian, one forward+backward pass instead of 2*(n+p) function
evaluations) and fall back to finite differences otherwise -- e.g. for plugins
whose ``dynamics()`` uses plain numpy/math and isn't torch-differentiable.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def finite_diff_jacobian(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    u: np.ndarray,
    h: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Central-difference Jacobians A = df/dx, B = df/du at (x, u)."""
    n, p = x.size, u.size
    A = np.zeros((n, n))
    B = np.zeros((n, p))

    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = h
        A[:, i] = (f(x + dx, u) - f(x - dx, u)) / (2 * h)

    for j in range(p):
        du = np.zeros_like(u)
        du[j] = h
        B[:, j] = (f(x, u + du) - f(x, u - du)) / (2 * h)

    return A, B


def autograd_jacobian(
    f: Callable, x: np.ndarray, u: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Jacobians via torch autograd. ``f`` must accept and return
    torch tensors and be written with differentiable ops (torch.sin,
    torch.cos, etc. instead of numpy/math)."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is not installed; use finite_diff_jacobian instead.")

    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    u_t = torch.tensor(u, dtype=torch.float64, requires_grad=True)
    dx_t = f(x_t, u_t)

    n = dx_t.shape[0]
    A = torch.zeros((n, x_t.shape[0]), dtype=torch.float64)
    B = torch.zeros((n, u_t.shape[0]), dtype=torch.float64)

    for i in range(n):
        grad_x, grad_u = torch.autograd.grad(dx_t[i], (x_t, u_t), retain_graph=True)
        A[i, :] = grad_x
        B[i, :] = grad_u

    return A.detach().numpy(), B.detach().numpy()


def linearize(
    dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    u: np.ndarray,
    *,
    torch_dynamics_fn: "Callable | None" = None,
    h: float = 1e-4,
    prefer_analytic: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Best-available linearization: analytic (torch) if a torch-compatible
    version of the dynamics is supplied and available, finite-diff otherwise.

    Plugins are not required to provide a torch version -- most won't, and
    that's fine (finite-diff is still correct, just slower/noisier). Plugins
    that do provide one (by implementing ``dynamics_torch(self, x, u)`` next
    to ``dynamics``) get the faster, exact path for free.
    """
    if prefer_analytic and _HAS_TORCH and torch_dynamics_fn is not None:
        try:
            return autograd_jacobian(torch_dynamics_fn, x, u)
        except Exception:
            pass  # fall through to finite-diff if autograd path breaks
    return finite_diff_jacobian(dynamics_fn, x, u, h=h)
