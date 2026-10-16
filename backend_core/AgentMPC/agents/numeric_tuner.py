"""
================================================================================
agents/numeric_tuner.py
================================================================================
A small, dependency-light numeric optimizer over MPC parameters (Q, R, Np,
Nc). 

*** THIS MODULE IS NOT USED BY THE DEFAULT TUNING GRAPH. ***
``graph/workflow.py``'s ``build_mpc_tuning_graph`` / ``build_ui_tuning_graph``
(used by both run_agents.py and app.py) never import anything from this file.
The Actor's parameter proposals are, and always have been, 100% LLM-driven --
this is a separate, optional utility offered as an alternative or complement,
not a replacement, for anyone who wants a fast/free/offline baseline search.
If you never call anything in this file, nothing about your Agent-driven
tuning loop changes.

This is NOT the same thing as the QP solver in mpc/solver.py (OSQP). That one
is a different, unavoidable piece: given a *specific* Q/R/Np/Nc (however it
was chosen -- by the LLM, by hand, by this file, whatever), actually running
one closed-loop simulation with it requires computing the control input `u`
at every timestep, which is what MPC mathematically *is* -- solving a
receding-horizon optimization problem at each step. That has nothing to do
with how the parameters themselves are chosen, and removing it isn't
possible without abandoning MPC itself. Confusing the two is an easy mistake
-- "a numeric solver is involved somewhere" -- but they answer completely
different questions:
    - mpc/solver.py (OSQP): "given these Q/R/Np/Nc, what's the best control
      action *right now*?" -- runs on every timestep, unavoidable.
    - agents/numeric_tuner.py (this file): "what Q/R/Np/Nc should we try
      *next*?" -- an optional alternative to the LLM Actor, off by default.

Why this exists (see the review of the original notebook): the LLM
Actor/Critic/Terminator/Juror loop can call an LLM 3-4 times per tuning
iteration. That's slow, non-deterministic, and costs money/tokens for what is,
numerically, a fairly standard bounded search problem. This module lets you:

  * run a cheap random/coordinate search purely in Python to get a decent
    starting point (or the final answer) before ever calling an LLM, and/or
  * let the LLM operate at a coarser "strategy" level (explore vs exploit,
    which state matters most) while this module handles the fine numeric
    search within whatever region the LLM points it at.

This intentionally does not depend on optuna/skopt so the package has no
extra hard dependency; swapping in Optuna's TPE sampler behind the same
``search()`` interface is a natural upgrade if that dependency is acceptable
in your environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..dynamics.base import BaseDynamics
from ..mpc.config import Config
from ..utils.logging_utils import get_logger
from .evaluator import run_closed_loop
from .metrics import scalar_cost

log = get_logger(__name__)

ObjectiveFn = Callable[[Dict[str, Any]], float]


@dataclass
class SearchBounds:
    # log10 ranges. Wide on purpose: only the Q/R ratio matters, and the
    # weights that make a fast plant converge can sit at ratios of 1e4..1e11
    # (quadcopter campaign). The old (-2, 3) x (-3, 1) capped the ratio at
    # 1e6 and excluded the user's hand-tuned solution outright (bug 11).
    q_log_range: Tuple[float, float] = (-2.0, 6.0)   # Q entries searched in log10-space
    r_log_range: Tuple[float, float] = (-6.0, 1.0)
    np_range: Tuple[int, int] = (5, 30)
    nc_range: Tuple[int, int] = (2, 15)


@dataclass
class SearchResult:
    best_params: Dict[str, Any]
    best_cost: float
    history: List[Tuple[Dict[str, Any], float]] = field(default_factory=list)


def make_objective(
    dynamics: BaseDynamics, cfg: Config, cost_weights: Optional[dict] = None, max_steps: int = 150
) -> ObjectiveFn:
    """Build a scalar objective function params -> cost, for use with search()
    or with any external optimizer (Optuna, scipy.optimize, CMA-ES, ...)."""

    def objective(params: Dict[str, Any]) -> float:
        result = run_closed_loop(dynamics, cfg, params, max_steps=max_steps)
        if "error" in result:
            return 1e6  # heavily penalize solver/simulation failures
        return scalar_cost(result["metrics"], weights=cost_weights)

    return objective


def _sample_params(n_states: int, n_inputs: int, bounds: SearchBounds, rng: np.random.Generator) -> Dict[str, Any]:
    q = 10 ** rng.uniform(*bounds.q_log_range, size=n_states)
    r = 10 ** rng.uniform(*bounds.r_log_range, size=n_inputs)
    Np = int(rng.integers(bounds.np_range[0], bounds.np_range[1] + 1))
    Nc = int(rng.integers(bounds.nc_range[0], min(bounds.nc_range[1], Np) + 1))
    return {"Np": Np, "Nc": Nc, "Q": q.tolist(), "R": r.tolist(), "P": q.tolist()}


def random_search(
    objective: ObjectiveFn,
    n_states: int,
    n_inputs: int,
    n_trials: int = 40,
    bounds: Optional[SearchBounds] = None,
    seed: int = 0,
) -> SearchResult:
    """Plain random search over the parameter space -- a strong, simple
    baseline for MPC weight tuning (the response surface is usually noisy and
    non-convex enough that gradient methods aren't obviously better)."""
    bounds = bounds or SearchBounds()
    rng = np.random.default_rng(seed)

    best_params: Optional[Dict[str, Any]] = None
    best_cost = float("inf")
    history: List[Tuple[Dict[str, Any], float]] = []

    for trial in range(n_trials):
        params = _sample_params(n_states, n_inputs, bounds, rng)
        cost = objective(params)
        history.append((params, cost))
        if cost < best_cost:
            best_cost = cost
            best_params = params
            log.info("[numeric_tuner] trial %d: new best cost=%.6f  Np=%d Nc=%d", trial, cost, params["Np"], params["Nc"])

    assert best_params is not None
    return SearchResult(best_params=best_params, best_cost=best_cost, history=history)


def coordinate_refine(
    objective: ObjectiveFn,
    start_params: Dict[str, Any],
    n_iters: int = 20,
    step_scale: float = 0.3,
    seed: int = 1,
) -> SearchResult:
    """Local coordinate-descent-style refinement around a starting point
    (e.g. the LLM Actor's last proposal, or random_search's winner): perturb
    one weight at a time (multiplicatively, in log-space) and keep the move
    only if it improves the cost. Cheap, robust, no gradients required."""
    rng = np.random.default_rng(seed)
    params = {**start_params, "Q": list(start_params["Q"]), "R": list(start_params["R"])}
    cost = objective(params)
    history = [({**params}, cost)]

    keys = [("Q", i) for i in range(len(params["Q"]))] + [("R", i) for i in range(len(params["R"]))]

    for _ in range(n_iters):
        key, idx = keys[rng.integers(len(keys))]
        trial_params = {**params, key: list(params[key])}
        factor = float(np.exp(rng.normal(scale=step_scale)))
        trial_params[key][idx] = max(trial_params[key][idx] * factor, 1e-6)

        trial_cost = objective(trial_params)
        history.append((trial_params, trial_cost))
        if trial_cost < cost:
            params, cost = trial_params, trial_cost

    return SearchResult(best_params=params, best_cost=cost, history=history)


def grouped_search(
    objective: ObjectiveFn,
    n_states: int,
    n_inputs: int,
    groups: List[List[int]],
    n_trials: int = 48,
    bounds: Optional[SearchBounds] = None,
    Np: int = 20,
    Nc: int = 4,
    seed: int = 0,
    local_fraction: float = 1.0 / 3.0,
    local_sigma: float = 0.5,
) -> SearchResult:
    """Global search over ONE weight per group of states, plus one R.

    Searching every Q and R entry independently (random_search) spends the
    budget on a 20-dimensional space where almost nothing converges: on the
    quadcopter campaign it found no converging weights in 48 simulations.
    States that play the same physical role -- the three positions, the three
    attitude angles -- need the same kind of weight, so sharing one exponent
    per group cuts the space to len(groups) + 1 dimensions (bug 11).

    Two stages, all in log10 space:
      1. a Latin hypercube over the whole box -- every exponent's range is
         covered evenly, not left to luck;
      2. Gaussian steps of ``local_sigma`` decades around the best point so
         far, for the last ``local_fraction`` of the budget.
    States missing from ``groups`` keep weight 1.
    """
    bounds = bounds or SearchBounds()
    rng = np.random.default_rng(seed)
    dims = len(groups) + 1
    lo = np.array([bounds.q_log_range[0]] * len(groups) + [bounds.r_log_range[0]])
    hi = np.array([bounds.q_log_range[1]] * len(groups) + [bounds.r_log_range[1]])

    def params_from(e):
        q = np.ones(n_states)
        for g, ex in zip(groups, e[:-1]):
            q[list(g)] = 10.0 ** ex
        return {"Np": int(Np), "Nc": int(Nc), "Q": q.tolist(), "R": [float(10.0 ** e[-1])] * n_inputs,
                "P": q.tolist()}

    n_global = max(1, int(round(n_trials * (1.0 - local_fraction))))
    strata = np.stack([rng.permutation(n_global) for _ in range(dims)], axis=1)
    E = lo + (strata + rng.uniform(size=(n_global, dims))) / n_global * (hi - lo)

    history: List[Tuple[Dict[str, Any], float]] = []
    best_e, best_cost = None, float("inf")
    for k in range(n_trials):
        if k < n_global:
            e = E[k]
        else:
            e = np.clip(best_e + rng.normal(scale=local_sigma, size=dims), lo, hi)
        params = params_from(e)
        cost = objective(params)
        history.append((params, cost))
        if cost < best_cost:
            best_e, best_cost = e, cost
            log.info("[numeric_tuner] grouped trial %d: new best cost=%.6f", k, cost)
    return SearchResult(best_params=params_from(best_e), best_cost=best_cost, history=history)
