"""
================================================================================
agents/evaluator.py
================================================================================
Evaluator node: deterministic, no LLM call. Runs a closed-loop simulation
with the Actor's current proposal and computes performance metrics.

Because this node has no LLM dependency, it is also reused directly by
``agents/numeric_tuner.py`` as the objective function for a fast,
LLM-free parameter search.

v2 changes:
  * Generic (dynamics-agnostic) divergence detection: the simulation stops
    early and is flagged ``unstable`` if the state becomes non-finite
    (NaN/Inf) or blows up far beyond its starting magnitude -- this doesn't
    depend on a plugin declaring state_bounds (dynamics.check_termination()
    is still checked too, and also flags ``unstable``, since it means the
    controller failed to respect a declared constraint).
  * ``success`` is now read directly from ``metrics.settled`` (computed
    once, correctly, in metrics.py) instead of being re-derived here with
    an off-by-one comparison that made it always False.
  * "Best so far" is now ranked by ``scalar_cost()`` (MSE + overshoot +
    settling behavior + effort, with a penalty for never actually settling)
    instead of raw MSE, so a run that briefly dips to a low instantaneous
    error while still oscillating doesn't get treated as the best result.
  * Per-state MSE (mapped to the dynamics' own state_names, so this stays
    generic across plugins) is exposed for the Critic/Actor agents, so they
    can target the specific state/weight that's underperforming instead of
    only seeing one aggregate number.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..dynamics.base import BaseDynamics, SystemSimulator
from ..mpc.config import Config
from ..mpc.controller import GenericMPC
from ..utils.logging_utils import get_logger
from .metrics import compute_metrics, scalar_cost

log = get_logger(__name__)


def run_closed_loop(
    dynamics: BaseDynamics,
    cfg: Config,
    params: Dict[str, Any],
    max_steps: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Run one closed-loop MPC simulation with the given parameter proposal.
    Pure function: does not read or write any global state (unlike the
    original notebook's ``set_dynamics_plugin`` globals pattern).

    ``max_steps``: hard cap on the number of control steps, mainly useful for
    a deliberately short/fast check (e.g. app.py's "Test Dynamics" button
    passes a small explicit value). If left as ``None`` (the default), it's
    computed from ``cfg.data.simulation_time / cfg.data.dt_mpc`` -- i.e. the
    full duration the user actually configured is what runs. A previous
    version of this function defaulted ``max_steps`` to a flat 200 regardless
    of ``simulation_time``, which silently truncated every run to a fixed
    ~4s window no matter what the Streamlit UI's "Simulation Time" slider
    said -- that's why settling time looked suspiciously constant across
    very different runs, and why "Stable" rarely triggered (the response
    was being cut off right as it was converging, not given time to hold).

    Honors ``cfg.data.trajectory_mode`` ("reg"/"sin"/"pulse") and
    ``cfg.data.noise_std`` (additive Gaussian measurement noise on the
    realized state after each step) -- both are driven by the Streamlit UI's
    scenario-level / trajectory-type selectors (see agents/scenario_presets.py).

    On ANY failure -- controller construction, trajectory generation, or a
    per-step solve -- returns ``{"error": <message>, "traceback": <full
    traceback text>, "step": <int or None>}`` instead of raising.

    If the state diverges (NaN/Inf, or blows up far beyond its starting
    magnitude -- a generic, dynamics-agnostic check) or a plugin's own
    ``check_termination`` bound is hit, the simulation stops early and the
    result carries ``"unstable": True`` instead of silently running to the
    full horizon on garbage data.
    """
    import traceback as _traceback

    try:
        cfg.mpc.set_from_dict(params)
        controller = GenericMPC(dynamics, cfg)
        simulator = SystemSimulator(dynamics, dt=cfg.data.dt_mpc)
        rng = rng or np.random.default_rng(cfg.random_seed)

        x = dynamics.config.default_initial_state.copy()
        u = (np.asarray(cfg.data.feedforward_override, dtype=float)
             if cfg.data.feedforward_override is not None else dynamics.get_equilibrium_input())

        if cfg.data.trajectory_mode == "custom":
            if cfg.data.custom_trajectory_fn is None:
                return {
                    "error": "trajectory_mode is 'custom' but no custom trajectory function is loaded "
                             "(cfg.data.custom_trajectory_fn is None). Upload a trajectory file first.",
                    "traceback": None, "step": None,
                }
            n_steps_requested = max(int(cfg.data.simulation_time / cfg.data.dt_mpc), 1)
            ref_full = cfg.data.custom_trajectory_fn(
                cfg.data.dt_mpc, cfg.data.simulation_time, dynamics.n_states, dynamics.state_names
            )
            ref_full = np.asarray(ref_full, dtype=float)
            if ref_full.ndim != 2 or ref_full.shape[1] != dynamics.n_states:
                return {
                    "error": f"Custom trajectory function returned shape {ref_full.shape}, "
                             f"expected (n_steps, {dynamics.n_states}).",
                    "traceback": None, "step": None,
                }
            if ref_full.shape[0] < n_steps_requested:
                return {
                    "error": f"Custom trajectory function returned only {ref_full.shape[0]} steps, "
                             f"need at least {n_steps_requested} for simulation_time="
                             f"{cfg.data.simulation_time}s at dt_mpc={cfg.data.dt_mpc}s.",
                    "traceback": None, "step": None,
                }
        else:
            ref_full = dynamics.config.desired_trajectory(
                cfg.data.dt_mpc, cfg.data.simulation_time, mode=cfg.data.trajectory_mode,
                amplitude=cfg.data.trajectory_amplitude, frequency=cfg.data.trajectory_frequency,
                pulse_start=cfg.data.trajectory_pulse_start, pulse_end=cfg.data.trajectory_pulse_end,
                per_state_modes=cfg.data.trajectory_per_state_modes,
            )

        effective_max_steps = max_steps if max_steps is not None else int(
            cfg.data.simulation_time / cfg.data.dt_mpc
        ) + 10

        n_steps = min(effective_max_steps, len(ref_full) - controller.Np)
        if n_steps <= 0:
            return {
                "error": (
                    f"No simulation steps to run: prediction horizon Np={controller.Np} is >= the "
                    f"reference trajectory length ({len(ref_full)} samples, from simulation_time="
                    f"{cfg.data.simulation_time}s / dt_mpc={cfg.data.dt_mpc}s). Increase simulation_time "
                    f"or reduce Np."
                ),
                "traceback": None,
                "step": None,
            }

        # Generic divergence threshold: dynamics-agnostic, doesn't require the
        # plugin to declare state_bounds. Scales with the system's own
        # starting magnitude so it works whether states are ~O(1) radians or
        # ~O(1000) RPM.
        x0_norm = float(np.linalg.norm(x))
        diverge_threshold = max(1e6, 1000.0 * (x0_norm + 1.0))

        states, inputs, solve_times = [x.copy()], [], []
        unstable = False
        unstable_reason = None

        for k in range(n_steps):
            ref_window = ref_full[k : k + controller.Np]
            import time as _time

            t0 = _time.perf_counter()
            u = controller.control(x, u, ref_window)
            solve_times.append(_time.perf_counter() - t0)

            x = simulator.step(x, u)
            if np.any(cfg.data.noise_std > 0):
                x = x + rng.normal(scale=cfg.data.noise_std, size=x.shape)
            states.append(x.copy())
            inputs.append(u.copy())

            if not np.all(np.isfinite(x)):
                unstable, unstable_reason = True, "state became non-finite (NaN/Inf)"
                log.warning("Simulation diverged at step %d: %s", k, unstable_reason)
                break
            if np.linalg.norm(x) > diverge_threshold:
                unstable, unstable_reason = True, f"state magnitude exceeded {diverge_threshold:.3g} (diverged)"
                log.warning("Simulation diverged at step %d: %s", k, unstable_reason)
                break
            if dynamics.check_termination(x):
                unstable, unstable_reason = True, "hit the plugin's declared state bounds"
                log.info("Simulation terminated early at step %d: %s.", k, unstable_reason)
                break

    except Exception as e:  # noqa: BLE001
        tb = _traceback.format_exc()
        log.warning("Closed-loop simulation failed: %s\n%s", e, tb)
        return {"error": str(e) or type(e).__name__, "traceback": tb, "step": None}

    states = np.array(states)
    inputs = np.array(inputs) if inputs else np.zeros((0, dynamics.n_inputs))
    reference = ref_full[: len(states)]
    times = np.arange(len(states)) * cfg.data.dt_mpc

    # is_regulation: is the reference actually constant over time, regardless
    # of *how* it was generated (a fixed mode="reg", a mix via
    # trajectory_per_state_modes, or a custom trajectory file that happens to
    # be constant)? Checking the actual data is more robust than string-
    # matching cfg.data.trajectory_mode, and is the only sane way to handle
    # trajectory_per_state_modes, where some states can be "reg" and others
    # moving at the same time -- if ANY state moves, overshoot/settling/
    # oscillation (regulation-style metrics) aren't meaningful for the whole
    # response, so the reference counts as non-regulation as soon as any
    # component changes over time.
    is_regulation = bool(np.allclose(reference, reference[0]))

    metrics = compute_metrics(
        states, inputs, reference, dt=cfg.data.dt_mpc, settling_tolerance=cfg.data.settling_tolerance,
        is_regulation=is_regulation,
    )

    return {
        "metrics": metrics,
        "states": states,
        "inputs": inputs,
        "reference": reference,
        "times": times,
        "success": metrics.settled,
        "unstable": unstable,
        "unstable_reason": unstable_reason,
        "avg_solve_time": float(np.mean(solve_times)) if solve_times else float("nan"),
        "steps": len(states),
        "solver_diagnostics": dict(controller.solver.diagnostics),
    }


def evaluator_node(state: Dict[str, Any], *, dynamics: BaseDynamics, cfg: Config) -> Dict[str, Any]:
    """LangGraph node wrapper: reads ``state["current_params"]``, writes the
    metric fields back into the graph state.

    ``dynamics``/``cfg`` are bound via ``functools.partial`` when the node is
    registered (see graph/workflow.py) instead of being read from module-level
    globals, which is what made the original notebook's nodes hard to test in
    isolation (``find_dynamics_plugin_global`` walking ``__main__``/globals()).

    All fields returned here MUST be declared in graph/state.py's
    MPCGraphState -- LangGraph builds its state channels strictly from that
    TypedDict, so any key returned here that isn't declared there gets
    silently dropped when the graph merges/streams state. (This bit a
    previous version of this file: metrics/simulation_data/success were
    computed correctly but never reached the UI, because they weren't in the
    schema. If you add a new field here, add it to MPCGraphState too.)
    """
    params = state.get("current_params")
    if not params:
        return {**state, "eval_error": "No current_params in state.", "eval_traceback": None}

    # dt is tunable by the Actor now, exactly like Q/R/Np/Nc (see
    # agents/schemas.py's MPCParameters.dt and agents/actor.py's prompt) --
    # applied here, right before the simulation actually runs, so every
    # subsequent evaluation (and the Juror's own dt_mpc readout) reflects
    # whatever the Actor most recently proposed. Optional: if the Actor's
    # structured output omits it, the previous dt_mpc is left untouched.
    if params.get("dt"):
        cfg.data.dt_mpc = float(params["dt"])

    result = run_closed_loop(dynamics, cfg, params, max_steps=state.get("max_steps"))

    if "error" in result:
        history = state.get("history", []) + [f"[Evaluator] FAILED: {result['error']}"]
        return {**state, "eval_error": result["error"], "eval_traceback": result.get("traceback"), "history": history}

    m = result["metrics"]
    cost = scalar_cost(m, weights=state.get("cost_weights"))
    best_cost = state.get("best_cost", float("inf"))
    is_new_best = (not result["unstable"]) and cost < best_cost

    # Per-metric improvement vs. the best-so-far *before* this iteration
    # (not vs. the immediately preceding iteration, and not MSE-only like
    # the old single Improvement_Pct was) -- positive % = this iteration
    # beat the best result seen so far on that metric, negative % = it's
    # worse. Used to color-code each metric card green/red in the UI.
    def _pct_improvement(current: float, best_before: float) -> float:
        if best_before in (float("inf"), 0) or not np.isfinite(best_before):
            return 0.0  # nothing to compare against yet (first iteration)
        return 100.0 * (best_before - current) / abs(best_before)

    improvement = {
        "MSE": _pct_improvement(m.mse, state.get("best_mse", float("inf"))),
        "Overshoot": _pct_improvement(m.overshoot, state.get("best_overshoot", float("inf"))),
        "Settling_Time": _pct_improvement(m.settling_time, state.get("best_settling", float("inf"))),
        "Control_Effort": _pct_improvement(m.control_effort, state.get("best_effort", float("inf"))),
    }

    mse_history = state.get("mse_history", []) + [m.mse]

    simulation_data = {
        "states": result["states"],
        "inputs": result["inputs"],
        "refs": result["reference"],
        "times": result["times"],
        "n_states": dynamics.n_states,
        "n_inputs": dynamics.n_inputs,
    }

    per_state_mse = dict(zip(dynamics.state_names, m.per_state_mse))
    per_state_overshoot = dict(zip(dynamics.state_names, m.per_state_overshoot))
    per_state_iae = dict(zip(dynamics.state_names, m.per_state_iae))
    per_state_ise = dict(zip(dynamics.state_names, m.per_state_ise))
    per_state_summary = ", ".join(f"{name}={mse:.5f}" for name, mse in per_state_mse.items())
    per_state_ise_summary = ", ".join(f"{name}={ise:.5f}" for name, ise in per_state_ise.items())

    solver_diag = result.get("solver_diagnostics", {})
    n_bad_solves = solver_diag.get("solved_inaccurate", 0) + solver_diag.get("other", 0)
    solver_note = f"  Solver: {solver_diag.get('solved', 0)} clean" + (
        f", {n_bad_solves} inaccurate/other" if n_bad_solves else ""
    )

    status_tag = "UNSTABLE" if result["unstable"] else ("SETTLED" if m.settled else "RUNNING-OUT-CLOCK")
    if m.is_regulation:
        headline = (
            f"MSE={m.mse:.6f}  Overshoot={m.overshoot:.4f}  "
            f"Settling={m.settling_time:.2f}s  Effort={m.control_effort:.4f}  Oscillations={m.oscillation_count}"
        )
    else:
        # Tracking a moving reference (sin/pulse) -- overshoot/oscillation-count
        # aren't meaningful here (see agents/metrics.py), but settled/
        # settling_time ARE (computed relative to the reference signal's own
        # magnitude instead of the initial error) -- lead with the integral
        # error indices, which are the most standard tracking-quality signal.
        headline = (
            f"MSE={m.mse:.6f}  IAE={m.integral_abs_error:.4f}  ISE={m.integral_sq_error:.4f}  "
            f"Settling={m.settling_time:.2f}s  Effort={m.control_effort:.4f}  "
            f"(tracking mode -- overshoot/oscillation-count not applicable)"
        )
    history = state.get("history", []) + [
        f"[Evaluator] {status_tag}  {headline}"
        + (f"  ({result['unstable_reason']})" if result["unstable"] else "")
        + f"\n  Per-state MSE: {per_state_summary}"
        + f"\n  Per-state ISE (integral squared error -- use this to see which state accumulated the "
        + f"most total tracking error over the run): {per_state_ise_summary}"
        + solver_note
    ]

    new_state = {
        **state,
        # -- flat fields consumed by critic.py / terminator.py --
        "current_mse": m.mse,
        "current_overshoot": m.overshoot,
        "current_settling": m.settling_time,
        "current_effort": m.control_effort,
        "current_per_state_mse": per_state_mse,
        "current_per_state_overshoot": per_state_overshoot,
        "current_per_state_iae": per_state_iae,
        "current_per_state_ise": per_state_ise,
        "current_is_regulation": m.is_regulation,
        "current_oscillation_count": m.oscillation_count,
        "current_unstable": result["unstable"],
        "avg_solve_time": result["avg_solve_time"],
        "eval_error": None,
        "mse_history": mse_history,
        "overshoot_history": state.get("overshoot_history", []) + [m.overshoot],
        "settling_history": state.get("settling_history", []) + [m.settling_time],
        "effort_history": state.get("effort_history", []) + [m.control_effort],
        "params_history": state.get("params_history", []) + [params],
        "history": history,
        # -- UI-friendly fields consumed by app.py --
        "metrics": {
            "MSE": m.mse,
            "Max_Overshoot": m.overshoot,
            "Overshoot_Meaningful": m.overshoot_meaningful,
            "Settling_Time": m.settling_time,
            "Settled": m.settled,
            "Is_Stable": m.is_stable,
            "Control_Effort_RMS": m.control_effort,
            "Oscillation_Count": m.oscillation_count,
            "Per_State_MSE": per_state_mse,
            "Per_State_Overshoot": per_state_overshoot,
            "Integral_Abs_Error": m.integral_abs_error,
            "Integral_Sq_Error": m.integral_sq_error,
            "Per_State_IAE": per_state_iae,
            "Per_State_ISE": per_state_ise,
            "Is_Regulation": m.is_regulation,
            "Improvement": improvement,
            "Scenario_Level": state.get("ui_scenario_level", 1),
            "Cost": cost,
            "Unstable": result["unstable"],
            "Unstable_Reason": result["unstable_reason"],
            "Solver_Diagnostics": solver_diag,
            "Dt_Mpc": cfg.data.dt_mpc,
        },
        "simulation_data": simulation_data,
        "success": result["success"],
        "unstable": result["unstable"],
        "exploration_strategy": state.get("strategy", "explore"),
    }

    if is_new_best:
        new_state.update(
            best_mse=m.mse,
            best_cost=cost,
            best_params=params,
            best_overshoot=m.overshoot,
            best_settling=m.settling_time,
            best_effort=m.control_effort,
        )
        log.info("New best (cost=%.6f, mse=%.6f) at iteration %s", cost, m.mse, state.get("iteration"))

    return new_state
