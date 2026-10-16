"""
================================================================================
agents/export_script.py
================================================================================
Builds a single, self-contained .py file the user can download and run on
their OWN machine (only numpy/scipy/matplotlib required -- no osqp, no
torch, no LangGraph/LangChain, no Streamlit) to reproduce the exact final
tuning result: the same dynamics, the same MPC formulation, the same best
parameters, and the same state/input/metric plots.

Fidelity approach: rather than hand-retyping a "simplified" MPC controller
(which risks silently drifting from what was actually used to tune the
parameters), this reads the ACTUAL source of dynamics/base.py, mpc/config.py,
mpc/jacobian.py, mpc/solver.py and mpc/controller.py live from disk and
embeds them verbatim (only their cross-module relative imports are stripped,
since everything ends up in one flat file/namespace). osqp and torch stay
optional exactly as in the app: input-bound-only QPs are solved by the
solver's exact numpy/scipy path, and without osqp a state-constrained QP
falls back to the solver's own dense method.

The driver reproduces the evaluator's closed loop step for step -- the
nominal model the controller believed vs. the plant that was simulated, the
disturbance observer, actuator faults, disturbances (sampled per step, so a
time-varying one is reproduced too), measurement noise with the run's seed,
and the reference the run actually tracked. This used to be a different QP
method (SLSQP) and a nominal-only scenario, and on the quadcopter campaign the
exported script's MSE was 5x the app's (campaign bug 8).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parent.parent  # AgentMPC/

_RELATIVE_IMPORT_RE = re.compile(r"^from \.+[\w.]* import .*$", re.MULTILINE)
_FUTURE_IMPORT_RE = re.compile(r"^from __future__ import .*$", re.MULTILINE)


def _read_stripped(relative_path: str) -> str:
    """Reads a source file from the AgentMPC package and strips its
    relative (from . / from ..) imports -- those targets get inlined into
    the same flat script instead, so the imports would otherwise fail --
    and its own `from __future__ import annotations` line, since Python
    only allows that exact statement once, at the very top of a file, and
    the generated script already has its own copy there."""
    text = (_PKG_ROOT / relative_path).read_text(encoding="utf-8")
    text = _RELATIVE_IMPORT_RE.sub("", text)
    text = _FUTURE_IMPORT_RE.sub("", text)
    return text.strip()


_DRIVER_TEMPLATE = '''
# ============================================================================
# DRIVER -- runs the closed loop with the tuned parameters below, on the
# SAME scenario this was tuned against, step for step the way the app's
# evaluator runs it, and plots states, inputs and a short metrics summary.
# Edit anything below and re-run to explore further.
# ============================================================================

BEST_PARAMS = {best_params!r}
DT_MPC = {dt_mpc!r}
SIMULATION_TIME = {simulation_time!r}
FEEDFORWARD_OVERRIDE = {feedforward!r}   # None unless the "Use computed feedforward trim input" toggle was on

# The plant that was SIMULATED and the model the controller BELIEVED. They
# differ when a robustness scenario perturbed the plant: the controller was
# never told the true parameters, and neither is it here.
INITIAL_STATE = {initial_state!r}
PHYSICAL_PARAMS_OVERRIDE = {physical_params_override!r}   # the plant; None = the file's own defaults
NOMINAL_PARAMS = {nominal_params!r}   # the controller's model; None = same as the plant
DERIVATIVE_PAIRS = {derivative_pairs!r}

# The reference trajectory actually used.
TRAJECTORY_MODE = {trajectory_mode!r}
TRAJECTORY_AMPLITUDE = {trajectory_amplitude!r}
TRAJECTORY_FREQUENCY = {trajectory_frequency!r}
TRAJECTORY_PULSE_START = {trajectory_pulse_start!r}
TRAJECTORY_PULSE_END = {trajectory_pulse_end!r}
TRAJECTORY_PER_STATE_MODES = {trajectory_per_state_modes!r}   # None unless "Customize per state" was used
REFERENCE = {reference!r}   # the full reference, embedded when it came from an uploaded file

# Constraint bounds actually enforced during tuning (None = whatever the
# dynamics file itself declares, if anything).
U_BOUNDS = {u_bounds!r}
X_BOUNDS = {x_bounds!r}

# What the plant received besides the controller's command, one row per
# control step (constant ones repeat), and the measurement noise.
TERMINAL_WEIGHT_MODE = {terminal_weight_mode!r}   # "dare": P from the Riccati equation each step
USE_DISTURBANCE_OBSERVER = {use_disturbance_observer!r}
OBSERVER_GAIN = {observer_gain!r}
ACTUATOR_EFFECTIVENESS = {actuator_effectiveness!r}   # plant receives effectiveness * u
INPUT_DISTURBANCE = {input_disturbance!r}
STATE_DISTURBANCE = {state_disturbance!r}   # added to dx/dt during each step
NOISE_STD = {noise_std!r}
RANDOM_SEED = {random_seed!r}


def _per_step(table, k, width, default):
    if table is None:
        return default
    row = table[min(k, len(table) - 1)]
    return np.broadcast_to(np.asarray(row, dtype=float), (width,)).copy()


def main():
    plant = {class_name}(create_config())
    if PHYSICAL_PARAMS_OVERRIDE is not None:
        plant.params.update(PHYSICAL_PARAMS_OVERRIDE)
    model = {class_name}(create_config())
    model.params.update(NOMINAL_PARAMS if NOMINAL_PARAMS is not None else (PHYSICAL_PARAMS_OVERRIDE or {{}}))
    for d in (plant, model):
        d.config.derivative_pairs = [tuple(p) for p in DERIVATIVE_PAIRS] if DERIVATIVE_PAIRS else None

    cfg = Config()
    cfg.mpc.prediction_horizon = BEST_PARAMS["Np"]
    cfg.mpc.control_horizon = BEST_PARAMS["Nc"]
    cfg.mpc.state_weights = np.diag(np.asarray(BEST_PARAMS["Q"], dtype=float))
    cfg.mpc.input_weights = np.diag(np.asarray(BEST_PARAMS["R"], dtype=float))
    cfg.mpc.terminal_weights = np.diag(np.asarray(BEST_PARAMS.get("P") or BEST_PARAMS["Q"], dtype=float))
    cfg.mpc.terminal_weight_mode = TERMINAL_WEIGHT_MODE
    cfg.mpc.use_disturbance_observer = bool(USE_DISTURBANCE_OBSERVER)
    cfg.mpc.observer_gain = float(OBSERVER_GAIN)
    cfg.data.dt_mpc = DT_MPC
    cfg.data.simulation_time = SIMULATION_TIME
    cfg.random_seed = RANDOM_SEED
    if U_BOUNDS is not None:
        cfg.mpc.u_bounds = (np.asarray(U_BOUNDS[0], dtype=float), np.asarray(U_BOUNDS[1], dtype=float))
    if X_BOUNDS is not None:
        cfg.mpc.x_bounds = (np.asarray(X_BOUNDS[0], dtype=float), np.asarray(X_BOUNDS[1], dtype=float))

    controller = GenericMPC(model, cfg)
    simulator = SystemSimulator(plant, dt=DT_MPC)
    Np = cfg.mpc.prediction_horizon

    x = (np.asarray(INITIAL_STATE, dtype=float) if INITIAL_STATE is not None
         else plant.config.default_initial_state.copy())
    u = (np.asarray(FEEDFORWARD_OVERRIDE, dtype=float) if FEEDFORWARD_OVERRIDE is not None
         else plant.get_equilibrium_input())

    n_steps = max(int(round(SIMULATION_TIME / DT_MPC)), 1)
    # Np samples past the end, so the look-ahead window exists on the last
    # step too -- same as the app's evaluator.
    if REFERENCE is not None:
        ref_full = np.asarray(REFERENCE, dtype=float)[:n_steps]
        ref_full = np.vstack([ref_full, np.repeat(ref_full[-1:], Np, axis=0)])
    else:
        ref_full = plant.config.desired_trajectory(
            DT_MPC, SIMULATION_TIME, mode=TRAJECTORY_MODE or "reg", amplitude=TRAJECTORY_AMPLITUDE,
            frequency=TRAJECTORY_FREQUENCY, pulse_start=TRAJECTORY_PULSE_START,
            pulse_end=TRAJECTORY_PULSE_END, per_state_modes=TRAJECTORY_PER_STATE_MODES, extend_steps=Np,
        )

    n_u, n_x = plant.n_inputs, plant.n_states
    noise = np.zeros(n_x) if NOISE_STD is None else np.broadcast_to(np.asarray(NOISE_STD, dtype=float), (n_x,))
    rng = np.random.default_rng(RANDOM_SEED)
    diverge_threshold = max(1e6, 1000.0 * (float(np.linalg.norm(x)) + 1.0))

    states, inputs = [x.copy()], []
    for k in range(n_steps):
        u = controller.control(x, u, ref_full[k: k + Np])
        u_applied = (_per_step(ACTUATOR_EFFECTIVENESS, k, n_u, np.ones(n_u)) * u
                     + _per_step(INPUT_DISTURBANCE, k, n_u, np.zeros(n_u)))
        x = simulator.step(x, u_applied, d_state=_per_step(STATE_DISTURBANCE, k, n_x, np.zeros(n_x)))
        if np.any(noise > 0):
            x = x + rng.normal(scale=noise, size=x.shape)
        states.append(x.copy())
        inputs.append(u.copy())
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > diverge_threshold:
            print(f"Stopped early at step {{k}}: the state diverged.")
            break
        if plant.check_termination(x):
            print(f"Stopped early at step {{k}}: hit a declared state bound.")
            break

    states = np.array(states)
    inputs = np.array(inputs)
    times = np.arange(len(states)) * DT_MPC

    errors = states - ref_full[:len(states), :n_x]
    mse = float(np.mean(errors ** 2))
    final_error = np.linalg.norm(errors[-1])
    print(f"System: {class_name}")
    print(f"Scenario: trajectory={{TRAJECTORY_MODE or 'reg'}}, "
          f"plant params={{'perturbed' if NOMINAL_PARAMS is not None else 'as the model'}}, "
          f"observer={{'on' if USE_DISTURBANCE_OBSERVER else 'off'}}")
    print(f"Steps simulated: {{len(states) - 1}} / {{n_steps}}")
    print(f"MSE (all states, full run, vs the actual reference): {{mse:.10g}}")
    print(f"Final ||x - reference||: {{final_error:.6g}}")
    print(f"Q = {{np.diag(cfg.mpc.state_weights).tolist()}}")
    print(f"R = {{np.diag(cfg.mpc.input_weights).tolist()}}")
    print(f"Np={{cfg.mpc.prediction_horizon}}  Nc={{cfg.mpc.control_horizon}}  dt_mpc={{DT_MPC}}")
    dynamics = plant                     # the plots below show the simulated plant
    ref_full = ref_full[:len(states)]

''' + '''
    n_states, n_inputs = dynamics.n_states, dynamics.n_inputs
    state_names = dynamics.state_names
    input_names = dynamics.input_names
    fig, axs = plt.subplots(n_states + n_inputs, 1, figsize=(10, 2.2 * (n_states + n_inputs)), sharex=True)
    if n_states + n_inputs == 1:
        axs = [axs]
    ref_plot = ref_full[:len(states)]
    for i in range(n_states):
        axs[i].plot(times, states[:, i], label=state_names[i])
        axs[i].plot(times, ref_plot[:, i], color="r", linestyle="--", linewidth=1, alpha=0.6, label="reference")
        axs[i].set_ylabel(state_names[i])
        axs[i].legend(loc="best", fontsize=8)
        axs[i].grid(alpha=0.3)
    for j in range(n_inputs):
        ax = axs[n_states + j]
        ax.plot(times[:-1] if len(times) > len(inputs) else times, inputs[:, j], color="g", label=input_names[j])
        ax.set_ylabel(input_names[j])
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    axs[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{class_name} -- MPC closed-loop response (trajectory={{TRAJECTORY_MODE or 'reg'}})")
    plt.tight_layout()
    plt.savefig("mpc_result.png", dpi=150)
    print("Saved plot to mpc_result.png")
    plt.show()


if __name__ == "__main__":
    main()
'''


def _safe_num(v: Any) -> float:
    """Casts to a native Python float (never a numpy scalar -- NumPy 2.x's
    repr() wraps those as e.g. 'np.float64(1.5)', which happens to still be
    valid Python source since 1.5 is self-contained, but there's no reason
    to embed that visual noise into a script meant to be read by a human).
    Infinite values are replaced with a large-but-finite sentinel (+-1e9):
    Python's own repr(float('-inf')) is the bare text '-inf', which is NOT
    valid Python source on its own (there is no bare `inf` literal/name in
    the language) -- confirmed by reproducing exactly the NameError this
    was causing in exported scripts before this fix. +-1e9 is functionally
    equivalent to "unconstrained" for any realistic system's units.
    """
    f = float(v)
    if f == float("inf"):
        return 1e9
    if f == float("-inf"):
        return -1e9
    return f


def _safe_num_list(values: Any) -> List[float]:
    return [_safe_num(v) for v in values]


def generate_standalone_script(
    dynamics_source_code: str,
    class_name: str,
    best_params: Dict[str, Any],
    dt_mpc: float,
    simulation_time: float,
    system_name: str,
    feedforward: Optional[List[float]] = None,
    initial_state: Optional[List[float]] = None,
    physical_params_override: Optional[Dict[str, Any]] = None,
    trajectory_mode: Optional[str] = None,
    trajectory_amplitude: float = 0.5,
    trajectory_frequency: float = 0.5,
    trajectory_pulse_start: float = 0.2,
    trajectory_pulse_end: float = 0.7,
    trajectory_per_state_modes: Optional[List[str]] = None,
    u_bounds: Optional[Tuple[List[float], List[float]]] = None,
    x_bounds: Optional[Tuple[List[float], List[float]]] = None,
    nominal_params: Optional[Dict[str, Any]] = None,
    derivative_pairs: Optional[List[Tuple[int, int]]] = None,
    reference: Optional[List[List[float]]] = None,
    use_disturbance_observer: bool = False,
    observer_gain: float = 0.3,
    terminal_weight_mode: str = "dare",
    actuator_effectiveness: Optional[List[List[float]]] = None,
    input_disturbance: Optional[List[List[float]]] = None,
    state_disturbance: Optional[List[List[float]]] = None,
    noise_std: Optional[List[float]] = None,
    random_seed: int = 42,
) -> str:
    """Returns the full text of a standalone .py file. ``dynamics_source_code``
    should be the validated/final source of the user's uploaded plugin file
    (including its create_config()/class definition) -- it's embedded
    verbatim, same as the other pieces, for fidelity with what was actually
    tuned.

    ``initial_state``/``physical_params_override``/``trajectory_*``/
    ``u_bounds``/``x_bounds`` reproduce the ACTUAL scenario the run was
    tuned against (Scenario Level, trajectory type, constraints) rather
    than always testing on the dynamics file's own plain defaults --
    left as None/default reproduces the previous (nominal/regulation)
    behavior for callers that don't have this information.
    """
    base_src = _read_stripped("dynamics/base.py")
    config_src = _read_stripped("mpc/config.py")
    jacobian_src = _read_stripped("mpc/jacobian.py")
    solver_src = _read_stripped("mpc/solver.py")
    controller_src = _read_stripped("mpc/controller.py")

    def _table(rows):
        return None if rows is None else [_safe_num_list(np.ravel(r)) for r in rows]

    driver = _DRIVER_TEMPLATE.format(
        best_params={"Np": int(best_params.get("Np")), "Nc": int(best_params.get("Nc")),
                     "Q": _safe_num_list(best_params.get("Q") or []), "R": _safe_num_list(best_params.get("R") or []),
                     "P": _safe_num_list(best_params.get("P") or best_params.get("Q") or [])},
        dt_mpc=_safe_num(dt_mpc),
        simulation_time=_safe_num(simulation_time),
        class_name=class_name,
        feedforward=_safe_num_list(feedforward) if feedforward else None,
        initial_state=_safe_num_list(initial_state) if initial_state is not None else None,
        physical_params_override=(
            {k: (_safe_num(v) if isinstance(v, (int, float)) else v) for k, v in physical_params_override.items()}
            if physical_params_override else None
        ),
        trajectory_mode=trajectory_mode,
        trajectory_amplitude=_safe_num(trajectory_amplitude),
        trajectory_frequency=_safe_num(trajectory_frequency),
        trajectory_pulse_start=_safe_num(trajectory_pulse_start),
        trajectory_pulse_end=_safe_num(trajectory_pulse_end),
        trajectory_per_state_modes=list(trajectory_per_state_modes) if trajectory_per_state_modes else None,
        u_bounds=(_safe_num_list(u_bounds[0]), _safe_num_list(u_bounds[1])) if u_bounds is not None else None,
        x_bounds=(_safe_num_list(x_bounds[0]), _safe_num_list(x_bounds[1])) if x_bounds is not None else None,
        nominal_params=(
            {k: (_safe_num(v) if isinstance(v, (int, float)) else v) for k, v in nominal_params.items()}
            if nominal_params else None
        ),
        derivative_pairs=[[int(a), int(b)] for a, b in derivative_pairs] if derivative_pairs else None,
        reference=_table(reference),
        use_disturbance_observer=bool(use_disturbance_observer),
        terminal_weight_mode=str(terminal_weight_mode),
        observer_gain=_safe_num(observer_gain),
        actuator_effectiveness=_table(actuator_effectiveness),
        input_disturbance=_table(input_disturbance),
        state_disturbance=_table(state_disturbance),
        noise_std=_safe_num_list(np.ravel(noise_std)) if noise_std is not None else None,
        random_seed=int(random_seed),
    )

    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", system_name.lower())
    parts = [
        f'"""\nStandalone MPC reproduction script, exported from Agent-MPC Studio.\n'
        f'System: {system_name}\nOnly numpy, scipy, and matplotlib are required -- no other dependency '
        f'from the original tuning app.\n\nRun with:  python {safe_name}_mpc_export.py\n"""',
        "from __future__ import annotations\n\nimport numpy as np\nimport scipy.sparse as sp\n"
        "import matplotlib.pyplot as plt\nfrom abc import ABC, abstractmethod\n"
        "from dataclasses import dataclass, field\nfrom typing import Any, Callable, Dict, List, Optional, Tuple, Union",
        "# " + "=" * 78 + "\n# Part 1: dynamics/base.py (BaseDynamics, SystemConfig, SystemSimulator)\n# " + "=" * 78,
        base_src,
        "# " + "=" * 78 + "\n# Part 2: mpc/config.py (Config, MPCConfig, DataConfig)\n# " + "=" * 78,
        config_src,
        "# " + "=" * 78 + "\n# Part 3: mpc/jacobian.py + mpc/solver.py (torch and osqp stay optional)\n# " + "=" * 78,
        jacobian_src,
        solver_src,
        "# " + "=" * 78 + "\n# Part 4: mpc/controller.py (GenericMPC)\n# " + "=" * 78,
        controller_src,
        "# " + "=" * 78 + f"\n# Part 5: your dynamics plugin ({class_name})\n# " + "=" * 78,
        dynamics_source_code.strip(),
        driver,
    ]
    return "\n\n\n".join(parts) + "\n"


def export_scenario_from_cfg(cfg, dynamics, dt: Optional[float] = None,
                             max_reference_rows: int = 200000) -> Dict[str, Any]:
    """Everything about a run's scenario that generate_standalone_script needs,
    read off the Config the run used -- so both frontends export the same
    thing from the same source instead of each assembling it by hand.

    ``dynamics`` is the plant as simulated (its params may be perturbed);
    ``cfg.data.nominal_dynamics`` is what the controller believed. Signals
    that are callables of time are sampled once per control step, which is
    exactly how the evaluator holds them (zero-order hold), so a
    time-varying disturbance is reproduced too.
    """
    # dt: the exported run's sample time when it differs from cfg's (the Actor
    # may have tuned dt; time-varying signals must be sampled on the export's clock).
    dt, T = float(dt if dt else cfg.data.dt_mpc), float(cfg.data.simulation_time)
    n_steps = max(int(round(T / dt)), 1)

    def table(spec, width):
        if spec is None:
            return None
        if callable(spec):
            return [np.broadcast_to(np.asarray(spec(k * dt), dtype=float).ravel(), (width,)).tolist()
                    for k in range(n_steps)]
        return [np.broadcast_to(np.asarray(spec, dtype=float).ravel(), (width,)).tolist()]

    nominal = getattr(cfg.data, "nominal_dynamics", None)
    nominal_params = dict(nominal.params) if nominal is not None and nominal.params is not None else None
    plant_params = dict(dynamics.params) if dynamics.params else None
    if nominal_params == plant_params:
        nominal_params = None

    reference = None
    if cfg.data.trajectory_mode == "custom" and cfg.data.custom_trajectory_fn is not None:
        ref = np.asarray(cfg.data.custom_trajectory_fn(dt, T, dynamics.n_states, dynamics.state_names), dtype=float)
        if ref.size <= max_reference_rows * max(dynamics.n_states, 1):
            reference = ref[:n_steps].tolist()

    noise = np.asarray(cfg.data.noise_std, dtype=float) if cfg.data.noise_std is not None else None
    u_bounds = cfg.mpc.u_bounds if cfg.mpc.u_bounds is not None else dynamics.get_input_bounds()
    x_bounds = cfg.mpc.x_bounds if cfg.mpc.x_bounds is not None else dynamics.get_state_bounds()
    return {
        "initial_state": list(np.asarray(dynamics.config.default_initial_state, dtype=float)),
        "physical_params_override": plant_params,
        "nominal_params": nominal_params,
        "derivative_pairs": dynamics.config.derivative_pairs,
        "trajectory_mode": cfg.data.trajectory_mode if reference is None else "custom",
        "trajectory_amplitude": cfg.data.trajectory_amplitude,
        "trajectory_frequency": cfg.data.trajectory_frequency,
        "trajectory_pulse_start": cfg.data.trajectory_pulse_start,
        "trajectory_pulse_end": cfg.data.trajectory_pulse_end,
        "trajectory_per_state_modes": cfg.data.trajectory_per_state_modes,
        "reference": reference,
        "u_bounds": None if u_bounds is None else (list(u_bounds[0]), list(u_bounds[1])),
        "x_bounds": None if x_bounds is None else (list(x_bounds[0]), list(x_bounds[1])),
        "feedforward": (None if cfg.data.feedforward_override is None
                        else list(np.asarray(cfg.data.feedforward_override, dtype=float))),
        "use_disturbance_observer": bool(cfg.mpc.use_disturbance_observer),
        "terminal_weight_mode": str(getattr(cfg.mpc, "terminal_weight_mode", "dare")),
        "observer_gain": float(cfg.mpc.observer_gain),
        "actuator_effectiveness": table(cfg.data.actuator_effectiveness, dynamics.n_inputs),
        "input_disturbance": table(cfg.data.input_disturbance, dynamics.n_inputs),
        "state_disturbance": table(cfg.data.state_disturbance, dynamics.n_states),
        "noise_std": None if noise is None or not np.any(noise > 0) else noise.tolist(),
        "random_seed": int(cfg.random_seed),
    }
