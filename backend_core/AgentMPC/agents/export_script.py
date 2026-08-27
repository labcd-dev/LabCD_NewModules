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
and mpc/controller.py live from disk and embeds them close to verbatim
(only their cross-module relative imports are stripped, since everything
ends up in one flat file/namespace). The two pieces that genuinely need
simplification -- jacobian.py (drop the optional torch-autograd path) and
solver.py (drop the optional osqp path, keep only the scipy-based dense
fallback that's already used whenever osqp isn't installed) -- are written
fresh here, following the exact same algorithm as their real counterparts,
specifically so the exported script has no dependencies beyond
numpy/scipy/matplotlib.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


_JACOBIAN_BLOCK = '''
# ---- condensed from AgentMPC/mpc/jacobian.py (finite-difference only --
# the real module also supports torch autograd when available; dropped
# here so this script has no torch dependency) ----
def finite_diff_jacobian(f, x, u, h: float = 1e-4):
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


def linearize(dynamics_fn, x, u, torch_dynamics_fn=None, h: float = 1e-4, prefer_analytic: bool = True):
    return finite_diff_jacobian(dynamics_fn, x, u, h)
'''.strip("\n")


_SOLVER_BLOCK = '''
# ---- condensed from AgentMPC/mpc/solver.py (dense scipy fallback only --
# the real module prefers OSQP when installed for speed/warm-starting;
# dropped here so this script has no osqp dependency. Same algorithm as
# QPSolver._dense_box_fallback. ----
class QPSolver:
    def __init__(self):
        self.diagnostics = {"solved": 0, "other": 0}

    def solve(self, P, q, A, l, u):
        from scipy.optimize import minimize, LinearConstraint
        Pd = np.asarray(P.todense()) if sp.issparse(P) else np.asarray(P)
        Ad = np.asarray(A.todense()) if sp.issparse(A) else np.asarray(A)
        Pd = Pd + 1e-9 * np.eye(Pd.shape[0])
        x0 = np.zeros(Pd.shape[0])

        def obj(v):
            return 0.5 * v @ Pd @ v + q @ v

        def grad(v):
            return Pd @ v + q

        finite_rows = ~(np.isneginf(l) & np.isposinf(u))
        if not np.any(finite_rows):
            res = minimize(obj, x0, jac=grad, method="BFGS", options={"maxiter": 200})
            self.diagnostics["solved" if res.success else "other"] += 1
            return res.x

        constraint = LinearConstraint(Ad[finite_rows], l[finite_rows], u[finite_rows])
        res = minimize(obj, x0, jac=grad, constraints=[constraint], method="SLSQP",
                        options={"maxiter": 200, "ftol": 1e-9})
        self.diagnostics["solved" if res.success else "other"] += 1
        return res.x
'''.strip("\n")


_DRIVER_TEMPLATE = '''
# ============================================================================
# DRIVER -- runs the closed loop with the tuned parameters below, on the
# SAME scenario/trajectory/constraints this was actually tuned against, and
# plots states, inputs, and a short metrics summary. Edit anything below
# and re-run to explore further.
# ============================================================================

BEST_PARAMS = {best_params!r}
DT_MPC = {dt_mpc!r}
SIMULATION_TIME = {simulation_time!r}
FEEDFORWARD_OVERRIDE = {feedforward!r}   # None unless the "Use computed feedforward trim input" toggle was on

# The actual initial state and physical parameters used during tuning --
# NOT necessarily the dynamics file's own plain defaults, if a Scenario
# Level (Noise/Robust) was active: Level 3 in particular perturbs both the
# starting point and some physical parameters (a deliberate plant-model
# mismatch test -- see agents/scenario_presets.py in the original app).
INITIAL_STATE = {initial_state!r}
PHYSICAL_PARAMS_OVERRIDE = {physical_params_override!r}   # None if unchanged from the file's own defaults

# The reference trajectory actually used -- "reg" (regulation/constant),
# "sin", "pulse", or "custom" (per-state modes only, if set below).
TRAJECTORY_MODE = {trajectory_mode!r}
TRAJECTORY_AMPLITUDE = {trajectory_amplitude!r}
TRAJECTORY_FREQUENCY = {trajectory_frequency!r}
TRAJECTORY_PULSE_START = {trajectory_pulse_start!r}
TRAJECTORY_PULSE_END = {trajectory_pulse_end!r}
TRAJECTORY_PER_STATE_MODES = {trajectory_per_state_modes!r}   # None unless "Customize per state" was used

# Constraint bounds actually enforced during tuning (None = whatever the
# dynamics file itself declares, if anything).
U_BOUNDS = {u_bounds!r}
X_BOUNDS = {x_bounds!r}


def main():
    dynamics = {class_name}(create_config())
    if PHYSICAL_PARAMS_OVERRIDE is not None:
        dynamics.params.update(PHYSICAL_PARAMS_OVERRIDE)

    cfg = Config()
    cfg.mpc.prediction_horizon = BEST_PARAMS["Np"]
    cfg.mpc.control_horizon = BEST_PARAMS["Nc"]
    cfg.mpc.state_weights = np.diag(np.asarray(BEST_PARAMS["Q"], dtype=float))
    cfg.mpc.input_weights = np.diag(np.asarray(BEST_PARAMS["R"], dtype=float))
    cfg.mpc.terminal_weights = np.diag(np.asarray(BEST_PARAMS.get("P") or BEST_PARAMS["Q"], dtype=float))
    cfg.data.dt_mpc = DT_MPC
    cfg.data.simulation_time = SIMULATION_TIME
    if U_BOUNDS is not None:
        cfg.mpc.u_bounds = (np.asarray(U_BOUNDS[0], dtype=float), np.asarray(U_BOUNDS[1], dtype=float))
    if X_BOUNDS is not None:
        cfg.mpc.x_bounds = (np.asarray(X_BOUNDS[0], dtype=float), np.asarray(X_BOUNDS[1], dtype=float))

    controller = GenericMPC(dynamics, cfg)
    simulator = SystemSimulator(dynamics, dt=DT_MPC)

    x = (np.asarray(INITIAL_STATE, dtype=float) if INITIAL_STATE is not None
         else dynamics.config.default_initial_state.copy())
    u = (np.asarray(FEEDFORWARD_OVERRIDE, dtype=float) if FEEDFORWARD_OVERRIDE is not None
         else dynamics.get_equilibrium_input())
    target = dynamics.config.default_target.copy()

    n_steps = max(int(SIMULATION_TIME / DT_MPC), 1)
    # Reference needs to extend PREDICTION_HORIZON steps beyond the last
    # simulated step, so the controller always has a full lookahead window
    # even on the final iterations -- same reasoning as the original app.
    if TRAJECTORY_MODE == "reg" or TRAJECTORY_MODE is None:
        ref_full = np.tile(target, (n_steps + cfg.mpc.prediction_horizon, 1))
    else:
        ref_full = dynamics.config.desired_trajectory(
            DT_MPC, SIMULATION_TIME + cfg.mpc.prediction_horizon * DT_MPC,
            mode=TRAJECTORY_MODE, amplitude=TRAJECTORY_AMPLITUDE, frequency=TRAJECTORY_FREQUENCY,
            pulse_start=TRAJECTORY_PULSE_START, pulse_end=TRAJECTORY_PULSE_END,
            per_state_modes=TRAJECTORY_PER_STATE_MODES,
        )

    states = [x.copy()]
    inputs = []
    times = [0.0]

    for k in range(n_steps):
        ref_window = ref_full[k: k + cfg.mpc.prediction_horizon]
        u = controller.control(x, u, ref_window)
        x = simulator.step(x, u)
        states.append(x.copy())
        inputs.append(u.copy())
        times.append((k + 1) * DT_MPC)
        if dynamics.check_termination(x):
            print(f"Stopped early at step {{k}}: hit a declared state bound.")
            break

    states = np.array(states)
    inputs = np.array(inputs)
    times = np.array(times)

    errors = states[:-1] - ref_full[:len(states) - 1, :states.shape[1]]
    mse = float(np.mean(errors ** 2))
    final_error = np.linalg.norm(states[-1] - ref_full[len(states) - 1, :states.shape[1]])
    print(f"System: {class_name}")
    print(f"Scenario: trajectory={{TRAJECTORY_MODE or 'reg'}}, "
          f"initial_state={{'scenario-adjusted' if INITIAL_STATE is not None else 'file default'}}, "
          f"physical_params={{'perturbed' if PHYSICAL_PARAMS_OVERRIDE else 'file default'}}")
    print(f"Steps simulated: {{len(states) - 1}} / {{n_steps}}")
    print(f"MSE (all states, full run, vs the actual reference): {{mse:.6g}}")
    print(f"Final ||x - reference||: {{final_error:.6g}}")
    print(f"Q = {{np.diag(cfg.mpc.state_weights).tolist()}}")
    print(f"R = {{np.diag(cfg.mpc.input_weights).tolist()}}")
    print(f"Np={{cfg.mpc.prediction_horizon}}  Nc={{cfg.mpc.control_horizon}}  dt_mpc={{DT_MPC}}")

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
    controller_src = _read_stripped("mpc/controller.py")

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
        "# " + "=" * 78 + "\n# Part 3: condensed jacobian.py + solver.py (no torch/osqp dependency)\n# " + "=" * 78,
        _JACOBIAN_BLOCK,
        _SOLVER_BLOCK,
        "# " + "=" * 78 + "\n# Part 4: mpc/controller.py (GenericMPC)\n# " + "=" * 78,
        controller_src,
        "# " + "=" * 78 + f"\n# Part 5: your dynamics plugin ({class_name})\n# " + "=" * 78,
        dynamics_source_code.strip(),
        driver,
    ]
    return "\n\n\n".join(parts) + "\n"
