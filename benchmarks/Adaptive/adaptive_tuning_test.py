r"""Complete Groq-backed AgentAdaptive tuning example.

Run from the repository root:

    .venv\Scripts\python.exe benchmarks\Adaptive\artifacts\adaptive_tuning_test.py

The script follows the application flow:

    numerical Python plant + AgentPlant metadata
        -> PlantCompiler Adaptive system_spec
        -> Design Agent
        -> initial simulation
        -> Tuner Agent rounds
        -> final simulation and JSON report

No secret is stored in this file. Put ``GROQ_API_KEY=...`` in the repository
root ``.env`` file before running it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# 1. Repository and output paths
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
ADAPTIVE_DIR = THIS_FILE.parents[1]
BENCHMARKS_DIR = THIS_FILE.parents[2]
REPO_ROOT = THIS_FILE.parents[3]
RESULTS_DIR = ADAPTIVE_DIR / "results"
RESULT_FILE = RESULTS_DIR / "adaptive_double_integrator_groq_result.json"
ENV_FILE = REPO_ROOT / ".env"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# 2. Groq configuration for every Adaptive LLM role
# ---------------------------------------------------------------------------

LLM_PROVIDER = "groq"
DESIGN_MODEL = "openai/gpt-oss-120b"
TUNER_MODEL = "openai/gpt-oss-120b"
CLARIFIER_MODEL = "openai/gpt-oss-120b"
REPORTER_MODEL = "openai/gpt-oss-120b"

DESIGN_MAX_TOKENS = 1500
TUNER_MAX_TOKENS = 1500
CLARIFIER_MAX_TOKENS = 8000
REPORTER_MAX_TOKENS = 500


def configure_groq() -> None:
    """Load GROQ_API_KEY and map it to the Adaptive role configuration."""
    if not ENV_FILE.is_file():
        raise FileNotFoundError(f"Environment file not found: {ENV_FILE}")

    load_dotenv(dotenv_path=ENV_FILE, override=False)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError(
            f"GROQ_API_KEY is missing. Add it to {ENV_FILE} before running."
        )

    # AgentAdaptive currently uses OPENAI_* role-variable names regardless of
    # provider. LLM_PROVIDER tells LLMFactory that these models/keys belong to
    # Groq rather than OpenAI.
    os.environ["LLM_PROVIDER"] = LLM_PROVIDER
    os.environ["LLM_PROVIDER_TUNER"] = LLM_PROVIDER
    os.environ["LLM_PROVIDER_CLARIFIER"] = LLM_PROVIDER
    os.environ["LLM_PROVIDER_REPORTER"] = LLM_PROVIDER

    os.environ["OPENAI_MODEL"] = DESIGN_MODEL
    os.environ["OPENAI_MODEL_TUNER"] = TUNER_MODEL
    os.environ["OPENAI_MODEL_CLARIFIER"] = CLARIFIER_MODEL
    os.environ["OPENAI_MODEL_REPORTER"] = REPORTER_MODEL

    os.environ["OPENAI_API_KEY"] = groq_api_key
    os.environ["OPENAI_API_KEY_TUNER"] = groq_api_key
    os.environ["OPENAI_API_KEY_CLARIFIER"] = groq_api_key
    os.environ["OPENAI_API_KEY_REPORTER"] = groq_api_key

    os.environ["OPENAI_MAX_TOKENS"] = str(DESIGN_MAX_TOKENS)
    os.environ["OPENAI_MAX_TOKENS_TUNER"] = str(TUNER_MAX_TOKENS)
    os.environ["OPENAI_MAX_TOKENS_CLARIFIER"] = str(CLARIFIER_MAX_TOKENS)
    os.environ["OPENAI_MAX_TOKENS_REPORTER"] = str(REPORTER_MAX_TOKENS)

    # Prevent interactive plot windows during a script/benchmark run.
    os.environ["LABCD_ADAPTIVE_SHOW_PLOTS"] = "0"
    os.environ["MPLBACKEND"] = "Agg"


# ---------------------------------------------------------------------------
# 3. Numerical Python dynamic model
# ---------------------------------------------------------------------------

MASS_KG = 1.0


def double_integrator_dynamics(
    t: float,
    x: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Nominal double-integrator dynamics: x=[position, velocity], u=[force]."""
    del t  # This nominal model is time invariant.
    position = float(x[0])
    velocity = float(x[1])
    force = float(u[0])
    del position  # Position does not directly enter the nominal acceleration.
    acceleration = force / MASS_KG
    return np.array([velocity, acceleration], dtype=float)


# PlantCompiler expects the AgentPlant code in source-string form. It validates
# this source and uses the matching metadata to generate the Adaptive spec.
PLANT_PYTHON_CODE = '''import numpy as np

def dynamics(t, x, u):
    position = x[0]  # position (m)
    velocity = x[1]  # velocity (m/s)
    force = u[0]  # force (N)
    mass = 1.0  # mass (kg)
    acceleration = force / mass
    dx = np.array([velocity, acceleration], dtype=float)
    return dx
'''

PLANT_OUTPUT: dict[str, Any] = {
    "system_name": "AdaptiveDoubleIntegrator",
    "python_code": PLANT_PYTHON_CODE,
    "metadata": {
        "states": ["position", "velocity"],
        "state_meanings": ["position (m)", "velocity (m/s)"],
        "inputs": ["force"],
        "outputs": ["position"],
        "state_equations": [
            "velocity",
            "force / mass",
        ],
        "parameters": {
            "mass": MASS_KG,
        },
        "system_type": "SISO",
        "assumptions": [
            "The nominal mass is constant and equal to 1.0 kg.",
            "Position and velocity are available to the controller.",
            "The control input is an unconstrained force in this example.",
        ],
    },
}


# ---------------------------------------------------------------------------
# 4. Initial condition, reference, uncertainty, and disturbance
# ---------------------------------------------------------------------------

SIMULATION_TIME_S = 8.0
SOLVER_STEP_S = 0.001
INITIAL_STATE = [2.0, 0.0]
DEFAULT_FULL_STATE_TARGET = [0.0, 0.0]

# Adaptive tracks declared outputs, not a mandatory target for every state.
POSITION_REFERENCE_EXPRESSION = "0"

# The nominal Python model remains f(x,u). AgentAdaptive simulates the true
# plant as f(x,u) + Delta(x) + d(t), using these explicit expressions.
STATE_DEPENDENT_UNCERTAINTY = [
    {
        "state": "velocity",
        "expr": "-0.20 * velocity",
    }
]
TIME_DEPENDENT_DISTURBANCE = [
    {
        "state": "velocity",
        "expr": "0.15 * sin(1.5*t)",
    }
]

PRE_LAUNCH: dict[str, Any] = {
    "total_simulation_time": SIMULATION_TIME_S,
    "solver_sample_time": SOLVER_STEP_S,
    "initial_state": INITIAL_STATE,
    "default_target": DEFAULT_FULL_STATE_TARGET,
}


def build_adaptive_system_spec() -> dict[str, Any]:
    """Validate the Python plant and construct the complete Adaptive input."""
    from backend_core.plant_compiler import PlantCompiler

    compiler = PlantCompiler()
    validation = compiler.validate(PLANT_OUTPUT)
    validation.raise_if_invalid()
    spec = compiler.generate_adaptive_spec(
        plant_output=PLANT_OUTPUT,
        pre_launch=PRE_LAUNCH,
    )

    dynamics = spec["dynamics"]
    dynamics["x0"] = list(INITIAL_STATE)
    dynamics["references"] = [
        {
            "output": "position",
            "expr": POSITION_REFERENCE_EXPRESSION,
        }
    ]
    dynamics["uncertainty"] = [dict(item) for item in STATE_DEPENDENT_UNCERTAINTY]
    dynamics["disturbance"] = [dict(item) for item in TIME_DEPENDENT_DISTURBANCE]
    dynamics["sim_time"] = SIMULATION_TIME_S
    dynamics["solver_step"] = SOLVER_STEP_S
    spec["status"] = "complete"
    return spec


# ---------------------------------------------------------------------------
# 5. Every public tuning-run input
# ---------------------------------------------------------------------------

DESCRIPTION = (
    "Design and tune an adaptive controller for a force-driven double "
    "integrator. Track zero position from the declared initial state while "
    "handling the explicit state-dependent damping uncertainty and sinusoidal "
    "acceleration disturbance."
)

ENABLE_TUNING = True
TARGET_STEADY_RMS_FRACTION = 0.02
MAX_TUNING_ROUNDS = 4

# Importance values use the repository's 1..5 scale. The runner automatically
# removes objectives that do not apply to the method selected by the Designer.
TUNING_OBJECTIVES = {
    "steady_state_error": 5,
    "tracking_mse": 5,
    "overshoot": 3,
    "settling_time": 4,
    "transient_error": 3,
    "control_effort": 3,
    "control_max": 3,
    "control_smoothness": 3,
    "chattering": 2,
    "estimator_accuracy": 4,
}

SIMULATION_OVERRIDES = {
    "t_end": SIMULATION_TIME_S,
    "dt": SOLVER_STEP_S,
}

# A complete system_spec is supplied directly, so this standalone example does
# not run the interactive Clarifier conversation.
CLARIFICATION_RECORD: list[dict[str, Any]] = []
CLARIFIER_USAGE = None
SAVE_TIME_SERIES = True


# ---------------------------------------------------------------------------
# 6. Progress callback and JSON conversion
# ---------------------------------------------------------------------------

PROGRESS_EVENTS: list[dict[str, Any]] = []


def on_event(**event: Any) -> None:
    """Collect and print the important Design/Tuner progress events."""
    PROGRESS_EVENTS.append(dict(event))
    kind = event.get("kind", "event")
    stage = event.get("stage", "")
    round_number = event.get("round")
    suffix = f", round={round_number}" if round_number is not None else ""
    print(f"[{kind}] stage={stage}{suffix}")


def should_stop() -> bool:
    """Cancellation callback required by the runner; always continue here."""
    return False


def to_jsonable(value: Any) -> Any:
    """Convert arrays, message objects, and nested results to plain JSON data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    content = getattr(value, "content", None)
    if content is not None:
        return {"type": type(value).__name__, "content": to_jsonable(content)}
    return str(value)


# ---------------------------------------------------------------------------
# 7. Run Design -> simulation -> tuning -> final simulation
# ---------------------------------------------------------------------------

def main() -> None:
    configure_groq()

    # Import after provider configuration so every Adaptive role resolves Groq.
    from backend_core.AgentAdaptive.agents import run_full_pipeline

    system_spec = build_adaptive_system_spec()

    # Confirm that the numerical Python plant has the expected shape and finite
    # output before spending any API tokens.
    preflight_dx = double_integrator_dynamics(
        t=0.0,
        x=np.asarray(INITIAL_STATE, dtype=float),
        u=np.array([0.0], dtype=float),
    )
    if preflight_dx.shape != (2,) or not np.all(np.isfinite(preflight_dx)):
        raise RuntimeError(f"Python dynamics preflight failed: {preflight_dx}")

    print("Starting Groq-backed Adaptive design and tuning...")
    result, usage, tuning_log, tuning_best = run_full_pipeline(
        description=DESCRIPTION,
        enable_tuning=ENABLE_TUNING,
        target_rms_frac=TARGET_STEADY_RMS_FRACTION,
        max_tuning_rounds=MAX_TUNING_ROUNDS,
        on_event=on_event,
        should_stop=should_stop,
        clarification_record=CLARIFICATION_RECORD,
        sim_overrides=SIMULATION_OVERRIDES,
        clarifier_usage=CLARIFIER_USAGE,
        tuning_objectives=TUNING_OBJECTIVES,
        system_spec=system_spec,
    )

    stored_result = dict(result) if isinstance(result, dict) else {"value": result}
    if not SAVE_TIME_SERIES:
        stored_result.pop("series", None)

    report = {
        "run_configuration": {
            "provider": LLM_PROVIDER,
            "models": {
                "design": DESIGN_MODEL,
                "tuner": TUNER_MODEL,
                "clarifier": CLARIFIER_MODEL,
                "reporter": REPORTER_MODEL,
            },
            "max_tokens": {
                "design": DESIGN_MAX_TOKENS,
                "tuner": TUNER_MAX_TOKENS,
                "clarifier": CLARIFIER_MAX_TOKENS,
                "reporter": REPORTER_MAX_TOKENS,
            },
            "enable_tuning": ENABLE_TUNING,
            "target_steady_rms_fraction": TARGET_STEADY_RMS_FRACTION,
            "max_tuning_rounds": MAX_TUNING_ROUNDS,
            "tuning_objectives": TUNING_OBJECTIVES,
            "simulation_overrides": SIMULATION_OVERRIDES,
            "save_time_series": SAVE_TIME_SERIES,
        },
        "python_dynamics_preflight": {
            "initial_state": INITIAL_STATE,
            "zero_input": [0.0],
            "state_derivative": preflight_dx.tolist(),
        },
        "system_spec": system_spec,
        "result": stored_result,
        "final_metrics": stored_result.get("final_metrics"),
        "tuning_log": tuning_log,
        "tuning_best": tuning_best,
        "token_usage": usage,
        "progress_events": PROGRESS_EVENTS,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(
        json.dumps(to_jsonable(report), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    print(f"Tuning finished. Result saved to: {RESULT_FILE}")
    print("Best tuning:", to_jsonable(tuning_best))
    print("Final metrics:", to_jsonable(stored_result.get("final_metrics")))
    print("Token usage:", to_jsonable(usage.get("total")))


if __name__ == "__main__":
    main()
