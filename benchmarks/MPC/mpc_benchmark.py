"""Run MPC Actor/Critic/Terminator/Juror on selected artifact tests.

Edit USER SETTINGS, then run this file. Mock mode uses the agents under
benchmarks/MPC/mock. Real mode uses backend_core/AgentMPC.
Each model/seed gets <artifact>_results.json under
json_reports/results/<model>/seed<value>/, alongside trajectories/.
Results follow json_reports/accepted_results_template.json: compact tuning
summary plus complete verification and short output summaries. Finalization
updates complexity and success scores in the same file; workflow snapshots
are not saved.
Importing this module does not start a benchmark or create files.
"""

from __future__ import annotations

import copy
import json
import math
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np

MPC_DIR = Path(__file__).resolve().parent
REPO_ROOT = MPC_DIR.parents[1]
ARTIFACTS_DIR = MPC_DIR / "artifacts"
RESULTS_DIR = MPC_DIR / "json_reports" / "results"
TRAJECTORY_FIELDS = ("states", "inputs", "reference", "times")
PROGRESS_FILENAME = "benchmark_progress.json"
for directory in (REPO_ROOT, REPO_ROOT / "packages" / "labcd_agents" / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from backend_core.AgentMPC.agents.evaluator import run_closed_loop
from backend_core.AgentMPC.agents.llm_base import TokenUsageTracker, configure_llm
from backend_core.AgentMPC.agents.scenario_presets import apply_scenario_level
from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig
from backend_core.AgentMPC.graph.workflow import build_ui_tuning_graph, initial_state
from backend_core.AgentMPC.mpc.config import Config, DataConfig, MPCConfig

# ------------------------- USER SETTINGS -------------------------
RUN_BENCHMARK = True
USE_MOCK_AGENTS = False  # True avoids API calls and uses reproducible random seeds.
# None selects all artifacts; a list selects only those plants.
ARTIFACT_FILES: list[str] | None = None
# Example: ARTIFACT_FILES = ["DCMotor.json", "InvPendulum.json"]
# True selects only the cheapest entry according to the repository price table.
# False benchmarks every entry in LLM_CONFIGS. Mock mode ignores both settings.
CHEAPEST_MODEL_ONLY = False
LLM_CONFIGS = [
    {"folder_name": "gpt-5.4-mini", "provider": "openai",
     "model": "gpt-5.4-mini", "api_key_env_var": "OPENAI_API_KEY"},
    {"folder_name": "gpt-5.4-nano", "provider": "openai",
     "model": "gpt-5.4-nano", "api_key_env_var": "OPENAI_API_KEY"},
    {"folder_name": "gpt-4o-mini", "provider": "openai",
     "model": "gpt-4o-mini", "api_key_env_var": "OPENAI_API_KEY"},
    {"folder_name": "gpt-4.1-nano", "provider": "openai",
     "model": "gpt-4.1-nano", "api_key_env_var": "OPENAI_API_KEY"},
    {"folder_name": "groq-gpt-oss-120b", "provider": "groq",
     "model": "openai/gpt-oss-120b", "api_key_env_var": "GROQ_API_KEY"},
]
# To force one model, leave only that entry in LLM_CONFIGS.

LLM_TEMPERATURE = 0.3
# Each seed repeats every selected artifact with a fresh tuning workflow.
# Used for LLM sampling, simulation randomness, and mock agents.
RUN_SEEDS: list[int] = [42, 123, 2024]
LLM_MAX_RETRIES = 10
LLM_TIMEOUT_SECONDS = 120.0
LLM_MAX_TOKENS = 4000

MAX_ITERATIONS = 20
MIN_EXPLORE_ITERATIONS = 4
EXPLORATION_INTENSITY = 50
RECURSION_LIMIT = 500
# Free-text direction for the agents; numeric cost weights come from artifacts.
# USER_GUIDANCE = (
#     "Prioritize low overshoot and accurate reference tracking, followed by "
#     "short settling time, while keeping control effort reasonable."
# )
USER_GUIDANCE = ""
# None lets the Actor propose the first MPC parameters.
SEED_PARAMS = None

# Simulation time, time steps, and settling tolerance come from the artifact.
MOCK_COST_USD_PER_TOKEN = 0.001
# ---------------------------------------------------------------


def _safe_folder(name: str) -> bool:
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    return (isinstance(name, str) and bool(name) and name not in (".", "..")
            and not any(char in '<>:"/\\|?*' for char in name)
            and not name.endswith((".", " ")) and name.upper().split(".")[0] not in reserved)


def selected_models() -> list[dict[str, Any]]:
    if USE_MOCK_AGENTS:
        return [{"folder_name": "mock", "provider": None, "model": "mock-random"}]
    if not LLM_CONFIGS:
        raise ValueError("LLM_CONFIGS must contain at least one paid-model configuration.")
    names = [config.get("folder_name") for config in LLM_CONFIGS]
    if not all(_safe_folder(name) for name in names) or len(names) != len(set(names)):
        raise ValueError("LLM_CONFIGS must have unique, safe folder_name values.")
    for config in LLM_CONFIGS:
        if any(not isinstance(config.get(key), str) or not config[key].strip()
               for key in ("provider", "model", "api_key_env_var")):
            raise ValueError("Each LLM needs provider, model, and api_key_env_var.")
    if not CHEAPEST_MODEL_ONLY:
        return list(LLM_CONFIGS)
    from labcd_agents.pricing import CostCalculator
    calculator = CostCalculator()
    ranked = []
    for index, config in enumerate(LLM_CONFIGS):
        price = calculator.resolve_price(config["model"])
        if price is None:
            raise ValueError(f"No configured price for {config['model']}; add it before comparing costs.")
        # Equal input/output counts are a selection estimate, not a prediction
        # of actual run cost. Recorded cost uses actual token counts.
        ranked.append((price.input_per_million + price.output_per_million, index))
    _cost, index = min(ranked)
    return [LLM_CONFIGS[index]]


def validate_complexity(value: Any) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10
    ):
        raise ValueError("total_complexity must be null or an integer from 1 to 10.")


def artifact_bounds(
    artifact: dict[str, Any], key: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read optional [lower vector, upper vector] limits in metadata order."""
    names_key = {"state_bounds": "states", "input_bounds": "inputs"}[key]
    metadata = artifact["metadata"]
    bounds = metadata.get(key)
    if bounds is None:
        return None  # Missing or null means no declared limits, not zero limits.
    count = len(metadata[names_key])
    label = f"{artifact['system_name']} metadata.{key}"
    if (not isinstance(bounds, list) or len(bounds) != 2
            or any(not isinstance(vector, list) or len(vector) != count for vector in bounds)):
        raise ValueError(f"{label} must be [lower, upper], each with {count} values.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           for vector in bounds for value in vector):
        raise ValueError(f"{label} must contain numeric limits, not booleans or strings.")
    lower, upper = np.asarray(bounds, dtype=float)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError(f"{label} limits must be finite; use null for undeclared bounds.")
    if np.any(lower > upper):
        raise ValueError(f"{label} lower limits must not exceed upper limits.")
    return lower.copy(), upper.copy()


def artifact_trim_input(artifact: dict[str, Any]) -> np.ndarray:
    count = len(artifact["metadata"]["inputs"])
    trim = np.asarray(artifact["metadata"].get("trim_values", np.zeros(count)), dtype=float).reshape(-1)
    if trim.shape != (count,) or not np.all(np.isfinite(trim)):
        raise ValueError(f"metadata.trim_values must be a finite vector of length {count}.")
    return trim


def validate_operating_bounds(
    artifact: dict[str, Any], initial_condition: np.ndarray, target: np.ndarray,
) -> None:
    """Reject incompatible declared conditions before any tuning/API calls."""
    state_bounds = artifact_bounds(artifact, "state_bounds")
    input_bounds = artifact_bounds(artifact, "input_bounds")
    if state_bounds is not None:
        lower, upper = state_bounds
        for name, vector in (("initial_condition", initial_condition), ("target", target)):
            if np.any(vector < lower) or np.any(vector > upper):
                raise ValueError(f"{artifact['system_name']} {name} lies outside metadata.state_bounds.")
    trim = artifact_trim_input(artifact)
    if input_bounds is not None:
        lower, upper = input_bounds
        if np.any(trim < lower) or np.any(trim > upper):
            raise ValueError(f"{artifact['system_name']} trim input lies outside metadata.input_bounds.")


def load_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or not artifact.get("system_name"):
        raise ValueError(f"{path.name} must define system_name.")
    if not isinstance(artifact.get("python_code"), str) or not artifact["python_code"].strip():
        raise ValueError(f"{path.name} must define python_code.")
    tests = artifact.get("tests")
    combinations = tests.get("combinations") if isinstance(tests, dict) else None
    if not isinstance(combinations, list) or not combinations:
        raise ValueError(f"{path.name} tests.combinations must be a non-empty list.")
    test_ids = [test.get("test_id") for test in combinations if isinstance(test, dict)]
    if (len(test_ids) != len(combinations) or any(not isinstance(value, str) or not value.strip()
                                                  for value in test_ids)
            or len(test_ids) != len(set(test_ids))):
        raise ValueError(f"{path.name} must give every test a unique, non-empty test_id.")
    if not _safe_folder(path.stem) or not all(_safe_folder(test_id) for test_id in test_ids):
        raise ValueError(f"{path.name}: artifact and test IDs must be safe trajectory-filename components.")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path.name} must define metadata.")
    for key in ("states", "inputs"):
        names = metadata.get(key)
        if (
            not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError(f"{path.name} metadata.{key} must contain unique, non-empty names.")
    n_states = len(metadata["states"])
    outputs = metadata.get("outputs")
    if (not isinstance(outputs, list) or not outputs
            or any(not isinstance(name, str) or name not in metadata["states"] for name in outputs)
            or len(outputs) != len(set(outputs))):
        raise ValueError(f"{path.name} metadata.outputs must name unique declared states.")
    artifact_runtime_settings(artifact)
    artifact_cost_weights(artifact)
    for test in combinations:
        label = f"{path.name} test {test['test_id']!r}"
        validate_complexity(test.get("total_complexity"))
        for key in ("initial_condition", "target"):
            vector = np.asarray(test[key], dtype=float)
            if vector.shape != (n_states,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{label} {key} must be a finite vector of length {n_states}.")
        validate_operating_bounds(artifact, np.asarray(test["initial_condition"], dtype=float),
                                  np.asarray(test["target"], dtype=float))
        if isinstance(test["scenario_level"], bool) or test["scenario_level"] not in (1, 2, 3):
            raise ValueError(f"{label} scenario_level must be 1, 2, or 3.")
        settings = test["scenario_settings"]
        mask = settings["noise_state_mask"]
        if len(mask) != n_states or any(not isinstance(value, bool) for value in mask):
            raise ValueError(f"{label} noise_state_mask must have one boolean per state.")
        for key in ("noise_std_value", "robust_noise_fraction", "max_param_uncertainty"):
            value = float(settings[key])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{label} {key} must be finite and between 0 and 1.")
        if not isinstance(settings["perturb_physical_params"], bool):
            raise ValueError(f"{label} perturb_physical_params must be a boolean.")
    return artifact


def selected_artifacts() -> list[tuple[Path, dict[str, Any]]]:
    paths = {path.name: path for path in ARTIFACTS_DIR.glob("*.json")}
    if not paths:
        raise FileNotFoundError(f"No artifacts found in {ARTIFACTS_DIR}.")
    names = sorted(paths) if ARTIFACT_FILES is None else ARTIFACT_FILES
    if not names or len(names) != len(set(names)):
        raise ValueError("ARTIFACT_FILES must contain unique artifact filenames.")
    unknown = set(names) - paths.keys()
    if unknown:
        raise ValueError(f"Unknown artifacts: {sorted(unknown)}.")
    return [(paths[name], load_artifact(paths[name])) for name in names]


def artifact_runtime_settings(artifact: dict[str, Any]) -> dict[str, float]:
    """Read simulation settings from this artifact, not benchmark-wide defaults."""
    values = artifact.get("simulation_parameters")
    if not isinstance(values, dict):
        raise ValueError("Every artifact must define simulation_parameters.")
    settings = {}
    for key in ("simulation_time", "dt_simulation", "dt_mpc", "settling_tolerance"):
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"simulation_parameters.{key} must be a positive number.")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"simulation_parameters.{key} must be finite and positive.")
        settings[key] = float(value)
    if max(settings["dt_simulation"], settings["dt_mpc"]) >= settings["simulation_time"]:
        raise ValueError("Simulation time must exceed both time steps.")
    return settings


def artifact_cost_weights(artifact: dict[str, Any]) -> dict[str, float] | None:
    """Read one plant's composite-cost coefficients, not MPC's Q/R/P."""
    values = artifact.get("simulation_parameters")
    if not isinstance(values, dict) or "cost_weights" not in values:
        raise ValueError("Every artifact must define simulation_parameters.cost_weights (object or null).")
    weights = values["cost_weights"]
    if weights is None:
        return None  # Use the core's balanced defaults explicitly.
    if not isinstance(weights, dict):
        raise ValueError("simulation_parameters.cost_weights must be a dictionary or null.")
    supported = {"mse", "overshoot", "settling_time", "control_effort",
                 "oscillation", "unsettled_penalty"}
    if set(weights) - supported:
        raise ValueError(f"Unsupported cost weights: {sorted(set(weights) - supported)}.")
    for name, value in weights.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"Cost weight {name!r} must be finite and nonnegative.")
    return {name: float(value) for name, value in weights.items()}


def validate_settings() -> None:
    """Validate shared run controls before configuring an LLM or creating reports."""
    if (not isinstance(RUN_SEEDS, list) or not RUN_SEEDS
            or any(isinstance(seed, bool) or not isinstance(seed, int)
                   or not 0 <= seed < 2**32 for seed in RUN_SEEDS)
            or len(RUN_SEEDS) != len(set(RUN_SEEDS))):
        raise ValueError("RUN_SEEDS must be a nonempty list of unique integers in [0, 2**32).")
    for name, value in (("MAX_ITERATIONS", MAX_ITERATIONS), ("RECURSION_LIMIT", RECURSION_LIMIT)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if (isinstance(MIN_EXPLORE_ITERATIONS, bool) or not isinstance(MIN_EXPLORE_ITERATIONS, int)
            or MIN_EXPLORE_ITERATIONS < 0):
        raise ValueError("MIN_EXPLORE_ITERATIONS must be a nonnegative integer.")


def compile_artifact_dynamics(
    artifact: dict[str, Any], initial_condition: np.ndarray, target: np.ndarray,
) -> BaseDynamics:
    """Adapt a Plant-Agent-style ``dynamics(t, x, u)`` artifact to MPC.

    This adapter is benchmark-local. It deliberately leaves AgentMPC core
    untouched while allowing MPC and Adaptive to consume the same plant
    equation and metadata representation.
    """
    metadata = artifact["metadata"]
    n_states = len(metadata["states"])
    n_inputs = len(metadata["inputs"])
    validate_operating_bounds(artifact, initial_condition, target)
    namespace: dict[str, Any] = {}
    exec(compile(artifact["python_code"], f"<{artifact['system_name']}>", "exec"), namespace)
    rhs = namespace.get("dynamics")
    if not callable(rhs):
        raise ValueError("python_code must define dynamics(t, x, u).")

    equilibrium_input = artifact_trim_input(artifact)

    config = SystemConfig(
        n_states=n_states,
        n_inputs=n_inputs,
        params=copy.deepcopy(metadata.get("parameters", {})),
        state_names=list(metadata["states"]),
        input_names=list(metadata["inputs"]),
        default_initial_state=initial_condition.copy(),
        default_target=target.copy(),
        state_bounds=artifact_bounds(artifact, "state_bounds"),
        input_bounds=artifact_bounds(artifact, "input_bounds"),
    )

    def wrapped_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        derivative = np.asarray(rhs(0.0, np.asarray(x, dtype=float), np.asarray(u, dtype=float)), dtype=float)
        if derivative.shape != (n_states,):
            raise ValueError(f"dynamics returned shape {derivative.shape}; expected ({n_states},).")
        return derivative

    def get_equilibrium_input(self) -> np.ndarray:
        return equilibrium_input.copy()

    dynamics_class = type(
        artifact["system_name"],
        (BaseDynamics,),
        {"__module__": __name__, "dynamics": wrapped_dynamics,
         "get_equilibrium_input": get_equilibrium_input},
    )
    dynamics = dynamics_class(config)
    derivative = dynamics.dynamics(initial_condition, equilibrium_input)
    if not np.all(np.isfinite(derivative)):
        raise ValueError("dynamics is non-finite at the declared initial condition and trim input.")
    return dynamics


def build_mpc_config(
    artifact: dict[str, Any], settings: dict[str, float], seed: int,
) -> Config:
    """Build a fresh MPC configuration for one test, using its artifact's timings."""
    return Config(
        data=DataConfig(
            simulation_time=settings["simulation_time"],
            dt_simulation=settings["dt_simulation"],
            dt_mpc=settings["dt_mpc"],
            settling_tolerance=settings["settling_tolerance"],
            trajectory_mode="reg",
        ),
        mpc=MPCConfig(),
        system_name=artifact["system_name"],
        random_seed=seed,
    )


def json_safe(value: Any):
    if isinstance(value, TokenUsageTracker):
        # Runtime callback objects/locks are not JSON; preserve their data.
        return json_safe(value.snapshot())
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return json_safe(value.as_dict())
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return json_safe(value.model_dump())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def configure_model(config: dict[str, Any], seed: int) -> None:
    if USE_MOCK_AGENTS:
        return
    from dotenv import load_dotenv
    from labcd_agents import LLMFactory, require_api_key
    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=True)
    api_key = require_api_key(provider=config["provider"], env_var=config["api_key_env_var"])

    def factory():
        return LLMFactory.create(
            model=config["model"], provider=config["provider"], api_key=api_key,
            temperature=LLM_TEMPERATURE, seed=seed, max_retries=LLM_MAX_RETRIES,
            timeout=LLM_TIMEOUT_SECONDS, max_tokens=LLM_MAX_TOKENS,
        )
    configure_llm(factory=factory)


@contextmanager
def benchmark_agent_routing(use_mock_agents: bool):
    """Temporarily route graph construction to benchmark-local mock agents.

    The upstream core remains untouched. The graph captures the selected node
    callables while it is built, after which the original core attributes are
    restored immediately.
    """
    if not use_mock_agents:
        yield
        return

    import backend_core.AgentMPC.agents.actor as core_actor
    import backend_core.AgentMPC.agents.critic as core_critic
    import backend_core.AgentMPC.agents.juror as core_juror
    import backend_core.AgentMPC.agents.terminator as core_terminator
    from benchmarks.MPC.mock.actor import actor_node
    from benchmarks.MPC.mock.critic import critic_node
    from benchmarks.MPC.mock.juror import juror_node
    from benchmarks.MPC.mock.terminator import should_continue

    replacements = (
        (core_actor, "actor_node", actor_node),
        (core_critic, "critic_node", critic_node),
        (core_juror, "juror_node", juror_node),
        (core_terminator, "should_continue", should_continue),
    )
    originals = [(module, name, getattr(module, name)) for module, name, _ in replacements]
    try:
        for module, name, replacement in replacements:
            setattr(module, name, replacement)
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


def simulated_mock_token_usage(final_state, agent_messages):
    """Create reproducible, nonzero fake usage for JSON-format testing."""
    call_count = sum(len(messages) for messages in agent_messages.values())
    rng = np.random.default_rng(
        seed=int(final_state.get("mock_seed", RUN_SEEDS[0])) + 701
    )

    prompt_tokens = int(
        sum(rng.integers(250, 901) for _ in range(call_count))
    )
    completion_tokens = int(
        sum(rng.integers(40, 221) for _ in range(call_count))
    )
    total_tokens = prompt_tokens + completion_tokens

    return {
        "simulated": True,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "call_count": call_count,
        "unparsed_calls": 0,
        "per_model": {
            "mock-random": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens,
            }
        },
        "cost_usd": round(total_tokens * MOCK_COST_USD_PER_TOKEN, 6),
        "unpriced_models": [],
    }


def clean_tuning_result(final_state, token_tracker, tuning_cfg):
    """Return the compact tuning report written directly by the benchmark."""
    parameter_history = final_state.get("params_history") or []
    best_params = copy.deepcopy(final_state.get("best_params"))
    if best_params is not None and best_params.get("dt") is None:
        # A null Actor dt means "keep the previous dt". Replay the parameter
        # history to recover the dt that was active when this candidate was
        # evaluated, rather than accidentally using a later candidate's dt.
        effective_dt = float(final_state.get("dt_mpc", tuning_cfg.data.dt_mpc))
        best_effective_dt = effective_dt
        for historical_params in parameter_history:
            if historical_params.get("dt") is not None:
                effective_dt = float(historical_params["dt"])
            if historical_params == final_state.get("best_params"):
                best_effective_dt = effective_dt
                break
        best_params["dt"] = best_effective_dt

    full_history = final_state.get("history") or []

    def messages_for(agent_name):
        prefixes = (f"[{agent_name}]", f"[Mock {agent_name}]")
        return [message for message in full_history if message.startswith(prefixes)]

    agent_messages = {
        "actor": messages_for("Actor"),
        "critic": messages_for("Critic"),
        "terminator": messages_for("Terminator"),
        "juror": messages_for("Juror"),
    }
    token_usage = (
        simulated_mock_token_usage(final_state, agent_messages)
        if USE_MOCK_AGENTS
        else {"simulated": False, **token_tracker.snapshot()}
    )

    return json_safe(
        {
            "best_params": best_params,
            "best_cost": final_state.get("best_cost"),
            "best_mse": final_state.get("best_mse"),
            "iterations": final_state.get("iteration"),
            "termination_reason": final_state.get("termination_reason"),
            "juror_verdict": final_state.get("juror_verdict"),
            "token_usage": token_usage,
        }
    )


def capture_tuning_run(graph, state, token_tracker, tuning_cfg, record):
    """Keep workflow memory internally, but save only the accepted summary.

    The last merged state is needed to select/verify the best controller and
    resolve its sampling interval. It is not serialized into the report.
    On failure, retain the summary of the last available state.
    """
    final_state = state
    completed = False
    try:
        for payload in graph.stream(
            input=state, config={"recursion_limit": RECURSION_LIMIT},
            stream_mode="values",
        ):
            final_state = payload
        completed = True
    finally:
        record["tuning"] = clean_tuning_result(final_state, token_tracker, tuning_cfg)
        record["tuning"]["completed"] = completed
    return final_state


def clean_verification(verification, state_names: list[str], output_names: list[str]):
    """Return only the accepted template's short per-output summaries.

    run_one saves verification metrics as raw_result and trajectories as NPZ. Do not
    duplicate its metrics, status, solver diagnostics, or solve-time fields.
    These arrays contain one value per output, not per simulation timestep.
    """
    if verification is None:
        return None
    if "error" in verification:
        return json_safe(
            {
                "outputs": list(output_names),
                "tracking_mse": None,
                "task_scale": None,
                "task_scale_basis": "mse",
            }
        )

    metrics = verification["metrics"].as_dict()
    indices = [state_names.index(name) for name in output_names]
    states = np.asarray(verification["states"], dtype=float)
    reference = np.asarray(verification["reference"], dtype=float)
    if (states.ndim != 2 or states.shape != reference.shape or not len(states)
            or states.shape[1] != len(state_names)):
        raise ValueError("Verification states/reference must have matching nonempty full-state shapes.")
    # Mean-squared error of holding each output at y(0) against the actual
    # reference. For regulation this is (y(0) - target)**2, without a square root.
    with np.errstate(over="ignore", invalid="ignore"):
        task_scale = np.mean(
            (states[0, indices] - reference[:, indices]) ** 2, axis=0,
        )
    output_mse = [metrics["per_state_mse"][index] for index in indices]
    return json_safe(
        {
            "outputs": list(output_names),
            "tracking_mse": {"full": output_mse},
            "task_scale": task_scale,
            "task_scale_basis": "mse",
        }
    )


def result_directory(config: dict[str, Any]) -> Path:
    """Use one model/seed directory for both reports and relative NPZ links."""
    folder = config["folder_name"]
    if not _safe_folder(folder):
        raise ValueError("Model folder must be a safe name.")
    directory = RESULTS_DIR / folder
    seed = config.get("run_seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("run_seed must be an integer in [0, 2**32).")
        directory /= f"seed{seed}"
    return directory


def save_verification_trajectories(
    artifact_path: Path, test_id: str, config: dict[str, Any], verification: dict[str, Any],
) -> str | None:
    """Save full numeric arrays before JSON conversion; return a relative link.

    No downsampling or non-finite-value replacement occurs in the NPZ. A
    verification error without trajectories has no archive and returns None.
    """
    if not any(key in verification for key in TRAJECTORY_FIELDS):
        return None
    if not all(key in verification for key in TRAJECTORY_FIELDS):
        raise ValueError("Verification returned incomplete trajectory arrays.")
    filename = f"{artifact_path.stem}_{test_id}_verification.npz"
    folder = config["folder_name"]
    if not _safe_folder(test_id) or not _safe_folder(filename) or not _safe_folder(folder):
        raise ValueError("Trajectory filename, test ID and model folder must be safe names.")
    arrays = {key: np.asarray(verification[key]) for key in TRAJECTORY_FIELDS}
    if any(array.dtype.kind not in "biuf" for array in arrays.values()):
        raise ValueError("Verification trajectories must contain numeric arrays, not Python objects.")
    relative_path = Path("trajectories") / filename
    destination = result_directory(config) / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as archive:
            temporary = Path(archive.name)
            np.savez_compressed(archive, **arrays)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return relative_path.as_posix()


def run_one(
    artifact_path: Path, artifact: dict[str, Any], test: dict[str, Any],
    config: dict[str, Any], seed: int,
) -> dict[str, Any]:
    settings = artifact_runtime_settings(artifact)
    record = {
        "artifact": artifact_path.name,
        "test_id": test["test_id"],
        "model": {
            "name": artifact["system_name"],
            "number_of_states": len(artifact["metadata"]["states"]),
            "number_of_inputs": len(artifact["metadata"]["inputs"]),
        },
        "initial_condition": list(test["initial_condition"]),
        "target": list(test["target"]),
        "scenario_level": int(test["scenario_level"]),
        "scenario_name": test["scenario_name"],
        "agent_mode": "mock" if USE_MOCK_AGENTS else config["model"],
        "seeds": {
            "run": config.get("run_seed", seed),
            "llm": None if USE_MOCK_AGENTS else config.get("run_seed", seed),
            "simulation": seed,
            "mock": seed if USE_MOCK_AGENTS else None,
        },
        "total_complexity": None,
        "success_rate_1": None,
        "success_rate_2": None,
        "tuning": None,
        "verification": None,
    }
    try:
        declared_initial = np.asarray(test["initial_condition"], dtype=float)
        declared_target = np.asarray(test["target"], dtype=float)
        dynamics = compile_artifact_dynamics(artifact, declared_initial, declared_target)
        if (dynamics.__class__.__name__ != artifact["system_name"]
                or dynamics.n_states != record["model"]["number_of_states"]
                or dynamics.n_inputs != record["model"]["number_of_inputs"]):
            raise ValueError(f"{artifact_path.name}: dynamics dimensions/name disagree with metadata.")
        dynamics.config.default_initial_state = declared_initial.copy()
        dynamics.config.default_target = declared_target.copy()
        cfg = build_mpc_config(artifact, settings, seed)
        scenario = test["scenario_settings"]
        applied_initial, applied_target, perturbed = apply_scenario_level(
            dynamics=dynamics, cfg=cfg, level=int(test["scenario_level"]),
            noise_std_value=float(scenario["noise_std_value"]),
            noise_state_mask=np.asarray(scenario["noise_state_mask"], dtype=bool),
            robust_noise_fraction=float(scenario["robust_noise_fraction"]),
            perturb_physical_params=scenario["perturb_physical_params"],
            max_param_uncertainty=float(scenario["max_param_uncertainty"]),
        )
        if not np.allclose(applied_initial, declared_initial):
            record["simulated_initial_condition"] = applied_initial.tolist()
        if not np.allclose(applied_target, declared_target):
            record["simulated_target"] = applied_target.tolist()
        if perturbed:
            record["perturbed_physical_parameters"] = json_safe(perturbed)
        tracker = TokenUsageTracker(default_model=config["model"])
        with benchmark_agent_routing(USE_MOCK_AGENTS):
            graph = build_ui_tuning_graph(
                dynamics=dynamics, cfg=cfg,
                entry_node="evaluator" if SEED_PARAMS else "actor",
                max_iterations=MAX_ITERATIONS,
            )
        state = initial_state(
            dynamics=dynamics, system_name=artifact["system_name"],
            max_iterations=MAX_ITERATIONS, ui_scenario_level=int(test["scenario_level"]),
            seed_params=copy.deepcopy(SEED_PARAMS), user_guidance=USER_GUIDANCE,
            min_explore_iterations=MIN_EXPLORE_ITERATIONS,
            cost_weights=artifact_cost_weights(artifact), exploration_intensity=EXPLORATION_INTENSITY,
            dt_mpc=settings["dt_mpc"], token_tracker=tracker,
        )
        if USE_MOCK_AGENTS:
            state["mock_seed"] = seed
        capture_tuning_run(graph, state, tracker, cfg, record)
        best_params = record["tuning"]["best_params"]
        if best_params is None:
            raise RuntimeError("The tuning graph produced no best_params.")
        cfg.data.dt_mpc = float(best_params["dt"])
        verification = run_closed_loop(
            dynamics=dynamics, cfg=cfg,
            params={key: best_params[key] for key in ("Np", "Nc", "Q", "R", "P")},
            max_steps=None, rng=np.random.default_rng(seed=seed),
        )
        # Keep metrics/errors in JSON without converting the large arrays to lists.
        record["verification"] = {"raw_result": json_safe({
            key: value for key, value in verification.items() if key not in TRAJECTORY_FIELDS
        })}
        record["verification"]["raw_result"]["trajectory_file"] = None
        record["verification"]["raw_result"]["trajectory_file"] = save_verification_trajectories(
            artifact_path, test["test_id"], config, verification,
        )
        verification_summary = clean_verification(
            verification, artifact["metadata"]["states"], artifact["metadata"]["outputs"],
        )
        record["verification"].update(verification_summary)
        if "error" in verification:
            raise RuntimeError(f"Verification failed: {verification['error']}")
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    return json_safe(record)


def write_json_checkpoint(output: Path, value: Any) -> None:
    """Replace a JSON checkpoint atomically, leaving the old file on failure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, suffix=".tmp", delete=False,
        ) as checkpoint:
            temporary = Path(checkpoint.name)
            json.dump(json_safe(value), checkpoint, indent=2, ensure_ascii=False, allow_nan=False)
            checkpoint.write("\n")
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def completion_status(record: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """Completion is a saved pipeline result, not the controller's success score."""
    verification = record.get("verification")
    verification = verification if isinstance(verification, dict) else {}
    raw = verification.get("raw_result")
    raw = raw if isinstance(raw, dict) else {}
    error = record.get("error") or raw.get("error")
    if error:
        return "failed", (error if isinstance(error, dict)
                          else {"type": "RunError", "message": str(error)})
    tuning = record.get("tuning")
    tuning = tuning if isinstance(tuning, dict) else {}
    if (tuning.get("completed") is True and isinstance(tuning.get("best_params"), dict)
            and tuning["best_params"] and isinstance(raw.get("metrics"), dict)):
        return "completed", None
    return "failed", {"type": "IncompleteResult",
                      "message": "Saved tuning or verification did not finish."}


def build_progress_plan(
    artifacts: list[tuple[Path, dict[str, Any]]], models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Inventory matching saved seeded runs for the current selection, without APIs."""
    plan = []
    for config in models:
        for seed in RUN_SEEDS:
            run_config = {**config, "run_seed": seed}
            for path, artifact in artifacts:
                report = result_directory(run_config) / f"{path.stem}_results.json"
                saved_records = []
                read_error = None
                if report.is_file():
                    try:
                        saved_records = json.loads(report.read_text(encoding="utf-8"))
                        if not isinstance(saved_records, list):
                            raise ValueError("The saved report must contain a list.")
                    except (OSError, ValueError) as error:
                        read_error = {"type": type(error).__name__, "message": str(error)}
                        saved_records = []
                for test in artifact["tests"]["combinations"]:
                    item = {
                        "llm_model": config["model"],
                        "model_folder": config["folder_name"],
                        "seed": seed,
                        "artifact": path.name,
                        "dynamic_model": artifact["system_name"],
                        "test_id": test["test_id"],
                        "result_file": report.relative_to(RESULTS_DIR).as_posix(),
                        "status": "pending",
                    }
                    matches = [
                        record for record in saved_records if isinstance(record, dict)
                        and record.get("artifact") == path.name
                        and record.get("test_id") == test["test_id"]
                        and isinstance(record.get("model"), dict)
                        and (record.get("model") or {}).get("name") == artifact["system_name"]
                        and isinstance(record.get("seeds"), dict)
                        and (record.get("seeds") or {}).get("run") == seed
                        and record.get("agent_mode") == ("mock" if USE_MOCK_AGENTS else config["model"])
                        and record.get("initial_condition") == test.get("initial_condition")
                        and record.get("target") == test.get("target")
                        and record.get("scenario_level") == test.get("scenario_level")
                    ]
                    if matches:
                        item["status"], error = completion_status(matches[-1])
                        if error:
                            item["error"] = error
                    elif read_error:
                        item["saved_report_issue"] = read_error
                    plan.append(item)
    return plan


def progress_key(model_folder: str, seed: int, artifact: str, test_id: str) -> tuple:
    return model_folder, seed, artifact, test_id


def save_progress(
    plan: list[dict[str, Any]], status: str, last_error: dict[str, Any] | None = None,
) -> Path:
    """Save completed/remaining lists and counts before/after each combination."""
    completed = [item for item in plan if item["status"] == "completed"]
    remaining = [item for item in plan if item["status"] != "completed"]
    failed = [item for item in plan if item["status"] == "failed"]
    running = [item for item in plan if item["status"] == "running"]
    total, done = len(plan), len(completed)
    snapshot = {
        "scope": "Current selected models, RUN_SEEDS, artifacts and tests.",
        "completion_definition": "Tuning and verification finished and the JSON was saved; not controller success.",
        "status": status,
        "summary": {
            "done": done,
            "total": total,
            "done_out_of_total": f"{done}/{total}",
            "progress_percent": round(100 * done / total, 2) if total else 0.0,
            "remaining": len(remaining),
            "pending": sum(item["status"] == "pending" for item in plan),
            "running": len(running),
            "failed": len(failed),
        },
        "current_combination": running[0] if running else None,
        "completed_combinations": completed,
        "remaining_combinations": remaining,
        "failed_combinations": failed,
        "last_error": last_error,
    }
    output = RESULTS_DIR / PROGRESS_FILENAME
    write_json_checkpoint(output, snapshot)
    return output


def save_artifact_results(
    artifact_path: Path, config: dict[str, Any], records: list[dict[str, Any]],
) -> Path:
    """Checkpoint this artifact's test list without replacing other plants."""
    filename = f"{artifact_path.stem}_results.json"
    if not _safe_folder(filename) or not _safe_folder(config["folder_name"]):
        raise ValueError("Artifact result filename and model folder must be safe names.")
    output = result_directory(config) / filename
    write_json_checkpoint(output, records)
    return output


def main() -> None:
    if not RUN_BENCHMARK:
        print("Benchmark disabled. Set RUN_BENCHMARK=True to run it.")
        return
    validate_settings()
    artifacts = selected_artifacts()  # Validate the selection before any API calls.
    models = selected_models()
    plan = build_progress_plan(artifacts, models)
    progress_items = {
        progress_key(item["model_folder"], item["seed"], item["artifact"], item["test_id"]): item
        for item in plan
    }
    progress_path = save_progress(plan, "running")
    total_tests = sum(len(artifact["tests"]["combinations"]) for _, artifact in artifacts)
    print(f"Selected {len(artifacts)} artifacts, {total_tests} tests, "
          f"{len(models)} agent mode(s), {len(RUN_SEEDS)} seeds.")
    print(f"Run seeds: {RUN_SEEDS}; {total_tests * len(models) * len(RUN_SEEDS)} tuning runs.")
    active = None
    try:
        for config in models:
            print(f"Selected model: {config['model']} -> {config['folder_name']}")
            for seed in RUN_SEEDS:
                run_config = {**config, "run_seed": seed}
                configure_model(run_config, seed)
                completed_tests = 0
                verified_tests = 0
                for path, artifact in artifacts:
                    records = []
                    # This artifact's file is replaced as its tests are rerun.
                    # Do not keep counting old later tests that may be overwritten.
                    for test in artifact["tests"]["combinations"]:
                        item = progress_items[progress_key(config["folder_name"], seed, path.name, test["test_id"])]
                        item["status"] = "pending"
                        item.pop("error", None)
                        item.pop("saved_report_issue", None)
                    for test in artifact["tests"]["combinations"]:
                        active = progress_items[progress_key(config["folder_name"], seed, path.name, test["test_id"])]
                        active["status"] = "running"
                        save_progress(plan, "running")
                        print(f"[{completed_tests + 1}/{total_tests}] {config['folder_name']}/seed{seed} "
                              f"{artifact['system_name']} {test['test_id']}", flush=True)
                        record = run_one(path, artifact, test, run_config, seed)
                        records.append(record)
                        output = save_artifact_results(path, run_config, records)
                        active["status"], error = completion_status(record)
                        if error:
                            active["error"] = error
                        active = None
                        save_progress(plan, "running")
                        completed_tests += 1
                        raw = (record.get("verification") or {}).get("raw_result") or {}
                        verified_tests += int("metrics" in raw and "error" not in raw)
                        if record.get("error"):
                            print(f"  Failed: {record['error']['message']}", flush=True)
                    print(f"Accepted results report: {output}")
                print(f"Verified {verified_tests}/{total_tests} for {config['folder_name']}/seed{seed}.")
    except BaseException as error:
        if active is not None:
            active["status"] = "pending"
        save_progress(plan, "interrupted", {"type": type(error).__name__, "message": str(error)})
        raise
    save_progress(plan, "finished")
    done = sum(item["status"] == "completed" for item in plan)
    print(f"Progress: {done}/{len(plan)} done. Tracker: {progress_path}")
    print("Next: run finalize_results.py, then plot_dashboard.py.")


if __name__ == "__main__":
    main()
