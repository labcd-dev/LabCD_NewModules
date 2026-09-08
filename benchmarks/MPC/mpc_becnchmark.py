# BATCH MPC TUNING BENCHMARK
# Mock mode tests the full simulation/JSON pipeline without making API calls.

from pathlib import Path
from typing import Any
import atexit
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import numpy as np


def find_repo_root(start_directory):
    current = Path(start_directory).resolve()
    candidates = [current, *current.parents]
    candidates.extend(path for path in current.iterdir() if path.is_dir())
    for candidate in candidates:
        if (candidate / "backend_core" / "AgentMPC").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find backend_core/AgentMPC. Start Jupyter in the repository or its parent folder."
    )


REPO_ROOT = find_repo_root(Path.cwd())
BENCHMARK_DIR = REPO_ROOT / "benchmarks"
MPC_DIR = BENCHMARK_DIR / "MPC"
JSON_REPORTS_DIR = MPC_DIR / "json_reports"
RESULTS_DIR = JSON_REPORTS_DIR / "results"
ARTIFACTS_DIR = BENCHMARK_DIR / "artifacts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MPC_DIR) not in sys.path:
    sys.path.insert(0, str(MPC_DIR))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from backend_core.AgentMPC.agents.evaluator import run_closed_loop
from backend_core.AgentMPC.agents.llm_base import TokenUsageTracker, configure_llm
from backend_core.AgentMPC.agents.scenario_presets import apply_scenario_level
from backend_core.AgentMPC.dynamics.loader import DynamicLoader
from backend_core.AgentMPC.graph.workflow import build_ui_tuning_graph, initial_state
from backend_core.AgentMPC.mpc.config import Config, DataConfig, MPCConfig
from make_tests import TESTS_KEY


# ------------------------- USER INPUTS -------------------------
RUN_BATCH_LLM_TUNING = True
USE_MOCK_AGENTS = False
# USE_MOCK_AGENTS = True
MOCK_RANDOM_SEED = 42
MOCK_COST_USD_PER_TOKEN = 0.001

# Select artifacts and exact combinations here. Each key is an artifact JSON
# filename and each value is a list of test_id values from tests.combinations.
# For example, ic1_s1 means IC/target case 1 with scenario level 1.
# An empty list selects every combination of that one artifact. Set the whole
# value to None only when you intentionally want every artifact and every test.
# Examples:
# BENCHMARK_SELECTION = {
#     "double_integrator.json": ["ic1_s1", "ic2_s2"],
#     "cart_pole.json": ["ic1_s1"],
# }
# BENCHMARK_SELECTION = {"double_integrator.json": []}  # all 9 of this plant
# BENCHMARK_SELECTION = None  # all artifacts and all combinations

# Ten tests spread across the current total-complexity range (about 1.06-6.18).
BENCHMARK_SELECTION: dict[str, list[str]] | None = {
    "double_integrator.json": ["ic1_s1", "ic1_s2"],
    "two_mass_spring.json": ["ic1_s1"],
    "planar_drone.json": ["ic2_s1", "ic3_s2"],
    "cart_pole.json": ["ic1_s2", "ic2_s3"],
    "triple_mass_spring.json": ["ic1_s2", "ic3_s3"],
    "mass_spring_damper.json": ["ic2_s3"],
}

# Every enabled entry is benchmarked when USE_MOCK_AGENTS=False. Folder names
# must be unique and should be safe Windows directory names.
LLM_CONFIGS = [
    # {
    #     "folder_name": "gpt-4o-mini",
    #     "provider": "openai",
    #     "model": "gpt-4o-mini",
    #     "api_key_env_var": "OPENAI_API_KEY",
    # },
    # Add more models by copying this dictionary, for example:
    {
        "folder_name": "groq-gpt-oss-120b",
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "api_key_env_var": "GROQ_API_KEY",
    },
]

# Internal variable used only when this script launches one isolated process
# per real LLM configuration. Users should not set it manually.
ACTIVE_LLM_INDEX_ENV = "LABCD_MPC_ACTIVE_LLM_INDEX"
active_llm_index_text = os.environ.get(ACTIVE_LLM_INDEX_ENV)
ACTIVE_LLM_INDEX = (
    int(active_llm_index_text) if active_llm_index_text is not None else None
)

if USE_MOCK_AGENTS:
    ACTIVE_AGENT_CONFIG = {
        "folder_name": "mock",
        "provider": None,
        "model": "mock-random",
        "api_key_env_var": None,
    }
elif ACTIVE_LLM_INDEX is not None:
    if not 0 <= ACTIVE_LLM_INDEX < len(LLM_CONFIGS):
        raise IndexError(f"Invalid {ACTIVE_LLM_INDEX_ENV}={ACTIVE_LLM_INDEX}.")
    ACTIVE_AGENT_CONFIG = LLM_CONFIGS[ACTIVE_LLM_INDEX]
else:
    ACTIVE_AGENT_CONFIG = None


def validate_result_folder_name(folder_name):
    name = str(folder_name)
    invalid_characters = set('<>:"/\\|?*')
    if (
        not name
        or name in {".", ".."}
        or Path(name).name != name
        or any(character in invalid_characters for character in name)
    ):
        raise ValueError(f"Unsafe result folder_name: {name!r}.")
    return name


ACTIVE_FOLDER_NAME = (
    validate_result_folder_name(ACTIVE_AGENT_CONFIG["folder_name"])
    if ACTIVE_AGENT_CONFIG is not None
    else "_llm_orchestrator"
)
ACTIVE_RESULTS_DIR = RESULTS_DIR / (
    ACTIVE_FOLDER_NAME
)
if ACTIVE_AGENT_CONFIG is not None:
    ACTIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = ACTIVE_RESULTS_DIR / "batch_tuning_results.json"

LLM_TEMPERATURE = 0.3
LLM_SEED = 42
LLM_MAX_RETRIES = 2
LLM_TIMEOUT_SECONDS = 60.0
LLM_MAX_TOKENS = 2000

# Tuning behavior.
MAX_ITERATIONS = 10
MIN_EXPLORE_ITERATIONS = 4
EXPLORATION_INTENSITY = 50
# ENTRY_NODE = "evaluator"
RECURSION_LIMIT = 500
# USER_GUIDANCE = (
#     "Prioritize position/configuration tracking and closed-loop stability. "
#     "Keep velocity/rate errors controlled, avoid excessive actuator effort, "
#     "and keep the controller suitable for a 50 Hz control loop."
# )
# COST_WEIGHTS = {
#     "mse": 1.0,
#     "overshoot": 2.0,
#     "settling_time": 0.5,
#     "control_effort": 0.05,
#     "oscillation": 0.05,
#     "unsettled_penalty": 1.0,
# }

# Scenario and simulation settings.
SIMULATION_TIME = 5.0
DT_SIMULATION = 0.001
DT_MPC = 0.02
SETTLING_TOLERANCE = 0.05
# SCENARIO_NOISE_STD_VALUE = 0.01
# ROBUST_PUSH_SCALE = 1.0
# ROBUST_NOISE_FRACTION = 0.5
# PERTURB_PHYSICAL_PARAMETERS = False
BASE_RANDOM_SEED = 42

# # Initial MPC guess supplied to the evaluator before LLM exploration.
# INITIAL_NP = 20
# INITIAL_NC = 5
# INITIAL_POSITION_Q = 100.0
# INITIAL_RATE_Q = 10.0
# INITIAL_INPUT_R = 0.1
# INITIAL_POSITION_P = 200.0
# INITIAL_RATE_P = 20.0

# ---------------------------------------------------------------


_ARTIFACT_MODULE_DIR: tempfile.TemporaryDirectory | None = None


def artifact_library() -> list[Path]:
    """Every artifact in the library, in a stable order.

    This order is the basis of the per-plant random seed, so it is always the
    full library -- never the BENCHMARK_SELECTION subset. Benchmarking a subset
    therefore reproduces exactly the runs it would have produced as part of a
    full sweep.
    """
    paths = sorted(
        path.resolve()
        for path in ARTIFACTS_DIR.glob("*.json")
        if not path.name.startswith("_")
    )
    if not paths:
        raise FileNotFoundError(f"No JSON artifacts found in {ARTIFACTS_DIR}.")
    return paths


def artifact_paths() -> list[Path]:
    """The artifacts to benchmark, in a stable order."""
    library = artifact_library()
    if BENCHMARK_SELECTION is None:
        return library
    if not BENCHMARK_SELECTION:
        raise ValueError(
            "BENCHMARK_SELECTION cannot be empty. Provide at least one artifact "
            "or set it to None for a deliberate full sweep."
        )

    paths = [(ARTIFACTS_DIR / name).resolve() for name in BENCHMARK_SELECTION]
    unknown = [path for path in paths if path not in library]
    if unknown:
        raise FileNotFoundError(
            "BENCHMARK_SELECTION names files that are not in "
            f"{ARTIFACTS_DIR}: " + ", ".join(path.name for path in unknown)
        )
    return sorted(paths, key=library.index)


def load_artifact(path: Path) -> dict[str, Any]:
    """Read one artifact and check it carries a generated test matrix."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise TypeError(f"{path.name} must contain a single JSON artifact object.")

    tests = artifact.get(TESTS_KEY)
    if not isinstance(tests, dict) or not tests.get("combinations"):
        raise ValueError(
            f"{path.name} has no {TESTS_KEY!r} combinations. "
            "Run benchmarks/make_tests.py first."
        )

    return artifact


def load_artifact_dynamics(artifact: dict[str, Any], path: Path) -> DynamicLoader:
    """Materialise the artifact's `python_code` so DynamicLoader can load it.

    DynamicLoader only accepts a real ``.py`` file, and it injects
    ``BaseDynamics`` / ``SystemConfig`` / ``np`` into the module namespace, so the
    embedded source is written out verbatim to a throwaway file named after the
    artifact and loaded from there. The temporary directory is removed when the
    process exits.
    """
    global _ARTIFACT_MODULE_DIR

    python_code = artifact.get("python_code")
    if not isinstance(python_code, str) or not python_code.strip():
        raise ValueError(f"{path.name} has no 'python_code' to load.")

    if _ARTIFACT_MODULE_DIR is None:
        _ARTIFACT_MODULE_DIR = tempfile.TemporaryDirectory(prefix="labcd_artifacts_")
        atexit.register(_ARTIFACT_MODULE_DIR.cleanup)

    module_path = Path(_ARTIFACT_MODULE_DIR.name) / f"{path.stem}.py"
    module_path.write_text(python_code, encoding="utf-8")

    loaded_plugin = DynamicLoader.load_from_path(path=str(module_path))

    declared_name = artifact.get("system_name")
    actual_name = loaded_plugin.dynamics_class.__name__
    if declared_name and declared_name != actual_name:
        raise ValueError(
            f"{path.name} declares system_name {declared_name!r} but its "
            f"python_code defines {actual_name!r}."
        )
    return loaded_plugin


def selected_tests(artifact: dict[str, Any], artifact_path: Path) -> list[dict[str, Any]]:
    """Return all tests or the exact test IDs selected for this artifact."""
    available = list(artifact[TESTS_KEY]["combinations"])
    if BENCHMARK_SELECTION is None:
        return available

    requested = list(BENCHMARK_SELECTION[artifact_path.name])
    if not requested:
        return available
    if len(requested) != len(set(requested)):
        raise ValueError(
            f"BENCHMARK_SELECTION contains duplicate test IDs for {artifact_path.name}."
        )

    available_by_id = {str(test["test_id"]): test for test in available}
    unknown = [test_id for test_id in requested if test_id not in available_by_id]
    if unknown:
        raise ValueError(
            f"Unknown test_id values for {artifact_path.name}: {unknown}. "
            f"Available values: {list(available_by_id)}."
        )
    return [available_by_id[test_id] for test_id in requested]


def json_safe(value: Any):
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


def simulated_mock_token_usage(final_state, agent_messages):
    """Create reproducible, nonzero fake usage for JSON-format testing."""
    call_count = sum(len(messages) for messages in agent_messages.values())
    rng = np.random.default_rng(
        seed=int(final_state.get("mock_seed", MOCK_RANDOM_SEED)) + 701
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
        "price_usd_per_token": MOCK_COST_USD_PER_TOKEN,
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


def clean_verification(verification):
    """Keep final verification summaries and exclude all trajectory arrays.

    The success rate is not stored here -- it is a top-level record field.
    """
    if verification is None:
        return None
    if "error" in verification:
        return json_safe(
            {
                "error": verification.get("error"),
                "traceback": verification.get("traceback"),
            }
        )

    metrics = verification["metrics"].as_dict()
    return json_safe(
        {
            "metrics": {
                "mse": metrics["mse"],
                "rmse": metrics["rmse"],
                "overshoot": metrics["overshoot"],
                "settling_time": metrics["settling_time"],
                "control_effort": metrics["control_effort"],
                "oscillation_count": metrics["oscillation_count"],
            },
            "status": {
                "success": verification["success"],
                "settled": metrics["settled"],
                "is_stable": metrics["is_stable"],
                "unstable": verification["unstable"],
                "unstable_reason": verification["unstable_reason"],
            },
            "average_mpc_solution_time_seconds": verification["avg_solve_time"],
            "solver_diagnostics": verification["solver_diagnostics"],
        }
    )


def save_checkpoint(records):
    clean_records = json_safe(records)
    OUTPUT_FILE.write_text(
        json.dumps(clean_records, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


if not RUN_BATCH_LLM_TUNING:
    print("Batch LLM tuning is disabled. Set RUN_BATCH_LLM_TUNING=True to run it.")
elif not USE_MOCK_AGENTS and ACTIVE_AGENT_CONFIG is None:
    if not LLM_CONFIGS:
        raise ValueError("LLM_CONFIGS must contain at least one model in real-LLM mode.")
    folder_names = [
        validate_result_folder_name(config["folder_name"]) for config in LLM_CONFIGS
    ]
    if len(folder_names) != len(set(folder_names)):
        raise ValueError("Every LLM_CONFIGS folder_name must be unique.")

    print(f"Real-LLM mode: running {len(LLM_CONFIGS)} model configurations.")
    for llm_index, llm_config in enumerate(LLM_CONFIGS):
        print(
            f"Starting model {llm_index + 1}/{len(LLM_CONFIGS)}: "
            f"{llm_config['model']} -> results/{llm_config['folder_name']}"
        )
        child_environment = os.environ.copy()
        child_environment[ACTIVE_LLM_INDEX_ENV] = str(llm_index)
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=str(REPO_ROOT),
            env=child_environment,
            check=True,
        )
    print("All configured LLM benchmarks finished.")
else:
    # if ENTRY_NODE != "evaluator":
    #     raise ValueError("This batch cell creates seed parameters, so ENTRY_NODE must be 'evaluator'.")

    if USE_MOCK_AGENTS:
        print("Mock-agent mode enabled: no .env file, API key, or LLM calls are used.")
    else:
        from dotenv import load_dotenv
        from labcd_agents import LLMFactory, require_api_key

        dotenv_path = (REPO_ROOT / ".env").resolve()
        if not load_dotenv(dotenv_path=dotenv_path, override=True):
            raise RuntimeError(f"Could not load environment file: {dotenv_path}")

        llm_provider = str(ACTIVE_AGENT_CONFIG["provider"])
        llm_model = str(ACTIVE_AGENT_CONFIG["model"])
        api_key_env_var = str(ACTIVE_AGENT_CONFIG["api_key_env_var"])
        api_key = require_api_key(
            provider=llm_provider,
            env_var=api_key_env_var,
        )

        def create_chat_model():
            return LLMFactory.create(
                model=llm_model,
                temperature=LLM_TEMPERATURE,
                seed=LLM_SEED,
                provider=llm_provider,
                api_key=api_key,
                max_retries=LLM_MAX_RETRIES,
                timeout=LLM_TIMEOUT_SECONDS,
                max_tokens=LLM_MAX_TOKENS,
            )

        configure_llm(factory=create_chat_model)

    # Load and validate every artifact test matrix up front. Complexity is not
    # copied here; finalize_results.py adds it after the benchmark finishes.
    ARTIFACT_LIBRARY = artifact_library()
    loaded_artifacts = []
    for artifact_path in artifact_paths():
        artifact = load_artifact(artifact_path)
        loaded_artifacts.append(
            (
                ARTIFACT_LIBRARY.index(artifact_path),
                artifact_path,
                artifact,
                selected_tests(artifact, artifact_path),
            )
        )

    total_runs = sum(len(tests) for *_, tests in loaded_artifacts)
    # "mock", or the model name in real-LLM mode.
    agent_mode = "mock" if USE_MOCK_AGENTS else str(ACTIVE_AGENT_CONFIG["model"])
    print(
        f"Starting {total_runs} independent {agent_mode} tuning runs across "
        f"{len(loaded_artifacts)} artifacts."
    )
    print(f"Each run allows up to {MAX_ITERATIONS} tuning iterations.")
    print("Tests and complexity scores come from:", ARTIFACTS_DIR.resolve())
    print("Clean results are checkpointed after every run:", OUTPUT_FILE.resolve())

    batch_results = []
    completed_runs = 0

    for plant_index, artifact_path, artifact, tests in loaded_artifacts:
        loaded_plugin = load_artifact_dynamics(artifact, artifact_path)
        probe_dynamics = loaded_plugin.create_dynamics(params={})
        plant_name = loaded_plugin.dynamics_class.__name__
        # position_indices = {
        #     int(pair[0]) for pair in (probe_dynamics.config.derivative_pairs or [])
        # }
        # seed_q = [
        #     INITIAL_POSITION_Q if index in position_indices else INITIAL_RATE_Q
        #     for index in range(probe_dynamics.n_states)
        # ]
        # seed_r = [INITIAL_INPUT_R] * probe_dynamics.n_inputs
        # seed_p = [
        #     INITIAL_POSITION_P if index in position_indices else INITIAL_RATE_P
        #     for index in range(probe_dynamics.n_states)
        # ]
        # seed_parameters = {
        #     "Np": INITIAL_NP,
        #     "Nc": INITIAL_NC,
        #     "Q": seed_q,
        #     "R": seed_r,
        #     "P": seed_p,
        #     "dt": DT_MPC,
        # }

        for test in tests:
            ic_target_id = int(test["ic_target_id"])
            scenario_level = int(test["scenario_level"])
            # Same shape as before the artifact migration, but `plant_index` is
            # now the artifact's position in the alphabetically sorted library
            # rather than its position in the old plugin list, so mock runs are
            # reproducible from here on without matching the pre-migration
            # results in `benchmarks_backup_2026-09-08/`.
            run_seed = (
                (MOCK_RANDOM_SEED if USE_MOCK_AGENTS else BASE_RANDOM_SEED)
                + plant_index * 1000
                + (ic_target_id - 1) * 10
                + scenario_level
            )
            completed_runs += 1
            print(
                f"[{completed_runs}/{total_runs}] {plant_name} {test['test_id']}, "
                f"IC={test['initial_condition']}, target={test['target']}, "
                f"scenario={scenario_level} ({test.get('scenario_name')})"
            )

            record = {
                "model": {
                    "name": plant_name,
                    "number_of_states": int(probe_dynamics.n_states),
                    "number_of_inputs": int(probe_dynamics.n_inputs),
                },
                "test_id": test["test_id"],
                "initial_condition": list(test["initial_condition"]),
                "target": list(test["target"]),
                "scenario_level": scenario_level,
                "scenario_name": test.get("scenario_name"),
                "agent_mode": agent_mode,
                # Filled later by finalize_results.py from this artifact test.
                "total_complexity": None,
                # Filled later by finalize_results.py from verification RMSE.
                "success_rate": None,
                "tuning": None,
                "verification": None,
            }
            final_state = None
            verification = None
            token_tracker = None

            try:
                tuning_dynamics = loaded_plugin.create_dynamics(params={})

                # The artifact stores complete state vectors, so they are used
                # as-is rather than expanded onto a primary derivative pair.
                declared_initial = np.asarray(test["initial_condition"], dtype=float)
                declared_target = np.asarray(test["target"], dtype=float)
                expected_shape = (int(tuning_dynamics.n_states),)
                if declared_initial.shape != expected_shape:
                    raise ValueError(
                        f"{test['test_id']} initial_condition has shape "
                        f"{declared_initial.shape}, expected {expected_shape}."
                    )
                if declared_target.shape != expected_shape:
                    raise ValueError(
                        f"{test['test_id']} target has shape "
                        f"{declared_target.shape}, expected {expected_shape}."
                    )
                tuning_dynamics.config.default_initial_state = declared_initial.copy()
                tuning_dynamics.config.default_target = declared_target.copy()

                tuning_cfg = Config(
                    data=DataConfig(
                        simulation_time=SIMULATION_TIME,
                        dt_simulation=DT_SIMULATION,
                        dt_mpc=DT_MPC,
                        # noise_std=np.zeros(tuning_dynamics.n_states, dtype=float),
                        # feedforward_override=None,
                        # trajectory_mode="reg",
                        # trajectory_amplitude=0.5,
                        # trajectory_frequency=0.5,
                        # trajectory_pulse_start=0.2,
                        # trajectory_pulse_end=0.7,
                        # trajectory_per_state_modes=["reg"] * tuning_dynamics.n_states,
                        # custom_trajectory_fn=None,
                        # settling_tolerance=SETTLING_TOLERANCE,
                    ),
                    mpc=MPCConfig(
                        # prediction_horizon=INITIAL_NP,
                        # control_horizon=INITIAL_NC,
                        # state_weights=np.diag(np.asarray(seed_q, dtype=float)),
                        # input_weights=np.diag(np.asarray(seed_r, dtype=float)),
                        # terminal_weights=np.diag(np.asarray(seed_p, dtype=float)),
                        # u_bounds=tuning_dynamics.get_input_bounds(),
                        # x_bounds=tuning_dynamics.get_state_bounds(),
                        # finite_diff_step=1e-4,
                        # use_analytic_jacobian=False,
                    ),
                    system_name=plant_name,
                    random_seed=run_seed,
                )

                applied_initial, applied_target, _ = apply_scenario_level(
                    dynamics=tuning_dynamics,
                    cfg=tuning_cfg,
                    level=scenario_level,
                    noise_std_value=(test.get("scenario_settings") or {}).get(
                        "noise_std_value"
                    ),
                    noise_state_mask=np.asarray(
                        (test.get("scenario_settings") or {}).get(
                            "noise_state_mask",
                            [True] * tuning_dynamics.n_states,
                        ),
                        dtype=bool,
                    ),
                    robust_push_scale=(test.get("scenario_settings") or {}).get(
                        "robust_push_scale"
                    ),
                    robust_state_mask=np.asarray(
                        (test.get("scenario_settings") or {}).get(
                            "robust_state_mask",
                            [True] * tuning_dynamics.n_states,
                        ),
                        dtype=bool,
                    ),
                    robust_noise_fraction=(test.get("scenario_settings") or {}).get(
                        "robust_noise_fraction"
                    ),
                    perturb_physical_params=bool(
                        (test.get("scenario_settings") or {}).get(
                            "perturb_physical_params", True
                        )
                    ),
                )
                # Level 3 replaces the declared IC with a pushed state, so the
                # state actually simulated is recorded whenever it differs.
                if not np.allclose(applied_initial, declared_initial):
                    record["simulated_initial_condition"] = applied_initial.tolist()
                if not np.allclose(applied_target, declared_target):
                    record["simulated_target"] = applied_target.tolist()

                token_tracker = TokenUsageTracker(
                    default_model=str(ACTIVE_AGENT_CONFIG["model"])
                )
                tuning_graph = build_ui_tuning_graph(
                    dynamics=tuning_dynamics,
                    cfg=tuning_cfg,
                    # entry_node=ENTRY_NODE,
                    use_mock_agents=USE_MOCK_AGENTS,
                )
                tuning_state = initial_state(
                    dynamics=tuning_dynamics,
                    system_name=plant_name,
                    max_iterations=MAX_ITERATIONS,
                    ui_scenario_level=scenario_level,
                    # seed_params=copy.deepcopy(seed_parameters),
                    # user_guidance=USER_GUIDANCE,
                    min_explore_iterations=MIN_EXPLORE_ITERATIONS,
                    # cost_weights=copy.deepcopy(COST_WEIGHTS),
                    exploration_intensity=EXPLORATION_INTENSITY,
                    dt_mpc=DT_MPC,
                    token_tracker=token_tracker,
                    mock_seed=run_seed,
                )
                final_state = tuning_graph.invoke(
                    input=tuning_state,
                    config={"recursion_limit": RECURSION_LIMIT},
                )
                record["tuning"] = clean_tuning_result(
                    final_state=final_state,
                    token_tracker=token_tracker,
                    tuning_cfg=tuning_cfg,
                )

                best_parameters = final_state.get("best_params")
                if best_parameters is None:
                    raise RuntimeError("The tuning graph produced no best_params.")
                saved_best_parameters = record["tuning"]["best_params"]
                tuning_cfg.data.dt_mpc = float(saved_best_parameters["dt"])

                verification = run_closed_loop(
                    dynamics=tuning_dynamics,
                    cfg=tuning_cfg,
                    params={
                        "Np": int(saved_best_parameters["Np"]),
                        "Nc": int(saved_best_parameters["Nc"]),
                        "Q": list(saved_best_parameters["Q"]),
                        "R": list(saved_best_parameters["R"]),
                        "P": list(saved_best_parameters["P"]),
                    },
                    max_steps=None,
                    rng=np.random.default_rng(seed=run_seed),
                )
                if "error" in verification:
                    record["verification"] = clean_verification(verification)
                    raise RuntimeError(
                        f"Best-parameter verification failed: {verification['error']}"
                    )
                record["verification"] = clean_verification(verification)

            except Exception as error:
                record["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }

            batch_results.append(json_safe(record))
            save_checkpoint(batch_results)

    successful_runs = sum(
        isinstance(item.get("verification"), dict)
        and "metrics" in item["verification"]
        for item in batch_results
    )
    print(f"Finished: {successful_runs}/{total_runs} runs verified successfully.")
    print("Clean base JSON written to:", OUTPUT_FILE.resolve())
    print("Next: run benchmarks/MPC/finalize_results.py to create the _full JSON.")
