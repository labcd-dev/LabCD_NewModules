"""Run one manually configured test per MPC artifact.

Edit artifacts/*.json, choose the LLM settings below, and run this script.
Results are compact: trajectory arrays and agent histories are not saved.
No tuning, file creation, or API calls occur when this module is imported.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np

MPC_DIR = Path(__file__).resolve().parent
REPO_ROOT = MPC_DIR.parents[1]
ARTIFACTS_DIR = MPC_DIR / "artifacts"
RESULTS_DIR = MPC_DIR / "json_reports" / "results"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend_core.AgentMPC.agents.evaluator import run_closed_loop
from backend_core.AgentMPC.agents.llm_base import TokenUsageTracker, configure_llm
from backend_core.AgentMPC.agents.scenario_presets import apply_scenario_level
from backend_core.AgentMPC.dynamics.loader import DynamicLoader
from backend_core.AgentMPC.graph.workflow import build_ui_tuning_graph, initial_state
from backend_core.AgentMPC.mpc.config import Config, DataConfig, MPCConfig

# ------------------------- USER SETTINGS -------------------------
RUN_BENCHMARK = True
USE_MOCK_AGENTS = False  # True avoids API calls and uses reproducible random seeds.
# None selects all artifacts; a list selects only those plants.
ARTIFACT_FILES: list[str] | None = None
# Example: ARTIFACT_FILES = ["double_integrator.json", "cart_pole.json"]

# True selects only the cheapest entry according to the repository price table.
# False benchmarks every entry in LLM_CONFIGS. Mock mode ignores both settings.
CHEAPEST_MODEL_ONLY = True
LLM_CONFIGS = [
    # {"folder_name": "gpt-5.4-mini", "provider": "openai",
    #  "model": "gpt-5.4-mini", "api_key_env_var": "OPENAI_API_KEY"},
    # {"folder_name": "gpt-5.4-nano", "provider": "openai",
    #  "model": "gpt-5.4-nano", "api_key_env_var": "OPENAI_API_KEY"},
    # {"folder_name": "gpt-4o-mini", "provider": "openai",
    #  "model": "gpt-4o-mini", "api_key_env_var": "OPENAI_API_KEY"},
    # {"folder_name": "gpt-4.1-nano", "provider": "openai",
    #  "model": "gpt-4.1-nano", "api_key_env_var": "OPENAI_API_KEY"},
    {"folder_name": "groq-gpt-oss-120b", "provider": "groq",
     "model": "openai/gpt-oss-120b", "api_key_env_var": "GROQ_API_KEY"},
]
# To force one model, leave only that entry in LLM_CONFIGS.

LLM_TEMPERATURE = 0.3
LLM_SEED = 42
LLM_MAX_RETRIES = 2
LLM_TIMEOUT_SECONDS = 60.0
LLM_MAX_TOKENS = 2000

MAX_ITERATIONS = 20
MIN_EXPLORE_ITERATIONS = 4
EXPLORATION_INTENSITY = 50
RECURSION_LIMIT = 500
USER_GUIDANCE = ""
COST_WEIGHTS = None
# None lets the Actor propose the first MPC parameters.
SEED_PARAMS = None

SIMULATION_TIME = 5.0
DT_SIMULATION = 0.001
DT_MPC = 0.02
SETTLING_TOLERANCE = 0.05
BASE_RANDOM_SEED = 42
MOCK_RANDOM_SEED = 42
MOCK_COST_USD_PER_TOKEN = 0.001
OUTPUT_FILENAME = "batch_tuning_results.json"
# ---------------------------------------------------------------


def validate_complexity(value: Any) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10
    ):
        raise ValueError("total_complexity must be null or an integer from 1 to 10.")


def load_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or not artifact.get("system_name"):
        raise ValueError(f"{path.name} must define system_name.")
    if not isinstance(artifact.get("python_code"), str) or not artifact["python_code"].strip():
        raise ValueError(f"{path.name} must define python_code.")
    validate_complexity(artifact.get("total_complexity"))
    test = artifact.get("test")
    if not isinstance(test, dict):
        raise ValueError(f"{path.name} must contain exactly one test object.")
    n_states = int(artifact["metadata"]["n_states"])
    for key in ("initial_condition", "target"):
        vector = np.asarray(test[key], dtype=float)
        if vector.shape != (n_states,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{path.name} {key} must be a finite vector of length {n_states}.")
    if isinstance(test["scenario_level"], bool) or test["scenario_level"] not in (1, 2, 3):
        raise ValueError(f"{path.name} scenario_level must be 1, 2, or 3.")
    settings = test["scenario_settings"]
    for key in ("noise_state_mask", "robust_state_mask"):
        mask = settings[key]
        if len(mask) != n_states or any(not isinstance(value, bool) for value in mask):
            raise ValueError(f"{path.name} {key} must have one boolean per state.")
    for key in ("noise_std_value", "robust_push_scale", "robust_noise_fraction"):
        value = float(settings[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{path.name} {key} must be finite and nonnegative.")
    if not isinstance(settings["perturb_physical_params"], bool):
        raise ValueError(f"{path.name} perturb_physical_params must be a boolean.")
    return artifact


def selected_artifacts() -> list[tuple[int, Path, dict[str, Any]]]:
    library = sorted(ARTIFACTS_DIR.glob("*.json"))
    if not library:
        raise FileNotFoundError(f"No artifacts found in {ARTIFACTS_DIR}.")
    names = [path.name for path in library] if ARTIFACT_FILES is None else ARTIFACT_FILES
    if not names or len(names) != len(set(names)):
        raise ValueError("ARTIFACT_FILES must contain unique artifact filenames.")
    unknown = set(names) - {path.name for path in library}
    if unknown:
        raise ValueError(f"Unknown artifacts: {sorted(unknown)}.")
    return [
        (index, path, load_artifact(path))
        for index, path in enumerate(library) if path.name in names
    ]


def selected_models() -> list[dict[str, Any]]:
    if USE_MOCK_AGENTS:
        return [{"folder_name": "mock", "provider": None, "model": "mock-random"}]
    if not LLM_CONFIGS:
        raise ValueError("LLM_CONFIGS must contain at least one paid-model configuration.")
    names = [str(config["folder_name"]) for config in LLM_CONFIGS]
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    for name in names:
        if (not name or name in (".", "..") or any(char in '<>:"/\\|?*' for char in name)
                or name.endswith((".", " ")) or name.upper().split(".")[0] in reserved):
            raise ValueError(f"Unsafe model result folder: {name!r}.")
    if len(names) != len(set(names)):
        raise ValueError("LLM_CONFIGS folder_name values must be unique.")
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


def configure_model(config: dict[str, Any]) -> None:
    if USE_MOCK_AGENTS:
        return
    from dotenv import load_dotenv
    from labcd_agents import LLMFactory, require_api_key
    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=True)
    api_key = require_api_key(provider=config["provider"], env_var=config["api_key_env_var"])

    def factory():
        return LLMFactory.create(
            model=config["model"], provider=config["provider"], api_key=api_key,
            temperature=LLM_TEMPERATURE, seed=LLM_SEED, max_retries=LLM_MAX_RETRIES,
            timeout=LLM_TIMEOUT_SECONDS, max_tokens=LLM_MAX_TOKENS,
        )
    configure_llm(factory=factory)


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


def run_one_artifact(
    config: dict[str, Any], plant_index: int, path: Path,
    artifact: dict[str, Any], module_dir: Path,
) -> dict[str, Any]:
    test = artifact["test"]
    seed = (MOCK_RANDOM_SEED if USE_MOCK_AGENTS else BASE_RANDOM_SEED) + plant_index * 1000 + int(test["scenario_level"])
    record = {
        "artifact": path.name,
        "model": {
            "name": artifact["system_name"],
            "number_of_states": int(artifact["metadata"]["n_states"]),
            "number_of_inputs": int(artifact["metadata"]["n_inputs"]),
        },
        "initial_condition": list(test["initial_condition"]),
        "target": list(test["target"]),
        "scenario_level": int(test["scenario_level"]),
        "scenario_name": test["scenario_name"],
        "agent_mode": "mock" if USE_MOCK_AGENTS else config["model"],
        "total_complexity": None,
        "success_rate": None,
        "tuning": None,
        "verification": None,
    }
    try:
        module_path = module_dir / f"{path.stem}.py"
        module_path.write_text(artifact["python_code"], encoding="utf-8")
        plugin = DynamicLoader.load_from_path(path=str(module_path))
        dynamics = plugin.create_dynamics(params={})
        if (plugin.dynamics_class.__name__ != artifact["system_name"]
                or dynamics.n_states != record["model"]["number_of_states"]
                or dynamics.n_inputs != record["model"]["number_of_inputs"]):
            raise ValueError(f"{path.name}: dynamics dimensions/name disagree with metadata.")
        declared_initial = np.asarray(test["initial_condition"], dtype=float)
        declared_target = np.asarray(test["target"], dtype=float)
        bounds = dynamics.get_state_bounds()
        if bounds is not None:
            lower, upper = bounds
            for label, vector in (("initial_condition", declared_initial), ("target", declared_target)):
                if np.any(vector < lower) or np.any(vector > upper):
                    raise ValueError(f"{path.name} {label} lies outside the declared state bounds.")
        dynamics.config.default_initial_state = declared_initial.copy()
        dynamics.config.default_target = declared_target.copy()
        cfg = Config(
            data=DataConfig(
                simulation_time=SIMULATION_TIME, dt_simulation=DT_SIMULATION,
                dt_mpc=DT_MPC, settling_tolerance=SETTLING_TOLERANCE,
                trajectory_mode="reg",
            ),
            mpc=MPCConfig(), system_name=artifact["system_name"], random_seed=seed,
        )
        settings = test["scenario_settings"]
        applied_initial, applied_target, perturbed = apply_scenario_level(
            dynamics=dynamics, cfg=cfg, level=int(test["scenario_level"]),
            noise_std_value=float(settings["noise_std_value"]),
            noise_state_mask=np.asarray(settings["noise_state_mask"], dtype=bool),
            robust_push_scale=float(settings["robust_push_scale"]),
            robust_state_mask=np.asarray(settings["robust_state_mask"], dtype=bool),
            robust_noise_fraction=float(settings["robust_noise_fraction"]),
            perturb_physical_params=settings["perturb_physical_params"],
        )
        if not np.allclose(applied_initial, declared_initial):
            record["simulated_initial_condition"] = applied_initial.tolist()
        if not np.allclose(applied_target, declared_target):
            record["simulated_target"] = applied_target.tolist()
        if perturbed:
            record["perturbed_physical_parameters"] = json_safe(perturbed)
        tracker = TokenUsageTracker(default_model=config["model"])
        graph = build_ui_tuning_graph(
            dynamics=dynamics, cfg=cfg,
            entry_node="evaluator" if SEED_PARAMS else "actor",
            use_mock_agents=USE_MOCK_AGENTS,
        )
        state = initial_state(
            dynamics=dynamics, system_name=artifact["system_name"],
            max_iterations=MAX_ITERATIONS, ui_scenario_level=int(test["scenario_level"]),
            seed_params=copy.deepcopy(SEED_PARAMS), user_guidance=USER_GUIDANCE,
            min_explore_iterations=MIN_EXPLORE_ITERATIONS,
            cost_weights=copy.deepcopy(COST_WEIGHTS), exploration_intensity=EXPLORATION_INTENSITY,
            dt_mpc=DT_MPC, token_tracker=tracker, mock_seed=seed,
        )
        final_state = graph.invoke(input=state, config={"recursion_limit": RECURSION_LIMIT})
        record["tuning"] = clean_tuning_result(final_state, tracker, cfg)
        best_params = record["tuning"]["best_params"]
        if best_params is None:
            raise RuntimeError("The tuning graph produced no best_params.")
        cfg.data.dt_mpc = float(best_params["dt"])
        verification = run_closed_loop(
            dynamics=dynamics, cfg=cfg,
            params={key: best_params[key] for key in ("Np", "Nc", "Q", "R", "P")},
            max_steps=None, rng=np.random.default_rng(seed=seed),
        )
        record["verification"] = clean_verification(verification)
        if "error" in verification:
            raise RuntimeError(f"Verification failed: {verification['error']}")
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    return json_safe(record)


def main() -> None:
    if not RUN_BENCHMARK:
        print("Benchmark disabled. Set RUN_BENCHMARK=True to run it.")
        return
    artifacts = selected_artifacts()  # Validate every chosen test before using an API.
    models = selected_models()
    if Path(OUTPUT_FILENAME).name != OUTPUT_FILENAME or not OUTPUT_FILENAME.endswith(".json"):
        raise ValueError("OUTPUT_FILENAME must be a plain JSON filename.")
    print(f"One test per artifact: {len(artifacts)} plants, {len(models)} selected LLM configuration(s).")
    with tempfile.TemporaryDirectory(prefix="labcd_discrete_mpc_") as module_dir:
        for config in models:
            print(f"Selected model: {config['model']} -> {config['folder_name']}")
            configure_model(config)
            output = RESULTS_DIR / config["folder_name"] / OUTPUT_FILENAME
            output.parent.mkdir(parents=True, exist_ok=True)
            records = []
            for index, (plant_index, path, artifact) in enumerate(artifacts, start=1):
                print(f"[{index}/{len(artifacts)}] {artifact['system_name']}")
                record = run_one_artifact(config, plant_index, path, artifact, Path(module_dir))
                records.append(record)
                output.write_text(
                    json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
                )
                if record.get("error"):
                    print(f"  Failed: {record['error']['message']}")
            verified = sum("metrics" in (record.get("verification") or {}) for record in records)
            print(f"Verified {verified}/{len(records)}. Base report: {output}")
    print("Next: fill each artifact's total_complexity, run finalize_results.py, then plot_dashboard.py.")


if __name__ == "__main__":
    main()
