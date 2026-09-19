"""Run Adaptive Designer/Tuner/Reporter on selected artifact tests.

Edit USER SETTINGS, then run this file. Mock mode never makes LLM API calls.
Each model gets a full JSON array under json_reports/results/<model>/.
Run finalize_results.py afterwards to make the brief report and success score.
Importing this module does not start a benchmark or create files.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ADAPTIVE_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTIVE_DIR.parents[1]
ARTIFACTS_DIR = ADAPTIVE_DIR / "artifacts"
RESULTS_DIR = ADAPTIVE_DIR / "json_reports" / "results"
for directory in (REPO_ROOT, REPO_ROOT / "packages" / "labcd_agents" / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

# ------------------------------ USER SETTINGS ------------------------------
RUN_BENCHMARK = True
USE_MOCK_AGENTS = False  # False uses the models and API keys below.
# ARTIFACT_FILES: list[str] | None = ["single_pole.json"]  # None runs every artifact.
ARTIFACT_FILES: list[str] | None = None  # None runs every artifact.

# Mock mode ignores this list entirely. Every enabled real model runs every
# selected artifact/test; start with one model and one artifact to limit cost.
LLM_CONFIGS = [
    # {"folder_name": "groq-gpt-oss-120b", "provider": "groq",
    #  "model": "openai/gpt-oss-120b", "api_key_env_var": "GROQ_API_KEY"},
    # {"folder_name": "gpt-4o-mini", "provider": "openai",
    #  "model": "gpt-4o-mini", "api_key_env_var": "OPENAI_API_KEY"},
    # {"folder_name": "gpt-5.4-mini", "provider": "openai",
    #  "model": "gpt-5.4-mini", "api_key_env_var": "OPENAI_API_KEY"},
    {"folder_name": "gpt-5.4-nano", "provider": "openai",
     "model": "gpt-5.4-nano", "api_key_env_var": "OPENAI_API_KEY"},
    # {"folder_name": "gpt-4.1-nano", "provider": "openai",
    #  "model": "gpt-4.1-nano", "api_key_env_var": "OPENAI_API_KEY"},
]

ENABLE_TUNING = True
TARGET_STEADY_RMS_FRACTION = 0.02
MAX_TUNING_ROUNDS = 10
TUNING_OBJECTIVES: dict[str, int] | None = {"control_effort": 3}
SIMULATION_TIME = 10.0
SOLVER_STEP = 0.001
MOCK_RANDOM_SEED = 42
ROLE_MAX_TOKENS = {"design": 1500, "tuner": 2000,
                   "clarifier": 8000, "reporter": 500}
SAVE_TIME_SERIES = False  # Histories/metrics remain; large trajectory arrays do not.
OUTPUT_FILENAME = "batch_tuning_results.json"
# ---------------------------------------------------------------------------


def _safe_folder(name: str) -> bool:
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    return (isinstance(name, str) and bool(name) and name not in (".", "..")
            and not any(char in '<>:"/\\|?*' for char in name)
            and not name.endswith((".", " ")) and name.upper().split(".")[0] not in reserved)


def selected_models() -> list[dict[str, str]]:
    if USE_MOCK_AGENTS:
        return [{"folder_name": "mock", "provider": "mock", "model": "mock-adaptive"}]
    if not LLM_CONFIGS:
        raise ValueError("Select at least one real LLM in LLM_CONFIGS.")
    names = [config.get("folder_name") for config in LLM_CONFIGS]
    if not all(_safe_folder(name) for name in names) or len(names) != len(set(names)):
        raise ValueError("LLM_CONFIGS must have unique, safe folder_name values.")
    for config in LLM_CONFIGS:
        if any(not isinstance(config.get(key), str) or not config[key].strip()
               for key in ("provider", "model", "api_key_env_var")):
            raise ValueError("Each LLM needs provider, model, and api_key_env_var.")
    return list(LLM_CONFIGS)


def selected_artifacts() -> list[tuple[Path, dict[str, Any]]]:
    paths = {path.name: path for path in ARTIFACTS_DIR.glob("*.json")}
    names = sorted(paths) if ARTIFACT_FILES is None else ARTIFACT_FILES
    if not names or len(names) != len(set(names)):
        raise ValueError("ARTIFACT_FILES must select one or more unique files.")
    unknown = set(names) - paths.keys()
    if unknown:
        raise FileNotFoundError(f"Unknown Adaptive artifacts: {sorted(unknown)}")
    from backend_core.plant_compiler import PlantCompiler

    compiler = PlantCompiler()
    selected = []
    for name in names:
        path = paths[name]
        artifact = json.loads(path.read_text(encoding="utf-8"))
        complexity = artifact.get("total_complexity")
        if (isinstance(complexity, bool) or not isinstance(complexity, int)
                or not 1 <= complexity <= 10):
            raise ValueError(f"{name}: total_complexity must be an integer from 1 to 10.")
        validation = compiler.validate(artifact)
        if not validation.ok:
            raise ValueError(f"{name}: invalid Plant Agent artifact: {validation.errors}")
        tests = (artifact.get("tests") or {}).get("combinations")
        if not isinstance(tests, list) or not tests:
            raise ValueError(f"{name}: tests.combinations must contain at least one test.")
        ids = [test.get("test_id") for test in tests]
        if any(not isinstance(test_id, str) or not test_id for test_id in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{name}: test_id values must be nonempty and unique.")
        selected.append((path, artifact))
    return selected


def validate_settings() -> None:
    if SIMULATION_TIME not in (5.0, 10.0, 30.0, 100.0):
        raise ValueError("SIMULATION_TIME must be one of 5, 10, 30, or 100 seconds.")
    if SOLVER_STEP not in (0.0001, 0.001, 0.01):
        raise ValueError("SOLVER_STEP must be one of 0.0001, 0.001, or 0.01 seconds.")
    if (not math.isfinite(TARGET_STEADY_RMS_FRACTION)
            or not 0.005 <= TARGET_STEADY_RMS_FRACTION <= 0.10
            or not math.isclose(TARGET_STEADY_RMS_FRACTION / 0.005,
                                round(TARGET_STEADY_RMS_FRACTION / 0.005))):
        raise ValueError("TARGET_STEADY_RMS_FRACTION must be 0.005..0.10 in 0.005 steps.")
    if (isinstance(MAX_TUNING_ROUNDS, bool) or not isinstance(MAX_TUNING_ROUNDS, int)
            or not 1 <= MAX_TUNING_ROUNDS <= 10):
        raise ValueError("MAX_TUNING_ROUNDS must be an integer from 1 to 10.")
    if TUNING_OBJECTIVES is not None:
        from backend_core.AgentAdaptive.tools.tuning_objectives import OBJECTIVES
        if not isinstance(TUNING_OBJECTIVES, dict) or any(
            key not in OBJECTIVES or isinstance(weight, bool)
            or not isinstance(weight, int) or not 1 <= weight <= 5
            for key, weight in TUNING_OBJECTIVES.items()
        ):
            raise ValueError("TUNING_OBJECTIVES needs supported objectives with weights 1..5.")
    if Path(OUTPUT_FILENAME).name != OUTPUT_FILENAME or not OUTPUT_FILENAME.endswith(".json"):
        raise ValueError("OUTPUT_FILENAME must be a plain .json filename.")


def build_system_spec(artifact: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    """Use the same PlantCompiler -> confirmed Adaptive spec path as the notebooks."""
    from backend_core.plant_compiler import PlantCompiler
    from backend_core.AgentAdaptive.tools import system_spec as spec_tools
    from backend_core.AgentAdaptive.controller.structure_build import _build_structure_from_spec

    metadata = artifact["metadata"]
    states = metadata["states"]
    outputs = metadata["outputs"]
    initial = np.asarray(test["initial_condition"], dtype=float)
    if initial.shape != (len(states),) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_condition must have one finite value per state.")
    references = test.get("references")
    if not isinstance(references, list) or len(references) != len(outputs):
        raise ValueError("Provide exactly one reference per declared output.")
    ref_map = {item.get("output"): item.get("expr") for item in references if isinstance(item, dict)}
    if (len(ref_map) != len(outputs) or set(ref_map) != set(outputs)
            or any(not isinstance(expr, str) or not expr.strip() for expr in ref_map.values())):
        raise ValueError("References must name each output once and have nonempty expressions.")

    spec = PlantCompiler().generate_adaptive_spec(
        plant_output=artifact,
        pre_launch={"initial_state": initial.tolist(),
                    "total_simulation_time": SIMULATION_TIME,
                    "solver_sample_time": SOLVER_STEP},
    )
    dynamics = spec["dynamics"]
    dynamics["x0"] = initial.tolist()
    dynamics["references"] = [{"output": name, "expr": ref_map[name]} for name in outputs]
    dynamics["uncertainty"] = copy.deepcopy(test.get("uncertainty", []))
    dynamics["disturbance"] = copy.deepcopy(test.get("disturbance", []))
    dynamics["sim_time"] = SIMULATION_TIME
    dynamics["solver_step"] = SOLVER_STEP
    gaps = spec_tools.missing_items(spec)
    if gaps:
        raise ValueError(f"Incomplete Adaptive system specification: {gaps}")
    _, limitations = _build_structure_from_spec(spec_tools.substitute_parameters(spec))
    if limitations:
        raise ValueError(f"Unsupported Adaptive structure/reference: {limitations}")
    return spec


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "content"):
        return {"type": type(value).__name__, "content": json_safe(value.content)}
    raise TypeError(f"Cannot serialize {type(value).__name__} in a benchmark result.")


def role_environment(config: dict[str, str]) -> dict[str, str]:
    """Return temporary role settings; never save an API key in JSON."""
    routing = {"LABCD_ADAPTIVE_SHOW_PLOTS": "0", "MPLBACKEND": "Agg"}
    if USE_MOCK_AGENTS:
        return routing
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)
    api_key = os.getenv(config["api_key_env_var"])
    if not api_key:
        raise RuntimeError(f"Set {config['api_key_env_var']} in the environment or repository .env.")
    for role in ("design", "tuner", "clarifier", "reporter"):
        suffix = "" if role == "design" else f"_{role.upper()}"
        routing[f"LLM_PROVIDER{suffix}"] = config["provider"]
        routing[f"OPENAI_MODEL{suffix}"] = config["model"]
        routing[f"OPENAI_API_KEY{suffix}"] = api_key
        routing[f"OPENAI_MAX_TOKENS{suffix}"] = str(ROLE_MAX_TOKENS[role])
    return routing


def run_one(
    artifact_path: Path, artifact: dict[str, Any], test: dict[str, Any],
    config: dict[str, str], mock_seed: int,
) -> dict[str, Any]:
    record = {
        "artifact": artifact_path.name,
        "plant": artifact["system_name"],
        "test_id": test["test_id"],
        "test": copy.deepcopy(test),
        "agent_mode": "mock" if USE_MOCK_AGENTS else config["provider"],
        "llm_model": None if USE_MOCK_AGENTS else config["model"],
        "total_complexity": artifact["total_complexity"],
        "mock_seed": mock_seed if USE_MOCK_AGENTS else None,
        "tuning_inputs": {
            "enabled": ENABLE_TUNING,
            "target_error_fraction": TARGET_STEADY_RMS_FRACTION,
            "max_rounds": MAX_TUNING_ROUNDS,
            "objectives": copy.deepcopy(TUNING_OBJECTIVES),
        },
        "applied_system_spec": None,
        "tuning": None,
        "verification": None,
        "pipeline_completed": False,
    }
    try:
        spec = build_system_spec(artifact, test)
        record["applied_system_spec"] = spec
        if USE_MOCK_AGENTS:
            from backend_core.AgentAdaptive.agents.tuner_agent import run_full_pipeline
            agent_options = {"mock_seed": mock_seed}
        else:
            from backend_core.AgentAdaptive.agents.tuner_agent_main import run_full_pipeline
            agent_options = {}

        result, usage, history, best = run_full_pipeline(
            description="",
            enable_tuning=ENABLE_TUNING,
            target_rms_frac=TARGET_STEADY_RMS_FRACTION,
            max_tuning_rounds=MAX_TUNING_ROUNDS,
            on_event=None,
            should_stop=None,
            clarification_record=[],
            sim_overrides={"t_end": SIMULATION_TIME, "dt": SOLVER_STEP},
            clarifier_usage=None,
            tuning_objectives=TUNING_OBJECTIVES,
            system_spec=spec,
            **agent_options,
        )
        messages = result.get("messages") or []
        record["tuning"] = {
            "best_params": (best or {}).get("tuning"),
            "best_metrics": (best or {}).get("metrics"),
            "best_round": (best or {}).get("round"),
            "history": history,
            "agent_messages": messages,
            "mock_agent_messages": result.get("mock_agent_messages", []),
            "token_usage": usage,
        }
        record["verification"] = result.get("final_metrics")
        record["summary"] = getattr(messages[-1], "content", "") if messages else ""
        record["abstract"] = result.get("abstract")
        record["pipeline_completed"] = (
            isinstance(result.get("final_metrics"), dict) and bool(result.get("series"))
        )
        if SAVE_TIME_SERIES:
            record["series"] = result.get("series")
        if not record["pipeline_completed"]:
            record["error"] = {"type": "PipelineIncomplete", "message": record["summary"]}
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    return json_safe(record)


def main() -> None:
    if not RUN_BENCHMARK:
        print("Benchmark disabled. Set RUN_BENCHMARK=True to run it.")
        return
    validate_settings()
    artifacts = selected_artifacts()  # Validate the selection before any API calls.
    models = selected_models()
    total_tests = sum(len(artifact["tests"]["combinations"]) for _, artifact in artifacts)
    print(f"Selected {len(artifacts)} artifacts, {total_tests} tests, {len(models)} agent mode(s).")

    for config in models:
        routing = role_environment(config)  # Check API key before touching its report.
        previous = {key: os.environ.get(key) for key in routing}
        output = RESULTS_DIR / config["folder_name"] / OUTPUT_FILENAME
        records = []
        try:
            os.environ.update(routing)
            output.parent.mkdir(parents=True, exist_ok=True)
            for artifact_index, (path, artifact) in enumerate(artifacts):
                for test_index, test in enumerate(artifact["tests"]["combinations"]):
                    print(f"[{len(records) + 1}/{total_tests}] {config['folder_name']} "
                          f"{artifact['system_name']} {test['test_id']}", flush=True)
                    seed = MOCK_RANDOM_SEED + 1000 * artifact_index + test_index
                    record = run_one(path, artifact, test, config, seed)
                    records.append(record)
                    output.write_text(
                        json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                    if record.get("error"):
                        print(f"  Failed: {record['error']['message']}", flush=True)
            completed = sum(record["pipeline_completed"] for record in records)
            print(f"Completed {completed}/{total_tests}. Full report: {output}")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    print("Next: run benchmarks/Adaptive/finalize_results.py to write brief reports.")


if __name__ == "__main__":
    main()
