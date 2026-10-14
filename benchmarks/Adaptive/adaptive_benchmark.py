"""Run Adaptive Designer/Tuner/Reporter on selected artifact tests.

Edit USER SETTINGS, then run this file. Mock mode uses only the agents under
``benchmarks/Adaptive/mock`` and never makes LLM API calls. Real mode uses the
normal agents under ``backend_core/AgentAdaptive``.
Each plant gets ``<artifact>_results.json`` under json_reports/results/<model>/seed<value>/.
Saved JSON fields follow json_reports/accepted_results_template.json; the
removed histories and core_outputs are not saved. Final series arrays live in
linked trajectories/*.npz files. Rerunning a plant replaces only that plant's
report. Run finalize_results.py to score that same JSON in place, then plot.
Importing this module does not start a benchmark or create files.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any

import numpy as np

ADAPTIVE_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTIVE_DIR.parents[1]
ARTIFACTS_DIR = ADAPTIVE_DIR / "artifacts"
RESULTS_DIR = ADAPTIVE_DIR / "json_reports" / "results"
ACCEPTED_TEMPLATE_PATH = ADAPTIVE_DIR / "json_reports" / "accepted_results_template.json"
PROGRESS_FILENAME = "benchmark_progress.json"
for directory in (REPO_ROOT, REPO_ROOT / "packages" / "labcd_agents" / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

# ------------------------------ USER SETTINGS ------------------------------
RUN_BENCHMARK = True
USE_MOCK_AGENTS = False  # True ignores LLM_CONFIGS and makes no API calls.
# ARTIFACT_FILES: list[str] | None = ["InvPendulum.json"]  # None runs every artifact.

ARTIFACT_FILES: list[str] | None = ["CartPole.json"]  # None runs every artifact.
# ARTIFACT_FILES: list[str] | None = None  # None runs every artifact.

# Mock mode ignores this list entirely. Every enabled real model runs every
# selected artifact/test; start with one model and one artifact to limit cost.
LLM_CONFIGS = [
    # {"folder_name": "gpt-5.4-mini", "provider": "openai",
    #  "model": "gpt-5.4-mini", "api_key_env_var": "OPENAI_API_KEY"},
    # {"folder_name": "gpt-5.4-nano", "provider": "openai",
    #  "model": "gpt-5.4-nano", "api_key_env_var": "OPENAI_API_KEY"},
    # {"folder_name": "gpt-4o-mini", "provider": "openai",
    #  "model": "gpt-4o-mini", "api_key_env_var": "OPENAI_API_KEY"},
    {"folder_name": "gpt-4.1-nano", "provider": "openai",
     "model": "gpt-4.1-nano", "api_key_env_var": "OPENAI_API_KEY"},
    # {"folder_name": "groq-gpt-oss-120b", "provider": "groq",
    #  "model": "openai/gpt-oss-120b", "api_key_env_var": "GROQ_API_KEY"},
]

ENABLE_TUNING = True  # False runs only the initial design and verification.
# Simulation time, dt, target fraction, tuning rounds, objectives, and fixed
# targets are read from each artifact's simulation_parameters block.
# Repeat every selected LLM/artifact/test with a fresh workflow for each seed.
# Real LLM sampling and mock agents use this seed. The core's RBF seed stays 0.
RUN_SEEDS: list[int] = [42, 123, 2024]
ROLE_MAX_TOKENS = {"design": 1500, "tuner": 4000,
                   "clarifier": 8000, "reporter": 500}
# ---------------------------------------------------------------------------


def _safe_folder(name: str) -> bool:
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    return (isinstance(name, str) and bool(name) and name not in (".", "..")
            and not any(char in '<>:"/\\|?*' for char in name)
            and not name.endswith((".", " ")) and name.upper().split(".")[0] not in reserved)


def validate_settings() -> None:
    if (not isinstance(RUN_SEEDS, list) or not RUN_SEEDS
            or any(isinstance(seed, bool) or not isinstance(seed, int)
                   or not 0 <= seed < 2**32 for seed in RUN_SEEDS)
            or len(RUN_SEEDS) != len(set(RUN_SEEDS))):
        raise ValueError("RUN_SEEDS must contain unique integers in [0, 2**32).")


def result_directory(config: dict[str, Any]) -> Path:
    """Seeded layout for new runs; preserve compatibility for older callers."""
    if not _safe_folder(config.get("folder_name")):
        raise ValueError("Model folder must be a safe name.")
    directory = RESULTS_DIR / config["folder_name"]
    if "run_seed" in config:
        seed = config["run_seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("run_seed must be an integer in [0, 2**32).")
        directory /= f"seed{seed}"
    return directory


@contextmanager
def seeded_llm_routing(seed: int):
    """Inject sampling seeds only while this sequential benchmark run executes.

    This is runtime routing, not a core-file edit. Preserve role temperatures,
    token limits and normal real-agent behavior; restore the factory even on error.
    """
    if USE_MOCK_AGENTS:
        yield
        return
    from backend_core.AgentAdaptive.agents import llm_factory
    original = llm_factory.build_llm

    def seeded_factory(*args, **kwargs):
        kwargs["seed"] = seed
        return original(*args, **kwargs)

    llm_factory.build_llm = seeded_factory
    try:
        yield
    finally:
        llm_factory.build_llm = original


@contextmanager
def full_series_export(settings: dict[str, Any]):
    """Keep final samples for offline overshoot scoring, using the core's knob."""
    key = "LABCD_ADAPTIVE_SERIES_MAX_POINTS"
    previous = os.environ.get(key)
    needed = max(2000, math.ceil(float(settings["simulation_time"]) / float(settings["dt"])) + 2)
    os.environ[key] = str(needed)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


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
        metadata = artifact.get("metadata") or {}
        system_type = str(metadata.get("system_type", "")).upper()
        control_input_affine = metadata.get("control_input_affine")
        inputs = metadata.get("inputs") or []
        outputs = metadata.get("outputs") or []
        if not isinstance(control_input_affine, bool):
            raise ValueError(
                f"{name}: metadata.control_input_affine must be true or false."
            )
        if not control_input_affine:
            print(
                f"Skipping {name}: the control input is non-affine "
                "and is unsupported by the current Adaptive controllers."
            )
            continue
        if system_type == "MIMO" and len(inputs) != len(outputs):
            input_word = "input" if len(inputs) == 1 else "inputs"
            output_word = "output" if len(outputs) == 1 else "outputs"
            print(
                f"Skipping {name}: non-square MIMO system "
                f"({len(inputs)} {input_word}, {len(outputs)} {output_word})."
            )
            continue
        compiler_artifact = copy.deepcopy(artifact)
        compiler_artifact["metadata"].setdefault("assumptions", [])
        validation = compiler.validate(compiler_artifact)
        if not validation.ok:
            raise ValueError(f"{name}: invalid Plant Agent artifact: {validation.errors}")
        tests = (artifact.get("tests") or {}).get("combinations")
        if not isinstance(tests, list) or not tests:
            raise ValueError(f"{name}: tests.combinations must contain at least one test.")
        ids = [test.get("test_id") for test in tests]
        if any(not isinstance(test_id, str) or not test_id for test_id in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{name}: test_id values must be nonempty and unique.")
        for test in tests:
            complexity = test.get("total_complexity")
            if (isinstance(complexity, bool) or not isinstance(complexity, int)
                    or not 1 <= complexity <= 10):
                raise ValueError(
                    f"{name}/{test.get('test_id')}: total_complexity must be 1..10."
                )
        artifact_runtime_settings(artifact)
        selected.append((path, artifact))
    return selected


def _validate_runtime_values(
    simulation_time: Any, solver_step: Any, target_fraction: Any,
    tuning_rounds: Any, objectives: Any, *, label: str,
) -> None:
    if (isinstance(simulation_time, bool) or not isinstance(simulation_time, (int, float))
            or not math.isfinite(float(simulation_time)) or float(simulation_time) <= 0.0):
        raise ValueError(f"{label}: simulation_time must be finite and positive.")
    if (isinstance(solver_step, bool) or not isinstance(solver_step, (int, float))
            or not math.isfinite(float(solver_step)) or float(solver_step) <= 0.0
            or float(solver_step) >= float(simulation_time)):
        raise ValueError(f"{label}: dt must be positive and smaller than simulation_time.")
    if (isinstance(target_fraction, bool) or not isinstance(target_fraction, (int, float))
            or not math.isfinite(float(target_fraction))
            or not 0.0 < float(target_fraction) <= 1.0):
        raise ValueError(f"{label}: target_steady_rms_fraction must be in (0, 1].")
    if (isinstance(tuning_rounds, bool) or not isinstance(tuning_rounds, int)
            or not 1 <= tuning_rounds <= 10):
        raise ValueError(f"{label}: tuning_rounds must be an integer from 1 to 10.")
    if objectives is not None:
        from backend_core.AgentAdaptive.tools.tuning_objectives import OBJECTIVES
        if not isinstance(objectives, dict) or any(
            key not in OBJECTIVES or isinstance(weight, bool)
            or not isinstance(weight, int) or not 1 <= weight <= 5
            for key, weight in objectives.items()
        ):
            raise ValueError(
                f"{label}: tuning_objectives needs supported objectives with weights 1..5."
            )


def artifact_runtime_settings(artifact: dict[str, Any]) -> dict[str, Any]:
    """Read the complete runtime/tuning configuration from one artifact."""
    raw = artifact.get("simulation_parameters") or {}
    required = {
        "simulation_time", "dt", "target_steady_rms_fraction",
        "tuning_rounds", "tuning_objectives", "fixed_targets",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(
            f"{artifact.get('system_name') or 'artifact'}: simulation_parameters "
            f"is missing {missing}."
        )
    settings = {
        "simulation_time": raw["simulation_time"],
        "dt": raw["dt"],
        "target_fraction": raw["target_steady_rms_fraction"],
        "tuning_rounds": raw["tuning_rounds"],
        "objectives": copy.deepcopy(raw["tuning_objectives"]),
        "fixed_targets": copy.deepcopy(raw["fixed_targets"]),
    }
    _validate_runtime_values(
        settings["simulation_time"], settings["dt"], settings["target_fraction"],
        settings["tuning_rounds"], settings["objectives"],
        label=str(artifact.get("system_name") or "artifact"),
    )
    from benchmarks.Adaptive.success_rate.success_rate_2 import validate_fixed_targets
    validate_fixed_targets(settings["fixed_targets"])
    return settings


def build_system_spec(
    artifact: dict[str, Any], test: dict[str, Any], settings: dict[str, Any],
) -> dict[str, Any]:
    """Pass one artifact directly through the normal Adaptive core path."""
    from backend_core.plant_compiler import PlantCompiler
    from backend_core.AgentAdaptive.tools import system_spec as spec_tools
    from backend_core.AgentAdaptive.controller.structure_build import _build_structure_from_spec

    metadata = artifact["metadata"]
    states = metadata["states"]
    outputs = metadata["outputs"]
    initial = np.asarray(test["initial_condition"], dtype=float)
    if initial.shape != (len(states),) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_condition must have one finite value per state.")
    targets = test.get("targets")
    if not isinstance(targets, list) or len(targets) != len(outputs):
        raise ValueError("Provide exactly one target per declared output.")
    ref_map = {item.get("output"): item.get("expr") for item in targets if isinstance(item, dict)}
    if (len(ref_map) != len(outputs) or set(ref_map) != set(outputs)
            or any(not isinstance(expr, str) or not expr.strip() for expr in ref_map.values())):
        raise ValueError("Targets must name each output once and have nonempty expressions.")

    compiler_artifact = copy.deepcopy(artifact)
    compiler_artifact["metadata"].setdefault("assumptions", [])
    validation = PlantCompiler().validate(compiler_artifact)
    if not validation.ok:
        raise ValueError(f"Invalid Adaptive artifact: {validation.errors}")

    spec = PlantCompiler().generate_adaptive_spec(
        plant_output=compiler_artifact,
        pre_launch={"initial_state": initial.tolist(),
                    "total_simulation_time": float(settings["simulation_time"]),
                    "solver_sample_time": float(settings["dt"])},
    )
    dynamics = spec["dynamics"]
    dynamics["x0"] = initial.tolist()
    dynamics["references"] = [{"output": name, "expr": ref_map[name]} for name in outputs]
    dynamics["uncertainty"] = copy.deepcopy(test.get("uncertainty", []))
    dynamics["disturbance"] = copy.deepcopy(test.get("disturbance", []))
    dynamics["sim_time"] = float(settings["simulation_time"])
    dynamics["solver_step"] = float(settings["dt"])
    gaps = spec_tools.missing_items(spec)
    if gaps:
        raise ValueError(f"Incomplete Adaptive system specification: {gaps}")
    _, limitations = _build_structure_from_spec(spec_tools.substitute_parameters(spec))
    if limitations:
        raise ValueError(f"Unsupported Adaptive structure/reference: {limitations}")
    return spec


def json_safe(value: Any) -> Any:
    """Keep complete returned data, including LLM message metadata, JSON-safe."""
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
    # LangChain messages include tool calls, token/response metadata, IDs, etc.
    # Saving only .content would discard part of the core's returned output.
    if callable(getattr(value, "model_dump", None)):
        return json_safe(value.model_dump(mode="python"))
    if hasattr(value, "content"):
        fields = dict(vars(value)) if hasattr(value, "__dict__") else {}
        fields.setdefault("type", type(value).__name__)
        fields.setdefault("content", value.content)
        return json_safe(fields)
    # Designer progress events can contain symbolic expressions/matrices.
    import sympy as sp
    if isinstance(value, sp.MatrixBase):
        return json_safe(value.tolist())
    if isinstance(value, sp.Basic):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} in a benchmark result.")


class _CapturedStream(io.TextIOBase):
    """Save console output while keeping the user's normal live output."""

    def __init__(self, original, captured: io.StringIO):
        self.original = original
        self.captured = captured

    def write(self, text: str) -> int:
        self.captured.write(text)
        return self.original.write(text)

    def flush(self) -> None:
        self.original.flush()


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
    config: dict[str, Any], seed: int,
) -> dict[str, Any]:
    settings = artifact_runtime_settings(artifact)
    record = {
        "artifact": artifact_path.name,
        "plant": artifact["system_name"],
        "test_id": test["test_id"],
        "test": copy.deepcopy(test),
        "agent_mode": "mock" if USE_MOCK_AGENTS else config["provider"],
        "llm_model": None if USE_MOCK_AGENTS else config["model"],
        "total_complexity": test["total_complexity"],
        "success_rate_1": None,
        "success_rate_2": None,
        "success_rate_2_details": None,
        "mock_seed": seed if USE_MOCK_AGENTS else None,
        "seeds": {
            "run": seed, "llm": None if USE_MOCK_AGENTS else seed,
            "mock": seed if USE_MOCK_AGENTS else None,
            "estimator": 0, "simulation": None,
        },
        "tuning_inputs": {
            "enabled": ENABLE_TUNING,
            "target_error_fraction": settings["target_fraction"],
            "max_rounds": settings["tuning_rounds"],
            "objectives": copy.deepcopy(settings["objectives"]),
            "fixed_targets": copy.deepcopy(settings["fixed_targets"]),
            "simulation_time": settings["simulation_time"],
            "dt": settings["dt"],
        },
        "applied_system_spec": None,
        "tuning": None,
        "verification": None,
        "series": None,
        "core_outputs": {
            "result": None,
            "usage": None,
            "tuning_history": None,
            "best": None,
            "events": [],
            "console_output": {"stdout": "", "stderr": ""},
        },
        "pipeline_completed": False,
    }
    stdout_log, stderr_log = io.StringIO(), io.StringIO()

    def on_event(event: dict[str, Any]) -> None:
        # Freeze each event at emission: later mutations must not change it.
        record["core_outputs"]["events"].append(json_safe(event))

    try:
        spec = build_system_spec(artifact, test, settings)
        record["applied_system_spec"] = spec
        if USE_MOCK_AGENTS:
            from benchmarks.Adaptive.mock.tuner_agent import run_full_pipeline
            agent_options = {"mock_seed": seed}
        else:
            from backend_core.AgentAdaptive.agents.tuner_agent import run_full_pipeline
            agent_options = {}

        from benchmarks.Adaptive.fast_report import fast_two_link_report

        with (redirect_stdout(_CapturedStream(sys.stdout, stdout_log)),
              redirect_stderr(_CapturedStream(sys.stderr, stderr_log)),
              fast_two_link_report(artifact_path.stem.startswith("TwoLinkManip")),
              seeded_llm_routing(seed), full_series_export(settings)):
            result, usage, history, best = run_full_pipeline(
                description="",
                enable_tuning=ENABLE_TUNING,
                target_rms_frac=float(settings["target_fraction"]),
                max_tuning_rounds=int(settings["tuning_rounds"]),
                on_event=on_event,
                should_stop=None,
                clarification_record=[],
                sim_overrides={
                    "t_end": float(settings["simulation_time"]),
                    "dt": float(settings["dt"]),
                },
                clarifier_usage=None,
                tuning_objectives=settings["objectives"],
                system_spec=spec,
                **agent_options,
            )
        record["core_outputs"].update({
            "result": result,
            "usage": usage,
            "tuning_history": history,
            "best": best,
        })
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
        record["series"] = result.get("series")
        if not record["pipeline_completed"]:
            record["error"] = {"type": "PipelineIncomplete", "message": record["summary"]}
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
    finally:
        record["core_outputs"]["console_output"] = {
            "stdout": stdout_log.getvalue(), "stderr": stderr_log.getvalue(),
        }
    return json_safe(record)


def save_verification_trajectories(
    artifact_path: Path, test_id: str, config: dict[str, str], series: dict[str, Any],
) -> dict[str, Any]:
    """Move exported numeric channels into an NPZ; return metadata and its link.

    Preserve every sample returned by the core, including any nonfinite values.
    The core may already have downsampled its export; no further sampling occurs.
    """
    filename = f"{artifact_path.stem}_{test_id}_verification.npz"
    folder = config["folder_name"]
    if not _safe_folder(test_id) or not _safe_folder(filename) or not _safe_folder(folder):
        raise ValueError("Trajectory filename, test ID and model folder must be safe names.")
    channel_keys = {
        "t": "times", "x": "states", "y": "outputs", "ref": "reference", "u": "inputs",
    }
    channels = series.get("channels")
    if not isinstance(channels, dict) or not all(
        isinstance(channels.get(key), dict) and "data" in channels[key]
        for key in channel_keys
    ):
        raise ValueError("Verification series must contain t, x, y, ref and u data.")
    arrays = {}
    metadata = {key: copy.deepcopy(value) for key, value in series.items() if key != "channels"}
    metadata["channels"] = {}
    for channel, array_key in channel_keys.items():
        details = channels[channel]
        array = np.asarray(details["data"])
        # The core exports missing/nonfinite measurements as None. Preserve these
        # as NaN in the numeric archive, without enabling pickled object arrays.
        if array.dtype.kind == "O":
            if not all(value is None or isinstance(value, (int, float, np.number))
                       for value in array.flat):
                raise ValueError(f"Verification channel {channel!r} must be numeric.")
            array = array.astype(float)
        if array.dtype.kind not in "biuf":
            raise ValueError(f"Verification channel {channel!r} must be numeric.")
        if array.ndim != (1 if channel == "t" else 2):
            raise ValueError(f"Verification channel {channel!r} has an invalid array shape.")
        names = details.get("names")
        if channel != "t" and isinstance(names, list) and len(names) != array.shape[1]:
            raise ValueError(f"Verification channel {channel!r} columns do not match its names.")
        arrays[array_key] = array
        metadata["channels"][channel] = {
            key: copy.deepcopy(value) for key, value in details.items() if key != "data"
        }
        metadata["channels"][channel]["array_key"] = array_key
    count = len(arrays["times"])
    if any(len(array) != count for array in arrays.values()):
        raise ValueError("Verification channels must align with the exported timestamps.")
    if "n_points" in series and series["n_points"] != count:
        raise ValueError("Verification n_points does not match the exported timestamps.")
    # Do not silently drop unknown future channels that might carry extra data.
    if set(channels) != set(channel_keys):
        raise ValueError("Verification series has unsupported extra channels.")
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
    metadata["trajectory_file"] = relative_path.as_posix()
    return json_safe(metadata)


def load_accepted_template() -> dict[str, Any]:
    """Read the accepted field structure; descriptions are never result values."""
    template = json.loads(ACCEPTED_TEMPLATE_PATH.read_text(encoding="utf-8"))
    if (not isinstance(template, list) or len(template) != 1
            or not isinstance(template[0], dict)
            or not isinstance(template[0].get("tuning"), dict)
            or not isinstance(template[0].get("series"), dict)):
        raise ValueError("Accepted results template must contain one record with tuning and series objects.")
    return template[0]


def accepted_result_record(value: Any, structure: Any) -> Any:
    """Keep only accepted fields, preserving actual values and optional branches.

    A <placeholder> key permits plant/provider-specific keys. Array examples
    describe repeated elements, not limits on their count. Absent fields stay
    absent; None stays None. Template descriptions are documentation only.
    """
    if isinstance(value, dict) and isinstance(structure, dict):
        wildcard = next((key for key in structure
                         if key.startswith("<") and key.endswith(">")), None)
        saved = {}
        for key, item in value.items():
            if key == "_description":
                continue
            if key in structure:
                saved[key] = accepted_result_record(item, structure[key])
            elif wildcard is not None:
                saved[key] = accepted_result_record(item, structure[wildcard])
        return saved
    if isinstance(value, list) and isinstance(structure, list):
        if not structure:
            return copy.deepcopy(value)
        return [accepted_result_record(item, structure[min(index, len(structure) - 1)])
                for index, item in enumerate(value)]
    return copy.deepcopy(value)


def save_artifact_results(
    artifact_path: Path, config: dict[str, str], records: list[dict[str, Any]],
) -> Path:
    """Checkpoint accepted records and external trajectories for one plant."""
    structure = load_accepted_template()
    filename = f"{artifact_path.stem}_results.json"
    if not _safe_folder(filename) or not _safe_folder(config["folder_name"]):
        raise ValueError("Artifact result filename and model folder must be safe names.")
    saved_records = copy.deepcopy(records)
    for record in saved_records:
        series = record.get("series")
        if series:
            metadata = save_verification_trajectories(
                artifact_path, record["test_id"], config, series,
            )
            record["series"] = metadata
    accepted_records = [accepted_result_record(json_safe(record), structure) for record in saved_records]
    payload = json.dumps(accepted_records, indent=2, ensure_ascii=False,
                         allow_nan=False) + "\n"
    output = result_directory(config) / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                         dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


def write_progress(plan: list[dict[str, Any]], status: str,
                   last_error: dict[str, Any] | None = None) -> Path:
    """Checkpoint selected combinations; completion is not controller success."""
    done = [item for item in plan if item["status"] == "completed"]
    remaining = [item for item in plan if item["status"] != "completed"]
    failed = [item for item in plan if item["status"] == "failed"]
    running = [item for item in plan if item["status"] == "running"]
    total = len(plan)
    snapshot = {
        "scope": "Current selected models, RUN_SEEDS, supported artifacts and tests.",
        "completion_definition": "Pipeline finished and JSON/trajectories were saved; not controller success.",
        "status": status,
        "summary": {
            "done": len(done), "total": total, "done_out_of_total": f"{len(done)}/{total}",
            "progress_percent": round(100 * len(done) / total, 2) if total else 0.0,
            "remaining": len(remaining), "failed": len(failed), "running": len(running),
            "pending": sum(item["status"] == "pending" for item in plan),
        },
        "current_combination": running[0] if running else None,
        "completed_combinations": done, "remaining_combinations": remaining,
        "failed_combinations": failed, "last_error": last_error,
    }
    output = RESULTS_DIR / PROGRESS_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                         dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(snapshot, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


def completion_status(record: dict[str, Any]) -> tuple[str, dict | None]:
    error = record.get("error")
    if error:
        return "failed", error if isinstance(error, dict) else {"message": str(error)}
    if (record.get("pipeline_completed") is True
            and isinstance(record.get("verification"), dict)):
        # ENABLE_TUNING=False legitimately has no tuning_best/best_params.
        # The initial-controller simulation is still a completed pipeline.
        return "completed", None
    return "failed", {"type": "IncompleteResult", "message": "Saved pipeline did not finish."}


def progress_key(folder: str, seed: int, artifact: str, test_id: str) -> tuple:
    return folder, seed, artifact, test_id


def build_progress_plan(artifacts, models) -> list[dict[str, Any]]:
    """Read only matching saved seeded runs; do not call LLMs or resume runs."""
    plan = []
    for config in models:
        for seed in RUN_SEEDS:
            for path, artifact in artifacts:
                report = result_directory({**config, "run_seed": seed}) / f"{path.stem}_results.json"
                saved, read_error = [], None
                if report.is_file():
                    try:
                        saved = json.loads(report.read_text(encoding="utf-8"))
                        if not isinstance(saved, list):
                            raise ValueError("Saved report must be a JSON array.")
                    except (OSError, ValueError) as error:
                        saved = []
                        read_error = {"type": type(error).__name__, "message": str(error)}
                for test in artifact["tests"]["combinations"]:
                    item = {
                        "model_folder": config["folder_name"], "llm_model": config["model"],
                        "seed": seed, "artifact": path.name, "dynamic_model": artifact["system_name"],
                        "test_id": test["test_id"],
                        "result_file": report.relative_to(RESULTS_DIR).as_posix(), "status": "pending",
                    }
                    matches = [
                        record for record in saved if isinstance(record, dict)
                        and record.get("artifact") == path.name
                        and record.get("plant") == artifact["system_name"]
                        and record.get("test_id") == test["test_id"]
                        and isinstance(record.get("seeds"), dict) and record["seeds"].get("run") == seed
                        and record.get("llm_model") == (None if USE_MOCK_AGENTS else config["model"])
                        and record.get("agent_mode") == ("mock" if USE_MOCK_AGENTS else config["provider"])
                        and isinstance(record.get("test"), dict)
                        and all(record["test"].get(key) == test.get(key)
                                for key in ("initial_condition", "targets", "uncertainty", "disturbance"))
                    ]
                    if matches:
                        item["status"], error = completion_status(matches[-1])
                        if error:
                            item["error"] = error
                    elif read_error:
                        item["saved_report_issue"] = read_error
                    plan.append(item)
    return plan


def main() -> None:
    if not RUN_BENCHMARK:
        print("Benchmark disabled. Set RUN_BENCHMARK=True to run it.")
        return
    validate_settings()
    load_accepted_template()  # Check the output structure before any API calls.
    artifacts = selected_artifacts()  # Validate the selection before any API calls.
    if not artifacts:
        print("No supported artifacts remain after skipping non-square MIMO systems.")
        return
    models = selected_models()
    plan = build_progress_plan(artifacts, models)
    by_key = {progress_key(item["model_folder"], item["seed"], item["artifact"], item["test_id"]): item
              for item in plan}
    progress_path = write_progress(plan, "running")
    total_tests = sum(len(artifact["tests"]["combinations"]) for _, artifact in artifacts)
    print(f"Selected {len(artifacts)} artifacts, {total_tests} tests, {len(models)} agent mode(s), "
          f"{len(RUN_SEEDS)} seeds; {len(plan)} tuning runs. Seeds: {RUN_SEEDS}")
    active = None
    try:
        for config in models:
            routing = role_environment(config)  # Check API key before touching its report.
            previous = {key: os.environ.get(key) for key in routing}
            try:
                os.environ.update(routing)
                for seed in RUN_SEEDS:
                    run_config = {**config, "run_seed": seed}
                    completed_tests = completed_pipelines = 0
                    for path, artifact in artifacts:
                        records = []
                        # Rerunning replaces this artifact checkpoint, not other plants/seeds.
                        for test in artifact["tests"]["combinations"]:
                            item = by_key[progress_key(config["folder_name"], seed, path.name, test["test_id"])]
                            item["status"] = "pending"
                            item.pop("error", None)
                            item.pop("saved_report_issue", None)
                        for test in artifact["tests"]["combinations"]:
                            active = by_key[progress_key(config["folder_name"], seed, path.name, test["test_id"])]
                            active["status"] = "running"
                            write_progress(plan, "running")
                            print(f"[{completed_tests + 1}/{total_tests}] {config['folder_name']}/seed{seed} "
                                  f"{artifact['system_name']} {test['test_id']}", flush=True)
                            record = run_one(path, artifact, test, run_config, seed)
                            records.append(record)
                            output = save_artifact_results(path, run_config, records)
                            active["status"], error = completion_status(record)
                            if error:
                                active["error"] = error
                            active = None
                            write_progress(plan, "running")
                            completed_tests += 1
                            completed_pipelines += int(bool(record["pipeline_completed"]))
                            if record.get("error"):
                                print(f"  Failed: {record['error']['message']}", flush=True)
                        print(f"Accepted artifact report: {output}")
                    print(f"Completed {completed_pipelines}/{total_tests} for {config['folder_name']}/seed{seed}.")
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
    except BaseException as error:
        if active is not None:
            active["status"] = "pending"
        write_progress(plan, "interrupted", {"type": type(error).__name__, "message": str(error)})
        raise
    write_progress(plan, "finished")
    done = sum(item["status"] == "completed" for item in plan)
    print(f"Progress: {done}/{len(plan)} done. Tracker: {progress_path}")
    print("Accepted results and linked trajectories are saved separately for each model/seed/artifact.")
    print("Next: finalize_results.py, plot_dashboard.py, and plot_trajectories.py.")


if __name__ == "__main__":
    main()
