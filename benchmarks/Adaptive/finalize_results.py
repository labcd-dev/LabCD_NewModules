"""Finalize accepted-format Adaptive result JSON in place with two success rates.

No tuning, simulation, or LLM API calls happen here. No brief/full copies are made.
Like the MPC finalizer, this uses an exponential of full-window verification
MSE normalized by the initial task-scale MSE:
    success_rate_1 = exp(-10 * mean_i(full_tracking_MSE_i / task_scale_MSE_i))
Success Rate 2 is the fraction of configured fixed targets passed, using
controlled-output MSE, reached settling time, mean-square input effort and
true target-crossing overshoot from the saved final output trajectories.
Only declared controlled outputs are scored. The core's task_scale is RMS;
task_scale_MSE_i = task_scale_i**2, not the square of an averaged scale.
Original core metrics stay unchanged; verification adds outputs and
task_scale_mse. A zero scale scores 1 only with zero MSE; otherwise the score is 0.
The score is distinct from verification.success, which checks steady-state
error and numerical health.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
import sys
from typing import Any

ADAPTIVE_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTIVE_DIR.parents[1]
ARTIFACTS_DIR = ADAPTIVE_DIR / "artifacts"
RESULTS_DIR = ADAPTIVE_DIR / "json_reports" / "results"
for directory in (REPO_ROOT, REPO_ROOT / "packages" / "labcd_agents" / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

try:
    from .adaptive_benchmark import accepted_result_record, load_accepted_template
    from .success_rate.success_rate_1 import calculate_success_rate_1, output_task_scales_mse
    from .success_rate.success_rate_2 import evaluate_fixed_targets, validate_fixed_targets
except ImportError:  # Direct script execution.
    from adaptive_benchmark import accepted_result_record, load_accepted_template
    from success_rate.success_rate_1 import calculate_success_rate_1, output_task_scales_mse
    from success_rate.success_rate_2 import evaluate_fixed_targets, validate_fixed_targets

# Backward-compatible Python API only; result JSON uses two named scores.
calculate_success_rate = calculate_success_rate_1

# --------------------------- FINALIZATION SETTINGS -------------------------
USE_MOCK_RESULTS = False  # True ignores the model list and finalizes only mock.
MODEL_FOLDERS_TO_UPDATE = [
    "groq-gpt-oss-120b",
    "gpt-4o-mini",
    "gpt-4.1-nano",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]
INPUT_FILENAME = "batch_tuning_results.json"
SEEDS_TO_UPDATE: list[int] | None = None  # None: all saved seeds; [42]: seed42 only.
# ---------------------------------------------------------------------------


def result_directories(model_dir: Path, seeds: list[int] | None) -> list[Path]:
    """Prefer seeded result reports over unseeded history; ignore old briefs."""
    if seeds is not None:
        if (not isinstance(seeds, list) or not seeds
                or any(isinstance(seed, bool) or not isinstance(seed, int)
                       or not 0 <= seed < 2**32 for seed in seeds)
                or len(seeds) != len(set(seeds))):
            raise ValueError("Selected seeds must be unique integers in [0, 2**32).")
        return [model_dir / f"seed{seed}" for seed in seeds]
    saved = sorted(
        (path for path in model_dir.glob("seed*") if path.is_dir()
         and path.name[4:].isascii() and path.name[4:].isdigit()
         and any(report.is_file() for report in path.glob("*_results.json"))),
        key=lambda path: int(path.name[4:]),
    )
    return saved or [model_dir]


def report_run_seed(record: dict[str, Any], report: Path) -> int | None:
    """Use the saved/folder seed, rejecting mislabeled seeded results."""
    name = report.parent.name
    folder_seed = (int(name[4:]) if name.startswith("seed")
                   and name[4:].isascii() and name[4:].isdigit() else None)
    saved = record.get("seeds")
    seed = saved.get("run") if isinstance(saved, dict) else None
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)
                             or not 0 <= seed < 2**32):
        raise ValueError(f"{report}: invalid saved run seed {seed!r}.")
    if folder_seed is not None and seed is not None and seed != folder_seed:
        raise ValueError(f"{report}: saved run seed {seed} disagrees with {name}.")
    return folder_seed if folder_seed is not None else seed


def select_result_paths(explicit_paths: list[Path], *, model_folders: list[str] | None = None,
                        seeds: list[int] | None = None) -> list[Path]:
    """Select only mock in mock mode, otherwise CLI inputs or configured models."""
    if USE_MOCK_RESULTS:
        if explicit_paths:
            print("Mock mode: ignoring explicit input paths and updating only mock.")
        folders = ["mock"]
    elif explicit_paths:
        return explicit_paths
    else:
        folders = MODEL_FOLDERS_TO_UPDATE if model_folders is None else model_folders
    paths = []
    for folder in dict.fromkeys(folders):
        if (not isinstance(folder, str) or not folder or folder in (".", "..")
                or any(char in folder for char in '/\\:<>"|?*')):
            raise ValueError(f"Unsafe model folder name: {folder!r}")
        model_dir = RESULTS_DIR / folder
        for directory in result_directories(model_dir, SEEDS_TO_UPDATE if seeds is None else seeds):
            plant_reports = sorted(path for path in directory.glob("*_results.json")
                                   if path.is_file() and path.name != INPUT_FILENAME)
            path = directory / INPUT_FILENAME
            if plant_reports:
                paths.extend(plant_reports)
            elif path.is_file():
                paths.append(path)
            else:
                print(f"Skipping missing report: {path}")
    return paths




def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> dict[str, dict[str, Any]]:
    artifacts = {}
    for path in sorted(artifacts_dir.glob("*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        validate_fixed_targets((artifact.get("simulation_parameters") or {}).get("fixed_targets"))
        tests = (artifact.get("tests") or {}).get("combinations") or []
        if not tests:
            raise ValueError(f"{path.name}: tests.combinations cannot be empty.")
        for test in tests:
            complexity = test.get("total_complexity")
            if (isinstance(complexity, bool) or not isinstance(complexity, int)
                    or not 1 <= complexity <= 10):
                raise ValueError(
                    f"{path.name}/{test.get('test_id')}: total_complexity must be 1..10."
                )
        artifacts[path.name] = artifact
    if not artifacts:
        raise FileNotFoundError(f"No Adaptive artifacts in {artifacts_dir}")
    return artifacts




def _same_test(saved: dict[str, Any], current: dict[str, Any]) -> bool:
    """Prevent attaching a current complexity to results from an edited test."""
    if saved.get("test_id") != current.get("test_id"):
        return False
    for key in ("initial_condition", "targets", "uncertainty", "disturbance"):
        if saved.get(key) != current.get(key):
            return False
    return True


def add_estimated_cost(record: dict[str, Any]) -> None:
    """Price real-agent token totals using the repository's local price table."""
    tuning = record.get("tuning")
    counts = tuning.get("token_usage") if isinstance(tuning, dict) else None
    if not isinstance(counts, dict):
        return
    if counts.get("simulated") or counts.get("cost_usd") is not None:
        return
    models = counts.get("models") or {}
    if not isinstance(models, dict):
        return
    from labcd_agents.pricing import CostCalculator

    calculator = CostCalculator()
    total_cost = 0.0
    unpriced = []
    any_tokens = False
    for role, field in (("agent", "agent"), ("tuner", "tuner"),
                        ("reporter", "reporter"), ("clarifier", "clarifier")):
        usage = counts.get(field) or {}
        if not isinstance(usage, dict):
            continue
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative.")
        if not input_tokens and not output_tokens:
            continue
        any_tokens = True
        model = models.get(role)
        if not isinstance(model, str) or calculator.resolve_price(model) is None:
            unpriced.append(model or f"unknown:{role}")
        else:
            total_cost += calculator.compute_cost(model, input_tokens, output_tokens)
    if any_tokens:
        # Do not report a partial amount as if it were the total cost.
        counts["cost_usd"] = None if unpriced else round(total_cost, 10)
        counts["unpriced_models"] = sorted(set(unpriced))
        counts["cost_source"] = "local_price_table_estimate"


def finalize_record(
    record: dict[str, Any], artifacts: dict[str, dict[str, Any]],
    report_path: Path | None = None,
) -> dict[str, Any]:
    artifact_name = record.get("artifact")
    if artifact_name not in artifacts:
        raise KeyError(f"No current Adaptive artifact matches {artifact_name!r}.")
    artifact = artifacts[artifact_name]
    if record.get("plant") != artifact.get("system_name"):
        raise ValueError(f"Plant name changed for {artifact_name}; rerun the benchmark.")
    tests = (artifact.get("tests") or {}).get("combinations") or []
    matches = [test for test in tests if test.get("test_id") == record.get("test_id")]
    if len(matches) != 1:
        raise ValueError(f"No unique current test for {artifact_name} {record.get('test_id')!r}.")
    saved_test = record.get("test")
    if saved_test is not None and not _same_test(saved_test, matches[0]):
        raise ValueError(f"Test conditions changed for {artifact_name}; rerun the benchmark.")

    updated = copy.deepcopy(record)
    updated["total_complexity"] = matches[0]["total_complexity"]
    applied = updated.get("applied_system_spec") or {}
    dynamics = applied.get("dynamics") or {}
    outputs = dynamics.get("outputs") or artifact["metadata"]["outputs"]
    verification = updated.get("verification")
    if isinstance(verification, dict):
        verification.setdefault("outputs", list(outputs))
        scales = output_task_scales_mse(verification)
        if scales is not None:
            verification["task_scale_mse"] = scales
    updated["success_rate_1"] = calculate_success_rate_1(verification, outputs)
    targets = (artifact.get("simulation_parameters") or {}).get("fixed_targets")
    fixed_result = evaluate_fixed_targets(verification, targets, outputs,
                                         record=updated, report_path=report_path)
    updated["success_rate_2"] = fixed_result.pop("success_rate_2")
    updated["success_rate_2_details"] = fixed_result
    updated.pop("success_rate", None)
    add_estimated_cost(updated)
    return updated


def update_results(
    input_path: Path, artifacts: dict[str, dict[str, Any]],
) -> Path:
    """Validate all records and atomically replace this JSON, preserving archives."""
    if input_path.name.endswith(("_brief.json", "_full.json")):
        raise ValueError("Select the canonical <artifact>_results.json, not a brief/full copy.")
    saved = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(saved, list):
        raise TypeError(f"{input_path} must contain an Adaptive benchmark JSON array.")
    structure = load_accepted_template()
    updated = []
    for record in saved:
        if not isinstance(record, dict):
            raise TypeError(f"{input_path}: every run must be a JSON object.")
        report_run_seed(record, input_path)
        finalized = finalize_record(record, artifacts, input_path)
        updated.append(accepted_result_record(finalized, structure))
    # Descriptive templates never supply missing values; they select saved fields.
    # Ensure a stale template cannot silently drop the newly computed scores.
    if any(any(field not in record for field in ("success_rate_1", "success_rate_2",
                                                 "success_rate_2_details")) for record in updated):
        raise ValueError("Accepted template must include both success scores and their check details.")
    content = json.dumps(updated, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=input_path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(input_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return input_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize Adaptive benchmark reports.")
    parser.add_argument("results_json", nargs="*", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--models", nargs="+", help="Model result folders to finalize.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Seed values to finalize.")
    args = parser.parse_args()
    if args.results_json and (args.models is not None or args.seeds is not None) and not USE_MOCK_RESULTS:
        parser.error("Use explicit JSON paths OR --models/--seeds, not both.")
    paths = select_result_paths(args.results_json, model_folders=args.models, seeds=args.seeds)
    if not paths:
        raise FileNotFoundError(
            "No selected full benchmark report exists. Run adaptive_benchmark.py first "
            "or change USE_MOCK_RESULTS/MODEL_FOLDERS_TO_UPDATE."
        )
    artifacts = load_artifacts(args.artifacts_dir)
    for path in paths:
        output = update_results(path, artifacts)
        print(f"Updated complexity and both success rates in place: {output}")


if __name__ == "__main__":
    main()
