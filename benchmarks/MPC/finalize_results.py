"""Update complexity and success scores in the existing results JSON.

Update total_complexity, success_rate_1 and success_rate_2 in place.
Success Rate 2 is the fraction of configured fixed targets passed, using
controlled-output MSE and the saved settling/overshoot/input-effort metrics.
Migrate legacy RMS task scales once.
No tuning, simulation,
or API calls occur here; no _full copies are created.
Score only the declared outputs (metadata.outputs), over the full simulation:
    task_scale_mse_i = mean_t((initial_output_i - reference_i(t))**2)
    success_rate = exp(-10 * mean_i(tracking_mse.full_i / task_scale_i))
Read only the saved tracking_mse.full and task_scale arrays, aligned by outputs.
task_scale_basis='mse' means use the denominator directly. Unmarked old reports
stored RMS scales; square them once and mark the converted JSON as 'mse'.
No fallback to raw metrics or artifact IC/target gaps for missing score data.
There is no plant/condition complexity calculation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from pathlib import Path
import sys
from typing import Any

MPC_DIR = Path(__file__).resolve().parent
REPO_ROOT = MPC_DIR.parents[1]
ARTIFACTS_DIR = MPC_DIR / "artifacts"
RESULTS_DIR = MPC_DIR / "json_reports" / "results"
for directory in (REPO_ROOT, REPO_ROOT / "packages" / "labcd_agents" / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

try:
    from .success_rate.success_rate_1 import calculate_success_rate_1, output_task_scales_mse
    from .success_rate.success_rate_2 import evaluate_fixed_targets, validate_fixed_targets
except ImportError:  # Direct script execution.
    from success_rate.success_rate_1 import calculate_success_rate_1, output_task_scales_mse
    from success_rate.success_rate_2 import evaluate_fixed_targets, validate_fixed_targets

# Preserve the old Python function API for callers, not a duplicate JSON field.
calculate_success_rate = calculate_success_rate_1

# ------------------------- FINALIZATION SETTINGS -------------------------
# True: ignore the model list and any CLI input paths; update only mock.
# False: update the listed folders, unless explicit CLI input paths are given.
USE_MOCK_RESULTS = False  # True ignores the model list and finalizes only mock.
MODEL_FOLDERS_TO_UPDATE = [
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-4o-mini",
    "gpt-4.1-nano",
    "groq-gpt-oss-120b",
]
INPUT_FILENAME = "batch_tuning_results.json"
SEEDS_TO_UPDATE: list[int] | None = None  # None: all saved seeds; [1]: only seed1.
# Legacy fallback; new runs use <artifact>_results.json.
# ------------------------------------------------------------------------


def result_directories(model_dir: Path, seeds: list[int] | None) -> list[Path]:
    """Prefer saved seed runs over older unseeded reports; never mix both."""
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
         and any(path.glob("*_results.json"))),
        key=lambda path: int(path.name[4:]),
    )
    return saved or [model_dir]


def select_result_paths(
    explicit_paths: list[Path], *, model_folders: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[Path]:
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
        if (
            not isinstance(folder, str) or not folder.strip()
            or folder in {".", ".."} or "/" in folder or "\\" in folder or ":" in folder
        ):
            raise ValueError("Model entries must be simple result-folder names.")
        model_dir = RESULTS_DIR / folder
        for directory in result_directories(model_dir, SEEDS_TO_UPDATE if seeds is None else seeds):
            plant_reports = sorted(path for path in directory.glob("*_results.json")
                                   if path.is_file() and path.name != INPUT_FILENAME)
            path = directory / INPUT_FILENAME
            if plant_reports:
                paths.extend(plant_reports)
            elif path.is_file():
                paths.append(path)  # Backward compatibility without duplicate runs.
            else:
                print(f"Skipping missing selected report: {path}")
    return paths


def report_run_seed(record: dict[str, Any], report: Path) -> int | None:
    """Read the run seed and reject a record filed under a different seed."""
    folder = report.parent.name
    folder_seed = (int(folder[4:]) if folder.startswith("seed")
                   and folder[4:].isascii() and folder[4:].isdigit() else None)
    saved = record.get("seeds")
    seed = saved.get("run") if isinstance(saved, dict) else None
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)
                             or not 0 <= seed < 2**32):
        raise ValueError(f"{report}: invalid saved run seed {seed!r}.")
    if folder_seed is not None and seed is not None and folder_seed != seed:
        raise ValueError(f"{report}: saved run seed {seed} disagrees with {folder}.")
    return folder_seed if folder_seed is not None else seed


def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> dict[str, dict[str, Any]]:
    paths = sorted(artifacts_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No artifacts found in {artifacts_dir}.")
    artifacts = {}
    for path in paths:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        validate_fixed_targets((artifact.get("simulation_parameters") or {}).get("fixed_targets"))
        combinations = (artifact.get("tests") or {}).get("combinations")
        if not isinstance(artifact, dict) or not isinstance(combinations, list) or not combinations:
            raise ValueError(f"{path.name} must contain a non-empty tests.combinations list.")
        test_ids = []
        for test in combinations:
            if not isinstance(test, dict):
                raise ValueError(f"{path.name}: every test combination must be an object.")
            test_id = test.get("test_id")
            if not isinstance(test_id, str) or not test_id.strip():
                raise ValueError(f"{path.name}: every test combination must have a test_id.")
            test_ids.append(test_id)
            value = test.get("total_complexity")
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10
            ):
                raise ValueError(
                    f"{path.name}, {test_id}: total_complexity must be null or an integer from 1 to 10."
                )
        if len(test_ids) != len(set(test_ids)):
            raise ValueError(f"{path.name}: test_id values must be unique.")
        states = (artifact.get("metadata") or {}).get("states")
        if (
            not isinstance(states, list) or not states
            or any(not isinstance(name, str) or not name.strip() for name in states)
            or len(states) != len(set(states))
        ):
            raise ValueError(f"{path.name}: metadata.states must contain unique, non-empty names.")
        n_states = len(states)
        outputs = (artifact.get("metadata") or {}).get("outputs")
        if (not isinstance(outputs, list) or not outputs
                or any(not isinstance(name, str) or name not in states for name in outputs)
                or len(outputs) != len(set(outputs))):
            raise ValueError(f"{path.name}: metadata.outputs must name unique declared states.")
        for test in combinations:
            try:
                if len(test.get("initial_condition", [])) != n_states:
                    raise ValueError("Initial condition must have n_states entries.")
                calculate_task_scale_mse(test.get("initial_condition"), test.get("target"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path.name}, {test['test_id']}: {exc}") from exc
        artifacts[path.name] = artifact
    return artifacts


def calculate_task_scale_mse(initial_condition: Any, target: Any) -> list[float]:
    """Return one squared starting error per state, not an all-state average."""
    if (
        not isinstance(initial_condition, list) or not isinstance(target, list)
        or not initial_condition or len(initial_condition) != len(target)
    ):
        raise ValueError("Initial condition and target must be equally sized, nonempty lists.")
    errors = []
    for initial, desired in zip(initial_condition, target):
        try:
            initial, desired = float(initial), float(desired)
        except (TypeError, ValueError) as exc:
            raise ValueError("Initial condition and target must contain finite numbers.") from exc
        error = initial - desired
        if not all(math.isfinite(value) for value in (initial, desired, error)):
            raise ValueError("Initial condition and target must contain finite numbers.")
        squared_error = error * error
        if not math.isfinite(squared_error):
            raise ValueError("Squared task scales must be finite.")
        errors.append(squared_error)
    return errors




def vectors_match(left: Any, right: Any) -> bool:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    return all(
        math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
        for a, b in zip(left, right)
    )


def find_matching_test(record: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    """Resolve a result to its artifact test, with support for legacy reports."""
    combinations = artifact["tests"]["combinations"]
    result_test_id = record.get("test_id")
    if result_test_id is not None:
        candidates = [test for test in combinations if test["test_id"] == result_test_id]
        if not candidates:
            raise ValueError(
                f"Unknown test_id {result_test_id!r} for artifact {record.get('artifact')!r}."
            )
    else:
        # Older reports did not store test_id, so resolve them by their conditions.
        candidates = combinations

    matches = [
        test for test in candidates
        if vectors_match(record.get("initial_condition"), test.get("initial_condition"))
        and vectors_match(record.get("target"), test.get("target"))
        and record.get("scenario_level") == test.get("scenario_level")
    ]
    if not matches:
        raise ValueError(
            f"Test conditions changed for {record.get('artifact')}; rerun its benchmark first."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous legacy result for {record.get('artifact')}; rerun it to record test_id."
        )
    return matches[0]


def finalize_record(
    record: dict[str, Any], artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate, migrate old scales, and calculate both scores without retuning."""
    artifact_name = record.get("artifact")
    if artifact_name not in artifacts:
        raise KeyError(f"No current artifact matches {artifact_name!r}.")
    artifact = artifacts[artifact_name]
    if (record.get("model") or {}).get("name") != artifact["system_name"]:
        raise ValueError(f"Dynamics-name mismatch for {artifact_name}.")
    test = find_matching_test(record, artifact)
    updated = copy.deepcopy(record)
    verification = updated.get("verification")
    if isinstance(verification, dict) and verification.get("task_scale_basis", "rms") == "rms":
        saved_outputs = verification.get("outputs")
        if isinstance(saved_outputs, list) and saved_outputs:
            scales = output_task_scales_mse(verification, saved_outputs)
            if scales is not None:
                verification["task_scale"] = scales
                verification["task_scale_basis"] = "mse"
    updated["total_complexity"] = test.get("total_complexity")
    outputs = artifact["metadata"]["outputs"]
    updated["success_rate_1"] = calculate_success_rate_1(verification, outputs)
    targets = (artifact.get("simulation_parameters") or {}).get("fixed_targets")
    fixed_result = evaluate_fixed_targets(verification, targets, outputs)
    updated["success_rate_2"] = fixed_result.pop("success_rate_2")
    updated["success_rate_2_details"] = fixed_result
    updated.pop("success_rate", None)  # Old single-score field is now Rate 1.
    return updated


def update_results(
    input_path: Path, artifacts: dict[str, dict[str, Any]],
) -> Path:
    """Update the existing JSON atomically; never create a separate report."""
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(f"{input_path} must contain a JSON array.")
    # Validate every record before writing anything. A mismatch leaves the file intact.
    updated = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError(f"{input_path}: every run must be a JSON object.")
        report_run_seed(record, input_path)
        updated.append(finalize_record(record, artifacts))
    content = json.dumps(updated, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=input_path.parent, suffix=".tmp", delete=False,
        ) as report:
            temporary = Path(report.name)
            report.write(content)
        temporary.replace(input_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    pending = sum(record["total_complexity"] is None for record in updated)
    if pending:
        print(f"{pending} manual complexity value(s) still null; fill them in artifacts and finalize again.")
    for field in ("success_rate_1", "success_rate_2"):
        unavailable = sum(record[field] is None for record in updated)
        if unavailable:
            print(f"{unavailable} run(s) lack required data/targets; {field} remains null.")
    return input_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Update MPC complexity and success scores in place.")
    parser.add_argument("results_json", nargs="*", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--models", nargs="+", help="Model result folders to update.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Seed values to update.")
    args = parser.parse_args()
    if args.results_json and (args.models is not None or args.seeds is not None) and not USE_MOCK_RESULTS:
        parser.error("Use explicit JSON paths OR --models/--seeds, not both.")
    paths = select_result_paths(args.results_json, model_folders=args.models, seeds=args.seeds)
    if not paths:
        raise FileNotFoundError(
            "No reports exist for the selected models. Check USE_MOCK_RESULTS and "
            "MODEL_FOLDERS_TO_UPDATE, or run mpc_benchmark.py for those models first."
        )
    artifacts = load_artifacts(args.artifacts_dir)
    for path in paths:
        output = update_results(path, artifacts)
        print(f"Updated complexity, Success Rate 1 and Success Rate 2: {output}")


if __name__ == "__main__":
    main()
