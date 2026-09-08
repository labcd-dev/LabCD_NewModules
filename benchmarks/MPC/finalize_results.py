r"""Fill complexity and success rate in clean MPC benchmark results.

The benchmark deliberately writes ``total_complexity`` and ``success_rate`` as
null placeholders. This postprocessor reads the scored artifact tests, fills
the matching total complexity, calculates ``exp(-verification RMSE)`` for each
run, and writes a separate ``*_full.json`` file. The clean base JSON is never
modified.

Run from the repository root after ``make_tests.py``,
``score_complexity.py``, and ``mpc_becnchmark.py``:

    .venv\Scripts\python.exe benchmarks\MPC\finalize_results.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


MPC_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = MPC_DIR.parent
DEFAULT_ARTIFACTS_DIR = BENCHMARK_DIR / "artifacts"
DEFAULT_RESULTS_DIR = MPC_DIR / "json_reports" / "results"


def calculate_success_rate(verification_rmse: float) -> float:
    """Return the continuous per-run success rate exp(-RMSE)."""
    rmse = float(verification_rmse)
    if not math.isfinite(rmse) or rmse < 0.0:
        raise ValueError("Verification RMSE must be finite and nonnegative.")
    return round(math.exp(-rmse), 6)


def reprice_token_usage(record: dict[str, Any]) -> None:
    """Refresh stored LLM cost using the current shared price resolver."""
    tuning = record.get("tuning")
    token_usage = tuning.get("token_usage") if isinstance(tuning, dict) else None
    per_model = token_usage.get("per_model") if isinstance(token_usage, dict) else None
    if not isinstance(per_model, dict):
        return
    if token_usage.get("simulated"):
        return

    try:
        from labcd_agents.pricing import CostCalculator
    except ImportError:
        return

    calculator = CostCalculator()
    total_cost = 0.0
    unpriced_models: list[str] = []
    for model_name, usage in per_model.items():
        if model_name == "unknown" or not isinstance(usage, dict):
            continue
        if calculator.resolve_price(model_name) is None:
            unpriced_models.append(model_name)
            continue
        total_cost += calculator.compute_cost(
            model_name,
            int(usage.get("prompt", 0) or 0),
            int(usage.get("completion", 0) or 0),
        )

    token_usage["cost_usd"] = round(total_cost, 10)
    token_usage["unpriced_models"] = unpriced_models


def load_artifact_tests(artifacts_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Index scored artifact tests by (dynamics class name, test_id)."""
    artifact_paths = sorted(
        path
        for path in artifacts_dir.glob("*.json")
        if not path.name.startswith("_")
    )
    if not artifact_paths:
        raise FileNotFoundError(f"No artifact JSON files found in {artifacts_dir}.")

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for artifact_path in artifact_paths:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        system_name = str(artifact.get("system_name") or "")
        combinations = (artifact.get("tests") or {}).get("combinations") or []
        if not system_name or not combinations:
            raise ValueError(
                f"{artifact_path.name} has no system_name or test combinations. "
                "Run make_tests.py and score_complexity.py first."
            )

        for test in combinations:
            test_id = str(test.get("test_id") or "")
            total_complexity = test.get("total_cmplx")
            if not test_id or not isinstance(total_complexity, dict):
                raise ValueError(
                    f"{artifact_path.name} contains an unscored test. "
                    "Run score_complexity.py first."
                )
            key = (system_name, test_id)
            if key in index:
                raise ValueError(f"Duplicate artifact test key: {key}.")
            index[key] = test
    return index


def _record_condition(record: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Read IC, target and level from the current or legacy result schema."""
    scenario = record.get("scenario") or {}
    initial_condition = record.get("initial_condition")
    target = record.get("target")
    scenario_level = record.get("scenario_level")
    if initial_condition is None:
        initial_condition = scenario.get("initial_state")
    if target is None:
        target = scenario.get("target")
    if scenario_level is None:
        scenario_level = scenario.get("level")
    return initial_condition, target, scenario_level


def _vectors_match(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    if not isinstance(left, list) or not isinstance(right, list):
        return False
    if len(left) != len(right):
        return False
    try:
        return all(
            math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)
            for a, b in zip(left, right)
        )
    except (TypeError, ValueError):
        return False


def resolve_artifact_test(
    record: dict[str, Any],
    artifact_tests: dict[tuple[str, str], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Resolve a current test_id or infer one for an older result schema."""
    system_name = str((record.get("model") or {}).get("name") or "")
    test_id = str(record.get("test_id") or "")
    if test_id:
        key = (system_name, test_id)
        if key not in artifact_tests:
            raise KeyError(
                f"No artifact test matches system_name={system_name!r}, "
                f"test_id={test_id!r}."
            )
        return test_id, artifact_tests[key]

    initial_condition, target, scenario_level = _record_condition(record)
    candidates = [
        test
        for (candidate_system, _), test in artifact_tests.items()
        if candidate_system == system_name
        and int(test["scenario_level"]) == int(scenario_level)
        and _vectors_match(test.get("target"), target)
    ]

    # Prefer an exact declared IC. A legacy level-3 result may instead contain
    # the scenario-pushed simulated IC, so a unique target+level match is the
    # safe fallback for those old reports.
    exact_candidates = [
        test
        for test in candidates
        if _vectors_match(test.get("initial_condition"), initial_condition)
    ]
    if len(exact_candidates) == 1:
        matched = exact_candidates[0]
    elif len(candidates) == 1:
        matched = candidates[0]
    else:
        raise KeyError(
            f"Cannot uniquely infer a test_id for legacy result "
            f"system_name={system_name!r}, scenario_level={scenario_level!r}, "
            f"target={target!r}; candidates={len(candidates)}."
        )
    return str(matched["test_id"]), matched


def fill_result_record(
    record: dict[str, Any],
    artifact_tests: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Return one copied record with both placeholders filled when possible."""
    updated = copy.deepcopy(record)
    system_name = str((updated.get("model") or {}).get("name") or "")
    test_id, artifact_test = resolve_artifact_test(updated, artifact_tests)
    initial_condition, target, scenario_level = _record_condition(updated)
    if int(scenario_level) != int(artifact_test["scenario_level"]):
        raise ValueError(f"Scenario mismatch for {system_name} {test_id}.")
    if not _vectors_match(initial_condition, artifact_test.get("initial_condition")):
        # Level 3 in legacy reports may store the pushed simulated IC instead
        # of the artifact's declared IC. The resolver permits that only when
        # target+level identified exactly one artifact test.
        if updated.get("test_id") or int(scenario_level) != 3:
            raise ValueError(f"Initial-condition mismatch for {system_name} {test_id}.")
    if not _vectors_match(target, artifact_test.get("target")):
        raise ValueError(f"Target mismatch for {system_name} {test_id}.")

    if not updated.get("test_id"):
        updated["test_id"] = test_id

    updated["total_complexity"] = copy.deepcopy(artifact_test["total_cmplx"])

    verification = updated.get("verification")
    metrics = verification.get("metrics") if isinstance(verification, dict) else None
    rmse = metrics.get("rmse") if isinstance(metrics, dict) else None
    updated["success_rate"] = (
        calculate_success_rate(rmse) if rmse is not None else None
    )
    reprice_token_usage(updated)
    return updated


def create_full_results(
    input_path: Path,
    artifact_tests: dict[tuple[str, str], dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    """Read one clean result JSON and write its separate ``*_full.json`` copy."""
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(f"{input_path} must contain a JSON array of runs.")

    full_records = [fill_result_record(record, artifact_tests) for record in records]
    destination = output_path or input_path.with_name(f"{input_path.stem}_full.json")
    if destination.resolve() == input_path.resolve():
        raise ValueError("The full-result output must differ from the base input file.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(full_records, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return destination


def default_result_files(results_dir: Path) -> list[Path]:
    """Find the one clean base report in each model subfolder."""
    paths = sorted(results_dir.glob("*/batch_tuning_results.json"))
    if not paths:
        raise FileNotFoundError(
            f"No model-folder batch_tuning_results.json files found in {results_dir}."
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create *_full.json reports with artifact complexity and exp(-RMSE)."
    )
    parser.add_argument(
        "results_json",
        nargs="*",
        type=Path,
        help="Clean base result files. Defaults to every model folder's standard base report.",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Custom output path; valid only when one input file is supplied.",
    )
    args = parser.parse_args()

    explicit_inputs = bool(args.results_json)
    input_paths = [path.resolve() for path in args.results_json]
    if not explicit_inputs:
        input_paths = default_result_files(DEFAULT_RESULTS_DIR)
    if args.output is not None and len(input_paths) != 1:
        raise ValueError("--output can be used only with exactly one input JSON.")

    artifact_tests = load_artifact_tests(args.artifacts_dir.resolve())
    outputs: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for input_path in input_paths:
        try:
            output = create_full_results(
                input_path=input_path,
                artifact_tests=artifact_tests,
                output_path=args.output.resolve() if args.output is not None else None,
            )
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            if explicit_inputs:
                raise
            skipped.append((input_path, str(exc)))
            print(f"Skipped stale/incompatible report: {input_path}")
            print(f"  Reason: {exc}")
            continue

        outputs.append(output)
        print(f"Base report preserved: {input_path}")
        print(f"Full report written:  {output}")

    if skipped:
        print(
            f"Skipped {len(skipped)} report(s) created from older artifact tests. "
            "Rerun the benchmark for those model folders before finalizing them."
        )
    if not outputs:
        raise RuntimeError(
            "No compatible report was finalized. Rerun mpc_becnchmark.py with "
            "the current artifacts, then run this command again."
        )


if __name__ == "__main__":
    main()
