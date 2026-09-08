"""Add the test matrix to every artifact.

Reads each artifact in ``artifacts/`` and writes a ``tests`` block holding

  1. ``weights``      -- the names of the total-complexity weights, values null,
  2. ``combinations`` -- every "Init Condition & Targets" x "Scenarios" pair,
                         each carrying its IC, target, scenario and a null
                         ``total_cmplx``.

Run from the repository root:

    .venv\\Scripts\\python.exe benchmarks\\make_tests.py

Only the structure is created here. ``score_complexity.py`` is what fills the
weights and every complexity value.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BENCHMARK_DIR / "artifacts"

IC_TARGETS_KEY = "Init Condition & Targets"
SCENARIOS_KEY = "Scenarios"
TESTS_KEY = "tests"

# Names only. score_complexity.py owns the values.
WEIGHT_NAMES = ("plant_weight", "ic_target_weight", "scenario_weight")

# Artifact key order, so a rewritten file stays readable.
KEY_ORDER = [
    "system_name",
    "python_code",
    "metadata",
    "plant_cmplx",
    IC_TARGETS_KEY,
    SCENARIOS_KEY,
    TESTS_KEY,
]


def empty_weights() -> dict[str, None]:
    return {name: None for name in WEIGHT_NAMES}


def build_tests(artifact: dict[str, Any], keep_complexity: bool = False) -> dict[str, Any]:
    """Build the tests block: weights first, then one entry per combination."""
    ic_targets = artifact.get(IC_TARGETS_KEY) or []
    scenarios = artifact.get(SCENARIOS_KEY) or []
    if not ic_targets:
        raise ValueError(f"Artifact has no {IC_TARGETS_KEY!r} entries.")
    if not scenarios:
        raise ValueError(f"Artifact has no {SCENARIOS_KEY!r} entries.")

    previous: dict[str, Any] = {}
    previous_weights = empty_weights()
    if keep_complexity:
        old_tests = artifact.get(TESTS_KEY) or {}
        previous = {
            str(test.get("test_id")): test.get("total_cmplx")
            for test in old_tests.get("combinations", [])
        }
        for name in WEIGHT_NAMES:
            previous_weights[name] = (old_tests.get("weights") or {}).get(name)

    combinations = []
    for case in ic_targets:
        for scenario in scenarios:
            test_id = f"ic{int(case['id'])}_s{int(scenario['level'])}"
            combinations.append(
                {
                    "test_id": test_id,
                    "ic_target_id": int(case["id"]),
                    "initial_condition": list(case["initial_state"]),
                    "target": list(case["target_state"]),
                    "scenario_level": int(scenario["level"]),
                    "scenario_name": scenario.get("name"),
                    "scenario_settings": copy.deepcopy(scenario.get("settings") or {}),
                    "total_cmplx": previous.get(test_id),
                }
            )

    return {"weights": previous_weights, "combinations": combinations}


def process_artifact_file(
    path: Path,
    keep_complexity: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise TypeError(f"{path} must contain a single JSON artifact object.")

    artifact[TESTS_KEY] = build_tests(artifact, keep_complexity=keep_complexity)

    unknown = [key for key in artifact if key not in KEY_ORDER]
    ordered = {key: artifact[key] for key in KEY_ORDER if key in artifact}
    ordered.update({key: artifact[key] for key in unknown})

    if not dry_run:
        path.write_text(
            json.dumps(ordered, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    return ordered


def artifact_files(artifacts_dir: Path) -> list[Path]:
    paths = sorted(
        path for path in artifacts_dir.glob("*.json") if not path.name.startswith("_")
    )
    if not paths:
        raise FileNotFoundError(f"No JSON artifacts found in {artifacts_dir}.")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the tests block (weights + combinations) in every artifact."
    )
    parser.add_argument("artifacts", nargs="*", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument(
        "--keep-complexity",
        action="store_true",
        help="Preserve total_cmplx and weight values already present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report without writing the artifacts back.",
    )
    args = parser.parse_args()

    paths = [path.resolve() for path in args.artifacts] or artifact_files(
        args.artifacts_dir
    )

    total_tests = 0
    for path in paths:
        artifact = process_artifact_file(
            path, keep_complexity=args.keep_complexity, dry_run=args.dry_run
        )
        tests = artifact[TESTS_KEY]["combinations"]
        total_tests += len(tests)
        print(
            f"{artifact['system_name']:<26} "
            f"{len(artifact[IC_TARGETS_KEY])} IC/targets x "
            f"{len(artifact[SCENARIOS_KEY])} scenarios = {len(tests)} tests"
        )

    action = "Would write" if args.dry_run else "Wrote"
    print(f"\n{action} {total_tests} tests across {len(paths)} artifacts.")
    print("Next: benchmarks/score_complexity.py fills the weights and every cmplx.")


if __name__ == "__main__":
    main()
