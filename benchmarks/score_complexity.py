"""Fill every complexity value in the JSON benchmark artifacts.

Run ``make_tests.py`` first to create the tests block, then this:

    .venv\\Scripts\\python.exe benchmarks\\score_complexity.py

For each artifact in ``artifacts/`` this fills

  1. ``plant_cmplx``                        -- from the state count,
  2. ``Init Condition & Targets[i].cmplx``  -- from initial/target distance and rate,
  3. ``Scenarios[i].cmplx``                 -- from the scenario level,
  4. ``tests.weights``                      -- the three total-complexity weights,
  5. ``tests.combinations[j].total_cmplx``  -- the weighted total per test,

and writes them back in place. Everything is read from the artifact JSON itself,
so no dynamics code is executed.

Formulas (v1)
-------------
    plant      = clip((n_states - 2) / (12 - 2), 0, 1)
    ic_target  = 0.5 * rms(position_error / bound_distance)
               + 0.5 * rms(rate_error / bound_distance)
    scenario   = (level - 1) / (3 - 1)
    total      = plant_weight     * plant
               + ic_target_weight * ic_target
               + scenario_weight  * scenario

The three weights are 0.30 / 0.30 / 0.40 and sum to 1. They weight plant size,
the IC/target distance, and scenario difficulty respectively.

Every score is reported twice: ``normalized_score`` in [0, 1] and
``score_1_to_10 = 1 + 9 * normalized_score`` for the dashboard axes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from make_tests import (
    IC_TARGETS_KEY,
    SCENARIOS_KEY,
    TESTS_KEY,
    WEIGHT_NAMES,
)


BENCHMARK_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BENCHMARK_DIR / "artifacts"

# ------------------------- COMPLEXITY DEFINITION -------------------------
# Plant: state count only (v1).
STATE_COUNT_EASY = 2
STATE_COUNT_HARD = 12

# Scenario: level 1 is nominal, level 3 is robust.
MIN_SCENARIO_LEVEL = 1
MAX_SCENARIO_LEVEL = 3

# IC/target halves.
POSITION_WEIGHT = 0.5
RATE_WEIGHT = 0.5

# Total-complexity weights written into tests.weights. One value per name in
# WEIGHT_NAMES order (plant, ic_target, scenario); they must sum to 1.
TOTAL_WEIGHTS = (0.30, 0.30, 0.40)
# -------------------------------------------------------------------------

ROUND_DIGITS = 6


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _rms(values: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values else 0.0


def _score(normalized: float, **extra: Any) -> dict[str, Any]:
    """Wrap a normalized score with its 1-10 display value."""
    normalized = _clip01(normalized)
    return {
        "normalized_score": round(normalized, ROUND_DIGITS),
        "score_1_to_10": round(1.0 + 9.0 * normalized, ROUND_DIGITS),
        **extra,
    }


def total_weights() -> dict[str, float]:
    """The named total-complexity weights."""
    if len(TOTAL_WEIGHTS) != len(WEIGHT_NAMES):
        raise ValueError(
            f"TOTAL_WEIGHTS must supply one value per weight name "
            f"({len(WEIGHT_NAMES)}); got {len(TOTAL_WEIGHTS)}."
        )
    if not np.isclose(sum(TOTAL_WEIGHTS), 1.0):
        raise ValueError(f"TOTAL_WEIGHTS must sum to 1; got {sum(TOTAL_WEIGHTS)}.")
    return {
        name: float(value) for name, value in zip(WEIGHT_NAMES, TOTAL_WEIGHTS)
    }


# ----------------------------- 1. PLANT -----------------------------


def calculate_plant_cmplx(metadata: dict[str, Any]) -> dict[str, Any]:
    """Score the plant from its state count."""
    n_states = int(metadata["n_states"])
    if n_states < 1:
        raise ValueError(f"n_states must be positive; got {n_states}.")

    normalized = _clip01(
        (n_states - STATE_COUNT_EASY) / (STATE_COUNT_HARD - STATE_COUNT_EASY)
    )
    return _score(
        normalized,
        state_count_reference={"easy": STATE_COUNT_EASY, "hard": STATE_COUNT_HARD},
    )


# -------------------------- 2. IC AND TARGET --------------------------


def calculate_ic_target_cmplx(
    metadata: dict[str, Any],
    initial_state: list[float],
    target_state: list[float],
) -> dict[str, Any]:
    """Score how hard one initial-condition/target pair is for this plant.

    Position and rate errors are normalized by the largest target-to-bound
    distance of the corresponding state, so 1.0 means "as far from the target as
    this state is allowed to be". Multiple degrees of freedom combine with RMS
    rather than a mean, so one difficult coordinate is not averaged away by easy
    ones.
    """
    n_states = int(metadata["n_states"])
    expected_shape = (n_states,)
    initial = np.asarray(initial_state, dtype=float)
    target = np.asarray(target_state, dtype=float)
    if initial.shape != expected_shape or target.shape != expected_shape:
        raise ValueError(
            f"Initial and target states must both have shape {expected_shape}; "
            f"got {initial.shape} and {target.shape}."
        )
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(target)):
        raise ValueError("Initial and target states must be finite.")

    lower = np.asarray(metadata["state_bounds"]["lower"], dtype=float)
    upper = np.asarray(metadata["state_bounds"]["upper"], dtype=float)
    if lower.shape != expected_shape or upper.shape != expected_shape:
        raise ValueError(f"State bounds must both have shape {expected_shape}.")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("All state-bound values must be finite.")
    if np.any(upper <= lower):
        raise ValueError("Every upper state bound must exceed its lower bound.")

    pairs = [
        (int(position), int(rate)) for position, rate in metadata["derivative_pairs"]
    ]
    if not pairs:
        raise ValueError("At least one position/rate state pair is required.")

    position_components: list[float] = []
    rate_components: list[float] = []

    for position_index, rate_index in pairs:
        position_reference = max(
            abs(lower[position_index] - target[position_index]),
            abs(upper[position_index] - target[position_index]),
        )
        rate_reference = max(
            abs(lower[rate_index] - target[rate_index]),
            abs(upper[rate_index] - target[rate_index]),
        )
        if position_reference <= 0.0 or rate_reference <= 0.0:
            raise ValueError(
                "State bounds must provide a nonzero normalization range for "
                f"states {position_index} and {rate_index}."
            )

        position_components.append(
            _clip01(
                abs(initial[position_index] - target[position_index])
                / position_reference
            )
        )
        rate_components.append(
            _clip01(abs(initial[rate_index] - target[rate_index]) / rate_reference)
        )

    normalized = POSITION_WEIGHT * _clip01(
        _rms(position_components)
    ) + RATE_WEIGHT * _clip01(_rms(rate_components))
    return _score(normalized)


# ---------------------------- 3. SCENARIO ----------------------------


def calculate_scenario_cmplx(level: int) -> dict[str, Any]:
    """Score one scenario level on the declared 1-3 range."""
    level = int(level)
    if not MIN_SCENARIO_LEVEL <= level <= MAX_SCENARIO_LEVEL:
        raise ValueError(
            f"scenario level must be between {MIN_SCENARIO_LEVEL} and "
            f"{MAX_SCENARIO_LEVEL}; got {level}."
        )
    normalized = _clip01(
        (level - MIN_SCENARIO_LEVEL) / (MAX_SCENARIO_LEVEL - MIN_SCENARIO_LEVEL)
    )
    return _score(normalized)


# ------------------------------ 4. TOTAL ------------------------------


def calculate_total_cmplx(
    plant_cmplx: dict[str, Any],
    ic_target_cmplx: dict[str, Any],
    scenario_cmplx: dict[str, Any],
    weights: dict[str, float],
) -> dict[str, Any]:
    """Blend the three stored scores into one test's total complexity."""
    plant_weight, ic_target_weight, scenario_weight = (
        float(weights[name]) for name in WEIGHT_NAMES
    )
    total = (
        plant_weight * float(plant_cmplx["normalized_score"])
        + ic_target_weight * float(ic_target_cmplx["normalized_score"])
        + scenario_weight * float(scenario_cmplx["normalized_score"])
    )
    return _score(total)


# ------------------------------ DRIVER ------------------------------


def fill_artifact_complexity(artifact: dict[str, Any]) -> dict[str, Any]:
    """Calculate and write every complexity value of one artifact in place."""
    metadata = artifact["metadata"]

    artifact["plant_cmplx"] = calculate_plant_cmplx(metadata)

    for case in artifact[IC_TARGETS_KEY]:
        case["cmplx"] = calculate_ic_target_cmplx(
            metadata=metadata,
            initial_state=case["initial_state"],
            target_state=case["target_state"],
        )

    for scenario in artifact[SCENARIOS_KEY]:
        scenario["cmplx"] = calculate_scenario_cmplx(scenario["level"])

    tests = artifact.get(TESTS_KEY)
    if not isinstance(tests, dict) or not tests.get("combinations"):
        raise ValueError(
            f"Artifact has no {TESTS_KEY!r} combinations. "
            "Run make_tests.py first."
        )

    weights = total_weights()
    tests["weights"] = weights

    ic_target_by_id = {int(case["id"]): case for case in artifact[IC_TARGETS_KEY]}
    scenario_by_level = {int(item["level"]): item for item in artifact[SCENARIOS_KEY]}

    for test in tests["combinations"]:
        case = ic_target_by_id.get(int(test["ic_target_id"]))
        scenario = scenario_by_level.get(int(test["scenario_level"]))
        if case is None or scenario is None:
            raise ValueError(
                f"Test {test.get('test_id')!r} references an IC/target or "
                "scenario that this artifact does not declare."
            )
        test["total_cmplx"] = calculate_total_cmplx(
            plant_cmplx=artifact["plant_cmplx"],
            ic_target_cmplx=case["cmplx"],
            scenario_cmplx=scenario["cmplx"],
            weights=weights,
        )
    return artifact


def process_artifact_file(path: Path, dry_run: bool = False) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise TypeError(f"{path} must contain a single JSON artifact object.")

    fill_artifact_complexity(artifact)

    if not dry_run:
        path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    return artifact


def artifact_files(artifacts_dir: Path) -> list[Path]:
    paths = sorted(
        path for path in artifacts_dir.glob("*.json") if not path.name.startswith("_")
    )
    if not paths:
        raise FileNotFoundError(f"No JSON artifacts found in {artifacts_dir}.")
    return paths


def _report(artifact: dict[str, Any]) -> None:
    plant = artifact["plant_cmplx"]["score_1_to_10"]
    totals = [
        test["total_cmplx"]["score_1_to_10"]
        for test in artifact[TESTS_KEY]["combinations"]
    ]
    print(
        f"{artifact['system_name']:<26} plant={plant:>5.2f}  "
        f"tests={len(totals):>2}  "
        f"total={min(totals):>5.2f}-{max(totals):<5.2f}  "
        f"mean={sum(totals) / len(totals):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill plant, IC/target, scenario and total complexity in "
        "every JSON benchmark artifact."
    )
    parser.add_argument(
        "artifacts",
        nargs="*",
        type=Path,
        help="Specific artifact files. Defaults to every JSON in benchmarks/artifacts/.",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate and report without writing the artifacts back.",
    )
    args = parser.parse_args()

    paths = [path.resolve() for path in args.artifacts] or artifact_files(
        args.artifacts_dir
    )

    total_tests = 0
    for path in paths:
        artifact = process_artifact_file(path, dry_run=args.dry_run)
        _report(artifact)
        total_tests += len(artifact[TESTS_KEY]["combinations"])

    weights = ", ".join(f"{name}={value}" for name, value in total_weights().items())
    action = "Calculated" if args.dry_run else "Updated"
    print(f"\n{action} {len(paths)} artifacts, {total_tests} tests total.")
    print(f"Weights: {weights}")
    if args.dry_run:
        print("Dry run: no files were written.")


if __name__ == "__main__":
    main()
