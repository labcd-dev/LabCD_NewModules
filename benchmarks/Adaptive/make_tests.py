"""Prepare the independent AgentAdaptive artifact test matrices.

The files in ``artifacts/`` remain direct AgentAdaptive ``system_spec`` objects.
This script adds only Adaptive benchmark data: bounds, IC/target cases,
scenarios, and the Cartesian test combinations with empty complexity marks.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


ADAPTIVE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ADAPTIVE_DIR / "artifacts"

BENCHMARK_KEY = "benchmark"
IC_TARGETS_KEY = "initial_conditions_and_targets"
SCENARIOS_KEY = "scenarios"
TESTS_KEY = "tests"

PLANT_WEIGHT_NAMES = (
    "state_no_weight",
    "dof_weight",
    "stable_dynamics_weight",
)
CONDITION_WEIGHT_NAMES = ("ic_target_weight", "scenario_weight")
TOTAL_WEIGHT_NAMES = ("plant_weight", "condition_weight")
WEIGHT_GROUPS = {
    "plant": PLANT_WEIGHT_NAMES,
    "condition": CONDITION_WEIGHT_NAMES,
    "total": TOTAL_WEIGHT_NAMES,
}


ARTIFACT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "double_integrator.json": {
        "references": [{"output": "position", "expr": "0"}],
        "state_bounds": [[-12.0, -8.0], [12.0, 8.0]],
        "input_bounds": [[-10.0], [10.0]],
        "derivative_pairs": [[0, 1]],
        "cases": [
            [[0.5, 0.0], [0.0, 0.0]],
            [[2.0, 0.3], [0.0, 0.0]],
            [[-4.0, 0.6], [3.0, 0.0]],
        ],
        "noise_std": 0.01,
    },
    "mass_spring_damper.json": {
        "references": [{"output": "position", "expr": "0"}],
        "state_bounds": [[-8.0, -10.0], [8.0, 10.0]],
        "input_bounds": [[-20.0], [20.0]],
        "derivative_pairs": [[0, 1]],
        "cases": [
            [[0.5, 0.0], [0.0, 0.0]],
            [[2.0, 0.3], [0.0, 0.0]],
            [[-4.0, 0.6], [2.0, 0.0]],
        ],
        "noise_std": 0.01,
    },
    "two_mass_spring.json": {
        "references": [
            {"output": "position_1", "expr": "0"},
            {"output": "position_2", "expr": "0"},
        ],
        "state_bounds": [[-8.0, -10.0, -8.0, -10.0], [8.0, 10.0, 8.0, 10.0]],
        "input_bounds": [[-20.0, -20.0], [20.0, 20.0]],
        "derivative_pairs": [[0, 1], [2, 3]],
        "cases": [
            [[0.4, 0.0, -0.2, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[1.75, 0.3, -1.0, -0.15], [0.0, 0.0, 0.0, 0.0]],
            [[-3.5, 0.5, 3.0, -0.4], [0.0, 0.0, 0.0, 0.0]],
        ],
        "noise_std": 0.01,
    },
    "triple_mass_spring.json": {
        "references": [{"output": "x1", "expr": "0"}],
        "state_bounds": [
            [-8.0, -10.0, -8.0, -10.0, -8.0, -10.0],
            [8.0, 10.0, 8.0, 10.0, 8.0, 10.0],
        ],
        "input_bounds": [[-20.0], [20.0]],
        "derivative_pairs": [[0, 1], [2, 3], [4, 5]],
        "cases": [
            [[0.4, 0.0, 0.2, 0.0, -0.2, 0.0], [0.0] * 6],
            [[1.5, 0.25, 0.75, 0.0, -0.75, -0.1], [0.0] * 6],
            [[-3.0, 0.5, 2.0, -0.25, 3.0, 0.25], [0.0] * 6],
        ],
        "noise_std": 0.01,
    },
    "cart_pole.json": {
        "references": [{"output": "pole_angle", "expr": "0"}],
        "state_bounds": [[-6.0, -8.0, -1.2, -8.0], [6.0, 8.0, 1.2, 8.0]],
        "input_bounds": [[-20.0], [20.0]],
        "derivative_pairs": [[0, 1], [2, 3]],
        "cases": [
            [[0.25, 0.0, 0.02, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[1.2, 0.2, 0.08, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[-2.5, 0.4, 0.12, -0.05], [2.0, 0.0, 0.0, 0.0]],
        ],
        "noise_std": 0.005,
    },
    "planar_drone.json": {
        "references": [
            {"output": "z", "expr": "1"},
            {"output": "pitch", "expr": "0"},
        ],
        "state_bounds": [
            [-15.0, -10.0, 0.0, -10.0, -1.0, -6.0],
            [15.0, 10.0, 20.0, 10.0, 1.0, 6.0],
        ],
        "input_bounds": [[0.0, -4.0], [35.316, 4.0]],
        "derivative_pairs": [[0, 1], [2, 3], [4, 5]],
        "cases": [
            [[0.4, 0.0, 2.2, 0.0, 0.02, 0.0], [0.0, 0.0, 2.0, 0.0, 0.0, 0.0]],
            [[2.0, 0.25, 4.0, -0.25, 0.08, 0.0], [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]],
            [[-5.0, 0.5, 8.0, -0.5, 0.15, 0.1], [4.0, 0.0, 4.0, 0.0, 0.0, 0.0]],
        ],
        "noise_std": 0.005,
    },
}


def empty_weights() -> dict[str, dict[str, None]]:
    return {
        group: {name: None for name in names}
        for group, names in WEIGHT_GROUPS.items()
    }


def _scenarios(n_states: int, noise_std: float) -> list[dict[str, Any]]:
    disabled = [False] * n_states
    enabled = [True] * n_states
    return [
        {
            "level": 1,
            "name": "Nominal",
            "settings": {
                "noise_std_value": 0.0,
                "noise_state_mask": disabled,
                "robust_push_scale": 1.0,
                "robust_state_mask": disabled,
                "robust_noise_fraction": 0.0,
                "perturb_physical_params": False,
            },
            "cmplx": None,
        },
        {
            "level": 2,
            "name": "Noise",
            "settings": {
                "noise_std_value": noise_std,
                "noise_state_mask": enabled,
                "robust_push_scale": 1.0,
                "robust_state_mask": disabled,
                "robust_noise_fraction": 1.0,
                "perturb_physical_params": False,
            },
            "cmplx": None,
        },
        {
            "level": 3,
            "name": "Robust",
            "settings": {
                "noise_std_value": noise_std,
                "noise_state_mask": enabled,
                "robust_push_scale": 1.0,
                "robust_state_mask": disabled,
                "robust_noise_fraction": 0.5,
                "perturb_physical_params": True,
            },
            "cmplx": None,
        },
    ]


def prepare_artifact(path: Path, keep_complexity: bool = False) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    definition = ARTIFACT_DEFINITIONS.get(path.name)
    if definition is None:
        raise KeyError(f"No Adaptive benchmark definition for {path.name!r}.")

    dynamics = artifact.get("dynamics") or {}
    n_states = len(dynamics.get("states") or [])
    n_inputs = len(dynamics.get("inputs") or [])
    if n_states == 0 or n_inputs == 0:
        raise ValueError(f"{path.name} is not a complete Adaptive system_spec.")
    dynamics["references"] = copy.deepcopy(definition["references"])

    lower, upper = definition["state_bounds"]
    input_lower, input_upper = definition["input_bounds"]
    artifact[BENCHMARK_KEY] = {
        "n_states": n_states,
        "n_inputs": n_inputs,
        "state_bounds": {"lower": lower, "upper": upper},
        "input_bounds": {"lower": input_lower, "upper": input_upper},
        "derivative_pairs": copy.deepcopy(definition["derivative_pairs"]),
        "plant_cmplx": artifact.get(BENCHMARK_KEY, {}).get("plant_cmplx")
        if keep_complexity
        else None,
    }

    old_cases = artifact.get(IC_TARGETS_KEY) or []
    old_case_scores = {int(item["id"]): item.get("cmplx") for item in old_cases}
    artifact[IC_TARGETS_KEY] = [
        {
            "id": index,
            "initial_state": initial,
            "target_state": target,
            "cmplx": old_case_scores.get(index) if keep_complexity else None,
        }
        for index, (initial, target) in enumerate(definition["cases"], start=1)
    ]

    old_scenarios = artifact.get(SCENARIOS_KEY) or []
    old_scenario_scores = {
        int(item["level"]): item.get("cmplx") for item in old_scenarios
    }
    scenarios = _scenarios(n_states, float(definition["noise_std"]))
    if keep_complexity:
        for scenario in scenarios:
            scenario["cmplx"] = old_scenario_scores.get(int(scenario["level"]))
    artifact[SCENARIOS_KEY] = scenarios

    old_tests = artifact.get(TESTS_KEY) or {}
    old_test_scores = {
        str(item["test_id"]): (
            item.get("condition_cmplx"),
            item.get("total_cmplx"),
        )
        for item in old_tests.get("combinations", [])
    }
    combinations = []
    for case in artifact[IC_TARGETS_KEY]:
        for scenario in scenarios:
            test_id = f"ic{case['id']}_s{scenario['level']}"
            previous = old_test_scores.get(test_id, (None, None))
            combinations.append(
                {
                    "test_id": test_id,
                    "ic_target_id": case["id"],
                    "initial_condition": list(case["initial_state"]),
                    "target": list(case["target_state"]),
                    "scenario_level": scenario["level"],
                    "scenario_name": scenario["name"],
                    "scenario_settings": copy.deepcopy(scenario["settings"]),
                    "condition_cmplx": previous[0] if keep_complexity else None,
                    "total_cmplx": previous[1] if keep_complexity else None,
                }
            )
    artifact[TESTS_KEY] = {
        "weights": old_tests.get("weights", empty_weights())
        if keep_complexity
        else empty_weights(),
        "combinations": combinations,
    }
    return artifact


def artifact_files(artifacts_dir: Path) -> list[Path]:
    paths = sorted(artifacts_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No Adaptive artifacts found in {artifacts_dir}.")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Adaptive artifact tests.")
    parser.add_argument("artifacts", nargs="*", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--keep-complexity", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = [path.resolve() for path in args.artifacts] or artifact_files(
        args.artifacts_dir
    )
    total = 0
    for path in paths:
        artifact = prepare_artifact(path, keep_complexity=args.keep_complexity)
        combinations = artifact[TESTS_KEY]["combinations"]
        total += len(combinations)
        if not args.dry_run:
            path.write_text(
                json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
        print(f"{path.name:<28} {len(combinations)} tests")

    action = "Would prepare" if args.dry_run else "Prepared"
    print(f"{action} {total} Adaptive tests across {len(paths)} artifacts.")
    print("Next: benchmarks/Adaptive/score_complexity.py")


if __name__ == "__main__":
    main()
