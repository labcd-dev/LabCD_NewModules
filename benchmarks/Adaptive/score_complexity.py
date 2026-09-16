"""Score complexity for the independent AgentAdaptive artifacts.

Run ``make_tests.py`` first. This scorer evaluates the symbolic Adaptive
equations directly; it does not import MPC plugins or MPC benchmark artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from make_tests import (
    ARTIFACTS_DIR,
    BENCHMARK_KEY,
    CONDITION_WEIGHT_NAMES,
    IC_TARGETS_KEY,
    PLANT_WEIGHT_NAMES,
    SCENARIOS_KEY,
    TESTS_KEY,
    TOTAL_WEIGHT_NAMES,
)


STATE_COUNT_EASY = 2
STATE_COUNT_HARD = 6
DOF_COUNT_EASY = 1
DOF_COUNT_HARD = 6

PLANT_WEIGHTS = (0.50, 0.25, 0.25)
CONDITION_WEIGHTS = (0.60, 0.40)
TOTAL_WEIGHTS = (0.50, 0.50)
POSITION_WEIGHT = 0.50
RATE_WEIGHT = 0.50

STABILITY_SAMPLE_COUNT = 20
STABILITY_SIMULATION_TIME = 10.0
STABILITY_DT = 0.01
STABILITY_BASE_SEED = 20260912
STABILITY_ZERO_TOLERANCE = 1e-10

MIN_SCENARIO_LEVEL = 1
MAX_SCENARIO_LEVEL = 3
ROUND_DIGITS = 6


class SymbolicDynamics:
    """Small numerical adapter around one Adaptive symbolic system_spec."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        dynamics = artifact["dynamics"]
        state_names = list(dynamics["states"])
        input_names = list(dynamics["inputs"])
        equations = list(dynamics["state_equations"])
        parameters = dict(dynamics.get("parameters") or {})

        self.n_states = len(state_names)
        self.n_inputs = len(input_names)
        if len(equations) != self.n_states:
            raise ValueError("Adaptive artifact needs one equation per state.")

        symbol_map = {
            name: sp.Symbol(name)
            for name in [*state_names, *input_names, *parameters.keys()]
        }
        substitutions = {
            symbol_map[name]: float(value) for name, value in parameters.items()
        }
        parsed = [
            sp.sympify(expression, locals=symbol_map).subs(substitutions)
            for expression in equations
        ]
        arguments = [
            *(symbol_map[name] for name in state_names),
            *(symbol_map[name] for name in input_names),
        ]
        self._function = sp.lambdify(arguments, parsed, modules="numpy")

    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(self.n_states)
        control = np.asarray(control, dtype=float).reshape(self.n_inputs)
        values = self._function(*state.tolist(), *control.tolist())
        result = np.asarray(values, dtype=float).reshape(-1)
        if result.shape != (self.n_states,):
            raise ValueError(
                f"Dynamics returned {result.shape}; expected ({self.n_states},)."
            )
        return result

    def rk4_step(
        self,
        state: np.ndarray,
        control: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        k1 = self.dynamics(state, control)
        k2 = self.dynamics(state + 0.5 * dt * k1, control)
        k3 = self.dynamics(state + 0.5 * dt * k2, control)
        k4 = self.dynamics(state + dt * k3, control)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _rms(values: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values else 0.0


def _score(normalized: float, **extra: Any) -> dict[str, Any]:
    normalized = _clip01(normalized)
    return {
        "normalized_score": round(normalized, ROUND_DIGITS),
        "score_1_to_10": round(1.0 + 9.0 * normalized, ROUND_DIGITS),
        **extra,
    }


def _named_weights(
    values: tuple[float, ...],
    names: tuple[str, ...],
    group: str,
) -> dict[str, float]:
    if len(values) != len(names) or not np.isclose(sum(values), 1.0):
        raise ValueError(f"Invalid {group} complexity weights: {values}.")
    return {name: float(value) for name, value in zip(names, values)}


def complexity_weights() -> dict[str, dict[str, float]]:
    return {
        "plant": _named_weights(PLANT_WEIGHTS, PLANT_WEIGHT_NAMES, "plant"),
        "condition": _named_weights(
            CONDITION_WEIGHTS,
            CONDITION_WEIGHT_NAMES,
            "condition",
        ),
        "total": _named_weights(TOTAL_WEIGHTS, TOTAL_WEIGHT_NAMES, "total"),
    }


def _derivative_pairs(config: dict[str, Any]) -> list[tuple[int, int]]:
    pairs = [
        (int(position), int(rate))
        for position, rate in config.get("derivative_pairs") or []
    ]
    if not pairs:
        raise ValueError("At least one derivative pair is required.")
    n_states = int(config["n_states"])
    flattened = [index for pair in pairs for index in pair]
    if any(index < 0 or index >= n_states for index in flattened):
        raise ValueError("A derivative-pair index is outside the state vector.")
    if len(flattened) != len(set(flattened)):
        raise ValueError("Derivative pairs must not reuse state indices.")
    return pairs


def _artifact_seed(system_name: str) -> int:
    digest = hashlib.sha256(system_name.encode("utf-8")).digest()
    return STABILITY_BASE_SEED + int.from_bytes(digest[:4], "big")


def calculate_stable_dynamics_cmplx(
    artifact: dict[str, Any],
    config: dict[str, Any],
    pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    n_states = int(config["n_states"])
    lower = np.asarray(config["state_bounds"]["lower"], dtype=float)
    upper = np.asarray(config["state_bounds"]["upper"], dtype=float)
    if lower.shape != (n_states,) or upper.shape != (n_states,):
        raise ValueError(f"State bounds must have shape ({n_states},).")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("State bounds must be finite.")
    if np.any(upper <= lower):
        raise ValueError("Every upper state bound must exceed its lower bound.")

    model = SymbolicDynamics(artifact)
    if model.n_states != n_states:
        raise ValueError("Adaptive equations and benchmark n_states disagree.")

    position_indices = np.asarray([position for position, _ in pairs], dtype=int)
    rate_indices = np.asarray([rate for _, rate in pairs], dtype=int)
    position_spans = upper[position_indices] - lower[position_indices]
    zero_input = np.zeros(model.n_inputs, dtype=float)
    n_steps = max(int(round(STABILITY_SIMULATION_TIME / STABILITY_DT)), 1)
    seed = _artifact_seed(str(artifact.get("system_name") or "artifact"))
    rng = np.random.default_rng(seed)
    samples: list[float] = []

    for _ in range(STABILITY_SAMPLE_COUNT):
        state = rng.uniform(lower, upper)
        state[rate_indices] = 0.0
        initial_positions = state[position_indices].copy()
        maximum_drift = 0.0

        for _step in range(n_steps):
            try:
                state = np.asarray(
                    model.rk4_step(state, zero_input, STABILITY_DT),
                    dtype=float,
                )
            except Exception:
                maximum_drift = 1.0
                break
            if state.shape != (n_states,) or not np.all(np.isfinite(state)):
                maximum_drift = 1.0
                break
            normalized_drift = np.abs(
                state[position_indices] - initial_positions
            ) / position_spans
            maximum_drift = max(maximum_drift, float(np.max(normalized_drift)))
            if np.any(state < lower) or np.any(state > upper):
                maximum_drift = 1.0
                break

        if maximum_drift <= STABILITY_ZERO_TOLERANCE:
            maximum_drift = 0.0
        samples.append(_clip01(maximum_drift))

    return _score(
        float(np.mean(samples)),
        sample_count=STABILITY_SAMPLE_COUNT,
        simulation_time=STABILITY_SIMULATION_TIME,
        dt=STABILITY_DT,
        random_seed=seed,
    )


def calculate_plant_cmplx(artifact: dict[str, Any]) -> dict[str, Any]:
    config = artifact[BENCHMARK_KEY]
    n_states = int(config["n_states"])
    pairs = _derivative_pairs(config)
    n_dof = len(pairs)
    state_score = _score(
        (n_states - STATE_COUNT_EASY) / (STATE_COUNT_HARD - STATE_COUNT_EASY),
        n_states=n_states,
        reference={"easy": STATE_COUNT_EASY, "hard": STATE_COUNT_HARD},
    )
    dof_score = _score(
        (n_dof - DOF_COUNT_EASY) / (DOF_COUNT_HARD - DOF_COUNT_EASY),
        n_dof=n_dof,
        reference={"easy": DOF_COUNT_EASY, "hard": DOF_COUNT_HARD},
    )
    motion_score = calculate_stable_dynamics_cmplx(artifact, config, pairs)
    normalized = sum(
        weight * component["normalized_score"]
        for weight, component in zip(
            PLANT_WEIGHTS,
            (state_score, dof_score, motion_score),
        )
    )
    return {
        "state_no_cmplx": state_score,
        "dof_cmplx": dof_score,
        "stable_dynamics_cmpx": motion_score,
        **_score(normalized),
    }


def calculate_ic_target_cmplx(
    config: dict[str, Any],
    initial_state: list[float],
    target_state: list[float],
) -> dict[str, Any]:
    n_states = int(config["n_states"])
    initial = np.asarray(initial_state, dtype=float)
    target = np.asarray(target_state, dtype=float)
    lower = np.asarray(config["state_bounds"]["lower"], dtype=float)
    upper = np.asarray(config["state_bounds"]["upper"], dtype=float)
    expected = (n_states,)
    if any(value.shape != expected for value in (initial, target, lower, upper)):
        raise ValueError(f"IC, target, and bounds must have shape {expected}.")

    position_components: list[float] = []
    rate_components: list[float] = []
    for position, rate in _derivative_pairs(config):
        position_reference = max(
            abs(lower[position] - target[position]),
            abs(upper[position] - target[position]),
        )
        rate_reference = max(
            abs(lower[rate] - target[rate]),
            abs(upper[rate] - target[rate]),
        )
        if position_reference <= 0.0 or rate_reference <= 0.0:
            raise ValueError("Bounds do not provide a nonzero normalization range.")
        position_components.append(
            _clip01(abs(initial[position] - target[position]) / position_reference)
        )
        rate_components.append(
            _clip01(abs(initial[rate] - target[rate]) / rate_reference)
        )

    normalized = (
        POSITION_WEIGHT * _clip01(_rms(position_components))
        + RATE_WEIGHT * _clip01(_rms(rate_components))
    )
    return _score(normalized)


def calculate_scenario_cmplx(level: int) -> dict[str, Any]:
    level = int(level)
    if not MIN_SCENARIO_LEVEL <= level <= MAX_SCENARIO_LEVEL:
        raise ValueError(f"Unsupported scenario level: {level}.")
    return _score(
        (level - MIN_SCENARIO_LEVEL)
        / (MAX_SCENARIO_LEVEL - MIN_SCENARIO_LEVEL)
    )


def fill_artifact_complexity(artifact: dict[str, Any]) -> dict[str, Any]:
    config = artifact[BENCHMARK_KEY]
    weights = complexity_weights()
    config["plant_cmplx"] = calculate_plant_cmplx(artifact)

    for case in artifact[IC_TARGETS_KEY]:
        case["cmplx"] = calculate_ic_target_cmplx(
            config,
            case["initial_state"],
            case["target_state"],
        )
    for scenario in artifact[SCENARIOS_KEY]:
        scenario["cmplx"] = calculate_scenario_cmplx(scenario["level"])

    tests = artifact.get(TESTS_KEY) or {}
    combinations = tests.get("combinations") or []
    if not combinations:
        raise ValueError("Run Adaptive/make_tests.py before scoring complexity.")
    tests["weights"] = weights

    cases = {int(item["id"]): item for item in artifact[IC_TARGETS_KEY]}
    scenarios = {
        int(item["level"]): item for item in artifact[SCENARIOS_KEY]
    }
    for test in combinations:
        case = cases[int(test["ic_target_id"])]
        scenario = scenarios[int(test["scenario_level"])]
        condition = (
            weights["condition"]["ic_target_weight"]
            * case["cmplx"]["normalized_score"]
            + weights["condition"]["scenario_weight"]
            * scenario["cmplx"]["normalized_score"]
        )
        test["condition_cmplx"] = _score(condition)
        total = (
            weights["total"]["plant_weight"]
            * config["plant_cmplx"]["normalized_score"]
            + weights["total"]["condition_weight"]
            * test["condition_cmplx"]["normalized_score"]
        )
        test["total_cmplx"] = _score(total)
    return artifact


def artifact_files(artifacts_dir: Path) -> list[Path]:
    paths = sorted(artifacts_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No Adaptive artifacts found in {artifacts_dir}.")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Adaptive artifacts.")
    parser.add_argument("artifacts", nargs="*", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = [path.resolve() for path in args.artifacts] or artifact_files(
        args.artifacts_dir
    )
    for path in paths:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        fill_artifact_complexity(artifact)
        if not args.dry_run:
            path.write_text(
                json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
        plant = artifact[BENCHMARK_KEY]["plant_cmplx"]["score_1_to_10"]
        totals = [
            item["total_cmplx"]["score_1_to_10"]
            for item in artifact[TESTS_KEY]["combinations"]
        ]
        print(
            f"{path.name:<28} plant={plant:>5.2f} "
            f"total={min(totals):>5.2f}-{max(totals):<5.2f}"
        )

    action = "Would score" if args.dry_run else "Scored"
    print(f"{action} {len(paths)} Adaptive artifacts.")


if __name__ == "__main__":
    main()
