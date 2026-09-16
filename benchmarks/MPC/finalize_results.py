"""Copy manual discrete complexity and success scores into *_full.json.

The base JSON is preserved. No tuning, simulation, or API calls occur here.
Normalize verification RMSE by the RMS of the artifact's state-bound ranges.
There is no plant/condition complexity calculation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any

MPC_DIR = Path(__file__).resolve().parent
REPO_ROOT = MPC_DIR.parents[1]
ARTIFACTS_DIR = MPC_DIR / "artifacts"
RESULTS_DIR = MPC_DIR / "json_reports" / "results"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ------------------------- FINALIZATION SETTINGS -------------------------
# True: ignore the model list and any CLI input paths; update only mock.
# False: update the listed folders, unless explicit CLI input paths are given.
USE_MOCK_RESULTS = False
MODEL_FOLDERS_TO_UPDATE = [
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-4o-mini",
    "gpt-4.1-nano",
    "groq-gpt-oss-120b",
]
# ------------------------------------------------------------------------


def select_result_paths(explicit_paths: list[Path]) -> list[Path]:
    """Select only mock in mock mode, otherwise CLI inputs or configured models."""
    if USE_MOCK_RESULTS:
        if explicit_paths:
            print("Mock mode: ignoring explicit input paths and updating only mock.")
        folders = ["mock"]
    elif explicit_paths:
        return explicit_paths
    else:
        folders = MODEL_FOLDERS_TO_UPDATE
    paths = []
    for folder in dict.fromkeys(folders):
        if (
            not isinstance(folder, str) or not folder.strip()
            or folder in {".", ".."} or "/" in folder or "\\" in folder or ":" in folder
        ):
            raise ValueError("Model entries must be simple result-folder names.")
        path = RESULTS_DIR / folder / "batch_tuning_results.json"
        if path.is_file():
            paths.append(path)
        else:
            print(f"Skipping missing selected report: {path}")
    return paths


def calculate_state_bound_scale(state_bounds: Any, n_states: int) -> float:
    """Return RMS(upper - lower) across all state dimensions."""
    if isinstance(n_states, bool) or not isinstance(n_states, int) or n_states < 1:
        raise ValueError("n_states must be a positive integer.")
    if not isinstance(state_bounds, dict):
        raise ValueError("Artifact metadata must define state_bounds with lower and upper lists.")
    lower, upper = state_bounds.get("lower"), state_bounds.get("upper")
    if (
        not isinstance(lower, list) or not isinstance(upper, list)
        or len(lower) != n_states or len(upper) != n_states
    ):
        raise ValueError("State-bound lower and upper lists must each have n_states entries.")
    ranges = []
    for lo, hi in zip(lower, upper):
        try:
            lo, hi = float(lo), float(hi)
        except (TypeError, ValueError) as exc:
            raise ValueError("State bounds must be finite numbers.") from exc
        width = hi - lo
        if not math.isfinite(lo) or not math.isfinite(hi) or not math.isfinite(width) or width <= 0:
            raise ValueError("Each state bound must be finite with upper strictly greater than lower.")
        ranges.append(width)
    # Scale before squaring to avoid overflow for large, valid bounds.
    largest = max(ranges)
    return largest * math.sqrt(sum((width / largest) ** 2 for width in ranges) / n_states)


def calculate_success_rate(verification_rmse: float, bound_scale: float) -> float:
    """Return exp(-verification RMSE / RMS state-bound range), in [0, 1]."""
    rmse = float(verification_rmse)
    bound_scale = float(bound_scale)
    if not math.isfinite(rmse) or rmse < 0:
        raise ValueError("Verification RMSE must be finite and nonnegative.")
    if not math.isfinite(bound_scale) or bound_scale <= 0:
        raise ValueError("State-bound scale must be finite and positive.")
    return round(math.exp(-20 * rmse / bound_scale), 6)


def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> dict[str, dict[str, Any]]:
    paths = sorted(artifacts_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No artifacts found in {artifacts_dir}.")
    artifacts = {}
    for path in paths:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(artifact, dict) or not isinstance(artifact.get("test"), dict):
            raise ValueError(f"{path.name} must contain one test object.")
        value = artifact.get("total_complexity")
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10
        ):
            raise ValueError(f"{path.name}: total_complexity must be null or an integer from 1 to 10.")
        metadata = artifact.get("metadata") or {}
        try:
            calculate_state_bound_scale(metadata.get("state_bounds"), metadata.get("n_states"))
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        artifacts[path.name] = artifact
    return artifacts


def vectors_match(left: Any, right: Any) -> bool:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    return all(
        math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
        for a, b in zip(left, right)
    )


def reprice_token_usage(record: dict[str, Any]) -> None:
    usage = (record.get("tuning") or {}).get("token_usage")
    if not isinstance(usage, dict) or usage.get("simulated"):
        return
    per_model = usage.get("per_model")
    if not isinstance(per_model, dict):
        return
    try:
        from labcd_agents.pricing import CostCalculator
    except ImportError:
        return
    calculator = CostCalculator()
    cost = 0.0
    unpriced = []
    for model, tokens in per_model.items():
        if model == "unknown" or not isinstance(tokens, dict):
            continue
        if calculator.resolve_price(model) is None:
            unpriced.append(model)
        else:
            cost += calculator.compute_cost(
                model, int(tokens.get("prompt", 0) or 0), int(tokens.get("completion", 0) or 0),
            )
    usage["cost_usd"] = round(cost, 10)
    usage["unpriced_models"] = unpriced


def fill_result_record(
    record: dict[str, Any], artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    artifact_name = record.get("artifact")
    if artifact_name not in artifacts:
        raise KeyError(f"No current artifact matches {artifact_name!r}.")
    artifact = artifacts[artifact_name]
    test = artifact["test"]
    if (record.get("model") or {}).get("name") != artifact["system_name"]:
        raise ValueError(f"Dynamics-name mismatch for {artifact_name}.")
    if (
        not vectors_match(record.get("initial_condition"), test["initial_condition"])
        or not vectors_match(record.get("target"), test["target"])
        or record.get("scenario_level") != test["scenario_level"]
    ):
        raise ValueError(f"Test conditions changed for {artifact_name}; rerun its benchmark first.")
    updated = copy.deepcopy(record)
    updated["total_complexity"] = artifact.get("total_complexity")
    metadata = artifact["metadata"]
    bound_scale = calculate_state_bound_scale(metadata.get("state_bounds"), metadata.get("n_states"))
    updated["success_rate_bound_scale"] = bound_scale
    rmse = ((updated.get("verification") or {}).get("metrics") or {}).get("rmse")
    if rmse is None:
        updated["success_rate"] = None
    else:
        updated["success_rate"] = calculate_success_rate(rmse, bound_scale)
    reprice_token_usage(updated)
    return updated


def create_full_results(
    input_path: Path, artifacts: dict[str, dict[str, Any]], output_path: Path | None = None,
) -> Path:
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(f"{input_path} must contain a JSON array.")
    updated = [fill_result_record(record, artifacts) for record in records]
    destination = output_path or input_path.with_name(f"{input_path.stem}_full.json")
    if destination.resolve() == input_path.resolve():
        raise ValueError("The full report must have a different filename from the base report.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(updated, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    pending = sum(record["total_complexity"] is None for record in updated)
    if pending:
        print(f"{pending} manual complexity value(s) still null; fill them in artifacts and finalize again.")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize manually rated MPC benchmark reports.")
    parser.add_argument("results_json", nargs="*", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    paths = select_result_paths(args.results_json)
    if not paths:
        raise FileNotFoundError(
            "No base reports exist for the selected models. Check USE_MOCK_RESULTS and "
            "MODEL_FOLDERS_TO_UPDATE, or run mpc_benchmark.py for those models first."
        )
    if args.output is not None and len(paths) != 1:
        raise ValueError("--output requires exactly one input file.")
    artifacts = load_artifacts(args.artifacts_dir)
    for path in paths:
        output = create_full_results(path, artifacts, args.output)
        print(f"Base report preserved: {path}")
        print(f"Full report written: {output}")


if __name__ == "__main__":
    main()
