"""Turn saved Adaptive full benchmark JSON into brief results with success rates.

No tuning, simulation, or LLM API calls happen here. The full report is kept.
The score uses the final verification's *full-window* output tracking MSE:
    success_rate = exp(-max_i(RMSE_i / task_scale_i))
where task_scale_i is the RMS error of holding output i at its initial value.
The worst normalized output is used for multi-output systems. This is a
continuous score in [0, 1], distinct from verification.success (a pass/fail
check of steady-state error and numerical health). It is not directly the
same normalization used by the MPC benchmark, whose artifacts have bounds.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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

from benchmarks.Adaptive.result_brief import build_brief_result

# --------------------------- FINALIZATION SETTINGS -------------------------
USE_MOCK_RESULTS = False  # True ignores the model list and finalizes only mock.
MODEL_FOLDERS_TO_UPDATE = [
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-4.1-nano",
    "groq-gpt-oss-120b",
    "gpt-4o-mini",
]
INPUT_FILENAME = "batch_tuning_results.json"
# ---------------------------------------------------------------------------


def selected_paths(explicit_paths: list[Path]) -> list[Path]:
    if USE_MOCK_RESULTS:
        folders = ["mock"]
    elif explicit_paths:
        return explicit_paths
    else:
        folders = MODEL_FOLDERS_TO_UPDATE
    paths = []
    for folder in dict.fromkeys(folders):
        if (not isinstance(folder, str) or not folder or folder in (".", "..")
                or any(char in folder for char in '/\\:<>"|?*')):
            raise ValueError(f"Unsafe model folder name: {folder!r}")
        path = RESULTS_DIR / folder / INPUT_FILENAME
        if path.is_file():
            paths.append(path)
        else:
            print(f"Skipping missing report: {path}")
    return paths


def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> dict[str, dict[str, Any]]:
    artifacts = {}
    for path in sorted(artifacts_dir.glob("*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        complexity = artifact.get("total_complexity")
        if (isinstance(complexity, bool) or not isinstance(complexity, int)
                or not 1 <= complexity <= 10):
            raise ValueError(f"{path.name}: total_complexity must be an integer from 1 to 10.")
        artifacts[path.name] = artifact
    if not artifacts:
        raise FileNotFoundError(f"No Adaptive artifacts in {artifacts_dir}")
    return artifacts


def calculate_success_rate(verification: Any) -> float | None:
    """Return a fraction 0..1; return None when verification data is absent."""
    if not isinstance(verification, dict):
        return None
    if verification.get("numerically_healthy") is False:
        return 0.0
    checks = verification.get("success_checks") or {}
    if checks.get("finite") is False or checks.get("bounded") is False:
        return 0.0
    mse = (verification.get("tracking_mse") or {}).get("full")
    scale = verification.get("task_scale")
    trivial = verification.get("task_trivial")
    if (not isinstance(mse, list) or not isinstance(scale, list)
            or not mse or len(mse) != len(scale)):
        return None
    if not isinstance(trivial, list) or len(trivial) != len(mse):
        trivial = [False] * len(mse)

    normalized_errors = []
    for mse_i, scale_i, trivial_i in zip(mse, scale, trivial):
        try:
            mse_value = float(mse_i)
            scale_value = float(scale_i)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(mse_value) or mse_value < 0:
            return None
        # Follow Adaptive's own tracking-score convention for trivial tasks.
        if trivial_i:
            scale_value = 1e-6
        if not math.isfinite(scale_value) or scale_value <= 0:
            return None
        normalized_errors.append(math.sqrt(mse_value) / scale_value)
    return round(math.exp(-max(normalized_errors)), 6)


def _same_test(saved: dict[str, Any], current: dict[str, Any]) -> bool:
    """Prevent attaching a current complexity to results from an edited test."""
    if saved.get("test_id") != current.get("test_id"):
        return False
    for key in ("initial_condition", "references", "uncertainty", "disturbance"):
        if saved.get(key) != current.get(key):
            return False
    return True


def add_estimated_cost(brief: dict[str, Any]) -> None:
    """Price real-agent token totals using the repository's local price table."""
    counts = brief["token_counts"]
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
    for role, field in (("agent", "designer"), ("tuner", "tuner"),
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


def make_brief_record(
    record: dict[str, Any], artifacts: dict[str, dict[str, Any]],
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

    brief = build_brief_result(record)
    brief["llm_model"] = record.get("llm_model")
    brief["total_complexity"] = artifact["total_complexity"]
    brief["success_rate"] = calculate_success_rate(record.get("verification"))
    add_estimated_cost(brief)
    if record.get("error"):
        brief["error"] = copy.deepcopy(record["error"])
    # If an error occurred before spec creation, still identify the saved test.
    if not brief["dynamics"]["initial_condition"]:
        brief["dynamics"]["initial_condition"] = matches[0]["initial_condition"]
    if not brief["dynamics"]["references"]:
        brief["dynamics"]["references"] = matches[0]["references"]
    return brief


def create_brief_results(
    input_path: Path, artifacts: dict[str, dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    saved = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(saved, list):
        raise TypeError(f"{input_path} must contain an Adaptive benchmark JSON array.")
    briefs = [make_brief_record(record, artifacts) for record in saved]
    destination = output_path or input_path.with_name(f"{input_path.stem}_brief.json")
    if destination.resolve() == input_path.resolve():
        raise ValueError("Brief and full result paths must be different.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(briefs, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize Adaptive benchmark reports.")
    parser.add_argument("results_json", nargs="*", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    paths = selected_paths(args.results_json)
    if not paths:
        raise FileNotFoundError(
            "No selected full benchmark report exists. Run adaptive_benchmark.py first "
            "or change USE_MOCK_RESULTS/MODEL_FOLDERS_TO_UPDATE."
        )
    if args.output is not None and len(paths) != 1:
        raise ValueError("--output requires exactly one input report.")
    artifacts = load_artifacts(args.artifacts_dir)
    for path in paths:
        output = create_brief_results(path, artifacts, args.output)
        print(f"Full report preserved: {path}")
        print(f"Brief report written: {output}")


if __name__ == "__main__":
    main()
