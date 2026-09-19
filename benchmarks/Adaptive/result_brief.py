"""Create a compact JSON companion for an Adaptive tuning result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def build_brief_result(full_result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the useful final fields without histories or long agent text."""
    applied_spec = full_result.get("applied_system_spec") or {}
    dynamics = applied_spec.get("dynamics") or {}
    tuning = full_result.get("tuning") or {}
    usage = tuning.get("token_usage") or {}

    return {
        "artifact": full_result.get("artifact"),
        "plant": full_result.get("plant") or applied_spec.get("system_name"),
        "test_id": full_result.get("test_id"),
        "agent_mode": full_result.get("agent_mode"),
        "dynamics": {
            "system_name": applied_spec.get("system_name") or full_result.get("plant"),
            "system_type": dynamics.get("system_type"),
            "states": dynamics.get("states"),
            "state_meanings": dynamics.get("state_meanings"),
            "inputs": dynamics.get("inputs"),
            "outputs": dynamics.get("outputs"),
            "state_equations": dynamics.get("state_equations"),
            "parameters": dynamics.get("parameters"),
            "initial_condition": dynamics.get("x0"),
            "references": dynamics.get("references"),
            "uncertainty": dynamics.get("uncertainty"),
            "disturbance": dynamics.get("disturbance"),
            "simulation_time": dynamics.get("sim_time"),
            "solver_step": dynamics.get("solver_step"),
        },
        "tuning": {
            "best_round": tuning.get("best_round"),
            "best_params": tuning.get("best_params"),
        },
        "verification": full_result.get("verification"),
        "token_counts": {
            "models": usage.get("models"),
            "designer": usage.get("agent"),
            "tuner": usage.get("tuner"),
            "reporter": usage.get("reporter"),
            "clarifier": usage.get("clarifier"),
            "total": usage.get("total"),
            "simulated": usage.get("simulated", False),
            "mock_seed": usage.get("mock_seed"),
            "cost_usd": usage.get("cost_usd"),
            "actual_cost_usd": usage.get("actual_cost_usd"),
        },
        "pipeline_completed": full_result.get("pipeline_completed"),
    }


def write_brief_result(full_json_path: Path | str) -> Path:
    """Write ``<full-result-stem>_brief.json`` beside the full JSON file."""
    full_path = Path(full_json_path).resolve()
    if not full_path.is_file():
        raise FileNotFoundError(full_path)

    full_result = json.loads(full_path.read_text(encoding="utf-8"))
    brief_path = full_path.with_name(f"{full_path.stem}_brief.json")
    brief_path.write_text(
        json.dumps(
            build_brief_result(full_result),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return brief_path
