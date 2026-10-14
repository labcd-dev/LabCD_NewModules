"""Benchmark-local mock Tuner and offline Adaptive pipeline adapter.

The normal core modules are never edited. During this function call only,
their agent entry points are routed to deterministic mock functions while the
real numerical controller design, simulation, metrics, and scoring are reused.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from threading import RLock

from backend_core.AgentAdaptive.agents import tuner_agent as core_tuner
from backend_core.AgentAdaptive.agents.agent_io import _SyntheticMessage

from .designer_agent import run_extraction
from .mock_random import (
    MOCK_COST_USD_PER_TOKEN,
    MockAgent,
    MockSession,
    current_session,
    mock_session,
    resolved_models,
)
from .report_writer import write_abstract


_PATCH_LOCK = RLock()


class MockTuner(MockAgent):
    def payload(self):
        current = self.context["current_tuning"]
        allowed = self.context.get("allowed_params")
        candidates = (
            ["c_gains"]
            if self.context["method"] == "backstepping"
            else ["K", "Lam", "surface_lambda"]
        )
        candidates = [
            key
            for key in candidates
            if current.get(key) is not None
            and (allowed is None or key in allowed)
        ]
        proposal = {
            "reasoning": (
                "MOCK Tuner made a seeded gain adjustment to exercise the "
                "tuning loop. Real simulation metrics decide whether it improves."
            ),
            "diagnosed_symptom": "slow_transient",
        }
        for key in self.rng.sample(candidates, min(len(candidates), 2)):
            factor = self.rng.uniform(0.85, 1.20)
            value = current[key]
            if isinstance(value, (list, tuple)):
                proposal[key] = [round(float(item) * factor, 8) for item in value]
            elif math.isfinite(float(value)) and float(value) > 0:
                proposal[key] = round(float(value) * factor, 8)
        return proposal


def run_tuner_round(
    fixed_args,
    current_tuning,
    session,
    prompt,
    structure_cache=None,
    allowed_params=None,
    round_num=None,
):
    del prompt, round_num
    mocks = current_session() or MockSession(42)
    response = MockTuner(
        mocks,
        "tuner",
        {
            "current_tuning": current_tuning,
            "method": fixed_args.get("method", "smc"),
            "allowed_params": allowed_params,
        },
    ).invoke([])
    payload = mocks.messages[-1]["response"]
    reasoning, symptom, overrides = core_tuner._coerce_tuning_payload(payload)
    report = core_tuner._apply_tuning_proposal(
        reasoning,
        symptom,
        overrides,
        fixed_args,
        current_tuning,
        session,
        structure_cache=structure_cache,
        allowed_params=allowed_params,
    )
    if report.startswith("DESIGN FAILED") or report.startswith("NO CHANGE APPLIED"):
        return {"messages": [response, _SyntheticMessage(report)]}
    return {"messages": [response]}


@contextmanager
def _route_core_to_mocks():
    replacements = {
        "run_extraction": run_extraction,
        "run_tuner_round": run_tuner_round,
        "write_abstract": write_abstract,
        "resolved_models": resolved_models,
    }
    with _PATCH_LOCK:
        previous = {name: getattr(core_tuner, name) for name in replacements}
        try:
            for name, replacement in replacements.items():
                setattr(core_tuner, name, replacement)
            yield
        finally:
            for name, original in previous.items():
                setattr(core_tuner, name, original)


def run_full_pipeline(*args, mock_seed=42, **kwargs):
    """Run mock agents and real numerical simulations with no API calls."""
    with mock_session(seed=mock_seed) as session, _route_core_to_mocks():
        result, usage, tuning_log, best = core_tuner.run_full_pipeline(*args, **kwargs)
    result["agent_mode"] = "mock"
    result["mock_agent_messages"] = session.messages
    total_tokens = (usage.get("total") or {}).get("total_tokens", 0)
    usage.update(
        {
            "simulated": True,
            "mock_seed": mock_seed,
            "price_usd_per_token": MOCK_COST_USD_PER_TOKEN,
            "cost_usd": round(total_tokens * MOCK_COST_USD_PER_TOKEN, 6),
            "actual_cost_usd": 0.0,
        }
    )
    return result, usage, tuning_log, best
