"""Service adapter tests (mocked Clarifier + pipeline, no live LLM)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from backend_api.AgentAdaptive.job_store import InMemoryJobStore
from backend_api.AgentAdaptive.schemas import (
    AdaptiveClarifyRequest,
    AdaptiveJobCreateRequest,
    AdaptiveJobOptions,
)
from backend_api.AgentAdaptive.service import (
    cancel_job,
    clarify_job,
    get_job,
    get_results,
    list_jobs,
    submit_job,
)


def _minimal_spec():
    return {
        "status": "complete",
        "system_name": "simple_integrator",
        "dynamics": {
            "states": ["x"],
            "state_meanings": ["state"],
            "inputs": ["u"],
            "outputs": ["x"],
            "state_equations": ["u"],
            "parameters": {},
            "system_type": "SISO",
            "assumptions": [],
            "x0": [0.0],
            "references": [],
            "uncertainty": [],
            "disturbance": [],
            "sim_time": 5.0,
            "solver_step": 0.01,
        },
    }


def test_submit_skip_clarify_runs_pipeline():
    store = InMemoryJobStore()
    fake_result = {
        "messages": [SimpleNamespace(content="Design report text")],
        "abstract": "Short abstract.",
        "final_metrics": {"success": True, "tracking_pct_headline": 99.0},
    }
    fake_usage = {"total": {"total_tokens": 10}}

    with patch(
        "backend_api.AgentAdaptive.service.run_full_pipeline",
        return_value=(fake_result, fake_usage, [], None),
    ) as mock_pipe:
        # Run pipeline synchronously by invoking the thread target inline
        with patch(
            "backend_api.AgentAdaptive.service._start_pipeline_async",
            side_effect=lambda job_id, s: __import__(
                "backend_api.AgentAdaptive.service", fromlist=["_run_pipeline_thread"]
            )._run_pipeline_thread(job_id, s),
        ):
            resp = submit_job(
                AdaptiveJobCreateRequest(
                    system_spec=_minimal_spec(),
                    options=AdaptiveJobOptions(skip_clarify=True, enable_tuning=False),
                ),
                store=store,
            )

    assert resp.job_id
    assert resp.status in ("designing", "completed")
    mock_pipe.assert_called_once()

    status = get_job(resp.job_id, store=store)
    assert status.status == "completed"
    assert status.stage == "done"

    results = get_results(resp.job_id, store=store)
    assert results.report == "Design report text"
    assert results.abstract == "Short abstract."
    assert results.final_metrics is not None
    assert results.final_metrics["success"] is True


def test_submit_starts_clarifier():
    store = InMemoryJobStore()

    def fake_turn(messages, on_event=None, round_num=1, force_finish=False, _nudged=False):
        return (
            "continue",
            "Is there model uncertainty?",
            None,
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "cached_input_tokens": 0},
            "",
            list(messages) + [{"role": "assistant", "content": "{}"}],
        )

    with patch(
        "backend_api.AgentAdaptive.service.clarifier.start_conversation",
        return_value=[{"role": "user", "content": "plant"}],
    ):
        with patch(
            "backend_api.AgentAdaptive.service.clarifier.run_clarifier_turn",
            side_effect=fake_turn,
        ):
            resp = submit_job(
                AdaptiveJobCreateRequest(
                    system_spec=_minimal_spec(),
                    options=AdaptiveJobOptions(skip_clarify=False),
                ),
                store=store,
            )

    assert resp.status == "clarifying"
    status = get_job(resp.job_id, store=store)
    assert status.clarify_pending is True
    assert status.last_clarifier_reply == "Is there model uncertainty?"


def test_clarify_complete_starts_pipeline():
    store = InMemoryJobStore()

    turns = {
        "first": (
            "continue",
            "Any disturbance?",
            None,
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "cached_input_tokens": 0},
            "",
        ),
        "second": (
            "complete",
            "No uncertainty or disturbance.",
            {"uncertainty": [], "disturbance": [], "references": []},
            {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4, "cached_input_tokens": 0},
            "",
        ),
    }
    call_n = {"n": 0}

    def fake_turn(messages, on_event=None, round_num=1, force_finish=False, _nudged=False):
        call_n["n"] += 1
        key = "first" if call_n["n"] == 1 else "second"
        status, reply, dynamics, usage, error = turns[key]
        updated = list(messages) + [{"role": "assistant", "content": reply}]
        return status, reply, dynamics, usage, error, updated

    fake_result = {
        "messages": [SimpleNamespace(content="ok report")],
        "abstract": None,
        "final_metrics": {"success": True},
    }

    with patch(
        "backend_api.AgentAdaptive.service.clarifier.start_conversation",
        return_value=[{"role": "user", "content": "plant"}],
    ):
        with patch(
            "backend_api.AgentAdaptive.service.clarifier.run_clarifier_turn",
            side_effect=fake_turn,
        ):
            with patch(
                "backend_api.AgentAdaptive.service.run_full_pipeline",
                return_value=(fake_result, {}, [], None),
            ):
                with patch(
                    "backend_api.AgentAdaptive.service._start_pipeline_async",
                    side_effect=lambda job_id, s: __import__(
                        "backend_api.AgentAdaptive.service",
                        fromlist=["_run_pipeline_thread"],
                    )._run_pipeline_thread(job_id, s),
                ):
                    created = submit_job(
                        AdaptiveJobCreateRequest(system_spec=_minimal_spec()),
                        store=store,
                    )
                    clarified = clarify_job(
                        created.job_id,
                        AdaptiveClarifyRequest(answer="none"),
                        store=store,
                    )

    assert clarified.clarifier_status == "complete"
    status = get_job(created.job_id, store=store)
    assert status.status == "completed"
    results = get_results(created.job_id, store=store)
    assert results.report == "ok report"


def test_cancel_while_clarifying():
    store = InMemoryJobStore()

    def fake_turn(messages, on_event=None, round_num=1, force_finish=False, _nudged=False):
        return (
            "continue",
            "question?",
            None,
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cached_input_tokens": 0},
            "",
            list(messages),
        )

    with patch(
        "backend_api.AgentAdaptive.service.clarifier.start_conversation",
        return_value=[{"role": "user", "content": "p"}],
    ):
        with patch(
            "backend_api.AgentAdaptive.service.clarifier.run_clarifier_turn",
            side_effect=fake_turn,
        ):
            created = submit_job(
                AdaptiveJobCreateRequest(system_spec=_minimal_spec()),
                store=store,
            )

    cancelled = cancel_job(created.job_id, store=store)
    assert cancelled.status == "cancelled"


def test_list_jobs():
    store = InMemoryJobStore()
    with patch(
        "backend_api.AgentAdaptive.service._start_pipeline_async",
        lambda *a, **k: None,
    ):
        submit_job(
            AdaptiveJobCreateRequest(
                system_spec=_minimal_spec(),
                options=AdaptiveJobOptions(skip_clarify=True),
                user_id=9,
            ),
            store=store,
        )
    items = list_jobs(user_id=9, store=store)
    assert len(items) == 1
    assert items[0].system_name == "simple_integrator"
