"""HTTP-level tests for AgentAdaptive routes (mocked core, no live LLM)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend_api.AgentAdaptive.app import app
from backend_api.AgentAdaptive.job_store import InMemoryJobStore


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


@pytest.fixture
def client():
    store = InMemoryJobStore()
    with patch("backend_api.AgentAdaptive.router.default_job_store", store):
        with patch("backend_api.AgentAdaptive.service.default_job_store", store):
            yield TestClient(app), store


def test_health(client):
    http, _store = client
    r = http.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "agent-adaptive"


def test_create_skip_clarify_and_results(client):
    http, store = client
    fake_result = {
        "messages": [SimpleNamespace(content="Final report")],
        "abstract": "Abstract line.",
        "final_metrics": {"success": True},
    }

    with patch(
        "backend_api.AgentAdaptive.service.run_full_pipeline",
        return_value=(fake_result, {"total": {}}, [], None),
    ):
        with patch(
            "backend_api.AgentAdaptive.service._start_pipeline_async",
            side_effect=lambda job_id, s: __import__(
                "backend_api.AgentAdaptive.service",
                fromlist=["_run_pipeline_thread"],
            )._run_pipeline_thread(job_id, s),
        ):
            r = http.post(
                "/api/adaptive/jobs",
                json={
                    "system_spec": _minimal_spec(),
                    "options": {"skip_clarify": True, "enable_tuning": False},
                },
            )
    assert r.status_code == 201, r.text
    data = r.json()
    job_id = data["job_id"]

    status = http.get(f"/api/adaptive/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"

    results = http.get(f"/api/adaptive/jobs/{job_id}/results")
    assert results.status_code == 200
    body = results.json()
    assert body["report"] == "Final report"
    assert body["abstract"] == "Abstract line."

    listed = http.get("/api/adaptive/jobs")
    assert listed.status_code == 200
    assert any(item["job_id"] == job_id for item in listed.json())


def test_clarify_flow(client):
    http, _store = client
    call_n = {"n": 0}

    def fake_turn(messages, on_event=None, round_num=1, force_finish=False, _nudged=False):
        call_n["n"] += 1
        usage = {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "cached_input_tokens": 0,
        }
        if call_n["n"] == 1:
            return (
                "continue",
                "Any uncertainty?",
                None,
                usage,
                "",
                list(messages) + [{"role": "assistant", "content": "q"}],
            )
        return (
            "complete",
            "Done clarifying.",
            {"uncertainty": [], "disturbance": [], "references": []},
            usage,
            "",
            list(messages) + [{"role": "assistant", "content": "done"}],
        )

    fake_result = {
        "messages": [SimpleNamespace(content="report")],
        "abstract": None,
        "final_metrics": {},
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
                    created = http.post(
                        "/api/adaptive/jobs",
                        json={"system_spec": _minimal_spec()},
                    )
                    assert created.status_code == 201
                    job_id = created.json()["job_id"]
                    assert created.json()["status"] == "clarifying"

                    clarified = http.post(
                        f"/api/adaptive/jobs/{job_id}/clarify",
                        json={"answer": "no"},
                    )
                    assert clarified.status_code == 200
                    assert clarified.json()["clarifier_status"] == "complete"

    assert http.get(f"/api/adaptive/jobs/{job_id}").json()["status"] == "completed"


def test_cancel(client):
    http, _store = client

    def fake_turn(messages, on_event=None, round_num=1, force_finish=False, _nudged=False):
        return (
            "continue",
            "q?",
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
            created = http.post(
                "/api/adaptive/jobs",
                json={"system_spec": _minimal_spec()},
            )
    job_id = created.json()["job_id"]
    cancelled = http.post(f"/api/adaptive/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"


def test_job_not_found(client):
    http, _store = client
    assert http.get("/api/adaptive/jobs/missing").status_code == 404
    assert http.get("/api/adaptive/jobs/missing/results").status_code == 404
    assert (
        http.post("/api/adaptive/jobs/missing/clarify", json={"answer": "x"}).status_code
        == 404
    )
    assert http.post("/api/adaptive/jobs/missing/cancel").status_code == 404
