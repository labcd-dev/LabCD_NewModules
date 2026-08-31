"""HTTP route smoke tests with TestClient (mocked service pipeline)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend_api.AgentMPC.app import app
from backend_api.AgentMPC.job_store import InMemoryJobStore
from backend_api.AgentMPC.schemas import (
    MPCJobCreateResponse,
    MPCJobResultsResponse,
    MPCJobStatusResponse,
)

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "agent-mpc"


def test_jobs_flow_mocked():
    store = InMemoryJobStore()
    now = datetime.now(timezone.utc)

    def fake_submit(request, store=None):
        s = store or InMemoryJobStore()
        rec = s.create(
            dynamics_ref={"plugin_id": "example_pendulum"},
            options={"max_iterations": 3},
            system_name="test",
        )
        s.update(
            rec.job_id,
            status="completed",
            stage="done",
            message="ok",
            best_params={"Np": 8},
            best_mse=0.02,
            iteration=2,
            termination_reason="done",
        )
        return MPCJobCreateResponse(
            job_id=rec.job_id, status="completed", stage="done", message="ok"
        )

    with patch("backend_api.AgentMPC.router._store", return_value=store):
        with patch("backend_api.AgentMPC.router.submit_job", side_effect=fake_submit):
            r = client.post(
                "/api/mpc/jobs",
                json={
                    "dynamics": {"plugin_id": "example_pendulum"},
                    "options": {"max_iterations": 3},
                },
            )
        assert r.status_code == 201
        body = r.json()
        job_id = body["job_id"]

        with patch(
            "backend_api.AgentMPC.router.get_job",
            return_value=MPCJobStatusResponse(
                job_id=job_id,
                status="completed",
                stage="done",
                message="ok",
                iteration=2,
                max_iterations=3,
                progress=[],
                created_at=now,
                updated_at=now,
                system_name="test",
            ),
        ):
            r2 = client.get(f"/api/mpc/jobs/{job_id}")
        assert r2.status_code == 200
        assert r2.json()["status"] == "completed"

        with patch(
            "backend_api.AgentMPC.router.get_results",
            return_value=MPCJobResultsResponse(
                job_id=job_id,
                status="completed",
                stage="done",
                best_params={"Np": 8},
                best_mse=0.02,
                iteration=2,
                termination_reason="done",
            ),
        ):
            r3 = client.get(f"/api/mpc/jobs/{job_id}/results")
        assert r3.status_code == 200
        assert r3.json()["best_params"]["Np"] == 8


def test_job_not_found():
    r = client.get("/api/mpc/jobs/nonexistent")
    assert r.status_code == 404
