"""In-memory job store unit tests."""

from __future__ import annotations

from backend_api.AgentMPC.job_store import InMemoryJobStore


def test_create_get_update_list():
    store = InMemoryJobStore()
    rec = store.create(
        dynamics_ref={"plugin_id": "example_pendulum"},
        options={"max_iterations": 5},
        user_id=1,
        project_id="p1",
        system_name="pendulum",
    )
    assert rec.job_id
    assert rec.status == "queued"
    assert store.get(rec.job_id).system_name == "pendulum"

    store.update(rec.job_id, status="running", stage="actor", iteration=2)
    got = store.get(rec.job_id)
    assert got.status == "running"
    assert got.iteration == 2

    store.append_progress(rec.job_id, {"kind": "stage_start", "stage": "actor"})
    got = store.get(rec.job_id)
    assert len(got.progress) == 1

    listed = store.list_jobs(user_id=1)
    assert len(listed) == 1
    assert store.list_jobs(user_id=99) == []


def test_cancel_flag():
    store = InMemoryJobStore()
    rec = store.create(dynamics_ref=None, options={})
    assert not store.is_cancel_requested(rec.job_id)
    store.request_cancel(rec.job_id)
    assert store.is_cancel_requested(rec.job_id)
