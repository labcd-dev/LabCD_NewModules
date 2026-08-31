"""Unit tests for the in-memory adaptive job store."""

from __future__ import annotations

from backend_api.AgentAdaptive.job_store import InMemoryJobStore


def test_create_get_list_update():
    store = InMemoryJobStore()
    rec = store.create(
        system_spec={"system_name": "demo", "dynamics": {"states": ["x"]}},
        options={"skip_clarify": True},
        user_id=3,
    )
    assert rec.job_id
    assert rec.status == "queued"

    got = store.get(rec.job_id)
    assert got is not None
    assert got.system_spec["system_name"] == "demo"

    listed = store.list_jobs(user_id=3)
    assert len(listed) == 1
    assert listed[0].job_id == rec.job_id

    updated = store.update(rec.job_id, status="designing", stage="design", message="go")
    assert updated is not None
    assert updated.status == "designing"
    assert updated.message == "go"


def test_cancel_and_progress():
    store = InMemoryJobStore()
    rec = store.create(system_spec=None, options={})
    store.append_progress(rec.job_id, {"kind": "note", "stage": "clarify", "text": "hi"})
    got = store.get(rec.job_id)
    assert got is not None
    assert len(got.progress) == 1

    store.request_cancel(rec.job_id)
    assert store.is_cancel_requested(rec.job_id) is True
