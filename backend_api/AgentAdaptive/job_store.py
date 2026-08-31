"""In-memory job store for standalone AgentAdaptive runs.

Mirrors the role of LabCD_Application ``job_store`` for NewModules demos.
Replace with the platform DB-backed store on merge; keep the record fields
stable so routers and services stay thin.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

JobStatus = Literal[
    "queued",
    "clarifying",
    "designing",
    "building",
    "tuning",
    "reporting",
    "completed",
    "failed",
    "cancelled",
]

JobStage = Literal[
    "queued",
    "clarify",
    "design",
    "build",
    "tune",
    "report",
    "done",
    "error",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid4().hex[:12]


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus = "queued"
    stage: JobStage = "queued"
    message: str = ""
    error: str | None = None
    system_spec: dict[str, Any] | None = None
    options: dict[str, Any] = field(default_factory=dict)
    user_id: int | None = None
    project_id: str | None = None
    # Clarifier conversation state
    clarify_messages: list[dict[str, str]] = field(default_factory=list)
    clarify_chat_log: list[dict[str, str]] = field(default_factory=list)
    clarify_round: int = 0
    last_clarifier_reply: str | None = None
    clarification_record: list[dict[str, Any]] = field(default_factory=list)
    clarifier_usage: dict[str, Any] = field(default_factory=dict)
    # Progress + cancel
    progress: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False
    # Results
    abstract: str | None = None
    report: str | None = None
    method: str | None = None
    final_metrics: dict[str, Any] | None = None
    tuning_log: list[dict[str, Any]] = field(default_factory=list)
    tuning_best: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    series: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


class InMemoryJobStore:
    """Thread-safe in-memory job registry."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._jobs: dict[str, JobRecord] = {}

    def create(
        self,
        *,
        system_spec: dict[str, Any] | None,
        options: dict[str, Any],
        user_id: int | None = None,
        project_id: str | None = None,
    ) -> JobRecord:
        with self._lock:
            job_id = _new_id()
            while job_id in self._jobs:
                job_id = _new_id()
            record = JobRecord(
                job_id=job_id,
                system_spec=deepcopy(system_spec) if system_spec else None,
                options=dict(options or {}),
                user_id=user_id,
                project_id=project_id,
            )
            self._jobs[job_id] = record
            return deepcopy(record)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return deepcopy(record) if record is not None else None

    def list_jobs(self, user_id: int | None = None) -> list[JobRecord]:
        with self._lock:
            records = list(self._jobs.values())
        if user_id is not None:
            records = [r for r in records if r.user_id == user_id]
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return [deepcopy(r) for r in records]

    def update(self, job_id: str, **fields: Any) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            for key, value in fields.items():
                if not hasattr(record, key):
                    raise AttributeError(f"JobRecord has no field {key!r}")
                setattr(record, key, value)
            record.updated_at = _now()
            return deepcopy(record)

    def append_progress(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.progress.append(dict(event))
            record.updated_at = _now()

    def request_cancel(self, job_id: str) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            record.cancel_requested = True
            record.updated_at = _now()
            return deepcopy(record)

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            record = self._jobs.get(job_id)
            return bool(record and record.cancel_requested)


# Process-wide default store for the standalone app.
default_job_store = InMemoryJobStore()
