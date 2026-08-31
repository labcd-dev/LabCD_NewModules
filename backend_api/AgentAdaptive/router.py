"""AgentAdaptive job routes.

Standalone NewModules mode uses an in-memory job store. When merging into
LabCD_Application, swap for the platform ``job_store`` and re-attach auth /
project dependencies. Module name for a shared job router: ``adaptive``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend_api.AgentAdaptive.job_store import InMemoryJobStore, default_job_store
from backend_api.AgentAdaptive.schemas import (
    AdaptiveClarifyRequest,
    AdaptiveClarifyResponse,
    AdaptiveJobCreateRequest,
    AdaptiveJobCreateResponse,
    AdaptiveJobResultsResponse,
    AdaptiveJobStatusResponse,
    AdaptiveJobSummary,
)
from backend_api.AgentAdaptive.service import (
    cancel_job,
    clarify_job,
    get_job,
    get_results,
    list_jobs,
    submit_job,
)

router = APIRouter(prefix="/adaptive", tags=["adaptive"])


def _store() -> InMemoryJobStore:
    return default_job_store


@router.post(
    "/jobs",
    response_model=AdaptiveJobCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_adaptive_job(request: AdaptiveJobCreateRequest) -> AdaptiveJobCreateResponse:
    return submit_job(request, store=_store())


@router.get("/jobs", response_model=list[AdaptiveJobSummary])
def list_adaptive_jobs(user_id: int | None = None) -> list[AdaptiveJobSummary]:
    return list_jobs(user_id=user_id, store=_store())


@router.get("/jobs/{job_id}", response_model=AdaptiveJobStatusResponse)
def get_adaptive_job(job_id: str) -> AdaptiveJobStatusResponse:
    try:
        return get_job(job_id, store=_store())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@router.post("/jobs/{job_id}/clarify", response_model=AdaptiveClarifyResponse)
def clarify_adaptive_job(
    job_id: str,
    request: AdaptiveClarifyRequest,
) -> AdaptiveClarifyResponse:
    try:
        return clarify_job(job_id, request, store=_store())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@router.post("/jobs/{job_id}/cancel", response_model=AdaptiveJobStatusResponse)
def cancel_adaptive_job(job_id: str) -> AdaptiveJobStatusResponse:
    try:
        return cancel_job(job_id, store=_store())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@router.get("/jobs/{job_id}/results", response_model=AdaptiveJobResultsResponse)
def get_adaptive_job_results(job_id: str) -> AdaptiveJobResultsResponse:
    try:
        return get_results(job_id, store=_store())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
