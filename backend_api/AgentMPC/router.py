"""AgentMPC job routes.

Standalone NewModules mode uses an in-memory job store. When merging into
LabCD_Application, swap for the platform ``job_store`` and re-attach auth /
project dependencies. Module name for a shared job router: ``mpc``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend_api.AgentMPC.job_store import InMemoryJobStore, default_job_store
from backend_api.AgentMPC.schemas import (
    MPCJobCreateRequest,
    MPCJobCreateResponse,
    MPCJobResultsResponse,
    MPCJobStatusResponse,
    MPCJobSummary,
)
from backend_api.AgentMPC.service import (
    cancel_job,
    get_job,
    get_results,
    list_jobs,
    submit_job,
)

router = APIRouter(prefix="/mpc", tags=["mpc"])


def _store() -> InMemoryJobStore:
    return default_job_store


@router.post(
    "/jobs",
    response_model=MPCJobCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_mpc_job(request: MPCJobCreateRequest) -> MPCJobCreateResponse:
    try:
        return submit_job(request, store=_store())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs", response_model=list[MPCJobSummary])
def list_mpc_jobs(user_id: int | None = None) -> list[MPCJobSummary]:
    return list_jobs(user_id=user_id, store=_store())


@router.get("/jobs/{job_id}", response_model=MPCJobStatusResponse)
def get_mpc_job(job_id: str) -> MPCJobStatusResponse:
    try:
        return get_job(job_id, store=_store())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@router.post("/jobs/{job_id}/cancel", response_model=MPCJobStatusResponse)
def cancel_mpc_job(job_id: str) -> MPCJobStatusResponse:
    try:
        return cancel_job(job_id, store=_store())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@router.get("/jobs/{job_id}/results", response_model=MPCJobResultsResponse)
def get_mpc_job_results(job_id: str) -> MPCJobResultsResponse:
    try:
        return get_results(job_id, store=_store())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
