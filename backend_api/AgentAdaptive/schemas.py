"""AgentAdaptive job API schemas.

Job-oriented contracts for long-running adaptive design runs (clarify →
design → build → tune → report). Shapes are intentionally close to
LabCD_Application job status / artefact patterns so a future merge stays clean.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


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


class AdaptiveJobOptions(BaseModel):
    """Knobs for a design run (mirrors Streamlit / run_full_pipeline options)."""

    enable_tuning: bool = False
    target_rms_frac: float = Field(default=0.02, gt=0, le=1.0)
    max_tuning_rounds: int = Field(default=4, ge=0, le=20)
    skip_clarify: bool = False
    model: str | None = None
    # Optional free-text description (legacy CLI); pipeline primarily uses system_spec.
    description: str = ""


class AdaptiveJobCreateRequest(BaseModel):
    """Start a design job from a system_spec (plant JSON + sim knobs).

    ``system_spec`` should match the shape produced by AgentPlant artifact
    adaptive-spec / Streamlit plant+knobs (``status`` + ``system_name`` +
    ``dynamics``). When omitted, the job fails at design with a clear error
    (same behaviour as ``run_full_pipeline`` without a spec).
    """

    system_spec: dict[str, Any] | None = None
    options: AdaptiveJobOptions = Field(default_factory=AdaptiveJobOptions)
    user_id: int | None = None
    project_id: str | None = None


class AdaptiveJobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage
    message: str = ""


class AdaptiveClarifyRequest(BaseModel):
    """Submit a clarification answer, or force-finish the clarifier."""

    answer: str = ""
    force_finish: bool = False


class AdaptiveClarifyResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage
    clarifier_status: Literal["continue", "complete", "error", "skipped"]
    reply: str = ""
    round: int = 0


class AdaptiveJobProgressEvent(BaseModel):
    kind: str = ""
    stage: str = ""
    text: str = ""
    round: int | None = None
    ts: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class AdaptiveJobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage
    message: str = ""
    error: str | None = None
    round: int = 0
    clarify_pending: bool = False
    last_clarifier_reply: str | None = None
    progress: list[AdaptiveJobProgressEvent] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    user_id: int | None = None
    project_id: str | None = None
    options: AdaptiveJobOptions | None = None


class AdaptiveJobResultsResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage
    abstract: str | None = None
    report: str | None = None
    method: str | None = None
    final_metrics: dict[str, Any] | None = None
    tuning_log: list[dict[str, Any]] = Field(default_factory=list)
    tuning_best: dict[str, Any] | None = None
    system_spec: dict[str, Any] | None = None
    clarification_record: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] | None = None
    series: dict[str, Any] | None = None
    error: str | None = None


class AdaptiveJobSummary(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage
    system_name: str | None = None
    created_at: datetime
    updated_at: datetime
    user_id: int | None = None
