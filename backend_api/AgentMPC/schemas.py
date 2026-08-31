"""AgentMPC job API schemas.

Job-oriented contracts for long-running MPC auto-tuning runs (multi-agent
graph over dynamics plugin + MPC solver). Shapes align with LabCD_Application
job status / artefact patterns and with ``backend_api/AgentAdaptive``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


JobStatus = Literal[
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
]

JobStage = Literal[
    "queued",
    "scenarist",
    "actor",
    "evaluator",
    "terminator",
    "critic",
    "juror",
    "done",
    "error",
]


class MPCJobOptions(BaseModel):
    """Knobs for an MPC tuning run (mirrors Streamlit / run_agents options)."""

    max_iterations: int = Field(default=15, ge=1, le=100)
    prediction_horizon: int = Field(default=12, ge=1, le=200)
    control_horizon: int = Field(default=4, ge=1, le=100)
    dt_mpc: float = Field(default=0.02, gt=0)
    simulation_time: float = Field(default=3.0, gt=0)
    ui_scenario_level: int = Field(default=1, ge=1, le=3)
    user_guidance: str = ""
    min_explore_iterations: int = Field(default=4, ge=0, le=50)
    exploration_intensity: int = Field(default=50, ge=1, le=100)
    # When True, use UI graph (no Scenarist); scenario comes from ui_scenario_level.
    use_ui_graph: bool = True
    # Optional seed params for entry at evaluator (Np, Nc, Q, R, P).
    seed_params: dict[str, Any] | None = None
    # LLM model name (passed through when configure_llm is set by caller).
    model: str | None = None
    system_name: str = "mpc_system"


class MPCDynamicsInput(BaseModel):
    """Dynamics plugin reference.

    Provide **one** of:
    - ``plugin_path``: path to a .py plugin (relative to repo or absolute)
    - ``plugin_id``: short id of a bundled plugin under
      ``backend_core/AgentMPC/dynamics/plugins/`` (e.g. ``example_pendulum``)
    - ``source``: full Python source of a plugin (written to a temp file)
    """

    plugin_path: str | None = None
    plugin_id: str | None = None
    source: str | None = None


class MPCJobCreateRequest(BaseModel):
    """Start an MPC tuning job."""

    dynamics: MPCDynamicsInput | None = None
    options: MPCJobOptions = Field(default_factory=MPCJobOptions)
    user_id: int | None = None
    project_id: str | None = None


class MPCJobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage
    message: str = ""


class MPCJobProgressEvent(BaseModel):
    kind: str = ""
    stage: str = ""
    text: str = ""
    round: int | None = None
    ts: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class MPCJobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage
    message: str = ""
    error: str | None = None
    iteration: int = 0
    max_iterations: int = 0
    progress: list[MPCJobProgressEvent] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    user_id: int | None = None
    project_id: str | None = None
    options: MPCJobOptions | None = None
    system_name: str | None = None


class MPCJobResultsResponse(BaseModel):
    """Final job artefacts. Fields are intentionally permissive: the LangGraph
    state mixes str logs, dicts, and numpy scalars."""

    job_id: str
    status: JobStatus
    stage: JobStage
    best_params: Any = None
    best_mse: float | None = None
    iteration: int = 0
    termination_reason: str | None = None
    mse_history: list[Any] = Field(default_factory=list)
    overshoot_history: list[Any] = Field(default_factory=list)
    settling_history: list[Any] = Field(default_factory=list)
    effort_history: list[Any] = Field(default_factory=list)
    params_history: list[Any] = Field(default_factory=list)
    # Core agents append plain log strings (List[str]); also accept dict entries.
    history: list[Any] = Field(default_factory=list)
    report: str | None = None
    export_script: str | None = None
    metrics: Any = None
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class MPCJobSummary(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage
    system_name: str | None = None
    created_at: datetime
    updated_at: datetime
    user_id: int | None = None
