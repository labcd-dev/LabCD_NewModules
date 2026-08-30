"""Plant-model chat and artifact HTTP service adapters.

Chat adapter drives ``backend_core.AgentPlant.PlantModelAgent``.
Artifact helpers thin-wrap ``PlantCompiler`` + ``ArtifactStore`` so the unified
Plant → Pre-Launch → Artifact hand-off matches ``frontend_streamlit/unified_app``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from backend_core.AgentPlant import (
    PlantModelAgent,
    PlantModelSessionState as AgentSessionState,
    apply_session_state,
    export_session_state,
)
from backend_core.artifact_store import ArtifactStore
from backend_core.plant_compiler import PlantCompiler, validate_pre_launch
from backend_api.AgentPlant.schemas import (
    ArtifactCreateRequest,
    ArtifactCreateResponse,
    ArtifactDetail,
    ArtifactPluginResponse,
    ArtifactSummary,
    PlantModelChatRequest,
    PlantModelChatResponse,
    PlantModelResult,
    PlantModelSessionStateOut,
    PlantPayload,
    PreLaunchConfig,
    TokenUsageOut,
    ValidationRequest,
    ValidationResponse,
)


def _to_agent_session_state(
    state: PlantModelSessionStateOut | None,
) -> AgentSessionState | None:
    if state is None:
        return None
    latest = None
    if state.latest_draft is not None:
        latest = {
            "system_name": state.latest_draft.system_name,
            "python_code": state.latest_draft.python_code,
        }
        if state.latest_draft.metadata is not None:
            latest["metadata"] = state.latest_draft.metadata
    return AgentSessionState(draft_count=state.draft_count, latest_draft=latest)


def _from_agent_session_state(state: AgentSessionState) -> PlantModelSessionStateOut:
    latest = None
    if state.latest_draft is not None:
        latest = PlantModelResult(
            system_name=state.latest_draft["system_name"],
            python_code=state.latest_draft["python_code"],
            metadata=state.latest_draft.get("metadata"),
        )
    return PlantModelSessionStateOut(draft_count=state.draft_count, latest_draft=latest)


def _infer_status(
    *,
    prev_draft_count: int,
    draft_count: int,
    final_result: dict | None,
) -> Literal["continue", "draft", "complete"]:
    if final_result is not None:
        return "complete"
    if draft_count > prev_draft_count:
        return "draft"
    return "continue"


def run_plant_model_chat(request: PlantModelChatRequest) -> PlantModelChatResponse:
    """Run one plant-model turn and return a structured response."""
    agent = PlantModelAgent(
        model=request.model,
        max_drafts=request.max_drafts,
        min_user_turns_before_completion=request.min_user_turns_before_completion,
    )
    apply_session_state(agent, _to_agent_session_state(request.session_state))
    prev_draft_count = agent._draft_count

    history = [{"role": m.role, "content": m.content} for m in request.messages]
    reply, final_payload = agent.step(history, request.user_message.strip())

    session_state = _from_agent_session_state(export_session_state(agent))
    status = _infer_status(
        prev_draft_count=prev_draft_count,
        draft_count=agent._draft_count,
        final_result=final_payload,
    )

    final_result = None
    if final_payload is not None:
        final_result = PlantModelResult(
            system_name=final_payload["system_name"],
            python_code=final_payload["python_code"],
            metadata=final_payload.get("metadata"),
        )

    usage_totals = agent.total_usage
    usage = TokenUsageOut(
        input_tokens=usage_totals.input_tokens,
        output_tokens=usage_totals.output_tokens,
        estimated_cost=agent.total_cost,
    )

    return PlantModelChatResponse(
        reply=reply,
        status=status,
        final_result=final_result,
        session_state=session_state,
        usage=usage,
        conversation_id=request.conversation_id,
    )


# ---------------------------------------------------------------------------
# Artifact store / compile
# ---------------------------------------------------------------------------


def default_artifacts_dir() -> str:
    """Resolve artifact base directory (env or repo-root ``artifacts/``)."""
    env = os.getenv("LABCD_ARTIFACTS_DIR")
    if env:
        return env
    # backend_api/AgentPlant/service.py → parents[2] == repo root
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "artifacts")


def get_artifact_store(base_dir: str | None = None) -> ArtifactStore:
    return ArtifactStore(base_dir=base_dir or default_artifacts_dir())


def plant_payload_to_dict(plant: PlantPayload | PlantModelResult) -> dict[str, Any]:
    data: dict[str, Any] = {
        "system_name": plant.system_name,
        "python_code": plant.python_code,
    }
    meta = getattr(plant, "metadata", None)
    if meta is not None:
        data["metadata"] = meta
    return data


def pre_launch_to_dict(pre_launch: PreLaunchConfig) -> dict[str, Any]:
    return {
        "total_simulation_time": float(pre_launch.total_simulation_time),
        "solver_sample_time": float(pre_launch.solver_sample_time),
        "initial_state": [float(x) for x in pre_launch.initial_state],
        "default_target": [float(x) for x in pre_launch.default_target],
    }


class ArtifactValidationError(Exception):
    """Raised when plant or pre-launch validation fails before persist."""

    def __init__(self, errors: list[str], warnings: list[str] | None = None) -> None:
        self.errors = list(errors)
        self.warnings = list(warnings or [])
        super().__init__("; ".join(self.errors) if self.errors else "validation failed")


def validate_plant_and_pre_launch(
    plant: dict[str, Any],
    pre_launch: dict[str, Any] | None = None,
) -> ValidationResponse:
    """Validate plant (and optional pre-launch) without writing files."""
    compiler = PlantCompiler()
    plant_result = compiler.validate(plant)
    errors = list(plant_result.errors)
    warnings = list(plant_result.warnings)

    if pre_launch is not None and plant_result.ok:
        meta = plant.get("metadata") or {}
        if not meta and plant.get("python_code"):
            # Legacy plant: allow pre-launch length checks against empty states
            meta = {"states": [], "outputs": []}
        pl_result = validate_pre_launch(pre_launch, meta)
        errors.extend(pl_result.errors)
        warnings.extend(pl_result.warnings)

    return ValidationResponse(
        ok=not errors,
        errors=errors,
        warnings=warnings,
    )


def create_artifact(
    request: ArtifactCreateRequest,
    *,
    plant_override: dict[str, Any] | None = None,
    store: ArtifactStore | None = None,
) -> ArtifactCreateResponse:
    """Compile and persist an artifact. Raises ArtifactValidationError on failure."""
    if plant_override is not None:
        plant = dict(plant_override)
    elif request.plant is not None:
        plant = plant_payload_to_dict(request.plant)
    else:
        raise ArtifactValidationError(
            ["plant payload is required (or a completed conversation_id)"]
        )

    pre_launch = pre_launch_to_dict(request.pre_launch)
    validation = validate_plant_and_pre_launch(plant, pre_launch)
    if not validation.ok:
        raise ArtifactValidationError(validation.errors, validation.warnings)

    art_store = store or get_artifact_store()
    artifact_id = art_store.save_from_plant(plant, pre_launch)
    data = art_store.load(artifact_id)
    return ArtifactCreateResponse(
        artifact_id=artifact_id,
        system_name=str(data.get("system_name") or plant.get("system_name") or ""),
        created_at=str(data.get("created_at") or ""),
        version=str(data.get("version") or "1.0"),
        warnings=list(validation.warnings),
    )


def list_artifacts(store: ArtifactStore | None = None) -> list[ArtifactSummary]:
    art_store = store or get_artifact_store()
    return [
        ArtifactSummary(
            artifact_id=str(item.get("artifact_id") or ""),
            system_name=str(item.get("system_name") or ""),
            created_at=str(item.get("created_at") or ""),
            version=str(item.get("version") or ""),
        )
        for item in art_store.list_artifacts()
    ]


def get_artifact(
    artifact_id: str,
    store: ArtifactStore | None = None,
) -> ArtifactDetail:
    art_store = store or get_artifact_store()
    try:
        data = art_store.load(artifact_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Artifact not found: {artifact_id}") from exc
    return ArtifactDetail(
        artifact_id=str(data.get("artifact_id") or artifact_id),
        system_name=str(data.get("system_name") or ""),
        created_at=str(data.get("created_at") or ""),
        version=str(data.get("version") or "1.0"),
        plant=dict(data.get("plant") or {}),
        pre_launch=dict(data.get("pre_launch") or {}),
        module_specific=dict(data.get("module_specific") or {}),
    )


def get_artifact_plugin(
    artifact_id: str,
    store: ArtifactStore | None = None,
) -> ArtifactPluginResponse:
    art_store = store or get_artifact_store()
    try:
        path = art_store.load_plugin_path(artifact_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Plugin not found for artifact: {artifact_id}") from exc
    source = Path(path).read_text(encoding="utf-8")
    return ArtifactPluginResponse(
        artifact_id=artifact_id,
        plugin_path=path,
        source=source,
    )


def get_adaptive_spec(
    artifact_id: str,
    store: ArtifactStore | None = None,
) -> dict[str, Any]:
    art_store = store or get_artifact_store()
    try:
        return art_store.get_adaptive_spec(artifact_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Artifact not found: {artifact_id}") from exc


def run_validation(request: ValidationRequest, plant: dict[str, Any] | None) -> ValidationResponse:
    """Service entry for POST /validate."""
    if plant is None and request.plant is not None:
        plant = plant_payload_to_dict(request.plant)
    if plant is None:
        return ValidationResponse(
            ok=False,
            errors=["plant payload is required (or a completed conversation_id)"],
        )
    pre_launch = None
    if request.pre_launch is not None:
        pre_launch = pre_launch_to_dict(request.pre_launch)
    return validate_plant_and_pre_launch(plant, pre_launch)
