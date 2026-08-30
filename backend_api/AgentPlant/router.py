"""Plant-model chat and artifact routes.

Adapted from LabCD_Application ``backend_api/http/routers/plant_model.py``.
Standalone NewModules mode uses an in-memory conversation store and filesystem
``ArtifactStore``. When merging into LabCD_Application, swap the conversation
store for the DB-backed service and re-attach ``require_action`` / ``get_db``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from backend_api.AgentPlant.conversation_store import (
    ConversationAccessDenied,
    InMemoryConversationStore,
    default_store,
)
from backend_api.AgentPlant.schemas import (
    ArtifactCreateRequest,
    ArtifactCreateResponse,
    ArtifactDetail,
    ArtifactPluginResponse,
    ArtifactSummary,
    PlantModelChatRequest,
    PlantModelChatResponse,
    PlantModelConversationDetail,
    PlantModelConversationSummary,
    ValidationRequest,
    ValidationResponse,
)
from backend_api.AgentPlant.service import (
    ArtifactValidationError,
    create_artifact,
    get_adaptive_spec,
    get_artifact,
    get_artifact_plugin,
    list_artifacts,
    plant_payload_to_dict,
    run_plant_model_chat,
    run_validation,
)

router = APIRouter(prefix="/plant-model", tags=["plant-model"])


def _store() -> InMemoryConversationStore:
    return default_store


def _assert_access(
    conversation_user_id: int | None,
    caller_user_id: int | None,
) -> None:
    """Standalone mode: allow if either side is anonymous or ids match."""
    if caller_user_id is None or conversation_user_id is None:
        return
    if conversation_user_id != caller_user_id:
        raise ConversationAccessDenied("Conversation access denied")


def _resolve_plant_from_conversation(
    conversation_id: int | None,
    user_id: int | None,
) -> dict[str, Any] | None:
    """Load final_result from a completed conversation, or None if not provided."""
    if conversation_id is None:
        return None
    conversation = _store().get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        _assert_access(conversation.user_id, user_id)
    except ConversationAccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if conversation.status != "complete" or conversation.final_result is None:
        raise HTTPException(
            status_code=400,
            detail="Conversation is not complete; finish the plant chat first",
        )
    return plant_payload_to_dict(conversation.final_result)


# ---------------------------------------------------------------------------
# Conversations / chat (existing)
# ---------------------------------------------------------------------------


@router.get("/conversations", response_model=list[PlantModelConversationSummary])
def list_plant_model_conversations(
    user_id: int | None = None,
) -> list[PlantModelConversationSummary]:
    conversations = _store().list_for_user(user_id)
    return [
        PlantModelConversationSummary(
            id=c.id,
            title=c.title,
            status=c.status,
            llm_model=c.llm_model,
            system_name=c.final_result.system_name if c.final_result else None,
            user_id=c.user_id,
            owner_email=c.owner_email,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in conversations
    ]


@router.get("/conversations/{conversation_id}", response_model=PlantModelConversationDetail)
def get_plant_model_conversation(
    conversation_id: int,
    user_id: int | None = None,
) -> PlantModelConversationDetail:
    conversation = _store().get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        _assert_access(conversation.user_id, user_id)
    except ConversationAccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return PlantModelConversationDetail(
        id=conversation.id,
        title=conversation.title,
        status=conversation.status,
        llm_model=conversation.llm_model,
        messages=conversation.messages,
        session_state=conversation.session_state,
        final_result=conversation.final_result,
        user_id=conversation.user_id,
        owner_email=conversation.owner_email,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_plant_model_conversation(
    conversation_id: int,
    user_id: int | None = None,
) -> None:
    conversation = _store().get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        _assert_access(conversation.user_id, user_id)
    except ConversationAccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    _store().delete(conversation_id)


@router.post("/chat", response_model=PlantModelChatResponse)
def plant_model_chat(
    request: PlantModelChatRequest,
    user_id: int | None = None,
) -> PlantModelChatResponse:
    if request.conversation_id is not None:
        existing = _store().get(request.conversation_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        try:
            _assert_access(existing.user_id, user_id)
        except ConversationAccessDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    response = run_plant_model_chat(request)
    conversation = _store().persist_turn(
        user_id=user_id,
        conversation_id=request.conversation_id,
        user_message=request.user_message.strip(),
        assistant_reply=response.reply,
        llm_model=request.model,
        session_state=response.session_state,
        final_result=response.final_result,
    )
    response.conversation_id = conversation.id
    return response


# ---------------------------------------------------------------------------
# Artifacts (unified hand-off)
# ---------------------------------------------------------------------------


@router.post(
    "/artifacts",
    response_model=ArtifactCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_plant_artifact(
    request: ArtifactCreateRequest,
    user_id: int | None = None,
) -> ArtifactCreateResponse:
    plant: dict[str, Any] | None = None
    if request.conversation_id is not None:
        plant = _resolve_plant_from_conversation(request.conversation_id, user_id)
    elif request.plant is not None:
        plant = plant_payload_to_dict(request.plant)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide plant payload or conversation_id of a completed plant chat",
        )

    try:
        return create_artifact(request, plant_override=plant)
    except ArtifactValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Plant or pre-launch validation failed",
                "errors": exc.errors,
                "warnings": exc.warnings,
            },
        ) from exc


@router.get("/artifacts", response_model=list[ArtifactSummary])
def list_plant_artifacts() -> list[ArtifactSummary]:
    return list_artifacts()


@router.get("/artifacts/{artifact_id}", response_model=ArtifactDetail)
def get_plant_artifact(artifact_id: str) -> ArtifactDetail:
    try:
        return get_artifact(artifact_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/artifacts/{artifact_id}/plugin",
    response_model=ArtifactPluginResponse,
)
def get_plant_artifact_plugin(artifact_id: str) -> ArtifactPluginResponse:
    try:
        return get_artifact_plugin(artifact_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/artifacts/{artifact_id}/adaptive-spec")
def get_plant_artifact_adaptive_spec(artifact_id: str) -> dict[str, Any]:
    try:
        return get_adaptive_spec(artifact_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/validate", response_model=ValidationResponse)
def validate_plant_or_pre_launch(
    request: ValidationRequest,
    user_id: int | None = None,
) -> ValidationResponse:
    plant: dict[str, Any] | None = None
    if request.conversation_id is not None:
        plant = _resolve_plant_from_conversation(request.conversation_id, user_id)
    elif request.plant is not None:
        plant = plant_payload_to_dict(request.plant)
    return run_validation(request, plant)
