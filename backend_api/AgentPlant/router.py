"""Plant-model chat routes.

Adapted from LabCD_Application ``backend_api/http/routers/plant_model.py``.
Standalone NewModules mode uses an in-memory conversation store and no
platform auth. When merging into LabCD_Application, swap the store for the
DB-backed service and re-attach ``require_action`` / ``get_db`` dependencies.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend_api.AgentPlant.conversation_store import (
    ConversationAccessDenied,
    InMemoryConversationStore,
    default_store,
)
from backend_api.AgentPlant.schemas import (
    PlantModelChatRequest,
    PlantModelChatResponse,
    PlantModelConversationDetail,
    PlantModelConversationSummary,
)
from backend_api.AgentPlant.service import run_plant_model_chat

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
