"""Plant-model chat API schemas.

Contracts mirror LabCD_Application ``backend_api/http/schemas/plant_model.py``
so clients and a future merge stay compatible.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend_core.AgentPlant import (
    DEFAULT_MAX_DRAFTS,
    DEFAULT_MIN_USER_TURNS_BEFORE_COMPLETION,
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class PlantModelResult(BaseModel):
    system_name: str
    python_code: str


class PlantModelSessionStateOut(BaseModel):
    draft_count: int = 0
    latest_draft: PlantModelResult | None = None


class PlantModelChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    user_message: str
    model: str = "gpt-4o-mini"
    session_state: PlantModelSessionStateOut | None = None
    conversation_id: int | None = None
    max_drafts: int = Field(default=DEFAULT_MAX_DRAFTS, ge=1, le=10)
    min_user_turns_before_completion: int = Field(
        default=DEFAULT_MIN_USER_TURNS_BEFORE_COMPLETION,
        ge=1,
        le=5,
    )


class TokenUsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    estimated_cost: float


class PlantModelChatResponse(BaseModel):
    reply: str
    status: Literal["continue", "draft", "complete"]
    final_result: PlantModelResult | None = None
    session_state: PlantModelSessionStateOut
    usage: Optional[TokenUsageOut] = None
    conversation_id: int | None = None


class PlantModelConversationSummary(BaseModel):
    id: int
    title: str
    status: Literal["active", "complete"]
    llm_model: str
    system_name: str | None = None
    user_id: int | None = None
    owner_email: str | None = None
    created_at: datetime
    updated_at: datetime


class PlantModelConversationDetail(BaseModel):
    id: int
    title: str
    status: Literal["active", "complete"]
    llm_model: str
    messages: list[ChatMessage]
    session_state: PlantModelSessionStateOut | None = None
    final_result: PlantModelResult | None = None
    user_id: int | None = None
    owner_email: str | None = None
    created_at: datetime
    updated_at: datetime
