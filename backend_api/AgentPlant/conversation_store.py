"""In-memory conversation store for standalone NewModules runs.

Mirrors the persistence shape of LabCD_Application's
``plant_model_chat_service`` (messages, session_state, final result) so the
router contracts stay the same. Replace with the Application DB-backed
service when merging into the full product.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Literal

from backend_api.AgentPlant.schemas import (
    ChatMessage,
    PlantModelResult,
    PlantModelSessionStateOut,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ConversationRecord:
    id: int
    title: str
    status: Literal["active", "complete"]
    llm_model: str
    messages: list[ChatMessage] = field(default_factory=list)
    session_state: PlantModelSessionStateOut | None = None
    final_result: PlantModelResult | None = None
    user_id: int | None = None
    owner_email: str | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


class ConversationAccessDenied(Exception):
    """Raised when a caller may not access a conversation."""


class InMemoryConversationStore:
    """Thread-safe in-memory store keyed by conversation id."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._next_id = 1
        self._conversations: dict[int, ConversationRecord] = {}

    def list_for_user(self, user_id: int | None = None) -> list[ConversationRecord]:
        with self._lock:
            records = list(self._conversations.values())
        if user_id is not None:
            records = [c for c in records if c.user_id == user_id]
        records.sort(key=lambda c: c.updated_at, reverse=True)
        return [deepcopy(c) for c in records]

    def get(self, conversation_id: int) -> ConversationRecord | None:
        with self._lock:
            record = self._conversations.get(conversation_id)
            return deepcopy(record) if record is not None else None

    def delete(self, conversation_id: int) -> bool:
        with self._lock:
            return self._conversations.pop(conversation_id, None) is not None

    def persist_turn(
        self,
        *,
        user_id: int | None,
        conversation_id: int | None,
        user_message: str,
        assistant_reply: str,
        llm_model: str,
        session_state: PlantModelSessionStateOut,
        final_result: PlantModelResult | None,
    ) -> ConversationRecord:
        with self._lock:
            conversation: ConversationRecord | None = None
            if conversation_id is not None:
                conversation = self._conversations.get(conversation_id)
                if conversation is not None and user_id is not None:
                    if conversation.user_id is not None and conversation.user_id != user_id:
                        conversation = None

            if conversation is None:
                conversation = ConversationRecord(
                    id=self._next_id,
                    title=_title_from_message(user_message),
                    status="active",
                    llm_model=llm_model,
                    user_id=user_id,
                )
                self._conversations[conversation.id] = conversation
                self._next_id += 1

            conversation.llm_model = llm_model
            conversation.session_state = session_state
            conversation.updated_at = _now()
            conversation.messages.append(ChatMessage(role="user", content=user_message))
            conversation.messages.append(
                ChatMessage(role="assistant", content=assistant_reply)
            )

            if final_result is not None:
                conversation.status = "complete"
                conversation.final_result = final_result
                if final_result.system_name.strip():
                    conversation.title = final_result.system_name.strip()[:120]

            return deepcopy(conversation)


def _title_from_message(message: str) -> str:
    text = " ".join(message.strip().split())
    if not text:
        return "Untitled plant"
    return text[:80] + ("…" if len(text) > 80 else "")


# Process-wide default store for the standalone app.
default_store = InMemoryConversationStore()
