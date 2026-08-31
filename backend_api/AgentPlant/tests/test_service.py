"""Unit tests for the AgentPlant FastAPI service adapter (no live LLM)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend_api.AgentPlant.schemas import (
    ChatMessage,
    PlantModelChatRequest,
    PlantModelSessionStateOut,
)
from backend_api.AgentPlant.service import run_plant_model_chat
from backend_core.AgentPlant import (
    PlantModelSessionState,
    apply_session_state,
    export_session_state,
)


def test_session_state_roundtrip():
    agent = MagicMock()
    agent._draft_count = 0
    agent._latest_draft = None
    agent.reset_conversation_state = MagicMock()

    apply_session_state(
        agent,
        PlantModelSessionState(
            draft_count=2,
            latest_draft={
                "system_name": "motor",
                "python_code": "def dynamics(t, x, u):\n    return x",
            },
        ),
    )
    assert agent._draft_count == 2
    assert agent._latest_draft["system_name"] == "motor"

    snap = export_session_state(agent)
    assert snap.draft_count == 2
    assert snap.latest_draft["python_code"].startswith("def dynamics")


def test_run_plant_model_chat_continue():
    mock_agent = MagicMock()
    mock_agent._draft_count = 0
    mock_agent._latest_draft = None
    mock_agent.step.return_value = ("What kind of plant is it?", None)
    mock_agent.total_usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    mock_agent.total_cost = 0.001

    with patch("backend_api.AgentPlant.service.PlantModelAgent", return_value=mock_agent):
        with patch(
            "backend_api.AgentPlant.service.export_session_state",
            return_value=PlantModelSessionState(draft_count=0, latest_draft=None),
        ):
            response = run_plant_model_chat(
                PlantModelChatRequest(user_message="hello", messages=[])
            )

    assert response.status == "continue"
    assert response.final_result is None
    assert response.reply == "What kind of plant is it?"
    assert response.usage is not None
    assert response.usage.input_tokens == 10


def test_run_plant_model_chat_draft_and_complete():
    draft_payload = {
        "system_name": "dc_motor",
        "python_code": "def dynamics(t, x, u):\n    return x",
    }

    mock_agent = MagicMock()
    mock_agent._draft_count = 0
    mock_agent._latest_draft = None
    mock_agent.total_usage = SimpleNamespace(input_tokens=20, output_tokens=40)
    mock_agent.total_cost = 0.002

    def _step_draft(history, user_message):
        # Simulate agent bumping draft count during step (after apply_session_state).
        mock_agent._draft_count = 1
        mock_agent._latest_draft = draft_payload
        return ("Here is a draft.", None)

    mock_agent.step.side_effect = _step_draft

    with patch("backend_api.AgentPlant.service.PlantModelAgent", return_value=mock_agent):
        with patch(
            "backend_api.AgentPlant.service.export_session_state",
            return_value=PlantModelSessionState(
                draft_count=1,
                latest_draft=draft_payload,
            ),
        ):
            response = run_plant_model_chat(
                PlantModelChatRequest(
                    user_message="a DC motor",
                    messages=[ChatMessage(role="user", content="hi")],
                    session_state=PlantModelSessionStateOut(draft_count=0),
                )
            )

    assert response.status == "draft"
    assert response.session_state.draft_count == 1
    assert response.session_state.latest_draft is not None
    assert response.session_state.latest_draft.system_name == "dc_motor"

    # Complete path — status is inferred from a non-null final_payload
    def _step_complete(history, user_message):
        mock_agent._draft_count = 1
        mock_agent._latest_draft = draft_payload
        return (
            "Model ready — **dc_motor**.",
            dict(draft_payload),
        )

    mock_agent.step.side_effect = _step_complete
    with patch("backend_api.AgentPlant.service.PlantModelAgent", return_value=mock_agent):
        with patch(
            "backend_api.AgentPlant.service.export_session_state",
            return_value=PlantModelSessionState(
                draft_count=1,
                latest_draft=draft_payload,
            ),
        ):
            done = run_plant_model_chat(
                PlantModelChatRequest(user_message="finish", messages=[])
            )

    assert done.status == "complete"
    assert done.final_result is not None
    assert done.final_result.system_name == "dc_motor"


def test_conversation_store_persist_and_list():
    from backend_api.AgentPlant.conversation_store import InMemoryConversationStore
    from backend_api.AgentPlant.schemas import PlantModelResult, PlantModelSessionStateOut

    store = InMemoryConversationStore()
    state = PlantModelSessionStateOut(draft_count=1)
    c1 = store.persist_turn(
        user_id=7,
        conversation_id=None,
        user_message="hello motor",
        assistant_reply="What voltage?",
        llm_model="gpt-4o-mini",
        session_state=state,
        final_result=None,
    )
    assert c1.id == 1
    assert c1.status == "active"
    assert len(c1.messages) == 2

    c2 = store.persist_turn(
        user_id=7,
        conversation_id=c1.id,
        user_message="finish",
        assistant_reply="done",
        llm_model="gpt-4o-mini",
        session_state=state,
        final_result=PlantModelResult(
            system_name="motor",
            python_code="def dynamics(t, x, u):\n    return x",
        ),
    )
    assert c2.id == 1
    assert c2.status == "complete"
    assert c2.final_result is not None
    assert c2.title == "motor"

    listed = store.list_for_user(7)
    assert len(listed) == 1
    assert store.get(1) is not None
    assert store.delete(1) is True
    assert store.get(1) is None
