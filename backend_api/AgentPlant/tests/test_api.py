"""HTTP-level tests for AgentPlant FastAPI routes (no live LLM)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend_api.AgentPlant.app import app
from backend_api.AgentPlant.conversation_store import InMemoryConversationStore
from backend_api.AgentPlant.schemas import (
    PlantModelChatResponse,
    PlantModelResult,
    PlantModelSessionStateOut,
    TokenUsageOut,
)


@pytest.fixture
def client():
    store = InMemoryConversationStore()
    with patch("backend_api.AgentPlant.router.default_store", store):
        yield TestClient(app)


def _fake_chat_response(*, reply="ok", status="continue", conversation_id=None):
    return PlantModelChatResponse(
        reply=reply,
        status=status,
        final_result=None,
        session_state=PlantModelSessionStateOut(draft_count=0),
        usage=TokenUsageOut(input_tokens=1, output_tokens=1, estimated_cost=0.0),
        conversation_id=conversation_id,
    )


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "agent-plant"


def test_chat_continue(client: TestClient):
    fake = _fake_chat_response(reply="What kind of plant is it?", status="continue")
    with patch(
        "backend_api.AgentPlant.router.run_plant_model_chat",
        return_value=fake,
    ) as mock_run:
        r = client.post(
            "/api/plant-model/chat",
            json={"user_message": "hello", "messages": []},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "continue"
    assert data["reply"] == "What kind of plant is it?"
    assert data["conversation_id"] is not None
    mock_run.assert_called_once()


def test_chat_complete_and_list(client: TestClient):
    complete = PlantModelChatResponse(
        reply="Model ready — **dc_motor**.",
        status="complete",
        final_result=PlantModelResult(
            system_name="dc_motor",
            python_code="def dynamics(t, x, u):\n    return x",
        ),
        session_state=PlantModelSessionStateOut(
            draft_count=1,
            latest_draft=PlantModelResult(
                system_name="dc_motor",
                python_code="def dynamics(t, x, u):\n    return x",
            ),
        ),
        usage=TokenUsageOut(input_tokens=10, output_tokens=20, estimated_cost=0.001),
    )
    with patch(
        "backend_api.AgentPlant.router.run_plant_model_chat",
        return_value=complete,
    ):
        r = client.post(
            "/api/plant-model/chat",
            json={"user_message": "finish", "messages": []},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "complete"
    assert data["final_result"]["system_name"] == "dc_motor"
    cid = data["conversation_id"]

    listed = client.get("/api/plant-model/conversations")
    assert listed.status_code == 200
    items = listed.json()
    assert len(items) == 1
    assert items[0]["id"] == cid
    assert items[0]["status"] == "complete"
    assert items[0]["system_name"] == "dc_motor"

    detail = client.get(f"/api/plant-model/conversations/{cid}")
    assert detail.status_code == 200
    body = detail.json()
    assert len(body["messages"]) == 2
    assert body["final_result"]["system_name"] == "dc_motor"

    deleted = client.delete(f"/api/plant-model/conversations/{cid}")
    assert deleted.status_code == 204
    assert client.get(f"/api/plant-model/conversations/{cid}").status_code == 404


def test_chat_unknown_conversation(client: TestClient):
    r = client.post(
        "/api/plant-model/chat",
        json={
            "user_message": "hi",
            "messages": [],
            "conversation_id": 99999,
        },
    )
    assert r.status_code == 404