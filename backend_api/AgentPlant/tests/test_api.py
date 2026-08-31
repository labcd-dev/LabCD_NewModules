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

# ---------------------------------------------------------------------------
# Artifact routes
# ---------------------------------------------------------------------------


def _minimal_plant_json():
    return {
        "system_name": "simple_integrator",
        "python_code": "def dynamics(t, x, u):\n    return [u[0]]",
    }


def _minimal_pre_launch(n_states: int = 0):
    return {
        "total_simulation_time": 10.0,
        "solver_sample_time": 0.001,
        "initial_state": [0.0] * n_states,
        "default_target": [0.0] * n_states,
    }


@pytest.fixture
def artifact_client(tmp_path):
    """TestClient with isolated conversation store and artifact directory."""
    store = InMemoryConversationStore()
    with patch("backend_api.AgentPlant.router.default_store", store):
        with patch(
            "backend_api.AgentPlant.service.default_artifacts_dir",
            return_value=str(tmp_path),
        ):
            yield TestClient(app), store, tmp_path


def test_create_list_get_artifact(artifact_client):
    client, _store, tmp_path = artifact_client
    body = {
        "plant": _minimal_plant_json(),
        "pre_launch": _minimal_pre_launch(0),
    }
    r = client.post("/api/plant-model/artifacts", json=body)
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["artifact_id"]
    assert data["system_name"] == "simple_integrator"
    artifact_id = data["artifact_id"]

    listed = client.get("/api/plant-model/artifacts")
    assert listed.status_code == 200
    ids = [item["artifact_id"] for item in listed.json()]
    assert artifact_id in ids

    detail = client.get(f"/api/plant-model/artifacts/{artifact_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["artifact_id"] == artifact_id
    assert payload["plant"]["python_code"].startswith("def dynamics")
    assert payload["pre_launch"]["total_simulation_time"] == 10.0

    plugin = client.get(f"/api/plant-model/artifacts/{artifact_id}/plugin")
    assert plugin.status_code == 200
    plugin_body = plugin.json()
    assert "BaseDynamics" in plugin_body["source"]
    assert plugin_body["plugin_path"].endswith(".py")

    adaptive = client.get(f"/api/plant-model/artifacts/{artifact_id}/adaptive-spec")
    assert adaptive.status_code == 200
    spec = adaptive.json()
    assert spec["system_name"] == "simple_integrator"
    assert "dynamics" in spec


def test_create_artifact_from_conversation(artifact_client):
    client, store, _tmp = artifact_client
    complete = PlantModelChatResponse(
        reply="done",
        status="complete",
        final_result=PlantModelResult(
            system_name="simple_integrator",
            python_code="def dynamics(t, x, u):\n    return [u[0]]",
        ),
        session_state=PlantModelSessionStateOut(draft_count=1),
        usage=TokenUsageOut(input_tokens=1, output_tokens=1, estimated_cost=0.0),
    )
    with patch(
        "backend_api.AgentPlant.router.run_plant_model_chat",
        return_value=complete,
    ):
        chat = client.post(
            "/api/plant-model/chat",
            json={"user_message": "finish", "messages": []},
        )
    assert chat.status_code == 200
    cid = chat.json()["conversation_id"]

    r = client.post(
        "/api/plant-model/artifacts",
        json={
            "conversation_id": cid,
            "pre_launch": _minimal_pre_launch(0),
        },
    )
    assert r.status_code == 201, r.text
    assert r.json()["system_name"] == "simple_integrator"


def test_create_artifact_validation_error(artifact_client):
    client, _store, _tmp = artifact_client
    r = client.post(
        "/api/plant-model/artifacts",
        json={
            "plant": {"system_name": "", "python_code": "no dynamics here"},
            "pre_launch": _minimal_pre_launch(0),
        },
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "errors" in detail
    assert detail["errors"]


def test_validate_endpoint(artifact_client):
    client, _store, _tmp = artifact_client
    ok = client.post(
        "/api/plant-model/validate",
        json={
            "plant": _minimal_plant_json(),
            "pre_launch": _minimal_pre_launch(0),
        },
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["ok"] is True

    bad = client.post(
        "/api/plant-model/validate",
        json={
            "plant": {"system_name": "x", "python_code": "pass"},
        },
    )
    assert bad.status_code == 200
    assert bad.json()["ok"] is False
    assert bad.json()["errors"]


def test_artifact_not_found(artifact_client):
    client, _store, _tmp = artifact_client
    assert client.get("/api/plant-model/artifacts/does-not-exist").status_code == 404
    assert (
        client.get("/api/plant-model/artifacts/does-not-exist/plugin").status_code == 404
    )
    assert (
        client.get(
            "/api/plant-model/artifacts/does-not-exist/adaptive-spec"
        ).status_code
        == 404
    )
