"""Smoke-test AgentPlant HTTP API against a running uvicorn server.

Usage (server already running on :8003):

    python scripts/smoke_plant_api.py

Optional: set BASE_URL if the port/prefix differs.
"""

from __future__ import annotations

import json
import os
import sys

import httpx

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8003")
API = f"{BASE}/api/plant-model"


def pretty(label: str, data) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(data, indent=2, default=str))


def main() -> int:
    with httpx.Client(
            timeout=60.0,
            trust_env=False,  # ignore HTTP_PROXY / HTTPS_PROXY
    ) as client:
        # 1) Health
        r = client.get(f"{BASE}/health")
        r.raise_for_status()
        pretty("health", r.json())

        # 2) First chat turn (calls real LLM unless you mock at the service layer)
        payload = {
            "user_message": "I want a simple DC motor model",
            "messages": [],
            "model": "gpt-4o-mini",
        }
        r = client.post(f"{API}/chat", json=payload)
        r.raise_for_status()
        chat1 = r.json()
        pretty("chat #1", chat1)
        cid = chat1["conversation_id"]
        session = chat1.get("session_state")

        # 3) Follow-up turn in the same conversation
        history = [
            {"role": "user", "content": payload["user_message"]},
            {"role": "assistant", "content": chat1["reply"]},
        ]
        payload2 = {
            "user_message": "use armature voltage as input and angular velocity as state",
            "messages": history,
            "conversation_id": cid,
            "session_state": session,
            "model": "gpt-4o-mini",
        }
        r = client.post(f"{API}/chat", json=payload2)
        r.raise_for_status()
        chat2 = r.json()
        pretty("chat #2", chat2)

        # 4) List + get conversation
        r = client.get(f"{API}/conversations")
        r.raise_for_status()
        pretty("list conversations", r.json())

        r = client.get(f"{API}/conversations/{cid}")
        r.raise_for_status()
        pretty(f"conversation {cid}", r.json())

        # 5) Delete
        r = client.delete(f"{API}/conversations/{cid}")
        assert r.status_code == 204, r.text
        print(f"\n=== deleted conversation {cid} ===")

        r = client.get(f"{API}/conversations/{cid}")
        print(f"get after delete -> HTTP {r.status_code} (expect 404)")

        # 6) Artifact hand-off (no LLM): validate + compile from explicit plant
        plant = {
            "system_name": "simple_integrator",
            "python_code": "def dynamics(t, x, u):\n    return [u[0]]",
        }
        pre_launch = {
            "total_simulation_time": 5.0,
            "solver_sample_time": 0.01,
            "initial_state": [],
            "default_target": [],
        }
        r = client.post(
            f"{API}/validate",
            json={"plant": plant, "pre_launch": pre_launch},
        )
        r.raise_for_status()
        pretty("validate", r.json())

        r = client.post(
            f"{API}/artifacts",
            json={"plant": plant, "pre_launch": pre_launch},
        )
        r.raise_for_status()
        art = r.json()
        pretty("create artifact", art)
        aid = art["artifact_id"]

        r = client.get(f"{API}/artifacts/{aid}")
        r.raise_for_status()
        pretty(f"artifact {aid}", r.json())

        r = client.get(f"{API}/artifacts/{aid}/plugin")
        r.raise_for_status()
        plugin = r.json()
        print(f"\n=== plugin path ===\n{plugin.get('plugin_path')}")

        r = client.get(f"{API}/artifacts/{aid}/adaptive-spec")
        r.raise_for_status()
        pretty("adaptive-spec", r.json())

    print("\nSmoke test finished OK.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except httpx.ConnectError:
        print(
            f"Cannot connect to {BASE}. Start the server first:\n"
            f'  $env:PYTHONPATH = "."\n'
            f"  uvicorn backend_api.AgentPlant.app:app --host 127.0.0.1 --port 8003",
            file=sys.stderr,
        )
        raise SystemExit(1)