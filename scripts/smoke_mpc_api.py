"""Smoke-test AgentMPC HTTP API against a running uvicorn server.

Exercises the production-shaped path toward the React client:

  health → create job (plugin_id / options) → list → poll status → results
  optional second job + cancel

Usage (server already running on :8005):

    PYTHONPATH=. uvicorn backend_api.AgentMPC.app:app --host 127.0.0.1 --port 8005
    python scripts/smoke_mpc_api.py

Env:

    BASE_URL            default http://127.0.0.1:8005
    PLUGIN_ID           default example_pendulum
    MAX_ITERATIONS      default 3  (keep low for smoke; live LLM runs are slow)
    USE_UI_GRAPH        default 1  (0 = full graph with Scenarist)
    POLL_TIMEOUT_SEC    default 900
    POLL_INTERVAL_SEC   default 3
    SKIP_LIVE_RUN       default 0  (set 1 to only hit health + 404 paths)
"""

from __future__ import annotations

import json
import os
import sys
import time

import httpx

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8005")
API = f"{BASE}/api/mpc"
PLUGIN_ID = os.getenv("PLUGIN_ID", "example_pendulum")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
USE_UI_GRAPH = os.getenv("USE_UI_GRAPH", "1") != "0"
POLL_TIMEOUT_SEC = float(os.getenv("POLL_TIMEOUT_SEC", "900"))
POLL_INTERVAL_SEC = float(os.getenv("POLL_INTERVAL_SEC", "3"))
SKIP_LIVE_RUN = os.getenv("SKIP_LIVE_RUN", "0") != "0"


def pretty(label: str, data) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(data, indent=2, default=str))


def poll_until_terminal(client: httpx.Client, job_id: str) -> dict:
    deadline = time.time() + POLL_TIMEOUT_SEC
    last = None
    while time.time() < deadline:
        r = client.get(f"{API}/jobs/{job_id}")
        r.raise_for_status()
        last = r.json()
        status = last.get("status")
        stage = last.get("stage")
        iteration = last.get("iteration")
        print(
            f"  poll status={status} stage={stage} "
            f"iter={iteration}/{last.get('max_iterations')} "
            f"msg={last.get('message')!r}"
        )
        if status in ("completed", "failed", "cancelled"):
            return last
        time.sleep(POLL_INTERVAL_SEC)
    raise TimeoutError(
        f"Job {job_id} did not finish within {POLL_TIMEOUT_SEC}s; last={last}"
    )


def create_body() -> dict:
    return {
        "dynamics": {"plugin_id": PLUGIN_ID},
        "options": {
            "max_iterations": MAX_ITERATIONS,
            "prediction_horizon": 12,
            "control_horizon": 4,
            "dt_mpc": 0.02,
            "simulation_time": 3.0,
            "ui_scenario_level": 1,
            "use_ui_graph": USE_UI_GRAPH,
            "user_guidance": "Smoke test: prefer low control effort if MSE is acceptable.",
            "min_explore_iterations": 1,
            "exploration_intensity": 40,
            "system_name": f"smoke_{PLUGIN_ID}",
        },
    }


def main() -> int:
    timeout = httpx.Timeout(POLL_TIMEOUT_SEC + 60.0, connect=30.0)
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        # 1) Health
        r = client.get(f"{BASE}/health")
        r.raise_for_status()
        pretty("health", r.json())
        assert r.json().get("service") == "agent-mpc", r.text

        # 2) Not-found shape (React error handling)
        r = client.get(f"{API}/jobs/does-not-exist")
        assert r.status_code == 404, r.text
        print("\n=== 404 job not found OK ===")

        if SKIP_LIVE_RUN:
            print("\nSKIP_LIVE_RUN=1 — skipping create/poll/results.")
            return 0

        # 3) Create tuning job
        body = create_body()
        r = client.post(f"{API}/jobs", json=body)
        if r.status_code == 400:
            pretty("create rejected (400)", r.json())
            print(
                "Create failed validation (plugin path / dynamics). "
                "Check PLUGIN_ID and server logs.",
                file=sys.stderr,
            )
            return 3
        r.raise_for_status()
        created = r.json()
        pretty("create job", created)
        job_id = created["job_id"]
        assert created.get("status") in ("queued", "running"), created

        # 4) List jobs
        r = client.get(f"{API}/jobs")
        r.raise_for_status()
        listed = r.json()
        pretty("list jobs (count)", {"count": len(listed), "ids": [j.get("job_id") for j in listed]})
        assert any(j.get("job_id") == job_id for j in listed), listed

        # 5) Poll to terminal
        print("\n=== polling until terminal ===")
        terminal = poll_until_terminal(client, job_id)
        pretty("terminal status", terminal)

        # 6) Results
        r = client.get(f"{API}/jobs/{job_id}/results")
        r.raise_for_status()
        results = r.json()
        summary = {
            "job_id": results.get("job_id"),
            "status": results.get("status"),
            "stage": results.get("stage"),
            "best_mse": results.get("best_mse"),
            "best_params": results.get("best_params"),
            "iteration": results.get("iteration"),
            "termination_reason": results.get("termination_reason"),
            "mse_history_len": len(results.get("mse_history") or []),
            "params_history_len": len(results.get("params_history") or []),
            "history_len": len(results.get("history") or []),
            "has_report": bool(results.get("report")),
            "has_export_script": bool(results.get("export_script")),
            "metrics_keys": list((results.get("metrics") or {}).keys()),
            "error": results.get("error"),
        }
        pretty("results summary", summary)

        # 7) Cancel a second job quickly (structure only; may complete first if graph is instant)
        r = client.post(f"{API}/jobs", json=create_body())
        if r.status_code == 201:
            other = r.json()["job_id"]
            cr = client.post(f"{API}/jobs/{other}/cancel")
            cr.raise_for_status()
            pretty("cancel second job", cr.json())
            assert cr.json()["status"] in ("cancelled", "completed", "failed"), cr.text
        else:
            pretty("skip cancel demo (create failed)", r.json() if r.content else r.status_code)

        if terminal.get("status") == "failed":
            print(
                "\nSmoke finished but job FAILED "
                "(check GROQ_API_KEY / LLM config, langgraph, plugin).",
                file=sys.stderr,
            )
            pretty("failure detail", {"error": terminal.get("error"), "message": terminal.get("message")})
            return 2

        if terminal.get("status") == "cancelled":
            print("\nSmoke finished: primary job was cancelled.", file=sys.stderr)
            return 2

    print("\nSmoke test finished OK.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except httpx.ConnectError:
        print(
            f"Cannot connect to {BASE}. Start the server first:\n"
            f"  PYTHONPATH=. uvicorn backend_api.AgentMPC.app:app "
            f"--host 127.0.0.1 --port 8005",
            file=sys.stderr,
        )
        raise SystemExit(1)
