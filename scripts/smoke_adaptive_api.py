"""Smoke-test AgentAdaptive HTTP API against a running uvicorn server.

Exercises the production-shaped path toward the React client:

  health → create job (skip clarify) → poll status → results
  optional interactive clarify path when SKIP_CLARIFY=0

Usage (server already running on :8004):

    PYTHONPATH=. uvicorn backend_api.AgentAdaptive.app:app --host 127.0.0.1 --port 8004
    python scripts/smoke_adaptive_api.py

Env:

    BASE_URL          default http://127.0.0.1:8004
    SKIP_CLARIFY      default 1 (set 0 to exercise /clarify with canned answers)
    POLL_TIMEOUT_SEC  default 600 (design/sim can be slow with a live LLM)
    ENABLE_TUNING     default 0
"""

from __future__ import annotations

import json
import os
import sys
import time

import httpx

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8004")
API = f"{BASE}/api/adaptive"
SKIP_CLARIFY = os.getenv("SKIP_CLARIFY", "1") != "0"
ENABLE_TUNING = os.getenv("ENABLE_TUNING", "0") != "0"
POLL_TIMEOUT_SEC = float(os.getenv("POLL_TIMEOUT_SEC", "600"))
POLL_INTERVAL_SEC = float(os.getenv("POLL_INTERVAL_SEC", "2"))


def pretty(label: str, data) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(data, indent=2, default=str))


def minimal_system_spec() -> dict:
    """Small SISO plant suitable for API wiring tests (not a fidelity benchmark)."""
    return {
        "status": "complete",
        "system_name": "smoke_integrator",
        "dynamics": {
            "states": ["x"],
            "state_meanings": ["integrator state"],
            "inputs": ["u"],
            "outputs": ["x"],
            "state_equations": ["u"],
            "parameters": {},
            "system_type": "SISO",
            "assumptions": ["unit integrator for API smoke only"],
            "x0": [0.0],
            "references": [{"output": "x", "expr": "1"}],
            "uncertainty": [],
            "disturbance": [],
            "sim_time": 5.0,
            "solver_step": 0.01,
        },
    }


def poll_until_terminal(client: httpx.Client, job_id: str) -> dict:
    deadline = time.time() + POLL_TIMEOUT_SEC
    last = None
    while time.time() < deadline:
        r = client.get(f"{API}/jobs/{job_id}")
        r.raise_for_status()
        last = r.json()
        status = last.get("status")
        stage = last.get("stage")
        print(f"  poll status={status} stage={stage} msg={last.get('message')!r}")
        if status in ("completed", "failed", "cancelled"):
            return last
        time.sleep(POLL_INTERVAL_SEC)
    raise TimeoutError(
        f"Job {job_id} did not finish within {POLL_TIMEOUT_SEC}s; last={last}"
    )


def main() -> int:
    timeout = httpx.Timeout(POLL_TIMEOUT_SEC + 30.0, connect=30.0)
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        # 1) Health
        r = client.get(f"{BASE}/health")
        r.raise_for_status()
        pretty("health", r.json())

        # 2) Create job
        body = {
            "system_spec": minimal_system_spec(),
            "options": {
                "skip_clarify": SKIP_CLARIFY,
                "enable_tuning": ENABLE_TUNING,
                "target_rms_frac": 0.02,
                "max_tuning_rounds": 2,
            },
        }
        r = client.post(f"{API}/jobs", json=body)
        r.raise_for_status()
        created = r.json()
        pretty("create job", created)
        job_id = created["job_id"]

        # 3) Optional clarify loop (when SKIP_CLARIFY=0)
        if not SKIP_CLARIFY:
            answers = [
                "No significant model uncertainty for this smoke test.",
                "No external disturbance for this smoke test.",
                "References as given are fine.",
            ]
            for i, answer in enumerate(answers):
                status = client.get(f"{API}/jobs/{job_id}")
                status.raise_for_status()
                st = status.json()
                pretty(f"status before clarify #{i+1}", st)
                if st["status"] != "clarifying":
                    break
                force = i == len(answers) - 1
                r = client.post(
                    f"{API}/jobs/{job_id}/clarify",
                    json={"answer": answer, "force_finish": force},
                )
                r.raise_for_status()
                pretty(f"clarify #{i+1}", r.json())
                if r.json().get("clarifier_status") in ("complete", "error"):
                    break

        # 4) List jobs
        r = client.get(f"{API}/jobs")
        r.raise_for_status()
        pretty("list jobs", r.json())

        # 5) Poll to terminal
        print("\n=== polling until terminal ===")
        terminal = poll_until_terminal(client, job_id)
        pretty("terminal status", terminal)

        # 6) Results
        r = client.get(f"{API}/jobs/{job_id}/results")
        r.raise_for_status()
        results = r.json()
        # Trim huge report for console
        series = results.get("series") or {}
        channels = series.get("channels") or {}
        summary = {
            "job_id": results.get("job_id"),
            "status": results.get("status"),
            "stage": results.get("stage"),
            "method": results.get("method"),
            "abstract": results.get("abstract"),
            "error": results.get("error"),
            "has_report": bool(results.get("report")),
            "report_chars": len(results.get("report") or ""),
            "final_metrics_keys": list((results.get("final_metrics") or {}).keys()),
            "tuning_rounds": len(results.get("tuning_log") or []),
            "system_name": (results.get("system_spec") or {}).get("system_name"),
            "series_n_points": series.get("n_points"),
            "series_downsampled": series.get("downsampled"),
            "series_channel_keys": list(channels.keys()),
            "series_y_names": (channels.get("y") or {}).get("names"),
        }
        pretty("results summary", summary)

        # 7) Cancel a second job while clarifying (structure only)
        r = client.post(
            f"{API}/jobs",
            json={
                "system_spec": minimal_system_spec(),
                "options": {"skip_clarify": False, "enable_tuning": False},
            },
        )
        if r.status_code == 201:
            other = r.json()["job_id"]
            cr = client.post(f"{API}/jobs/{other}/cancel")
            cr.raise_for_status()
            pretty("cancel while clarifying", cr.json())
            assert cr.json()["status"] == "cancelled", cr.text

        if terminal.get("status") == "failed":
            print("\nSmoke finished but job FAILED (check API keys / plant fidelity).", file=sys.stderr)
            return 2

    print("\nSmoke test finished OK.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except httpx.ConnectError:
        print(
            f"Cannot connect to {BASE}. Start the server first:\n"
            f"  PYTHONPATH=. uvicorn backend_api.AgentAdaptive.app:app "
            f"--host 127.0.0.1 --port 8004",
            file=sys.stderr,
        )
        raise SystemExit(1)
