"""Standalone FastAPI application for AgentMPC.

Run from the repository root:

    PYTHONPATH=. uvicorn backend_api.AgentMPC.app:app --reload --port 8005

Or:

    PYTHONPATH=. python -m backend_api.AgentMPC.app
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from labcd_agents import ensure_env_loaded

    _env = _REPO_ROOT / ".env"
    ensure_env_loaded(str(_env) if _env.is_file() else None)
except ImportError:
    pass

from backend_api.AgentMPC.router import router as mpc_router

API_PREFIX = os.getenv("LABCD_API_PREFIX", "/api")

app = FastAPI(
    title="LabCD AgentMPC API",
    description=(
        "FastAPI layer for AgentMPC (multi-agent MPC auto-tuning over a "
        "dynamics plugin + MPC solver). Job-oriented contracts for long-running "
        "tuning runs."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("LABCD_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mpc_router, prefix=API_PREFIX)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "agent-mpc"}


def main() -> None:
    import uvicorn

    host = os.getenv("LABCD_MPC_HOST", "0.0.0.0")
    port = int(os.getenv("LABCD_MPC_PORT", "8005"))
    uvicorn.run(
        "backend_api.AgentMPC.app:app",
        host=host,
        port=port,
        reload=os.getenv("LABCD_RELOAD", "0") == "1",
    )


if __name__ == "__main__":
    main()
