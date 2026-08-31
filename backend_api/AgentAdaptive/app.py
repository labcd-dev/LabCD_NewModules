"""Standalone FastAPI application for AgentAdaptive.

Run from the repository root:

    PYTHONPATH=. uvicorn backend_api.AgentAdaptive.app:app --reload --port 8004

Or:

    PYTHONPATH=. python -m backend_api.AgentAdaptive.app
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

from backend_api.AgentAdaptive.router import router as adaptive_router

API_PREFIX = os.getenv("LABCD_API_PREFIX", "/api")

app = FastAPI(
    title="LabCD AgentAdaptive API",
    description=(
        "FastAPI layer for AgentAdaptive (Clarifier → Designer → build → "
        "Tuner → Report). Job-oriented contracts for long-running design runs."
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

app.include_router(adaptive_router, prefix=API_PREFIX)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "agent-adaptive"}


def main() -> None:
    import uvicorn

    host = os.getenv("LABCD_ADAPTIVE_HOST", "0.0.0.0")
    port = int(os.getenv("LABCD_ADAPTIVE_PORT", "8004"))
    uvicorn.run(
        "backend_api.AgentAdaptive.app:app",
        host=host,
        port=port,
        reload=os.getenv("LABCD_RELOAD", "0") == "1",
    )


if __name__ == "__main__":
    main()
