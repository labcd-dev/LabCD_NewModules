"""Standalone FastAPI application for AgentPlant.

Run from the repository root:

    PYTHONPATH=. uvicorn backend_api.AgentPlant.app:app --reload --port 8003

Or:

    PYTHONPATH=. python -m backend_api.AgentPlant.app
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load repo-root .env for LLM keys when available.
_REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from labcd_agents import ensure_env_loaded

    _env = _REPO_ROOT / ".env"
    ensure_env_loaded(str(_env) if _env.is_file() else None)
except ImportError:
    pass

from backend_api.AgentPlant.router import router as plant_model_router

API_PREFIX = os.getenv("LABCD_API_PREFIX", "/api")

app = FastAPI(
    title="LabCD AgentPlant API",
    description=(
        "FastAPI layer for AgentPlant (natural-language plant → dynamics). "
        "Contracts align with LabCD_Application plant-model routes."
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

app.include_router(plant_model_router, prefix=API_PREFIX)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "agent-plant"}


def main() -> None:
    import uvicorn

    host = os.getenv("LABCD_PLANT_HOST", "0.0.0.0")
    port = int(os.getenv("LABCD_PLANT_PORT", "8003"))
    uvicorn.run(
        "backend_api.AgentPlant.app:app",
        host=host,
        port=port,
        reload=os.getenv("LABCD_RELOAD", "0") == "1",
    )


if __name__ == "__main__":
    main()
