"""FastAPI layer for AgentMPC.

Exposes MPC auto-tuning (multi-agent graph over a plugin dynamics + MPC
solver) as job-oriented HTTP routes. Core logic stays in
``backend_core.AgentMPC``; this package is a thin adapter + job store.

Primary entry points:

- :mod:`backend_api.AgentMPC.app` — standalone FastAPI app
- :mod:`backend_api.AgentMPC.router` — include in a larger platform app
"""

from backend_api.AgentMPC.router import router
from backend_api.AgentMPC.service import (
    cancel_job,
    get_job,
    get_results,
    list_jobs,
    submit_job,
)

__all__ = [
    "router",
    "submit_job",
    "get_job",
    "cancel_job",
    "get_results",
    "list_jobs",
]
