"""FastAPI layer for AgentAdaptive.

Exposes adaptive / nonlinear control design (Clarifier → Designer → build →
Tuner → Report) as job-oriented HTTP routes. Core logic stays in
``backend_core.AgentAdaptive``; this package is a thin adapter + job store.

Primary entry points:

- :mod:`backend_api.AgentAdaptive.app` — standalone FastAPI app
- :mod:`backend_api.AgentAdaptive.router` — include in a larger platform app
"""

from backend_api.AgentAdaptive.router import router
from backend_api.AgentAdaptive.service import (
    cancel_job,
    clarify_job,
    get_job,
    get_results,
    list_jobs,
    submit_job,
)

__all__ = [
    "router",
    "submit_job",
    "get_job",
    "clarify_job",
    "cancel_job",
    "get_results",
    "list_jobs",
]
