"""FastAPI layer for AgentPlant.

Adapts the LabCD_Application PlantModelChat HTTP surface to this repository's
``backend_core.AgentPlant`` core. Primary entry points:

- :mod:`backend_api.AgentPlant.app` — standalone FastAPI app
- :mod:`backend_api.AgentPlant.router` — include in a larger platform app
- :func:`backend_api.AgentPlant.service.run_plant_model_chat` — pure adapter
"""

from backend_api.AgentPlant.service import run_plant_model_chat
from backend_api.AgentPlant.router import router

__all__ = [
    "run_plant_model_chat",
    "router",
]
