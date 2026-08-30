"""FastAPI layer for AgentPlant.

Adapts the LabCD_Application PlantModelChat HTTP surface to this repository's
``backend_core.AgentPlant`` core, and exposes the unified artifact hand-off
(PlantCompiler + ArtifactStore) used by ``frontend_streamlit/unified_app``.

Primary entry points:

- :mod:`backend_api.AgentPlant.app` — standalone FastAPI app
- :mod:`backend_api.AgentPlant.router` — include in a larger platform app
- :func:`backend_api.AgentPlant.service.run_plant_model_chat` — chat adapter
- :func:`backend_api.AgentPlant.service.create_artifact` — compile + persist
"""

from backend_api.AgentPlant.service import create_artifact, run_plant_model_chat
from backend_api.AgentPlant.router import router

__all__ = [
    "create_artifact",
    "run_plant_model_chat",
    "router",
]
