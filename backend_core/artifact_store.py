"""Filesystem-based artifact store for LabCD unified plant artifacts.

Layout under base_dir:
  {artifact_id}.json   — full artifact (plant + pre_launch + module_specific)
  {artifact_id}.py     — AgentMPC plugin source
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend_core.plant_compiler import Artifact, PlantCompiler


class ArtifactStore:
    """Simple filesystem-based artifact store."""

    def __init__(self, base_dir: str = "artifacts") -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._compiler = PlantCompiler()

    def _json_path(self, artifact_id: str) -> Path:
        return self.base_dir / f"{artifact_id}.json"

    def _py_path(self, artifact_id: str) -> Path:
        return self.base_dir / f"{artifact_id}.py"

    def _resolve_collision(self, artifact_id: str) -> str:
        """If id already exists, append _1, _2, ..."""
        candidate = artifact_id
        counter = 1
        while self._json_path(candidate).exists():
            candidate = f"{artifact_id}_{counter}"
            counter += 1
        return candidate

    def save(self, artifact: Artifact) -> str:
        """Persist artifact + generate .py plugin. Returns artifact_id."""
        artifact_id = self._resolve_collision(artifact.artifact_id)
        payload = dict(artifact.full_payload)
        payload["artifact_id"] = artifact_id

        json_path = self._json_path(artifact_id)
        py_path = self._py_path(artifact_id)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        with open(py_path, "w", encoding="utf-8") as f:
            f.write(artifact.mpc_plugin_source)

        return artifact_id

    def save_from_plant(
        self,
        plant_output: dict,
        pre_launch: dict,
    ) -> str:
        """Compile + persist in one step. Returns artifact_id."""
        art = self._compiler.compile_artifact(plant_output, pre_launch)
        return self.save(art)

    def load(self, artifact_id: str) -> dict:
        """Load full artifact JSON."""
        path = self._json_path(artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def load_plugin_path(self, artifact_id: str) -> str:
        """Return absolute path to the .py plugin file."""
        path = self._py_path(artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"Plugin not found for artifact: {artifact_id}")
        return str(path.resolve())

    def list_artifacts(self) -> List[dict]:
        """List all artifacts (metadata only, no full payload)."""
        results: List[dict] = []
        for path in sorted(self.base_dir.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                results.append(
                    {
                        "artifact_id": data.get("artifact_id", path.stem),
                        "system_name": data.get("system_name", ""),
                        "created_at": data.get("created_at", ""),
                        "version": data.get("version", ""),
                    }
                )
            except (json.JSONDecodeError, OSError):
                continue
        return results

    def update_pre_launch(self, artifact_id: str, pre_launch: dict) -> None:
        """Update only the pre_launch section (user edited knobs)."""
        data = self.load(artifact_id)
        data["pre_launch"] = dict(pre_launch)
        with open(self._json_path(artifact_id), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def regenerate_plugin(self, artifact_id: str) -> None:
        """Re-run PlantCompiler to refresh .py (and adaptive-relevant fields).

        Call this after parameter edits that require plugin regeneration.
        """
        data = self.load(artifact_id)
        plant_output = {
            "system_name": data["system_name"],
            "python_code": data["plant"]["python_code"],
            "metadata": data["plant"].get("metadata"),
        }
        pre_launch = data.get("pre_launch") or {}
        art = self._compiler.compile_artifact(plant_output, pre_launch)
        # Preserve artifact_id and module_specific
        payload = art.full_payload
        payload["artifact_id"] = artifact_id
        payload["module_specific"] = data.get("module_specific", payload.get("module_specific"))
        with open(self._json_path(artifact_id), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(self._py_path(artifact_id), "w", encoding="utf-8") as f:
            f.write(art.mpc_plugin_source)

    def get_adaptive_spec(self, artifact_id: str) -> dict:
        """Build AgentAdaptive system_spec from stored artifact."""
        data = self.load(artifact_id)
        plant_output = {
            "system_name": data["system_name"],
            "python_code": data["plant"]["python_code"],
            "metadata": data["plant"].get("metadata"),
        }
        return self._compiler.generate_adaptive_spec(
            plant_output, data.get("pre_launch") or {}
        )
