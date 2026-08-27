"""PlantCompiler: validate enriched AgentPlant output and generate downstream artifacts.

Converts AgentPlant (python_code + metadata) + pre-launch config into:
- AgentMPC BaseDynamics plugin (.py source)
- AgentAdaptive system_spec dict
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None  # type: ignore


REQUIRED_METADATA_KEYS = (
    "states",
    "state_meanings",
    "inputs",
    "outputs",
    "state_equations",
    "parameters",
    "system_type",
    "assumptions",
)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("Plant validation failed:\n  - " + "\n  - ".join(self.errors))


@dataclass
class Artifact:
    """Handle returned by compile_artifact."""

    artifact_id: str
    system_name: str
    plant: Dict[str, Any]
    pre_launch: Dict[str, Any]
    adaptive_spec: Dict[str, Any]
    mpc_plugin_source: str
    created_at: str
    full_payload: Dict[str, Any]


def _short_hash(payload: Dict[str, Any]) -> str:
    import json

    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:6]


def _safe_class_name(system_name: str) -> str:
    """Turn 'Cart-Pole' into 'CartPoleDynamics' base identifier."""
    parts = re.findall(r"[A-Za-z0-9]+", system_name or "System")
    if not parts:
        parts = ["System"]
    return "".join(p[:1].upper() + p[1:] for p in parts)


class PlantCompiler:
    """Compiles an enriched AgentPlant output into downstream-ready artifacts."""

    def validate(self, plant_output: dict) -> ValidationResult:
        """Check metadata completeness and equation syntax."""
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(plant_output, dict):
            return ValidationResult(ok=False, errors=["plant_output must be a dict"])

        system_name = plant_output.get("system_name")
        python_code = plant_output.get("python_code")
        if not isinstance(system_name, str) or not system_name.strip():
            errors.append("system_name is required and must be a non-empty string")
        if not isinstance(python_code, str) or not python_code.strip():
            errors.append("python_code is required and must be a non-empty string")
        elif "def dynamics" not in python_code:
            errors.append("python_code must define a dynamics(t, x, u) function")

        meta = plant_output.get("metadata")
        if meta is None:
            warnings.append(
                "metadata missing — legacy AgentPlant output; downstream integration limited"
            )
            return ValidationResult(ok=not errors, errors=errors, warnings=warnings)

        if not isinstance(meta, dict):
            errors.append("metadata must be an object")
            return ValidationResult(ok=False, errors=errors, warnings=warnings)

        for key in REQUIRED_METADATA_KEYS:
            if key not in meta:
                errors.append(f"metadata missing required key: {key}")

        if errors:
            return ValidationResult(ok=False, errors=errors, warnings=warnings)

        states = meta.get("states") or []
        meanings = meta.get("state_meanings") or []
        inputs = meta.get("inputs") or []
        outputs = meta.get("outputs") or []
        eqs = meta.get("state_equations") or []
        params = meta.get("parameters") or {}
        system_type = meta.get("system_type") or ""
        assumptions = meta.get("assumptions")

        if not isinstance(states, list) or not states:
            errors.append("metadata.states must be a non-empty list")
        else:
            for s in states:
                if not isinstance(s, str) or not _IDENT_RE.match(s):
                    errors.append(f"invalid state name: {s!r}")

        if not isinstance(meanings, list):
            errors.append("metadata.state_meanings must be a list")
        elif len(meanings) != len(states):
            errors.append(
                f"state_meanings length ({len(meanings)}) != states length ({len(states)})"
            )

        if not isinstance(inputs, list) or not inputs:
            errors.append("metadata.inputs must be a non-empty list")
        else:
            for i in inputs:
                if not isinstance(i, str) or not _IDENT_RE.match(i):
                    errors.append(f"invalid input name: {i!r}")

        if not isinstance(outputs, list) or not outputs:
            errors.append("metadata.outputs must be a non-empty list")
        else:
            state_set = set(states) if isinstance(states, list) else set()
            for o in outputs:
                if o not in state_set:
                    errors.append(f"output {o!r} is not in states")

        if not isinstance(eqs, list):
            errors.append("metadata.state_equations must be a list")
        elif len(eqs) != len(states):
            errors.append(
                f"state_equations length ({len(eqs)}) != states length ({len(states)})"
            )

        if not isinstance(params, dict):
            errors.append("metadata.parameters must be a dict")
        else:
            for k, v in params.items():
                if not isinstance(k, str) or not _IDENT_RE.match(k):
                    errors.append(f"invalid parameter name: {k!r}")
                try:
                    float(v)
                except (TypeError, ValueError):
                    errors.append(f"parameter {k!r} value is not numeric: {v!r}")

        if system_type not in ("SISO", "MIMO"):
            errors.append(f"system_type must be 'SISO' or 'MIMO', got {system_type!r}")

        if assumptions is not None and not isinstance(assumptions, list):
            errors.append("metadata.assumptions must be a list")

        # Sympy parse check for equations
        if sp is not None and isinstance(eqs, list) and isinstance(states, list):
            symbols: Dict[str, Any] = {}
            for name in list(states) + list(inputs) + list(params.keys()):
                if isinstance(name, str) and _IDENT_RE.match(name):
                    symbols[name] = sp.symbols(name)
            symbols["t"] = sp.symbols("t")
            symbols["pi"] = sp.pi
            symbols["E"] = sp.E
            # common functions
            local_dict = {
                **symbols,
                "sin": sp.sin,
                "cos": sp.cos,
                "tan": sp.tan,
                "asin": sp.asin,
                "acos": sp.acos,
                "atan": sp.atan,
                "atan2": sp.atan2,
                "exp": sp.exp,
                "log": sp.log,
                "sqrt": sp.sqrt,
                "Abs": sp.Abs,
                "sign": sp.sign,
                "sinh": sp.sinh,
                "cosh": sp.cosh,
                "tanh": sp.tanh,
                "Heaviside": sp.Heaviside,
            }
            for i, eq in enumerate(eqs):
                if not isinstance(eq, str) or not eq.strip():
                    errors.append(f"state_equations[{i}] is empty")
                    continue
                try:
                    sp.sympify(eq, locals=local_dict)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"state_equations[{i}] not sympy-parseable: {eq!r} ({exc})")

        return ValidationResult(ok=not errors, errors=errors, warnings=warnings)

    def generate_mpc_plugin(self, plant_output: dict, pre_launch: dict) -> str:
        """Return the full .py source for an AgentMPC BaseDynamics plugin."""
        meta = plant_output.get("metadata") or {}
        system_name = plant_output.get("system_name") or "System"
        python_code = plant_output.get("python_code") or ""
        class_base = _safe_class_name(system_name)
        class_name = f"{class_base}Dynamics"

        states = meta.get("states") or []
        inputs = meta.get("inputs") or []
        params = meta.get("parameters") or {}
        n_states = len(states)
        n_inputs = len(inputs)

        initial_state = pre_launch.get("initial_state")
        if not isinstance(initial_state, list) or len(initial_state) != n_states:
            initial_state = [0.0] * n_states
        default_target = pre_launch.get("default_target")
        if not isinstance(default_target, list) or len(default_target) != n_states:
            default_target = [0.0] * n_states

        # Format params / names for insertion
        params_repr = repr({str(k): float(v) for k, v in params.items()})
        state_names_repr = repr(list(states))
        input_names_repr = repr(list(inputs))
        init_repr = repr([float(x) for x in initial_state])
        target_repr = repr([float(x) for x in default_target])

        # Ensure user code is indented correctly as module-level
        user_code = python_code.strip()

        source = f'''# Auto-generated by PlantCompiler from AgentPlant output
# System: {system_name}
# Do not edit by hand unless you know the AgentMPC plugin contract.
import numpy as np
from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig

# --- user-provided dynamics ---
{user_code}

# --- auto-generated config ---
def create_config() -> SystemConfig:
    return SystemConfig(
        n_states={n_states},
        n_inputs={n_inputs},
        params={params_repr},
        state_names={state_names_repr},
        input_names={input_names_repr},
        default_initial_state=np.array({init_repr}),
        default_target=np.array({target_repr}),
    )

class {class_name}(BaseDynamics):
    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return dynamics(0.0, x, u)  # t is unused in most plants
'''
        return source

    def generate_adaptive_spec(self, plant_output: dict, pre_launch: dict) -> dict:
        """Return a system_spec-compatible dict for AgentAdaptive."""
        meta = plant_output.get("metadata") or {}
        system_name = plant_output.get("system_name") or "System"

        states = list(meta.get("states") or [])
        meanings = list(meta.get("state_meanings") or [])
        inputs = list(meta.get("inputs") or [])
        outputs = list(meta.get("outputs") or [])
        eqs = list(meta.get("state_equations") or [])
        params = dict(meta.get("parameters") or {})
        system_type = meta.get("system_type") or "SISO"
        assumptions = list(meta.get("assumptions") or [])

        n = len(states)
        x0 = pre_launch.get("initial_state")
        if not isinstance(x0, list) or len(x0) != n:
            x0 = [0.0] * n
        else:
            x0 = [float(v) for v in x0]

        sim_time = float(pre_launch.get("total_simulation_time") or 10.0)
        solver_step = float(pre_launch.get("solver_sample_time") or 0.001)

        references = self._build_references(meta, pre_launch)

        return {
            "status": "complete",
            "system_name": system_name,
            "dynamics": {
                "states": states,
                "state_meanings": meanings,
                "inputs": inputs,
                "outputs": outputs,
                "state_equations": eqs,
                "x0": x0,
                "references": references,
                "parameters": params,
                "uncertainty": [],
                "disturbance": [],
                "system_type": system_type,
                "sim_time": sim_time,
                "solver_step": solver_step,
                "assumptions": assumptions,
            },
        }

    def _build_references(self, meta: dict, pre_launch: dict) -> List[Dict[str, str]]:
        """Translate trajectory knobs into AgentAdaptive references list."""
        outputs = list(meta.get("outputs") or [])
        states = list(meta.get("states") or [])
        mode = (pre_launch.get("trajectory_mode") or "reg").lower()
        amplitude = float(pre_launch.get("trajectory_amplitude") or 0.5)
        frequency = float(pre_launch.get("trajectory_frequency") or 0.5)
        offset = float(pre_launch.get("trajectory_offset") or 0.0)
        default_target = pre_launch.get("default_target") or [0.0] * len(states)
        sim_time = float(pre_launch.get("total_simulation_time") or 10.0)

        refs: List[Dict[str, str]] = []
        for out_name in outputs:
            try:
                idx = states.index(out_name)
            except ValueError:
                idx = 0
            target_val = float(default_target[idx]) if idx < len(default_target) else 0.0

            if mode == "sin":
                omega = 2.0 * math.pi * frequency
                expr = f"{amplitude} * sin({omega} * t) + {offset}"
            elif mode == "pulse":
                t_rise = sim_time * 0.2
                t_fall = sim_time * 0.7
                expr = (
                    f"{amplitude} * (Heaviside(t - {t_rise}) - Heaviside(t - {t_fall}))"
                    f" + {offset}"
                )
            else:  # reg
                expr = str(target_val)
            refs.append({"output": out_name, "expr": expr})
        return refs

    def compile_artifact(self, plant_output: dict, pre_launch: dict) -> Artifact:
        """Validate, generate both outputs, return artifact handle (no I/O)."""
        result = self.validate(plant_output)
        result.raise_if_invalid()

        system_name = plant_output["system_name"]
        short = _short_hash(
            {
                "system_name": system_name,
                "python_code": plant_output.get("python_code"),
                "metadata": plant_output.get("metadata"),
            }
        )
        artifact_id = f"{system_name.replace(' ', '-')}_{short}"

        mpc_source = self.generate_mpc_plugin(plant_output, pre_launch)
        adaptive_spec = self.generate_adaptive_spec(plant_output, pre_launch)
        created_at = datetime.now(timezone.utc).isoformat()

        full_payload = {
            "artifact_id": artifact_id,
            "system_name": system_name,
            "created_at": created_at,
            "version": "1.0",
            "plant": {
                "python_code": plant_output.get("python_code"),
                "metadata": plant_output.get("metadata"),
            },
            "pre_launch": dict(pre_launch),
            "module_specific": {
                "adaptive": {
                    "clarifier_record": None,
                    "designer_method": None,
                    "tuning_best": None,
                },
                "mpc": {
                    "suggested_dt": None,
                    "suggested_Q": None,
                    "suggested_R": None,
                    "suggested_feedforward": None,
                    "derivative_pairs": None,
                },
            },
        }

        return Artifact(
            artifact_id=artifact_id,
            system_name=system_name,
            plant=full_payload["plant"],
            pre_launch=full_payload["pre_launch"],
            adaptive_spec=adaptive_spec,
            mpc_plugin_source=mpc_source,
            created_at=created_at,
            full_payload=full_payload,
        )


def default_pre_launch(n_states: int = 0) -> Dict[str, Any]:
    """Return a pre-launch config with sensible defaults."""
    return {
        "total_simulation_time": 10.0,
        "solver_sample_time": 0.001,
        "initial_state": [0.0] * n_states,
        "default_target": [0.0] * n_states,
        "trajectory_mode": "reg",
        "trajectory_amplitude": 0.5,
        "trajectory_frequency": 0.5,
        "trajectory_offset": 0.0,
    }


def validate_pre_launch(pre_launch: dict, metadata: dict) -> ValidationResult:
    """Validate pre-launch knobs against metadata metadata."""
    errors: List[str] = []
    states = metadata.get("states") or []
    n = len(states)

    for key in (
        "total_simulation_time",
        "solver_sample_time",
        "initial_state",
        "default_target",
        "trajectory_mode",
    ):
        if key not in pre_launch:
            errors.append(f"pre_launch missing key: {key}")

    if errors:
        return ValidationResult(ok=False, errors=errors)

    try:
        t_sim = float(pre_launch["total_simulation_time"])
        dt = float(pre_launch["solver_sample_time"])
    except (TypeError, ValueError):
        return ValidationResult(ok=False, errors=["simulation times must be numeric"])

    if t_sim <= 0:
        errors.append("total_simulation_time must be > 0")
    if dt <= 0:
        errors.append("solver_sample_time must be > 0")
    elif t_sim > 0 and dt > t_sim / 100.0:
        errors.append("solver_sample_time must be <= total_simulation_time / 100")

    x0 = pre_launch.get("initial_state")
    if not isinstance(x0, list) or len(x0) != n:
        errors.append(f"initial_state must be a list of length {n}")

    target = pre_launch.get("default_target")
    if not isinstance(target, list) or len(target) != n:
        errors.append(f"default_target must be a list of length {n}")

    mode = pre_launch.get("trajectory_mode")
    if mode not in ("reg", "sin", "pulse"):
        errors.append("trajectory_mode must be one of: reg, sin, pulse")

    outputs = metadata.get("outputs") or []
    state_set = set(states)
    for o in outputs:
        if o not in state_set:
            errors.append(f"output {o!r} not in states")

    return ValidationResult(ok=not errors, errors=errors)
