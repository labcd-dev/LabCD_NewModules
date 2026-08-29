"""PlantCompiler: validate enriched AgentPlant output and generate downstream artifacts.

Converts AgentPlant (python_code + metadata) + pre-launch config into:
- AgentMPC BaseDynamics plugin (.py source)
- AgentAdaptive system_spec dict
"""

from __future__ import annotations

import hashlib
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



# Identifiers that may appear in sympy RHS without being states/inputs/params.
_MATH_NAMES = frozenset({
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "exp", "log", "sqrt", "Abs", "abs", "sign",
    "sinh", "cosh", "tanh", "Heaviside", "pi", "E", "t",
})

_SYMPY_IMPORT_RE = re.compile(
    r"^\s*(from\s+sympy(\.[\w.]+)?\s+import\s+[^\n]+|import\s+sympy(\s+as\s+\w+)?)\s*$",
    re.MULTILINE,
)
_BARE_MATH_FN_RE = re.compile(
    r"(?<![\w.])(sin|cos|tan|asin|acos|atan|atan2|exp|log|sqrt|sinh|cosh|tanh|abs|sign)\s*\("
)


def _identifiers_in_expr(expr: str) -> List[str]:
    """Rough identifier scan for free names in a sympy-style expression."""
    return re.findall(r"\b[A-Za-z_][A-Za-z_0-9]*\b", expr or "")


def sanitize_python_code(python_code: str) -> str:
    """Make AgentPlant python_code safe for numeric MPC plugins.

    - Drop sympy imports (AgentMPC evaluates with numpy arrays).
    - Rewrite bare sin(/cos(/... to np.sin(/np.cos(/... when not already qualified.
    """
    if not python_code:
        return python_code
    code = _SYMPY_IMPORT_RE.sub("", python_code)

    def _repl(m: re.Match) -> str:
        fn = m.group(1)
        return f"np.{fn}("

    code = _BARE_MATH_FN_RE.sub(_repl, code)
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code.strip() + ("\n" if python_code.endswith("\n") else "")


def align_equation_inputs(eqs: List[str], inputs: List[str]) -> List[str]:
    """If equations use bare ``u`` but the sole declared input is e.g. ``tau``, rewrite.

    Adaptive's structure_build only injects declared input names into the symbol
    map. A free ``u`` in the RHS leaves the control channel at 0 and yields
    ComplexInfinity during Lie-derivative / relative-degree calculations.
    """
    if not isinstance(eqs, list) or not inputs:
        return eqs
    if len(inputs) != 1:
        return eqs
    in_name = inputs[0]
    if in_name == "u":
        return eqs
    out: List[str] = []
    for eq in eqs:
        if not isinstance(eq, str):
            out.append(eq)
            continue
        if re.search(r"\bu\b", eq) and not re.search(rf"\b{re.escape(in_name)}\b", eq):
            out.append(re.sub(r"\bu\b", in_name, eq))
        else:
            out.append(eq)
    return out



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
            _NP_QUALIFIED_RE = re.compile(
                r"\bnp\.(sin|cos|tan|asin|acos|atan|atan2|exp|log|sqrt|abs|sign|"
                r"sinh|cosh|tanh|pi|e)\b",
                re.IGNORECASE,
            )
            for i, eq in enumerate(eqs):
                if not isinstance(eq, str) or not eq.strip():
                    errors.append(f"state_equations[{i}] is empty")
                    continue
                np_hit = _NP_QUALIFIED_RE.search(eq)
                if np_hit:
                    errors.append(
                        f"state_equations[{i}] not sympy-parseable: {eq!r} "
                        f"(uses numpy-qualified '{np_hit.group(0)}' — rewrite with "
                        f"bare sympy function names: sin, cos, exp, ... — no np./numpy. prefixes)"
                    )
                    continue
                try:
                    sp.sympify(eq, locals=local_dict)
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    if "has no attribute" in msg and "Symbol" in msg:
                        errors.append(
                            f"state_equations[{i}] not sympy-parseable: {eq!r} "
                            f"({msg}). Use bare sympy names (sin, cos, exp, ...) "
                            f"— no module prefixes such as np. or math."
                        )
                    else:
                        errors.append(
                            f"state_equations[{i}] not sympy-parseable: {eq!r} ({exc})"
                        )

        # Flag free identifiers that are not states/inputs/params. Special case:
        # sole input is e.g. ``tau`` but equations say ``u`` — auto-fixed at
        # adaptive-spec generation; emit a warning, not a hard error.
        if isinstance(eqs, list):
            allowed = set()
            for name in list(states) + list(inputs) + list(params.keys()):
                if isinstance(name, str):
                    allowed.add(name)
            allowed |= _MATH_NAMES
            aligned = align_equation_inputs(list(eqs), list(inputs) if isinstance(inputs, list) else [])
            for i, eq in enumerate(eqs):
                if not isinstance(eq, str):
                    continue
                unknown = sorted({
                    tok for tok in _identifiers_in_expr(eq)
                    if tok not in allowed and not tok.isnumeric()
                })
                if not unknown:
                    continue
                # Would alignment remove the unknowns?
                aligned_eq = aligned[i] if i < len(aligned) else eq
                still = sorted({
                    tok for tok in _identifiers_in_expr(aligned_eq)
                    if tok not in allowed and not tok.isnumeric()
                })
                if still:
                    errors.append(
                        f"state_equations[{i}] uses unknown identifier(s) {still} "
                        f"(not in states={list(states)}, inputs={list(inputs)}, "
                        f"parameters={list(params.keys())}). Use the same names as "
                        f"metadata.inputs (e.g. if inputs=['tau'], write tau not u)."
                    )
                else:
                    warnings.append(
                        f"state_equations[{i}] used {{u}} but inputs={list(inputs)}; "
                        f"will rewrite to '{inputs[0]}' for Adaptive."
                    )

        # python_code must not import sympy for numeric simulation
        if isinstance(python_code, str) and re.search(
            r"(from\s+sympy|import\s+sympy)", python_code
        ):
            warnings.append(
                "python_code imports sympy; PlantCompiler will rewrite it to numpy "
                "for the MPC plugin. Prefer `import numpy as np` and np.sin/np.cos."
            )

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

        # Sanitize: strip sympy imports, rewrite bare sin→np.sin, etc.
        user_code = sanitize_python_code(python_code.strip())

        source = f'''# Auto-generated by PlantCompiler from AgentPlant output
# System: {system_name}
# Do not edit by hand unless you know the AgentMPC plugin contract.
import numpy as np
from backend_core.AgentMPC.dynamics.base import BaseDynamics, SystemConfig

# --- user-provided dynamics (sanitized for numpy) ---
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
        # Always pass 1-D float arrays. AgentPlant python_code is expected to
        # index u[0], u[1], ... (never treat u as a bare Python float).
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        u_arr = np.atleast_1d(np.asarray(u, dtype=float).reshape(-1))
        out = dynamics(0.0, x_arr, u_arr)
        out_arr = np.atleast_1d(np.asarray(out, dtype=float)).reshape(-1)
        if out_arr.size != {n_states}:
            raise ValueError(
                f"dynamics returned shape {{out_arr.shape}}, expected ({n_states},)"
            )
        return out_arr
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
        eqs = align_equation_inputs(list(meta.get("state_equations") or []), inputs)
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

        # References are owned by AgentAdaptive (Clarifier / sim knobs).
        # Pre-Launch does not define trajectory; leave empty for Adaptive to fill.
        references: List[Dict[str, str]] = []

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
    """Return a pre-launch config with sensible defaults.

    Trajectory / reference knobs are intentionally absent: each downstream
    module owns its own reference configuration.
    """
    return {
        "total_simulation_time": 10.0,
        "solver_sample_time": 0.001,
        "initial_state": [0.0] * n_states,
        "default_target": [0.0] * n_states,
    }


def validate_pre_launch(pre_launch: dict, metadata: dict) -> ValidationResult:
    """Validate pre-launch knobs against metadata."""
    errors: List[str] = []
    states = metadata.get("states") or []
    n = len(states)

    for key in (
        "total_simulation_time",
        "solver_sample_time",
        "initial_state",
        "default_target",
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

    outputs = metadata.get("outputs") or []
    state_set = set(states)
    for o in outputs:
        if o not in state_set:
            errors.append(f"output {o!r} not in states")

    return ValidationResult(ok=not errors, errors=errors)
