"""Unified Pre-Launch Panel — module-agnostic simulation config form.

Collects parameters shared by AgentAdaptive and AgentMPC, then compiles
the artifact via PlantCompiler + ArtifactStore.

Reference trajectories are NOT configured here: each downstream module
owns its own (AgentMPC Scenario tab; AgentAdaptive Clarifier / sim knobs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from backend_core.plant_compiler import (
    PlantCompiler,
    default_pre_launch,
    validate_pre_launch,
)
from backend_core.artifact_store import ArtifactStore


def _n_states_from_plant(plant_output: dict) -> int:
    meta = plant_output.get("metadata") or {}
    states = meta.get("states") or []
    return len(states) if isinstance(states, list) else 0


def render_pre_launch_form(
    plant_output: dict,
    *,
    store: Optional[ArtifactStore] = None,
    key_prefix: str = "prelaunch",
) -> Optional[str]:
    """Render the pre-launch form. Returns artifact_id on successful compile, else None.

    Expected to be called when AgentPlant has emitted status=complete with
    plant_output containing system_name, python_code, and preferably metadata.
    """
    if not plant_output or not plant_output.get("system_name"):
        st.info("Complete a plant model first.")
        return None

    system_name = plant_output["system_name"]
    meta = plant_output.get("metadata") or {}
    n_states = _n_states_from_plant(plant_output)
    states = list(meta.get("states") or [f"x{i+1}" for i in range(n_states)])
    outputs = list(meta.get("outputs") or (states[-1:] if states else []))

    st.subheader(f"Pre-Launch Config — {system_name}")

    if not meta:
        st.warning(
            "This plant was produced without metadata (legacy). "
            "Downstream Adaptive/MPC integration will be limited."
        )

    compiler = PlantCompiler()
    validation = compiler.validate(plant_output)
    if validation.warnings:
        for w in validation.warnings:
            st.warning(w)
    if not validation.ok:
        st.error("Plant validation failed:")
        for e in validation.errors:
            st.write(f"- {e}")
        st.caption(
            "Send these errors back to AgentPlant to repair the equations "
            "without losing conversation history."
        )
        if st.button(
            "Send errors back to AgentPlant →",
            type="primary",
            key=f"{key_prefix}_send_errors_back",
        ):
            st.session_state["plant_validation_errors"] = list(validation.errors)
            st.session_state["final_result"] = None
            st.session_state["stage"] = "plant"
            st.rerun()
        return None

    defaults = default_pre_launch(n_states)

    col1, col2 = st.columns(2)
    with col1:
        total_sim = st.number_input(
            "Total simulation time (s)",
            min_value=0.01,
            value=float(defaults["total_simulation_time"]),
            step=0.5,
            key=f"{key_prefix}_tsim",
        )
    with col2:
        dt = st.number_input(
            "Solver sample time (s)",
            min_value=1e-6,
            value=float(defaults["solver_sample_time"]),
            format="%.6f",
            key=f"{key_prefix}_dt",
        )
        st.caption(
            "Hint: AgentMPC may suggest a different dt after plugin load "
            "(estimate_dt). Accept or override there."
        )

    st.markdown("**Initial state**")
    x0_cols = st.columns(min(n_states, 4) or 1)
    x0: List[float] = []
    for i, name in enumerate(states):
        with x0_cols[i % len(x0_cols)]:
            val = st.number_input(
                name,
                value=0.0 if i != 2 else 0.1,  # mild tip for cart-pole angle
                key=f"{key_prefix}_x0_{i}",
                format="%.4f",
            )
            x0.append(float(val))

    st.markdown("**Default target**")
    st.caption(
        "Used as the MPC plugin default_target (regulation setpoint vector). "
        "Reference trajectories are configured inside each module "
        "(MPC Scenario tab; Adaptive Clarifier / sim knobs)."
    )
    tgt_cols = st.columns(min(n_states, 4) or 1)
    target: List[float] = []
    for i, name in enumerate(states):
        with tgt_cols[i % len(tgt_cols)]:
            val = st.number_input(
                f"target {name}",
                value=0.0,
                key=f"{key_prefix}_tgt_{i}",
                format="%.4f",
            )
            target.append(float(val))

    if outputs:
        st.caption(f"Tracked outputs: {outputs}")

    pre_launch = {
        "total_simulation_time": float(total_sim),
        "solver_sample_time": float(dt),
        "initial_state": x0,
        "default_target": target,
    }

    pl_val = validate_pre_launch(pre_launch, meta if meta else {"states": states, "outputs": outputs})
    if not pl_val.ok:
        st.error("Pre-launch validation failed:")
        for e in pl_val.errors:
            st.write(f"- {e}")
        return None

    if st.button("Compile & Save Artifact", type="primary", key=f"{key_prefix}_compile"):
        try:
            art = compiler.compile_artifact(plant_output, pre_launch)
            if store is None:
                store = ArtifactStore(base_dir=str(Path("artifacts").resolve()))
            artifact_id = store.save(art)
            st.success(f"Artifact saved: **{artifact_id}**")
            st.session_state["last_artifact_id"] = artifact_id
            st.session_state["last_artifact"] = store.load(artifact_id)
            return artifact_id
        except Exception as exc:  # noqa: BLE001
            st.error(f"Compile failed: {exc}")
            return None

    return None


def list_and_select_artifact(
    store: Optional[ArtifactStore] = None,
    key_prefix: str = "artifact_select",
) -> Optional[str]:
    """Sidebar/helper to pick an existing artifact. Returns artifact_id or None."""
    if store is None:
        store = ArtifactStore(base_dir=str(Path("artifacts").resolve()))
    items = store.list_artifacts()
    if not items:
        st.info("No artifacts yet. Complete a plant and pre-launch first.")
        return None
    labels = [f"{it['artifact_id']}  ({it.get('system_name', '')})" for it in items]
    ids = [it["artifact_id"] for it in items]
    choice = st.selectbox("Select artifact", options=list(range(len(ids))), format_func=lambda i: labels[i], key=f"{key_prefix}_sel")
    return ids[choice]
