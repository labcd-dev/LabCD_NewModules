"""LabCD Unified App — Plant → Pre-Launch → Module selection.

Run from repository root:

    PYTHONPATH=. streamlit run frontend_streamlit/unified_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from labcd_agents import ensure_env_loaded
    _env = _REPO_ROOT / ".env"
    ensure_env_loaded(str(_env) if _env.is_file() else None)
except ImportError:
    pass

from backend_core.AgentPlant.agent import (
    DEFAULT_MAX_DRAFTS,
    DEFAULT_MIN_USER_TURNS_BEFORE_COMPLETION,
    PlantModelAgent,
)
from backend_core.artifact_store import ArtifactStore
from frontend_streamlit.pre_launch_panel import render_pre_launch_form, list_and_select_artifact

DEFAULT_MODEL = os.getenv("LABCD_DEMO_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = 0.0
ARTIFACTS_DIR = _REPO_ROOT / "artifacts"

st.set_page_config(page_title="LabCD Unified", page_icon="🔬", layout="wide")


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("final_result", None)
    st.session_state.setdefault("agent", None)
    st.session_state.setdefault("agent_key", None)
    st.session_state.setdefault("agent_error", None)
    st.session_state.setdefault("last_artifact_id", None)
    st.session_state.setdefault("last_artifact", None)
    st.session_state.setdefault("stage", "plant")  # plant | prelaunch | module


def _reset_plant() -> None:
    st.session_state["messages"] = []
    st.session_state["final_result"] = None
    st.session_state["agent"] = None
    st.session_state["agent_key"] = None
    st.session_state["agent_error"] = None
    st.session_state.pop("latest_plant_draft", None)
    st.session_state.pop("plant_draft_count", None)
    st.session_state.pop("plant_validation_errors", None)
    st.session_state["stage"] = "plant"


def _get_agent(model: str, temperature: float) -> PlantModelAgent:
    key = f"{model}|{temperature}"
    if st.session_state.get("agent") is None or st.session_state.get("agent_key") != key:
        agent = PlantModelAgent(
            model=model,
            temperature=temperature,
            max_drafts=DEFAULT_MAX_DRAFTS,
            min_user_turns_before_completion=DEFAULT_MIN_USER_TURNS_BEFORE_COMPLETION,
        )
        saved = st.session_state.get("latest_plant_draft")
        if isinstance(saved, dict) and saved.get("system_name") and saved.get("python_code"):
            agent._latest_draft = dict(saved)
            agent._draft_count = max(1, int(st.session_state.get("plant_draft_count") or 1))
        st.session_state["agent"] = agent
        st.session_state["agent_key"] = key
    return st.session_state["agent"]


def _render_plant_stage() -> None:
    st.header("1. Plant Model (AgentPlant)")
    col_a, col_b = st.columns([3, 1])
    with col_b:
        model = st.text_input("Model", value=DEFAULT_MODEL, key="plant_model")
        temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05, key="plant_temp")
        if st.button("Reset conversation"):
            _reset_plant()
            st.rerun()

    agent = _get_agent(model, temperature)

    pending_errors = st.session_state.pop("plant_validation_errors", None)
    if pending_errors:
        err_text = "\n".join(f"- {e}" for e in pending_errors)
        repair_msg = (
            "Pre-Launch validation rejected the plant metadata. Please fix the "
            "issues below and resubmit status \"draft\" with corrected "
            "state_equations using bare sympy function names (sin, cos, exp, "
            "... — no np./numpy. prefixes):\n"
            f"{err_text}"
        )
        st.session_state["messages"].append({"role": "user", "content": repair_msg})
        try:
            display, final = agent.step(st.session_state["messages"][:-1], repair_msg)
            st.session_state["messages"].append({"role": "assistant", "content": display})
            if agent._latest_draft is not None:
                st.session_state["latest_plant_draft"] = dict(agent._latest_draft)
                st.session_state["plant_draft_count"] = agent._draft_count
            if final is not None:
                st.session_state["final_result"] = final
                st.session_state.pop("latest_plant_draft", None)
        except Exception as exc:  # noqa: BLE001
            st.session_state["agent_error"] = str(exc)
            st.error(f"Agent error during repair: {exc}")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.get("final_result"):
        st.success(f"Plant complete: **{st.session_state['final_result'].get('system_name')}**")
        if st.button("Continue to Pre-Launch →", type="primary"):
            st.session_state["stage"] = "prelaunch"
            st.rerun()
        return

    user_msg = st.chat_input("Describe your plant or refine the draft…")
    if user_msg:
        st.session_state["messages"].append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        try:
            display, final = agent.step(st.session_state["messages"][:-1], user_msg)
            st.session_state["messages"].append({"role": "assistant", "content": display})
            with st.chat_message("assistant"):
                st.markdown(display)
            if agent._latest_draft is not None:
                st.session_state["latest_plant_draft"] = dict(agent._latest_draft)
                st.session_state["plant_draft_count"] = agent._draft_count
            if final is not None:
                st.session_state["final_result"] = final
                st.session_state.pop("latest_plant_draft", None)
                st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.session_state["agent_error"] = str(exc)
            st.error(f"Agent error: {exc}")


def _render_prelaunch_stage() -> None:
    st.header("2. Pre-Launch Configuration")
    plant = st.session_state.get("final_result")
    if not plant:
        st.warning("No completed plant. Go back to Plant stage.")
        if st.button("← Back to Plant"):
            st.session_state["stage"] = "plant"
            st.rerun()
        return

    store = ArtifactStore(base_dir=str(ARTIFACTS_DIR))
    artifact_id = render_pre_launch_form(plant, store=store, key_prefix="unified_pl")
    if artifact_id or st.session_state.get("last_artifact_id"):
        if st.button("Continue to Module Selection →", type="primary"):
            st.session_state["stage"] = "module"
            st.rerun()

    if st.button("← Back to Plant"):
        st.session_state["stage"] = "plant"
        st.rerun()


def _render_module_stage() -> None:
    st.header("3. Launch Module")
    store = ArtifactStore(base_dir=str(ARTIFACTS_DIR))
    artifact_id = st.session_state.get("last_artifact_id")
    if not artifact_id:
        artifact_id = list_and_select_artifact(store=store, key_prefix="unified_mod")
    else:
        st.info(f"Using artifact: **{artifact_id}**")
        if st.button("Choose a different artifact"):
            st.session_state["last_artifact_id"] = None
            st.rerun()

    if not artifact_id:
        return

    try:
        data = store.load(artifact_id)
    except FileNotFoundError:
        st.error(f"Artifact {artifact_id} not found.")
        return

    st.json(
        {
            "artifact_id": data.get("artifact_id"),
            "system_name": data.get("system_name"),
            "created_at": data.get("created_at"),
            "pre_launch": data.get("pre_launch"),
        }
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("AgentAdaptive")
        st.caption("Clarifier → Designer → Tuner")
        st.markdown(
            "Launch Adaptive with the compiled `system_spec` (plant + pre-launch references)."
        )
        if st.button("Open AgentAdaptive", key="launch_adaptive"):
            st.session_state["launch_module"] = "adaptive"
            st.session_state["launch_artifact_id"] = artifact_id
            st.info(
                "Wire this into agent_adaptive_app.py: pass "
                f"`store.get_adaptive_spec('{artifact_id}')` as the starting system_spec."
            )
            # Practical hand-off: write a small launch hint file
            hint = ARTIFACTS_DIR / f"{artifact_id}_adaptive_spec.json"
            import json
            with open(hint, "w", encoding="utf-8") as f:
                json.dump(store.get_adaptive_spec(artifact_id), f, indent=2)
            st.success(f"Adaptive spec written to {hint.name}")

    with col2:
        st.subheader("AgentMPC")
        st.caption("Plugin loaded → SetupAgent → Actor loop")
        plugin_path = store.load_plugin_path(artifact_id)
        st.code(plugin_path, language=None)
        if st.button("Open AgentMPC", key="launch_mpc"):
            st.session_state["launch_module"] = "mpc"
            st.session_state["launch_artifact_id"] = artifact_id
            st.info(
                "Wire this into agent_mpc_app.py: load plugin from "
                f"`{plugin_path}` and seed Config from pre_launch."
            )
            st.success("Plugin path ready for DynamicLoader.load_from_path")

    if st.button("← Back to Pre-Launch"):
        st.session_state["stage"] = "prelaunch"
        st.rerun()


def main() -> None:
    _init_state()
    st.title("LabCD Unified System")
    st.caption("AgentPlant → Plant Compiler → Pre-Launch → Adaptive / MPC")

    stages = {"plant": "1. Plant", "prelaunch": "2. Pre-Launch", "module": "3. Module"}
    stage = st.session_state.get("stage", "plant")
    cols = st.columns(len(stages))
    for i, (key, label) in enumerate(stages.items()):
        with cols[i]:
            if st.button(label, key=f"nav_{key}", type="primary" if key == stage else "secondary"):
                st.session_state["stage"] = key
                st.rerun()

    st.divider()
    if stage == "plant":
        _render_plant_stage()
    elif stage == "prelaunch":
        _render_prelaunch_stage()
    else:
        _render_module_stage()


if __name__ == "__main__":
    main()
