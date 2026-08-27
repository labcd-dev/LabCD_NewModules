"""LabCD Plant-Model Agent — Streamlit UI.

Run from repository root:

    PYTHONPATH=. streamlit run frontend_streamlit/agent_plant_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

# Repo root on path
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

DEFAULT_MODEL = os.getenv("LABCD_DEMO_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = 0.0

st.set_page_config(page_title="LabCD Plant-Model Agent", page_icon="🧪", layout="wide")


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("final_result", None)
    st.session_state.setdefault("agent", None)
    st.session_state.setdefault("agent_key", None)
    st.session_state.setdefault("agent_error", None)


def _reset_conversation() -> None:
    st.session_state["messages"] = []
    st.session_state["final_result"] = None
    st.session_state["agent_error"] = None
    agent = st.session_state.get("agent")
    if agent is not None and hasattr(agent, "reset_conversation_state"):
        agent.reset_conversation_state()


def _get_or_create_agent(
    model: str, temperature: float, provider: Optional[str], min_turns: int, max_drafts: int
):
    key = (model, temperature, provider, min_turns, max_drafts)
    if st.session_state["agent"] is not None and st.session_state["agent_key"] == key:
        return st.session_state["agent"], None

    try:
        agent = PlantModelAgent(
            model=model,
            temperature=temperature,
            provider=provider,
            min_user_turns_before_completion=min_turns,
            max_drafts=max_drafts,
        )
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)

    st.session_state["agent"] = agent
    st.session_state["agent_key"] = key
    st.session_state["agent_error"] = None
    return agent, None


def _render_sidebar() -> tuple[str, float, Optional[str], int, int]:
    with st.sidebar:
        st.header("🧪 LabCD Plant-Model Agent")
        st.caption("Demo chatbot built on `labcd_agents`.")

        model = st.text_input(
            "Model",
            value=DEFAULT_MODEL,
            help="Model name recognized by labcd_agents.LLMFactory.",
        )
        provider_input = st.text_input(
            "Provider override (optional)",
            value="",
            help="Leave blank to auto-detect from the model name.",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05)
        provider = provider_input.strip() or None

        min_turns = st.slider(
            "Min. user turns before auto-complete",
            1,
            5,
            DEFAULT_MIN_USER_TURNS_BEFORE_COMPLETION,
            1,
            help=(
                "Soft floor before accepting status complete without an explicit "
                "user accept. Explicit finish/accept after a draft always wins."
            ),
        )
        max_drafts = st.slider(
            "Max drafts before auto-accept",
            1,
            10,
            DEFAULT_MAX_DRAFTS,
            1,
            help=(
                "After this many draft iterations the latest draft is accepted "
                "even if the user never says finish."
            ),
        )

        st.divider()
        if st.button("🔄 Reset conversation", use_container_width=True):
            _reset_conversation()
            st.rerun()

        st.divider()
        st.subheader("Session stats")
        agent = st.session_state.get("agent")
        if agent is not None:
            usage = agent.total_usage
            col1, col2 = st.columns(2)
            col1.metric("Input tokens", usage.input_tokens)
            col2.metric("Output tokens", usage.output_tokens)
            st.metric("Estimated cost", f"${agent.total_cost:.6f}")
            st.caption(f"Drafts this conversation: {getattr(agent, '_draft_count', 0)}")
        else:
            st.caption("No LLM calls yet this session.")

        if st.session_state.get("final_result"):
            st.divider()
            st.subheader("System identified")
            st.write(st.session_state["final_result"].get("system_name", "—"))

        st.divider()
        st.caption(
            "Provider construction, retries, tokens and cost come from "
            "`labcd_agents.BaseAgent`."
        )

    return model, temperature, provider, min_turns, max_drafts


def main() -> None:
    _init_state()
    model, temperature, provider, min_turns, max_drafts = _render_sidebar()

    st.title("🧪 LabCD Plant-Model Agent — Demo")
    st.caption(
        "Describe your system. The agent will ask only as long as needed, then "
        "sketch a draft `dynamics(t, x, u)` for you to refine. Say **finish** "
        "(or keep iterating until the draft limit) to lock it in."
    )

    if st.session_state.get("agent_error"):
        st.error(f"Could not initialize the LLM client: {st.session_state['agent_error']}")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    final_result = st.session_state.get("final_result")
    if final_result:
        st.success(f"✅ Model complete — **{final_result.get('system_name', 'Unnamed system')}**")
        python_code = final_result.get("python_code", "")
        st.code(python_code, language="python")
        st.download_button(
            "⬇️ Download dynamics.py",
            data=python_code,
            file_name="dynamics.py",
            mime="text/x-python",
            use_container_width=False,
        )

    chat_disabled = final_result is not None
    placeholder = (
        "Conversation complete — reset to design a new system."
        if chat_disabled
        else "Describe your system, request changes to a draft, or say finish..."
    )
    user_message = st.chat_input(placeholder, disabled=chat_disabled)

    if not user_message:
        return

    agent, error = _get_or_create_agent(model, temperature, provider, min_turns, max_drafts)
    if error:
        st.session_state["agent_error"] = error
        st.rerun()

    history_before_this_turn = list(st.session_state["messages"])

    st.session_state["messages"].append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response_text, final_result = agent.step(history_before_this_turn, user_message)
            except Exception as exc:  # noqa: BLE001
                response_text, final_result = f"⚠️ LLM call failed: {exc}", None

        display_text = response_text
        st.markdown(display_text)

    st.session_state["messages"].append({"role": "assistant", "content": display_text})
    if final_result:
        st.session_state["final_result"] = final_result

    st.rerun()


if __name__ == "__main__":
    main()