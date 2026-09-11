"""LabCD Plant-Model Agent — Streamlit UI.

Run from repository root:

    PYTHONPATH=. streamlit run frontend_streamlit/agent_plant_app.py

LAYOUT
--------------------------------------------------------------------------------
The agent's draft messages carry a full ``dynamics(t, x, u)`` listing plus a
metadata block, so a plain single-column chat re-dumps the entire model on every
turn and the conversation becomes unreadable after two or three drafts. The
screen is therefore split: the transcript stays a conversation on the left,
while the right-hand inspector shows the *current* model as structured panels --
state table, equations, parameters, assumptions, code. The transcript is
history; the inspector is state.

THEME
--------------------------------------------------------------------------------
Colours mirror the palette in ``agent_mpc_ui_components.py`` so the Plant agent
reads as part of the same product rather than a separate demo. They are
redeclared here rather than imported: this is AgentPlant's UI, and importing
AgentMPC's module to borrow a dict would couple two otherwise independent apps.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

st.set_page_config(
    page_title="LabCD Plant-Model Agent",
    page_icon=":material/science:",
    layout="wide",
)


# ============================================================================
# THEME
# ============================================================================

_CSS = """
<style>
  :root {
    --bg:#0a0d12; --panel:#11161d; --elevated:#161d27;
    --border:rgba(255,255,255,0.07); --border-soft:rgba(255,255,255,0.045);
    --text:#eef2f8; --text-2:#8d9bb0; --muted:#556074;
    --accent:#5b7fff; --accent-2:#8b7dff; --plant:#22d3a7;
    --success:#22d3a7; --warn:#f0b429; --danger:#f0576b;
    --mono:"IBM Plex Mono","JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
    --sans:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  }
  .stApp { background:var(--bg); }
  html, body, .stApp, [data-testid="stMarkdownContainer"] {
      font-family:var(--sans); color:var(--text); font-size:14px; }
  [data-testid="stHeader"] { display:none; }
  .block-container { padding:0.8rem 1.6rem 6rem 1.6rem !important; max-width:1560px; }
  [data-testid="stIconMaterial"], [class*="material-symbol"], [class*="material-icon"] {
      font-family:'Material Symbols Rounded','Material Icons',sans-serif !important; }

  /* ---- app header ---- */
  .ph { display:flex; align-items:center; justify-content:space-between; gap:16px;
        padding:14px 18px; margin-bottom:14px; border-radius:12px;
        background:linear-gradient(135deg, var(--panel) 0%, var(--elevated) 100%);
        border:1px solid var(--border); }
  .ph .l { display:flex; align-items:center; gap:13px; }
  .ph .mark { width:38px; height:38px; border-radius:10px; flex-shrink:0;
        display:flex; align-items:center; justify-content:center;
        background:rgba(34,211,167,0.11); border:1px solid rgba(34,211,167,0.28);
        color:var(--plant); }
  .ph .mark svg { width:20px; height:20px; }
  .ph h1 { font-size:17px; font-weight:700; margin:0; letter-spacing:-0.2px; color:var(--text); }
  .ph p  { font-size:12px; color:var(--text-2); margin:2px 0 0 0; max-width:78ch; line-height:1.45; }

  /* ---- status pill ---- */
  .pill { display:inline-flex; align-items:center; gap:6px; flex-shrink:0;
        font-family:var(--mono); font-size:10.5px; font-weight:700; letter-spacing:0.6px;
        text-transform:uppercase; padding:6px 11px; border-radius:999px;
        background:var(--elevated); border:1px solid var(--border); color:var(--text-2); }
  .pill .dot { width:6px; height:6px; border-radius:50%; background:currentColor; }
  .pill.idle     { color:var(--muted); }
  .pill.drafting { color:var(--warn);    border-color:rgba(240,180,41,0.32);
                   background:rgba(240,180,41,0.08); }
  .pill.done     { color:var(--success); border-color:rgba(34,211,167,0.32);
                   background:rgba(34,211,167,0.08); }

  /* ---- section tag above each panel ---- */
  .tag { display:inline-flex; align-items:center; gap:6px; font-family:var(--mono);
        font-size:10px; font-weight:700; letter-spacing:1.1px; text-transform:uppercase;
        color:var(--muted); margin:2px 0 7px 2px; }
  .tag svg { width:13px; height:13px; }

  /* ---- key/value grid for parameters ---- */
  .kv { display:grid; grid-template-columns:repeat(auto-fill,minmax(132px,1fr)); gap:7px; }
  .kv .c { background:var(--elevated); border:1px solid var(--border-soft);
        border-radius:8px; padding:8px 10px; }
  .kv .c .k { font-size:10px; color:var(--muted); font-family:var(--mono);
        text-transform:uppercase; letter-spacing:0.5px; }
  .kv .c .v { font-size:14px; font-weight:700; font-family:var(--mono);
        color:var(--text); margin-top:2px; }

  /* ---- equation rows ---- */
  .eq { background:var(--elevated); border:1px solid var(--border-soft); border-radius:8px;
        padding:8px 11px; margin-bottom:6px; font-family:var(--mono); font-size:12px;
        overflow-x:auto; }
  .eq .lhs { color:var(--plant); font-weight:700; }
  .eq .rhs { color:var(--text); }
  .eq .why { display:block; color:var(--muted); font-family:var(--sans);
        font-size:11px; margin-top:3px; }

  .note { color:var(--text-2); font-size:12px; line-height:1.55; margin:0 0 6px 0; }
  .empty { color:var(--muted); font-size:12.5px; font-style:italic;
        padding:22px 14px; text-align:center; border:1px dashed var(--border);
        border-radius:10px; background:var(--panel); }

  /* ---- streamlit surfaces ---- */
  [data-testid="stChatMessage"] { background:var(--panel); border:1px solid var(--border-soft);
        border-radius:11px; padding:11px 13px; margin-bottom:9px; }
  [data-testid="stVerticalBlockBorderWrapper"] { border-radius:12px; }
  [data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--border); }
  /* st.subheader renders an <h3>; st.header an <h2>. Style both so the
     sidebar's group labels read as section tags rather than page headings. */
  [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-size:10.5px !important; font-family:var(--mono) !important;
        font-weight:700 !important; letter-spacing:1.1px; text-transform:uppercase;
        color:var(--muted) !important; padding:0 !important;
        margin:18px 0 2px 0 !important; }
  [data-testid="stSidebar"] .ph h1 { text-transform:none; letter-spacing:-0.2px; }
  code, pre, .stCode { font-family:var(--mono) !important; }
  [data-testid="stMetricValue"] { font-family:var(--mono); font-size:19px; }
</style>
"""

# Inline SVG rather than emoji, matching the rest of the LabCD UI.
_ICON_FLASK = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
               'stroke-linecap="round" stroke-linejoin="round">'
               '<path d="M9 3h6M10 3v6L4.5 18a2 2 0 0 0 1.7 3h11.6a2 2 0 0 0 1.7-3L14 9V3"/>'
               '<path d="M7.5 14h9"/></svg>')


def _tag(label: str) -> str:
    return f'<div class="tag">{label}</div>'


# ============================================================================
# STATE
# ============================================================================

def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("final_result", None)
    st.session_state.setdefault("agent", None)
    st.session_state.setdefault("agent_key", None)
    st.session_state.setdefault("agent_error", None)
    st.session_state.setdefault("pending_prompt", None)


def _reset_conversation() -> None:
    st.session_state["messages"] = []
    st.session_state["final_result"] = None
    st.session_state["agent_error"] = None
    st.session_state["pending_prompt"] = None
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


def _current_model() -> Tuple[Optional[Dict[str, Any]], bool]:
    """The model the inspector should show, and whether it is final.

    A finished conversation has ``final_result``; before that, the agent holds
    the latest draft, which is what the user is actually iterating on.
    """
    final = st.session_state.get("final_result")
    if final:
        return final, True
    agent = st.session_state.get("agent")
    draft = getattr(agent, "_latest_draft", None) if agent is not None else None
    return (draft, False) if draft else (None, False)


# ============================================================================
# SIDEBAR
# ============================================================================

def _render_sidebar() -> Tuple[str, float, Optional[str], int, int]:
    with st.sidebar:
        st.markdown(
            f'<div class="ph" style="margin-bottom:10px;"><div class="l">'
            f'<div class="mark">{_ICON_FLASK}</div>'
            f'<div><h1>Plant-Model Agent</h1>'
            f'<p style="max-width:none;">Built on <code>labcd_agents</code></p>'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )

        st.subheader("Model")
        model = st.text_input(
            "Model", value=DEFAULT_MODEL,
            help="Model name recognized by labcd_agents.LLMFactory.",
        )
        provider_input = st.text_input(
            "Provider override", value="",
            placeholder="auto-detect",
            help="Leave blank to auto-detect from the model name.",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05)
        provider = provider_input.strip() or None

        st.subheader("Conversation")
        min_turns = st.slider(
            "Min. turns before auto-complete", 1, 5,
            DEFAULT_MIN_USER_TURNS_BEFORE_COMPLETION, 1,
            help=(
                "Soft floor before accepting status complete without an explicit "
                "user accept. Explicit finish/accept after a draft always wins."
            ),
        )
        max_drafts = st.slider(
            "Max drafts before auto-accept", 1, 10, DEFAULT_MAX_DRAFTS, 1,
            help=(
                "After this many draft iterations the latest draft is accepted "
                "even if the user never says finish."
            ),
        )

        st.subheader("Session")
        agent = st.session_state.get("agent")
        if agent is not None:
            usage = agent.total_usage
            c1, c2 = st.columns(2)
            c1.metric("Input tokens", f"{usage.input_tokens:,}")
            c2.metric("Output tokens", f"{usage.output_tokens:,}")
            st.metric("Estimated cost", f"${agent.total_cost:.6f}")

            used = int(getattr(agent, "_draft_count", 0))
            st.caption(f"Drafts: {used} of {max_drafts}")
            st.progress(min(used / max_drafts, 1.0) if max_drafts else 0.0)
        else:
            st.caption("No LLM calls yet this session.")

        st.subheader("Actions")
        if st.button(
            "Reset conversation", icon=":material/refresh:",
            width="stretch", type="secondary",
        ):
            _reset_conversation()
            st.rerun()

        st.caption(
            "Provider construction, retries, tokens and cost come from "
            "`labcd_agents.BaseAgent`."
        )

    return model, temperature, provider, min_turns, max_drafts


# ============================================================================
# HEADER
# ============================================================================

def _render_header() -> None:
    model, is_final = _current_model()
    if is_final:
        cls, label = "done", "Model complete"
    elif model or st.session_state["messages"]:
        cls, label = "drafting", "Drafting"
    else:
        cls, label = "idle", "Awaiting description"

    st.markdown(
        f'<div class="ph"><div class="l">'
        f'<div class="mark">{_ICON_FLASK}</div>'
        f'<div><h1>Plant-Model Agent</h1>'
        f'<p>Describe your system in plain language. The agent asks only as long as it '
        f'needs to, then drafts a runnable <code>dynamics(t, x, u)</code> for you to '
        f'refine. Say <b>finish</b> to lock it in.</p></div></div>'
        f'<div class="pill {cls}"><span class="dot"></span>{label}</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# INSPECTOR
# ============================================================================

def _render_state_table(meta: Dict[str, Any]) -> None:
    states = meta.get("states") or []
    meanings = meta.get("state_meanings") or []
    equations = meta.get("state_equations") or []
    if not states:
        return

    st.markdown(_tag("State variables"), unsafe_allow_html=True)
    for i, name in enumerate(states):
        meaning = meanings[i] if i < len(meanings) else ""
        equation = equations[i] if i < len(equations) else None
        body = (
            f'<span class="lhs">d{name}/dt</span> '
            f'<span class="rhs">= {equation}</span>'
            if equation is not None
            else f'<span class="lhs">{name}</span>'
        )
        why = f'<span class="why">{meaning}</span>' if meaning else ""
        st.markdown(f'<div class="eq">{body}{why}</div>', unsafe_allow_html=True)


def _render_parameters(meta: Dict[str, Any]) -> None:
    params = meta.get("parameters") or {}
    if not isinstance(params, dict) or not params:
        return
    st.markdown(_tag("Parameters"), unsafe_allow_html=True)
    cells = "".join(
        f'<div class="c"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in params.items()
    )
    st.markdown(f'<div class="kv">{cells}</div>', unsafe_allow_html=True)


def _render_signature(meta: Dict[str, Any]) -> None:
    """System type / inputs / outputs as badges -- the one-line identity of the
    model, which is what a reader checks first."""
    bits = []
    system_type = meta.get("system_type")
    if system_type:
        bits.append(f":green-badge[{system_type}]")
    for label, key in (("in", "inputs"), ("out", "outputs")):
        values = meta.get(key) or []
        if values:
            bits.append(f":blue-badge[{label}: {', '.join(str(v) for v in values)}]")
    states = meta.get("states") or []
    if states:
        bits.append(f":gray-badge[{len(states)} states]")
    if bits:
        st.markdown(" ".join(bits))


def _render_inspector(model: Optional[Dict[str, Any]], is_final: bool) -> None:
    st.markdown(_tag("System inspector"), unsafe_allow_html=True)

    if not model:
        st.markdown(
            '<div class="empty">No model yet.<br>Describe your system in the chat '
            'and the draft will appear here.</div>',
            unsafe_allow_html=True,
        )
        return

    name = model.get("system_name") or "Unnamed system"
    if is_final:
        st.success(f"**{name}** — accepted", icon=":material/check_circle:")
    else:
        st.info(f"**{name}** — draft in progress", icon=":material/edit_note:")

    meta = model.get("metadata")
    code = model.get("python_code", "")

    structure, source, notes = st.tabs(["Structure", "Source", "Assumptions"])

    with structure:
        if isinstance(meta, dict) and meta:
            _render_signature(meta)
            _render_state_table(meta)
            _render_parameters(meta)
        else:
            st.caption("The agent has not attached structured metadata to this draft yet.")

    with source:
        if code:
            st.code(code, language="python", line_numbers=True)
            st.download_button(
                "Download dynamics.py",
                data=code,
                file_name=f"{name.strip().replace(' ', '_').lower() or 'dynamics'}.py",
                mime="text/x-python",
                icon=":material/download:",
                width="stretch",
                type="primary" if is_final else "secondary",
            )
        else:
            st.caption("No source generated yet.")

    with notes:
        assumptions = (meta or {}).get("assumptions") if isinstance(meta, dict) else None
        if assumptions:
            for item in assumptions:
                st.markdown(f"- {item}")
        else:
            st.caption("No assumptions recorded for this draft.")


# ============================================================================
# CHAT
# ============================================================================

SUGGESTIONS = {
    ":green[:material/balance:] Inverted pendulum":
        "Model an inverted pendulum on a cart, force-driven, for MPC tuning.",
    ":blue[:material/thermostat:] Heated tank":
        "Model a stirred tank heated by an electric element, temperature as the output.",
    ":violet[:material/rocket_launch:] DC motor":
        "Model a DC motor driven by armature voltage, with shaft speed as the output.",
}


# Every draft turn from the agent carries the whole dynamics listing and a
# metadata dump (see PlantModelAgent._format_draft_display). Rendered inline,
# three drafts bury the actual conversation under three copies of the model.
# Both are folded away here instead: the prose stays in the bubble, and the
# artifact itself is always one click away -- and permanently visible in the
# inspector beside it.
_FENCE_RE = re.compile(r"```[ \t]*([\w+-]*)[ \t]*\n(.*?)```", re.DOTALL)

# The first line _format_metadata_block emits. Matching on it is a deliberate
# coupling to that one string; if the agent ever stops emitting it the block
# simply renders inline again, which is the old behaviour rather than a break.
_METADATA_MARKER = "**Metadata**"

# _format_draft_display appends its "say what to change, or finish" hint *after*
# the metadata block. That hint is the call to action and has to stay visible,
# so it is lifted back out rather than folded away with the metadata.
_TRAILING_HINT_RE = re.compile(r"\n\s*(_[^\n_][^\n]*_)\s*$")


def _render_message(content: str, *, collapse: bool) -> None:
    """Render one chat message, optionally folding code and metadata away."""
    if not collapse:
        st.markdown(content)
        return

    prose, _, metadata = content.partition(_METADATA_MARKER)

    pos = 0
    for match in _FENCE_RE.finditer(prose):
        before = prose[pos:match.start()].strip()
        if before:
            st.markdown(before)
        language, code = match.group(1) or "python", match.group(2)
        with st.expander(f"{language} · {len(code.splitlines())} lines",
                          icon=":material/code:"):
            st.code(code, language=language, line_numbers=True)
        pos = match.end()

    tail = prose[pos:].strip()
    if tail:
        st.markdown(tail)

    if metadata.strip():
        hint = _TRAILING_HINT_RE.search(metadata)
        if hint:
            metadata = metadata[: hint.start()]
        with st.expander("Model metadata", icon=":material/data_object:"):
            st.markdown(_METADATA_MARKER + metadata)
        if hint:
            st.markdown(hint.group(1))


def _render_chat() -> None:
    st.markdown(_tag("Conversation"), unsafe_allow_html=True)
    messages = st.session_state["messages"]

    if not messages:
        st.markdown(
            '<p class="note">Start from scratch, or pick a system to describe:</p>',
            unsafe_allow_html=True,
        )
        picked = st.pills(
            "Try one", list(SUGGESTIONS), label_visibility="collapsed", key="suggestion_pills",
        )
        if picked:
            st.session_state["pending_prompt"] = SUGGESTIONS[picked]
            st.rerun()
        return

    for message in messages:
        avatar = ":material/person:" if message["role"] == "user" else ":material/science:"
        with st.chat_message(message["role"], avatar=avatar):
            _render_message(message["content"], collapse=message["role"] == "assistant")


# ============================================================================
# TURN
# ============================================================================

def _run_turn(
    user_message: str, model: str, temperature: float,
    provider: Optional[str], min_turns: int, max_drafts: int,
) -> None:
    agent, error = _get_or_create_agent(model, temperature, provider, min_turns, max_drafts)
    if error:
        # Keep the typed message in state: losing it would make the user retype
        # a long system description just because a model name was wrong.
        st.session_state["agent_error"] = error
        st.session_state["pending_prompt"] = user_message
        st.rerun()

    history_before_this_turn = list(st.session_state["messages"])
    st.session_state["messages"].append({"role": "user", "content": user_message})

    try:
        with st.spinner("Thinking...", show_time=True):
            response_text, final_result = agent.step(history_before_this_turn, user_message)
    except Exception as exc:  # noqa: BLE001
        response_text, final_result = f"⚠️ LLM call failed: {exc}", None

    st.session_state["messages"].append({"role": "assistant", "content": response_text})
    if final_result:
        st.session_state["final_result"] = final_result
    st.rerun()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    _init_state()
    st.markdown(_CSS, unsafe_allow_html=True)

    model, temperature, provider, min_turns, max_drafts = _render_sidebar()
    _render_header()

    if st.session_state.get("agent_error"):
        st.error(
            f"Could not initialize the LLM client: {st.session_state['agent_error']}",
            icon=":material/error:",
        )

    current_model, is_final = _current_model()

    chat_col, inspect_col = st.columns([1.15, 1], gap="large")
    with chat_col:
        _render_chat()
    with inspect_col:
        _render_inspector(current_model, is_final)

    placeholder = (
        "Conversation complete — reset to design a new system."
        if is_final
        else "Describe your system, request changes to a draft, or say finish..."
    )
    typed = st.chat_input(placeholder, disabled=is_final)

    # A suggestion chip queues its prompt and reruns, so it arrives here on the
    # following pass exactly as if the user had typed it.
    user_message = typed or st.session_state.pop("pending_prompt", None)
    if user_message and not is_final:
        _run_turn(user_message, model, temperature, provider, min_turns, max_drafts)


if __name__ == "__main__":
    main()
