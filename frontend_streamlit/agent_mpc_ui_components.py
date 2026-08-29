"""
================================================================================
ui_components.py -- Agent-MPC Studio (compact operator console)
================================================================================
A standalone Streamlit app: `streamlit run ui_components.py` (or
`python run_ui_components.py`). Does not import or modify app.py -- the two
are independent front-ends over the same AgentMPC backend.

--------------------------------------------------------------------------
SECTION MAP -- which functions build which part of the screen
--------------------------------------------------------------------------
Every visible region renders a small monospace tag (e.g. `SETUP · CONSULT`)
so anyone reading this file can match a region on screen to the function
that draws it. The tags below are the complete list:

  THEME / CHROME
    inject_theme()          global CSS: palette, panels, chat, tags
    render_section_tag()    the small tag drawn above every region
    render_topbar()         breadcrumb + run status + active model
    render_stage_rail()     numbered stage progress strip
    metric_cards()          the metric card row primitive

  SETUP · UPLOAD          section_upload()
        Dynamics file intake, validation/auto-repair, Setup Agent analysis.

  SETUP · CONSULT         section_consult()
        Human-in-the-loop chat + structured JSON config suggestions
        (agents/config_advisor_agent.py provides both).

  LAUNCH · GENERAL        section_launch_general()
        Controller-agnostic settings: simulation time, settling tolerance,
        iteration budget, exploration behavior + constraint bounds.

  LAUNCH · MPC            section_launch_mpc()
        MPC-specific: scenario level, reference trajectory, seed Np/Nc/Q/R,
        dt, optimization focus, free-text guidance.

  RUN · METRICS           panel_metrics()          live metric cards
  RUN · PLOTS             panel_plots()            selector + chart canvas
  RUN · STATE FLOW        panel_state_flow()       agent graph position
  RUN · CANDIDATE         panel_candidate()        current parameters
  RUN · ACTIVITY          panel_activity()         agent event feed
  RUN · REASONING         panel_reasoning()        per-agent prompt/output
  RUN · TABLE             panel_iteration_table()  every iteration's numbers

  RESULT · EXPORT         section_export()
        PDF report + standalone .py script.

--------------------------------------------------------------------------
No emoji anywhere -- icons are inline SVG (ICON_* constants).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback as tb_module

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Agent-MPC Studio", page_icon=None, layout="wide",
                    initial_sidebar_state="collapsed")

# Repo root (parent of backend_core/ and frontend_streamlit/) must be on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend_core.AgentMPC.agents.config_advisor_agent import (
    chat as advisor_chat,
    clamp_general_settings,
    suggest_config,
)
from backend_core.AgentMPC.agents.dynamics_validator import analyze_and_setup
from backend_core.AgentMPC.agents.export_script import generate_standalone_script
from backend_core.AgentMPC.agents.formatting import fmt_num
from backend_core.AgentMPC.agents.llm_base import configure_llm
from backend_core.AgentMPC.agents.metrics import OPTIMIZATION_FOCUS_LABELS, OPTIMIZATION_FOCUS_PRESETS
from backend_core.AgentMPC.agents.report_agent import generate_report_analysis
from backend_core.AgentMPC.agents.report import build_pdf_report
from backend_core.AgentMPC.agents.scenario_presets import apply_scenario_level
from backend_core.AgentMPC.agents.seed_params import parse_seed_params
from backend_core.AgentMPC.dynamics.loader import DynamicLoader
from backend_core.AgentMPC.graph.workflow import build_ui_tuning_graph, initial_state
from backend_core.AgentMPC.mpc.config import Config

try:
    from backend_core.AgentMPC.agents.llm_base import TokenUsageTracker
    _TOKEN_TRACKING = True
except ImportError:
    _TOKEN_TRACKING = False

matplotlib.use("Agg")


# ============================================================================
# PALETTE -- one source of truth, shared by CSS and matplotlib so charts sit
# inside the panels rather than looking pasted on top of them.
# ============================================================================

C = {
    "bg": "#0a0d12", "panel": "#11161d", "elevated": "#161d27",
    "text": "#eef2f8", "text_2": "#8d9bb0", "muted": "#556074",
    "accent": "#5b7fff", "accent_2": "#8b7dff", "mpc": "#b98bff",
    "success": "#22d3a7", "warn": "#f0b429", "danger": "#f0576b",
}
# Hex-only versions for matplotlib (it will not accept rgba() strings).
MPL = {"bg": "#11161d", "grid": "#1e252f", "axis": "#2b3441", "text": "#8d9bb0",
        "series": ["#b98bff", "#5b7fff", "#22d3a7", "#f0b429", "#f0576b",
                   "#8b7dff", "#4cc9f0", "#f7768e"],
        "ref": "#4a5d78", "input": "#22d3a7"}


# ============================================================================
# ICONS -- inline SVG, never emoji
# ============================================================================

def _svg(d: str, extra: str = "") -> str:
    return (f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
            f'stroke-linecap="round" stroke-linejoin="round" {extra}>{d}</svg>')


ICON_UPLOAD = _svg('<path d="M12 3v12"/><path d="M7 8l5-5 5 5"/>'
                    '<path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>')
ICON_CHAT = _svg('<path d="M4 5h16v11H8l-4 4V5Z"/>')
ICON_SLIDERS = _svg('<line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/>'
                     '<line x1="4" y1="18" x2="20" y2="18"/>'
                     '<circle cx="9" cy="6" r="1.8" fill="currentColor" stroke="none"/>'
                     '<circle cx="16" cy="12" r="1.8" fill="currentColor" stroke="none"/>'
                     '<circle cx="11" cy="18" r="1.8" fill="currentColor" stroke="none"/>')
ICON_CPU = _svg('<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
                 '<path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/>')


# ============================================================================
# THEME / CHROME
# ============================================================================

def inject_theme():
    st.markdown("""
<style>
  :root {
    --bg:#0a0d12; --panel:#11161d; --elevated:#161d27;
    --border:rgba(255,255,255,0.07); --border-soft:rgba(255,255,255,0.045);
    --text:#eef2f8; --text-2:#8d9bb0; --muted:#556074;
    --accent:#5b7fff; --accent-2:#8b7dff; --mpc:#b98bff;
    --success:#22d3a7; --warn:#f0b429; --danger:#f0576b;
    --mono:"IBM Plex Mono","JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
    --sans:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  }
  .stApp { background:var(--bg); }
  html, body, .stApp, [data-testid="stMarkdownContainer"] {
      font-family:var(--sans); color:var(--text); font-size:14px; }
  [data-testid="stHeader"] { display:none; }
  .block-container { padding:0.7rem 1.5rem 2rem 1.5rem !important; max-width:1560px; }
  [data-testid="stIconMaterial"], [class*="material-symbol"], [class*="material-icon"] {
      font-family:'Material Symbols Rounded','Material Icons',sans-serif !important; }

  /* ---- region tag drawn above every panel ---- */
  .sec-tag { display:inline-flex; align-items:center; gap:6px; font-family:var(--mono);
      font-size:9.5px; font-weight:600; letter-spacing:.12em; text-transform:uppercase;
      color:var(--muted); background:rgba(255,255,255,0.025); border:1px solid var(--border-soft);
      padding:2px 8px; border-radius:5px; margin-bottom:7px; }
  .sec-tag .bar { width:3px; height:9px; border-radius:2px; background:var(--mpc); }
  .sec-title { font-size:15px; font-weight:650; color:var(--text); margin-bottom:2px; }
  .sec-sub { font-size:11.5px; color:var(--muted); margin-bottom:10px; line-height:1.5; }

  /* ---- topbar ---- */
  .topbar { display:flex; align-items:center; justify-content:space-between; gap:14px;
      padding:9px 15px; border:1px solid var(--border); border-radius:12px;
      background:var(--panel); margin-bottom:11px; }
  .crumbs { display:flex; align-items:center; gap:8px; font-size:12.5px; color:var(--muted); }
  .crumbs .current { color:var(--text); font-weight:600; }
  .crumbs .sep { opacity:.45; }
  .chip { display:inline-flex; align-items:center; gap:6px; font-size:10.5px; font-weight:700;
      letter-spacing:.03em; padding:3px 9px; border-radius:20px; white-space:nowrap; }
  .chip.mpc { color:var(--mpc); background:rgba(139,125,255,.14); border:1px solid rgba(139,125,255,.34); }
  .chip.ok { color:var(--success); background:rgba(34,211,167,.12); border:1px solid rgba(34,211,167,.3); }
  .chip.err { color:var(--danger); background:rgba(240,87,107,.12); border:1px solid rgba(240,87,107,.3); }
  .chip.model { color:var(--text-2); background:var(--elevated); border:1px solid var(--border);
      font-family:var(--mono); font-weight:600; }
  .dot { width:6px; height:6px; border-radius:50%; }
  .dot.live { background:var(--success); box-shadow:0 0 8px rgba(34,211,167,.5);
      animation:pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }

  /* ---- stage rail ---- */
  .rail { display:flex; align-items:center; gap:0; margin-bottom:13px; flex-wrap:wrap; }
  .rail-step { display:flex; align-items:center; gap:7px; padding:5px 11px; border-radius:8px;
      font-size:11.5px; color:var(--muted); }
  .rail-step .num { width:19px; height:19px; border-radius:6px; display:grid; place-items:center;
      font-family:var(--mono); font-size:10px; font-weight:700; background:var(--elevated);
      border:1px solid var(--border); }
  .rail-step.done { color:var(--text-2); }
  .rail-step.done .num { background:rgba(34,211,167,.12); border-color:rgba(34,211,167,.3); color:var(--success); }
  .rail-step.active { color:var(--text); background:rgba(139,125,255,.08); }
  .rail-step.active .num { background:rgba(139,125,255,.16); border-color:rgba(139,125,255,.4); color:var(--mpc); }
  .rail-line { flex:0 0 16px; height:1px; background:var(--border); }

  /* ---- panels ---- */
  .panel { background:var(--panel); border:1px solid var(--border); border-radius:14px;
      padding:13px 15px; margin-bottom:10px; }

  /* ---- metric cards ---- */
  .mrow { display:grid; gap:9px; margin-bottom:4px; }
  .mcard { padding:10px 12px; border-radius:11px; background:var(--panel);
      border:1px solid var(--border); position:relative; overflow:hidden; }
  .mcard::before { content:""; position:absolute; top:0; left:0; right:0; height:2px; background:var(--mpc); }
  .mcard.good::before { background:var(--success); }
  .mcard.warn::before { background:var(--warn); }
  .mcard.bad::before  { background:var(--danger); }
  .mcard .l { font-size:9.5px; color:var(--muted); font-weight:650; letter-spacing:.06em;
      text-transform:uppercase; margin-bottom:5px; }
  .mcard .v { font-size:18px; font-weight:700; font-family:var(--mono); color:var(--text);
      display:flex; align-items:baseline; gap:5px; }
  .mcard .v .u { font-size:10.5px; color:var(--muted); font-family:var(--sans); font-weight:500; }
  .mcard .s { font-size:10px; color:var(--muted); margin-top:4px; }
  .mcard .s.up { color:var(--success); }
  .mcard .s.down { color:var(--danger); }

  /* ---- key/value grid ---- */
  .kv { display:grid; gap:7px; }
  .kv .cell { text-align:center; padding:7px 4px; border-radius:9px; background:var(--elevated);
      border:1px solid var(--border-soft); }
  .kv .cell .k { font-size:9.5px; color:var(--muted); font-weight:650; margin-bottom:3px; }
  .kv .cell .v { font-size:13px; font-weight:700; font-family:var(--mono); color:var(--text); }

  /* ---- agent flow strip ---- */
  .flow { display:flex; align-items:center; }
  .fnode { flex:1; display:flex; flex-direction:column; align-items:center; gap:4px; position:relative; }
  .fnode .ic { width:27px; height:27px; border-radius:9px; display:grid; place-items:center;
      font-family:var(--mono); font-size:11px; font-weight:700; background:var(--elevated);
      border:1px solid var(--border); color:var(--muted); }
  .fnode .nm { font-size:9.5px; color:var(--muted); font-weight:600; }
  .fnode.on .ic { background:rgba(139,125,255,.16); border-color:rgba(139,125,255,.4); color:var(--mpc); }
  .fnode.on .nm { color:var(--text); }
  .fnode.on .pip { position:absolute; top:-2px; right:26%; width:6px; height:6px; border-radius:50%;
      background:var(--success); box-shadow:0 0 8px rgba(34,211,167,.6); animation:pulse 2s ease-in-out infinite; }
  .fconn { flex:0 0 13px; height:1px; background:var(--border); margin-bottom:15px; }
  .fconn.on { background:var(--mpc); }

  /* ---- activity feed ---- */
  .feed { display:flex; flex-direction:column; gap:7px; max-height:250px; overflow-y:auto; }
  .fitem { font-size:11px; line-height:1.5; display:flex; gap:8px; }
  .fitem .t { color:var(--muted); font-family:var(--mono); flex-shrink:0; font-size:10px; }
  .fitem .a { font-weight:650; color:#c3aeff; }
  .fitem .hl { color:var(--text); font-family:var(--mono); }

  /* ---- chat ---- */
  .msg { display:flex; gap:10px; margin-bottom:13px; }
  .msg.u { flex-direction:row-reverse; }
  .msg .av { width:25px; height:25px; border-radius:8px; flex-shrink:0; display:grid; place-items:center;
      font-size:9.5px; font-weight:700; margin-top:2px; font-family:var(--mono); }
  .msg.a .av { background:linear-gradient(150deg,var(--accent),var(--accent-2)); color:#fff; }
  .msg.u .av { background:var(--elevated); border:1px solid var(--border); color:var(--text-2); }
  .msg .bub { padding:9px 13px; border-radius:12px; font-size:13px; line-height:1.6; max-width:78%; }
  .msg.a .bub { background:var(--panel); border:1px solid var(--border); color:var(--text); }
  .msg.u .bub { background:rgba(91,127,255,.12); border:1px solid rgba(91,127,255,.28); color:#dbe3ff; }

  /* ---- streamlit widget restyle ---- */
  .stButton button { border-radius:9px !important; font-weight:600 !important; font-size:12.5px !important;
      border:1px solid var(--border) !important; background:var(--elevated) !important;
      color:var(--text-2) !important; transition:all .13s ease !important; }
  .stButton button:hover { color:var(--text) !important; border-color:rgba(255,255,255,.15) !important; }
  .stButton button[kind="primary"] { background:linear-gradient(135deg,var(--mpc),#6f5de0) !important;
      color:#fff !important; border:none !important; box-shadow:0 2px 12px rgba(139,125,255,.26) !important; }
  div[data-testid="stSlider"] div[data-baseweb="slider"] > div:nth-child(2) {
      background:linear-gradient(90deg,#2a3f5f,var(--accent)) !important; height:4px !important; }
  div[data-testid="stSlider"] div[role="slider"] { width:16px !important; height:16px !important;
      background:radial-gradient(circle at 35% 30%,#fff,#8fc4ff 55%,var(--accent)) !important;
      box-shadow:0 0 0 3px rgba(91,127,255,.18) !important; border:none !important; }
  [data-testid="stFileUploaderDropzone"] { background:rgba(255,255,255,.015) !important;
      border:1.5px dashed rgba(255,255,255,.14) !important; border-radius:13px !important; }
  [data-testid="stFileUploaderDropzone"]:hover { border-color:rgba(139,125,255,.45) !important; }
  [data-testid="stExpander"] { border:1px solid var(--border) !important; border-radius:11px !important;
      background:rgba(255,255,255,.012) !important; }
  .stTabs [data-baseweb="tab-list"] { gap:3px; border-bottom:1px solid var(--border); }
  .stTabs [data-baseweb="tab"] { font-size:12.5px; font-weight:600; color:var(--muted); }
  .stTabs [aria-selected="true"] { color:var(--mpc) !important; }
  div[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:10px; }
  .stRadio label { font-size:12px !important; }
  .stRadio [role="radiogroup"] { gap:0px; }
</style>
""", unsafe_allow_html=True)


def render_section_tag(tag: str, title: str = "", sub: str = ""):
    """THEME/CHROME -- the small monospace region label above every panel.
    `tag` matches an entry in this module's SECTION MAP docstring, so a
    reader can go from a region on screen to the function that renders it."""
    html = f'<div class="sec-tag"><span class="bar"></span>{tag}</div>'
    if title:
        html += f'<div class="sec-title">{title}</div>'
    if sub:
        html += f'<div class="sec-sub">{sub}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_topbar():
    """THEME/CHROME -- breadcrumb, active model, run status."""
    sysname = (st.session_state.uic_plugin.dynamics_class.__name__
               if st.session_state.get("uic_plugin") else "No system loaded")
    if st.session_state.get("uic_running"):
        status = '<span class="chip ok"><span class="dot live"></span>Tuning live</span>'
    elif st.session_state.get("uic_results"):
        status = '<span class="chip ok">Run complete</span>'
    else:
        status = '<span class="chip model">Idle</span>'
    model_chip = (f'<span class="chip model">{LLM_PROVIDER} &middot; {LLM_MODEL}</span>'
                  if LLM_READY else '<span class="chip err">LLM not configured</span>')
    st.markdown(
        f'<div class="topbar"><div class="crumbs">'
        f'<span>Agent-MPC Studio</span><span class="sep">/</span>'
        f'<span class="current">{sysname}</span>'
        f'<span class="chip mpc">Agentic MPC</span></div>'
        f'<div style="display:flex;align-items:center;gap:8px;">{model_chip}{status}</div></div>',
        unsafe_allow_html=True)


_STAGES = [("Upload", "upload"), ("Consult", "consult"), ("General", "launch_general"),
           ("MPC", "launch_mpc"), ("Run", "run"), ("Export", "export")]


def render_stage_rail():
    """THEME/CHROME -- numbered stage strip showing where the user is."""
    order = [s[1] for s in _STAGES]
    cur = st.session_state.uic_stage
    ci = order.index(cur) if cur in order else 0
    parts = []
    for i, (label, _key) in enumerate(_STAGES):
        cls = "done" if i < ci else ("active" if i == ci else "")
        parts.append(f'<div class="rail-step {cls}"><span class="num">{i+1}</span>{label}</div>')
        if i < len(_STAGES) - 1:
            parts.append('<div class="rail-line"></div>')
    st.markdown(f'<div class="rail">{"".join(parts)}</div>', unsafe_allow_html=True)


def metric_cards(items, cols=4):
    """THEME/CHROME -- a row of metric cards. Each item is a dict with
    label / value / optional unit, sub, sub_tone, tone("",good,warn,bad)."""
    html = f'<div class="mrow" style="grid-template-columns:repeat({cols},1fr);">'
    for it in items:
        unit = f'<span class="u">{it["unit"]}</span>' if it.get("unit") else ""
        sub = f'<div class="s {it.get("sub_tone","")}">{it["sub"]}</div>' if it.get("sub") else ""
        html += (f'<div class="mcard {it.get("tone","")}"><div class="l">{it["label"]}</div>'
                 f'<div class="v">{it["value"]}{unit}</div>{sub}</div>')
    st.markdown(html + "</div>", unsafe_allow_html=True)


# ============================================================================
# LLM SETUP -- standalone (app.py is never imported, so this file cannot
# rely on app.py having configured the client).
# ============================================================================

try:
    from labcd_agents import ensure_env_loaded as _ensure_env
    _ensure_env()
except ImportError:
    pass

BUILTIN_DEFAULT_MODEL = "gpt-4o-mini"
_ENV_MODEL = os.getenv("DEFAULT_LLM_MODEL")
# Whether the active default came from .env or from the line above. Tracked
# because a .env left over from configuring the other app silently wins
# here otherwise, and "why is it not using gpt-4o-mini?" is invisible
# without saying so explicitly in the UI.
MODEL_FROM_ENV = bool(_ENV_MODEL)
DEFAULT_LLM_MODEL = _ENV_MODEL or BUILTIN_DEFAULT_MODEL

LLM_MODEL_PRESETS = [
    "gpt-4o-mini", "gpt-4o", "gpt-5.4-mini",                       # OpenAI
    "openai/gpt-oss-120b", "openai/gpt-oss-20b",                   # Groq
    "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",     # Anthropic
    "llama-3.3-70b",                                                # Cerebras
]


@st.cache_resource
def _init_llm(model_name: str):
    try:
        from labcd_agents import LLMFactory, ensure_env_loaded, get_api_key
    except ImportError as e:
        return False, model_name, None, (
            f"labcd_agents not found ({e}) -- install with: "
            f"pip install -e 'packages/labcd_agents[all]'")
    ensure_env_loaded()
    provider = LLMFactory.resolve_provider(model_name)
    if provider is None:
        return False, model_name, None, (
            f"Unrecognized model '{model_name}'. See "
            f"packages/labcd_agents/src/labcd_agents/providers.py for the recognized names.")
    if not get_api_key(provider):
        env_var = {"openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY", "cerebras": "CEREBRAS_API_KEY",
                   "nvidia": "NVIDIA_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(
                       provider, f"{provider.upper()}_API_KEY")
        return False, model_name, provider, f"{env_var} not set. Add it to .env next to this file, then restart."
    try:
        inst = LLMFactory.create(model_name, temperature=0.3, seed=42, max_retries=2)
    except Exception as e:  # noqa: BLE001
        return False, model_name, provider, f"Could not initialize the {provider} client for '{model_name}': {e}"
    configure_llm(lambda: inst)
    return True, model_name, provider, None


# ============================================================================
# SESSION STATE
# ============================================================================

_DEFAULTS = {
    "uic_stage": "upload",
    "uic_dyn": None, "uic_plugin": None, "uic_source": None, "uic_setup": None,
    "uic_chat": [], "uic_suggestion": None,
    "uic_general": None, "uic_mpc_cfg": None,
    "uic_results": [], "uic_run_error": None, "uic_running": False,
    "uic_iterator": None, "uic_stop_requested": False,
    "uic_activity": [], "uic_active_node": None, "uic_candidate": None,
    "uic_reasoning": [], "uic_perturbed": {},
    "uic_pdf": None, "uic_script": None, "uic_tracker": None,
    "selected_llm_model": None,
}


def ensure_state():
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = ([] if isinstance(v, list) else ({} if isinstance(v, dict) else v))
    if st.session_state.selected_llm_model is None:
        st.session_state.selected_llm_model = DEFAULT_LLM_MODEL


def reset_all():
    for k, v in _DEFAULTS.items():
        st.session_state[k] = ([] if isinstance(v, list) else ({} if isinstance(v, dict) else v))
    st.session_state.selected_llm_model = DEFAULT_LLM_MODEL


ensure_state()
inject_theme()
LLM_READY, LLM_MODEL, LLM_PROVIDER, LLM_ERROR = _init_llm(st.session_state.selected_llm_model)


# ============================================================================
# PLOTTING -- dark-background matplotlib matching the panel palette
# ============================================================================

def _style_axes(ax, title=None):
    ax.set_facecolor(MPL["bg"])
    for s in ax.spines.values():
        s.set_color(MPL["axis"])
        s.set_linewidth(0.8)
    ax.tick_params(colors=MPL["text"], labelsize=8, length=3, width=0.8)
    ax.grid(True, color=MPL["grid"], linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=C["text"], fontsize=10.5, fontweight="600", pad=8)


def _new_fig(w=9.0, h=3.4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(MPL["bg"])
    return fig, ax


def _draw_bounds(ax, bounds, idx):
    """Constraint lines, drawn ONLY for finite bounds -- an unbounded
    (+-inf) side draws nothing, so unconstrained charts stay clean."""
    if bounds is None:
        return
    lo, hi = bounds
    for arr in (lo, hi):
        if arr is not None and idx < len(arr) and np.isfinite(arr[idx]):
            ax.axhline(arr[idx], color=C["danger"], ls=":", lw=1.2, alpha=.75, zorder=2)


def _legend(ax):
    lg = ax.legend(loc="best", fontsize=8, framealpha=.85)
    if lg:
        lg.get_frame().set_facecolor(MPL["bg"])
        lg.get_frame().set_edgecolor(MPL["axis"])
        for t in lg.get_texts():
            t.set_color(MPL["text"])


def plot_single_state(sim, idx, name, x_bounds=None):
    """RUN·PLOTS -- one state against its reference."""
    fig, ax = _new_fig()
    t, X, R = sim["times"], sim["states"], sim.get("refs")
    ax.plot(t, X[:, idx], color=MPL["series"][idx % len(MPL["series"])], lw=2.0, label=name, zorder=3)
    if R is not None and len(R) and idx < R.shape[1]:
        ax.plot(t, R[:len(X), idx], "--", color=MPL["ref"], lw=1.4, label="reference", zorder=2)
    _draw_bounds(ax, x_bounds, idx)
    _style_axes(ax, f"{name} \u2014 response vs reference")
    ax.set_xlabel("Time [s]", color=MPL["text"], fontsize=9)
    _legend(ax)
    fig.tight_layout()
    return fig


def plot_single_input(sim, j, name, u_bounds=None):
    """RUN·PLOTS -- one control input, with saturation lines when finite."""
    fig, ax = _new_fig()
    t, U = sim["times"], sim["inputs"]
    ax.plot(t[:len(U)], U[:, j], color=MPL["input"], lw=2.0, label=name, zorder=3)
    ax.axhline(0, color=MPL["axis"], lw=0.8, zorder=1)
    _draw_bounds(ax, u_bounds, j)
    _style_axes(ax, f"{name} \u2014 control input")
    ax.set_xlabel("Time [s]", color=MPL["text"], fontsize=9)
    _legend(ax)
    fig.tight_layout()
    return fig


def plot_all_states(sim, names, x_bounds=None):
    """RUN·PLOTS -- every state stacked; the see-everything-at-once view."""
    n = sim["n_states"]
    fig, axs = plt.subplots(n, 1, figsize=(9.0, 1.35 * n + 0.9), sharex=True)
    fig.patch.set_facecolor(MPL["bg"])
    if n == 1:
        axs = [axs]
    t, X, R = sim["times"], sim["states"], sim.get("refs")
    for i in range(n):
        nm = names[i] if i < len(names) else f"x{i}"
        axs[i].plot(t, X[:, i], color=MPL["series"][i % len(MPL["series"])], lw=1.7, zorder=3)
        if R is not None and len(R) and i < R.shape[1]:
            axs[i].plot(t, R[:len(X), i], "--", color=MPL["ref"], lw=1.2, zorder=2)
        _draw_bounds(axs[i], x_bounds, i)
        _style_axes(axs[i])
        axs[i].set_ylabel(nm, color=MPL["text"], fontsize=8.5)
    axs[-1].set_xlabel("Time [s]", color=MPL["text"], fontsize=9)
    fig.suptitle("All states", color=C["text"], fontsize=10.5, fontweight="600")
    fig.tight_layout()
    return fig


def plot_all_inputs(sim, names, u_bounds=None):
    """RUN·PLOTS -- every control input stacked."""
    m = sim["n_inputs"]
    fig, axs = plt.subplots(m, 1, figsize=(9.0, 1.5 * m + 0.9), sharex=True)
    fig.patch.set_facecolor(MPL["bg"])
    if m == 1:
        axs = [axs]
    t, U = sim["times"], sim["inputs"]
    for j in range(m):
        nm = names[j] if j < len(names) else f"u{j}"
        axs[j].plot(t[:len(U)], U[:, j], color=MPL["input"], lw=1.8, zorder=3)
        axs[j].axhline(0, color=MPL["axis"], lw=0.8, zorder=1)
        _draw_bounds(axs[j], u_bounds, j)
        _style_axes(axs[j])
        axs[j].set_ylabel(nm, color=MPL["text"], fontsize=8.5)
    axs[-1].set_xlabel("Time [s]", color=MPL["text"], fontsize=9)
    fig.suptitle("All control inputs", color=C["text"], fontsize=10.5, fontweight="600")
    fig.tight_layout()
    return fig


def plot_convergence(rows):
    """RUN·PLOTS -- tracking error per attempt, with running best."""
    ok = [r for r in rows if r.get("ok") and r.get("mse") is not None]
    if not ok:
        return None
    it = [r["iteration"] for r in ok]
    mse = [r["mse"] for r in ok]
    best, run = [], float("inf")
    for v in mse:
        run = min(run, v)
        best.append(run)
    fig, ax = _new_fig(9.0, 3.6)
    ax.plot(it, mse, "o-", color=MPL["text"], lw=1.0, ms=4, alpha=.55, label="each attempt", zorder=2)
    ax.plot(it, best, "-", color=MPL["series"][2], lw=2.4, label="best so far", zorder=3)
    if all(v > 0 for v in mse):
        ax.set_yscale("log")
    _style_axes(ax, "Convergence \u2014 tracking error per attempt")
    ax.set_xlabel("Iteration", color=MPL["text"], fontsize=9)
    ax.set_ylabel("MSE", color=MPL["text"], fontsize=9)
    _legend(ax)
    fig.tight_layout()
    return fig


def plot_metric_history(rows, key, label):
    """RUN·PLOTS -- any single metric's trajectory across iterations."""
    ok = [r for r in rows
          if r.get("ok") and r.get(key) is not None and np.isfinite(r.get(key, np.inf))]
    if not ok:
        return None
    fig, ax = _new_fig(9.0, 3.2)
    ax.plot([r["iteration"] for r in ok], [r[key] for r in ok], "o-",
            color=MPL["series"][3], lw=1.8, ms=4.5, zorder=3)
    _style_axes(ax, f"{label} per attempt")
    ax.set_xlabel("Iteration", color=MPL["text"], fontsize=9)
    ax.set_ylabel(label, color=MPL["text"], fontsize=9)
    fig.tight_layout()
    return fig


def show_fig(fig):
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ============================================================================
# SETUP · UPLOAD
# ============================================================================

def section_upload():
    """SETUP·UPLOAD -- dynamics file intake. Runs analyze_and_setup(), which
    validates the file, auto-repairs it via the LLM if it doesn't match the
    plugin contract, and then runs the deterministic setup analyses
    (derivative pairs, initial Q/R, dt, feedforward trim)."""
    render_section_tag(
        "SETUP · UPLOAD", "Load a system",
        "A Python file defining create_config() and a BaseDynamics subclass. "
        "It is validated on upload, auto-repaired if needed, then analyzed for "
        "timescales and sensible starting weights.")

    up = st.file_uploader("System file", type=["py"], label_visibility="collapsed", key="uic_up")
    if up is None:
        with st.expander("What the file needs to contain"):
            st.markdown(
                "- `create_config()` returning a `SystemConfig`\n"
                "- a class inheriting from `BaseDynamics` implementing `dynamics(self, x, u)`\n\n"
                "`np`, `BaseDynamics` and `SystemConfig` are injected at load time -- you do not "
                "need to import them. See `AgentMPC/dynamics/plugins/example_pendulum.py`.")
        return

    if st.button("Analyze and continue", type="primary", key="uic_up_go"):
        src = up.getvalue().decode("utf-8")
        with st.spinner("Validating and analyzing..."):
            setup = analyze_and_setup(src)
        if not setup.fix.valid:
            st.error(f"Could not load this file: "
                     f"{setup.fix.still_broken_error or setup.fix.original_error}")
            return
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(setup.fix.final_code)
                tmp = f.name
            plugin = DynamicLoader.load_from_path(tmp)
            dyn = plugin.create_dynamics()
        except Exception as e:  # noqa: BLE001
            st.error(f"Error instantiating the system: {e}")
            return
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

        st.session_state.uic_dyn = dyn
        st.session_state.uic_plugin = plugin
        st.session_state.uic_source = setup.fix.final_code
        st.session_state.uic_setup = setup
        st.session_state.uic_stage = "consult"
        st.rerun()


# ============================================================================
# SETUP · CONSULT
# ============================================================================

def section_consult():
    """SETUP·CONSULT -- the human-in-the-loop step. Free-form chat about the
    system (is MPC even right here? what should I watch out for?) plus a
    structured suggestion pass that fills the next two launch screens.
    Both come from agents/config_advisor_agent.py."""
    summary = st.session_state.uic_plugin.summary()
    setup = st.session_state.uic_setup

    render_section_tag(
        "SETUP · CONSULT", "Talk it through first",
        "Optional. Ask about control strategy, constraints, or anything else -- the advisor "
        "sees your actual system, not a generic description. It can also propose starting "
        "values for every setting on the next two screens.")

    left, right = st.columns([3, 2], gap="medium")

    with left:
        with st.container(border=True):
            if not st.session_state.uic_chat:
                st.markdown(
                    f'<div class="msg a"><div class="av">AI</div><div class="bub">'
                    f"I've loaded <b>{summary.get('dynamics_class')}</b> "
                    f"({summary.get('n_states')} states, {summary.get('n_inputs')} inputs). "
                    f"Ask me anything about it \u2014 whether MPC is the right fit, what "
                    f"constraints make sense, what to watch out for while tuning."
                    f'</div></div>', unsafe_allow_html=True)
            for turn in st.session_state.uic_chat:
                role = "u" if turn["role"] == "user" else "a"
                av = "YOU" if role == "u" else "AI"
                body = turn["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                st.markdown(f'<div class="msg {role}"><div class="av">{av}</div>'
                            f'<div class="bub">{body}</div></div>', unsafe_allow_html=True)

        msg = st.chat_input("Ask about control strategy, constraints, tuning...")
        if msg:
            if not LLM_READY:
                st.error(f"The LLM is not configured: {LLM_ERROR}")
            else:
                st.session_state.uic_chat.append({"role": "user", "content": msg})
                try:
                    with st.spinner("Thinking..."):
                        reply = advisor_chat(
                            msg, st.session_state.uic_chat[:-1], summary,
                            setup_notes=setup.setup_notes,
                            derivative_pairs=setup.derivative_pairs,
                            tracker=st.session_state.uic_tracker)
                except Exception as e:  # noqa: BLE001
                    reply = f"(The advisor hit an error: {e})"
                st.session_state.uic_chat.append({"role": "assistant", "content": reply})
                st.rerun()

    with right:
        render_section_tag("SETUP · CONSULT / SUGGEST")
        sug = st.session_state.uic_suggestion
        if sug is None:
            st.caption("Ask the advisor to propose starting values for constraints and the "
                       "general tuning settings. You can still edit every field afterwards, "
                       "or skip this entirely and set them yourself.")
            if st.button("Suggest a configuration", type="primary",
                          use_container_width=True, disabled=not LLM_READY, key="uic_sug_go"):
                try:
                    with st.spinner("Working out a starting configuration..."):
                        raw = suggest_config(
                            summary, setup_notes=setup.setup_notes,
                            derivative_pairs=setup.derivative_pairs,
                            conversation_history=st.session_state.uic_chat,
                            tracker=st.session_state.uic_tracker)
                        raw.general_settings = clamp_general_settings(raw.general_settings)
                        st.session_state.uic_suggestion = raw
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not produce a suggestion: {e}")
        else:
            st.markdown(f'<div class="panel" style="font-size:12.5px;line-height:1.6;">'
                        f'{sug.summary}</div>', unsafe_allow_html=True)
            g = sug.general_settings
            metric_cards([
                {"label": "Sim time", "value": f"{g.simulation_time:g}", "unit": "s"},
                {"label": "Settling tol", "value": f"{g.settling_tolerance_pct}", "unit": "%"},
                {"label": "Iterations", "value": f"{g.max_iterations}"},
            ], cols=3)
            with st.expander("Full suggestion (JSON)", expanded=False):
                st.json(sug.model_dump())
            if sug.warnings:
                for w in sug.warnings:
                    st.warning(w, icon=None)
            if st.button("Discard and start over", use_container_width=True, key="uic_sug_clear"):
                st.session_state.uic_suggestion = None
                st.rerun()

    st.divider()
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("Continue \u2192", type="primary", key="uic_consult_next"):
            st.session_state.uic_stage = "launch_general"
            st.rerun()
    with c2:
        if st.button("\u2190 Load a different system", key="uic_consult_back"):
            reset_all()
            st.rerun()


# ============================================================================
# LAUNCH · GENERAL
# ============================================================================

def section_launch_general():
    """LAUNCH·GENERAL -- settings that apply to ANY controller, not just MPC:
    how long each candidate is simulated, what counts as settled, how many
    attempts to make, how boldly to explore, and the hard constraint bounds
    the optimizer must respect."""
    summary = st.session_state.uic_plugin.summary()
    dyn = st.session_state.uic_dyn
    sug = st.session_state.uic_suggestion
    g = sug.general_settings if sug else None

    render_section_tag(
        "LAUNCH · GENERAL", "Study settings and constraints",
        "Controller-agnostic. These frame the search itself -- how each candidate is "
        "evaluated and what limits it must respect." +
        (" Pre-filled from the advisor's suggestion; edit anything." if g else ""))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Evaluation**")
        sim_time = st.slider("Simulation time (s)", 2.0, 20.0,
                              float(g.simulation_time) if g else 8.0, step=0.5, key="uic_simtime",
                              help="How long each candidate parameter set runs. Longer shows more "
                                   "post-settling behavior at the cost of slower iterations.")
        settle_tol = st.slider("Settling tolerance (%)", 1, 20,
                                int(g.settling_tolerance_pct) if g else 5, key="uic_settletol",
                                help="How close to target counts as settled, as a percent of the "
                                     "initial error. Lower is stricter.")
    with c2:
        st.markdown("**Search budget**")
        max_iter = st.slider("Max iterations", 3, 30,
                              int(g.max_iterations) if g else 10, key="uic_maxiter")
        min_explore = st.slider("Minimum explore iterations", 0, 15,
                                 int(g.min_explore_iterations) if g else 4, key="uic_minexp",
                                 help="Fine-tuning is blocked until this many attempts have run, "
                                      "so the search covers ground before narrowing.")
        explore_int = st.slider("Exploration intensity (%)", 1, 100,
                                 int(g.exploration_intensity) if g else 50, key="uic_expint",
                                 help="How bold parameter changes are while exploring. 50 is normal.")

    if g and g.rationale:
        st.caption(f"Advisor's reasoning: {g.rationale}")

    st.markdown("")
    render_section_tag("LAUNCH · GENERAL / CONSTRAINTS")
    st.caption("Hard limits enforced inside the optimization itself -- the controller cannot "
               "propose a solution that violates them. Pre-filled from the dynamics file's own "
               "declared bounds (and the advisor's suggestion, where it had one). "
               "Use -inf / inf for an unbounded side.")

    in_names = summary.get("input_names", []) or [f"u{i}" for i in range(dyn.n_inputs)]
    st_names = summary.get("state_names", []) or [f"x{i}" for i in range(dyn.n_states)]
    pub, psb = dyn.get_input_bounds(), dyn.get_state_bounds()

    u_lo = list(pub[0]) if pub is not None else [-np.inf] * dyn.n_inputs
    u_hi = list(pub[1]) if pub is not None else [np.inf] * dyn.n_inputs
    x_lo = list(psb[0]) if psb is not None else [-np.inf] * dyn.n_states
    x_hi = list(psb[1]) if psb is not None else [np.inf] * dyn.n_states

    # The advisor's numbers take precedence over the plugin defaults, but only
    # where it actually gave one -- a null side means "leave it unbounded",
    # not "overwrite whatever the plugin declared with nothing".
    if sug:
        for b in sug.input_bounds:
            if b.name in in_names:
                k = in_names.index(b.name)
                if b.lower is not None:
                    u_lo[k] = b.lower
                if b.upper is not None:
                    u_hi[k] = b.upper
        for b in sug.state_bounds:
            if b.name in st_names:
                k = st_names.index(b.name)
                if b.lower is not None:
                    x_lo[k] = b.lower
                if b.upper is not None:
                    x_hi[k] = b.upper

    cc1, cc2 = st.columns(2, gap="large")
    with cc1:
        st.markdown("**Input bounds**")
        u_df = st.data_editor(
            pd.DataFrame({"Input": in_names, "Min": u_lo, "Max": u_hi}),
            disabled=["Input"], hide_index=True, use_container_width=True, key="uic_ub")
    with cc2:
        st.markdown("**State bounds**")
        x_df = st.data_editor(
            pd.DataFrame({"State": st_names, "Min": x_lo, "Max": x_hi}),
            disabled=["State"], hide_index=True, use_container_width=True, key="uic_xb")

    st.divider()
    b1, b2 = st.columns([1, 5])
    with b1:
        if st.button("Continue \u2192", type="primary", key="uic_gen_next"):
            st.session_state.uic_general = {
                "simulation_time": sim_time, "settling_tolerance_pct": settle_tol,
                "max_iterations": max_iter, "min_explore_iterations": min_explore,
                "exploration_intensity": explore_int,
                "u_bounds": (u_df["Min"].to_numpy(dtype=float), u_df["Max"].to_numpy(dtype=float)),
                "x_bounds": (x_df["Min"].to_numpy(dtype=float), x_df["Max"].to_numpy(dtype=float)),
            }
            st.session_state.uic_stage = "launch_mpc"
            st.rerun()
    with b2:
        if st.button("\u2190 Back to consult", key="uic_gen_back"):
            st.session_state.uic_stage = "consult"
            st.rerun()


# ============================================================================
# LAUNCH · MPC
# ============================================================================

_SCENARIO_CARDS = [
    (1, "Nominal", "Clean run from the file's own starting point. No noise."),
    (2, "Noise", "Adds measurement noise to every state."),
    (3, "Robust", "Harder starting point, noise, and physical-parameter mismatch."),
]
_TRAJ_CARDS = [
    ("reg", "Regulation", "Hold steady at a fixed target."),
    ("sin", "Sinusoidal", "Track a smooth back-and-forth wave."),
    ("pulse", "Pulse", "Step to a target, then step back."),
]


def _card_choice(options, key, default):
    """LAUNCH·MPC -- clickable cards standing in for a dropdown. The active
    card is marked via the button's own type=primary rather than a wrapping
    div, because Streamlit does not nest widgets created after a raw
    st.markdown() as children of that markdown's HTML."""
    sk = f"_cardsel_{key}"
    if sk not in st.session_state:
        st.session_state[sk] = default
    cols = st.columns(len(options))
    for i, (col, (val, title, desc)) in enumerate(zip(cols, options)):
        with col:
            active = st.session_state[sk] == val
            if st.button(f"{title}\n{desc}", key=f"{key}_c{i}", use_container_width=True,
                          type="primary" if active else "secondary"):
                if st.session_state[sk] != val:
                    st.session_state[sk] = val
                    st.rerun()
    return st.session_state[sk]


def section_launch_mpc():
    """LAUNCH·MPC -- everything specific to the MPC controller and the
    scenario it is graded against: scenario level, reference trajectory,
    seed Np/Nc/Q/R, sample time, optimization focus, free-text guidance."""
    summary = st.session_state.uic_plugin.summary()
    setup = st.session_state.uic_setup
    n_states = summary.get("n_states", 4)
    n_inputs = summary.get("n_inputs", 1)

    render_section_tag(
        "LAUNCH · MPC", "Controller and scenario",
        "MPC-specific. What the controller starts from, and how hard a scenario it is graded against.")

    st.markdown("**Scenario level**")
    scenario = _card_choice(_SCENARIO_CARDS, "scenario", 1)
    st.markdown("")
    st.markdown("**Reference trajectory**")
    traj = _card_choice(_TRAJ_CARDS, "traj", "reg")

    traj_amp, traj_freq = 0.5, 0.5
    if traj in ("sin", "pulse"):
        t1, t2 = st.columns(2)
        with t1:
            traj_amp = st.slider("Amplitude", 0.05, 3.0, 0.5, step=0.05, key="uic_tamp")
        with t2:
            traj_freq = st.slider("Frequency (Hz)", 0.05, 3.0, 0.5, step=0.05, key="uic_tfreq")

    st.markdown("")
    render_section_tag("LAUNCH · MPC / SEED")
    st.caption("Where the Actor starts searching from. Q and R are pre-filled from the Setup "
               "Agent's own analysis of this system (Bryson's rule); dt from its timescale estimate.")

    sQ = setup.suggested_Q or [1.0] * n_states
    sR = setup.suggested_R or [0.1] * n_inputs
    s1, s2, s3 = st.columns([1, 1, 2])
    with s1:
        seed_np = st.number_input("Np", 1, 50, 12, key="uic_np")
    with s2:
        seed_nc = st.number_input("Nc", 1, 50, 5, key="uic_nc")
    with s3:
        dt_val = st.number_input("dt (s)", 0.0005, 1.0,
                                  float(setup.suggested_dt or 0.02), step=0.001,
                                  format="%.4f", key="uic_dt")
    q_text = st.text_input(f"Q \u2014 {n_states} values", ", ".join(f"{v:.4g}" for v in sQ), key="uic_q")
    r_text = st.text_input(f"R \u2014 {n_inputs} values", ", ".join(f"{v:.4g}" for v in sR), key="uic_r")
    seed_params, seed_err = parse_seed_params(seed_np, seed_nc, q_text, r_text, n_states, n_inputs)
    if seed_err:
        st.error(seed_err)

    st.markdown("")
    render_section_tag("LAUNCH · MPC / OBJECTIVE")
    o1, o2 = st.columns([1, 2])
    with o1:
        focus = st.selectbox("Optimization focus", list(OPTIMIZATION_FOCUS_LABELS.keys()),
                              format_func=lambda k: OPTIMIZATION_FOCUS_LABELS[k], key="uic_focus")
    with o2:
        guidance = st.text_area("Guidance for the agents (optional)", "", height=80,
                                 placeholder="e.g. keep the control effort low even if settling is slower",
                                 key="uic_guide")

    st.divider()
    b1, b2, b3 = st.columns([1, 1, 4])
    with b1:
        if st.button("Start tuning", type="primary", disabled=bool(seed_err) or not LLM_READY,
                      key="uic_mpc_go"):
            st.session_state.uic_mpc_cfg = {
                "scenario": scenario, "traj": traj, "traj_amp": traj_amp, "traj_freq": traj_freq,
                "seed_params": seed_params, "dt": dt_val, "focus": focus, "guidance": guidance,
            }
            st.session_state.uic_stage = "run"
            st.session_state.uic_running = True
            st.rerun()
    with b2:
        if st.button("\u2190 Back", key="uic_mpc_back"):
            st.session_state.uic_stage = "launch_general"
            st.rerun()
    with b3:
        if not LLM_READY:
            st.caption(f"Cannot start: {LLM_ERROR}")


# ============================================================================
# RUN -- row builder shared with the main app's schema
# ============================================================================

_NODE_LABEL = {"actor": "Actor proposed new parameters",
               "critic": "Critic reviewed the trend",
               "terminator": "Terminator checked the stopping criteria",
               "juror": "Juror reviewed the final candidate",
               "evaluator": "Simulated the candidate"}
_FLOW = [("A", "Actor", "actor"), ("E", "Eval", "evaluator"), ("C", "Critic", "critic"),
         ("T", "Term", "terminator"), ("J", "Juror", "juror")]


def build_row(update, iteration, n_states, n_inputs, scenario):
    """RUN -- converts one evaluator update into the row shape the rest of
    the pipeline (report_agent, report, export_script) already expects.
    Kept identical to the main app's version on purpose so those modules
    work here unmodified."""
    m = update.get("metrics") or {}
    if update.get("eval_error") or not m:
        return {"iteration": iteration, "ok": False, "unstable": False,
                "error": update.get("eval_error") or "Evaluator returned no metrics.",
                "traceback": update.get("eval_traceback"), "scenario": scenario,
                "np": 0, "nc": 0, "Q_formatted": "--", "R_formatted": "--", "P_formatted": "--",
                "mse": None, "overshoot": None, "settling": None, "effort": None, "cost": None,
                "oscillation_count": None, "iae": None, "ise": None,
                "per_state_mse": {}, "per_state_overshoot": {},
                "success": False, "is_stable": False,
                "strategy": update.get("strategy", "?"), "simulation_data": None}

    p = update.get("current_params") or {}
    q = list(p.get("Q") or [])
    r = list(p.get("R") or [])
    pp = list(p.get("P") or q)
    while len(q) < n_states: q.append(0.0)
    while len(r) < n_inputs: r.append(0.0)
    while len(pp) < n_states: pp.append(0.0)
    fw = lambda w: "[" + ", ".join(f"{fmt_num(x)}" for x in w) + "]"

    row = {"iteration": iteration, "ok": True, "unstable": bool(m.get("Unstable", False)),
           "error": None, "traceback": None, "scenario": scenario,
           "np": p.get("Np", 0), "nc": p.get("Nc", 0),
           "Q_formatted": fw(q), "R_formatted": fw(r), "P_formatted": fw(pp),
           "mse": m.get("MSE", 0.0), "overshoot": m.get("Max_Overshoot", 0.0),
           "settling": m.get("Settling_Time", float("inf")),
           "effort": m.get("Control_Effort_RMS", 0.0), "cost": m.get("Cost"),
           "oscillation_count": m.get("Oscillation_Count", 0),
           "iae": m.get("Integral_Abs_Error", 0.0), "ise": m.get("Integral_Sq_Error", 0.0),
           "is_regulation": m.get("Is_Regulation", True),
           "overshoot_meaningful": m.get("Overshoot_Meaningful", True),
           "per_state_mse": m.get("Per_State_MSE", {}),
           "per_state_overshoot": m.get("Per_State_Overshoot", {}),
           "improvement": m.get("Improvement", {}),
           "solver_diagnostics": m.get("Solver_Diagnostics", {}),
           "success": update.get("success", False),
           "is_stable": bool(m.get("Is_Stable", False)),
           "strategy": update.get("exploration_strategy", "explore"),
           "dt_mpc": m.get("Dt_Mpc"),
           "simulation_data": update.get("simulation_data")}
    for i, v in enumerate(q): row[f"q{i+1}"] = v
    for i, v in enumerate(r): row[f"r{i+1}"] = v
    return row


def best_row(rows):
    """RUN -- ranks by the composite cost the engine itself uses, among
    attempts that actually ran and did not diverge."""
    cand = [r for r in rows if r.get("ok") and not r.get("unstable") and r.get("cost") is not None]
    return min(cand, key=lambda r: r["cost"]) if cand else None


# ============================================================================
# RUN · panels
# ============================================================================

def panel_usage(tracker, compact: bool = False):
    """RUN·USAGE / RESULT·EXPORT·USAGE -- cumulative LLM spend for this
    session. `compact` renders a single inline line for the live run header;
    otherwise a full card row. Prices come from labcd_agents' own table and
    are applied per model (see TokenUsageTracker._compute_cost)."""
    if tracker is None:
        return
    u = tracker.snapshot()
    if not u["call_count"]:
        return
    cost = u.get("cost_usd")
    cost_str = f"${cost:,.4f}" if isinstance(cost, (int, float)) else "n/a"

    if compact:
        note = ""
        if u.get("unpriced_models"):
            note = (f' &middot; <span style="color:#f0b429;">'
                    f'{len(u["unpriced_models"])} model(s) unpriced</span>')
        st.markdown(
            f'<div style="font-size:11px;color:#556074;font-family:var(--mono);'
            f'text-align:right;margin:-4px 0 6px 0;">'
            f'{u["total_tokens"]:,} tokens &middot; {u["call_count"]} calls &middot; '
            f'<span style="color:#22d3a7;">{cost_str}</span>{note}</div>',
            unsafe_allow_html=True)
        return

    render_section_tag("RESULT · EXPORT / USAGE")
    metric_cards([
        {"label": "Estimated spend", "value": cost_str, "tone": "good",
         "sub": "this session, all agents"},
        {"label": "Total tokens", "value": f"{u['total_tokens']:,}",
         "sub": f"{u['prompt_tokens']:,} in / {u['completion_tokens']:,} out"},
        {"label": "LLM calls", "value": f"{u['call_count']}"},
        {"label": "Models used", "value": f"{len(u['per_model'])}"},
    ])
    if u.get("per_model"):
        rows = []
        try:
            from labcd_agents.pricing import CostCalculator
            calc = CostCalculator()
        except ImportError:
            calc = None
        for model, mu in u["per_model"].items():
            c = (calc.compute_cost(model, mu.get("prompt", 0), mu.get("completion", 0))
                 if calc and calc.resolve_price(model) else None)
            rows.append({"Model": model, "Prompt": f"{mu.get('prompt',0):,}",
                          "Completion": f"{mu.get('completion',0):,}",
                          "Total": f"{mu.get('total',0):,}",
                          "Cost": f"${c:,.4f}" if c is not None else "unpriced"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if u.get("unpriced_models"):
        st.caption(f"No published price in the built-in table for: "
                   f"{', '.join(u['unpriced_models'])} \u2014 their tokens are counted but "
                   f"excluded from the total above, so the real spend is higher than shown.")
    if u.get("unparsed_calls"):
        st.caption(f"{u['unparsed_calls']} call(s) returned usage data this app could not parse; "
                   f"their tokens and cost are not included.")


def panel_metrics(rows):
    """RUN·METRICS -- the live metric card strip. Reads the best attempt so
    far, not the latest, so the numbers only ever improve as the run goes."""
    render_section_tag("RUN · METRICS")
    b = best_row(rows)
    n_ok = sum(1 for r in rows if r.get("ok"))
    if b is None:
        metric_cards([
            {"label": "Tracking error", "value": "--"},
            {"label": "Settling time", "value": "--"},
            {"label": "Control effort", "value": "--"},
            {"label": "Attempts", "value": f"{len(rows)}", "sub": f"{n_ok} valid"},
        ])
        return

    imp = b.get("improvement", {}) or {}
    mse_imp = imp.get("MSE")
    settle = (f"{fmt_num(b['settling'])}" if b["settling"] not in (None, float("inf")) else "--")
    metric_cards([
        {"label": "Tracking error", "value": fmt_num(b["mse"]), "tone": "good",
         "sub": (f"{mse_imp:+.1f}% vs first" if isinstance(mse_imp, (int, float)) else
                 f"best of {n_ok}"),
         "sub_tone": ("up" if isinstance(mse_imp, (int, float)) and mse_imp < 0 else "")},
        {"label": "Settling time", "value": settle, "unit": "s" if settle != "--" else "",
         "tone": "good" if b.get("is_stable") else "warn",
         "sub": "stable" if b.get("is_stable") else "not settled"},
        {"label": "Control effort", "value": fmt_num(b["effort"]), "tone": "",
         "sub": f"osc {b.get('oscillation_count', 0)}"},
        {"label": "Attempts", "value": f"{len(rows)}", "tone": "",
         "sub": f"{n_ok} valid \u00b7 best #{b['iteration']}"},
    ])


def _plot_options(summary):
    """RUN·PLOTS -- the shared view list, so the live and interactive
    variants below cannot drift apart."""
    st_names = summary.get("state_names", []) or []
    in_names = summary.get("input_names", []) or []
    return (["All states", "All inputs"]
            + [f"State: {n}" for n in st_names]
            + [f"Input: {n}" for n in in_names]
            + ["Convergence (MSE)", "Overshoot history", "Settling history", "Effort history"])


def _render_plot_choice(choice, rows, summary, u_bounds, x_bounds):
    """RUN·PLOTS -- draws one named view. Display-only (no widgets), so it
    is safe to call repeatedly inside the live redraw loop."""
    b = best_row(rows)
    sim = b.get("simulation_data") if b else None
    st_names = summary.get("state_names", []) or []
    in_names = summary.get("input_names", []) or []

    if choice == "Convergence (MSE)":
        show_fig(plot_convergence(rows))
    elif choice == "Overshoot history":
        show_fig(plot_metric_history(rows, "overshoot", "Overshoot"))
    elif choice == "Settling history":
        show_fig(plot_metric_history(rows, "settling", "Settling time (s)"))
    elif choice == "Effort history":
        show_fig(plot_metric_history(rows, "effort", "Control effort"))
    elif sim is None:
        st.caption("No completed simulation yet \u2014 charts appear after the first attempt.")
    elif choice == "All states":
        show_fig(plot_all_states(sim, st_names, x_bounds))
    elif choice == "All inputs":
        show_fig(plot_all_inputs(sim, in_names, u_bounds))
    elif choice.startswith("State: "):
        nm = choice[7:]
        if nm in st_names:
            show_fig(plot_single_state(sim, st_names.index(nm), nm, x_bounds))
    elif choice.startswith("Input: "):
        nm = choice[7:]
        if nm in in_names:
            show_fig(plot_single_input(sim, in_names.index(nm), nm, u_bounds))


def panel_plots(rows, summary, u_bounds=None, x_bounds=None):
    """RUN·PLOTS -- selector on the left keeps the footprint small while
    still reaching every state, every input and every metric history; the
    canvas on the right shows one at a time. Safe to use during a live run
    because section_run advances only one graph node per script run, so
    this widget is created exactly once per run (see section_run's
    docstring for why that matters)."""
    render_section_tag("RUN · PLOTS")
    b = best_row(rows)
    opts = _plot_options(summary)
    sel_col, canvas = st.columns([1, 4], gap="medium")
    with sel_col:
        choice = st.radio("View", opts, label_visibility="collapsed", key="uic_plotsel")
        st.caption("Showing the best attempt" + (f" (#{b['iteration']})" if b else ""))
    with canvas:
        _render_plot_choice(choice, rows, summary, u_bounds, x_bounds)


def panel_state_flow(active_node):
    """RUN·STATE FLOW -- which agent in the LangGraph loop is running now."""
    render_section_tag("RUN · STATE FLOW")
    parts = []
    for i, (letter, name, key) in enumerate(_FLOW):
        on = "on" if key == active_node else ""
        pip = '<span class="pip"></span>' if on else ""
        parts.append(f'<div class="fnode {on}">{pip}<div class="ic">{letter}</div>'
                     f'<div class="nm">{name}</div></div>')
        if i < len(_FLOW) - 1:
            parts.append('<div class="fconn"></div>')
    st.markdown(f'<div class="panel"><div class="flow">{"".join(parts)}</div></div>',
                unsafe_allow_html=True)


def panel_candidate(cand, rows):
    """RUN·CANDIDATE -- the parameter set being evaluated right now."""
    render_section_tag("RUN · CANDIDATE")
    if not cand:
        st.markdown('<div class="panel"><div style="font-size:11.5px;color:#556074;">'
                    'Waiting for the first proposal...</div></div>', unsafe_allow_html=True)
        return
    qn = float(np.linalg.norm(cand.get("Q") or [0]))
    rn = float(np.linalg.norm(cand.get("R") or [0]))
    last = rows[-1] if rows else None
    strat = (last or {}).get("strategy", "\u2014")
    st.markdown(
        f'<div class="panel"><div class="kv" style="grid-template-columns:repeat(4,1fr);">'
        f'<div class="cell"><div class="k">Np</div><div class="v">{cand.get("Np","--")}</div></div>'
        f'<div class="cell"><div class="k">Nc</div><div class="v">{cand.get("Nc","--")}</div></div>'
        f'<div class="cell"><div class="k">||Q||</div><div class="v">{qn:.3g}</div></div>'
        f'<div class="cell"><div class="k">||R||</div><div class="v">{rn:.3g}</div></div>'
        f'</div><div style="font-size:11px;color:#8d9bb0;margin-top:9px;">'
        f'strategy <span style="font-family:var(--mono);color:#c3aeff;">{strat}</span></div></div>',
        unsafe_allow_html=True)


def panel_activity(events):
    """RUN·ACTIVITY -- timestamped feed of which agent did what."""
    render_section_tag("RUN · ACTIVITY")
    if not events:
        st.markdown('<div class="panel"><div style="font-size:11.5px;color:#556074;">'
                    'No activity yet.</div></div>', unsafe_allow_html=True)
        return
    items = "".join(
        f'<div class="fitem"><span class="t">{t}</span>'
        f'<span><span class="a">{a}</span> {msg}</span></div>'
        for t, a, msg in reversed(events[-40:]))
    st.markdown(f'<div class="panel"><div class="feed">{items}</div></div>', unsafe_allow_html=True)


def panel_reasoning(entries):
    """RUN·REASONING -- each agent's own stated reasoning, per iteration."""
    render_section_tag("RUN · REASONING")
    if not entries:
        st.caption("The agents' reasoning appears here as the run progresses.")
        return
    for e in reversed(entries[-12:]):
        with st.expander(f"#{e['iteration']} \u00b7 {e['agent']}", expanded=False):
            st.markdown(f"<div style='font-size:12.5px;line-height:1.6;'>{e['text']}</div>",
                        unsafe_allow_html=True)


def panel_iteration_table(rows):
    """RUN·TABLE -- every attempt's numbers, and the CSV of the same."""
    render_section_tag("RUN · TABLE")
    if not rows:
        st.caption("No attempts yet.")
        return
    df = pd.DataFrame([{
        "#": r["iteration"],
        "Status": "ok" if r.get("ok") and not r.get("unstable") else ("unstable" if r.get("unstable") else "failed"),
        "Np": r.get("np"), "Nc": r.get("nc"),
        "Q": r.get("Q_formatted"), "R": r.get("R_formatted"),
        "MSE": fmt_num(r["mse"]) if r.get("mse") is not None else "--",
        "Overshoot": fmt_num(r["overshoot"]) if r.get("overshoot") is not None else "--",
        "Settling": (fmt_num(r["settling"]) if r.get("settling") not in (None, float("inf")) else "--"),
        "Effort": fmt_num(r["effort"]) if r.get("effort") is not None else "--",
        "Strategy": r.get("strategy", ""),
    } for r in rows])
    st.dataframe(df, hide_index=True, use_container_width=True, height=260)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                        file_name="AgentMPC_iterations.csv", mime="text/csv", key="uic_csv")


# ============================================================================
# RUN -- orchestration
# ============================================================================

def section_run():
    """RUN -- the live console. One graph node is advanced per script run
    (the iterator lives in session state), rather than consuming the whole
    stream inside a single run.

    That structure is what makes the chart selector work DURING a run.
    Consuming the stream in one blocking loop forced every panel to be
    redrawn many times within a single script run, and while display
    elements can be replaced in an st.empty() container freely, a keyed
    widget is also entered into Streamlit's per-run element registry, which
    rejects a second registration outright. Advancing one node per run
    means every widget here is created exactly once per run, so the
    selector is just a normal widget again -- and because the iterator is
    held in session state, interacting with it re-runs the script without
    restarting the tuning."""
    summary = st.session_state.uic_plugin.summary()
    gen = st.session_state.uic_general
    mpc = st.session_state.uic_mpc_cfg
    rows = st.session_state.uic_results

    if st.session_state.uic_running and st.session_state.uic_iterator is None:
        _start_run(summary, gen, mpc)
        rows = st.session_state.uic_results

    if st.session_state.uic_run_error:
        st.error("The run stopped early.")
        with st.expander("Details"):
            st.code(st.session_state.uic_run_error, language="python")

    # ---- header: progress, stop, spend ----
    if st.session_state.uic_running:
        done = len(rows)
        h1, h2 = st.columns([5, 1])
        with h1:
            st.progress(min(done / max(gen["max_iterations"], 1), 1.0),
                         text=f"Attempt {done} / {gen['max_iterations']}")
        with h2:
            if st.button("Stop", type="primary", use_container_width=True, key="uic_stop"):
                st.session_state.uic_stop_requested = True
                st.session_state.uic_running = False
                st.session_state.uic_iterator = None
                st.rerun()
    panel_usage(st.session_state.uic_tracker, compact=True)

    panel_metrics(rows)
    if st.session_state.uic_perturbed:
        pp = ", ".join(f"**{k}** {old:.4g} \u2192 {new:.4g} ({(new/old-1)*100:+.1f}%)"
                       for k, (old, new) in st.session_state.uic_perturbed.items())
        st.info(f"Level 3 perturbed these physical parameters for this run: {pp}")

    left, right = st.columns([3, 1.15], gap="medium")
    with left:
        panel_plots(rows, summary, gen["u_bounds"], gen["x_bounds"])
    with right:
        panel_state_flow(st.session_state.uic_active_node)
        panel_candidate(st.session_state.uic_candidate, rows)
        panel_activity(st.session_state.uic_activity)

    t1, t2 = st.tabs(["Iterations", "Agent reasoning"])
    with t1:
        panel_iteration_table(rows)
    with t2:
        panel_reasoning(st.session_state.uic_reasoning)

    st.divider()
    if st.session_state.uic_running:
        st.caption("Tuning in progress -- you can still switch charts above while it runs.")
    else:
        section_export_actions(inline=True)
        b1, b2 = st.columns([1, 5])
        with b1:
            if st.button("Full export page \u2192", key="uic_run_export"):
                st.session_state.uic_stage = "export"
                st.rerun()
        with b2:
            if st.button("New run", key="uic_run_again"):
                for k in ("uic_results", "uic_activity", "uic_reasoning", "uic_perturbed",
                          "uic_candidate", "uic_run_error", "uic_pdf", "uic_script",
                          "uic_iterator", "uic_active_node", "uic_stop_requested"):
                    st.session_state[k] = ([] if isinstance(_DEFAULTS.get(k), list)
                                            else ({} if isinstance(_DEFAULTS.get(k), dict) else None))
                st.session_state.uic_stage = "launch_general"
                st.rerun()

    if st.session_state.uic_running:
        _advance_one_node(gen, mpc)


def _start_run(summary, gen, mpc):
    """RUN -- builds the graph and stores the stream iterator in session
    state. Called once, when the run stage is first entered."""
    dyn = st.session_state.uic_dyn

    cfg = Config()
    cfg.mpc.prediction_horizon = mpc["seed_params"]["Np"] if mpc["seed_params"] else 12
    cfg.mpc.control_horizon = mpc["seed_params"]["Nc"] if mpc["seed_params"] else 5
    cfg.data.dt_mpc = mpc["dt"]
    cfg.data.simulation_time = gen["simulation_time"]
    cfg.data.settling_tolerance = gen["settling_tolerance_pct"] / 100.0
    cfg.mpc.u_bounds = gen["u_bounds"]
    cfg.mpc.x_bounds = gen["x_bounds"]
    cfg.data.trajectory_mode = mpc["traj"]
    cfg.data.trajectory_amplitude = mpc["traj_amp"]
    cfg.data.trajectory_frequency = mpc["traj_freq"]

    if _TOKEN_TRACKING and st.session_state.uic_tracker is None:
        st.session_state.uic_tracker = TokenUsageTracker(default_model=LLM_MODEL)

    try:
        perturbed = {}
        if mpc["scenario"] != 1:
            _, _, perturbed = apply_scenario_level(dyn, cfg, mpc["scenario"])
        st.session_state.uic_perturbed = perturbed

        focus_line = ("" if mpc["focus"] == "balanced"
                      else f"Optimization focus: {OPTIMIZATION_FOCUS_LABELS[mpc['focus']]}. ")
        guidance = "\n".join(s for s in (focus_line, (mpc["guidance"] or "").strip()) if s)

        graph = build_ui_tuning_graph(dyn, cfg,
                                       entry_node="evaluator" if mpc["seed_params"] else "actor")
        state = initial_state(
            dyn, system_name=st.session_state.uic_plugin.dynamics_class.__name__,
            max_iterations=gen["max_iterations"], ui_scenario_level=mpc["scenario"],
            seed_params=mpc["seed_params"], user_guidance=guidance,
            min_explore_iterations=gen["min_explore_iterations"],
            cost_weights=OPTIMIZATION_FOCUS_PRESETS.get(mpc["focus"]),
            exploration_intensity=gen["exploration_intensity"],
            dt_mpc=mpc["dt"], token_tracker=st.session_state.uic_tracker)

        st.session_state.uic_iterator = graph.stream(state)
        st.session_state.uic_candidate = mpc["seed_params"]
    except Exception as e:  # noqa: BLE001
        st.session_state.uic_run_error = f"{e}\n\n{tb_module.format_exc()}"
        st.session_state.uic_running = False
        st.session_state.uic_iterator = None


@st.fragment(run_every=0.5)
def _advance_one_node(gen, mpc):
    """RUN -- pulls exactly ONE node off the graph stream per tick, appends
    whatever it produced to session state, and re-runs. Keeping this in a
    fragment means the tick does not re-execute the whole page."""
    if not st.session_state.uic_running or st.session_state.uic_iterator is None:
        return
    dyn = st.session_state.uic_dyn
    try:
        output = next(st.session_state.uic_iterator)
    except StopIteration:
        st.session_state.uic_running = False
        st.session_state.uic_iterator = None
        st.session_state.uic_active_node = None
        st.rerun()
        return
    except Exception as e:  # noqa: BLE001
        st.session_state.uic_run_error = f"{e}\n\n{tb_module.format_exc()}"
        st.session_state.uic_running = False
        st.session_state.uic_iterator = None
        st.rerun()
        return

    for node, update in output.items():
        st.session_state.uic_active_node = node
        st.session_state.uic_activity.append(
            (time.strftime("%H:%M:%S"), node.capitalize(), _NODE_LABEL.get(node, "ran")))
        if update.get("current_params"):
            st.session_state.uic_candidate = update["current_params"]
        # last_outputs is the channel EVERY agent writes its reasoning into
        # (see llm_base.merge_last_output) -- reading it here captures all
        # of them uniformly, rather than per-agent field names that do not
        # consistently exist (the Juror has no *_reasoning field at all).
        for agent_key, text in (update.get("last_outputs") or {}).items():
            it_no = update.get("iteration", len(st.session_state.uic_results))
            recent = st.session_state.uic_reasoning[-6:]
            if not any(r["iteration"] == it_no and r["agent"] == agent_key.capitalize()
                       and r["text"] == str(text) for r in recent):
                st.session_state.uic_reasoning.append(
                    {"iteration": it_no, "agent": agent_key.capitalize(), "text": str(text)})
        if node == "evaluator":
            it = update.get("iteration", len(st.session_state.uic_results))
            st.session_state.uic_results.append(
                build_row(update, it, dyn.n_states, dyn.n_inputs, mpc["scenario"]))
    st.rerun()



# ============================================================================
# RESULT · EXPORT
# ============================================================================

def section_export_actions(inline: bool = False):
    """RESULT·EXPORT -- the two generate/download pairs. Shared by the full
    export page and the inline block at the bottom of the run console, so
    the report and script are reachable without leaving the run view (the
    key suffix keeps the two placements from colliding when both render)."""
    summary = st.session_state.uic_plugin.summary()
    rows = st.session_state.uic_results
    gen = st.session_state.uic_general
    mpc = st.session_state.uic_mpc_cfg
    b = best_row(rows)
    sfx = "_inline" if inline else ""

    if b is None:
        st.caption("No stable, successful attempt yet -- nothing to export.")
        return

    c1, c2 = st.columns(2, gap="large")
    with c1:
        render_section_tag("RESULT · EXPORT / REPORT")
        if not inline:
            st.caption("The Report Agent reads the actual iteration history and writes an "
                       "analysis of it -- what worked, what did not, what it would try next.")
        if st.button("Generate PDF report", type="primary", use_container_width=True,
                      disabled=not LLM_READY, key=f"uic_pdf_go{sfx}"):
            try:
                with st.spinner("Writing the report..."):
                    analysis = generate_report_analysis(
                        system_name=summary.get("dynamics_class", "System"),
                        state_names=summary.get("state_names", []),
                        input_names=summary.get("input_names", []),
                        results_data=rows, best_row=b,
                        stopped_by_user=st.session_state.uic_stop_requested,
                        tracker=st.session_state.uic_tracker)
                    fd, path = tempfile.mkstemp(suffix=".pdf")
                    os.close(fd)
                    try:
                        build_pdf_report(path, summary.get("dynamics_class", "System"),
                                          summary, rows, b, analysis)
                        with open(path, "rb") as f:
                            st.session_state.uic_pdf = f.read()
                    finally:
                        if os.path.exists(path):
                            os.unlink(path)
            except Exception as e:  # noqa: BLE001
                st.error(f"Report generation failed: {e}")
        if st.session_state.uic_pdf:
            st.download_button("Download PDF", st.session_state.uic_pdf,
                                file_name="AgentMPC_report.pdf", mime="application/pdf",
                                use_container_width=True, key=f"uic_pdf_dl{sfx}")

    with c2:
        render_section_tag("RESULT · EXPORT / SCRIPT")
        if not inline:
            st.caption("One .py file with the dynamics, the MPC controller, the tuned "
                       "parameters and this run's exact scenario. Needs only numpy, scipy "
                       "and matplotlib.")
        if st.button("Generate standalone script", type="primary", use_container_width=True,
                      key=f"uic_scr_go{sfx}"):
            try:
                with st.spinner("Writing the script..."):
                    dyn = st.session_state.uic_dyn
                    qk = sorted((k for k in b if k.startswith("q") and k[1:].isdigit()),
                                 key=lambda k: int(k[1:]))
                    rk = sorted((k for k in b if k.startswith("r") and k[1:].isdigit()),
                                 key=lambda k: int(k[1:]))
                    st.session_state.uic_script = generate_standalone_script(
                        dynamics_source_code=st.session_state.uic_source,
                        class_name=summary.get("dynamics_class", "MyDynamics"),
                        best_params={"Np": b["np"], "Nc": b["nc"],
                                      "Q": [b[k] for k in qk], "R": [b[k] for k in rk]},
                        dt_mpc=b.get("dt_mpc") or mpc["dt"],
                        simulation_time=gen["simulation_time"],
                        system_name=summary.get("dynamics_class", "MyDynamics"),
                        initial_state=list(dyn.config.default_initial_state),
                        physical_params_override=dict(dyn.params) if dyn.params else None,
                        trajectory_mode=mpc["traj"],
                        trajectory_amplitude=mpc["traj_amp"],
                        trajectory_frequency=mpc["traj_freq"],
                        u_bounds=gen["u_bounds"], x_bounds=gen["x_bounds"])
            except Exception as e:  # noqa: BLE001
                st.error(f"Script generation failed: {e}")
        if st.session_state.uic_script:
            st.download_button("Download .py", st.session_state.uic_script,
                                file_name="AgentMPC_export.py", mime="text/x-python",
                                use_container_width=True, key=f"uic_scr_dl{sfx}")


def section_export():
    """RESULT·EXPORT -- the full export page: the same actions as the
    inline block on the run console, plus the metric summary and the
    session's LLM spend."""
    rows = st.session_state.uic_results
    b = best_row(rows)

    render_section_tag("RESULT · EXPORT", "Take the result with you",
                        "A written report of what the agents found, or a self-contained script "
                        "that reproduces this exact result on your own machine.")

    if b is None:
        st.warning("No stable, successful attempt in this run \u2014 nothing to export yet.")
        if st.button("\u2190 Back to the run", key="uic_exp_back0"):
            st.session_state.uic_stage = "run"
            st.rerun()
        return

    panel_metrics(rows)
    st.markdown("")
    section_export_actions(inline=False)

    if st.session_state.uic_tracker is not None:
        st.markdown("")
        panel_usage(st.session_state.uic_tracker)

    st.divider()
    e1, e2 = st.columns([1, 5])
    with e1:
        if st.button("\u2190 Back to the run", key="uic_exp_back"):
            st.session_state.uic_stage = "run"
            st.rerun()
    with e2:
        if st.button("Start over with a new system", key="uic_exp_reset"):
            reset_all()
            st.rerun()


# ============================================================================
# THEME/CHROME -- model selector
# ============================================================================

def render_model_selector(expanded: bool = False):
    """THEME/CHROME -- always available, not just when the LLM fails to
    initialize. Any model labcd_agents' LLMFactory recognizes can be typed
    in; the presets are only a shortlist. Changing it re-runs _init_llm
    (cached per model name), so switching providers mid-session works
    without restarting the app."""
    with st.expander("Model", expanded=expanded):
        opts = list(LLM_MODEL_PRESETS)
        cur = st.session_state.selected_llm_model
        if cur not in opts:
            opts.insert(0, cur)
        opts = opts + ["Other (type a name)..."]
        pick = st.selectbox("Active model", opts, index=opts.index(cur),
                             key="uic_model_pick", label_visibility="collapsed")
        if pick == "Other (type a name)...":
            typed = st.text_input("Model name", value="",
                                   placeholder="e.g. gpt-4o, claude-3-5-haiku-20241022",
                                   key="uic_model_typed", label_visibility="collapsed")
            if typed.strip() and typed.strip() != cur:
                st.session_state.selected_llm_model = typed.strip()
                st.rerun()
        elif pick != cur:
            st.session_state.selected_llm_model = pick
            st.rerun()
        if MODEL_FROM_ENV:
            st.caption(f"DEFAULT_LLM_MODEL in your .env file is set to `{DEFAULT_LLM_MODEL}`, "
                       f"which overrides this app's built-in default (`{BUILTIN_DEFAULT_MODEL}`). "
                       f"Edit or remove that line in .env to change what it opens with.")
            if st.session_state.selected_llm_model != BUILTIN_DEFAULT_MODEL:
                if st.button(f"Switch to {BUILTIN_DEFAULT_MODEL}", key="uic_model_builtin",
                              use_container_width=True):
                    st.session_state.selected_llm_model = BUILTIN_DEFAULT_MODEL
                    st.rerun()
        else:
            st.caption(f"Using the built-in default (`{BUILTIN_DEFAULT_MODEL}`). To change what "
                       f"the app opens with, set DEFAULT_LLM_MODEL in the .env file next to this "
                       f"script. See .env.example.")


# ============================================================================
# ROUTER -- the standalone entry point
# ============================================================================

render_topbar()

if not LLM_READY:
    st.error(f"{LLM_ERROR}")
    render_model_selector(expanded=True)
    st.stop()

_rail_col, _model_col = st.columns([5, 1])
with _rail_col:
    render_stage_rail()
with _model_col:
    render_model_selector(expanded=False)

_stage = st.session_state.uic_stage
if _stage == "upload":
    section_upload()
elif _stage == "consult":
    section_consult()
elif _stage == "launch_general":
    section_launch_general()
elif _stage == "launch_mpc":
    section_launch_mpc()
elif _stage == "run":
    section_run()
elif _stage == "export":
    section_export()
