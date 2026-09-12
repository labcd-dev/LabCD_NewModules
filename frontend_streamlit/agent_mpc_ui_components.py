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
        Dynamics file intake (a Python file, or described to AgentPlant's
        chat with no file at all), validation/auto-repair, Setup Agent
        analysis.

  SETUP · PRE-LAUNCH      section_prelaunch()
        One stage: for a described (file-less) system, initial_state first
        (_prelaunch_initial_conditions(), needed to even build the plugin);
        then, always, human-in-the-loop chat + structured JSON config
        suggestions covering every general setting and bound, PLUS the
        reference trajectory itself (built-in shapes, per-state override, or
        a custom-uploaded trajectory file) and the default target
        (_prelaunch_configure(), chat/suggestions backed by
        agents/config_advisor_agent.py), all still editable by hand.

  LAUNCH · MPC            section_launch_mpc()
        MPC-specific: scenario level + noise magnitude, seed Np/Nc/Q/R/dt,
        optimization focus, free-text guidance.

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
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field

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
try:
    from backend_core.AgentMPC.agents.diagnostics_agent import (
        ERROR_CATEGORY_TITLES, chat_about_issues, generate_diagnostics_report, scan_for_issues,
    )
    _DIAGNOSTICS_AVAILABLE = True
    _DIAGNOSTICS_ERROR = None
except ImportError as _diag_exc:  # pragma: no cover -- optional dependency
    _DIAGNOSTICS_AVAILABLE = False
    _DIAGNOSTICS_ERROR = str(_diag_exc)
from backend_core.AgentMPC.agents.evaluator import run_closed_loop
from backend_core.AgentMPC.agents.export_script import generate_standalone_script
from backend_core.AgentMPC.agents.formatting import fmt_num
from backend_core.AgentMPC.agents.llm_base import configure_llm, get_llm, invoke_with_retry
from backend_core.AgentMPC.agents.metrics import OPTIMIZATION_FOCUS_LABELS, OPTIMIZATION_FOCUS_PRESETS
from backend_core.AgentMPC.agents.report_agent import generate_report_analysis
from backend_core.AgentMPC.agents.report_pdf import build_pdf_report
from backend_core.AgentMPC.agents.scenario_presets import apply_scenario_level, suggested_noise_std
from backend_core.AgentMPC.agents.seed_params import parse_seed_params
from backend_core.AgentMPC.dynamics.loader import DynamicLoader
from backend_core.AgentMPC.dynamics.trajectory_loader import TrajectoryLoader
from backend_core.AgentMPC.graph.workflow import build_ui_tuning_graph, initial_state
from backend_core.AgentMPC.mpc.config import Config
from backend_core.AgentPlant.agent import (
    DEFAULT_MAX_DRAFTS,
    DEFAULT_MIN_USER_TURNS_BEFORE_COMPLETION,
    PlantModelAgent,
)
from backend_core.plant_compiler import PlantCompiler

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


_STAGES = [("Upload", "upload"), ("Setup", "setup"), ("Pre-Launch", "prelaunch"),
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
    "uic_iterator": None, "uic_stop_requested": False, "uic_advancing": False,
    "uic_activity": [], "uic_active_node": None, "uic_candidate": None,
    "uic_reasoning": [], "uic_perturbed": {},
    "uic_pdf": None, "uic_script": None, "uic_tracker": None,
    "selected_llm_model": None,
    "uic_upload_mode": "Upload a Python file", "uic_plant_agent": None, "uic_plant_messages": [],
    "uic_plant_final": None, "uic_prelaunch_suggestion": None,
    "uic_manual_result": None,
    "uic_traj_custom_fn": None, "uic_traj_custom_name": None,
    # Diagnostics Agent: the report is cached against a signature of WHAT is
    # wrong, so the automatic analysis costs one LLM call per distinct problem
    # rather than one per Streamlit rerun.
    "uic_diag_report": None, "uic_diag_chat": [], "uic_diag_signature": None,
    "uic_diag_error": None,
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


def plot_trajectory_preview(ref, times, state_names):
    """LAUNCH·MPC·PREVIEW -- one small subplot per state showing the
    reference trajectory Start tuning would actually use, generated the same
    way a real run would (SystemConfig.desired_trajectory), so a user
    customizing per-state modes can see the shape before committing to it."""
    n = ref.shape[1]
    fig, axs = plt.subplots(n, 1, figsize=(9.0, 1.1 * n + 0.9), sharex=True)
    fig.patch.set_facecolor(MPL["bg"])
    if n == 1:
        axs = [axs]
    for i in range(n):
        nm = state_names[i] if i < len(state_names) else f"x{i}"
        axs[i].plot(times, ref[:, i], color=MPL["ref"], lw=1.8, zorder=3)
        axs[i].fill_between(times, ref[:, i], color=MPL["ref"], alpha=0.12, zorder=2)
        _style_axes(axs[i])
        axs[i].set_ylabel(nm, color=MPL["text"], fontsize=8.5)
    axs[-1].set_xlabel("Time [s]", color=MPL["text"], fontsize=9)
    fig.suptitle("Reference trajectory preview", color=C["text"], fontsize=10.5, fontweight="600")
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
    (derivative pairs, initial Q/R, dt, feedforward trim). A system can also
    be obtained without a file at all via _section_describe_dynamics(), which
    hands off to the SETUP·PRE-LAUNCH stage instead of straight to consult."""
    render_section_tag(
        "SETUP · UPLOAD", "Load a system",
        "A Python file defining create_config() and a BaseDynamics subclass, or describe "
        "the plant in plain language and have the plant agent draft it with you.")

    mode = st.radio("Source", ["Upload a Python file", "Describe it to the AI"],
                     horizontal=True, label_visibility="collapsed", key="uic_upload_mode")
    if mode == "Describe it to the AI":
        _section_describe_dynamics()
        return

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
        if _load_plugin_from_source(src):
            st.rerun()


def _load_plugin_from_source(src: str) -> bool:
    """SETUP -- shared by the file-upload path above and the pre-launch
    "Build plugin" step below. Validates/auto-repairs src, instantiates the
    dynamics, and on success stores it in session state and advances to the
    consult stage. Returns whether it succeeded; the caller re-runs itself
    on True and otherwise has already shown the error."""
    with st.spinner("Validating and analyzing..."):
        setup = analyze_and_setup(src)
    if not setup.fix.valid:
        st.error(f"Could not load this system: "
                 f"{setup.fix.still_broken_error or setup.fix.original_error}")
        return False
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(setup.fix.final_code)
            tmp = f.name
        plugin = DynamicLoader.load_from_path(tmp)
        dyn = plugin.create_dynamics()
    except Exception as e:  # noqa: BLE001
        st.error(f"Error instantiating the system: {e}")
        return False
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)

    st.session_state.uic_dyn = dyn
    st.session_state.uic_plugin = plugin
    st.session_state.uic_source = setup.fix.final_code
    st.session_state.uic_setup = setup
    st.session_state.uic_stage = (
        "setup" if (setup.setup_notes or setup.qr_diagnostics or setup.derivative_pairs)
        else "prelaunch")
    return True


# ============================================================================
# SETUP · DESCRIBE -- obtain a system without a file, via AgentPlant's chat
# ============================================================================

def _get_plant_agent() -> PlantModelAgent:
    """SETUP·DESCRIBE -- one PlantModelAgent per session, reusing the same
    LLM the rest of this app is configured with, so there is only ever one
    active model rather than a second, independently-configured one."""
    if st.session_state.uic_plant_agent is None:
        st.session_state.uic_plant_agent = PlantModelAgent(
            model=LLM_MODEL, temperature=0.2,
            max_drafts=DEFAULT_MAX_DRAFTS,
            min_user_turns_before_completion=DEFAULT_MIN_USER_TURNS_BEFORE_COMPLETION)
    return st.session_state.uic_plant_agent


def _section_describe_dynamics():
    """SETUP·DESCRIBE -- a chat with AgentPlant's PlantModelAgent that ends
    in a system_name/python_code/metadata draft, same shape unified_app.py's
    Plant stage produces. From there control passes to SETUP·PRE-LAUNCH,
    which turns it into an actual AgentMPC plugin via PlantCompiler."""
    render_section_tag(
        "SETUP · UPLOAD / DESCRIBE", "Describe the plant instead",
        "No Python file? Describe the system in plain language -- the plant agent will "
        "draft the equations with you, asking a few questions at a time.")

    if not LLM_READY:
        st.error(f"The LLM is not configured: {LLM_ERROR}")
        return

    agent = _get_plant_agent()

    with st.container(border=True):
        if not st.session_state.uic_plant_messages:
            st.markdown(
                '<div class="msg a"><div class="av">AI</div><div class="bub">'
                "Describe the plant you want to control -- what it is, its parts and "
                "parameters, what you measure and actuate. I'll draft the equations and "
                "ask about anything unclear.</div></div>", unsafe_allow_html=True)
        for turn in st.session_state.uic_plant_messages:
            role = "u" if turn["role"] == "user" else "a"
            av = "YOU" if role == "u" else "AI"
            body = turn["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            st.markdown(f'<div class="msg {role}"><div class="av">{av}</div>'
                        f'<div class="bub">{body}</div></div>', unsafe_allow_html=True)

    if st.session_state.uic_plant_final is None:
        msg = st.chat_input("Describe your plant or refine the draft...", key="uic_plant_input")
        if msg:
            st.session_state.uic_plant_messages.append({"role": "user", "content": msg})
            try:
                with st.spinner("Thinking..."):
                    display, final = agent.step(st.session_state.uic_plant_messages[:-1], msg)
                st.session_state.uic_plant_messages.append({"role": "assistant", "content": display})
                if final is not None:
                    st.session_state.uic_plant_final = final
            except Exception as e:  # noqa: BLE001
                st.session_state.uic_plant_messages.append(
                    {"role": "assistant", "content": f"(The plant agent hit an error: {e})"})
            st.rerun()
    else:
        final = st.session_state.uic_plant_final
        st.success(f"Plant model ready: **{final.get('system_name')}**")
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("Continue →", type="primary", key="uic_plant_continue"):
                _s = st.session_state.uic_setup
                st.session_state.uic_stage = (
                    "setup" if _s is not None and (_s.setup_notes or _s.qr_diagnostics
                                                    or _s.derivative_pairs)
                    else "prelaunch")
                st.rerun()
        with c2:
            if st.button("Discard and restart", key="uic_plant_restart"):
                st.session_state.uic_plant_agent = None
                st.session_state.uic_plant_messages = []
                st.session_state.uic_plant_final = None
                st.session_state.uic_prelaunch_suggestion = None
                st.rerun()


# ============================================================================
# SETUP · PRE-LAUNCH -- initial state / target for a described (file-less)
# system, with the same LLM proposing starting values on request
# ============================================================================

class PreLaunchSuggestion(BaseModel):
    initial_state: List[float] = Field(
        description="One value per state, same order as given, for a realistic starting point "
                    "that actually exercises the dynamics (not necessarily an equilibrium).")
    rationale: str = Field(
        description="1-2 sentences on why this starting point makes physical sense for this system.")


def suggest_prelaunch(system_name, states, params, assumptions, tracker=None) -> PreLaunchSuggestion:
    """SETUP·PRE-LAUNCH -- a small, self-contained suggestion call (same
    get_llm()/invoke_with_retry pattern as config_advisor_agent.suggest_config)
    for the one thing a described-from-scratch system still needs a human (or
    the LLM) to pick before a plugin can even be built: where the simulation
    starts. The target/reference trajectory is decided later, in
    _prelaunch_configure(), once a dynamics instance exists to preview
    against -- see the "Reference trajectory" section there."""
    prompt = (
        f"You are a controls engineer picking a realistic initial state for a simulation of the "
        f"system \"{system_name}\".\n"
        f"States, in order: {states}\n"
        f"Parameters: {params or 'none declared'}\n"
        f"Assumptions: {assumptions or 'none stated'}\n\n"
        f"Propose a realistic initial state, one value per state, in exactly the order given -- "
        f"something that actually exercises the dynamics rather than sitting at equilibrium."
    )
    llm = get_llm().with_structured_output(PreLaunchSuggestion)
    return invoke_with_retry(llm, prompt, max_retries=1, node_name="PreLaunchAdvisor", tracker=tracker)


def section_prelaunch():
    """SETUP·PRE-LAUNCH -- one stage covering everything that can be settled
    before a tuning run starts. If we arrived via the describe-to-AI path and
    there is no dynamics instance yet, _prelaunch_initial_conditions() collects
    initial_state/default_target and builds the plugin first (needed before
    anything else here can even introspect the system). Once a dynamics
    instance exists -- from that build, or straight from a file upload --
    _prelaunch_configure() takes over: advisory chat, one LLM suggestion pass
    covering every general setting and bound, all still editable by hand."""
    if st.session_state.uic_dyn is None:
        _prelaunch_initial_conditions()
    else:
        _prelaunch_configure()


def _back_from_prelaunch() -> str:
    """Where Pre-Launch's back button goes: the Setup findings if this file
    produced any, otherwise straight to Upload."""
    s = st.session_state.uic_setup
    if s is not None and (s.setup_notes or s.qr_diagnostics or s.derivative_pairs):
        return "setup"
    return "upload"


def _prelaunch_initial_conditions():
    """SETUP·PRE-LAUNCH / INITIAL CONDITIONS -- only reached from the
    describe-to-AI path (a file upload already has a dynamics instance by the
    time this stage is entered). Turns the AgentPlant draft into an actual
    AgentMPC plugin via PlantCompiler.generate_mpc_plugin(), which needs an
    initial_state and default_target to fill in -- this screen collects
    those, optionally proposed by the LLM, then hands the compiled source to
    the same _load_plugin_from_source() the file-upload path uses. On success
    uic_dyn is set, so the next render of section_prelaunch() falls through
    to _prelaunch_configure() automatically -- no stage change needed."""
    final = st.session_state.uic_plant_final
    if final is None:
        st.warning("No described plant yet -- go back and describe one first.")
        if st.button("← Back", key="uic_pl_back0"):
            st.session_state.uic_stage = _back_from_prelaunch()
            st.rerun()
        return

    meta = final.get("metadata") or {}
    states = list(meta.get("states") or [])

    render_section_tag(
        "SETUP · PRE-LAUNCH", "Initial state",
        "Where the simulation starts. Ask the advisor to propose values from the plant's "
        "physical description, or set them yourself -- the target/reference trajectory is "
        "decided next, once the plugin exists to preview it against.")

    sug = st.session_state.uic_prelaunch_suggestion
    if st.button("Suggest initial state", disabled=not LLM_READY, key="uic_pl_sug_go"):
        try:
            with st.spinner("Proposing initial conditions..."):
                sug = suggest_prelaunch(
                    final.get("system_name"), states, meta.get("parameters"),
                    meta.get("assumptions"), tracker=st.session_state.uic_tracker)
                st.session_state.uic_prelaunch_suggestion = sug
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not produce a suggestion: {e}")

    if sug and sug.rationale:
        st.caption(f"Advisor's reasoning: {sug.rationale}")

    cols = st.columns(min(len(states), 4) or 1)
    x0 = []
    for i, name in enumerate(states):
        with cols[i % len(cols)]:
            default = sug.initial_state[i] if sug and i < len(sug.initial_state) else 0.0
            x0.append(st.number_input(name, value=float(default), key=f"uic_pl_x0_{i}", format="%.4f"))

    st.divider()
    b1, b2 = st.columns([1, 5])
    with b1:
        if st.button("Build plugin →", type="primary", key="uic_pl_next"):
            pre_launch = {"initial_state": x0}
            with st.spinner("Compiling plugin..."):
                src = PlantCompiler().generate_mpc_plugin(final, pre_launch)
            if _load_plugin_from_source(src):
                st.rerun()
    with b2:
        if st.button("← Back to describing the plant", key="uic_pl_back"):
            st.session_state.uic_stage = _back_from_prelaunch()
            st.rerun()


# ============================================================================
# SETUP · PRE-LAUNCH / CONFIGURE -- advisory chat + general settings + bounds,
# one screen, once a dynamics instance exists
# ============================================================================

def _prelaunch_configure():
    """SETUP·PRE-LAUNCH / CONFIGURE -- the human-in-the-loop step, merged with
    the general tuning settings and constraint bounds that used to be a
    separate stage: chat about the system (is MPC even right here? what
    should I watch out for?) and a structured suggestion pass fill in every
    general setting and bound right here, still fully editable by hand, with
    a single Continue taking the user straight to LAUNCH·MPC. Chat and
    suggestion both come from agents/config_advisor_agent.py."""
    summary = st.session_state.uic_plugin.summary()
    setup = st.session_state.uic_setup
    dyn = st.session_state.uic_dyn

    render_section_tag(
        "SETUP · PRE-LAUNCH", "Talk it through, then confirm the numbers",
        "Optional chat, then one suggestion pass that proposes every general setting and "
        "bound below -- the advisor sees your actual system, not a generic description. "
        "Edit anything, or skip the chat and suggestion entirely and set it all by hand.")

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
        render_section_tag("SETUP · PRE-LAUNCH / SUGGEST")
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

    sug = st.session_state.uic_suggestion
    g = sug.general_settings if sug else None

    st.markdown("")
    render_section_tag(
        "SETUP \u00b7 PRE-LAUNCH / SETTINGS", "Study settings and constraints",
        "Controller-agnostic. These frame the search itself -- how each candidate is "
        "evaluated and what limits it must respect." +
        (" Pre-filled from the advisor's suggestion above; edit anything." if g else ""))

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
    render_section_tag("SETUP · PRE-LAUNCH / CONSTRAINTS")
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

    st.markdown("")
    render_section_tag(
        "SETUP \u00b7 PRE-LAUNCH / TRAJECTORY", "Reference trajectory",
        "What the controller should track: one of the three built-in shapes (optionally "
        "per state), or a trajectory file you supply yourself.")

    with st.expander("Upload a custom trajectory file (.py)", expanded=False):
        st.caption("Must define create_trajectory(dt_mpc, simulation_time, n_states, state_names) "
                    "-> np.ndarray of shape (n_steps, n_states).")
        traj_up = st.file_uploader("Trajectory file", type=["py"], label_visibility="collapsed",
                                     key="uic_traj_up")
        if traj_up is not None and st.button("Validate and use", key="uic_traj_up_go"):
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                                  encoding="utf-8") as f:
                    f.write(traj_up.getvalue().decode("utf-8"))
                    tmp = f.name
                loader = TrajectoryLoader.load_from_path(tmp)
                st.session_state.uic_traj_custom_fn = loader.generate
                st.session_state.uic_traj_custom_name = loader.source_name
                st.success(f"Loaded: {loader.source_name}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not load: {e}")
            finally:
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)
        if st.session_state.uic_traj_custom_fn is not None:
            cu1, cu2 = st.columns([3, 1])
            with cu1:
                st.caption(f"Active custom trajectory: {st.session_state.uic_traj_custom_name}")
            with cu2:
                if st.button("Remove", key="uic_traj_up_clear"):
                    st.session_state.uic_traj_custom_fn = None
                    st.session_state.uic_traj_custom_name = None
                    st.rerun()

    traj_cards = list(_TRAJ_CARDS)
    if st.session_state.uic_traj_custom_fn is not None:
        traj_cards = traj_cards + [("custom", "Custom", st.session_state.uic_traj_custom_name)]
    traj = _card_choice(traj_cards, "traj", "reg")

    traj_amp, traj_freq, per_state_modes = 0.5, 0.5, None
    if traj == "custom":
        st.caption("Using the uploaded file -- its own logic decides the shape per state.")
    else:
        customize_per_state = st.checkbox(
            "Customize per state", value=False, key="uic_traj_per_state_toggle",
            help="Pick a different trajectory type (regulation/sinusoidal/cosinusoidal/pulse) for "
                 "each state individually, instead of the one global choice above.")
        if customize_per_state:
            _traj_labels = {"reg": "Regulation", "sin": "Sinusoidal", "cos": "Cosinusoidal", "pulse": "Pulse"}
            _traj_codes = {v: k for k, v in _traj_labels.items()}
            default_label = _traj_labels.get(traj, "Regulation")
            edited = st.data_editor(
                pd.DataFrame({"State": st_names, "Trajectory": [default_label] * len(st_names)}),
                disabled=["State"], hide_index=True, use_container_width=True, key="uic_traj_editor",
                column_config={"Trajectory": st.column_config.SelectboxColumn(options=list(_traj_labels.values()))})
            per_state_modes = [_traj_codes[v] for v in edited["Trajectory"]]
            if any(m in ("sin", "cos", "pulse") for m in per_state_modes):
                t1, t2 = st.columns(2)
                with t1:
                    traj_amp = st.slider("Amplitude", 0.05, 3.0, 0.5, step=0.05, key="uic_tamp")
                with t2:
                    traj_freq = st.slider("Frequency (Hz)", 0.05, 3.0, 0.5, step=0.05, key="uic_tfreq")
        elif traj in ("sin", "pulse"):
            t1, t2 = st.columns(2)
            with t1:
                traj_amp = st.slider("Amplitude", 0.05, 3.0, 0.5, step=0.05, key="uic_tamp")
            with t2:
                traj_freq = st.slider("Frequency (Hz)", 0.05, 3.0, 0.5, step=0.05, key="uic_tfreq")

    st.markdown("**Default target**")
    tgt_cols = st.columns(min(len(st_names), 4) or 1)
    target = []
    cur_target = dyn.config.default_target
    for i, name in enumerate(st_names):
        with tgt_cols[i % len(tgt_cols)]:
            target.append(st.number_input(f"target {name}", value=float(cur_target[i]),
                                           key=f"uic_traj_tgt_{i}", format="%.4f"))

    dt_preview = st.number_input(
        "dt_mpc for this preview (s)", 0.0005, 1.0, float(setup.suggested_dt or 0.02),
        step=0.001, format="%.4f", key="uic_traj_preview_dt",
        help="Only affects the preview's sample density -- the actual run's dt_mpc is chosen "
             "on the next screen (LAUNCH\u00b7MPC), where the Actor can also tune it.")
    with st.expander("Preview reference trajectory", expanded=False):
        try:
            if traj == "custom":
                ref = np.asarray(st.session_state.uic_traj_custom_fn(
                    dt_preview, float(sim_time), dyn.n_states, st_names))
            else:
                ref = dyn.config.desired_trajectory(
                    dt_mpc=dt_preview, simulation_time=float(sim_time), mode=traj,
                    amplitude=traj_amp, frequency=traj_freq, per_state_modes=per_state_modes)
            times = np.linspace(0, float(sim_time), ref.shape[0])
            show_fig(plot_trajectory_preview(ref, times, st_names))
        except Exception as e:  # noqa: BLE001
            st.caption(f"Could not preview: {e}")

    st.divider()
    b1, b2 = st.columns([1, 5])
    with b1:
        if st.button("Continue \u2192", type="primary", key="uic_gen_next"):
            dyn.config.default_target = np.array(target, dtype=float)
            st.session_state.uic_general = {
                "simulation_time": sim_time, "settling_tolerance_pct": settle_tol,
                "max_iterations": max_iter, "min_explore_iterations": min_explore,
                "exploration_intensity": explore_int,
                "u_bounds": (u_df["Min"].to_numpy(dtype=float), u_df["Max"].to_numpy(dtype=float)),
                "x_bounds": (x_df["Min"].to_numpy(dtype=float), x_df["Max"].to_numpy(dtype=float)),
                "trajectory_mode": traj, "trajectory_amplitude": traj_amp,
                "trajectory_frequency": traj_freq, "trajectory_per_state_modes": per_state_modes,
                "trajectory_custom_fn": (st.session_state.uic_traj_custom_fn
                                         if traj == "custom" else None),
            }
            st.session_state.uic_stage = "launch_mpc"
            st.rerun()
    with b2:
        if st.button("\u2190 Load a different system", key="uic_gen_back"):
            reset_all()
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


def plot_probe_states(diag, state_names):
    """SETUP -- the open-loop step-response probe, one trace per state, with
    the measured range drawn as an explicit bracket.

    "range" is the number Bryson's rule divides by, so it is the one quantity
    in the whole setup analysis worth being able to SEE rather than take on
    faith: the bracket is exactly max-min of the trace beside it.
    """
    traj = np.asarray(diag["trajectory"], dtype=float)
    probe_dt, ranges = diag["probe_dt"], diag["ranges"]
    t = np.arange(traj.shape[0]) * probe_dt
    n = traj.shape[1]
    ncols = max(1, min(3, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 1.9 * nrows),
                              facecolor=MPL["bg"], squeeze=False)
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i >= n:
            ax.axis("off")
            continue
        y = traj[:, i]
        ax.plot(t, y, color=MPL["series"][i % len(MPL["series"])], linewidth=1.6)
        lo, hi = float(np.min(y)), float(np.max(y))
        ax.axhline(lo, color=MPL["ref"], linestyle="--", linewidth=0.8)
        ax.axhline(hi, color=MPL["ref"], linestyle="--", linewidth=0.8)
        ax.annotate("", xy=(t[-1] * 0.92, hi), xytext=(t[-1] * 0.92, lo),
                     arrowprops=dict(arrowstyle="<->", color=MPL["text"], linewidth=0.9))
        ax.text(t[-1] * 0.9, (hi + lo) / 2, f"{ranges[i]:.3g}", color=MPL["text"],
                 fontsize=7, ha="right", va="center")
        _style_axes(ax, state_names[i] if i < len(state_names) else f"x{i}")
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def plot_probe_inputs(diag, input_names):
    """SETUP -- the +/- input step each R weight was measured from."""
    step_mag = list(diag["step_mag"])
    n = len(step_mag)
    fig, ax = plt.subplots(figsize=(2.4, max(1.9, 0.7 * n)), facecolor=MPL["bg"])
    pos = np.arange(n)
    ax.barh(pos, step_mag, color=MPL["input"], height=0.45)
    ax.barh(pos, [-s for s in step_mag], color=MPL["input"], alpha=0.35, height=0.45)
    ax.axvline(0, color=MPL["axis"], linewidth=1.0)
    ax.set_yticks(pos)
    ax.set_yticklabels([input_names[j] if j < len(input_names) else f"u{j}" for j in range(n)],
                        color=MPL["text"], fontsize=7)
    for j, s in enumerate(step_mag):
        ax.text(s, j, f"  \u00b1{s:.3g}", color=MPL["text"], fontsize=7, va="center")
    _style_axes(ax, "input step")
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def _bryson_rows(names, values, measured, symbol):
    """One substituted equation per variable, as a single aligned LaTeX block.

    A prose sentence about whichever state happened to get the largest weight
    does not let you check any of the others, so every state and every input
    gets its own line carrying the range the probe actually measured.
    """
    rows = r" \\ ".join(
        rf"{symbol}_{{\text{{{(names[i] if i < len(names) else f'{symbol.lower()}{i}')}}}}}"
        rf" &= \dfrac{{1}}{{{measured[i]:.3g}^{{2}}}}\ (\text{{rescaled}}) = \mathbf{{{fmt_num(values[i])}}}"
        for i in range(len(values))
    )
    return r"\begin{aligned}" + rows + r"\end{aligned}"


def section_setup():
    """SETUP -- what the Initial Setup Agent found for THIS file.

    The compact frontend used to go straight from Upload to Pre-Launch, so the
    suggested dt/Q/R simply turned up as pre-filled numbers with no account of
    where they came from. They are not guesses: they are measured off a real
    open-loop probe of the uploaded plant, and a user being asked to accept
    them is entitled to see that measurement. Same content as the main app's
    Setup Agent panel, laid out in this file's tighter idiom.
    """
    setup = st.session_state.uic_setup
    dyn = st.session_state.uic_dyn
    if setup is None or dyn is None:
        st.session_state.uic_stage = 'upload'
        st.rerun()

    summary = st.session_state.uic_plugin.summary()
    state_names = summary.get("state_names", []) or []
    input_names = summary.get("input_names", []) or []
    diag = setup.qr_diagnostics
    pairs = setup.derivative_pairs or []
    was_fixed = bool(getattr(setup.fix, "was_fixed", False) or getattr(setup.fix, "fixed_code", None))

    render_section_tag(
        "SETUP \u00b7 INITIAL ANALYSIS", "What the Setup Agent found",
        "Derivative pairs, a Bryson's-rule starting Q/R and a sample time \u2014 all measured from the "
        "plant itself by short simulations, never guessed by an LLM.")

    metric_cards([
        {"label": "Validation", "value": "Auto-repaired" if was_fixed else "Valid as uploaded"},
        {"label": "Derivative pairs", "value": str(len(pairs))},
        {"label": "dt_mpc (starting)",
         "value": f"{setup.suggested_dt:.4g}" if setup.suggested_dt else "n/a", "unit": "s"},
        {"label": "States / Inputs", "value": f"{len(state_names)} / {len(input_names)}"},
    ], cols=4)

    if pairs:
        st.caption("Derivative pairs: " + ", ".join(
            f"{state_names[j]} = d({state_names[i]})/dt"
            for i, j in pairs if i < len(state_names) and j < len(state_names)))

    if setup.setup_notes:
        with st.expander("Setup notes", expanded=False):
            for note in setup.setup_notes:
                st.markdown(f"- {note}")

    if diag and setup.suggested_Q:
        st.divider()
        render_section_tag("SETUP \u00b7 STEP-RESPONSE PROBE", "Where the weights were measured",
                            "Open-loop, a 25% input step on top of equilibrium. The bracket on each state "
                            "and the \u00b1 bar on each input are exactly what Q and R are computed from.")
        pc1, pc2 = st.columns([3, 1])
        with pc1:
            show_fig(plot_probe_states(diag, state_names))
        with pc2:
            show_fig(plot_probe_inputs(diag, input_names))

        st.latex(r"Q_{ii} = \frac{1}{\text{range}_i^{\,2}} \cdot "
                 r"\frac{Q_{\max}}{\max_k(1/\text{range}_k^{\,2})} \qquad "
                 r"R_{jj} = \frac{1}{\text{step}_j^{\,2}} \cdot "
                 r"\frac{R_{\max}}{\max_k(1/\text{step}_k^{\,2})}")
        st.caption("Weight each variable by the inverse square of its own characteristic scale, so no "
                    "state or input dominates the cost just because of its units.")
        qc, rc = st.columns(2)
        with qc:
            st.caption("State weights (Q) \u2014 measured range per state")
            st.latex(_bryson_rows(state_names, setup.suggested_Q, diag["ranges"], "Q"))
        with rc:
            st.caption("Input weights (R) \u2014 measured step per input")
            st.latex(_bryson_rows(input_names, setup.suggested_R, diag["step_mag"], "R"))
    else:
        st.caption("Step-response probe unavailable for this file \u2014 flat default weights will be used.")

    if setup.suggested_dt:
        st.divider()
        st.markdown(f"**Sample time \u2014 dt_mpc \u2248 {setup.suggested_dt:.4g}s**")
        st.caption("The smaller of two estimates: 1/15th of the fastest linearized time constant at "
                    "equilibrium, and 1/8th of the fastest state's step-response rise time. Used as the "
                    "STARTING value \u2014 whether the Actor may then tune it is your choice in the MPC step.")

    st.divider()
    b1, b2 = st.columns([1, 5])
    with b1:
        if st.button("Continue \u2192", type="primary", key="uic_setup_next"):
            st.session_state.uic_stage = "prelaunch"
            st.rerun()
    with b2:
        if st.button("\u2190 Load a different system", key="uic_setup_back"):
            reset_all()
            st.rerun()


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
    scenario it is graded against: scenario level + noise magnitude, seed
    Np/Nc/Q/R/dt, optimization focus, free-text guidance. The reference
    trajectory itself (built-in shapes or a custom upload) is decided
    earlier, in SETUP·PRE-LAUNCH / TRAJECTORY -- see _prelaunch_configure()."""
    summary = st.session_state.uic_plugin.summary()
    setup = st.session_state.uic_setup
    n_states = summary.get("n_states", 4)
    n_inputs = summary.get("n_inputs", 1)

    render_section_tag(
        "LAUNCH · MPC", "Controller and scenario",
        "MPC-specific. What the controller starts from, and how hard a scenario it is graded against.")

    st.markdown("**Scenario level**")
    scenario = _card_choice(_SCENARIO_CARDS, "scenario", 1)
    noise_std_value = None
    if scenario in (2, 3):
        noise_std_value = st.slider(
            "Noise level (std)", 0.0, 1.0, float(suggested_noise_std(st.session_state.uic_dyn)),
            step=0.001, format="%.4f", key="uic_noise_std",
            help="Standard deviation of the additive measurement noise applied to every state "
                 "at each step (Robust applies half of this).")

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
        dt_tuned = st.checkbox(
            "tune dt", value=True, key="uic_dt_tune",
            help="On: dt starts here and the Actor may propose a different sample time "
                 "during the run, like Q/R/Np/Nc. Off: this value is used for every "
                 "iteration and the Actor is told not to propose one -- pick this when dt "
                 "is dictated by your hardware or an existing control loop.",
        )
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
                "scenario": scenario, "noise_std": noise_std_value,
                "seed_params": seed_params, "dt": dt_val, "tune_dt": dt_tuned,
                "focus": focus, "guidance": guidance,
            }
            st.session_state.uic_stage = "run"
            st.session_state.uic_running = True
            st.rerun()
    with b2:
        if st.button("\u2190 Back", key="uic_mpc_back"):
            st.session_state.uic_stage = "prelaunch"
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
    the pipeline (report_agent, report_pdf, export_script) already expects.
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
    attempts that actually ran, did not diverge, and preferably passed the
    stability check. Kept in step with agent_mpc_app.py:_best_row -- see the
    rationale there for why stability has to be part of the ranking and not
    only of the verdict."""
    cand = [r for r in rows if r.get("ok") and not r.get("unstable") and r.get("cost") is not None]
    if not cand:
        return None
    stable = [r for r in cand if r.get("is_stable")]
    return min(stable or cand, key=lambda r: r["cost"])


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
    """RUN REASONING -- every entry recorded for the run, uncapped: a long
    run produces far more than a handful of entries, and silently keeping
    only the last few would make most agents' reasoning permanently
    unreachable even though it is sitting right there in session state."""
    render_section_tag("RUN · REASONING")
    if not entries:
        st.caption("The agents' reasoning appears here as the run progresses.")
        return
    for e in reversed(entries):
        with st.expander(f"#{e['iteration']} \u00b7 {e['agent']}", expanded=True):
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


def panel_manual_simulation(summary, gen, candidate):
    """RUN·MANUAL -- a one-off closed-loop simulation with hand-chosen
    parameters, independent of the tuning loop and usable at any time --
    especially right after Stop, to inspect how the last candidate the
    agents produced actually behaves. Reuses run_closed_loop(), the same
    deterministic simulator the Evaluator agent calls; no LLM involved."""
    render_section_tag(
        "RUN · MANUAL SIMULATION", "Run one simulation by hand",
        "Uses the exact same closed-loop simulator the Evaluator agent calls, with parameters "
        "you choose -- pre-filled from the last candidate the agents produced.")
    dyn = st.session_state.uic_dyn
    setup = st.session_state.uic_setup
    n_states, n_inputs = dyn.n_states, dyn.n_inputs
    st_names = summary.get("state_names", []) or [f"x{i}" for i in range(n_states)]
    in_names = summary.get("input_names", []) or [f"u{i}" for i in range(n_inputs)]
    cand = candidate or {}

    sQ = cand.get("Q") or setup.suggested_Q or [1.0] * n_states
    sR = cand.get("R") or setup.suggested_R or [0.1] * n_inputs
    s1, s2, s3 = st.columns([1, 1, 2])
    with s1:
        m_np = st.number_input("Np", 1, 50, int(cand.get("Np", 12)), key="uic_man_np")
    with s2:
        m_nc = st.number_input("Nc", 1, 50, int(cand.get("Nc", 5)), key="uic_man_nc")
    with s3:
        m_dt = st.number_input("dt (s)", 0.0005, 1.0, float(setup.suggested_dt or 0.02),
                                step=0.001, format="%.4f", key="uic_man_dt")
    m_q_text = st.text_input(f"Q — {n_states} values", ", ".join(f"{v:.4g}" for v in sQ), key="uic_man_q")
    m_r_text = st.text_input(f"R — {n_inputs} values", ", ".join(f"{v:.4g}" for v in sR), key="uic_man_r")

    m_sim_time = st.slider("Simulation time (s)", 2.0, 30.0, float(gen["simulation_time"]),
                            step=0.5, key="uic_man_simtime")

    traj_opts = {"reg": "Regulation", "sin": "Sinusoidal", "pulse": "Pulse"}
    m_traj = st.selectbox("Reference trajectory", list(traj_opts.keys()),
                          format_func=lambda k: traj_opts[k], key="uic_man_traj")
    m_amp, m_freq = 0.5, 0.5
    if m_traj in ("sin", "pulse"):
        c1, c2 = st.columns(2)
        with c1:
            m_amp = st.slider("Amplitude", 0.05, 3.0, 0.5, step=0.05, key="uic_man_amp")
        with c2:
            m_freq = st.slider("Frequency (Hz)", 0.05, 3.0, 0.5, step=0.05, key="uic_man_freq")

    use_custom_x0 = st.checkbox("Override initial state", value=False, key="uic_man_x0_toggle")
    custom_x0 = None
    if use_custom_x0:
        cols = st.columns(min(n_states, 4) or 1)
        custom_x0 = []
        for i, name in enumerate(st_names):
            with cols[i % len(cols)]:
                custom_x0.append(st.number_input(
                    name, value=float(dyn.config.default_initial_state[i]),
                    key=f"uic_man_x0_{i}", format="%.4f"))

    if st.button("Run manual simulation", type="primary", key="uic_man_go"):
        params, err = parse_seed_params(m_np, m_nc, m_q_text, m_r_text, n_states, n_inputs)
        if err:
            st.session_state.uic_manual_result = {"error": err}
        else:
            cfg = Config()
            cfg.mpc.prediction_horizon = m_np
            cfg.mpc.control_horizon = m_nc
            cfg.data.dt_mpc = m_dt
            cfg.data.simulation_time = m_sim_time
            cfg.data.settling_tolerance = gen["settling_tolerance_pct"] / 100.0
            cfg.mpc.u_bounds = gen["u_bounds"]
            cfg.mpc.x_bounds = gen["x_bounds"]
            cfg.data.trajectory_mode = m_traj
            cfg.data.trajectory_amplitude = m_amp
            cfg.data.trajectory_frequency = m_freq

            # dyn is the SAME instance the tuning loop uses -- swap its
            # initial state only for this one call, always restoring it in
            # finally so a manual override never leaks into the next
            # tuning iteration or the next manual run.
            original_x0 = dyn.config.default_initial_state.copy()
            try:
                if custom_x0 is not None:
                    dyn.config.default_initial_state = np.array(custom_x0, dtype=float)
                with st.spinner("Running..."):
                    result = run_closed_loop(dyn, cfg, params)
            finally:
                dyn.config.default_initial_state = original_x0
            st.session_state.uic_manual_result = result
        st.rerun()

    res = st.session_state.uic_manual_result
    if res is None:
        st.caption("No manual simulation run yet.")
        return
    if "error" in res:
        st.error(f"Simulation failed: {res['error']}")
        return

    m = res["metrics"]
    metric_cards([
        {"label": "MSE", "value": fmt_num(m.mse)},
        {"label": "Overshoot", "value": fmt_num(m.overshoot)},
        {"label": "Settling", "value": fmt_num(m.settling_time) if np.isfinite(m.settling_time) else "--",
         "unit": "s"},
        {"label": "Effort", "value": fmt_num(m.control_effort)},
    ], cols=4)
    if res["unstable"]:
        st.warning(f"Unstable: {res['unstable_reason']}")

    sim = {"states": res["states"], "inputs": res["inputs"], "refs": res["reference"],
           "times": res["times"], "n_states": n_states, "n_inputs": n_inputs}
    show_fig(plot_all_states(sim, st_names, gen["x_bounds"]))
    show_fig(plot_all_inputs(sim, in_names, gen["u_bounds"]))


# ============================================================================
# RUN -- orchestration
# ============================================================================

def _uic_diagnostics_context() -> str:
    """Everything the Diagnostics Agent needs to say something specific.

    Without this the agent can only reason from issue CATEGORIES and gives
    generic advice ("consider lowering Q"), which is the complaint that got
    the richer context added on the main app in the first place. It sees the
    same material a person debugging would open: the plant, the settings the
    run used, what each iteration actually scored, and the raw error.
    """
    parts = []
    dyn = st.session_state.uic_dyn
    summary = st.session_state.uic_plugin.summary() if st.session_state.uic_plugin else {}
    if dyn is not None:
        parts.append(
            f"PLANT: {summary.get('dynamics_class', '?')} -- "
            f"{summary.get('n_states', '?')} states {summary.get('state_names', [])}, "
            f"{summary.get('n_inputs', '?')} inputs {summary.get('input_names', [])}"
        )
        try:
            parts.append(f"Initial state: {[round(float(v), 4) for v in dyn.config.default_initial_state]}")
            parts.append(f"Target state: {[round(float(v), 4) for v in dyn.config.default_target]}")
        except Exception:  # noqa: BLE001 -- context is best-effort, never fatal
            pass
        if getattr(dyn, "params", None):
            parts.append(f"Physical parameters: {dict(dyn.params)}")

    gen = st.session_state.uic_general or {}
    mpc = st.session_state.uic_mpc_cfg or {}
    if gen:
        parts.append(
            f"RUN SETTINGS: simulation_time={gen.get('simulation_time')}s, "
            f"iteration budget={gen.get('max_iterations')} (min explore "
            f"{gen.get('min_explore_iterations')}), exploration intensity="
            f"{gen.get('exploration_intensity')}%, trajectory={gen.get('trajectory_mode')}"
        )
        parts.append(f"Input bounds: {gen.get('u_bounds')}")
        parts.append(f"State bounds: {gen.get('x_bounds')}")
    if mpc:
        parts.append(
            f"MPC START: dt={mpc.get('dt')} "
            f"({'tuned during the run' if mpc.get('tune_dt', True) else 'fixed by the user'}), "
            f"scenario level={mpc.get('scenario')}, focus={mpc.get('focus')}"
        )
        if mpc.get("guidance"):
            parts.append(f"User guidance to the agents: {mpc['guidance']}")
    if st.session_state.uic_perturbed:
        parts.append("Level 3 parameter perturbation: " + ", ".join(
            f"{k} {old:.4g} -> {new:.4g}" for k, (old, new) in st.session_state.uic_perturbed.items()))

    rows = st.session_state.uic_results or []
    if rows:
        parts.append(f"ITERATIONS ({len(rows)} total):")
        for r in rows[-12:]:
            if r.get("ok"):
                parts.append(
                    f"  #{r.get('iteration')}: Np={r.get('np')} Nc={r.get('nc')} "
                    f"Q=[{r.get('Q_formatted')}] R=[{r.get('R_formatted')}] -> "
                    f"MSE={r.get('mse')}, overshoot={r.get('overshoot')}, "
                    f"settling={r.get('settling')}, effort={r.get('effort')}, "
                    f"stable={r.get('is_stable')}, solver={r.get('solver_diagnostics')}"
                )
            else:
                parts.append(f"  #{r.get('iteration')}: FAILED -- {r.get('error')}")
    if st.session_state.uic_run_error:
        parts.append(f"RUN ERROR: {st.session_state.uic_run_error}")
    for entry in (st.session_state.uic_reasoning or [])[-4:]:
        parts.append(f"AGENT: {entry.get('text', '')[:600]}")
    return "\n".join(parts)


def _uic_diag_signature(findings: dict, run_error) -> str:
    """Identifies WHAT is wrong right now, so the automatic analysis runs once
    per distinct problem instead of once per Streamlit rerun -- which, on the
    live-run path, would be an LLM call every few hundred milliseconds."""
    parts = [f"{k}:{v.get('count', 0)}" for k, v in sorted(findings.items())]
    parts.append(f"err:{hash(str(run_error)) if run_error else 'none'}")
    parts.append(f"iters:{len(st.session_state.uic_results or [])}")
    return "|".join(parts)


def _uic_findings():
    """The deterministic scan's view of this run.

    ``uic_activity`` holds 3-tuples ``(timestamp, agent, message)`` -- the
    shape panel_activity unpacks -- while scan_for_issues wants log dicts
    keyed by "message"/"node". Translating here, rather than changing what
    the activity feed stores, leaves each consumer with the shape it already
    expects.

    Deliberately total. This runs inside the Diagnostics tab, and Streamlit
    renders every tab's contents on every script tick INCLUDING mid-run, so
    anything that raises here does not just break diagnostics -- it takes
    down the live run page with it.
    """
    rows = st.session_state.uic_results or []
    if not rows or not _DIAGNOSTICS_AVAILABLE:
        return {}
    try:
        events = []
        for entry in (st.session_state.uic_activity or []):
            if isinstance(entry, dict):
                events.append({"message": entry.get("message") or entry.get("text", ""),
                               "node": entry.get("node", "")})
            else:
                fields = (list(entry) + ["", "", ""])[:3]
                events.append({"message": str(fields[2]), "node": str(fields[1])})
        return scan_for_issues(events, rows, {})
    except Exception:  # noqa: BLE001 -- a scan failure must not hide the run
        return {}


def panel_auto_diagnostics():
    """RUN -- surfaces problems at the TOP of the run page and explains them
    without waiting to be asked.

    Leaving the explanation behind a button means the moment you most need
    help -- a run that just failed -- you first have to notice the button.
    Three guards keep the automatic call cheap: never while a run is live,
    only when something is actually wrong, and only once per distinct problem
    signature.
    """
    if st.session_state.uic_running or not _DIAGNOSTICS_AVAILABLE:
        return
    rows = st.session_state.uic_results or []
    run_error = st.session_state.uic_run_error
    if not rows and not run_error:
        return

    findings = _uic_findings()
    all_failed = bool(rows) and all(not r.get("ok") for r in rows)
    if not findings and not run_error and not all_failed:
        return

    signature = _uic_diag_signature(findings, run_error)
    if st.session_state.uic_diag_signature != signature:
        if LLM_READY:
            try:
                with st.spinner("Diagnostics Agent analyzing what went wrong..."):
                    st.session_state.uic_diag_report = generate_diagnostics_report(
                        findings, len(rows), run_context=_uic_diagnostics_context(),
                        tracker=st.session_state.uic_tracker)
                st.session_state.uic_diag_error = None
            except Exception as exc:  # noqa: BLE001 -- diagnosis must never mask the original failure
                st.session_state.uic_diag_report = None
                st.session_state.uic_diag_error = str(exc)
        st.session_state.uic_diag_signature = signature

    headline = (f"{len(rows)} iteration(s) all failed" if all_failed and not findings
                else ", ".join(f"{v['count']}\u00d7 {k.replace('_', ' ')}" for k, v in findings.items())
                or "the run reported an error")
    with st.container(border=True):
        st.warning(f"**Diagnostics \u2014 {headline}.** Analyzed automatically; full detail and a chat "
                    f"about it are in the **Diagnostics** tab below.")
        report = st.session_state.uic_diag_report
        if report is not None:
            for rec in report.recommendations:
                st.markdown(f"**{ERROR_CATEGORY_TITLES.get(rec.category, rec.category)}**")
                st.write(rec.explanation)
                st.markdown(f"\u2192 {rec.recommendation}")
                st.caption(f"Contribution to this run's problems: {rec.contribution_estimate}")
        elif st.session_state.uic_diag_error:
            st.caption(f"The automatic analysis could not run: {st.session_state.uic_diag_error}")
        elif not LLM_READY:
            st.caption("The LLM is not configured, so only the deterministic scan is available.")


def panel_diagnostics():
    """RUN -- the Diagnostics Agent in full: deterministic scan, the LLM's
    grounded recommendations, and a free-form chat about them.

    Three layers, deliberately in this order: scan_for_issues() is free and
    runs on every render; the report is one LLM call already made by the
    automatic banner; the chat is where a specific question gets a specific
    answer, grounded in the same run context rather than in the category
    label alone.
    """
    if not _DIAGNOSTICS_AVAILABLE:
        st.warning(f"Diagnostics Agent unavailable: {_DIAGNOSTICS_ERROR}")
        return

    render_section_tag(
        "RUN \u00b7 DIAGNOSTICS", "What went wrong, and what to change",
        "The deterministic scan, the agent's recommendations, and somewhere to ask follow-up questions.")

    run_error = st.session_state.uic_run_error
    if run_error:
        st.error(f"This run failed with an error: {run_error}")

    findings = _uic_findings()
    context = _uic_diagnostics_context()

    if findings:
        categories = ", ".join(f"{v['count']}\u00d7 {k.replace('_', ' ')}" for k, v in findings.items())
        st.warning(f"**{len(findings)} issue type(s) detected this run** ({categories})")
        label = ("Re-analyze" if st.session_state.uic_diag_report is not None
                 else "Get detailed recommendations")
        if st.button(label, key="uic_diag_recs"):
            try:
                with st.spinner("Diagnostics Agent analyzing..."):
                    st.session_state.uic_diag_report = generate_diagnostics_report(
                        findings, len(st.session_state.uic_results or []), run_context=context,
                        tracker=st.session_state.uic_tracker)
                st.session_state.uic_diag_error = None
            except Exception as exc:  # noqa: BLE001
                st.session_state.uic_diag_error = str(exc)
            st.rerun()

        report = st.session_state.uic_diag_report
        if report is not None:
            for rec in report.recommendations:
                st.markdown(f"**{ERROR_CATEGORY_TITLES.get(rec.category, rec.category)}**")
                st.write(rec.explanation)
                st.caption(f"Contribution to this run's problems: {rec.contribution_estimate}")
                st.markdown(f"\u2192 {rec.recommendation}")
                st.divider()
    elif not run_error:
        st.success("No issues detected by the automatic scan so far.")

    st.divider()
    st.markdown("**Ask the Diagnostics Agent**")
    st.caption("Ask about a specific error, or describe what you have already tried \u2014 the agent sees "
                "the detected issues, its own report above, this run's settings and every iteration's "
                "numbers.")
    for turn in st.session_state.uic_diag_chat:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    question = st.chat_input("e.g. \"why did Q3 keep growing?\" or paste an error",
                              key="uic_diag_input")
    if question:
        st.session_state.uic_diag_chat = st.session_state.uic_diag_chat + [
            {"role": "user", "content": question}]
        try:
            with st.spinner("Thinking..."):
                reply = chat_about_issues(
                    question, findings, st.session_state.uic_diag_report,
                    st.session_state.uic_diag_chat[:-1], run_error=run_error,
                    run_context=context, tracker=st.session_state.uic_tracker)
        except Exception as exc:  # noqa: BLE001
            reply = f"Couldn't get a reply: {exc}"
        st.session_state.uic_diag_chat = st.session_state.uic_diag_chat + [
            {"role": "assistant", "content": reply}]
        st.rerun()


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
    panel_auto_diagnostics()

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

    # "Manual tuning" rather than "Manual simulation": the panel exists to let
    # you hand-tune a candidate after the agents stop, and the old label read as
    # a diagnostic tool nobody needed to open.
    #
    # These four labels are FIXED. An earlier version appended a hint to the
    # last one once a run stopped, which was wrong twice over: a changed label
    # gives Streamlit a different tabs element, so the strip remounts and jumps
    # back to tab 1 at exactly the moment you want tab 4 -- and the longer label
    # pushes the right-hand tabs off the end of a narrow strip, where a tab that
    # has to be scrolled to is a tab nobody finds.
    t1, t2, t3, t4 = st.tabs(["Iterations", "Agent reasoning", "Diagnostics", "Manual tuning"])
    with t1:
        panel_iteration_table(rows)
    with t2:
        panel_reasoning(st.session_state.uic_reasoning)
    with t3:
        panel_diagnostics()
    with t4:
        panel_manual_simulation(summary, gen, st.session_state.uic_candidate)

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
                st.session_state.uic_stage = "prelaunch"
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
    cfg.data.tune_dt = mpc.get("tune_dt", True)
    cfg.data.simulation_time = gen["simulation_time"]
    cfg.data.settling_tolerance = gen["settling_tolerance_pct"] / 100.0
    cfg.mpc.u_bounds = gen["u_bounds"]
    cfg.mpc.x_bounds = gen["x_bounds"]
    cfg.data.trajectory_mode = gen["trajectory_mode"]
    cfg.data.trajectory_amplitude = gen["trajectory_amplitude"]
    cfg.data.trajectory_frequency = gen["trajectory_frequency"]
    cfg.data.trajectory_per_state_modes = gen.get("trajectory_per_state_modes")
    cfg.data.custom_trajectory_fn = gen.get("trajectory_custom_fn")

    if _TOKEN_TRACKING and st.session_state.uic_tracker is None:
        st.session_state.uic_tracker = TokenUsageTracker(default_model=LLM_MODEL)

    try:
        perturbed = {}
        if mpc["scenario"] != 1:
            _, _, perturbed = apply_scenario_level(
                dyn, cfg, mpc["scenario"], noise_std_value=mpc.get("noise_std"))
        st.session_state.uic_perturbed = perturbed

        focus_line = ("" if mpc["focus"] == "balanced"
                      else f"Optimization focus: {OPTIMIZATION_FOCUS_LABELS[mpc['focus']]}. ")
        guidance = "\n".join(s for s in (focus_line, (mpc["guidance"] or "").strip()) if s)

        # max_iterations has to reach the builder as well as initial_state: it
        # sizes LangGraph's recursion budget so the run ends on the iteration
        # cap instead of GraphRecursionError (see graph/workflow.py).
        graph = build_ui_tuning_graph(dyn, cfg,
                                       entry_node="evaluator" if mpc["seed_params"] else "actor",
                                       max_iterations=gen["max_iterations"])
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
    # run_every fires this fragment on a timer regardless of whether the
    # previous tick's next() call (which can block for seconds inside an
    # LLM node) has returned yet. Without this guard two ticks can call
    # next() on the same generator concurrently, which raises
    # "ValueError: generator already executing".
    if st.session_state.uic_advancing:
        return
    st.session_state.uic_advancing = True
    dyn = st.session_state.uic_dyn
    try:
        output = next(st.session_state.uic_iterator)
    except StopIteration:
        st.session_state.uic_running = False
        st.session_state.uic_iterator = None
        st.session_state.uic_active_node = None
        st.session_state.uic_advancing = False
        st.rerun()
        return
    except Exception as e:  # noqa: BLE001
        st.session_state.uic_run_error = f"{e}\n\n{tb_module.format_exc()}"
        st.session_state.uic_running = False
        st.session_state.uic_iterator = None
        st.session_state.uic_advancing = False
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
    st.session_state.uic_advancing = False
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
                        trajectory_mode=gen["trajectory_mode"],
                        trajectory_amplitude=gen["trajectory_amplitude"],
                        trajectory_frequency=gen["trajectory_frequency"],
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
elif _stage == "setup":
    section_setup()
elif _stage == "prelaunch":
    section_prelaunch()
elif _stage == "launch_mpc":
    section_launch_mpc()
elif _stage == "run":
    section_run()
elif _stage == "export":
    section_export()
