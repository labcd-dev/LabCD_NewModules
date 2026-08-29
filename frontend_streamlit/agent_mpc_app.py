"""
================================================================================
app.py -- MPC Agent Tuner (Streamlit UI)
================================================================================
Live dashboard for the LLM-driven MPC parameter tuning loop, built on the
`backend_core.AgentMPC` package.

Design notes (v2 -- rewritten after the first version hid real failures
behind zeroed-out metrics):

  * ERRORS ARE NEVER SILENT. If a closed-loop simulation fails for any
    reason (bad plugin, mismatched Q/R length, Np too large for the
    trajectory length, a solver error, ...), the Results table marks that
    row FAILED and the full error + traceback is shown in an expander --
    instead of the UI falling back to metrics.get(..., 0.0) and rendering a
    misleading row of zeros. See agents/evaluator.py's run_closed_loop for
    the corresponding fix on the engine side (it now wraps the *entire*
    simulation, not just the per-step solve, so a controller-construction
    failure can't slip past uncaught either).

  * "Test Dynamics" button. Right after loading a plugin, you can run one
    quick closed-loop check (default params, a short simulation) before
    committing to a full multi-iteration tuning run. This is the fastest way
    to find out a plugin is broken -- in one click instead of after burning
    through several LLM-tuning iterations.

  * PERSISTENT RESULTS. All rendering reads from st.session_state and is
    called on every rerun (not just while `running` is True), so the
    dashboard doesn't go blank the moment a run finishes.

  * Tabbed layout: Live Run / Convergence / Simulation / Agent Reasoning /
    Data & Export, instead of one long vertically-stacked page.

Run:
    pip install -r requirements.txt
    echo "GROQ_API_KEY=your-groq-api-key" >> .env
    streamlit run app.py        (or: python run_ui.py)
"""

from __future__ import annotations

import html as html_module
import os
import sys
import tempfile
import time
import traceback as tb_module
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Repo root (parent of backend_core/ and frontend_streamlit/) must be on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend_core.AgentMPC.agents.evaluator import run_closed_loop
from backend_core.AgentMPC.agents.llm_base import configure_llm
from backend_core.AgentMPC.agents.formatting import fmt_num
from backend_core.AgentMPC.agents.metrics import OPTIMIZATION_FOCUS_LABELS, OPTIMIZATION_FOCUS_PRESETS
try:
    from backend_core.AgentMPC.agents.report_agent import generate_report_analysis
    from backend_core.AgentMPC.agents.report_pdf import build_pdf_report
    REPORT_FEATURE_AVAILABLE = True
    REPORT_FEATURE_ERROR = None
except ImportError as e:
    # reportlab is a real dependency (see requirements.txt) but an easy one
    # to end up missing -- e.g. after pulling an update that added it,
    # without re-running `pip install -r requirements.txt`. That should
    # disable just the Report button with a clear, actionable message, not
    # crash the entire app at import time before the user even sees a UI.
    REPORT_FEATURE_AVAILABLE = False
    REPORT_FEATURE_ERROR = (
        f"{e}. Running interpreter: {sys.executable} -- if `pip install reportlab` reported "
        f"it as already satisfied elsewhere, that pip likely belongs to a DIFFERENT Python "
        f"installation than the one shown above (common on Windows with more than one Python "
        f"installed). Fix: \"{sys.executable}\" -m pip install -r requirements.txt"
    )
try:
    from backend_core.AgentMPC.agents.export_script import generate_standalone_script
    EXPORT_SCRIPT_FEATURE_AVAILABLE = True
    EXPORT_SCRIPT_FEATURE_ERROR = None
except ImportError as e:
    # Only stdlib dependencies (re, pathlib, typing) -- this should
    # essentially never fail, but the same resilience pattern costs nothing
    # and protects against any unexpected environment issue.
    EXPORT_SCRIPT_FEATURE_AVAILABLE = False
    EXPORT_SCRIPT_FEATURE_ERROR = str(e)
try:
    from backend_core.AgentMPC.agents.animation_agent import generate_animation_code, render_animation_gif
    ANIMATION_FEATURE_AVAILABLE = True
    ANIMATION_FEATURE_ERROR = None
except ImportError as e:
    ANIMATION_FEATURE_AVAILABLE = False
    ANIMATION_FEATURE_ERROR = str(e)
try:
    from backend_core.AgentMPC.agents.diagnostics_agent import ERROR_CATEGORY_TITLES, generate_diagnostics_report, scan_for_issues
    DIAGNOSTICS_FEATURE_AVAILABLE = True
    DIAGNOSTICS_FEATURE_ERROR = None
except ImportError as e:
    DIAGNOSTICS_FEATURE_AVAILABLE = False
    DIAGNOSTICS_FEATURE_ERROR = str(e)
try:
    from backend_core.AgentMPC.agents.advisory_agent import chat as advisory_chat
    ADVISORY_CHAT_AVAILABLE = True
    ADVISORY_CHAT_ERROR = None
except ImportError as e:
    ADVISORY_CHAT_AVAILABLE = False
    ADVISORY_CHAT_ERROR = str(e)
try:
    from backend_core.AgentMPC.agents.llm_base import TokenUsageTracker
    TOKEN_TRACKING_AVAILABLE = True
    TOKEN_TRACKING_ERROR = None
except ImportError as e:
    TOKEN_TRACKING_AVAILABLE = False
    TOKEN_TRACKING_ERROR = str(e)
from backend_core.AgentMPC.agents.scenario_presets import SCENARIO_LEVEL_NAMES, apply_scenario_level, suggested_noise_std
from backend_core.AgentMPC.agents.seed_params import parse_seed_params
from backend_core.AgentMPC.dynamics.base import SystemSimulator
from backend_core.AgentMPC.dynamics.loader import DynamicLoader, DynamicsPluginError
from backend_core.AgentMPC.graph.workflow import build_ui_tuning_graph, initial_state
from backend_core.AgentMPC.mpc.config import Config
from backend_core.AgentMPC.utils.logging_utils import configure_logging, get_logger

# ============================================================================
# PAGE CONFIG + THEME
# ============================================================================

st.set_page_config(page_title="LabCD · MPC Studio", page_icon=":gear:", layout="wide", initial_sidebar_state="collapsed")
configure_logging()
log = get_logger(__name__)

st.markdown("""
<style>
    html, body, .stApp {
        background: radial-gradient(ellipse 1400px 900px at 15% -10%, rgba(77,159,255,0.04), transparent 60%),
                    #0a0d13;
    }
    section[data-testid="stSidebar"] { background: rgba(6,9,17,0.7); border-right: 1px solid rgba(255,255,255,0.04); }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(77,159,255,0.18); border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(77,159,255,0.32); }

    /* ---- LabCD-matching top bar ---- */
    .lcd-topbar { display:flex; align-items:center; justify-content:space-between;
        padding: 0.9rem 1.5rem; margin: -1rem -1rem 1.4rem -1rem; border-bottom: 1px solid rgba(255,255,255,0.07); }
    .lcd-brand { display:flex; align-items:center; gap:0.75rem; }
    .lcd-logo { width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center;
        background: linear-gradient(135deg, rgba(77,159,255,0.18), rgba(129,140,248,0.14)); font-size:1.25rem; }
    .lcd-brand-text .lcd-title { color:#f1f3f7; font-size:1.15rem; font-weight:700; line-height:1.2; }
    .lcd-brand-text .lcd-subtitle { color:#7d8598; font-size:0.72rem; line-height:1.2; }
    .lcd-nav { display:flex; align-items:center; gap:1.4rem; flex-wrap:wrap; }
    .lcd-nav-item { color:#a3aec2; font-size:0.85rem; font-weight:500; white-space:nowrap;
        display:inline-flex; align-items:center; gap:0.35rem; }
    .lcd-nav-item svg { width:15px; height:15px; stroke: currentColor; flex-shrink:0; }
    .lcd-nav-item svg polygon[fill] { fill: currentColor; }
    .lcd-logo svg { width:20px; height:20px; stroke: currentColor; color:#8fc4ff; }
    .lcd-nav-pill { background: linear-gradient(135deg,#4d9fff,#3a7fe0); color:#fff !important; padding:0.4rem 0.9rem;
        border-radius:8px; font-size:0.82rem; font-weight:600; white-space:nowrap; }
    .lcd-nav-outline { border:1px solid rgba(255,255,255,0.12); color:#c5cbda; padding:0.35rem 0.8rem;
        border-radius:8px; font-size:0.8rem; white-space:nowrap; }
    .lcd-avatar { width:26px; height:26px; border-radius:50%; background:linear-gradient(135deg,#818cf8,#4d9fff);
        display:inline-flex; align-items:center; justify-content:center; font-size:0.7rem; font-weight:700; color:#0a0d13; }

    /* ---- title card with icon-square, matching "Control Design Setup" ---- */
    .lcd-title-card { display:flex; align-items:flex-start; gap:1rem; background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 1.3rem 1.5rem; margin-bottom: 1rem; }
    .lcd-icon-square { min-width:44px; height:44px; border-radius:10px; background: rgba(77,159,255,0.14);
        display:flex; align-items:center; justify-content:center; font-size:1.3rem; color:#5b9dff; }
    .lcd-title-card .lcd-h { color:#f1f3f7; font-size:1.35rem; font-weight:700; margin:0 0 0.2rem 0; }
    .lcd-title-card .lcd-sub { color:#7d8598; font-size:0.88rem; margin:0; }

    /* ---- stepper ---- */
    .lcd-stepper { display:flex; align-items:flex-start; justify-content:space-between; background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 1.6rem 2rem; margin-bottom: 1rem; position:relative; }
    .lcd-step { display:flex; flex-direction:column; align-items:center; text-align:center; flex:1; position:relative; z-index:2; }
    /* Each step draws its OWN connector segment running from its circle's
       right edge to the next step's circle's left edge -- deliberately NOT
       one continuous line with circles layered on top, which was still
       faintly visible crossing through the "done" circles (their fill is a
       semi-transparent rgba(77,159,255,0.1), so a line behind them showed
       through). A segment that simply stops at calc(50% + 24px) can never
       render under a circle in the first place. */
    .lcd-step:not(:last-child)::after {
        content:""; position:absolute; top:24px; left:calc(50% + 24px); width:calc(100% - 48px); height:2px;
        background: rgba(255,255,255,0.1); z-index:1; border-radius:2px;
    }
    .lcd-step.done:not(:last-child)::after {
        background: linear-gradient(90deg, #4d9fff, #818cf8);
        animation: lcdSegAppear 0.6s ease-out;
    }
    @keyframes lcdSegAppear { from { opacity: 0; } to { opacity: 1; } }
    .lcd-step-circle { width:48px; height:48px; border-radius:50%; display:flex; align-items:center; justify-content:center;
        margin-bottom:0.6rem; border:1.5px solid rgba(255,255,255,0.14); background:#0f1219; color:#5f6a80;
        transition: border-color 0.3s ease, background 0.3s ease, box-shadow 0.3s ease, color 0.3s ease; }
    .lcd-step-circle svg { width:22px; height:22px; stroke: currentColor; }
    .lcd-step-circle svg circle[fill], .lcd-step-circle svg path[fill] { fill: currentColor; }
    .lcd-step.done .lcd-step-circle { border-color:#4d9fff; color:#4d9fff; background:#132338; }
    .lcd-step.active .lcd-step-circle { border-color:#4d9fff; color:#0a0d13; background:#4d9fff;
        box-shadow: 0 0 0 4px rgba(77,159,255,0.18), 0 0 18px rgba(77,159,255,0.35);
        animation: lcdStepPop 0.45s cubic-bezier(0.3,1.5,0.4,1); }
    @keyframes lcdStepPop { 0% { transform: scale(0.85); } 60% { transform: scale(1.08); } 100% { transform: scale(1); } }
    .lcd-step-label { color:#c5cbda; font-size:0.88rem; font-weight:700; }
    .lcd-step.active .lcd-step-label, .lcd-step.done .lcd-step-label { color:#f1f3f7; }
    .lcd-step-sub { color:#6b7488; font-size:0.72rem; margin-top:0.15rem; }

    /* ---- collapsible "Advanced Settings" look ---- */
    .lcd-advanced [data-testid="stExpander"] { border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 12px !important; background: rgba(255,255,255,0.015) !important; }

    /* ---- native file_uploader restyled to look like the LabCD dropzone ---- */
    [data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,0.015) !important;
        border: 1.5px dashed rgba(255,255,255,0.16) !important; border-radius: 14px !important;
        padding: 2.6rem 0 1.6rem 0 !important; position: relative !important; min-height: 120px !important;
        transition: border-color 0.2s ease, background 0.2s ease !important; }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: rgba(77,159,255,0.45) !important;
        background: rgba(77,159,255,0.03) !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] svg { display:none; }
    [data-testid="stFileUploaderDropzoneInstructions"]::before {
        content:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%235b9dff%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M12%203v12%22/%3E%3Cpath%20d%3D%22M7%208l5-5%205%205%22/%3E%3Cpath%20d%3D%22M4%2017v2a2%202%200%200%200%202%202h12a2%202%200%200%200%202-2v-2%22/%3E%3C/svg%3E");
        display:block; position:absolute; top:1.1rem; left:50%; transform:translateX(-50%);
        width:26px; height:26px; padding:11px; box-sizing:content-box;
        background:rgba(77,159,255,0.14); border-radius:50%;
    }

    .glass-card, .metric-card {
        background: rgba(20, 30, 50, 0.32);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.03);
        transition: all 0.25s cubic-bezier(0.2, 0.8, 0.2, 1);
    }
    .glass-card { padding: 1.2rem 1.5rem; margin-bottom: 0.8rem; }
    .glass-card:hover { border-color: rgba(77, 159, 255, 0.18); box-shadow: 0 8px 26px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04); }
    .metric-card { padding: 1rem 1.2rem; text-align: center; }
    .metric-card:hover { border-color: rgba(77, 159, 255, 0.22); transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.32), 0 0 0 1px rgba(77,159,255,0.06); }
    .metric-card .label { font-size: 0.65rem; color: #6a7a9a; text-transform: uppercase; letter-spacing: 2px; }
    .metric-card .value { font-size: 1.55rem; font-weight: 700; color: #e6f1ff; letter-spacing: -0.3px; }
    .value-cyan { color: #4d9fff; } .value-purple { color: #818cf8; }
    .value-yellow { color: #f1fa8c; } .value-green { color: #50fa7b; } .value-red { color: #ff5555; }
    .status-ready { color: #4d9fff; font-weight: 600; }
    .status-running { color: #f1fa8c; font-weight: 600; animation: pulse 1.5s ease-in-out infinite; }
    .status-failed { color: #ff5555; font-weight: 600; }
    @keyframes pulse { 0% {opacity:1;} 50% {opacity:0.5;} 100% {opacity:1;} }
    .progress-container { background: rgba(20,30,50,0.25); border-radius: 10px; padding: 0.6rem 1rem;
        border: 1px solid rgba(255,255,255,0.05); box-shadow: inset 0 1px 3px rgba(0,0,0,0.2); }
    .progress-bar-bg { width:100%; height:5px; background: rgba(45,51,73,0.35); border-radius:3px;
        overflow:hidden; margin-top:0.4rem; box-shadow: inset 0 1px 2px rgba(0,0,0,0.3); }
    .progress-bar-fill { height:100%; background: linear-gradient(90deg,#4d9fff,#818cf8);
        border-radius:3px; transition: width 0.4s cubic-bezier(0.2, 0.8, 0.2, 1); box-shadow: 0 0 10px rgba(77,159,255,0.5); }
    .log-container, .reasoning-container { background: rgba(6,10,18,0.55); border-radius: 12px;
        padding: 0.7rem; max-height: 420px; overflow-y: auto; border: 1px solid rgba(255,255,255,0.05);
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.25); }
    .log-entry { padding: 0.25rem 0.6rem; font-family: 'Consolas', monospace; font-size: 0.75rem;
        color: #aabbdd; border-radius: 4px; }
    .log-entry:hover { background: rgba(77,159,255,0.06); }
    .log-time { color: #3a4a6a; } .log-node { color: #4d9fff; font-weight: 600; }
    .log-metric { color: #f1fa8c; } .log-error { color: #ff5555; } .log-success { color: #50fa7b; }
    .reasoning-entry { background: rgba(20,30,50,0.28); border-radius: 10px; padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem; border-left: 3px solid rgba(77,159,255,0.35);
        box-shadow: 0 2px 10px rgba(0,0,0,0.15); transition: border-color 0.2s ease; }
    .reasoning-entry:hover { border-left-color: rgba(77,159,255,0.7); }
    .reasoning-entry .r-header { display:flex; justify-content:space-between; margin-bottom:0.3rem; font-size:0.7rem; }
    .reasoning-entry .r-node { font-weight:700; letter-spacing:1px; text-transform:uppercase; }
    .reasoning-entry .r-time { color: #3a4a6a; }
    .reasoning-entry .r-text { color: #c5d3ea; font-size: 0.85rem; line-height: 1.5; white-space: pre-wrap; }
    .r-node-scenarist { color: #818cf8; } .r-node-actor { color: #4d9fff; } .r-node-critic { color: #38bdf8; }
    .r-node-terminator { color: #60a5fa; } .r-node-juror { color: #ff5555; } .r-node-evaluator { color: #8be9fd; }
    .dataframe { background: rgba(20,30,50,0.18) !important; border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.05) !important; }
    .stButton button { background: linear-gradient(135deg, rgba(65,90,140,0.55), rgba(90,110,160,0.5)) !important;
        color: #e6f1ff !important; border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 10px !important; font-weight: 500 !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important; transition: all 0.2s cubic-bezier(0.2,0.8,0.2,1) !important; }
    .stButton button:hover { transform: translateY(-2px) !important; box-shadow: 0 10px 26px rgba(77,110,180,0.25) !important;
        border-color: rgba(77,159,255,0.35) !important; }
    .stButton button:active { transform: translateY(0) !important; }
    button[kind="primary"] { background: linear-gradient(135deg, #4d9fff, #3a7fe0) !important; border: none !important; }
    button[kind="primary"]:hover { box-shadow: 0 10px 28px rgba(77,159,255,0.35) !important; }
    hr { border:none !important; height:1px !important;
        background: linear-gradient(90deg, transparent, rgba(77,159,255,0.08), transparent) !important; margin: 1.2rem 0 !important; }
    .glow-text { background: linear-gradient(135deg, #4d9fff, #818cf8); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700;
        filter: drop-shadow(0 0 18px rgba(77,159,255,0.18)); }
    .header-glass { background: linear-gradient(135deg, rgba(20,30,50,0.35), rgba(20,30,50,0.15));
        backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.06); border-top: 1px solid rgba(77,159,255,0.2);
        border-radius: 14px; padding: 1.2rem 1.5rem; margin-bottom: 1.2rem; box-shadow: 0 8px 30px rgba(0,0,0,0.25); }
    .subheader { color: #8a9aba; font-size: 0.9rem; font-weight: 600; margin: 1.2rem 0 0.5rem 0;
        padding-bottom: 0.4rem; border-bottom: 1px solid rgba(255,255,255,0.06); letter-spacing: 0.3px; }
    .llm-badge { display:inline-block; background: rgba(77,159,255,0.1); border: 1px solid rgba(77,159,255,0.22);
        color: #4d9fff; border-radius: 999px; padding: 0.15rem 0.7rem; font-size: 0.7rem; font-weight: 600;
        letter-spacing: 0.5px; box-shadow: 0 0 12px rgba(77,159,255,0.12); }
    .fail-badge { display:inline-block; background: rgba(255,85,85,0.1); border: 1px solid rgba(255,85,85,0.25);
        color: #ff5555; border-radius: 999px; padding: 0.15rem 0.7rem; font-size: 0.7rem; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(255,255,255,0.06); }
    .stTabs [data-baseweb="tab"] { color: #6a7a9a; font-weight: 500; border-radius: 8px 8px 0 0; padding: 0.5rem 1rem; }
    .stTabs [data-baseweb="tab"]:hover { color: #c5d3ea; background: rgba(77,159,255,0.05); }
    .stTabs [aria-selected="true"] { color: #4d9fff !important; border-bottom: 2px solid #4d9fff !important; }
    div[data-testid="stMetric"] { background: rgba(20,30,50,0.28); border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px; padding: 0.7rem 0.9rem; box-shadow: 0 3px 12px rgba(0,0,0,0.18); }
    /* ---- neutralize Streamlit's built-in "stale content" fade during
       reruns -- with a fragment auto-ticking during a run, this default
       fade-while-updating behavior is what reads as periodic dimming/
       flicker. Forcing full, non-transitioning opacity on the main content
       containers keeps the page visually stable across reruns. ---- */
    [data-testid="stAppViewContainer"], [data-testid="stMain"], .main, .block-container,
    [data-testid="stVerticalBlock"], [data-testid="stElementContainer"] {
        opacity: 1 !important;
        transition: none !important;
    }
    div[data-stale="true"] { opacity: 1 !important; }

    /* ---- site-wide font (item 7) ----
       Deliberately NOT using a blanket "span, div, [data-testid] { ... !important }"
       selector here -- that was the actual root cause of icon text (e.g.
       "refresh", "arrow_right") rendering as literal words instead of
       glyphs: forcing !important onto every span/div also overrides the
       few of them that Streamlit itself gives an explicit icon font to,
       and there's no reliable way to enumerate every such element by hand.
       Setting the font at the very top of the DOM and letting normal CSS
       INHERITANCE carry it down instead means any element with its OWN
       explicit font-family (icons) simply keeps it -- inheritance never
       overrides an element's own explicit rule, with or without !important
       on the ancestor. */
    html, body, .stApp {
        font-family: 'Times New Roman', Times, serif;
    }
    /* Form controls (inputs, buttons, selects) are a long-standing CSS
       exception -- browsers give them their own default UI font instead of
       inheriting from the page by default. Explicitly telling them to
       inherit (not re-hardcoding the font name) keeps the same "icons stay
       exempt" property for anything nested inside them. */
    input, textarea, select, button {
        font-family: inherit;
    }
    /* Reinforce on plain text-carrying tags specifically -- p/h1-h6/label
       are, in practice, never used by Streamlit as icon-ligature
       containers (those are always span/div with a specific data-testid),
       so !important here is safe and helps override any Streamlit default
       stylesheet rule that might otherwise win on specificity alone. */
    p, h1, h2, h3, h4, h5, h6, label {
        font-family: 'Times New Roman', Times, serif !important;
    }
    code, pre, .stCode, [data-testid="stCodeBlock"] {
        font-family: 'Consolas', 'Courier New', monospace !important;
    }
    /* Reinforce on specific, high-confidence Streamlit text containers --
       these render only user-facing text content, never an icon ligature,
       so !important here is safe and fills in any gap plain inheritance
       might leave against Streamlit's own more-specific internal rules. */
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
    [data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"],
    [data-testid="stText"], [data-testid="stDataFrame"], [data-testid="stTable"],
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        font-family: 'Times New Roman', Times, serif !important;
    }
    /* Belt-and-suspenders: explicitly pin known Streamlit icon-font
       elements to their real icon font, in case some Streamlit-internal
       rule has higher specificity than plain inheritance would win against. */
    /* ================================================================
       DEEP WIDGET RESKIN -- sliders, and a from-scratch card-selector
       component system used to replace dropdowns for the highest-impact
       choices (Scenario Level, Trajectory Type).
       ================================================================ */

    /* ---- native slider: glowing gradient track + larger tactile thumb.
       Targets ARIA role (stable, web-standard) rather than BaseWeb's
       internal hashed classes (which change across Streamlit versions),
       so this degrades gracefully -- worst case it's plain, never broken. */
    div[data-testid="stSlider"] { padding: 0.35rem 0 0.15rem 0; }
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div:nth-child(2) {
        background: linear-gradient(90deg, #2a3f5f, #4d9fff) !important;
        height: 5px !important; border-radius: 4px;
        box-shadow: 0 0 10px rgba(77,159,255,0.35);
    }
    div[data-testid="stSlider"] div[role="slider"] {
        width: 20px !important; height: 20px !important;
        background: radial-gradient(circle at 35% 30%, #ffffff, #8fc4ff 55%, #4d9fff) !important;
        box-shadow: 0 0 0 4px rgba(77,159,255,0.18), 0 2px 8px rgba(0,0,0,0.5) !important;
        border: none !important; transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stSlider"] div[role="slider"]:hover,
    div[data-testid="stSlider"] div[role="slider"]:focus {
        transform: scale(1.15);
        box-shadow: 0 0 0 7px rgba(77,159,255,0.28), 0 2px 10px rgba(0,0,0,0.6) !important;
    }

    /* ---- card-selector: a grid of clickable cards standing in for a
       dropdown. Built entirely from st.button (full control, no BaseWeb
       internals to fight) -- the active card is marked via a CSS class
       toggled by a wrapper div (see render_card_selector). ---- */
    /* Card buttons (both states share the base look; kind="primary" is
       set programmatically on whichever card is currently selected --
       see render_card_selector -- since Streamlit doesn't actually nest
       widgets created after a raw st.markdown() div as its DOM children,
       so a parent-wrapper + nth-child approach can't reliably target the
       right card; styling the button's OWN kind attribute can. */
    div[data-testid="stButton"] button[kind="secondary"].card-btn,
    div[data-testid="stButton"] button {
        min-height: 84px; border-radius: 14px;
        text-align: left; white-space: pre-wrap; line-height: 1.35;
    }
    div[data-testid="column"] div[data-testid="stButton"] button[kind="secondary"],
    div[data-testid="column"] div[data-testid="stButton"] button[data-testid="baseButton-secondary"] {
        background: linear-gradient(165deg, rgba(19,35,56,0.9), rgba(10,15,25,0.9));
        border: 1.5px solid rgba(255,255,255,0.08); color: #8a9aba;
        transition: all 0.18s ease;
    }
    div[data-testid="column"] div[data-testid="stButton"] button[kind="secondary"]:hover,
    div[data-testid="column"] div[data-testid="stButton"] button[data-testid="baseButton-secondary"]:hover {
        border-color: rgba(77,159,255,0.45);
        background: linear-gradient(165deg, rgba(24,44,70,0.95), rgba(12,18,30,0.95));
        transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    }
    div[data-testid="column"] div[data-testid="stButton"] button[kind="primary"],
    div[data-testid="column"] div[data-testid="stButton"] button[data-testid="baseButton-primary"] {
        background: linear-gradient(165deg, rgba(30,60,100,0.6), rgba(15,30,50,0.8)) !important;
        border: 1.5px solid #4d9fff !important; color: #f1f3f7 !important;
        box-shadow: 0 0 0 1px rgba(77,159,255,0.4), 0 6px 20px rgba(77,159,255,0.15) !important;
    }
    .card-title { font-weight: 700; font-size: 0.95rem; color: inherit; margin-bottom: 2px; display:block; }
    .card-desc { font-weight: 400; font-size: 0.78rem; opacity: 0.85; display:block; }

    /* ---- per-state weight bar (used by the redesigned Q/R input) ---- */
    .weight-row { display:flex; align-items:center; gap:0.7rem; margin-bottom:0.35rem; }
    .weight-label { width:110px; flex-shrink:0; font-size:0.82rem; color:#8a9aba; text-align:right; }
    .weight-value { width:52px; flex-shrink:0; font-size:0.82rem; color:#4d9fff; font-weight:600; text-align:left; }

    [data-testid="stIconMaterial"], [data-testid*="Icon"], [data-testid="stExpanderIcon"],
    .material-icons, .material-icons-outlined, .material-icons-round, .material-icons-sharp,
    .material-symbols-outlined, .material-symbols-rounded, .material-symbols-sharp,
    [class*="material-symbol"], [class*="material-icon"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
    }

</style>
""", unsafe_allow_html=True)


def render_lcd_topbar():
    st.markdown(
        '<div class="lcd-topbar">'
        '<div class="lcd-brand">'
        f'<div class="lcd-logo">{LCD_ICON_LOGO}</div>'
        '<div class="lcd-brand-text"><div class="lcd-title">LabCD &middot; MPC</div>'
        '<div class="lcd-subtitle">AI-Powered Control System Design Studio</div></div>'
        '</div>'
        '<div class="lcd-nav">'
        f'<span class="lcd-nav-item">{LCD_ICON_FOLDER} Projects</span>'
        f'<span class="lcd-nav-item">{LCD_ICON_PLAY} Tutorials</span>'
        '<span class="lcd-nav-outline">PID</span>'
        '<span class="lcd-nav-pill">MPC</span>'
        '<span class="lcd-avatar">MPC</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_card_selector(options: list, key: str, default_value=None, columns_per_row: Optional[int] = None):
    """A row of clickable cards standing in for a dropdown -- each option is
    a dict with 'value', 'icon' (a short text/emoji glyph), 'title', and
    'desc'. Manages its own persisted selection in
    st.session_state[f"_card_selector_{key}"] and returns the currently
    selected value, same calling convention as st.selectbox.

    The active card is marked via the button's own type=primary/secondary
    (styled in the main CSS block) rather than a wrapping div + nth-child
    selector -- Streamlit doesn't actually nest widgets created after a raw
    st.markdown() injection as children of that markdown's HTML, so a
    parent-class approach couldn't reliably target the right card; styling
    the specific button's own variant attribute can, regardless of the
    surrounding DOM structure.
    """
    state_key = f"_card_selector_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value if default_value is not None else options[0]["value"]

    cols = st.columns(columns_per_row or len(options))
    for i, (col, opt) in enumerate(zip(cols, options)):
        with col:
            is_active = st.session_state[state_key] == opt["value"]
            label = f"{opt.get('icon', '')}  {opt['title']}\n{opt.get('desc', '')}".strip()
            if st.button(label, key=f"{key}_card_{i}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                if st.session_state[state_key] != opt["value"]:
                    st.session_state[state_key] = opt["value"]
                    st.rerun()
    return st.session_state[state_key]


def render_lcd_title_card(icon: str, title: str, subtitle: str):
    st.markdown(
        f'<div class="lcd-title-card"><div class="lcd-icon-square">{icon}</div>'
        f'<div><div class="lcd-h">{title}</div><div class="lcd-sub">{subtitle}</div></div></div>',
        unsafe_allow_html=True,
    )


LCD_ICON_FOLDER = ('<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                    '<path d="M3 7h5l2 2h11v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7z"/></svg>')
LCD_ICON_PLAY = '<svg viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="6,4 20,12 6,20"/></svg>'
LCD_ICON_STOP_SM = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
                     'style="width:0.85em;height:0.85em;vertical-align:-0.1em;"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>')
LCD_ICON_CHECK_SM = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" '
                      'stroke-linejoin="round" style="width:0.85em;height:0.85em;vertical-align:-0.1em;"><path d="M5 13l4 4L19 7"/></svg>')
LCD_ICON_WARN_SM = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
                     'stroke-linejoin="round" style="width:0.85em;height:0.85em;vertical-align:-0.1em;">'
                     '<path d="M12 3L22 20H2Z"/><line x1="12" y1="9" x2="12" y2="14"/><circle cx="12" cy="17" r="0.8" fill="currentColor"/></svg>')
LCD_ICON_WRENCH_SM = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
                       'style="width:0.9em;height:0.9em;vertical-align:-0.12em;"><line x1="4" y1="6" x2="20" y2="6"/>'
                       '<circle cx="9" cy="6" r="2" fill="currentColor"/><line x1="4" y1="12" x2="20" y2="12"/>'
                       '<circle cx="15" cy="12" r="2" fill="currentColor"/></svg>')

LCD_ICON_LOGO = ('<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
                  '<circle cx="6" cy="6" r="2.3"/><circle cx="18" cy="6" r="2.3"/><circle cx="12" cy="18" r="2.3"/>'
                  '<line x1="7.8" y1="7.3" x2="10.5" y2="16"/><line x1="16.2" y1="7.3" x2="13.5" y2="16"/>'
                  '<line x1="8.3" y1="6" x2="15.7" y2="6"/></svg>')

LCD_ICON_UPLOAD = ('<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                    '<path d="M12 3v12"/><path d="M7 8l5-5 5 5"/>'
                    '<path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>')
LCD_ICON_FLASK = ('<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                   '<path d="M9 3h6"/><path d="M10 3v6.5L4.5 19a1 1 0 0 0 .87 1.5h13.26a1 1 0 0 0 .87-1.5L14 9.5V3"/>'
                   '<path d="M6.5 14.5h11"/></svg>')
LCD_ICON_SLIDERS = ('<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                     '<line x1="4" y1="6" x2="20" y2="6"/><circle cx="9" cy="6" r="2" fill="currentColor"/>'
                     '<line x1="4" y1="12" x2="20" y2="12"/><circle cx="15" cy="12" r="2" fill="currentColor"/>'
                     '<line x1="4" y1="18" x2="20" y2="18"/><circle cx="7" cy="18" r="2" fill="currentColor"/></svg>')
LCD_ICON_ROCKET = ('<svg viewBox="0 0 24 24" fill="none" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
                    '<path d="M12 2 L15.5 9 L15.5 15.5 L8.5 15.5 L8.5 9 Z"/>'
                    '<circle cx="12" cy="9.5" r="1.6" fill="currentColor"/>'
                    '<path d="M8.5 13.5 L5.5 19 L8.5 17"/><path d="M15.5 13.5 L18.5 19 L15.5 17"/>'
                    '<path d="M10 15.5 L9.3 20 L12 18 L14.7 20 L14 15.5"/></svg>')


def render_advisory_chat():
    """Human-in-the-loop advisory chat (item 2): shown once, right after a
    dynamics file loads and before MPC configuration. Fully optional --
    "Skip, go straight to Setup" is always right there -- this is meant to
    help someone who wants to think out loud about their system, not to
    gate progress behind a conversation nobody asked for.
    """
    render_lcd_title_card(
        LCD_ICON_FLASK, "Let's talk about your system",
        "Ask about control strategy, tuning concerns, or anything else about what you uploaded -- "
        "optional, continue to Setup whenever you're ready.",
    )

    if not ADVISORY_CHAT_AVAILABLE:
        st.warning(f"Advisory chat unavailable: {ADVISORY_CHAT_ERROR}")
        if st.button("Continue to MPC Setup \u2192", type="primary", key="advisory_skip_unavailable"):
            st.session_state.advisory_chat_done = True
            st.rerun()
        return
    if not LLM_READY:
        st.warning(f"Advisory chat needs the LLM to be configured -- {LLM_INIT_ERROR}")
        if st.button("Continue to MPC Setup \u2192", type="primary", key="advisory_skip_no_llm"):
            st.session_state.advisory_chat_done = True
            st.rerun()
        return

    for turn in st.session_state.advisory_chat_history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    user_message = st.chat_input("Ask about control strategy, tuning, or anything else...")
    if user_message:
        st.session_state.advisory_chat_history.append({"role": "user", "content": user_message})
        try:
            with st.spinner("Thinking..."):
                reply = advisory_chat(
                    user_message, st.session_state.advisory_chat_history[:-1],
                    st.session_state.dynamics_summary,
                    setup_notes=st.session_state.get("setup_notes"),
                    derivative_pairs=st.session_state.get("derivative_pairs"),
                    tracker=st.session_state.get("token_tracker"),
                )
        except Exception as e:  # noqa: BLE001
            reply = f"(The advisory chat hit an error and couldn't respond: {e})"
        st.session_state.advisory_chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    st.divider()
    _adv_col1, _adv_col2 = st.columns([1, 4])
    with _adv_col1:
        _continue_label = "Continue to MPC Setup \u2192" if st.session_state.advisory_chat_history else "Skip, go straight to Setup \u2192"
        if st.button(_continue_label, type="primary", key="advisory_continue"):
            st.session_state.advisory_chat_done = True
            st.rerun()


def render_lcd_stepper(steps: list, current_index: int):
    """steps: list of (icon_svg, label, sublabel) tuples -- use the LCD_ICON_*
    constants above, never emoji (kept as plain line-art SVGs colored via
    currentColor, so CSS alone drives gray-inactive / blue-active). Each
    step draws its own connector segment (a ::after pseudo-element, see the
    CSS) running only from its own circle's edge to the next circle's edge
    -- so the line visibly starts and stops between icons, never crossing
    through one. Coloring is driven purely by the "done" CSS class (a
    segment is blue exactly when the step before it is done), no manual
    percentage math needed here.
    """
    cells = []
    for i, (icon, label, sub) in enumerate(steps):
        cls = "done" if i < current_index else ("active" if i == current_index else "")
        cells.append(
            f'<div class="lcd-step {cls}"><div class="lcd-step-circle">{icon}</div>'
            f'<div class="lcd-step-label">{label}</div><div class="lcd-step-sub">{sub}</div></div>'
        )
    html = '<div class="lcd-stepper">' + "".join(cells) + '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# LLM SETUP -- via labcd_agents.LLMFactory (see /packages/labcd_agents/src/labcd_agents), which
# supports OpenAI, Groq, Cerebras, NVIDIA NIM, and Anthropic from a single
# model name, auto-detecting the provider and reading the matching API key
# env var (OPENAI_API_KEY / GROQ_API_KEY / CEREBRAS_API_KEY / NVIDIA_API_KEY
# / ANTHROPIC_API_KEY). Add your key(s) to a .env file next to app.py.
# ============================================================================

try:
    from labcd_agents import ensure_env_loaded as _ensure_env_loaded_early
    _ensure_env_loaded_early()
except ImportError:
    pass

# DEFAULT_LLM_MODEL is read from .env (DEFAULT_LLM_MODEL=...), NOT
# hardcoded -- Streamlit's session_state is per-BROWSER-SESSION, not
# persisted across the app process restarting, so a hardcoded default here
# would silently win again every single time the app is relaunched,
# regardless of which provider was actually picked in the UI last time.
# Reading it from .env means picking a default ONCE actually sticks.
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "openai/gpt-oss-120b")

# Common presets shown in the model selector (see render_status_bar) --
# purely a convenience shortlist; any model labcd_agents' LLMFactory
# recognizes can still be typed in via the "Other" option regardless of
# whether it's listed here.
LLM_MODEL_PRESETS = [
    "openai/gpt-oss-120b", "openai/gpt-oss-20b",                  # Groq (llama-3.3-70b-versatile was retired Aug 16, 2026)
    "gpt-4o-mini", "gpt-4o", "gpt-5.4-mini",                     # OpenAI
    "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",   # Anthropic
    "llama-3.3-70b",                                              # Cerebras
]


@st.cache_resource
def _init_llm(model_name: str):
    try:
        from labcd_agents import LLMFactory, ensure_env_loaded, get_api_key
    except ImportError as e:
        return False, model_name, None, (
            f"labcd_agents package not found ({e}) -- see /packages/labcd_agents/src/labcd_agents "
            f"in the project root (install it via: pip install -e 'packages/labcd_agents[all]')."
        )

    ensure_env_loaded()
    provider = LLMFactory.resolve_provider(model_name)
    if provider is None:
        return False, model_name, None, (
            f"Unrecognized model '{model_name}' -- not matched to any known provider "
            f"(openai/groq/cerebras/nvidia/anthropic). Check the spelling, or see "
            f"packages/labcd_agents/src/labcd_agents/providers.py for the exact model lists each provider matches."
        )
    if not get_api_key(provider):
        env_var = {"openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY", "cerebras": "CEREBRAS_API_KEY",
                   "nvidia": "NVIDIA_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider, f"{provider.upper()}_API_KEY")
        return False, model_name, provider, f"{env_var} not found. Add it to a .env file next to app.py and restart."

    try:
        llm_instance = LLMFactory.create(model_name, temperature=0.3, seed=42, max_retries=2)
    except Exception as e:  # noqa: BLE001
        return False, model_name, provider, f"Failed to initialize {provider} client for '{model_name}': {e}"

    configure_llm(lambda: llm_instance)
    return True, model_name, provider, None


st.session_state.setdefault("selected_llm_model", DEFAULT_LLM_MODEL)
LLM_READY, LLM_MODEL, LLM_PROVIDER, LLM_INIT_ERROR = _init_llm(st.session_state.selected_llm_model)

# ============================================================================
# SESSION STATE DEFAULTS
# ============================================================================

_DEFAULTS = {
    "dynamics_loaded": False, "running": False,
    "results_data": [], "logs": [], "reasoning_entries": [], "_last_history_len": 0,
    "latest_params": {}, "best_row": None, "run_started_at": None, "run_finished_at": None,
    "test_result": None, "selected_sim_iteration": None, "last_raw_evaluator_update": None,
    "manual_sim_result": None,
    "derivative_pairs": [], "suggested_dt": None, "suggested_Q": None, "suggested_R": None, "setup_notes": [],
    "qr_diagnostics": None, "setup_panel_seen": False,
    "last_outputs": {}, "stop_requested": False, "graph_iterator": None, "stopped_by_user": False, "run_error": None,
    "report_pdf_bytes": None, "report_pdf_name": None, "run_perturbed_params": {},
    "upload_stage": None, "upload_review_code": "", "upload_review_filename": "", "upload_fix_result": None,
    "dynamics_source_code": None, "export_script_text": None, "export_script_name": None,
    "animation_gif_bytes": None, "animation_gif_name": None, "animation_description": None,
    "animation_render_note": None, "open_loop_result": None, "manual_best_iteration": None,
    "diagnostics_report": None, "token_tracker": None,
    "advisory_chat_done": False, "advisory_chat_history": [],
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================================
# DYNAMICS LOADING
# ============================================================================

def validate_or_fix_dynamics_code(source_code: str) -> dict:
    """Phase 1 of the upload review flow (item 5): validates the given
    source code, auto-fixing it with the LLM if needed. Does NOT
    instantiate anything or run the setup analyses -- purely a dry-run
    check, safe to call repeatedly as the user edits.

    Returns a dict: {"valid": bool, "was_fixed": bool, "final_code": str,
    "explanation": Optional[str], "error": Optional[str]}.
    """
    from backend_core.AgentMPC.agents.dynamics_validator import validate_and_fix_dynamics, validate_dynamics_source

    outcome = validate_dynamics_source(source_code)
    if outcome.valid:
        return {"valid": True, "was_fixed": False, "final_code": source_code, "explanation": None, "error": None}

    if not LLM_READY:
        return {"valid": False, "was_fixed": False, "final_code": source_code, "explanation": None,
                "error": f"{outcome.error} (auto-fix needs the LLM to be configured -- see GROQ_API_KEY)"}

    with st.spinner("Setup Agent checking your code against the standard..."):
        fix_result = validate_and_fix_dynamics(source_code, max_attempts=2)

    if not fix_result.valid:
        return {"valid": False, "was_fixed": True, "final_code": fix_result.final_code, "explanation": None,
                "error": f"Could not automatically fix this after {fix_result.attempts} attempt(s). "
                         f"Original error: {fix_result.original_error}. Still failing with: {fix_result.still_broken_error}"}

    return {"valid": True, "was_fixed": True, "final_code": fix_result.final_code,
            "explanation": fix_result.explanation, "error": None}


def finalize_dynamics_load(source_code: str, file_name: str) -> bool:
    """Phase 2 of the upload review flow (item 5): the given source code is
    assumed ALREADY validated (see validate_or_fix_dynamics_code) -- this
    does the actual instantiation and runs the deterministic setup
    analyses (derivative pairs, initial Q/R, dt, feedforward trim), same as
    the second half of the original single-shot load_dynamics_from_file.
    """
    import tempfile

    from backend_core.AgentMPC.agents.dynamics_validator import (
        detect_derivative_pairs, estimate_dt, estimate_feedforward_trim, estimate_initial_qr,
    )

    st.session_state.dynamics_source_code = source_code
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(source_code)
            temp_path = f.name

        plugin = DynamicLoader.load_from_path(temp_path)
        dyn = plugin.create_dynamics()

        setup_notes = []
        pairs = []
        qr_diagnostics = None
        try:
            with st.spinner("Analyzing dynamics (derivative pairs, initial Q/R, dt)..."):
                pairs = detect_derivative_pairs(dyn)
                dyn.config.derivative_pairs = pairs or None
                suggested_dt = estimate_dt(dyn)
                suggested_Q, suggested_R, qr_note, qr_diagnostics = estimate_initial_qr(dyn)
                if pairs:
                    names = dyn.state_names
                    setup_notes.append("Derivative pairs detected: " + ", ".join(f"{names[j]} = d({names[i]})/dt" for i, j in pairs))
                setup_notes.append(f"Suggested dt_mpc \u2248 {suggested_dt:.4g}s (from linearized/step-response analysis).")
                setup_notes.append(qr_note)
        except Exception as e:  # noqa: BLE001
            log.warning("Initial setup analysis failed, falling back to flat defaults: %s", e)
            suggested_dt, suggested_Q, suggested_R = None, None, None
            setup_notes.append(f"Initial setup analysis skipped ({e}); using flat defaults.")

        suggested_feedforward = None
        try:
            suggested_feedforward, ff_note = estimate_feedforward_trim(dyn)
            setup_notes.append(ff_note)
        except Exception as e:  # noqa: BLE001
            log.warning("Feedforward trim estimation failed: %s", e)
            setup_notes.append(f"Feedforward trim estimation skipped ({e}).")

        st.session_state.plugin = plugin
        st.session_state.dyn = dyn
        st.session_state.dynamics_loaded = True
        st.session_state.dynamics_file = file_name
        st.session_state.dynamics_summary = plugin.summary()
        st.session_state.test_result = None
        st.session_state.derivative_pairs = pairs
        st.session_state.suggested_dt = suggested_dt
        st.session_state.suggested_Q = suggested_Q
        st.session_state.suggested_R = suggested_R
        st.session_state.suggested_feedforward = suggested_feedforward
        st.session_state.setup_notes = setup_notes
        st.session_state.qr_diagnostics = qr_diagnostics
        try:
            st.session_state._suggested_noise_std = suggested_noise_std(dyn)
        except Exception:  # noqa: BLE001
            st.session_state._suggested_noise_std = 0.01
        st.session_state.setup_panel_seen = False  # show the setup-agent panel expanded once for this new upload
        st.session_state.advisory_chat_done = False
        st.session_state.advisory_chat_history = []
        return True

    except DynamicsPluginError as e:
        st.error(f"Plugin validation failed: {e}")
        return False
    except Exception as e:  # noqa: BLE001
        st.error(f"Error: {e}")
        return False
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def load_dynamics_from_file(uploaded_file) -> bool:
    """The ORIGINAL one-shot path (validate/fix + load, no review step) --
    kept for the main dropzone's quick "Load Dynamics" button, for anyone
    who doesn't need to look at/edit the code first. Just chains the two
    phase functions above with their own default UI messaging."""
    source_code = uploaded_file.getvalue().decode("utf-8")
    result = validate_or_fix_dynamics_code(source_code)
    st.session_state.fixed_dynamics_code = result["final_code"] if result["was_fixed"] else None
    st.session_state.fixed_dynamics_explanation = result["explanation"]

    if not result["valid"]:
        st.error(f"Plugin validation failed: {result['error']}")
        if result["was_fixed"]:
            with st.expander("Last attempted fix (didn't pass validation -- shown for reference)"):
                st.code(result["final_code"], language="python")
        return False

    if result["was_fixed"]:
        st.success("Fixed automatically.")
        st.info(result["explanation"])

    return finalize_dynamics_load(result["final_code"], uploaded_file.name)

def run_dynamics_test():
    """Quick pre-flight check: one short closed-loop simulation with default
    MPC parameters, so a broken plugin is caught in one click instead of
    after burning through a full tuning run."""
    dyn, summary = st.session_state.dyn, st.session_state.dynamics_summary
    cfg = Config()
    cfg.mpc.prediction_horizon = 8
    cfg.mpc.control_horizon = 4
    cfg.data.dt_mpc = st.session_state.get("suggested_dt") or 0.02
    cfg.data.simulation_time = 1.0
    n_states, n_inputs = summary["n_states"], summary["n_inputs"]
    params = {"Np": 8, "Nc": 4, "Q": [1.0] * n_states, "R": [0.1] * n_inputs, "P": [1.0] * n_states}

    result = run_closed_loop(dyn, cfg, params, max_steps=40)
    if "error" in result:
        st.session_state.test_result = {"ok": False, "error": result["error"], "traceback": result.get("traceback")}
    else:
        m = result["metrics"]
        st.session_state.test_result = {
            "ok": True, "steps": result["steps"], "mse": m.mse,
            "avg_solve_time": result["avg_solve_time"],
        }


def format_weight(w):
    if w is None:
        return "[]"
    if isinstance(w, (list, tuple, np.ndarray)):
        return "[" + ", ".join(f"{fmt_num(x)}" for x in np.asarray(w).flatten()) + "]"
    return str(w)


def prefill_manual_sim_from_params(params):
    """Carries the last agent-proposed Np/Nc/Q/R/P over into Manual
    Simulation's own widgets -- used when a run is stopped (or finishes) so
    the user can drop into manual fine-tuning from exactly where the Agents
    left off, without retyping the numbers by hand. Must be called BEFORE
    render_manual_simulation_tab() runs in the current script execution (its
    widgets already have a `key=` set, and Streamlit forbids writing to a
    widget's session_state key after that widget has been instantiated in
    the same run) -- in practice this means: set these, then st.rerun()."""
    if not params:
        return
    q, r, p = params.get("Q") or [], params.get("R") or [], params.get("P") or params.get("Q") or []
    st.session_state["manual_np"] = int(params.get("Np", 12))
    st.session_state["manual_nc"] = int(params.get("Nc", 5))
    st.session_state["manual_q"] = ", ".join(f"{v:.4g}" for v in q)
    st.session_state["manual_r"] = ", ".join(f"{v:.4g}" for v in r)
    st.session_state["manual_p"] = ", ".join(f"{v:.4g}" for v in p)
    st.session_state["manual_use_init_state"] = False


# ============================================================================
# PLOTTING
# ============================================================================

def _style_ax(ax):
    ax.grid(True, alpha=0.05, color="#2d3349")
    ax.tick_params(colors="#3a4a6a")
    ax.set_facecolor("#0a0e1a")
    for spine in ax.spines.values():
        spine.set_color("#1a2a3a")


def plot_step_response_probe(diagnostics: dict, state_names: list):
    """Visualizes the open-loop step-response probe the Initial Setup Agent
    used to derive Q via Bryson's rule (see agents/dynamics_validator.py:
    estimate_initial_qr) -- every state's trajectory during the probe, with
    the observed range (max-min, i.e. exactly what 1/range^2 uses) shaded in."""
    traj = diagnostics["trajectory"]
    probe_dt = diagnostics["probe_dt"]
    ranges = diagnostics["ranges"]
    T = traj.shape[0]
    t = np.arange(T) * probe_dt
    n_states = traj.shape[1]

    palette = ["#4d9fff", "#f59e0b", "#34d399", "#f472b6", "#a78bfa", "#22d3ee", "#fb7185", "#60a5fa",
               "#818cf8", "#38bdf8", "#fbbf24", "#a3e635"]

    ncols = 3
    nrows = int(np.ceil(n_states / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.3 * nrows), squeeze=False)
    fig.patch.set_facecolor("#0a0e1a")

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if i >= n_states:
            ax.axis("off")
            continue
        color = palette[i % len(palette)]
        y = traj[:, i]
        ax.plot(t, y, color=color, linewidth=1.8)
        ax.axhspan(y.min(), y.max(), color=color, alpha=0.08)
        name = state_names[i] if i < len(state_names) else f"x{i}"
        ax.set_title(f"{name}  (range={ranges[i]:.3g})", color="#8a9aba", fontsize=9)
        ax.tick_params(labelsize=7)
        _style_ax(ax)

    plt.tight_layout()
    return fig


def plot_convergence(df: pd.DataFrame):
    ok_df = df[df["ok"]]
    if ok_df.empty:
        return None

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.patch.set_facecolor("#0a0e1a")

    # Overshoot is only meaningful for regulation (fixed-target) runs -- for
    # tracking (sin/cos/pulse) runs it's not computed at all (see
    # agents/metrics.py), so plotting it for those rows would just draw a
    # flat, meaningless line at 0. Filter to regulation rows only, same as
    # "settling" already filters out the "never settled" placeholder value.
    if "overshoot_meaningful" in ok_df.columns:
        ok_df_reg = ok_df[ok_df["overshoot_meaningful"] == True]  # noqa: E712 -- explicit for clarity with pandas
    else:
        ok_df_reg = ok_df

    panels = [
        (axes[0, 0], ok_df, "mse", "o-", "#4d9fff", "MSE", "Mean Squared Error", True),
        (axes[0, 1], ok_df_reg, "overshoot", "s-", "#f59e0b", "Overshoot", "Overshoot", True),
        (axes[0, 2], ok_df, "settling", "d-", "#a78bfa", "Time (s)", "Settling Time", False),
        (axes[1, 0], ok_df, "effort", "^-", "#34d399", "Effort", "Control Effort", True),
        (axes[1, 1], ok_df, "iae", "o-", "#22d3ee", "IAE", "Integral Absolute Error", True),
        (axes[1, 2], ok_df, "ise", "s-", "#fb7185", "ISE", "Integral Squared Error", True),
    ]
    for ax, data_src, col, marker, color, ylabel, title, ylim0 in panels:
        data = data_src[data_src[col] < float("inf")] if col == "settling" else data_src
        if not data.empty:
            ax.plot(data["iteration"], data[col], marker, color=color, linewidth=2, markersize=7,
                    markerfacecolor="#0a0e1a", markeredgewidth=1.5)
        else:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center", color="#3a4a6a", fontsize=11)
        ax.set_xlabel("Iteration", color="#4a5a7a"); ax.set_ylabel(ylabel, color="#4a5a7a")
        ax.set_title(title, color="#8a9aba", fontsize=10)
        if ylim0:
            ax.set_ylim(bottom=0)
        _style_ax(ax)

    # dt_mpc -- a STEP plot (not a smooth line) since dt only ever changes
    # in discrete jumps, exclusively when the Juror decides to tune it (see
    # agents/juror.py); flat for most of the run by design (item 7's
    # "don't touch dt for the first several iterations" requirement).
    ax_dt = axes[2, 0]
    dt_data = ok_df[ok_df["dt_mpc"].notna()] if "dt_mpc" in ok_df.columns else ok_df.iloc[0:0]
    if not dt_data.empty:
        ax_dt.step(dt_data["iteration"], dt_data["dt_mpc"], where="post", color="#f472b6", linewidth=2)
        ax_dt.plot(dt_data["iteration"], dt_data["dt_mpc"], "o", color="#f472b6", markersize=5,
                   markerfacecolor="#0a0e1a", markeredgewidth=1.5)
        # mark the exact iteration(s) where dt actually changed
        changed = dt_data[dt_data["dt_mpc"].diff().fillna(0) != 0]
        for _, row in changed.iloc[1:].iterrows():
            ax_dt.axvline(row["iteration"], color="#f472b6", linestyle=":", linewidth=1, alpha=0.4)
    else:
        ax_dt.text(0.5, 0.5, "N/A", transform=ax_dt.transAxes, ha="center", va="center", color="#3a4a6a", fontsize=11)
    ax_dt.set_xlabel("Iteration", color="#4a5a7a"); ax_dt.set_ylabel("dt_mpc (s)", color="#4a5a7a")
    ax_dt.set_title("MPC Sample Time (dt) -- tuned only by the Juror", color="#8a9aba", fontsize=10)
    _style_ax(ax_dt)
    panels = panels + [(ax_dt, dt_data, "dt_mpc", "o-", "#f472b6", "dt_mpc (s)", "dt", False)]

    # failed iterations marked as vertical red dashed lines across all panels
    fail_iters = df[~df["ok"]]["iteration"].tolist()
    for ax, *_ in panels:
        for fi in fail_iters:
            ax.axvline(fi, color="#ff5555", linestyle=":", linewidth=1, alpha=0.5)

    # Same reasoning for the multi-line Q/R panels -- with several states/inputs
    # overlaid in ONE panel, they need to be distinguishable from EACH OTHER,
    # which an all-blue palette defeats the purpose of.
    q_colors = ["#4d9fff", "#f59e0b", "#34d399", "#f472b6", "#a78bfa", "#22d3ee"]
    ax = axes[2, 1]
    q_cols = sorted(c for c in ok_df.columns if c.startswith("q") and c[1:].isdigit())
    has_q = False
    for i, col in enumerate(q_cols):
        if ok_df[col].abs().sum() > 0:
            ax.plot(ok_df["iteration"], ok_df[col], "o-", color=q_colors[i % len(q_colors)], linewidth=1.5,
                    markersize=5, markerfacecolor="#0a0e1a", markeredgewidth=1.5, label=col.upper(), alpha=0.85)
            has_q = True
    if has_q:
        ax.legend(loc="best", facecolor="#0a0e1a", edgecolor="#1a2a3a", labelcolor="#6a7a9a", fontsize=8)
    ax.set_xlabel("Iteration", color="#4a5a7a"); ax.set_ylabel("Weight", color="#4a5a7a")
    ax.set_title("State Weights (Q)", color="#8a9aba", fontsize=10); _style_ax(ax)

    r_colors = ["#fb7185", "#a78bfa", "#22d3ee", "#f59e0b", "#34d399", "#4d9fff"]
    ax = axes[2, 2]
    r_cols = sorted(c for c in ok_df.columns if c.startswith("r") and c[1:].isdigit())
    has_r = False
    for i, col in enumerate(r_cols):
        if ok_df[col].abs().sum() > 0:
            ax.plot(ok_df["iteration"], ok_df[col], "s-", color=r_colors[i % len(r_colors)], linewidth=1.5,
                    markersize=5, markerfacecolor="#0a0e1a", markeredgewidth=1.5, label=col.upper(), alpha=0.85)
            has_r = True
    if has_r:
        ax.legend(loc="best", facecolor="#0a0e1a", edgecolor="#1a2a3a", labelcolor="#6a7a9a", fontsize=8)
    ax.set_xlabel("Iteration", color="#4a5a7a"); ax.set_ylabel("Weight", color="#4a5a7a")
    ax.set_title("Input Weights (R)", color="#8a9aba", fontsize=10); _style_ax(ax)

    plt.tight_layout()
    return fig


def plot_trajectory_preview(reference, times, state_names):
    """Shows what the reference trajectory looks like for the CURRENT
    Configure-section settings -- built-in (Regulation/Sinusoidal/Pulse,
    including any per-state mix) or a custom-uploaded one -- updating live
    as amplitude/frequency/etc change, so the user gets a visual feel for
    the target BEFORE spending a tuning run on it."""
    if reference is None or len(reference) == 0:
        return None
    n_states = reference.shape[1]
    ncols = min(3, n_states)
    nrows = int(np.ceil(n_states / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 2.3 * nrows), squeeze=False)
    fig.patch.set_facecolor("#0a0e1a")
    colors = ["#4d9fff", "#f1fa8c", "#60a5fa", "#818cf8", "#8be9fd", "#50fa7b", "#ff79c6", "#ffb86c"]

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axs[r][c]
        if i >= n_states:
            ax.axis("off")
            continue
        name = state_names[i] if i < len(state_names) else f"x{i}"
        ax.plot(times, reference[:, i], color=colors[i % len(colors)], linewidth=2)
        ax.fill_between(times, reference[:, i], alpha=0.08, color=colors[i % len(colors)])
        ax.set_title(name, color="#8a9aba", fontsize=10)
        ax.tick_params(labelsize=7)
        _style_ax(ax)

    plt.tight_layout()
    return fig


def render_trajectory_preview(dyn, selected_trajectory, per_state_trajectory_modes,
                               traj_amplitude, traj_frequency, traj_pulse_start, traj_pulse_end,
                               preview_duration: float = 8.0, preview_dt: float = 0.02):
    """Computes and renders the preview chart for the Configure section --
    reuses the EXACT same desired_trajectory()/custom-loader call the real
    run will make, just with a fixed short preview window, so what's shown
    here is genuinely what will be used (not an approximation of it)."""
    state_names = dyn.state_names
    try:
        if selected_trajectory == "custom":
            loader = st.session_state.get("custom_trajectory_loader")
            if loader is None:
                st.info("Upload a custom trajectory file above to preview it here.")
                return
            reference = loader.generate(preview_dt, preview_duration, dyn.n_states, state_names)
        else:
            reference = dyn.config.desired_trajectory(
                preview_dt, preview_duration,
                mode=selected_trajectory, amplitude=traj_amplitude, frequency=traj_frequency,
                pulse_start=traj_pulse_start, pulse_end=traj_pulse_end,
                per_state_modes=per_state_trajectory_modes,
            )
    except Exception as e:  # noqa: BLE001
        st.warning(f"Couldn't preview this trajectory: {e}")
        return

    n_steps = max(int(preview_duration / preview_dt), 1)
    times = np.linspace(0, preview_duration, n_steps)
    fig = plot_trajectory_preview(reference, times, state_names)
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)


def plot_simulation_results(sim_data, iteration, state_names, input_names, title: Optional[str] = None,
                             u_bounds: Optional[tuple] = None, x_bounds: Optional[tuple] = None):
    if sim_data is None:
        return None
    states, inputs, times, refs = sim_data["states"], sim_data["inputs"], sim_data["times"], sim_data["refs"]
    n_states, n_inputs = sim_data["n_states"], sim_data["n_inputs"]
    if states is None or len(states) == 0:
        return None

    n_plots = n_states + n_inputs
    fig, axs = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots + 2), sharex=True)
    if n_plots == 1:
        axs = [axs]
    fig.patch.set_facecolor("#0a0e1a")
    colors = ["#4d9fff", "#f1fa8c", "#60a5fa", "#818cf8", "#8be9fd", "#50fa7b"]

    for i in range(n_states):
        ax = axs[i]
        name = state_names[i] if i < len(state_names) else f"x{i}"
        ax.plot(times, states[:, i], color=colors[i % len(colors)], linewidth=2, label=name)
        if refs is not None and len(refs) > 0 and i < refs.shape[1]:
            ax.plot(times, refs[: len(states), i], "r--", linewidth=1.5, label=f"{name}_ref", alpha=0.7)
        _draw_bound_lines(ax, x_bounds, i)
        ax.set_ylabel(name, color="#8a9aba", fontsize=10)
        ax.legend(loc="best", facecolor="#0a0e1a", edgecolor="#1a2a3a", labelcolor="#6a7a9a", fontsize=8)
        _style_ax(ax)

    for j in range(n_inputs):
        ax = axs[n_states + j]
        name = input_names[j] if j < len(input_names) else f"u{j}"
        ax.plot(times[: len(inputs)], inputs[:, j], "g-", linewidth=2, label=name)
        _draw_bound_lines(ax, u_bounds, j)
        ax.set_ylabel(name, color="#8a9aba", fontsize=10)
        ax.legend(loc="best", facecolor="#0a0e1a", edgecolor="#1a2a3a", labelcolor="#6a7a9a", fontsize=8)
        ax.axhline(y=0, color="#1a2a3a", linestyle="-", linewidth=0.5)
        _style_ax(ax)

    axs[-1].set_xlabel("Time [s]", color="#4a5a7a", fontsize=12)
    fig.suptitle(title or f"Simulation Results - Iteration {iteration}", color="#8a9aba", fontsize=14)
    plt.tight_layout()
    return fig


def _draw_bound_lines(ax, bounds: Optional[tuple], idx: int):
    """Draws a horizontal reference line for a lower/upper bound, but ONLY
    when it's actually finite -- an unconstrained (+-inf) bound draws
    nothing, so charts for systems without constraints stay uncluttered."""
    if bounds is None:
        return
    lo, hi = bounds
    if lo is not None and idx < len(lo) and np.isfinite(lo[idx]):
        ax.axhline(y=lo[idx], color="#ff6b6b", linestyle=":", linewidth=1.3, alpha=0.8, label="bound")
    if hi is not None and idx < len(hi) and np.isfinite(hi[idx]):
        ax.axhline(y=hi[idx], color="#ff6b6b", linestyle=":", linewidth=1.3, alpha=0.8)


# ============================================================================
# RENDER HELPERS (all read from st.session_state -- safe to call any time)
# ============================================================================

def render_summary_cards():
    summary = st.session_state.get("dynamics_summary", {})
    df = pd.DataFrame(st.session_state.results_data)
    ok_df = df[df["ok"]] if not df.empty else df

    best = _best_row()
    best_mse = best["mse"] if best is not None else None
    n_ok = len(ok_df) if not df.empty else 0
    n_fail = len(df) - n_ok if not df.empty else 0
    n_unstable = int(df["unstable"].sum()) if not df.empty and "unstable" in df.columns else 0

    if st.session_state.running:
        status, status_class = "Running...", "status-running"
    elif n_fail > 0 and n_ok == 0:
        status, status_class = "Failed", "status-failed"
    elif st.session_state.results_data:
        status, status_class = "Done", "status-ready"
    else:
        status, status_class = "Ready", "status-ready"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="label">System</div>'
                     f'<div class="value">{summary.get("dynamics_class","--")}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="label">Status</div>'
                     f'<div class="value {status_class}">{status}</div></div>', unsafe_allow_html=True)
    with c3:
        val = f"{fmt_num(best_mse)}" if best_mse is not None else "--"
        st.markdown(f'<div class="metric-card"><div class="label">Best MSE (stable)</div>'
                     f'<div class="value value-cyan">{val}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="label">Iterations</div>'
                     f'<div class="value value-purple">{n_ok} ok / {n_unstable} unstable / {n_fail} failed</div></div>', unsafe_allow_html=True)
    with c5:
        elapsed = "--"
        if st.session_state.run_started_at:
            end = st.session_state.run_finished_at or datetime.now()
            elapsed = f"{(end - st.session_state.run_started_at).total_seconds():.0f}s"
        st.markdown(f'<div class="metric-card"><div class="label">Elapsed</div>'
                     f'<div class="value value-yellow">{elapsed}</div></div>', unsafe_allow_html=True)

    tracker = st.session_state.get("token_tracker")
    if tracker is not None:
        usage = tracker.snapshot()
        if usage["call_count"] > 0:
            unparsed_note = f" ({usage['unparsed_calls']} call(s) had unrecognized usage data)" if usage["unparsed_calls"] else ""
            st.caption(
                f"\U0001f4b0 LLM usage so far: {usage['total_tokens']:,} tokens "
                f"({usage['prompt_tokens']:,} prompt + {usage['completion_tokens']:,} completion) "
                f"across {usage['call_count']} call(s){unparsed_note} -- full breakdown in Data & Export."
            )


METRIC_FORMULAS = {
    "MSE": (r"MSE = \frac{1}{N \cdot n} \sum_{t=1}^{N} \sum_{i=1}^{n} \left(x_i(t) - x_{i,\text{target}}(t)\right)^2",
            "Mean over every state and every timestep. Lower is better."),
    "Overshoot": (r"\text{Overshoot} = \frac{\max\left(|e(t)| \text{ after } e(t) \text{ crosses zero}\right)}{|e(0)|}",
                  "Fraction of the initial error swung past the target. Only defined per state where the "
                  "target is fixed and the initial error is nonzero."),
    "Settling": (r"t_s = \min\left\{\,t : |e(\tau)| \leq \varepsilon \text{ for all } \tau \geq t\,\right\}, "
                 r"\qquad \varepsilon = \alpha \cdot \|e(0)\|",
                 r"First time the error enters tolerance $\varepsilon$ (a fraction $\alpha$ of the initial "
                 r"error) and never leaves it again."),
    "Effort": (r"J_u = \frac{1}{N \cdot m} \sum_{t=1}^{N} \sum_{j=1}^{m} u_j(t)^2",
               "Mean squared control input across every actuator and timestep -- a proxy for actuator "
               "energy/wear, not a physical energy unit."),
    "Stable": (r"\text{Stable if: } \ \text{RMS}_{\text{late quarter}}(e) \leq 1.05 \times \text{RMS}_{\text{early quarter}}(e)",
               "Compares the error's RMS level late in the run against early in the run -- looser than "
               "Settling Time, which needs the tight tolerance band."),
    "Oscillations": (r"N_{\text{osc}} = \#\left\{\,t : \operatorname{sign}(e(t)) \neq \operatorname{sign}(e(t-1))\,\right\}",
                      "Sign changes in the worst-off state's tracking error -- how many times it crossed "
                      "back and forth over the target."),
    "IAE": (r"IAE_i = \sum_{t} |e_i(t)| \cdot \Delta t", "Integral Absolute Error, per state, over the whole run."),
    "ISE": (r"ISE_i = \sum_{t} e_i(t)^2 \cdot \Delta t", "Integral Squared Error, per state, over the whole run."),
}


def render_metric_formulas(keys: list, panel_name: str = ""):
    """Collapsed-by-default expander with the REAL LaTeX formula (via
    st.latex, not plain text with unicode math symbols) for each metric key
    requested -- only rendered when the user actually opens it, not
    displayed unconditionally alongside every metric card. ``panel_name``
    makes the label text unique across the different places this gets
    called from within the same script run (Live Run panel, Best Result
    panel, ...) -- simpler and safer than relying on st.expander's key=
    parameter, whose exact minimum-version support isn't worth gambling on.
    """
    label = f"Show formulas ({panel_name})" if panel_name else "Show formulas"
    with st.expander(label, expanded=False):
        for key in keys:
            if key not in METRIC_FORMULAS:
                continue
            formula, note = METRIC_FORMULAS[key]
            st.markdown(f"**{key}**")
            st.latex(formula)
            st.caption(note)


def _normalized_effort(effort_mean_sq: Optional[float], u_bounds: Optional[tuple],
                        history_max: Optional[float] = None) -> Optional[float]:
    """Returns a ~0-1 scale value: RMS(u) as a fraction of the actuator's
    own bound magnitude when constraints are set (physically meaningful --
    "using 90% of available actuator authority"), or effort relative to the
    worst seen so far in this run otherwise (still 0-1, just relative
    instead of absolute). None if neither reference is available -- callers
    should fall back to showing the raw number only in that case."""
    if effort_mean_sq is None:
        return None
    rms = effort_mean_sq ** 0.5
    if u_bounds is not None:
        lo, hi = u_bounds
        finite_bounds = [abs(b) for b in (list(lo) + list(hi)) if np.isfinite(b)]
        if finite_bounds:
            avg_bound = float(np.mean(finite_bounds))
            if avg_bound > 0:
                return rms / avg_bound
    if history_max and history_max > 0:
        return float(effort_mean_sq / history_max)
    return None


def render_metrics_cards():
    if not st.session_state.results_data:
        return
    last = st.session_state.results_data[-1]
    if not last["ok"]:
        return
    improvement = last.get("improvement") or {}
    st.markdown('<div class="subheader">Current Performance (vs. best so far)</div>', unsafe_allow_html=True)
    overshoot_ok = last.get("overshoot_meaningful", True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("MSE", f"{fmt_num(last['mse'])}", delta=f"{improvement.get('MSE', 0.0):.1f}%",
                  help="Mean Squared Error across every state and timestep. Lower is better.")
    with c2:
        if overshoot_ok:
            st.metric("Overshoot", f"{fmt_num(last['overshoot'])}", delta=f"{improvement.get('Overshoot', 0.0):.1f}%",
                      help="How far past the target the response swings, as a fraction of the initial error.")
        else:
            st.metric("Overshoot", "N/A", help="Not computable: either every state started with zero initial error (nothing to overshoot from -- try Scenario Level 2/3 or a custom initial state), or every state with a nonzero initial error is tracking a moving reference. See IAE/ISE in the Convergence tab instead.")
    with c3:
        settling_str = f"{fmt_num(last['settling'])}s" if last["settling"] != float("inf") else "N/A"
        st.metric("Settling Time", settling_str, delta=f"{improvement.get('Settling_Time', 0.0):.1f}%",
                  help="First time the error enters tolerance and stays there.")
    with c4:
        _effort_history = [r["effort"] for r in st.session_state.results_data if r.get("ok") and r.get("effort") is not None]
        _effort_norm = _normalized_effort(last.get("effort"), st.session_state.get("run_u_bounds"),
                                            history_max=max(_effort_history) if _effort_history else None)
        if _effort_norm is not None:
            st.metric("Control Effort", f"{_effort_norm:.2f}", delta=f"{improvement.get('Control_Effort', 0.0):.1f}%",
                      help=f"Normalized 0-1ish: RMS control input as a fraction of the actuator's own bound "
                           f"(if set) or relative to the worst seen so far this run. Raw value (mean squared "
                           f"input, actual units): {fmt_num(last['effort'])}.")
        else:
            st.metric("Control Effort", f"{fmt_num(last['effort'])}", delta=f"{improvement.get('Control_Effort', 0.0):.1f}%",
                      help="Mean squared control input -- a proxy for actuator energy/wear. Set input bounds "
                           "in the Configure section for a more interpretable 0-1 normalized version.")
    render_metric_formulas(["MSE", "Overshoot", "Settling", "Effort"], panel_name="Live Run")

    solver_diag = last.get("solver_diagnostics") or {}
    if solver_diag:
        solved = solver_diag.get("solved", 0)
        inaccurate = solver_diag.get("solved_inaccurate", 0)
        other = solver_diag.get("other", 0)
        total = solved + inaccurate + other
        if total > 0:
            if other > 0:
                st.error(f"QP Solver: {solved}/{total} steps solved cleanly, {inaccurate} inaccurate, "
                         f"{other} infeasible/failed -- results may not be trustworthy.")
            elif inaccurate > 0:
                st.warning(f"QP Solver: {solved}/{total} steps solved cleanly, {inaccurate} solved "
                           f"inaccurately (still usable, just not to full tolerance).")
            else:
                st.caption(f"QP Solver: all {total} steps solved cleanly.")


def render_params_panel():
    if not st.session_state.results_data:
        return
    last = st.session_state.results_data[-1]
    tag = " (FAILED)" if not last["ok"] else (" (UNSTABLE)" if last.get("unstable") else "")
    with st.expander(f"Parameters - Iteration {last['iteration']}{tag}", expanded=False):
        if not last["ok"]:
            st.error(last.get("error", "Unknown error"))
            if last.get("traceback"):
                with st.expander("Technical details (traceback)"):
                    st.code(last["traceback"], language="python")
            return
        if last.get("unstable"):
            st.warning("Simulation diverged / became unstable and was stopped early -- metrics below "
                       "reflect only the trajectory up to that point.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**Prediction Horizon (Np):** {last['np']}")
            st.write(f"**Control Horizon (Nc):** {last['nc']}")
        with c2:
            st.write(f"**State Weights (Q):** {last['Q_formatted']}")
        with c3:
            st.write(f"**Input Weights (R):** {last['R_formatted']}")
            st.write(f"**Terminal Weights (P):** {last['P_formatted']}")
            st.write(f"**Strategy:** {str(last['strategy']).upper()}")

        per_state_mse = last.get("per_state_mse") or {}
        per_state_overshoot = last.get("per_state_overshoot") or {}
        if per_state_mse:
            st.markdown("**Per-state breakdown** (this is what the Actor/Critic agents actually see, "
                        "not just the aggregate MSE):")
            breakdown_df = pd.DataFrame({
                "State": list(per_state_mse.keys()),
                "MSE": [f"{fmt_num(v)}" for v in per_state_mse.values()],
                "Overshoot": [(f"{fmt_num(per_state_overshoot.get(k))}" if per_state_overshoot.get(k) is not None else "N/A") for k in per_state_mse.keys()],
            })
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)


def _reasoning_node_class(text: str) -> str:
    for name in ("Scenarist", "Actor", "Critic", "Terminator", "Juror", "Evaluator"):
        if text.startswith(f"[{name}]"):
            return f"r-node-{name.lower()}"
    return "r-node-actor"


def render_reasoning_panel():
    entries = st.session_state.reasoning_entries
    if not entries:
        st.info("No agent activity yet -- run a tuning session to see Actor/Critic/Terminator reasoning here.")
        return
    st.markdown('<div class="reasoning-container">', unsafe_allow_html=True)
    for entry in reversed(entries):
        text, time_str = entry["text"], entry["time"]
        node_label = text.split("]")[0].lstrip("[") if text.startswith("[") else "INFO"
        css_class = _reasoning_node_class(text)
        body = text.split("]", 1)[1].strip() if "]" in text else text
        st.markdown(f"""
        <div class="reasoning-entry">
            <div class="r-header"><span class="r-node {css_class}">{node_label}</span><span class="r-time">{time_str}</span></div>
            <div class="r-text">{body}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_log_panel():
    logs = st.session_state.logs
    if not logs:
        return
    st.markdown('<div class="log-container">', unsafe_allow_html=True)
    for entry in logs:
        node, message, time_str = entry["node"], entry["message"], entry["time"]
        cls = "log-metric" if "METRIC" in node else "log-error" if "ERROR" in node else "log-success" if "DONE" in node else ""
        st.markdown(f'<div class="log-entry"><span class="log-time">[{time_str}]</span> '
                     f'<span class="log-node">{node}:</span> <span class="{cls}">{message}</span></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _best_row():
    """Ranks by the composite scalar_cost the engine itself uses (see
    agents/metrics.py), among rows that actually ran successfully and didn't
    diverge -- not just the lowest raw MSE (see evaluator.py's docstring for
    why: a run that briefly dips to a low instantaneous error while still
    oscillating, or one that diverged right after a lucky first step,
    shouldn't be crowned "best"). A manual override (see the "Pick manually"
    control in render_best_result) takes priority over this automatic
    selection when set -- lets the user correct it for the cases where the
    automatic ranking doesn't match their own judgment of which iteration
    actually looks best."""
    manual_iter = st.session_state.get("manual_best_iteration")
    if manual_iter is not None:
        for r in st.session_state.results_data:
            if r.get("iteration") == manual_iter and r.get("ok"):
                return r
        # the manually-picked iteration no longer exists (e.g. a fresh run
        # started) -- fall through to automatic selection rather than
        # silently returning nothing.
    candidates = [r for r in st.session_state.results_data if r["ok"] and not r.get("unstable") and r.get("cost") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["cost"])


def render_best_result():
    is_manual = st.session_state.get("manual_best_iteration") is not None
    with st.expander("Pick manually" + (f" (currently: iteration {st.session_state.manual_best_iteration})" if is_manual else ""),
                      expanded=False):
        st.caption(
            "By default the \"best\" iteration is chosen automatically (see the ranking explained below) -- "
            "if you've looked through the results and think a different one actually looks best, pick it "
            "here instead. Applies everywhere \"best result\" is used: this tab, the PDF report, the "
            "exported script, and the animation."
        )
        _ok_iters = [r["iteration"] for r in st.session_state.results_data if r.get("ok")]
        if not _ok_iters:
            st.caption("No successful iterations yet to choose from.")
        else:
            _options = ["Automatic"] + _ok_iters
            _current = st.session_state.get("manual_best_iteration")
            _default_idx = _options.index(_current) if _current in _options else 0
            _choice = st.selectbox(
                "Which iteration is actually best?", options=_options, index=_default_idx,
                format_func=lambda x: "Automatic (default)" if x == "Automatic" else (
                    f"Iteration {x}  (MSE={fmt_num(next(r['mse'] for r in st.session_state.results_data if r['iteration']==x))}, "
                    f"stable={'Yes' if next(r.get('is_stable') for r in st.session_state.results_data if r['iteration']==x) else 'No'})"
                ),
                key="manual_best_selectbox",
            )
            new_value = None if _choice == "Automatic" else _choice
            if new_value != st.session_state.get("manual_best_iteration"):
                st.session_state.manual_best_iteration = new_value
                st.rerun()

    best = _best_row()
    if best is None:
        st.info("No stable, successful iteration yet -- the best result will appear here once one completes.")
        return
    if is_manual:
        st.info(f"\U0001f4cc Manually pinned to iteration {best['iteration']} (not the automatic choice). "
                f"Reset to \"Automatic\" above to go back to the default ranking.")

    st.markdown(f'<div class="subheader">Best Result -- Iteration {best["iteration"]}</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("MSE", f"{fmt_num(best['mse'])}", help="Mean Squared Error over every state and timestep.")
    with c2:
        st.metric("Overshoot", f"{fmt_num(best['overshoot'])}" if best.get("overshoot_meaningful", True) else "N/A",
                  help="Fraction of the initial error the response swings past the target by.")
    with c3:
        st.metric("Settling Time", f"{fmt_num(best['settling'])}s" if best["settling"] != float("inf") else "N/A",
                  help="First time the error enters tolerance and holds there.")
    with c4:
        st.metric("Stable", "Yes" if best.get("is_stable") else "No",
                  help="Whether the error's late-run RMS level held (didn't get meaningfully worse).")
    render_metric_formulas(["MSE", "Overshoot", "Settling", "Stable", "Oscillations"], panel_name="Best Result")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"**Np:** {best['np']}   **Nc:** {best['nc']}")
    with c2:
        st.write(f"**Q:** {best['Q_formatted']}")
        st.write(f"**R:** {best['R_formatted']}")
    with c3:
        st.write(f"**P:** {best['P_formatted']}")
        st.metric("Oscillations", best['oscillation_count'],
                  help="Number of times the worst-off state's tracking error changes sign (crosses zero) "
                       "over the run -- counts how many times it swung from one side of the target to the "
                       "other, a rough proxy for how 'ringy'/underdamped the response looks.")

    per_state_mse = best.get("per_state_mse") or {}
    if per_state_mse:
        with st.expander("Per-state breakdown", expanded=False):
            per_state_overshoot = best.get("per_state_overshoot") or {}
            breakdown_df = pd.DataFrame({
                "State": list(per_state_mse.keys()),
                "MSE": [f"{fmt_num(v)}" for v in per_state_mse.values()],
                "Overshoot": [(f"{fmt_num(per_state_overshoot.get(k))}" if per_state_overshoot.get(k) is not None else "N/A") for k in per_state_mse.keys()],
            })
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    if best.get("simulation_data") is not None:
        summary = st.session_state.get("dynamics_summary", {})
        fig = plot_simulation_results(best["simulation_data"], best["iteration"],
                                       summary.get("state_names", []), summary.get("input_names", []),
                                       u_bounds=st.session_state.get("run_u_bounds"),
                                       x_bounds=st.session_state.get("run_x_bounds"))
        if fig:
            st.pyplot(fig)
            plt.close(fig)


def render_open_loop_test():
    """Open Loop Test (added on request): simulates the RAW dynamics
    directly via SystemSimulator.simulate() -- no MPC controller, no
    feedback of any kind, just the plugin's own dynamics() evolving from a
    user-chosen initial state under a user-chosen (typically zero) input.
    Useful for exactly the kind of question that came up in conversation:
    "is this equilibrium actually stable on its own, before any controller
    gets involved?" -- collapsed by default since it's a diagnostic tool,
    not part of the normal tuning flow.
    """
    with st.expander("Open Loop Test", expanded=False):
        st.caption(
            "Simulates the dynamics directly -- no MPC, no feedback, nothing correcting anything. "
            "Set an initial state, optionally a constant input, and see how the system evolves on its "
            "own. Useful for sanity-checking the plugin itself, or seeing whether a target point is "
            "actually an equilibrium the system would stay near without a controller."
        )

        if not st.session_state.dynamics_loaded:
            st.info("Load a dynamics file first.")
            return

        summary = st.session_state.dynamics_summary
        n_states = summary.get("n_states", 4)
        n_inputs = summary.get("n_inputs", 1)
        state_names = summary.get("state_names", [f"x{i}" for i in range(n_states)])
        input_names = summary.get("input_names", [f"u{i}" for i in range(n_inputs)])
        dyn = st.session_state.dyn

        default_x0_hint = ", ".join(fmt_num(v) for v in dyn.config.default_initial_state)
        ol_init_text = st.text_input(
            f"Initial state -- {n_states} comma-separated values ({', '.join(state_names)})",
            value=default_x0_hint, key="ol_init_state_text",
        )
        ol_x0, ol_x0_error = None, None
        try:
            ol_x0 = np.array([float(v.strip()) for v in ol_init_text.split(",") if v.strip()])
            if ol_x0.size != n_states:
                ol_x0_error = f"Need exactly {n_states} value(s), got {ol_x0.size}."
                ol_x0 = None
        except ValueError:
            ol_x0_error = "Must be comma-separated numbers."
        if ol_x0_error:
            st.error(ol_x0_error)

        ol_use_input = st.checkbox(
            "Apply a constant nonzero input", value=False, key="ol_use_input",
            help="Off = pure zero-input open loop (the default and most common case -- 'if I let go, "
                 "what happens?'). On = hold a fixed, unchanging input throughout instead of zero.",
        )
        ol_u_const = np.zeros(n_inputs)
        ol_u_error = None
        if ol_use_input:
            ol_u_text = st.text_input(
                f"Constant input -- {n_inputs} comma-separated values ({', '.join(input_names)})",
                value=", ".join(["0.0"] * n_inputs), key="ol_input_text",
            )
            try:
                ol_u_const = np.array([float(v.strip()) for v in ol_u_text.split(",") if v.strip()])
                if ol_u_const.size != n_inputs:
                    ol_u_error = f"Need exactly {n_inputs} value(s), got {ol_u_const.size}."
            except ValueError:
                ol_u_error = "Must be comma-separated numbers."
            if ol_u_error:
                st.error(ol_u_error)

        c1, c2 = st.columns(2)
        with c1:
            ol_sim_time = st.number_input("Simulation Time (s)", min_value=0.1, max_value=600.0, value=5.0,
                                           step=0.5, key="ol_sim_time",
                                           help="Fully in your control -- no artificial cap beyond a generous ceiling.")
        with c2:
            ol_dt_default = float(st.session_state.get("suggested_dt") or 0.02)
            ol_dt = st.number_input("dt (s)", min_value=0.0005, max_value=1.0, value=ol_dt_default,
                                     step=0.001, format="%.4f", key="ol_dt")

        run_ol = st.button("Run Open Loop", type="primary", key="ol_run_button",
                            disabled=bool(ol_x0_error) or bool(ol_u_error))

        if run_ol and ol_x0 is not None:
            n_steps = max(int(ol_sim_time / ol_dt), 1)
            U = np.tile(ol_u_const, (n_steps, 1))
            simulator = SystemSimulator(dyn, dt=ol_dt)
            try:
                with st.spinner("Running open loop..."):
                    X, dX, U_used = simulator.simulate(ol_x0, U)
                times = np.arange(len(X)) * ol_dt
                target = dyn.config.default_target
                st.session_state.open_loop_result = {
                    "ok": True,
                    "terminated_early": len(X) < n_steps,
                    "final_state": X[-1].tolist(),
                    "sim_data": {
                        "states": X, "inputs": U_used,
                        "refs": np.tile(target, (len(X), 1)),
                        "times": times, "n_states": n_states, "n_inputs": n_inputs,
                    },
                }
            except Exception as e:  # noqa: BLE001
                st.session_state.open_loop_result = {"ok": False, "error": str(e), "traceback": tb_module.format_exc()}

        res = st.session_state.get("open_loop_result")
        if res is not None:
            st.divider()
            if not res["ok"]:
                st.error(f"Open loop simulation failed: {res['error']}")
                with st.expander("Traceback"):
                    st.code(res["traceback"], language="python")
            else:
                if res["terminated_early"]:
                    st.warning("Stopped early -- hit a declared state bound before the full simulation time "
                               "elapsed (a strong sign this starting point/input diverges rather than settles).")
                st.caption(f"Final state: {', '.join(f'{n}={fmt_num(v)}' for n, v in zip(state_names, res['final_state']))}")
                fig = plot_simulation_results(
                    res["sim_data"], iteration=None, state_names=state_names, input_names=input_names,
                    title="Open Loop Test (no controller, no feedback -- red dashed line is the target, for reference only)",
                    x_bounds=dyn.get_state_bounds(),
                )
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)


def render_manual_simulation_tab():
    st.markdown('<div class="subheader">Manual Simulation</div>', unsafe_allow_html=True)
    st.caption("Runs one closed-loop simulation with parameters you choose directly -- no Agents, "
               "no LLM calls, just the MPC controller. Useful for sanity-checking a parameter set "
               "by hand, independent of the tuning loop.")

    summary = st.session_state.get("dynamics_summary", {})
    n_states = summary.get("n_states", 4)
    n_inputs = summary.get("n_inputs", 1)
    dyn_for_defaults = st.session_state.get("dyn")

    c1, c2 = st.columns(2)
    with c1:
        m_np = st.number_input("Np (prediction horizon)", min_value=1, max_value=60,
                                value=int(st.session_state.get("manual_np", 12)), key="manual_np")
    with c2:
        m_nc = st.number_input("Nc (control horizon)", min_value=1, max_value=60,
                                value=int(st.session_state.get("manual_nc", 5)), key="manual_nc")

    m_q_text = st.text_input(f"Q -- {n_states} comma-separated values",
                              value=st.session_state.get("manual_q") or ", ".join(["1.0"] * n_states), key="manual_q")
    m_r_text = st.text_input(f"R -- {n_inputs} comma-separated values",
                              value=st.session_state.get("manual_r") or ", ".join(["0.1"] * n_inputs), key="manual_r")
    m_p_text = st.text_input(f"P (terminal weights) -- {n_states} values, or leave blank to reuse Q",
                              value=st.session_state.get("manual_p") or "", key="manual_p")

    c1, c2, c3 = st.columns(3)
    with c1:
        m_sim_time = st.slider("Simulation Time (s)", 2.0, 30.0, 10.0, step=0.5, key="manual_sim_time")
    with c2:
        m_trajectory = st.selectbox("Trajectory Type", options=["reg", "sin", "pulse"],
                                     format_func=lambda x: {"reg": "Regulation (Zero)", "sin": "Sinusoidal", "pulse": "Pulse"}[x],
                                     key="manual_trajectory")
    with c3:
        m_noise = st.slider("Measurement Noise (std)", 0.0, 0.5, 0.0, step=0.01, key="manual_noise")

    m_customize_per_state = st.checkbox(
        "Customize per state", value=False, key="manual_customize_per_state",
        help="Pick a different trajectory type per state (e.g. Sinusoidal for a position state, "
             "Cosinusoidal for its matching velocity state). Shares the Amplitude/Frequency below.",
    )
    m_per_state_modes = None
    if m_customize_per_state:
        m_per_state_options = {"reg": "Regulation", "sin": "Sinusoidal", "cos": "Cosinusoidal", "pulse": "Pulse"}
        state_names_list = summary.get("state_names", [f"x{i}" for i in range(n_states)])
        m_default_rows = pd.DataFrame({"State": state_names_list, "Trajectory": ["Regulation"] * len(state_names_list)})
        m_edited = st.data_editor(
            m_default_rows, hide_index=True, use_container_width=True, key="manual_per_state_traj_editor",
            column_config={
                "State": st.column_config.TextColumn(disabled=True),
                "Trajectory": st.column_config.SelectboxColumn(options=list(m_per_state_options.values())),
            },
        )
        m_label_to_code = {v: k for k, v in m_per_state_options.items()}
        m_per_state_modes = [m_label_to_code[v] for v in m_edited["Trajectory"]]

    m_amplitude, m_frequency, m_pulse_start, m_pulse_end = 0.5, 0.5, 0.2, 0.7
    m_show_sin = m_trajectory == "sin" or (m_per_state_modes and any(x in ("sin", "cos") for x in m_per_state_modes))
    m_show_pulse = m_trajectory == "pulse" or (m_per_state_modes and "pulse" in m_per_state_modes)
    if m_show_sin:
        c1, c2 = st.columns(2)
        with c1: m_amplitude = st.slider("Amplitude", 0.05, 3.0, 0.5, step=0.05, key="manual_sin_amplitude")
        with c2: m_frequency = st.slider("Frequency (Hz)", 0.05, 3.0, 0.5, step=0.05, key="manual_sin_frequency")
    if m_show_pulse:
        c1, c2, c3 = st.columns(3)
        with c1: m_amplitude = st.slider("Amplitude", 0.05, 3.0, 0.5, step=0.05, key="manual_pulse_amplitude")
        with c2: m_pulse_start = st.slider("Rise at (% of sim time)", 0, 90, 20, key="manual_pulse_start") / 100.0
        with c3: m_pulse_end = st.slider("Fall at (% of sim time)", 10, 100, 70, key="manual_pulse_end") / 100.0
        if m_pulse_end <= m_pulse_start:
            st.warning("Fall time should be after rise time.")

    m_dt_default = float(st.session_state.get("suggested_dt") or 0.02)
    m_dt = st.number_input("dt_mpc (s)", min_value=0.0005, max_value=1.0, value=m_dt_default, step=0.001, format="%.4f", key="manual_dt")

    use_manual_initial_state = st.checkbox("Set custom initial state", value=False, key="manual_use_init_state")
    manual_initial_state, manual_init_error = None, None
    if use_manual_initial_state:
        default_hint = (
            ", ".join(f"{fmt_num(v)}" for v in dyn_for_defaults.config.default_initial_state)
            if dyn_for_defaults is not None else ""
        )
        m_init_text = st.text_input(
            f"Initial state -- {n_states} comma-separated values", value=default_hint, key="manual_init_state_text",
        )
        try:
            manual_initial_state = np.array([float(v.strip()) for v in m_init_text.split(",") if v.strip()])
            if manual_initial_state.size != n_states:
                manual_init_error = f"Need exactly {n_states} value(s), got {manual_initial_state.size}."
                manual_initial_state = None
        except ValueError:
            manual_init_error = "Must be comma-separated numbers."
        if manual_init_error:
            st.error(manual_init_error)

    run_manual = st.button("Run Manual Simulation", type="primary", key="manual_run_button",
                            disabled=use_manual_initial_state and bool(manual_init_error))

    if run_manual:
        q_params, q_error = parse_seed_params(m_np, m_nc, m_q_text, m_r_text, n_states, n_inputs)
        if q_error:
            st.error(q_error)
        else:
            params = dict(q_params)
            if m_p_text.strip():
                p_vals, p_err = parse_seed_params(m_np, m_nc, m_p_text, m_r_text, n_states, n_inputs)
                if p_err:
                    st.error(f"P: {p_err}")
                    params = None
                else:
                    params["P"] = p_vals["Q"]

            if params is not None:
                dyn = st.session_state.dyn
                m_cfg = Config()
                m_cfg.mpc.prediction_horizon = m_np
                m_cfg.mpc.control_horizon = m_nc
                m_cfg.data.dt_mpc = m_dt
                m_cfg.data.simulation_time = m_sim_time
                m_cfg.data.trajectory_mode = m_trajectory
                m_cfg.data.trajectory_amplitude = m_amplitude
                m_cfg.data.trajectory_frequency = m_frequency
                m_cfg.data.trajectory_pulse_start = m_pulse_start
                m_cfg.data.trajectory_pulse_end = m_pulse_end
                m_cfg.data.trajectory_per_state_modes = m_per_state_modes
                m_cfg.data.noise_std = m_noise

                # Temporarily override the initial state for this one run only --
                # `dyn` is the same shared instance the main tuning run and the
                # sidebar's own "custom initial state" use, so we restore it
                # right after (success or failure) instead of leaving a manual-
                # simulation-only override leaking into the Agent tuning loop.
                original_initial_state = dyn.config.default_initial_state.copy()
                if use_manual_initial_state and manual_initial_state is not None:
                    dyn.config.default_initial_state = manual_initial_state.copy()

                try:
                    with st.spinner("Running..."):
                        result = run_closed_loop(dyn, m_cfg, params)
                finally:
                    dyn.config.default_initial_state = original_initial_state

                if "error" in result:
                    st.session_state.manual_sim_result = {"ok": False, "error": result["error"], "traceback": result.get("traceback")}
                else:
                    st.session_state.manual_sim_result = {
                        "ok": True, "unstable": result["unstable"], "unstable_reason": result.get("unstable_reason"),
                        "metrics": result["metrics"], "simulation_data": {
                            "states": result["states"], "inputs": result["inputs"],
                            "refs": result["reference"], "times": result["times"],
                            "n_states": n_states, "n_inputs": n_inputs,
                        },
                    }

    res = st.session_state.get("manual_sim_result")
    if res is not None:
        st.divider()
        if not res["ok"]:
            st.error(f"Simulation failed: {res['error']}")
            if res.get("traceback"):
                with st.expander("Traceback"):
                    st.code(res["traceback"], language="python")
        else:
            if res["unstable"]:
                st.warning(f"Unstable / diverged: {res['unstable_reason']}")
            m = res["metrics"]
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.metric("MSE", f"{fmt_num(m.mse)}")
            with c2:
                if m.is_regulation:
                    st.metric("Overshoot", f"{fmt_num(m.overshoot)}")
                else:
                    st.metric("Integral Sq. Error", f"{fmt_num(m.integral_sq_error)}")
            with c3:
                st.metric("Settling Time", f"{fmt_num(m.settling_time)}s" if m.settled else "N/A")
            with c4:
                _m_effort_norm = _normalized_effort(
                    m.control_effort,
                    (dyn_for_defaults.get_input_bounds() if dyn_for_defaults else None),
                )
                if _m_effort_norm is not None:
                    st.metric("Control Effort", f"{_m_effort_norm:.2f}",
                              help=f"Normalized 0-1ish, as a fraction of the input bound. Raw value: {fmt_num(m.control_effort)}.")
                else:
                    st.metric("Control Effort", f"{fmt_num(m.control_effort)}")
            with c5: st.metric("Stable", "Yes" if m.is_stable else "No")

            with st.expander("Per-state breakdown", expanded=False):
                state_names = summary.get("state_names", [f"x{i}" for i in range(n_states)])
                breakdown_df = pd.DataFrame({
                    "State": state_names,
                    "MSE": [f"{fmt_num(v)}" for v in m.per_state_mse],
                    "ISE (integral sq. error)": [f"{fmt_num(v)}" for v in m.per_state_ise],
                })
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

            fig = plot_simulation_results(res["simulation_data"], "manual", summary.get("state_names", []), summary.get("input_names", []),
                                            u_bounds=dyn_for_defaults.get_input_bounds() if dyn_for_defaults else None,
                                            x_bounds=dyn_for_defaults.get_state_bounds() if dyn_for_defaults else None)
            if fig:
                st.pyplot(fig)
                plt.close(fig)


def render_setup_agent_panel():
    """Graphical walkthrough of what the Initial Setup Agent did to THIS
    upload (agents/dynamics_validator.py:analyze_and_setup) -- shown once,
    prominently, right after a dynamics file is loaded and BEFORE the
    tuning-run state-flow diagram ever appears (that one only exists once
    "Run" is clicked, so the sequencing -- setup first, tuning-loop diagram
    second -- is automatic). Answers three questions visually: was the file
    OK as uploaded (or what got fixed), which states are derivative pairs,
    and where the suggested Q/R/dt numbers actually came from."""
    was_fixed = bool(st.session_state.get("fixed_dynamics_code"))
    summary = st.session_state.dynamics_summary
    state_names = summary.get("state_names", [])

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="subheader">{LCD_ICON_WRENCH_SM} Initial Setup Agent \u2014 what it found for this file</div>', unsafe_allow_html=True)

    # ---- 1. validation status ----
    c1, c2 = st.columns([1, 3])
    with c1:
        if was_fixed:
            st.markdown(f'<span class="fail-badge">{LCD_ICON_WARN_SM} Auto-repaired</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="llm-badge">{LCD_ICON_CHECK_SM} Valid as uploaded</span>', unsafe_allow_html=True)
    with c2:
        if was_fixed:
            st.caption(st.session_state.get("fixed_dynamics_explanation", ""))
        else:
            st.caption("Structurally matched the dynamics standard on the first check -- no LLM repair needed.")

    # ---- 2. derivative pairs ----
    pairs = st.session_state.get("derivative_pairs") or []
    if pairs:
        pair_strs = [f"**{state_names[j]}** = d(**{state_names[i]}**)/dt" for i, j in pairs
                     if i < len(state_names) and j < len(state_names)]
        st.markdown("**Derivative pairs detected** (verified numerically -- dx\u1d62/dt \u2261 x\u2c7c at many random points): " + ", ".join(pair_strs))
    else:
        st.caption("No derivative pairs detected -- per-state Sinusoidal/Cosinusoidal reference pairing will need to be set manually if you want it.")

    # ---- 3. step-response probe + Bryson's rule math ----
    diag = st.session_state.get("qr_diagnostics")
    Q, R = st.session_state.get("suggested_Q"), st.session_state.get("suggested_R")
    if diag and Q:
        st.markdown("**Step-response probe** (open-loop, {:.0%} input step on top of equilibrium) \u2014 this is what the Q weights below are measured from:".format(0.25))
        fig = plot_step_response_probe(diag, state_names)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Bryson's rule** \u2014 weight each variable by the inverse square of its own characteristic scale, so no state or input dominates the cost just because of its units:")
        st.latex(r"Q_{ii} = \frac{1}{\text{range}_i^{\,2}} \cdot \frac{Q_{\max}}{\max_k(1/\text{range}_k^{\,2})} \qquad\qquad R_{jj} = \frac{1}{\text{step}_j^{\,2}} \cdot \frac{R_{\max}}{\max_k(1/\text{step}_k^{\,2})}")
        idx = int(np.argmax(Q))
        r_idx = 0
        example_name = state_names[idx] if idx < len(state_names) else f"x{idx}"
        input_names = summary.get("input_names", [])
        example_input = input_names[r_idx] if r_idx < len(input_names) else "u0"
        st.caption(
            f"e.g. **{example_name}** moved by {diag['ranges'][idx]:.3g} during the probe \u2192 "
            f"Q = 1/{diag['ranges'][idx]:.3g}\u00b2 (rescaled) = **{fmt_num(Q[idx])}**  \u00b7  "
            f"**{example_input}** was stepped by {diag['step_mag'][r_idx]:.3g} \u2192 "
            f"R = 1/{diag['step_mag'][r_idx]:.3g}\u00b2 (rescaled) = **{fmt_num(R[r_idx])}**"
        )
    else:
        st.caption("Step-response probe unavailable for this file (see setup notes below).")

    # ---- 4. dt reasoning ----
    suggested_dt = st.session_state.get("suggested_dt")
    if suggested_dt:
        st.caption(
            f"**dt_mpc \u2248 {suggested_dt:.4g}s** \u2014 the smaller of two estimates: (a) 1/15th of the fastest "
            f"linearized time constant at equilibrium, and (b) 1/8th of the fastest state's step-response rise "
            f"time. Used as the STARTING value; the Actor may periodically adjust it during the run."
        )

    st.markdown('</div>', unsafe_allow_html=True)


def render_agent_flow_diagram(active_node: Optional[str] = None, iteration: int = 0, last_decision: str = "",
                               last_outputs: Optional[dict] = None):
    """A small Simulink-Stateflow-style live diagram of the tuning graph
    (Actor -> Evaluator -> Terminator -> {Critic|Juror}, with Critic feeding
    back to Actor and Juror -- now the run's mandatory final reviewer, not
    just an escalation handler -- feeding back to Actor OR ending the run).
    The node that just executed is highlighted
    with a glowing, pulsing border; everything else stays dim. Re-rendered
    (cheap -- it's just an SVG string) after every node during a live run,
    same pattern as the rest of the live-updating panels.

    Hovering any node reveals that agent's last REASONING/OUTPUT -- the
    same text shown in the Agent Reasoning tab (last_outputs, populated by
    each agent node -- see agents/llm_base.py:merge_last_output), not the
    prompt it was given. Via a pure-CSS tooltip: an invisible, percentage-
    positioned "hover zone" div sits on top of each SVG node (matching its
    position/size in the 700x285 viewBox), containing a styled tooltip div
    that's hidden by default and revealed on :hover -- no JavaScript
    needed, works the same way inside Streamlit's markdown sandbox as the
    rest of this diagram.
    """
    last_outputs = last_outputs or {}

    nodes = {
        "actor":      {"x": 30,  "y": 100, "w": 120, "h": 55, "label": "ACTOR"},
        "evaluator":  {"x": 210, "y": 100, "w": 120, "h": 55, "label": "EVALUATOR"},
        "terminator": {"x": 390, "y": 100, "w": 120, "h": 55, "label": "TERMINATOR"},
        "end":        {"x": 570, "y": 15,  "w": 90,  "h": 45, "label": "END"},
        "critic":     {"x": 300, "y": 210, "w": 120, "h": 55, "label": "CRITIC"},
        "juror":      {"x": 480, "y": 210, "w": 120, "h": 55, "label": "JUROR"},
    }
    colors = {
        "actor": "#4d9fff", "evaluator": "#8be9fd", "terminator": "#60a5fa",
        "critic": "#38bdf8", "juror": "#ff5555", "end": "#50fa7b",
    }
    VIEW_W, VIEW_H = 700, 285

    def node_svg(key: str, n: dict) -> str:
        is_active = key == active_node
        cx, cy = n["x"] + n["w"] / 2, n["y"] + n["h"] / 2
        color = colors[key]
        if is_active:
            fill, stroke, sw, text_fill, cls = color, color, 3, "#0a0e1a", "flow-node-active"
        else:
            fill, stroke, sw, text_fill, cls = "rgba(20,30,50,0.55)", "rgba(255,255,255,0.15)", 1.5, "#9fb0cc", ""
        glow = f'<rect x="{n["x"]-4}" y="{n["y"]-4}" width="{n["w"]+8}" height="{n["h"]+8}" rx="14" fill="none" stroke="{color}" stroke-width="2" opacity="0.5" class="flow-pulse-ring"/>' if is_active else ""
        # NOTE: built as a single concatenated line on purpose (no embedded
        # newlines/indentation) -- see the comment above svg_lines below for why.
        return (
            f'<g class="{cls}">{glow}'
            f'<rect x="{n["x"]}" y="{n["y"]}" width="{n["w"]}" height="{n["h"]}" rx="10" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
            f'<text x="{cx}" y="{cy+5}" text-anchor="middle" font-size="12.5" font-weight="700" '
            f'fill="{text_fill}" font-family="Consolas,monospace" letter-spacing="0.5">{n["label"]}</text>'
            f'</g>'
        )

    edges = [
        # (path "d", has_arrow)
        (f'M 150 127.5 L 210 127.5', True),                                   # actor -> evaluator
        (f'M 330 127.5 L 390 127.5', True),                                   # evaluator -> terminator
        (f'M 480 108 C 520 70, 540 45, 570 32', True),                        # terminator -> end
        (f'M 410 155 C 390 180, 380 195, 362 210', True),                     # terminator -> critic
        (f'M 445 155 C 470 180, 485 195, 500 210', True),                     # terminator -> juror
        (f'M 300 237 C 200 237, 150 190, 88 155', True),                      # critic -> actor
        (f'M 480 250 C 340 285, 140 270, 40 155', True),                      # juror -> actor
    ]

    edges_svg = "".join(
        f'<path d="{d}" fill="none" stroke="rgba(150,170,200,0.45)" stroke-width="1.75" '
        f'marker-end="url(#flowArrow)"/>' for d, _ in edges
    )
    nodes_svg = "".join(node_svg(k, n) for k, n in nodes.items())

    # IMPORTANT: Streamlit's st.markdown() runs its content through a
    # CommonMark-style parser even with unsafe_allow_html=True. Any line
    # that starts with 4+ spaces of leading whitespace is interpreted as an
    # *indented code block* -- which renders as literal escaped text instead
    # of being parsed as HTML. A multi-line f-string written inside an
    # indented Python function (like this one) naturally inherits that
    # indentation on every line, which is exactly what caused the raw
    # "<g class=...>" markup to show up as plain text instead of an actual
    # graphic. Building every piece as a single unindented line (above) and
    # joining them with no leading whitespace (below) avoids the problem
    # entirely -- this is a well-known Streamlit gotcha, not specific to SVG.
    svg_lines = [
        f'<svg viewBox="0 0 {VIEW_W} {VIEW_H}" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto; max-height:260px; display:block;">',
        '<defs>',
        '<marker id="flowArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '<path d="M0,0 L10,5 L0,10 z" fill="rgba(150,170,200,0.5)"/>',
        '</marker>',
        '<style>',
        '@keyframes flowPulse { 0% { opacity: 1; } 50% { opacity: 0.45; } 100% { opacity: 1; } }',
        '@keyframes flowRing { 0% { transform: scale(1); opacity: 0.6; } 100% { transform: scale(1.15); opacity: 0; } }',
        '.flow-node-active { animation: flowPulse 1.3s ease-in-out infinite; transform-origin: center; }',
        '.flow-pulse-ring { animation: flowRing 1.3s ease-out infinite; transform-box: fill-box; transform-origin: center; }',
        '</style>',
        '</defs>',
        edges_svg,
        nodes_svg,
        '</svg>',
    ]
    svg = "".join(svg_lines)

    # -- hover-zone overlays: one invisible, percentage-positioned div per
    # node (matching its SVG position exactly), each containing a tooltip
    # revealed by pure CSS :hover. ---
    tooltip_copy = {
        "evaluator": "No reasoning -- the Evaluator deterministically runs the closed-loop "
                     "simulation and computes metrics; nothing here is agent-generated text.",
        "end": "Terminal state -- reached once the Juror (the final reviewer) decides should_continue = False.",
    }

    def node_tooltip_text(key: str) -> str:
        if key in tooltip_copy:
            return tooltip_copy[key]
        text = last_outputs.get(key)
        if not text:
            return "No reasoning recorded yet for this agent in the current run."
        text = html_module.escape(text)
        if len(text) > 1600:
            text = text[:1600] + "\u2026 (truncated)"
        return text.replace("\n", "<br>")

    hover_zones = []
    for key, n in nodes.items():
        left_pct = n["x"] / VIEW_W * 100
        top_pct = n["y"] / VIEW_H * 100
        width_pct = n["w"] / VIEW_W * 100
        height_pct = n["h"] / VIEW_H * 100
        title = html_module.escape(nodes[key]["label"])
        body = node_tooltip_text(key)
        hover_zones.append(
            f'<div class="flow-hover-zone" style="left:{left_pct:.3f}%; top:{top_pct:.3f}%; '
            f'width:{width_pct:.3f}%; height:{height_pct:.3f}%;">'
            f'<div class="flow-tooltip"><div class="flow-tooltip-title">{title} \u2014 last reasoning</div>'
            f'<div class="flow-tooltip-body">{body}</div></div></div>'
        )
    hover_zones_html = "".join(hover_zones)

    container_lines = [
        '<div class="flow-diagram-wrap">',
        svg,
        hover_zones_html,
        '<style>',
        '.flow-diagram-wrap { position: relative; width: 100%; }',
        '.flow-hover-zone { position: absolute; cursor: help; }',
        '.flow-tooltip { visibility: hidden; opacity: 0; position: absolute; bottom: 108%; left: 50%; '
        'transform: translateX(-50%); width: 320px; max-height: 260px; overflow-y: auto; '
        'background: rgba(8,12,22,0.98); border: 1px solid #4d9fff; border-radius: 10px; '
        'padding: 10px 13px; z-index: 999; text-align: left; pointer-events: none; '
        'box-shadow: 0 10px 28px rgba(0,0,0,0.55); transition: opacity 0.15s ease; }',
        '.flow-hover-zone:hover .flow-tooltip { visibility: visible; opacity: 1; }',
        '.flow-tooltip-title { color: #4d9fff; font-weight: 700; font-size: 11px; letter-spacing: 0.4px; '
        'margin-bottom: 6px; font-family: Consolas, monospace; text-transform: uppercase; }',
        '.flow-tooltip-body { color: #c5d3ea; font-size: 10.5px; line-height: 1.55; '
        'font-family: Consolas, monospace; white-space: normal; }',
        '</style>',
        '</div>',
    ]
    st.markdown("".join(container_lines), unsafe_allow_html=True)

    caption = f"Iteration {iteration}"
    if last_decision:
        caption += f"  &middot;  {last_decision}"
    st.caption(caption + "  &middot;  hover a node to see its last reasoning")


def render_simulation_tab():
    ok_rows = [r for r in st.session_state.results_data if r["ok"] and r.get("simulation_data") is not None]
    if not ok_rows:
        st.info("No successful iteration to plot yet.")
        return
    options = {r["iteration"]: r for r in ok_rows}
    choice = st.selectbox("Iteration", options=list(options.keys()), index=len(options) - 1,
                           format_func=lambda i: f"Iteration {i}  (MSE={fmt_num(options[i]['mse'])})",
                           key="sim_iteration_select")
    row = options[choice]
    summary = st.session_state.get("dynamics_summary", {})
    fig = plot_simulation_results(row["simulation_data"], choice, summary.get("state_names", []), summary.get("input_names", []),
                                    u_bounds=st.session_state.get("run_u_bounds"), x_bounds=st.session_state.get("run_x_bounds"))
    if fig:
        st.pyplot(fig)
        plt.close(fig)


def render_simulation_live():
    """No selectbox (interactive widgets can't be safely re-instantiated
    every iteration within one script run) -- just always shows the most
    recent successful iteration's plot, refreshed live as the run
    progresses. The full picker (render_simulation_tab) takes over once the
    run is idle."""
    ok_rows = [r for r in st.session_state.results_data if r["ok"] and r.get("simulation_data") is not None]
    if not ok_rows:
        st.info("No successful iteration to plot yet.")
        return
    last = ok_rows[-1]
    summary = st.session_state.get("dynamics_summary", {})
    st.caption(f"Live preview -- latest successful iteration ({last['iteration']}). "
               f"Pick any past iteration once the run finishes.")
    fig = plot_simulation_results(last["simulation_data"], last["iteration"], summary.get("state_names", []), summary.get("input_names", []),
                                    u_bounds=st.session_state.get("run_u_bounds"), x_bounds=st.session_state.get("run_x_bounds"))
    if fig:
        st.pyplot(fig)
        plt.close(fig)


def render_data_live():
    """Plain dataframe, no download_button -- see render_results_table for why."""
    if not st.session_state.results_data:
        st.info("No results yet.")
        return
    df = pd.DataFrame(st.session_state.results_data)
    display_df = df.copy()
    display_df["MSE"] = display_df.apply(lambda r: f"{fmt_num(r['mse'])}" if r["ok"] else "--", axis=1)
    display_df["Overshoot"] = display_df.apply(
        lambda r: (f"{fmt_num(r['overshoot'])}" if r["ok"] and r.get("overshoot_meaningful", True) else
                   ("N/A" if r["ok"] else "--")), axis=1)
    display_df["IAE"] = display_df.apply(lambda r: f"{fmt_num(r['iae'])}" if r["ok"] and r.get("iae") is not None else "--", axis=1)
    display_df["ISE"] = display_df.apply(lambda r: f"{fmt_num(r['ise'])}" if r["ok"] and r.get("ise") is not None else "--", axis=1)
    display_df["ControlEffort"] = display_df.apply(lambda r: f"{fmt_num(r['effort'])}" if r["ok"] else "--", axis=1)
    display_df["Oscillations"] = display_df.apply(lambda r: (r["oscillation_count"] if r["ok"] else "--"), axis=1)
    display_df["Settling"] = display_df.apply(
        lambda r: (f"{fmt_num(r['settling'])}s" if r["ok"] and r["settling"] != float("inf") else "N/A"), axis=1)
    # "Stable" = is the response bounded/converging (not diverging or
    # growing) -- a lenient, human-intuitive notion of stability. This is
    # deliberately NOT the same as strict "settled" (fully converged to
    # within a tight tolerance band and held there): a response can be
    # visibly, genuinely stable -- steadily decaying error, no divergence --
    # while still not having crossed that tight threshold yet within the
    # simulation window. Showing strict "settled" here under the "Stable"
    # label was the actual bug behind "it always says No even though I can
    # see it's clearly stable" -- see agents/metrics.py's is_stable field.
    display_df["Stable"] = display_df.apply(lambda r: ("Yes" if r["ok"] and r.get("is_stable") else "No"), axis=1)
    display_df["Strategy"] = display_df["strategy"].astype(str).str.upper()
    display_df["Status"] = display_df.apply(
        lambda r: "UNSTABLE" if r.get("unstable") else ("OK" if r["ok"] else "FAILED"), axis=1)
    display_df["Q"], display_df["R"], display_df["P"] = display_df["Q_formatted"], display_df["R_formatted"], display_df["P_formatted"]
    display_df["Dt"] = display_df.apply(
        lambda r: (f"{fmt_num(r['dt_mpc'])}s" if r["ok"] and r.get("dt_mpc") is not None else "--"), axis=1)
    st.dataframe(
        display_df[["iteration", "Status", "np", "nc", "Q", "R", "P", "Dt", "MSE", "Overshoot", "IAE", "ISE",
                     "Oscillations", "Settling", "Stable", "ControlEffort", "Strategy"]],
        use_container_width=True, hide_index=True,
    )


def render_results_table():
    """Full version with CSV export -- only call this once per script run
    (i.e. from the idle branch, not inside the live streaming loop): like
    any interactive widget, calling st.download_button more than once per
    run raises a duplicate-element error. render_data_live() above is the
    loop-safe equivalent used while a run is in progress."""
    if not st.session_state.results_data:
        st.info("No results yet.")
        return
    render_data_live()
    df = pd.DataFrame(st.session_state.results_data)
    csv = df.drop(columns=["simulation_data"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("Download results as CSV", csv, file_name="mpc_tuning_results.csv", mime="text/csv", key="csv_download")


# ============================================================================
# STATUS BAR (LLM status + Reset -- always visible, replaces the old sidebar
# so a full restart is always one click away without a browser refresh)
# ============================================================================

def render_status_bar():
    _status_col1, _status_col2, _status_col3 = st.columns([3, 3, 1])
    with _status_col1:
        if LLM_READY:
            st.markdown(f'<span class="llm-badge">{LLM_PROVIDER.upper()} &middot; {LLM_MODEL}</span>', unsafe_allow_html=True)
        else:
            st.error(LLM_INIT_ERROR or "LLM not configured.")
    with _status_col2:
        _preset_options = list(LLM_MODEL_PRESETS)
        if st.session_state.selected_llm_model not in _preset_options:
            _preset_options = [st.session_state.selected_llm_model] + _preset_options
        _preset_options = _preset_options + ["Other (type a model name)..."]
        _model_choice = st.selectbox(
            "Model", options=_preset_options,
            index=_preset_options.index(st.session_state.selected_llm_model)
            if st.session_state.selected_llm_model in _preset_options else 0,
            label_visibility="collapsed", key="llm_model_selectbox",
            help="Any model labcd_agents' LLMFactory recognizes (OpenAI, Groq, Cerebras, NVIDIA NIM, "
                 "or Anthropic) -- the provider and its API key are resolved automatically from the name.",
        )
        if _model_choice == "Other (type a model name)...":
            _custom_model = st.text_input(
                "Custom model name", value="", placeholder="e.g. gpt-4o-mini, claude-3-5-haiku-20241022",
                label_visibility="collapsed", key="llm_model_custom_input",
            )
            if _custom_model.strip() and _custom_model.strip() != st.session_state.selected_llm_model:
                st.session_state.selected_llm_model = _custom_model.strip()
                st.rerun()
        elif _model_choice != st.session_state.selected_llm_model:
            st.session_state.selected_llm_model = _model_choice
            st.rerun()
    with _status_col3:
        if st.button("Reset", icon=":material/refresh:", use_container_width=True,
                      help="Clears everything and starts over -- same effect as restarting the app, "
                           "without needing a browser refresh."):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ============================================================================
# MAIN HEADER (LabCD-matching layout)
# ============================================================================

render_lcd_topbar()
render_status_bar()

render_lcd_title_card(
    LCD_ICON_SLIDERS, "MPC Design Setup",
    "Upload a system dynamics file, let the Setup Agent analyze it, and start the tuning process.",
)

_lcd_steps = [
    (LCD_ICON_UPLOAD, "Upload", "Dynamics file"),
    (LCD_ICON_FLASK, "Setup Agent", "Validate & analyze"),
    (LCD_ICON_CHECK_SM, "Discuss", "Optional advisory chat"),
    (LCD_ICON_SLIDERS, "Configure", "Scenario & trajectory"),
    (LCD_ICON_ROCKET, "Tune", "Run the agents"),
]
if not st.session_state.dynamics_loaded:
    _lcd_step_index = 0
elif not st.session_state.advisory_chat_done:
    _lcd_step_index = 2
elif not st.session_state.results_data and not st.session_state.running:
    _lcd_step_index = 3
else:
    _lcd_step_index = 4
render_lcd_stepper(_lcd_steps, _lcd_step_index)

if not st.session_state.dynamics_loaded:
    _upload_stage = st.session_state.get("upload_stage")

    if _upload_stage is None:
        # --- Unified artifact integration ---
        try:
            from backend_core.artifact_store import ArtifactStore
            _mpc_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _mpc_art_store = ArtifactStore(base_dir=os.path.join(_mpc_repo, "artifacts"))
            _mpc_arts = _mpc_art_store.list_artifacts()
        except Exception:
            _mpc_art_store = None
            _mpc_arts = []

        if _mpc_arts:
            with st.container(border=True):
                st.markdown(
                    '<div style="color:#f1f3f7; font-weight:700; margin-bottom:0.6rem;">'
                    "Load from LabCD artifact</div>",
                    unsafe_allow_html=True,
                )
                _mpc_labels = ["%s (%s)" % (a["artifact_id"], a.get("system_name", "")) for a in _mpc_arts]
                _mpc_ids = [a["artifact_id"] for a in _mpc_arts]
                _mpc_sel = st.selectbox(
                    "Artifact",
                    options=list(range(len(_mpc_ids))),
                    format_func=lambda i: _mpc_labels[i],
                    key="mpc_artifact_sel",
                )
                if st.button("Load artifact plugin", type="primary", key="mpc_load_artifact"):
                    try:
                        _plugin_path = _mpc_art_store.load_plugin_path(_mpc_ids[_mpc_sel])
                        # Trajectory / Scenario-tab knobs stay under MPC ownership.
                        # Do not seed trajectory_mode / amplitude / frequency from
                        # artifact pre_launch (those fields are no longer stored).
                        with open(_plugin_path, encoding="utf-8") as _pf:
                            _src = _pf.read()
                        st.session_state.upload_review_code = _src
                        st.session_state.upload_review_filename = Path(_plugin_path).name
                        st.session_state["loaded_artifact_id"] = _mpc_ids[_mpc_sel]
                        # Go through normal load path
                        class _FakeUpload:
                            def __init__(self, name, data):
                                self.name = name
                                self._data = data.encode("utf-8")
                            def getvalue(self):
                                return self._data
                        if load_dynamics_from_file(
                            _FakeUpload(Path(_plugin_path).name, _src)
                        ):
                            st.rerun()
                    except Exception as _exc:
                        st.error(f"Failed to load artifact plugin: {_exc}")

        with st.container(border=True):
            st.markdown('<div style="color:#f1f3f7; font-weight:700; margin-bottom:0.9rem;">System definition file</div>',
                        unsafe_allow_html=True)
            main_uploaded_file = st.file_uploader(
                "System definition file", type=["py"], label_visibility="collapsed", key="main_dropzone_uploader",
            )
            st.caption("Python (.py) &middot; must define create_config() and a BaseDynamics subclass -- see backend_core/AgentMPC/dynamics/plugins/example_pendulum.py")

        _dz_col1, _dz_col2 = st.columns(2)
        with _dz_col1:
            if main_uploaded_file is not None and st.button("Review code first", use_container_width=True, key="main_dropzone_review",
                                                              help="See and optionally edit the file's content before it's checked and loaded."):
                st.session_state.upload_review_code = main_uploaded_file.getvalue().decode("utf-8")
                st.session_state.upload_review_filename = main_uploaded_file.name
                st.session_state.upload_stage = "editing"
                st.rerun()
        with _dz_col2:
            if main_uploaded_file is not None and st.button("Load Dynamics", type="primary", use_container_width=True, key="main_dropzone_load",
                                                              help="Skip straight to validating and loading -- same as Review, minus the editing step."):
                with st.spinner("Loading dynamics..."):
                    if load_dynamics_from_file(main_uploaded_file):
                        st.rerun()

        with st.expander("Required file format", expanded=False):
            st.markdown("""
            - Must contain `create_config()` returning `SystemConfig`
            - Must contain a class inheriting from `BaseDynamics`

            See `backend_core/AgentMPC/dynamics/plugins/example_pendulum.py` for a reference implementation.
            Once loaded, use the sidebar's **Test Dynamics** button to catch plugin bugs before
            running a full tuning session. If the file doesn't quite match the standard, the
            Setup Agent will try to repair it automatically.
            """)
        st.stop()

    elif _upload_stage == "editing":
        render_lcd_title_card(LCD_ICON_FLASK, "Review your dynamics code",
                               "Shown exactly as uploaded -- edit anything you want before the Setup Agent checks it against the standard.")
        edited_code = st.text_area(
            "Dynamics code", value=st.session_state.upload_review_code, height=480,
            key="dynamics_code_editor", label_visibility="collapsed",
        )
        _ed_col1, _ed_col2 = st.columns([1, 5])
        with _ed_col1:
            if st.button("Continue", type="primary", use_container_width=True, key="dynamics_edit_continue"):
                st.session_state.upload_review_code = edited_code
                st.session_state.upload_fix_result = validate_or_fix_dynamics_code(edited_code)
                st.session_state.upload_stage = "reviewing_fix"
                st.rerun()
        with _ed_col2:
            if st.button("Cancel", use_container_width=True, key="dynamics_edit_cancel"):
                st.session_state.upload_stage = None
                st.rerun()
        st.stop()

    elif _upload_stage == "reviewing_fix":
        result = st.session_state.upload_fix_result or {}
        if not result.get("valid"):
            render_lcd_title_card(LCD_ICON_WARN_SM, "Validation failed",
                                   "The Setup Agent couldn't get this into a loadable state.")
            st.error(result.get("error", "Unknown validation error."))
            if result.get("was_fixed"):
                with st.expander("Last attempted fix (still didn't pass -- shown for reference)", expanded=True):
                    st.code(result.get("final_code", ""), language="python")
            if st.button("Back to edit", key="dynamics_fix_back_invalid"):
                st.session_state.upload_stage = "editing"
                st.rerun()
            st.stop()

        if result.get("was_fixed"):
            render_lcd_title_card(LCD_ICON_CHECK_SM, "Setup Agent adjusted your code",
                                   "It didn't match the standard as-is -- here's exactly what changed and why.")
            st.info(result.get("explanation") or "(no explanation returned)")
        else:
            render_lcd_title_card(LCD_ICON_CHECK_SM, "Looks good",
                                   "Your code already matches the standard -- nothing needed to change.")
        st.code(result["final_code"], language="python")

        _rv_col1, _rv_col2 = st.columns([1, 5])
        with _rv_col1:
            if st.button("Confirm and Load", type="primary", use_container_width=True, key="dynamics_confirm_load"):
                with st.spinner("Loading dynamics..."):
                    if finalize_dynamics_load(result["final_code"], st.session_state.upload_review_filename):
                        st.session_state.upload_stage = None
                        st.rerun()
        with _rv_col2:
            if st.button("Back to edit", use_container_width=True, key="dynamics_fix_back_valid"):
                st.session_state.upload_stage = "editing"
                st.rerun()
        st.stop()

if st.session_state.setup_notes:
    show_full = not st.session_state.setup_panel_seen and not st.session_state.results_data and not st.session_state.running
    if show_full:
        render_setup_agent_panel()
        if st.button("Got it -- hide this until the next upload", key="dismiss_setup_panel"):
            st.session_state.setup_panel_seen = True
            st.rerun()
    else:
        with st.expander("Initial Setup Agent (what it found for this file)", expanded=False):
            render_setup_agent_panel()


if st.session_state.dynamics_loaded and not st.session_state.advisory_chat_done:
    render_advisory_chat()
    st.stop()


if not st.session_state.results_data and not st.session_state.running:
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.6rem; margin:0.4rem 0 1rem 0;">'
        '<div style="width:34px; height:34px; border-radius:9px; background:linear-gradient(135deg,#1b3a63,#122238); '
        'display:flex; align-items:center; justify-content:center; flex-shrink:0;">' + LCD_ICON_SLIDERS +
        '</div><div><div style="font-weight:700; font-size:1.05rem; color:#f1f3f7;">Setup</div>'
        '<div style="font-size:0.82rem; color:#6a7a9a;">Three quick steps, then launch -- every default is sensible, edit only what you need to.</div>'
        '</div></div>', unsafe_allow_html=True,
    )
    tab_system, tab_scenario, tab_tuning = st.tabs(["1  \u00b7  System", "2  \u00b7  Scenario", "3  \u00b7  Tuning & Constraints"])

    with tab_system:

        st.subheader("Dynamics File")

        with st.expander("Dynamics File Standard (reference)", expanded=False):
            from backend_core.AgentMPC.agents.dynamics_validator import DYNAMICS_STANDARD
            st.markdown(DYNAMICS_STANDARD)

        uploaded_file = st.file_uploader("Upload dynamics .py file", type=["py"])

        if uploaded_file is not None and st.button("Load Dynamics", type="primary"):
            with st.spinner("Loading dynamics..."):
                if load_dynamics_from_file(uploaded_file):
                    st.rerun()

        if st.session_state.dynamics_loaded:
            summary = st.session_state.dynamics_summary
            st.success(f"Loaded: {summary.get('source_file', 'Unknown')}")
            st.info(f"States: {summary.get('n_states', 0)}  |  Inputs: {summary.get('n_inputs', 0)}")

            if st.session_state.get("fixed_dynamics_code"):
                st.warning("This file didn't match the standard and was automatically fixed by the Agent "
                           "before loading. Save the corrected version below to skip this step next time.")
                with st.expander("What was fixed", expanded=False):
                    st.write(st.session_state.get("fixed_dynamics_explanation", ""))
                st.download_button(
                    "Download corrected dynamics file", st.session_state["fixed_dynamics_code"],
                    file_name=f"fixed_{st.session_state.get('dynamics_file', 'dynamics.py')}",
                    mime="text/x-python", key="download_fixed_dynamics",
                )

            if st.button("Test Dynamics", use_container_width=True,
                          help="Runs one short closed-loop simulation with default parameters to catch plugin bugs early."):
                with st.spinner("Testing..."):
                    run_dynamics_test()

            tr = st.session_state.test_result
            if tr is not None:
                if tr["ok"]:
                    st.success(f"OK -- {tr['steps']} steps, MSE={fmt_num(tr['mse'])}, "
                               f"avg solve {fmt_num(tr['avg_solve_time']*1000)}ms/step")
                else:
                    st.error(f"Test failed: {tr['error']}")
                    with st.expander("Traceback"):
                        st.code(tr["traceback"] or "(no traceback captured)", language="python")


    with tab_scenario:

        st.subheader("Trajectory")

        with st.expander("Custom Reference Trajectory (optional)", expanded=False):
            with st.expander("Trajectory File Standard (reference)", expanded=False):
                from backend_core.AgentMPC.agents.trajectory_validator import TRAJECTORY_STANDARD
                st.markdown(TRAJECTORY_STANDARD)

            traj_file = st.file_uploader("Upload a custom trajectory .py file", type=["py"], key="traj_uploader")
            if traj_file is not None and st.button("Load Trajectory", key="load_traj_button"):
                from backend_core.AgentMPC.agents.trajectory_validator import validate_and_fix_trajectory, validate_trajectory_source
                from backend_core.AgentMPC.dynamics.trajectory_loader import TrajectoryLoader, TrajectoryPluginError

                traj_source = traj_file.getvalue().decode("utf-8")
                st.session_state.fixed_trajectory_code = None
                outcome = validate_trajectory_source(traj_source)

                if not outcome.valid and not LLM_READY:
                    st.error(f"Trajectory validation failed: {outcome.error}")
                    st.info("Auto-fix needs the LLM to be configured -- showing the raw error only.")
                else:
                    if not outcome.valid:
                        st.info("This trajectory file doesn't match the standard yet -- attempting an automatic fix...")
                        with st.spinner("Fixing trajectory file with the Agent..."):
                            fix_result = validate_and_fix_trajectory(traj_source, max_attempts=2)
                        if not fix_result.valid:
                            st.error(f"Could not automatically fix this file after {fix_result.attempts} attempt(s).")
                            st.error(f"Original error: {fix_result.original_error}")
                            st.error(f"Still failing with: {fix_result.still_broken_error}")
                        else:
                            st.success("Fixed automatically.")
                            st.info(fix_result.explanation)
                            st.session_state.fixed_trajectory_code = fix_result.final_code
                            traj_source = fix_result.final_code
                            outcome = validate_trajectory_source(traj_source)

                    if outcome.valid:
                        import tempfile as _tempfile
                        with _tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                            f.write(traj_source)
                            tpath = f.name
                        try:
                            st.session_state.custom_trajectory_loader = TrajectoryLoader.load_from_path(tpath)
                            st.session_state.trajectory_file_name = traj_file.name
                            st.success(f"Loaded: {traj_file.name}")
                        except TrajectoryPluginError as e:
                            st.error(f"Unexpected error loading the validated file: {e}")
                        finally:
                            os.unlink(tpath)

            if st.session_state.get("custom_trajectory_loader") is not None:
                st.success(f"Custom trajectory ready: {st.session_state.get('trajectory_file_name', 'unknown')}")
                if st.session_state.get("fixed_trajectory_code"):
                    st.download_button(
                        "Download corrected trajectory file", st.session_state["fixed_trajectory_code"],
                        file_name=f"fixed_{st.session_state.get('trajectory_file_name', 'trajectory.py')}",
                        mime="text/x-python", key="download_fixed_trajectory",
                    )

        _traj_card_options = [
            {"value": "reg", "icon": "\u2500", "title": "Regulation", "desc": "Hold steady at a fixed target."},
            {"value": "sin", "icon": "\u223f", "title": "Sinusoidal", "desc": "Track a smooth back-and-forth wave."},
            {"value": "pulse", "icon": "\u2594", "title": "Pulse", "desc": "Step to a target, then step back."},
        ]
        if st.session_state.get("custom_trajectory_loader") is not None:
            _traj_card_options.append(
                {"value": "custom", "icon": "\u2723", "title": "Custom", "desc": "Your uploaded trajectory file."}
            )
        selected_trajectory = render_card_selector(
            options=_traj_card_options, key="trajectory_type", default_value="reg",
        )
        if selected_trajectory in ("sin", "pulse"):
            st.caption(
                "\u2139\ufe0f With every state tracking a moving target, **Overshoot will show N/A for every "
                "iteration** -- that's expected, not a bug: overshoot is only defined relative to a FIXED "
                "target to swing past, and there isn't one here. Watch **IAE/ISE** (Convergence tab) instead "
                "for tracking-quality metrics that make sense for a moving reference."
            )

        per_state_trajectory_modes = None
        customize_per_state = False
        if selected_trajectory != "custom" and st.session_state.dynamics_loaded:
            customize_per_state = st.checkbox(
                "Customize per state", value=False,
                help="Pick a different trajectory type (Regulation/Sinusoidal/Cosinusoidal/Pulse) for each "
                     "state individually -- e.g. set a position state to Sinusoidal and its matching velocity "
                     "state to Cosinusoidal yourself (cos is exactly d/dt[sin]) for physically consistent "
                     "tracking, or mix freely for unrelated states. All states share the same Amplitude/"
                     "Frequency/pulse timing below -- only the trajectory *type* is per-state, to keep this "
                     "from turning into one control per state.",
            )
            if customize_per_state:
                state_names_list = st.session_state.dynamics_summary.get("state_names", [])
                per_state_options = {"reg": "Regulation", "sin": "Sinusoidal", "cos": "Cosinusoidal", "pulse": "Pulse"}
                default_rows = pd.DataFrame({
                    "State": state_names_list,
                    "Trajectory": ["Regulation"] * len(state_names_list),
                })
                edited = st.data_editor(
                    default_rows, hide_index=True, use_container_width=True, key="per_state_traj_editor",
                    column_config={
                        "State": st.column_config.TextColumn(disabled=True),
                        "Trajectory": st.column_config.SelectboxColumn(options=list(per_state_options.values())),
                    },
                )
                label_to_code = {v: k for k, v in per_state_options.items()}
                per_state_trajectory_modes = [label_to_code[v] for v in edited["Trajectory"]]

        traj_amplitude, traj_frequency = 0.5, 0.5
        traj_pulse_start, traj_pulse_end = 0.2, 0.7
        show_sin_controls = selected_trajectory == "sin" or (customize_per_state and per_state_trajectory_modes and
                                                                any(m in ("sin", "cos") for m in per_state_trajectory_modes))
        show_pulse_controls = selected_trajectory == "pulse" or (customize_per_state and per_state_trajectory_modes and
                                                                    "pulse" in per_state_trajectory_modes)
        if show_sin_controls:
            c1, c2 = st.columns(2)
            with c1:
                traj_amplitude = st.slider("Amplitude", 0.05, 3.0, 0.5, step=0.05, key="sin_amplitude")
            with c2:
                traj_frequency = st.slider("Frequency (Hz)", 0.05, 3.0, 0.5, step=0.05, key="sin_frequency")
        if show_pulse_controls:
            c1, c2, c3 = st.columns(3)
            with c1:
                traj_amplitude = st.slider("Amplitude", 0.05, 3.0, 0.5, step=0.05, key="pulse_amplitude")
            with c2:
                traj_pulse_start = st.slider("Rise at (% of sim time)", 0, 90, 20, key="pulse_start") / 100.0
            with c3:
                traj_pulse_end = st.slider("Fall at (% of sim time)", 10, 100, 70, key="pulse_end") / 100.0
            if traj_pulse_end <= traj_pulse_start:
                st.warning("Fall time should be after rise time.")

        if st.session_state.dynamics_loaded:
            with st.expander("Preview reference trajectory", expanded=True):
                st.caption(
                    "What the controller will actually be asked to track, for the CURRENT settings above -- "
                    "updates live as you adjust amplitude/frequency/trajectory type. Shown over a fixed "
                    f"{8.0:.0f}s preview window regardless of the actual Simulation Time setting."
                )
                render_trajectory_preview(
                    st.session_state.dyn, selected_trajectory, per_state_trajectory_modes,
                    traj_amplitude, traj_frequency, traj_pulse_start, traj_pulse_end,
                )

        scenario_level = render_card_selector(
            options=[
                {"value": 1, "icon": "\u25cf", "title": "Level 1 \u00b7 Nominal",
                 "desc": "Clean run, no noise, ideal starting point."},
                {"value": 2, "icon": "\u25d0", "title": "Level 2 \u00b7 Noise",
                 "desc": "Adds measurement noise to selected states."},
                {"value": 3, "icon": "\u25c9", "title": "Level 3 \u00b7 Robust",
                 "desc": "Harder start + noise + physical parameter mismatch."},
            ],
            key="scenario_level", default_value=1,
        )

        _scn_state_names = st.session_state.dynamics_summary.get("state_names", []) if st.session_state.dynamics_loaded else []
        scenario_noise_std_value = None
        scenario_noise_state_mask = None
        scenario_robust_push_scale = None
        scenario_robust_state_mask = None
        scenario_robust_noise_fraction = None
        scenario_perturb_physical_params = True

        if scenario_level == 2 and _scn_state_names:
            with st.expander("Noise settings (Level 2)", expanded=True):
                st.caption(
                    "By default, every state gets the same modest additive Gaussian measurement noise "
                    "(~1% of that state's declared range each step). Edit which states are affected and how "
                    "much below -- this was previously a fixed, invisible value."
                )
                _default_noise = float(st.session_state.get("_suggested_noise_std", 0.01))
                scenario_noise_std_value = st.number_input(
                    "Noise standard deviation (applied to every selected state)",
                    min_value=0.0, value=_default_noise, step=_default_noise / 10 if _default_noise > 0 else 0.001,
                    format="%.5f",
                )
                _noisy_states = st.multiselect(
                    "States that receive noise", options=_scn_state_names, default=_scn_state_names,
                    help="States left unchecked stay noise-free (exact measurement) even at this scenario level.",
                )
                scenario_noise_state_mask = np.array([s in _noisy_states for s in _scn_state_names])

        elif scenario_level == 3 and _scn_state_names:
            with st.expander("Robustness settings (Level 3)", expanded=True):
                st.caption(
                    "By default, every state's initial value is pushed toward the edge of its declared bounds "
                    "(a harder starting point), plus half of Level 2's noise magnitude. Edit the push strength, "
                    "which states get pushed, and the noise below."
                )
                scenario_robust_push_scale = st.slider(
                    "Push aggressiveness (relative to default)", 0.1, 2.0, 1.0, step=0.1,
                    help="1.0 = the original default push. Lower = a milder/easier starting point, "
                         "higher = an even harder one.",
                )
                _pushed_states = st.multiselect(
                    "States that get pushed to a harder starting point", options=_scn_state_names, default=_scn_state_names,
                    help="States left unchecked stay at their normal (Level 1) initial value instead.",
                )
                scenario_robust_state_mask = np.array([s in _pushed_states for s in _scn_state_names])

                _default_noise = float(st.session_state.get("_suggested_noise_std", 0.01))
                scenario_noise_std_value = st.number_input(
                    "Base noise standard deviation (before the 0.5x Level-3 fraction below)",
                    min_value=0.0, value=_default_noise, step=_default_noise / 10 if _default_noise > 0 else 0.001,
                    format="%.5f",
                )
                scenario_robust_noise_fraction = st.slider(
                    "Noise fraction applied at this level", 0.0, 2.0, 0.5, step=0.1,
                    help="Actual noise std = base noise x this fraction. 0.5 = the original default "
                         "(half of Level 2's magnitude); 0 = no noise at all, just the harder starting point.",
                )
                _noisy_states = st.multiselect(
                    "States that receive noise", options=_scn_state_names, default=_scn_state_names, key="robust_noisy_states",
                    help="States left unchecked stay noise-free even at this scenario level.",
                )
                scenario_noise_state_mask = np.array([s in _noisy_states for s in _scn_state_names])

                st.divider()
                scenario_perturb_physical_params = st.checkbox(
                    "Also perturb some physical parameters by up to 20% (plant-model mismatch)", value=True,
                    help="Beyond a harder initial condition and noise, this simulates the tuned controller "
                         "facing a REAL system whose physical parameters (mass, length, damping, etc) aren't "
                         "exactly what the model assumes -- a genuine robustness test, not just a harder start. "
                         "Each selected parameter gets its OWN random boost between 0% and 20% (not a uniform "
                         "20% across the board) -- e.g. one might end up +5%, another +17%.",
                )
                if scenario_perturb_physical_params and st.session_state.dyn is not None and st.session_state.dyn.params:
                    _preview_keys = sorted(
                        k for k, v in st.session_state.dyn.params.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    )[::2]
                    if _preview_keys:
                        _preview_text = ", ".join(
                            f"**{k}** (currently {st.session_state.dyn.params[k]:.4g})" for k in _preview_keys
                        )
                        st.caption(
                            f"Will perturb by a random amount up to +20% each (re-rolled fresh for this run): "
                            f"{_preview_text}."
                        )
                    else:
                        st.caption("No numeric physical parameters found on this plugin to perturb.")

        st.markdown('<div class="lcd-advanced">', unsafe_allow_html=True)
        with st.expander("Advanced Settings", expanded=False):
            st.caption("Sensible defaults are already set -- only change these if you know what you're after.")
            simulation_time = st.slider(
                "Simulation Time (s)", 2.0, 20.0, 8.0, step=0.5,
                help="How long each candidate parameter set is simulated for. Longer runs show more "
                     "post-settling behavior (useful for visually confirming true steady-state stability), "
                     "at the cost of slower iterations.",
            )
            settling_tolerance_pct = st.slider(
                "Settling Tolerance (%)", 1, 20, 5,
                help="How close to the target counts as 'settled', as a percent of the initial error. "
                     "Lower = stricter (only very tight convergence counts). If a response looks visually "
                     "flat/stable to you but still shows 'Stable: No', try raising this.",
            )
            max_iterations = st.slider("Max Iterations", 3, 30, 10)
            min_explore_iterations = st.slider(
                "Minimum Explore Iterations", 0, 15, 4,
                help="The Critic can't recommend 'exploit' (fine-tuning) before this many iterations have "
                     "run, regardless of what it thinks -- keeps the search from settling into local "
                     "fine-tuning before it's covered enough of the parameter space.",
            )
            exploration_intensity = st.slider(
                "Exploration Intensity (%)", 1, 100, 50,
                help="How bold the Actor is while in 'explore' mode. 50% = normal (the default behavior). "
                     "100% = wild, aggressive parameter changes each iteration. 1% = very cautious, small "
                     "adjustments only. Doesn't affect 'exploit' (fine-tuning near the best result).",
            )
        st.markdown('</div>', unsafe_allow_html=True)


    with tab_tuning:

        n_states_for_init = st.session_state.dynamics_summary.get("n_states", 4) if st.session_state.dynamics_loaded else 4
        default_init_hint = (
            ", ".join(f"{fmt_num(v)}" for v in st.session_state.dyn.config.default_initial_state)
            if st.session_state.dynamics_loaded else ""
        )
        use_custom_initial_state = st.checkbox(
            "Set custom initial state", value=False,
            help="Override the dynamics plugin's default_initial_state with your own exact values -- "
                 "applied AFTER the Scenario Level preset above, so this takes precedence over it.",
        )
        custom_initial_state, initial_state_error = None, None
        if use_custom_initial_state:
            init_state_text = st.text_input(
                f"Initial state -- {n_states_for_init} comma-separated values",
                value=default_init_hint,
                help="One value per state, in the order shown in the sidebar's 'States' summary "
                     "after loading the dynamics file.",
            )
            try:
                custom_initial_state = np.array([float(v.strip()) for v in init_state_text.split(",") if v.strip()])
                if custom_initial_state.size != n_states_for_init:
                    initial_state_error = f"Need exactly {n_states_for_init} value(s), got {custom_initial_state.size}."
                    custom_initial_state = None
            except ValueError:
                initial_state_error = "Must be comma-separated numbers."
            if initial_state_error:
                st.error(initial_state_error)

        st.divider()
        with st.expander("Guidance for the Agent (optional)", expanded=False):
            optimization_focus = st.selectbox(
                "Optimization Focus",
                options=list(OPTIMIZATION_FOCUS_LABELS.keys()),
                format_func=lambda k: OPTIMIZATION_FOCUS_LABELS[k],
                help="Determines what 'best result' means (both for the agents' own best-so-far "
                     "tracking and the Best Result tab) -- 'Balanced' considers MSE, overshoot, "
                     "settling time, and control effort together, same as before.",
            )
            user_guidance = st.text_area(
                "Anything else to tell the Actor/Critic, in plain language",
                value="",
                placeholder="e.g. \"This system has a fragile actuator, avoid large control inputs\" "
                            "or \"Prefer slower but very smooth responses over fast but jerky ones.\"",
                help="Passed directly into the Actor and Critic prompts, in addition to the "
                     "Optimization Focus above. Leave blank if you have nothing to add.",
            )

        st.divider()
        n_states_hint = st.session_state.dynamics_summary.get("n_states", 4) if st.session_state.dynamics_loaded else 4
        n_inputs_hint = st.session_state.dynamics_summary.get("n_inputs", 1) if st.session_state.dynamics_loaded else 1

        if st.session_state.setup_notes:
            with st.expander("Initial Setup Analysis", expanded=False):
                st.caption("Computed once, deterministically, from the dynamics itself (no LLM) -- see agents/dynamics_validator.py.")
                for note in st.session_state.setup_notes:
                    st.markdown(f"- {note}")

        suggested_dt = st.session_state.get("suggested_dt")
        use_manual_dt = st.checkbox(
            "Set dt_mpc manually", value=False,
            help="By default dt_mpc is estimated once from the system's own dynamics (see Initial Setup "
                 "Analysis above) as the STARTING point, and the Actor may periodically adjust it during "
                 "the run just like Q/R/Np/Nc. Check this to fix it to a specific value instead (the Actor "
                 "will then leave it alone).",
        )
        if use_manual_dt:
            dt_mpc_value = st.number_input(
                "dt_mpc (s)", min_value=0.001, max_value=1.0,
                value=float(suggested_dt) if suggested_dt else 0.02, step=0.001, format="%.4f",
            )
        else:
            dt_mpc_value = suggested_dt if suggested_dt else 0.02
            if suggested_dt:
                st.caption(f"Using suggested dt_mpc = {suggested_dt:.4g}s as the starting point (Actor may adjust it during the run)")

        suggested_feedforward = st.session_state.get("suggested_feedforward")
        use_feedforward = False
        if suggested_feedforward is not None:
            input_names_hint = st.session_state.dynamics_summary.get("input_names", []) if st.session_state.dynamics_loaded else []
            ff_display = ", ".join(f"{n}={v:.4g}" for n, v in zip(input_names_hint, suggested_feedforward)) \
                         if input_names_hint else ", ".join(f"{v:.4g}" for v in suggested_feedforward)
            use_feedforward = st.checkbox(
                "Use computed feedforward trim input", value=False,
                help="Linear-basis controllers like MPC/PID are normally built around a baseline input that "
                     "holds the system at its target on its own -- the controller only applies the CORRECTION "
                     "on top of that. Most dynamics files don't declare this explicitly (it defaults to zero), "
                     "so the Setup Agent numerically solved for it here: " + ff_display + ". Off by default -- "
                     "this changes the starting point of every run; leave off to keep the current behavior "
                     "exactly as it was.",
            )
            if use_feedforward:
                st.latex(r"u_k \;=\; u_{k-1} \;+\; \Delta u_0^{*}, \qquad u_{-1} \;=\; U_e")
                st.caption(
                    "The MPC solves for a sequence of increments (du) every step, not absolute inputs, and "
                    "applies only the first one on top of whatever the input currently is. This toggle changes "
                    "only the seed the very first step starts from (u at k=-1) -- from the plugin's own "
                    "default (usually all-zero) to the Setup Agent's computed trim Ue below, so the "
                    "controller's first correction is measured relative to an already-balanced baseline "
                    "instead of relative to zero."
                )
                _ue_rows = ", ".join(
                    f"{n} = {v:.4g}" for n, v in zip(
                        input_names_hint or [f"u{i}" for i in range(len(suggested_feedforward))], suggested_feedforward
                    )
                )
                st.latex(r"U_e = \begin{bmatrix}" + r" \\ ".join(f"{v:.4g}" for v in suggested_feedforward) + r"\end{bmatrix}")
                st.caption(f"({_ue_rows})")

        st.divider()
        st.markdown('<div style="color:#f1f3f7; font-weight:700;">Constraints (State &amp; Input Bounds)</div>', unsafe_allow_html=True)
        st.caption(
            "One of MPC's core strengths: these are HARD limits enforced directly inside the optimization "
            "itself (not just checked after the fact) -- the controller physically cannot propose a solution "
            "that violates them. Pre-filled from what the dynamics file itself declares, if anything; edit "
            "freely. Leave a cell as -inf/inf for 'no limit on this side'."
        )
        _cn_input_names = summary.get("input_names", []) if st.session_state.dynamics_loaded else []
        _cn_state_names = summary.get("state_names", []) if st.session_state.dynamics_loaded else []
        _plugin_u_bounds = st.session_state.dyn.get_input_bounds() if st.session_state.dynamics_loaded else None
        _plugin_x_bounds = st.session_state.dyn.get_state_bounds() if st.session_state.dynamics_loaded else None

        _u_lo = list(_plugin_u_bounds[0]) if _plugin_u_bounds is not None else [-np.inf] * n_inputs_hint
        _u_hi = list(_plugin_u_bounds[1]) if _plugin_u_bounds is not None else [np.inf] * n_inputs_hint
        _u_bounds_df = pd.DataFrame({"Input": _cn_input_names or [f"u{i}" for i in range(n_inputs_hint)],
                                      "Min": _u_lo, "Max": _u_hi})
        _u_bounds_edited = st.data_editor(
            _u_bounds_df, disabled=["Input"], hide_index=True, use_container_width=True, key="constraint_u_editor",
        )

        _x_lo = list(_plugin_x_bounds[0]) if _plugin_x_bounds is not None else [-np.inf] * n_states_hint
        _x_hi = list(_plugin_x_bounds[1]) if _plugin_x_bounds is not None else [np.inf] * n_states_hint
        _x_bounds_df = pd.DataFrame({"State": _cn_state_names or [f"x{i}" for i in range(n_states_hint)],
                                      "Min": _x_lo, "Max": _x_hi})
        _x_bounds_edited = st.data_editor(
            _x_bounds_df, disabled=["State"], hide_index=True, use_container_width=True, key="constraint_x_editor",
        )

        constraint_u_bounds = (
            _u_bounds_edited["Min"].to_numpy(dtype=float), _u_bounds_edited["Max"].to_numpy(dtype=float),
        )
        constraint_x_bounds = (
            _x_bounds_edited["Min"].to_numpy(dtype=float), _x_bounds_edited["Max"].to_numpy(dtype=float),
        )

        suggested_Q = st.session_state.get("suggested_Q")
        suggested_R = st.session_state.get("suggested_R")

        q_for_default = list(suggested_Q) if suggested_Q else [1.0] * n_states_hint
        _robust_boosted_states = []
        if scenario_level == 3 and scenario_robust_state_mask is not None and suggested_Q:
            _state_names_for_boost = st.session_state.dynamics_summary.get("state_names", [])
            for _i in range(min(len(q_for_default), len(scenario_robust_state_mask))):
                if scenario_robust_state_mask[_i]:
                    q_for_default[_i] = q_for_default[_i] * 1.2
                    _robust_boosted_states.append(
                        _state_names_for_boost[_i] if _i < len(_state_names_for_boost) else f"state{_i}"
                    )

        with st.expander("Initial Parameters", expanded=bool(_robust_boosted_states)):
            if suggested_Q and suggested_R:
                st.caption("Pre-filled from the Initial Setup Analysis (Bryson's rule) and used automatically "
                           "as the Actor's starting point -- edit freely to override.")
            else:
                st.caption("No automatic suggestion available for this file -- using flat defaults as the "
                           "starting point. Edit freely.")
            if _robust_boosted_states:
                st.info(
                    f"**Level 3 (Robust):** the Setup Agent increased Q by 20% for the state(s) being pushed "
                    f"to a harder starting point -- **{', '.join(_robust_boosted_states)}** -- for a more "
                    f"aggressive correction against the larger initial error. Shown pre-filled below; edit "
                    f"freely if you'd rather not have this."
                )
            c1, c2 = st.columns(2)
            with c1: init_np = st.number_input("Np", min_value=1, max_value=50, value=12)
            with c2: init_nc = st.number_input("Nc", min_value=1, max_value=50, value=5)
            q_default = ", ".join(f"{v:.4g}" for v in q_for_default)
            r_default = ", ".join(f"{v:.4g}" for v in suggested_R) if suggested_R else ", ".join(["0.1"] * n_inputs_hint)
            init_q_text = st.text_input(f"Q -- {n_states_hint} values", value=q_default)
            init_r_text = st.text_input(f"R -- {n_inputs_hint} values", value=r_default)
            seed_params, seed_error = parse_seed_params(init_np, init_nc, init_q_text, init_r_text, n_states_hint, n_inputs_hint)
            if seed_error:
                st.error(seed_error)


    st.divider()
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.5rem; margin:0.3rem 0 0.8rem 0;">'
        '<div style="font-weight:700; font-size:1.0rem; color:#f1f3f7;">Ready to launch</div></div>',
        unsafe_allow_html=True,
    )

    run_disabled = (
        not (st.session_state.dynamics_loaded and LLM_READY)
        or bool(seed_error)
        or (use_custom_initial_state and bool(initial_state_error))
        or st.session_state.running
    )
    run_button = st.button("Run", type="primary", use_container_width=True, disabled=run_disabled)
    if not st.session_state.dynamics_loaded:
        st.warning("Load a dynamics file first")
    elif not LLM_READY:
        st.warning("LLM not configured")

    if run_button:
        st.session_state.logs = []
        st.session_state.reasoning_entries = []
        st.session_state._last_history_len = 0
        st.session_state.results_data = []
        st.session_state.latest_params = {}
        st.session_state.last_outputs = {}
        st.session_state.run_started_at = datetime.now()
        st.session_state.run_finished_at = None
        st.session_state.stop_requested = False
        st.session_state.stopped_by_user = False
        st.session_state.report_pdf_bytes = None
        st.session_state.report_pdf_name = None
        st.session_state.export_script_text = None
        st.session_state.export_script_name = None
        st.session_state.animation_gif_bytes = None
        st.session_state.animation_gif_name = None
        st.session_state.animation_description = None
        st.session_state.animation_render_note = None
        st.session_state.manual_best_iteration = None
        st.session_state.diagnostics_report = None

        plugin, dyn = st.session_state.plugin, st.session_state.dyn
        cfg = Config()
        cfg.mpc.prediction_horizon = 12
        cfg.mpc.control_horizon = 5
        cfg.data.dt_mpc = dt_mpc_value
        cfg.data.feedforward_override = suggested_feedforward if use_feedforward and suggested_feedforward else None
        cfg.mpc.u_bounds = constraint_u_bounds
        cfg.mpc.x_bounds = constraint_x_bounds
        cfg.data.simulation_time = simulation_time
        cfg.data.settling_tolerance = settling_tolerance_pct / 100.0
        cfg.data.trajectory_mode = selected_trajectory
        cfg.data.trajectory_amplitude = traj_amplitude
        cfg.data.trajectory_frequency = traj_frequency
        cfg.data.trajectory_pulse_start = traj_pulse_start
        cfg.data.trajectory_pulse_end = traj_pulse_end
        cfg.data.trajectory_per_state_modes = per_state_trajectory_modes
        if selected_trajectory == "custom" and st.session_state.get("custom_trajectory_loader") is not None:
            cfg.data.custom_trajectory_fn = st.session_state.custom_trajectory_loader.generate
        _x0_after_scenario, _target_after_scenario, run_perturbed_params = apply_scenario_level(
            dyn, cfg, scenario_level,
            noise_std_value=scenario_noise_std_value,
            noise_state_mask=scenario_noise_state_mask,
            robust_push_scale=scenario_robust_push_scale,
            robust_state_mask=scenario_robust_state_mask,
            robust_noise_fraction=scenario_robust_noise_fraction,
            perturb_physical_params=scenario_perturb_physical_params,
        )
        st.session_state.run_perturbed_params = run_perturbed_params
        if use_custom_initial_state and custom_initial_state is not None:
            dyn.config.default_initial_state = custom_initial_state.copy()

        entry_node = "evaluator" if seed_params else "actor"
        graph = build_ui_tuning_graph(dyn, cfg, entry_node=entry_node)

        focus_sentence = (
            "" if optimization_focus == "balanced"
            else f"Optimization focus: {OPTIMIZATION_FOCUS_LABELS[optimization_focus]}. "
                 f"Prioritize this above the other metrics when proposing parameters."
        )
        combined_guidance = "\n".join(s for s in (focus_sentence, user_guidance.strip()) if s)

        st.session_state.token_tracker = TokenUsageTracker() if TOKEN_TRACKING_AVAILABLE else None

        state = initial_state(dyn, system_name=plugin.dynamics_class.__name__, max_iterations=max_iterations,
                               ui_scenario_level=scenario_level,
                               seed_params=seed_params if seed_params else None,
                               user_guidance=combined_guidance, min_explore_iterations=min_explore_iterations,
                               cost_weights=OPTIMIZATION_FOCUS_PRESETS.get(optimization_focus),
                               exploration_intensity=exploration_intensity, dt_mpc=dt_mpc_value,
                               token_tracker=st.session_state.token_tracker)

        st.session_state.logs.append({"time": time.strftime("%H:%M:%S"), "node": "SYSTEM",
            "message": f"Starting: {plugin.dynamics_class.__name__} "
                       f"(scenario={SCENARIO_LEVEL_NAMES[scenario_level]}, trajectory={selected_trajectory}, "
                       f"init={'seeded' if entry_node=='evaluator' else 'agent-proposed'})"})

        summary = st.session_state.dynamics_summary
        # n_states/n_inputs are needed again on every future rerun (each of which
        # only processes ONE node -- see below), so they're persisted in session
        # state rather than kept as a local variable that would vanish once this
        # script execution ends.
        st.session_state.run_n_states = summary.get("n_states", 4)
        st.session_state.run_n_inputs = summary.get("n_inputs", 1)
        st.session_state.run_scenario_level = scenario_level
        st.session_state.run_max_iterations = max_iterations
        st.session_state.run_simulation_time = simulation_time
        st.session_state.run_trajectory_mode = selected_trajectory
        st.session_state.run_trajectory_amplitude = traj_amplitude
        st.session_state.run_trajectory_frequency = traj_frequency
        st.session_state.run_trajectory_pulse_start = traj_pulse_start
        st.session_state.run_trajectory_pulse_end = traj_pulse_end
        st.session_state.run_trajectory_per_state_modes = per_state_trajectory_modes
        st.session_state.run_u_bounds = constraint_u_bounds
        st.session_state.run_x_bounds = constraint_x_bounds

        # The generator itself -- NOT consumed here. Storing it in session_state
        # and advancing it by exactly one `next()` call per script rerun (below)
        # is what lets the Stop button take effect between agent steps instead
        # of only after the entire multi-iteration run finishes: a single
        # blocking `for output in graph.stream(state): ...` loop, as this used
        # to be, runs to completion (or exception) within ONE script execution,
        # during which Streamlit cannot process any button click at all.
        st.session_state.graph_iterator = graph.stream(state)
        st.session_state.running = True
        st.rerun()

    st.stop()  # stay on the Configure view on every rerun where Run was NOT just clicked


def render_report_section():
    """The 'Generate Report', 'Export Script', and 'Generate Animation'
    buttons -- all disabled while a run is active, enabled once it's
    stopped or finished (there's a best result to work from). Report: runs
    the Report Agent (an LLM call analyzing the actual results) and builds
    a PDF via reportlab. Export: builds a self-contained .py file combining
    the actual dynamics/MPC source with the tuned best parameters,
    runnable on the user's own machine. Animation: the Animation Agent (see
    agents/animation_agent.py) writes a small, heavily-sandboxed per-frame
    drawing function, and ordinary Python code renders it against the
    ACTUAL best-result trajectory into a GIF -- replaces an earlier static-
    SVG-schematic feature that wasn't useful enough to keep. All three
    outputs are cached in session_state so re-rendering this section (every
    script rerun) doesn't redo the work unless the user explicitly clicks
    the button again.
    """
    has_results = bool(st.session_state.results_data)
    can_generate = has_results and not st.session_state.running and REPORT_FEATURE_AVAILABLE
    can_export = has_results and not st.session_state.running and EXPORT_SCRIPT_FEATURE_AVAILABLE and _best_row() is not None
    can_animate = has_results and not st.session_state.running and ANIMATION_FEATURE_AVAILABLE and _best_row() is not None

    if not REPORT_FEATURE_AVAILABLE and has_results:
        st.warning(
            f"Report generation needs the `reportlab` package, which isn't installed "
            f"(`{REPORT_FEATURE_ERROR}`). Run `pip install -r requirements.txt` (or just "
            f"`pip install reportlab`) and restart the app to enable it."
        )

    if has_results and ANIMATION_FEATURE_AVAILABLE:
        with st.expander("Animation context (optional)", expanded=False):
            st.caption(
                "The Animation Agent can't always tell things like which configuration a zero angle "
                "represents (e.g. an inverted pendulum's theta=0 is conventionally UPRIGHT, not hanging "
                "down -- the opposite of a plain pendulum) or which way a positive torque turns things. "
                "If the last animation looked physically wrong, describe the actual convention here."
            )
            animation_user_context = st.text_area(
                "Additional context for the Animation Agent", value="", height=80,
                placeholder="e.g. \"theta1=0 and theta2=0 both mean pointing straight UP (unstable "
                            "equilibrium), angles measured counterclockwise from vertical\"",
                key="animation_user_context", label_visibility="collapsed",
            )
    else:
        animation_user_context = ""

    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
    with col2:
        button_help = "Available once the run stops or finishes."
        if can_generate:
            button_help = ("The Report Agent analyzes the actual results (not a template) and produces "
                            "a PDF with charts, the full iteration table, and its own commentary.")
        elif has_results and not REPORT_FEATURE_AVAILABLE:
            button_help = "Needs the `reportlab` package -- see the message above."
        if st.button("Generate Report", icon=":material/description:", use_container_width=True,
                      disabled=not can_generate, help=button_help):
            summary = st.session_state.dynamics_summary
            best = _best_row()
            with st.spinner("Report Agent analyzing results..."):
                analysis = generate_report_analysis(
                    system_name=summary.get("dynamics_class", "Unknown System"),
                    state_names=summary.get("state_names", []),
                    input_names=summary.get("input_names", []),
                    results_data=st.session_state.results_data,
                    best_row=best,
                    stopped_by_user=st.session_state.stopped_by_user,
                    tracker=st.session_state.get("token_tracker"),
                )
            with st.spinner("Building PDF..."):
                df = pd.DataFrame(st.session_state.results_data)
                conv_fig = plot_convergence(df) if not df.empty else None
                sim_fig = None
                if best and best.get("simulation_data"):
                    sim_fig = plot_simulation_results(
                        best["simulation_data"], best["iteration"],
                        summary.get("state_names", []), summary.get("input_names", []),
                        u_bounds=st.session_state.get("run_u_bounds"), x_bounds=st.session_state.get("run_x_bounds"))

                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                        tmp_path = tf.name
                    build_pdf_report(
                        tmp_path,
                        system_name=summary.get("dynamics_class", "Unknown System"),
                        dynamics_summary=summary,
                        results_data=st.session_state.results_data,
                        best_row=best,
                        analysis=analysis,
                        convergence_fig=conv_fig,
                        simulation_fig=sim_fig,
                        stopped_by_user=st.session_state.stopped_by_user,
                    )
                    with open(tmp_path, "rb") as f:
                        st.session_state.report_pdf_bytes = f.read()
                    st.session_state.report_pdf_name = (
                        f"mpc_report_{summary.get('dynamics_class', 'system')}_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Report generation failed: {e}")
                finally:
                    if conv_fig is not None:
                        plt.close(conv_fig)
                    if sim_fig is not None:
                        plt.close(sim_fig)
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    with col3:
        export_help = "Available once the run stops or finishes."
        if can_export:
            export_help = ("Downloads a single .py file with the actual dynamics + MPC controller + the "
                            "best parameters found, runnable on your own machine with just "
                            "numpy/scipy/matplotlib -- reproduces the same state/input/metric plots.")
        elif has_results and not EXPORT_SCRIPT_FEATURE_AVAILABLE:
            export_help = f"Unavailable: {EXPORT_SCRIPT_FEATURE_ERROR}"
        if st.button("Export Script", icon=":material/download:", use_container_width=True,
                      disabled=not can_export, help=export_help):
            best = _best_row()
            summary = st.session_state.dynamics_summary
            dyn = st.session_state.dyn
            try:
                script_text = generate_standalone_script(
                    dynamics_source_code=st.session_state.get("dynamics_source_code") or "",
                    class_name=summary.get("dynamics_class", "MyDynamics"),
                    best_params={
                        "Np": best["np"], "Nc": best["nc"],
                        "Q": [best[k] for k in sorted((k for k in best if k.startswith("q") and k[1:].isdigit()),
                                                        key=lambda k: int(k[1:]))],
                        "R": [best[k] for k in sorted((k for k in best if k.startswith("r") and k[1:].isdigit()),
                                                        key=lambda k: int(k[1:]))],
                    },
                    dt_mpc=best.get("dt_mpc") or st.session_state.get("suggested_dt") or 0.02,
                    simulation_time=st.session_state.get("run_simulation_time", 8.0),
                    system_name=summary.get("dynamics_class", "MyDynamics"),
                    feedforward=st.session_state.get("suggested_feedforward"),
                    # dyn retains whatever apply_scenario_level actually set for THIS run (Level 2/3
                    # mutate its config.default_initial_state and params in place; see
                    # agents/scenario_presets.py) -- reading it back here, after the run, reproduces
                    # the exact scenario rather than the dynamics file's own plain defaults.
                    initial_state=list(dyn.config.default_initial_state) if dyn is not None else None,
                    physical_params_override=dict(dyn.params) if dyn is not None and dyn.params else None,
                    trajectory_mode=st.session_state.get("run_trajectory_mode"),
                    trajectory_amplitude=st.session_state.get("run_trajectory_amplitude", 0.5),
                    trajectory_frequency=st.session_state.get("run_trajectory_frequency", 0.5),
                    trajectory_pulse_start=st.session_state.get("run_trajectory_pulse_start", 0.2),
                    trajectory_pulse_end=st.session_state.get("run_trajectory_pulse_end", 0.7),
                    trajectory_per_state_modes=st.session_state.get("run_trajectory_per_state_modes"),
                    u_bounds=st.session_state.get("run_u_bounds"),
                    x_bounds=st.session_state.get("run_x_bounds"),
                )
                st.session_state.export_script_text = script_text
                st.session_state.export_script_name = (
                    f"{summary.get('dynamics_class', 'mpc')}_export_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M')}.py"
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Script export failed: {e}")

    if st.session_state.get("export_script_text"):
        with col1:
            st.download_button(
                "Download Standalone Script", st.session_state.export_script_text,
                file_name=st.session_state.get("export_script_name", "mpc_export.py"),
                mime="text/x-python", use_container_width=True, key="export_script_download",
            )

    with col4:
        animate_help = "Available once the run stops or finishes."
        if can_animate:
            animate_help = ("The Animation Agent writes a small drawing function for this system (heavily "
                             "sandboxed before it's ever run), and the ACTUAL best-result trajectory is "
                             "played through it frame by frame into a GIF -- an extra LLM call, and "
                             "rendering can take a little while.")
        elif has_results and not ANIMATION_FEATURE_AVAILABLE:
            animate_help = f"Unavailable: {ANIMATION_FEATURE_ERROR}"
        if st.button("Generate Animation (for simple systems)", icon=":material/movie:", use_container_width=True,
                      disabled=not can_animate, help=animate_help):
            best = _best_row()
            summary = st.session_state.dynamics_summary
            sim_data = best.get("simulation_data")
            if not sim_data:
                st.error("The best iteration has no stored simulation data to animate.")
            else:
                with st.spinner("Animation Agent writing the drawing code..."):
                    draw_fn, is_3d, note = generate_animation_code(
                        class_name=summary.get("dynamics_class", "UnknownSystem"),
                        state_names=summary.get("state_names", []),
                        input_names=summary.get("input_names", []),
                        params=summary.get("params", {}),
                        user_context=animation_user_context,
                        tracker=st.session_state.get("token_tracker"),
                    )
                if draw_fn is None:
                    st.warning(note)
                else:
                    with st.spinner("Rendering the animation..."):
                        gif_bytes, render_note = render_animation_gif(
                            draw_fn, is_3d, sim_data["states"], sim_data["times"],
                            summary.get("params", {}), summary.get("state_names", []),
                        )
                    if gif_bytes is None:
                        st.warning(render_note)
                    else:
                        st.session_state.animation_gif_bytes = gif_bytes
                        st.session_state.animation_gif_name = (
                            f"{summary.get('dynamics_class', 'mpc')}_animation_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M')}.gif"
                        )
                        st.session_state.animation_description = note
                        st.session_state.animation_render_note = render_note

    if st.session_state.get("animation_gif_bytes"):
        st.divider()
        st.markdown('<div class="subheader">Animation</div>', unsafe_allow_html=True)
        st.caption(st.session_state.get("animation_description", ""))
        st.image(st.session_state.animation_gif_bytes)
        st.caption(st.session_state.get("animation_render_note", ""))
        st.download_button(
            "Download Animation (GIF)", st.session_state.animation_gif_bytes,
            file_name=st.session_state.get("animation_gif_name", "mpc_animation.gif"),
            mime="image/gif", key="animation_gif_download",
        )

    if st.session_state.get("report_pdf_bytes"):
        with col1:
            st.download_button(
                "Download Report PDF", st.session_state.report_pdf_bytes,
                file_name=st.session_state.get("report_pdf_name", "mpc_report.pdf"),
                mime="application/pdf", use_container_width=True, key="report_pdf_download",
            )


if st.session_state.running:
    _stop_col1, _stop_col2 = st.columns([5, 1])
    with _stop_col2:
        if st.button("Stop", icon=":material/stop:", use_container_width=True,
                      help="Stops after the current agent step finishes. Everything completed so far is kept -- "
                           "nothing is discarded -- and the last parameters become available to Manual Simulation."):
            st.session_state.stop_requested = True

render_report_section()


def render_diagnostics_banner():
    """Diagnostics Agent, stage 1: the deterministic scan (see
    agents/diagnostics_agent.py) runs on every render -- it's free (no LLM
    call), so there's no cost to always checking. If it finds nothing, this
    renders nothing at all. If it finds something, a warning banner names
    exactly what was detected immediately (this is the part that directly
    fixes "hit the API limit with no visible error" -- these events were
    already being logged, just easy to miss in the log panel), with a
    button for stage 2 (an LLM call for grounded, specific recommendations)
    that only runs if the user actually wants that detail.
    """
    if not DIAGNOSTICS_FEATURE_AVAILABLE or not st.session_state.results_data:
        return

    findings = scan_for_issues(
        st.session_state.get("logs", []), st.session_state.results_data, st.session_state.get("last_outputs", {}),
    )
    if not findings:
        return

    total_hits = sum(f["count"] for f in findings.values())
    categories_text = ", ".join(f"{v['count']}\u00d7 {k.replace('_', ' ')}" for k, v in findings.items())
    with st.container(border=True):
        st.warning(f"\u26a0\ufe0f **Diagnostics: {len(findings)} issue type(s) detected this run** ({categories_text})")
        if st.button("Get detailed recommendations", key="diagnostics_get_recs"):
            with st.spinner("Diagnostics Agent analyzing..."):
                st.session_state.diagnostics_report = generate_diagnostics_report(
                    findings, len(st.session_state.results_data), tracker=st.session_state.get("token_tracker"))
            st.rerun()

        report = st.session_state.get("diagnostics_report")
        if report is not None:
            for rec in report.recommendations:
                cat_title = ERROR_CATEGORY_TITLES.get(rec.category, rec.category)
                st.markdown(f"**{cat_title}**")
                st.write(rec.explanation)
                st.caption(f"Contribution to this run's problems: {rec.contribution_estimate}")
                st.markdown(f"\u2192 {rec.recommendation}")
                st.divider()


def render_full_tuning_ui(active_node: Optional[str] = None):
    """Renders the entire tabbed tuning UI FRESH on every call -- no
    st.empty() placeholders. Called identically from run_one_step() (the
    @st.fragment, once per node while running) and the idle branch (once
    the run is stopped/finished, or before any run has started). This is
    deliberate: a Streamlit fragment cannot reliably push updates into
    elements that were created OUTSIDE its own render tree (e.g. by the
    surrounding script, only once, before the fragment first ticks) -- that
    mismatch was the root cause of two bugs: the Data & Export table only
    ever showing iteration 1 (only the very first tick, which runs as part
    of the outer script when Run is clicked, could actually write into
    those externally-created placeholders), and the whole page visually
    "jumping" the moment Stop was pressed (the idle branch was the FIRST
    render pass all run to ever actually show the full accumulated data).
    Every st.tabs(...) call here is fresh, so there is nothing to update
    from outside -- each tick simply redraws everything from current state.
    """
    tab_names = ["Live Run", "Convergence", "Simulation", "Best Result", "Agent Reasoning", "Data & Export"]
    render_summary_cards()
    render_diagnostics_banner()
    if st.session_state.get("run_perturbed_params"):
        _pp = st.session_state.run_perturbed_params
        _pp_text = ", ".join(
            f"**{k}**: {old:.4g} \u2192 {new:.4g} ({(new / old - 1) * 100:+.1f}%)" for k, (old, new) in _pp.items()
        )
        st.info(f"\U0001f527 **Level 3 (Robust):** the Setup Agent perturbed these physical parameters by a "
                f"random amount up to +20% each for this run (genuine plant-model mismatch, not just a harder "
                f"initial condition) -- here's exactly what each one became: {_pp_text}")
    tab_live, tab_convergence, tab_simulation, tab_best, tab_reasoning, tab_data = st.tabs(tab_names)

    current_iter = len(st.session_state.results_data)
    last_decision = ""
    if st.session_state.results_data:
        last = st.session_state.results_data[-1]
        last_decision = f"last: {'FAILED' if not last['ok'] else last['strategy'].upper()}"

    with tab_live:
        render_agent_flow_diagram(active_node=active_node, iteration=current_iter, last_decision=last_decision,
                                   last_outputs=st.session_state.get("last_outputs", {}))
        if st.session_state.stopped_by_user:
            st.warning(
                "Run stopped by user after "
                f"{current_iter} iteration(s). Everything completed is kept below."
                + (" The last parameters have been carried over to the Manual Simulation tab."
                   if st.session_state.latest_params else "")
            )
        if current_iter == 0:
            st.info("Waiting for the first evaluation..." if st.session_state.running
                     else "Configure the run above and click **Run** to start.")
        max_iterations = st.session_state.get("run_max_iterations", 10)
        progress = min(current_iter / max_iterations, 1.0) if max_iterations else 0.0
        st.markdown(f"""
        <div class="progress-container">
            <div style="display:flex; justify-content:space-between; color:#4a5a7a; font-size:0.75rem;">
                <span>Iteration {current_iter} / {max_iterations}</span><span>{progress:.0%}</span>
            </div>
            <div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{progress*100:.1f}%;"></div></div>
        </div>""", unsafe_allow_html=True)
        render_metrics_cards()
        render_params_panel()
        render_log_panel()
        if st.session_state.last_raw_evaluator_update is not None:
            with st.expander("Debug: raw Evaluator node output (last iteration)", expanded=False):
                raw = st.session_state.last_raw_evaluator_update
                st.write("Keys present in the node's return value:", sorted(raw.keys()))
                st.write("`eval_error`:", raw.get("eval_error"))
                st.write("`metrics` dict:", raw.get("metrics"))
                st.write("`simulation_data` present:", raw.get("simulation_data") is not None)

    with tab_convergence:
        df = pd.DataFrame(st.session_state.results_data)
        if df.empty:
            st.info("No data yet.")
        else:
            fig = plot_convergence(df)
            if fig:
                st.pyplot(fig)
                plt.close(fig)
                render_metric_formulas(
                    ["MSE", "Overshoot", "Settling", "Effort", "Stable", "Oscillations", "IAE", "ISE"],
                    panel_name="Convergence",
                )
            else:
                st.warning("No successful iterations to chart yet -- check Live Run / Debug for error details.")

    with tab_simulation:
        if st.session_state.running:
            render_simulation_live()
        else:
            render_simulation_tab()
        st.divider()
        st.markdown('<div class="subheader">Manual Simulation</div>', unsafe_allow_html=True)
        st.caption("Run a single closed-loop simulation with your own Np/Nc/Q/R/P/dt -- no Agents involved. "
                   "If you just stopped a tuning run, the fields below are pre-filled with its last parameters.")
        render_manual_simulation_tab()
        st.divider()
        render_open_loop_test()

    with tab_best:
        render_best_result()

    with tab_reasoning:
        render_reasoning_panel()

    with tab_data:
        if st.session_state.running:
            render_data_live()
        else:
            render_results_table()
        render_token_usage_summary()


def render_token_usage_summary():
    """Full LLM token usage breakdown for this run -- see llm_base.py's
    TokenUsageTracker for how this is collected (a LangChain callback
    attached to every agent's LLM call, Actor/Critic/Terminator/Juror plus
    the on-demand Report/Animation/Diagnostics agents).

    Deliberately does NOT hardcode a dollar cost: Groq's per-model pricing
    changes over time and this can't be kept in sync reliably from inside
    the app. Instead the price-per-token is a field the user fills in
    themselves (from their own current Groq pricing page), and the cost
    estimate is computed live from that -- accurate whenever the rate is,
    rather than silently going stale.
    """
    st.divider()
    st.markdown('<div class="subheader">LLM Token Usage</div>', unsafe_allow_html=True)
    tracker = st.session_state.get("token_tracker")
    if tracker is None or tracker.snapshot()["call_count"] == 0:
        st.caption("No LLM calls recorded yet for this run.")
        return

    usage = tracker.snapshot()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Tokens", f"{usage['total_tokens']:,}")
    with c2:
        st.metric("Prompt Tokens", f"{usage['prompt_tokens']:,}")
    with c3:
        st.metric("Completion Tokens", f"{usage['completion_tokens']:,}")
    with c4:
        st.metric("LLM Calls", usage["call_count"])
    if usage["unparsed_calls"]:
        st.caption(f"Note: {usage['unparsed_calls']} call(s) returned usage data in a format this app "
                   f"couldn't parse -- the totals above are a slight undercount for those calls.")

    if usage["per_model"]:
        st.caption("Per model:")
        per_model_df = pd.DataFrame([
            {"Model": model, "Prompt": v["prompt"], "Completion": v["completion"], "Total": v["total"]}
            for model, v in usage["per_model"].items()
        ])
        st.dataframe(per_model_df, hide_index=True, use_container_width=True)

    with st.expander("Estimate cost", expanded=False):
        st.caption(
            "Groq's per-model pricing changes over time, so it isn't hardcoded here -- enter the current "
            "rate from your own Groq pricing page (console.groq.com) for an accurate estimate."
        )
        price_per_million = st.number_input(
            "Price per 1,000,000 tokens (USD)", min_value=0.0, value=0.0, step=0.01, format="%.4f",
            key="token_price_per_million",
            help="Blended rate applied to the total token count -- if your prompt/completion rates differ "
                 "significantly, use a weighted average or just check the more precise per-model numbers above.",
        )
        if price_per_million > 0:
            estimated_cost = (usage["total_tokens"] / 1_000_000) * price_per_million
            st.metric("Estimated Cost", f"${estimated_cost:.4f}")


# ============================================================================
# RUN LOGIC
# ============================================================================


@st.fragment(run_every=0.4 if st.session_state.running else None)
def run_one_step():
    if not st.session_state.running:
        return  # nothing to do -- this fragment keeps ticking (run_every was fixed at the last
                  # app-level rerun) until that next app-level rerun turns the ticking off

    n_states = st.session_state.get("run_n_states", 4)
    n_inputs = st.session_state.get("run_n_inputs", 1)
    scenario_level_active = st.session_state.get("run_scenario_level", 1)

    if st.session_state.stop_requested:
        st.session_state.logs.append({"time": time.strftime("%H:%M:%S"), "node": "SYSTEM",
                                       "message": "Stopped by user. Everything completed so far is kept."})
        st.session_state.running = False
        st.session_state.run_finished_at = datetime.now()
        st.session_state.stopped_by_user = True
        st.session_state.graph_iterator = None
        if st.session_state.latest_params:
            prefill_manual_sim_from_params(st.session_state.latest_params)
        st.rerun(scope="app")  # run truly ends here -- refresh the whole page so the sidebar's Run/Stop buttons update

    else:
        try:
            output = next(st.session_state.graph_iterator)
        except StopIteration:
            st.session_state.logs.append({"time": time.strftime("%H:%M:%S"), "node": "DONE", "message": "Optimization completed."})
            st.session_state.running = False
            st.session_state.run_finished_at = datetime.now()
            st.session_state.graph_iterator = None
            if st.session_state.latest_params:
                prefill_manual_sim_from_params(st.session_state.latest_params)
            st.rerun(scope="app")  # run truly ends here -- same reasoning as above
        except Exception as e:  # noqa: BLE001
            st.session_state.logs.append({"time": time.strftime("%H:%M:%S"), "node": "ERROR", "message": str(e)})
            st.session_state.running = False
            st.session_state.run_finished_at = datetime.now()
            st.session_state.graph_iterator = None
            # Stored (not rendered directly) -- this whole function is about
            # to be abandoned by the st.rerun() below anyway; render_full_tuning_ui
            # picks this up and displays it persistently on the next (and every
            # subsequent) render, which a direct st.error() here would not.
            st.session_state.run_error = f"Run failed: {e}\n\n{tb_module.format_exc()}"
            st.rerun(scope="app")  # run truly ends here too -- refresh the sidebar
        else:
            for node, update in output.items():
                st.session_state.logs.append({"time": time.strftime("%H:%M:%S"), "node": node.upper(), "message": f"Executed: {node}"})

                prompt_used = update.get("last_outputs") or {}
                if prompt_used:
                    st.session_state.last_outputs = {**st.session_state.last_outputs, **prompt_used}

                node_history = update.get("history", [])
                last_len = st.session_state._last_history_len
                for new_entry in node_history[last_len:]:
                    st.session_state.reasoning_entries.append({"time": time.strftime("%H:%M:%S"), "text": new_entry})
                st.session_state._last_history_len = len(node_history)

                if "current_params" in update and update["current_params"]:
                    st.session_state.latest_params = update["current_params"]

                if node == "evaluator":
                    iteration = update.get("iteration", len(st.session_state.results_data))
                    metrics = update.get("metrics") or {}

                    if update.get("eval_error"):
                        row = {
                            "iteration": iteration, "ok": False, "unstable": False, "error": update["eval_error"],
                            "traceback": update.get("eval_traceback"), "scenario": scenario_level_active,
                            "np": 0, "nc": 0, "Q_formatted": "--", "R_formatted": "--", "P_formatted": "--",
                            "mse": None, "overshoot": None, "settling": None, "effort": None, "cost": None,
                            "oscillation_count": None, "iae": None, "ise": None, "per_state_mse": {}, "per_state_overshoot": {},
                            "success": False, "is_stable": False, "strategy": update.get("strategy", "?"), "simulation_data": None,
                        }
                        st.session_state.results_data.append(row)
                        st.session_state.last_raw_evaluator_update = update
                        st.session_state.logs.append({"time": time.strftime("%H:%M:%S"), "node": "ERROR",
                            "message": f"Iteration {iteration} failed: {update['eval_error']}"})

                    elif not metrics:
                        row = {
                            "iteration": iteration, "ok": False, "unstable": False,
                            "error": "Evaluator returned no error but also no metrics -- unexpected internal state. "
                                     "See the Debug panel below for the raw node output.",
                            "traceback": None, "scenario": scenario_level_active,
                            "np": 0, "nc": 0, "Q_formatted": "--", "R_formatted": "--", "P_formatted": "--",
                            "mse": None, "overshoot": None, "settling": None, "effort": None, "cost": None,
                            "oscillation_count": None, "iae": None, "ise": None, "per_state_mse": {}, "per_state_overshoot": {},
                            "success": False, "is_stable": False, "strategy": update.get("strategy", "?"), "simulation_data": None,
                        }
                        st.session_state.results_data.append(row)
                        st.session_state.last_raw_evaluator_update = update
                        st.session_state.logs.append({"time": time.strftime("%H:%M:%S"), "node": "ERROR",
                            "message": f"Iteration {iteration}: evaluator returned empty metrics with no eval_error."})

                    else:
                        params = update.get("current_params") or st.session_state.latest_params
                        q = list(params.get("Q") or [])
                        r = list(params.get("R") or [])
                        p = list(params.get("P") or q)
                        while len(q) < n_states: q.append(0.0)
                        while len(r) < n_inputs: r.append(0.0)
                        while len(p) < n_states: p.append(0.0)

                        row = {
                            "iteration": iteration, "ok": True, "unstable": bool(metrics.get("Unstable", False)),
                            "error": None, "traceback": None,
                            "scenario": scenario_level_active, "np": params.get("Np", 0), "nc": params.get("Nc", 0),
                            "Q_formatted": format_weight(q), "R_formatted": format_weight(r), "P_formatted": format_weight(p),
                            "mse": metrics.get("MSE", 0.0), "overshoot": metrics.get("Max_Overshoot", 0.0),
                            "settling": metrics.get("Settling_Time", float("inf")),
                            "effort": metrics.get("Control_Effort_RMS", 0.0),
                            "cost": metrics.get("Cost"),
                            "oscillation_count": metrics.get("Oscillation_Count", 0),
                            "iae": metrics.get("Integral_Abs_Error", 0.0),
                            "ise": metrics.get("Integral_Sq_Error", 0.0),
                            "is_regulation": metrics.get("Is_Regulation", True),
                            "overshoot_meaningful": metrics.get("Overshoot_Meaningful", True),
                            "per_state_mse": metrics.get("Per_State_MSE", {}),
                            "per_state_overshoot": metrics.get("Per_State_Overshoot", {}),
                            "improvement": metrics.get("Improvement", {}),
                            "solver_diagnostics": metrics.get("Solver_Diagnostics", {}),
                            "success": update.get("success", False),
                            "is_stable": bool(metrics.get("Is_Stable", False)),
                            "strategy": update.get("exploration_strategy", "explore"),
                            "dt_mpc": metrics.get("Dt_Mpc"),
                            "simulation_data": update.get("simulation_data"),
                        }
                        for i, v in enumerate(q): row[f"q{i+1}"] = v
                        for i, v in enumerate(r): row[f"r{i+1}"] = v
                        st.session_state.results_data.append(row)
                        st.session_state.last_raw_evaluator_update = update

                        unstable_tag = "  [UNSTABLE: " + str(metrics.get("Unstable_Reason")) + "]" if row["unstable"] else ""
                        st.session_state.logs.append({"time": time.strftime("%H:%M:%S"), "node": "METRIC",
                            "message": f"MSE={fmt_num(row['mse'])}  OS={fmt_num(row['overshoot'])}  Np={row['np']}  Nc={row['nc']}{unstable_tag}"})

                # -- render the ENTIRE tuning UI fresh on every step. Each
                # fragment tick processes exactly one node (see the Stop-
                # button note above); render_full_tuning_ui rebuilds
                # everything from current session_state rather than trying
                # to push updates into externally-created placeholders,
                # which is what a fragment cannot reliably do. --
                render_full_tuning_ui(active_node=node)

            # No manual st.rerun() here -- run_every (see the decorator above)
            # automatically re-invokes just this fragment for the next node.
            # A manual st.rerun(scope="fragment") call was tried here first,
            # but that raises StreamlitAPIException: scope="fragment" is only
            # valid once a fragment rerun is already in progress, not on this
            # function's normal (first, app-level-triggered) invocation.


if st.session_state.running:
    run_one_step()

else:
    if st.session_state.get("run_error"):
        st.error(st.session_state.run_error)
        if st.button("Dismiss error", key="dismiss_run_error"):
            st.session_state.run_error = None
            st.rerun()
    render_full_tuning_ui(active_node=None)
